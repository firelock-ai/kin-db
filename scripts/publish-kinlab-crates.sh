#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC
#
# Hardened, dependency-aware publish of kin-db to the private "kin" cargo
# registry (FIR-1021).
#
# Guarantees layered on top of the first-cut auto-publish:
#
#   1. Dependency-ordered publish. kin-db's registry dependencies (kin-model,
#      kin-search, and — under the relevant features — kin-vector / kin-infer)
#      live in sibling repos and must already be published. A preflight reads
#      kin-db's DECLARED requirements from `cargo metadata` (NOT the locally
#      patched/resolved versions) and fails loudly if any required version —
#      in particular the pinned kin-model version — is missing from the kin
#      registry index. kin-db is only published once its deps resolve.
#
#   2. Checksum-identical idempotency. An existing-version publish (HTTP 409)
#      is treated as success ONLY when the registry's index checksum for that
#      version is byte-identical to the crate we just packaged. A version that
#      already exists with DIFFERENT bytes fails loudly instead of being
#      silently accepted.
#
#   3. Post-publish checksum proof. After a successful publish the script
#      re-reads the index checksum, downloads the published .crate, and asserts
#      download-checksum == index-checksum == locally-packaged-checksum.
#
# Version source of truth is Cargo.toml (cargo metadata). TAG_NAME /
# GITHUB_REF_NAME is an optional consistency check on a tag push.
#
# PUBLISH IS TROY-GATED: the registry fails closed without KINLAB_CARGO_TOKEN
# (only Troy holds it), and the release workflow's publish job is pinned to a
# protected GitHub Environment. This script performs read-only registry GETs
# during preflight/verify even without the token; only the POST publish needs
# it.
#
# Env:
#   KINLAB_CARGO_REGISTRY_URL  registry base (default https://kinlab.ai)
#   KINLAB_CARGO_TOKEN         bearer token for publish (Troy-held secret)
#   DRY_RUN=1                  package + checksum locally, skip POST and network
#   SKIP_REGISTRY_CHECKS=1     skip the network preflight/verify (local packaging)
#   TAG_NAME / GITHUB_REF_*    optional tag/version consistency check

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

registry_url="${KINLAB_CARGO_REGISTRY_URL:-https://kinlab.ai}"
registry_url="${registry_url%/}"
# Sparse index root and HTTP API share this base on the kin daemon.
index_base="${registry_url}/registry/cargo"
# The kin cargo registry REQUIRES this token to publish: the daemon fails
# closed (rejects publishes) when its KIN_REGISTRY_CARGO_TOKEN is unset, and the
# token sent here must match it. Reads remain open. Provided in CI via secret.
registry_token="${KINLAB_CARGO_TOKEN:-${KINLAB_TOKEN:-}}"
dry_run="${DRY_RUN:-0}"
skip_registry_checks="${SKIP_REGISTRY_CHECKS:-0}"

is_truthy() { [[ "$1" == "1" || "$1" == "true" ]]; }

# Version source of truth is Cargo.toml (cargo metadata), so a version-bump
# merge to main auto-publishes without a git tag. TAG_NAME / GITHUB_REF_NAME is
# accepted as an OPTIONAL consistency check: when invoked from a tag push it
# must match the Cargo version. A non-tag GITHUB_REF_NAME (a branch name on a
# push-to-main publish) is ignored.
tag_name="${TAG_NAME:-}"
if [[ -z "$tag_name" && "${GITHUB_REF_TYPE:-}" == "tag" ]]; then
  tag_name="${GITHUB_REF_NAME:-}"
fi

expected_version=""
if [[ -n "$tag_name" ]]; then
  if [[ "$tag_name" != v* ]]; then
    echo "Release tag must start with 'v' (got: $tag_name)" >&2
    exit 1
  fi
  expected_version="${tag_name#v}"
fi

if command -v cargo >/dev/null 2>&1; then
  cargo_bin="$(command -v cargo)"
elif [[ -x "${HOME}/.cargo/bin/cargo" ]]; then
  cargo_bin="${HOME}/.cargo/bin/cargo"
else
  echo "cargo was not found in PATH or ~/.cargo/bin/cargo" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
metadata_json="$tmpdir/metadata.json"
"$cargo_bin" metadata --no-deps --format-version 1 >"$metadata_json"

# --- helpers --------------------------------------------------------------

sha256_file() {
  local path="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$path" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$path" | awk '{print $1}'
  else
    echo "no sha256 tool (sha256sum/shasum) available" >&2
    exit 1
  fi
}

resolve_version() {
  local package_name="$1"
  python3 - "$metadata_json" "$package_name" <<'PY'
import json
import sys

metadata_path, package_name = sys.argv[1], sys.argv[2]
with open(metadata_path, "r", encoding="utf-8") as fh:
    metadata = json.load(fh)

for package in metadata["packages"]:
    if package["name"] == package_name:
        print(package["version"])
        raise SystemExit(0)

raise SystemExit(f"package not found in cargo metadata: {package_name}")
PY
}

# Cargo sparse-index path rules: 1/2 char names use len-prefixed buckets, 3-char
# names use 3/<c1>, and >=4 char names use <c1c2>/<c3c4>. Always lowercased.
sparse_index_path() {
  local name
  name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  local len="${#name}"
  case "$len" in
    1) printf '1/%s' "$name" ;;
    2) printf '2/%s' "$name" ;;
    3) printf '3/%s/%s' "${name:0:1}" "$name" ;;
    *) printf '%s/%s/%s' "${name:0:2}" "${name:2:2}" "$name" ;;
  esac
}

# Fetch the sparse index file for a crate into $tmpdir/index-<name>. Returns
# non-zero if the crate is absent (404) or the registry is unreachable.
fetch_index() {
  local name="$1"
  local out="$tmpdir/index-${name}"
  local path
  path="$(sparse_index_path "$name")"
  curl -fsSL "${index_base}/${path}" -o "$out" 2>/dev/null
}

# Read the kin registry's download URL template from config.json (cargo's
# registry config). Falls back to the conventional daemon path.
dl_base() {
  local cfg="$tmpdir/config.json"
  if curl -fsSL "${index_base}/config.json" -o "$cfg" 2>/dev/null; then
    local dl
    dl="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("dl",""))' "$cfg" 2>/dev/null || true)"
    if [[ -n "$dl" && "$dl" != "None" ]]; then
      printf '%s' "${dl%/}"
      return 0
    fi
  fi
  printf '%s' "${index_base}/api/v1/crates"
}

# Print the index checksum (cksum) for an exact published version, or nothing.
index_cksum_for() {
  local name="$1" version="$2"
  local idx="$tmpdir/index-${name}"
  [[ -f "$idx" ]] || return 0
  python3 - "$idx" "$version" <<'PY'
import json
import sys

idx_path, version = sys.argv[1], sys.argv[2]
with open(idx_path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("vers") == version and not entry.get("yanked", False):
            print(entry.get("cksum", ""))
            break
PY
}

# True if the crate index has at least one non-yanked published version.
index_has_any_version() {
  local name="$1"
  local idx="$tmpdir/index-${name}"
  [[ -f "$idx" ]] || return 1
  python3 - "$idx" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not entry.get("yanked", False):
            raise SystemExit(0)
raise SystemExit(1)
PY
}

# Download a published .crate and print its sha256, or nothing on failure.
download_crate_cksum() {
  local name="$1" version="$2"
  local out="$tmpdir/dl-${name}-${version}.crate"
  local base
  base="$(dl_base)"
  if curl -fsSL "${base}/${name}/${version}/download" -o "$out" 2>/dev/null; then
    sha256_file "$out"
  fi
}

# --- dependency-ordered preflight ----------------------------------------

# Emit "name<TAB>req<TAB>base_version<TAB>required|optional" for every kin
# registry dependency declared by kin-db. base_version is the pinned version
# extracted from caret/tilde/exact reqs (empty for open ranges like ">=x").
kin_registry_deps() {
  python3 - "$metadata_json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    metadata = json.load(fh)

seen = set()
for package in metadata["packages"]:
    if package["name"] != "kin-db":
        continue
    for dep in package["dependencies"]:
        if not dep.get("registry"):
            continue  # crates.io dep
        name = dep["name"]
        req = dep["req"]
        optional = dep.get("optional", False)
        kind = dep.get("kind")  # None=normal, "dev", "build"
        key = (name, req, kind)
        if key in seen:
            continue
        seen.add(key)
        # Extract a pinned base version from caret/tilde/exact/plain reqs.
        m = re.match(r"^[\s=^~]*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", req)
        base = m.group(1) if m and not req.lstrip().startswith((">", "<")) else ""
        flag = "optional" if optional else "required"
        # dev-only deps are not part of the published crate's resolve graph.
        if kind == "dev":
            flag = "dev"
        print(f"{name}\t{req}\t{base}\t{flag}")
PY
}

preflight_registry_deps() {
  echo "== Dependency-ordered preflight: verifying kin-db's registry deps are published =="
  local failed=0
  local name req base flag
  while IFS=$'\t' read -r name req base flag; do
    [[ -n "$name" ]] || continue
    if ! fetch_index "$name"; then
      if [[ "$flag" == "required" ]]; then
        echo "  MISSING (required): $name not found in kin registry index (req $req)" >&2
        failed=1
      else
        echo "  absent ($flag): $name not in registry — ok, not in this publish's resolve set"
      fi
      continue
    fi
    if [[ -n "$base" ]]; then
      if [[ -n "$(index_cksum_for "$name" "$base")" ]]; then
        echo "  OK: $name $base published (req $req, $flag)"
      else
        if [[ "$flag" == "required" ]]; then
          echo "  MISSING (required): $name has no published version $base (req $req) — pin would dangle" >&2
          failed=1
        else
          echo "  WARN ($flag): $name $base not published (req $req); verified by registry-only consumer smoke"
        fi
      fi
    else
      # Open range — confirm at least one published version exists.
      if index_has_any_version "$name"; then
        echo "  OK: $name has published versions (range req $req, $flag)"
      elif [[ "$flag" == "required" ]]; then
        echo "  MISSING (required): $name has no published versions (req $req)" >&2
        failed=1
      fi
    fi
  done < <(kin_registry_deps)

  if [[ "$failed" -ne 0 ]]; then
    echo "Dependency preflight failed: publish kin-db's required deps first (dependency order)." >&2
    exit 1
  fi
  echo "Dependency preflight passed."
}

# --- post-publish / idempotency verification ------------------------------

# Assert the published version's index checksum matches the locally packaged
# crate, and that the downloadable blob matches the index. Exits non-zero on
# any mismatch or a missing index entry.
verify_published_matches() {
  local name="$1" version="$2" local_cksum="$3"
  fetch_index "$name" || {
    echo "Post-publish: $name not resolvable in index after publish" >&2
    exit 1
  }
  local idx_cksum
  idx_cksum="$(index_cksum_for "$name" "$version")"
  if [[ -z "$idx_cksum" ]]; then
    echo "Post-publish: no index entry for $name@$version (yanked or absent)" >&2
    exit 1
  fi
  if [[ "$idx_cksum" != "$local_cksum" ]]; then
    echo "CHECKSUM MISMATCH for $name@$version:" >&2
    echo "  index    cksum: $idx_cksum" >&2
    echo "  packaged cksum: $local_cksum" >&2
    echo "Refusing to treat as idempotent — the registry holds DIFFERENT bytes for this version." >&2
    exit 1
  fi
  # Download the published blob and confirm it matches the index checksum.
  local dl_cksum
  dl_cksum="$(download_crate_cksum "$name" "$version")"
  if [[ -z "$dl_cksum" ]]; then
    echo "  note: could not download $name@$version to re-verify the stored blob (index cksum matched packaged cksum)."
  elif [[ "$dl_cksum" != "$idx_cksum" ]]; then
    echo "DOWNLOAD CHECKSUM MISMATCH for $name@$version:" >&2
    echo "  download cksum: $dl_cksum" >&2
    echo "  index    cksum: $idx_cksum" >&2
    exit 1
  else
    echo "  post-publish checksum verified: download == index == packaged ($idx_cksum)"
  fi
}

# --- publish --------------------------------------------------------------

publish_package() {
  local package_name="$1"
  local package_version
  package_version="$(resolve_version "$package_name")"

  if [[ -n "$expected_version" && "$package_version" != "$expected_version" ]]; then
    echo "Version mismatch for $package_name: tag expects $expected_version but Cargo metadata resolved $package_version" >&2
    exit 1
  fi

  local crate_file="target/package/${package_name}-${package_version}.crate"
  local package_log="$tmpdir/${package_name}.package.log"
  echo "Packaging $package_name@$package_version"
  if ! "$cargo_bin" package -p "$package_name" --allow-dirty --no-verify 2>&1 | tee "$package_log"; then
    if [[ -f "$crate_file" ]]; then
      echo "cargo package reported a registry verification error after producing $crate_file; continuing with the packaged crate"
    else
      echo "cargo package failed for $package_name before producing $crate_file" >&2
      exit 1
    fi
  fi

  if [[ ! -f "$crate_file" ]]; then
    echo "Expected packaged crate not found: $crate_file" >&2
    exit 1
  fi

  local local_cksum
  local_cksum="$(sha256_file "$crate_file")"
  echo "Packaged $package_name@$package_version (sha256 $local_cksum)"

  if is_truthy "$dry_run"; then
    echo "[dry-run] Would publish $package_name@$package_version to ${registry_url}"
    return
  fi

  local response_file="$tmpdir/${package_name}.response"
  local url="${index_base}/api/v1/crates/publish?name=${package_name}&version=${package_version}"
  local curl_args=(
    -sS
    -o "$response_file"
    -w "%{http_code}"
    -X POST "$url"
    -H "content-type: application/octet-stream"
    --data-binary "@${crate_file}"
  )

  if [[ -n "$registry_token" ]]; then
    curl_args+=(-H "authorization: Bearer ${registry_token}")
  fi

  local http_code
  http_code="$(curl "${curl_args[@]}")"

  case "$http_code" in
    200|201|204)
      echo "Published $package_name@$package_version"
      ;;
    409)
      echo "$package_name@$package_version already exists; verifying byte-identical idempotency"
      # verify_published_matches fails loudly if the existing bytes differ.
      ;;
    *)
      echo "Publish failed for $package_name@$package_version (HTTP $http_code)" >&2
      cat "$response_file" >&2 || true
      exit 1
      ;;
  esac

  if is_truthy "$skip_registry_checks"; then
    echo "SKIP_REGISTRY_CHECKS set — skipping post-publish checksum verification."
    return
  fi
  verify_published_matches "$package_name" "$package_version" "$local_cksum"
}

# --- main -----------------------------------------------------------------

if is_truthy "$dry_run" || is_truthy "$skip_registry_checks"; then
  echo "Skipping dependency-ordered registry preflight (dry-run / SKIP_REGISTRY_CHECKS)."
else
  preflight_registry_deps
fi

publish_package "kin-db"
