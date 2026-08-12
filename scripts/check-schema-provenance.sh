#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC
#
# Refuses a kin-model version bump that leaves schemas/ behind.
#
# The JSON schemas under schemas/ are schemars output for kin-model types.
# kin-model owns the generator, so nothing in this repo can regenerate them,
# and nothing here reads them at runtime either. That combination is what let
# them rot unnoticed: the pin moved from 0.5.0 to =0.7.8 and the snapshot
# stayed put, so eleven types the newer kin-model defines had no schema at all
# and two dropped types still did, with every CI job green throughout.
#
# The one fact this repo can check on its own is whether the version the
# schemas were generated from still matches the version the crate depends on.
# It cannot prove a file was not hand-edited, and it does not claim to. Say
# what it covers and let the gap be visible rather than implied.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="${repo_root}/crates/kin-db/Cargo.toml"
stamp="${repo_root}/schemas/kin-model-version.txt"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

[ -f "${manifest}" ] || fail "missing ${manifest}"
[ -f "${stamp}" ] || fail "missing ${stamp}; schemas/README.md explains what it records"

# Both reads below must produce a version or abort. An unreadable input and a
# matching pair are the same silent success otherwise, which is the exact shape
# of check this guard exists to replace.
pinned="$(sed -n 's/^kin-model[[:space:]]*=.*version[[:space:]]*=[[:space:]]*"=\{0,1\}\([0-9][^"]*\)".*/\1/p' "${manifest}" | head -n 1)"
[ -n "${pinned}" ] || fail "could not read the kin-model version from ${manifest}"

generated="$(tr -d '[:space:]' < "${stamp}")"
[ -n "${generated}" ] || fail "${stamp} is empty"

echo "Schema provenance"
echo "  kin-model pinned by crates/kin-db/Cargo.toml : ${pinned}"
echo "  kin-model recorded by schemas/               : ${generated}"

if [ "${pinned}" != "${generated}" ]; then
  echo ""
  fail "the kin-model pin moved to ${pinned} but schemas/ was generated from ${generated}.
Regenerate the schemas and update schemas/kin-model-version.txt.
schemas/README.md carries the command."
fi

echo "OK: the checked-in schemas were generated from the pinned kin-model."
