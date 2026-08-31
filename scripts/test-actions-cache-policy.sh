#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
check="${root}/scripts/check-actions-cache-policy.sh"
fixtures="$(mktemp -d)"
trap 'rm -rf "${fixtures}"' EXIT

make_case() {
  local name="$1"
  local case_root="${fixtures}/${name}"
  local workflow_root="${case_root}/.github/workflows"
  local action_root="${case_root}/.github/actions"
  mkdir -p "${workflow_root}" "${action_root}"
  cp "${root}"/.github/workflows/*.yml "${workflow_root}/"
  cp -R "${root}/.github/actions/." "${action_root}/"
  printf '%s\n' "${workflow_root}"
}

expect_rejection() {
  local name="$1"
  local workflow_root="$2"
  local expected="$3"
  local output
  if output="$("${check}" "${workflow_root}" 2>&1)"; then
    echo "FAIL: ${name} falsifier was accepted" >&2
    exit 1
  fi
  if ! grep -Fq "${expected}" <<<"${output}"; then
    echo "FAIL: ${name} failed for the wrong reason" >&2
    printf '%s\n' "${output}" >&2
    exit 1
  fi
  echo "OK: ${name} rejected"
}

"${check}" "${root}/.github/workflows"

case_root="$(make_case target-output)"
perl -0pi -e 's#(~/.cargo/git\n)#$1            target\n#' "${case_root}/ci.yml"
expect_rejection target-output "${case_root}" "target output is forbidden"

case_root="$(make_case dynamic-key)"
perl -0pi -e 's/cargo-sources-v2/cargo-sources-\$\{\{ github.sha \}\}/' "${case_root}/ci.yml"
expect_rejection dynamic-key "${case_root}" "cache keys must not expand"

case_root="$(make_case lookup-only)"
perl -0pi -e 's/(          restore-keys: \|\n)/          lookup-only: true\n$1/' "${case_root}/ci.yml"
expect_rejection lookup-only "${case_root}" "restore inputs must be exactly"

case_root="$(make_case fail-on-cache-miss)"
perl -0pi -e 's/(          restore-keys: \|\n)/          fail-on-cache-miss: true\n$1/' "${case_root}/ci.yml"
expect_rejection fail-on-cache-miss "${case_root}" "restore inputs must be exactly"

case_root="$(make_case non-main-save)"
perl -0pi -e "s/if: github.ref == 'refs\/heads\/main' && steps.cargo-sources.outputs.cache-hit != 'true'/if: always()/" "${case_root}/ci.yml"
expect_rejection non-main-save "${case_root}" "cache save must be restricted to a main cache miss"

case_root="$(make_case monolithic-action)"
perl -0pi -e 's#actions/cache/restore\@v6#actions/cache\@v6#' "${case_root}/ci.yml"
expect_rejection monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

case_root="$(make_case uppercase-monolithic-action)"
perl -0pi -e 's#actions/cache/restore\@v6#Actions/cache\@v6#' "${case_root}/ci.yml"
expect_rejection uppercase-monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

case_root="$(make_case escaped-monolithic-action)"
perl -0pi -e 's#uses: actions/cache/restore\@v6#uses: "actions/\\u0063ache\@v6"#' "${case_root}/ci.yml"
expect_rejection escaped-monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

case_root="$(make_case conditional-restore)"
perl -0pi -e "s/(      - name: Restore cargo sources\n)/\$1        if: github.event_name == 'pull_request'\n/" "${case_root}/ci.yml"
expect_rejection conditional-restore "${case_root}" "cache restore must run on every workflow ref"

case_root="$(make_case conditional-restore-spaced-key)"
perl -0pi -e "s/(      - name: Restore cargo sources\n)/\$1        if : github.event_name == 'pull_request'\n/" "${case_root}/ci.yml"
expect_rejection conditional-restore-spaced-key "${case_root}" "cache restore must run on every workflow ref"

case_root="$(make_case conditional-restore-quoted-key)"
perl -0pi -e "s/(      - name: Restore cargo sources\n)/\$1        'if' : github.event_name == 'pull_request'\n/" "${case_root}/ci.yml"
expect_rejection conditional-restore-quoted-key "${case_root}" "cache restore must run on every workflow ref"

case_root="$(make_case conditional-restore-double-quoted-key)"
perl -0pi -e 's/(      - name: Restore cargo sources\n)/$1        "if" : github.event_name == '\''pull_request'\''\n/' "${case_root}/ci.yml"
expect_rejection conditional-restore-double-quoted-key "${case_root}" "cache restore must run on every workflow ref"

case_root="$(make_case early-save)"
perl -0pi -e '
  if (s#\n(      - name: Save cargo sources on main\n.*?          key: \$\{\{ steps\.cargo-sources\.outputs\.cache-primary-key \}\}\n)#$save = $1; "\n"#se) {
    s#(      - name: Restore cargo sources\n.*?            \$\{\{ runner\.os \}\}-cargo-sources-\n)#$1\n$save#s;
  }
' "${case_root}/ci.yml"
expect_rejection early-save "${case_root}" "cache save must be the last declared job step"

case_root="$(make_case soft-fetch)"
perl -0pi -e 's/cargo fetch/cargo fetch || true/' "${case_root}/ci.yml"
expect_rejection soft-fetch "${case_root}" "cache save must immediately follow an unconditional fail-hard cargo fetch"

case_root="$(make_case conditional-fetch)"
perl -0pi -e 's/(      - name: Fetch complete Cargo source graph\n)/$1        if: always()\n/' "${case_root}/ci.yml"
expect_rejection conditional-fetch "${case_root}" "cache save must immediately follow an unconditional fail-hard cargo fetch"

case_root="$(make_case bare-dash-late-step)"
perl -0pi -e 's#\n  \# Security audit moved#\n      -\n        name: Late Cargo work\n        run: cargo fetch\n\n  \# Security audit moved#' "${case_root}/ci.yml"
expect_rejection bare-dash-late-step "${case_root}" "cache save must be the last declared job step"

case_root="$(make_case late-restore)"
perl -0pi -e 's#(      - name: Restore cargo sources\n)#      - name: Cargo work before restore\n        run: cargo fetch\n\n$1#' "${case_root}/ci.yml"
expect_rejection late-restore "${case_root}" "cache restore must precede every run step"

case_root="$(make_case comment-before-condition)"
perl -0pi -e "s/(            \\$\{\{ runner.os \}\}-cargo-sources-\n)/\$1# parser boundary comment\n        if: github.event_name == 'pull_request'\n/" "${case_root}/ci.yml"
expect_rejection comment-before-condition "${case_root}" "cache restore must run on every workflow ref"

case_root="$(make_case comment-before-late-step)"
perl -0pi -e 's!\n  # Security audit moved!\n# parser boundary comment\n      - name: Later work\n        run: echo later\n\n  # Security audit moved!' "${case_root}/ci.yml"
expect_rejection comment-before-late-step "${case_root}" "cache save must be the last declared job step"

case_root="$(make_case duplicate-condition-key)"
perl -0pi -e "s/(      - name: Restore cargo sources\n)/\$1        if: true\n        if: false\n/" "${case_root}/ci.yml"
expect_rejection duplicate-condition-key "${case_root}" "duplicate YAML mapping key"

case_root="$(make_case composite-cache-action)"
perl -0pi -e 's!\z!\n    - name: Hidden target cache\n      uses: Actions/cache\@v6\n      with:\n        path: target\n        key: hidden-target\n!' "${case_root}/../actions/rust-toolchain/action.yml"
expect_rejection composite-cache-action "${case_root}" "repo-local composite actions must not invoke actions/cache"

case_root="$(make_case setup-node-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden Node cache\n        uses: actions/setup-node\@v6\n!' "${case_root}/ci.yml"
expect_rejection setup-node-default-cache "${case_root}" "actions/setup-node must set package-manager-cache: false"

case_root="$(make_case setup-go-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden Go cache\n        uses: actions/setup-go\@v6\n!' "${case_root}/ci.yml"
expect_rejection setup-go-default-cache "${case_root}" "actions/setup-go must set cache: false"

case_root="$(make_case setup-gradle-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden Gradle cache\n        uses: gradle/actions/setup-gradle\@v4\n!' "${case_root}/ci.yml"
expect_rejection setup-gradle-default-cache "${case_root}" "gradle/actions/setup-gradle must set cache-disabled: true"

case_root="$(make_case setup-uv-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden uv cache\n        uses: astral-sh/setup-uv\@v6\n!' "${case_root}/ci.yml"
expect_rejection setup-uv-default-cache "${case_root}" "astral-sh/setup-uv must set enable-cache: false"

case_root="$(make_case buildkit-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden BuildKit cache\n        uses: docker/build-push-action\@v6\n        with:\n          cache-to: type=gha\n!' "${case_root}/ci.yml"
expect_rejection buildkit-cache "${case_root}" 'hidden cache input "cache-to" is forbidden'

case_root="$(make_case guarded-job-condition)"
perl -0pi -e 's/(  schema-provenance:\n)/$1    if: false\n/' "${case_root}/ci.yml"
expect_rejection guarded-job-condition "${case_root}" "schema-provenance job must not declare if"

case_root="$(make_case guarded-step-soft-fail)"
perl -0pi -e 's/(      - name: Check Actions cache policy\n)/$1        continue-on-error: true\n/' "${case_root}/ci.yml"
expect_rejection guarded-step-soft-fail "${case_root}" "cache policy guard must be the first unconditional step after checkout"

echo "OK: all Actions cache policy falsifiers were rejected."
