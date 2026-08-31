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
  mkdir -p "${case_root}"
  cp "${root}"/.github/workflows/*.yml "${case_root}/"
  printf '%s\n' "${case_root}"
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

case_root="$(make_case non-main-save)"
perl -0pi -e "s/if: github.ref == 'refs\/heads\/main' && steps.cargo-sources.outputs.cache-hit != 'true'/if: always()/" "${case_root}/ci.yml"
expect_rejection non-main-save "${case_root}" "cache save must be restricted to a main cache miss"

case_root="$(make_case monolithic-action)"
perl -0pi -e 's#actions/cache/restore\@v6#actions/cache\@v6#' "${case_root}/ci.yml"
expect_rejection monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

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

echo "OK: all Actions cache policy falsifiers were rejected."
