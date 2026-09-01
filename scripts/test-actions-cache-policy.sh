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
  mkdir -p "${workflow_root}" "${action_root}" "${case_root}/.cargo" "${case_root}/scripts"
  cp -p "${root}"/.github/workflows/*.yml "${workflow_root}/"
  cp -Rp "${root}/.github/actions/." "${action_root}/"
  cp -p "${root}/.cargo/config.toml" "${case_root}/.cargo/config.toml"
  cp -p "${root}/rust-toolchain.toml" "${case_root}/rust-toolchain.toml"
  cp -p "${root}/scripts/check-actions-cache-policy.sh" "${case_root}/scripts/"
  cp -p "${root}/scripts/test-actions-cache-policy.sh" "${case_root}/scripts/"
  printf '%s\n' "${workflow_root}"
}

expect_rejection() {
  local name="$1"
  local workflow_root="$2"
  shift 2
  local candidate_root
  local output
  candidate_root="$(cd "${workflow_root}/../.." && pwd)"
  if output="$("${check}" "${workflow_root}" "${candidate_root}/.github/actions" "${candidate_root}" 2>&1)"; then
    echo "FAIL: ${name} falsifier was accepted" >&2
    exit 1
  fi
  local expected
  for expected in "$@"; do
    if ! grep -Fq "${expected}" <<<"${output}"; then
      echo "FAIL: ${name} failed for the wrong reason; missing: ${expected}" >&2
      printf '%s\n' "${output}" >&2
      exit 1
    fi
  done
  echo "OK: ${name} rejected"
}

add_target_path() {
  local workflow_root="$1"
  perl -0pi -e 's#(~/.cargo/git\n)#$1            target\n#' "${workflow_root}/ci.yml"
}

"${check}" "${root}/.github/workflows" "${root}/.github/actions" "${root}"

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
perl -0pi -e 's#actions/cache/restore\@[0-9a-f]{40}#actions/cache\@55cc8345863c7cc4c66a329aec7e433d2d1c52a9#' "${case_root}/ci.yml"
expect_rejection monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

case_root="$(make_case uppercase-monolithic-action)"
perl -0pi -e 's#actions/cache/restore\@[0-9a-f]{40}#Actions/cache\@55cc8345863c7cc4c66a329aec7e433d2d1c52a9#' "${case_root}/ci.yml"
expect_rejection uppercase-monolithic-action "${case_root}" "use actions/cache/restore@v6 or actions/cache/save@v6"

case_root="$(make_case escaped-monolithic-action)"
perl -0pi -e 's#uses: actions/cache/restore\@[0-9a-f]{40}#uses: "actions/\\u0063ache\@55cc8345863c7cc4c66a329aec7e433d2d1c52a9"#' "${case_root}/ci.yml"
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
expect_rejection guarded-job-condition "${case_root}" "schema-provenance job permits only"

case_root="$(make_case guarded-step-soft-fail)"
perl -0pi -e 's/(      - name: Check Actions cache policy\n)/$1        continue-on-error: true\n/' "${case_root}/ci.yml"
expect_rejection guarded-step-soft-fail "${case_root}" "cache policy guard must be the first exact fail-hard step after checkout"

case_root="$(make_case guarded-checkout-main)"
perl -0pi -e 's/(  schema-provenance:.*?      - uses: actions\/checkout\@[0-9a-f]{40}[^\n]*\n)/$1        with:\n          ref: main\n/s' "${case_root}/ci.yml"
expect_rejection guarded-checkout-main "${case_root}" "must begin with the exact immutable checkout identity"

case_root="$(make_case guarded-shell-mask)"
perl -0pi -e 's/(      - name: Check Actions cache policy\n        shell:) bash/$1 "bash {0} || true"/' "${case_root}/ci.yml"
expect_rejection guarded-shell-mask "${case_root}" "cache policy guard must be the first exact fail-hard step after checkout"

case_root="$(make_case guarded-job-default-mask)"
perl -0pi -e 's/(  schema-provenance:\n)/$1    defaults:\n      run:\n        shell: "bash {0} || true"\n/' "${case_root}/ci.yml"
expect_rejection guarded-job-default-mask "${case_root}" "schema-provenance job permits only"

case_root="$(make_case workflow-default-mask)"
perl -0pi -e 's/(jobs:\n)/defaults:\n  run:\n    shell: "bash {0} || true"\n\n$1/' "${case_root}/ci.yml"
expect_rejection workflow-default-mask "${case_root}" "workflow defaults are forbidden"

case_root="$(make_case setup-buildx-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden Buildx cache\n        uses: docker/setup-buildx-action\@v3\n!' "${case_root}/ci.yml"
expect_rejection setup-buildx-default-cache "${case_root}" "unapproved action identity"

case_root="$(make_case setup-qemu-default-cache)"
perl -0pi -e 's!\z!\n      - name: Hidden QEMU cache\n        uses: docker/setup-qemu-action\@v3\n!' "${case_root}/ci.yml"
expect_rejection setup-qemu-default-cache "${case_root}" "unapproved action identity"

case_root="$(make_case copied-reusable-workflow)"
cp "${case_root}/kin-dependency-wave.yml" "${case_root}/kin-dependency-wave-copy.yml"
expect_rejection copied-reusable-workflow "${case_root}" "uses unapproved reusable workflow"

case_root="$(make_case local-action-outside-root)"
mkdir -p "${case_root}/../cache-toolchain"
cp "${case_root}/../actions/rust-toolchain/action.yml" "${case_root}/../cache-toolchain/action.yml"
perl -0pi -e 's!\./\.github/actions/rust-toolchain!./.github/cache-toolchain!' "${case_root}/ci.yml"
expect_rejection local-action-outside-root "${case_root}" "unapproved action identity"

case_root="$(make_case build-continue-on-error)"
perl -0pi -e 's/(      - name: Build\n)/$1        continue-on-error: true\n/' "${case_root}/ci.yml"
expect_rejection build-continue-on-error "${case_root}" "cache-owner steps must not continue on error"

case_root="$(make_case build-command-mask)"
perl -0pi -e 's/run: cargo build --all-targets/run: cargo build --all-targets || true/' "${case_root}/ci.yml"
expect_rejection build-command-mask "${case_root}" 'protected "Build" step drifted or became non-authoritative'

case_root="$(make_case fetch-cargo-home)"
perl -0pi -e 's/(      - name: Fetch complete Cargo source graph\n)/$1        env:\n          CARGO_HOME: \/tmp\/other-cargo\n/' "${case_root}/ci.yml"
expect_rejection fetch-cargo-home "${case_root}" "protected runner environment CARGO_HOME must not be overridden"

case_root="$(make_case save-home)"
perl -0pi -e 's/(      - name: Save cargo sources on main\n)/$1        env:\n          HOME: \/tmp\/hidden-target\n/' "${case_root}/ci.yml"
expect_rejection save-home "${case_root}" "protected runner environment HOME must not be overridden"

case_root="$(make_case target-under-source-cache)"
perl -0pi -e 's/(      - name: Build\n)/$1        env:\n          CARGO_TARGET_DIR: \/home\/runner\/.cargo\/git\/target\n/' "${case_root}/ci.yml"
expect_rejection target-under-source-cache "${case_root}" "protected runner environment CARGO_TARGET_DIR must not be overridden"

case_root="$(make_case owner-job-condition)"
perl -0pi -e 's/(  check:\n)/$1    if: github.ref == '\''refs\/heads\/main'\''\n/' "${case_root}/ci.yml"
expect_rejection owner-job-condition "${case_root}" "cache-owner job permits only"

# Compound fixtures prove a candidate cannot hide a real cache violation by
# weakening the same candidate-controlled self-check.
case_root="$(make_case compound-shell-mask)"
add_target_path "${case_root}"
perl -0pi -e 's/(      - name: Check Actions cache policy\n        shell:) bash/$1 "bash {0} || true"/' "${case_root}/ci.yml"
expect_rejection compound-shell-mask "${case_root}" \
  "target output is forbidden" \
  "candidate self-check job drifted or became non-authoritative"

case_root="$(make_case compound-checkout-main)"
add_target_path "${case_root}"
perl -0pi -e 's/(  schema-provenance:.*?      - uses: actions\/checkout\@[0-9a-f]{40}[^\n]*\n)/$1        with:\n          ref: main\n/s' "${case_root}/ci.yml"
expect_rejection compound-checkout-main "${case_root}" \
  "target output is forbidden" \
  "candidate self-check job drifted or became non-authoritative"

case_root="$(make_case compound-skipped-job)"
add_target_path "${case_root}"
perl -0pi -e 's/(  schema-provenance:\n)/$1    if: github.event_name == '\''workflow_dispatch'\''\n/' "${case_root}/ci.yml"
expect_rejection compound-skipped-job "${case_root}" \
  "target output is forbidden" \
  "candidate self-check job drifted or became non-authoritative"

case_root="$(make_case compound-soft-job)"
add_target_path "${case_root}"
perl -0pi -e 's/(  schema-provenance:\n)/$1    continue-on-error: true\n/' "${case_root}/ci.yml"
expect_rejection compound-soft-job "${case_root}" \
  "target output is forbidden" \
  "candidate self-check job drifted or became non-authoritative"

case_root="$(make_case candidate-checker-replacement)"
add_target_path "${case_root}"
candidate_root="$(cd "${case_root}/../.." && pwd)"
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "${candidate_root}/scripts/check-actions-cache-policy.sh"
chmod +x "${candidate_root}/scripts/check-actions-cache-policy.sh"
expect_rejection candidate-checker-replacement "${case_root}" \
  "trusted policy implementation drifted from default-branch authority" \
  "target output is forbidden"

# The pull_request_target / merge_group authority is itself protected as data,
# including the event, permissions, refs, repository, shell, and fail semantics.
case_root="$(make_case authority-shell-mask)"
add_target_path "${case_root}"
perl -0pi -e 's/(      - name: Evaluate candidate with trusted checker\n        shell:) bash/$1 "bash {0} || true"/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-shell-mask "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-trusted-ref)"
add_target_path "${case_root}"
perl -0pi -e 's/github\.event\.pull_request\.base\.sha/github.event.pull_request.head.sha/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-trusted-ref "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-candidate-ref)"
add_target_path "${case_root}"
perl -0pi -e "s#ref: \\\$\{\{ github\.event_name == 'pull_request_target' && github\.event\.pull_request\.head\.sha \|\| github\.event\.merge_group\.head_sha \}\}#ref: main#" "${case_root}/cache-policy-authority.yml"
expect_rejection authority-candidate-ref "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-candidate-repository)"
add_target_path "${case_root}"
perl -0pi -e "s#repository: \\\$\{\{ github\.event_name == 'pull_request_target' && github\.event\.pull_request\.head\.repo\.full_name \|\| github\.repository \}\}#repository: \\\$\{\{ github.repository \}\}#" "${case_root}/cache-policy-authority.yml"
expect_rejection authority-candidate-repository "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-job-condition)"
add_target_path "${case_root}"
perl -0pi -e 's/(  candidate-policy:\n)/$1    if: false\n/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-job-condition "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-job-soft-fail)"
add_target_path "${case_root}"
perl -0pi -e 's/(  candidate-policy:\n)/$1    continue-on-error: true\n/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-job-soft-fail "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-trigger)"
add_target_path "${case_root}"
perl -0pi -e 's/pull_request_target:/pull_request:/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-trigger "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

case_root="$(make_case authority-permission)"
add_target_path "${case_root}"
perl -0pi -e 's/contents: read/contents: write/' "${case_root}/cache-policy-authority.yml"
expect_rejection authority-permission "${case_root}" \
  "trusted authority workflow drifted or became non-authoritative" \
  "target output is forbidden"

# The registry workflow is a one-step fail-hard hold. Reintroducing v0.1.31,
# even by immutable commit, is forbidden because it would still archive target/.
case_root="$(make_case registry-hold-exit-zero)"
perl -0pi -e 's/          exit 1/          exit 0/' "${case_root}/registry-publish.yml"
expect_rejection registry-hold-exit-zero "${case_root}" "registry release hold drifted or became non-authoritative"

case_root="$(make_case registry-hold-shell-mask)"
perl -0pi -e 's/(      - name: Refuse the unsafe registry workflow\n        shell:) bash/$1 "bash {0} || true"/' "${case_root}/registry-publish.yml"
expect_rejection registry-hold-shell-mask "${case_root}" "registry release hold drifted or became non-authoritative"

case_root="$(make_case registry-hold-job-condition)"
perl -0pi -e 's/(  release:\n)/$1    if: false\n/' "${case_root}/registry-publish.yml"
expect_rejection registry-hold-job-condition "${case_root}" "registry release hold drifted or became non-authoritative"

case_root="$(make_case registry-hold-step-soft-fail)"
perl -0pi -e 's/(      - name: Refuse the unsafe registry workflow\n)/$1        continue-on-error: true\n/' "${case_root}/registry-publish.yml"
expect_rejection registry-hold-step-soft-fail "${case_root}" "registry release hold drifted or became non-authoritative"

case_root="$(make_case registry-hold-removed)"
perl -0pi -e 's/\njobs:\n  release:.*\z/\njobs: {}\n/s' "${case_root}/registry-publish.yml"
expect_rejection registry-hold-removed "${case_root}" 'required protected job "release" is missing'

case_root="$(make_case unsafe-registry-tag)"
perl -0pi -e 's!\z!\n  unsafe-release:\n    uses: firelock-ai/kin-actions/.github/workflows/cargo-registry-release.yml\@v0.1.31\n!' "${case_root}/registry-publish.yml"
expect_rejection unsafe-registry-tag "${case_root}" \
  "unsafe target-caching registry release is forbidden" \
  "reusable workflow must use a full immutable commit SHA"

case_root="$(make_case unsafe-registry-commit)"
perl -0pi -e 's!\z!\n  unsafe-release:\n    uses: firelock-ai/kin-actions/.github/workflows/cargo-registry-release.yml\@d6b6585d0b5902437d2745a94a960fe0d7d27f0e\n!' "${case_root}/registry-publish.yml"
expect_rejection unsafe-registry-commit "${case_root}" "unsafe target-caching registry release is forbidden"

# Target output cannot be redirected under the two literal source roots through
# the local toolchain, GITHUB_ENV, or either tracked Cargo config filename.
case_root="$(make_case local-action-target-env)"
perl -0pi -e 's!\z!\n    - name: Redirect target into cached sources\n      shell: bash\n      run: echo "CARGO_TARGET_DIR=\$HOME/.cargo/git/target" >> "\$GITHUB_ENV"\n!' "${case_root}/../actions/rust-toolchain/action.yml"
expect_rejection local-action-target-env "${case_root}" \
  "protected local rust-toolchain action drifted" \
  "protected target or HOME write through GITHUB_ENV is forbidden"

case_root="$(make_case cargo-build-target-dir)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
printf '%s\n' '' '[build]' 'target-dir = "/home/runner/.cargo/git/target"' >> "${candidate_root}/.cargo/config.toml"
expect_rejection cargo-build-target-dir "${case_root}" "tracked Cargo config drifted or can redirect target output"

case_root="$(make_case cargo-env-target-dir)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
printf '%s\n' '' '[env]' 'CARGO_TARGET_DIR = "/home/runner/.cargo/git/target"' >> "${candidate_root}/.cargo/config.toml"
expect_rejection cargo-env-target-dir "${case_root}" "tracked Cargo config drifted or can redirect target output"

case_root="$(make_case legacy-cargo-config)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
printf '%s\n' '[build]' 'target-dir = "/home/runner/.cargo/git/target"' > "${candidate_root}/.cargo/config"
expect_rejection legacy-cargo-config "${case_root}" "legacy tracked Cargo config is forbidden"

# Bind each cache owner to its exact runner, matrix, step, and action topology.
case_root="$(make_case remove-macos-owner)"
perl -0pi -e 's/\[ubuntu-latest, macos-latest\]/[ubuntu-latest]/' "${case_root}/ci.yml"
expect_rejection remove-macos-owner "${case_root}" 'job "check" cache-owner job contract drifted'

case_root="$(make_case windows-runner-substitution)"
perl -0pi -e 's/runs-on: windows-latest/runs-on: ubuntu-latest/' "${case_root}/windows-nightly.yml"
expect_rejection windows-runner-substitution "${case_root}" 'job "windows" Windows cache-owner job contract drifted'

case_root="$(make_case linux-runner-substitution)"
perl -0pi -e 's/runs-on: ubuntu-latest/runs-on: macos-latest/' "${case_root}/ci-linux.yml"
expect_rejection linux-runner-substitution "${case_root}" 'job "linux-build-test" Linux cache-owner job contract drifted'

case_root="$(make_case coverage-runner-substitution)"
perl -0pi -e 's/(  coverage:.*?    runs-on:) ubuntu-latest/$1 macos-latest/s' "${case_root}/ci.yml"
expect_rejection coverage-runner-substitution "${case_root}" 'job "coverage" coverage cache-owner job contract drifted'

case_root="$(make_case rust-action-substitution)"
perl -0pi -e 's/(  check:.*?      - name: Install Rust toolchain\n        uses:) \.\/\.github\/actions\/rust-toolchain/$1 actions\/checkout\@3d3c42e5aac5ba805825da76410c181273ba90b1/s' "${case_root}/ci.yml"
expect_rejection rust-action-substitution "${case_root}" \
  'job "check" cache-owner job contract drifted' \
  "action identity drifted"

# Every remote action and reusable workflow is both full-SHA-shaped and bound
# to the exact reviewed identity, so tags and arbitrary commits fail closed.
case_root="$(make_case mutable-checkout-ref)"
perl -0pi -e 's/actions\/checkout\@[0-9a-f]{40}/actions\/checkout\@v7/' "${case_root}/ci.yml"
expect_rejection mutable-checkout-ref "${case_root}" "remote action must use a full immutable commit SHA"

case_root="$(make_case mutable-cache-ref)"
perl -0pi -e 's/actions\/cache\/restore\@[0-9a-f]{40}/actions\/cache\/restore\@v6/' "${case_root}/ci.yml"
expect_rejection mutable-cache-ref "${case_root}" "remote action must use a full immutable commit SHA"

case_root="$(make_case mutable-third-party-ref)"
perl -0pi -e 's/EmbarkStudios\/cargo-deny-action\@[0-9a-f]{40}/EmbarkStudios\/cargo-deny-action\@v2/' "${case_root}/sast.yml"
expect_rejection mutable-third-party-ref "${case_root}" "remote action must use a full immutable commit SHA"

case_root="$(make_case mutable-reusable-ref)"
perl -0pi -e 's/cargo-dependency-wave\.yml\@[0-9a-f]{40}/cargo-dependency-wave.yml\@v0.1.31/' "${case_root}/kin-dependency-wave.yml"
expect_rejection mutable-reusable-ref "${case_root}" "reusable workflow must use a full immutable commit SHA"

case_root="$(make_case arbitrary-reusable-commit)"
perl -0pi -e 's/cargo-dependency-wave\.yml\@[0-9a-f]{40}/cargo-dependency-wave.yml\@0000000000000000000000000000000000000000/' "${case_root}/kin-dependency-wave.yml"
expect_rejection arbitrary-reusable-commit "${case_root}" "uses unapproved reusable workflow"

# Symlinks are rejected before any candidate-controlled target is read.
case_root="$(make_case symlink-local-action)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
mv "${candidate_root}/.github/actions/rust-toolchain/action.yml" "${candidate_root}/.github/actions/rust-toolchain/action.real.yml"
ln -s action.real.yml "${candidate_root}/.github/actions/rust-toolchain/action.yml"
expect_rejection symlink-local-action "${case_root}" "must be a regular file, not a symlink"

case_root="$(make_case symlink-cargo-config)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
mv "${candidate_root}/.cargo/config.toml" "${candidate_root}/.cargo/config.real.toml"
ln -s config.real.toml "${candidate_root}/.cargo/config.toml"
expect_rejection symlink-cargo-config "${case_root}" "must be a regular file, not a symlink"

case_root="$(make_case symlink-policy-checker)"
candidate_root="$(cd "${case_root}/../.." && pwd)"
mv "${candidate_root}/scripts/check-actions-cache-policy.sh" "${candidate_root}/scripts/check-actions-cache-policy.real.sh"
ln -s check-actions-cache-policy.real.sh "${candidate_root}/scripts/check-actions-cache-policy.sh"
expect_rejection symlink-policy-checker "${case_root}" "must be a regular file, not a symlink"

case_root="$(make_case symlink-authority-workflow)"
mv "${case_root}/cache-policy-authority.yml" "${case_root}/cache-policy-authority.real.yml"
ln -s cache-policy-authority.real.yml "${case_root}/cache-policy-authority.yml"
expect_rejection symlink-authority-workflow "${case_root}" "must be a regular file, not a symlink"

echo "OK: all Actions cache policy falsifiers were rejected."
