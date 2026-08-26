#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC
#
# FIR-2615's acceptance measurement: does a one-file commit's peak resident
# set grow sublinearly in store size?
#
# Runs `fir2615_commit_peak_bench` at every store size named on the command
# line (default 512 2048 8192), one seed process and one commit process per
# sample, and reports the commit process's peak from two independent
# readings: the bench's own getrusage call and `/usr/bin/time -l`. They
# measure the same quantity, so a disagreement is the news.
#
# Usage: scripts/fir2615-peak-matrix.sh [samples] [size...]

set -euo pipefail

TEST_PATH="storage::repository::tests::fir2615_commit_peak_bench"
SAMPLES="${1:-3}"
shift || true
SIZES=("$@")
if [ "${#SIZES[@]}" -eq 0 ]; then
  SIZES=(512 2048 8192)
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# A measurement is only as good as the tree it ran on, and a dirty tree has no
# sha anyone can cite. Refuse rather than produce an unattributable number.
if [ -n "$(git status --porcelain)" ]; then
  echo "REFUSING: the tree is dirty, so these numbers would have no sha to cite" >&2
  git status --porcelain >&2
  exit 2
fi

SHA="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "tree=$REPO_ROOT sha=$SHA branch=$BRANCH"

BIN="${KIN_FIR2615_BIN:-}"
if [ -z "$BIN" ]; then
  echo "REFUSING: set KIN_FIR2615_BIN to the release kin-db unittest binary" >&2
  exit 2
fi
echo "bin=$BIN"

# A filter that matches nothing grades nothing and exits 0, which is the same
# status a clean pass returns. Prove the binary carries this test before
# trusting any run of it, and prove the check can still say no.
listed="$("$BIN" --list 2>/dev/null | grep -c "^${TEST_PATH}: test$" || true)"
if [ "$listed" != "1" ]; then
  echo "REFUSING: the binary lists $listed tests named $TEST_PATH, expected exactly 1" >&2
  exit 2
fi
absent="$("$BIN" --list 2>/dev/null | grep -c "^storage::repository::tests::a_test_that_does_not_exist: test$" || true)"
if [ "$absent" != "0" ]; then
  echo "REFUSING: the absent-test control matched $absent, so the listing check cannot fail" >&2
  exit 2
fi
echo "control: the test is listed once and a fabricated name is listed zero times"

run_phase() {
  # $1 phase, $2 dir, $3 files, $4 round, $5 log
  local phase="$1" dir="$2" files="$3" round="$4" log="$5"
  local rc=0
  set +e
  env KIN_FIR2615_PHASE="$phase" KIN_FIR2615_DIR="$dir" KIN_FIR2615_FILES="$files" \
      KIN_FIR2615_ROUND="$round" KIN_FIR2615_DEPTH="${KIN_FIR2615_DEPTH:-1}" \
      KIN_EMBED_BACKEND=cpu \
      /usr/bin/time -l "$BIN" "$TEST_PATH" --exact --ignored --nocapture > "$log" 2>&1
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "REFUSING: $phase at files=$files exited $rc" >&2
    cat "$log" >&2
    exit 3
  fi
  if ! grep -q "^test result: ok\. 1 passed; 0 failed" "$log"; then
    echo "REFUSING: $phase at files=$files did not report exactly one passing test" >&2
    grep "test result" "$log" >&2 || echo "(no test result line at all)" >&2
    exit 3
  fi
}

printf '%s\n' "files,depth,round,store_kib,open_ms,commit_ms,peak_after_open_bytes,bench_peak_bytes,time_l_peak_bytes"
for files in "${SIZES[@]}"; do
  for round in $(seq 1 "$SAMPLES"); do
    dir="$(mktemp -d "${TMPDIR:-/tmp}/fir2615-${files}-XXXXXX")"
    rm -rf "$dir"
    seed_log="$(mktemp "${TMPDIR:-/tmp}/fir2615-seed-XXXXXX.log")"
    commit_log="$(mktemp "${TMPDIR:-/tmp}/fir2615-commit-XXXXXX.log")"

    run_phase seed "$dir" "$files" 1 "$seed_log"
    store_kib="$(du -sk "$dir" | awk '{print $1}')"
    run_phase commit "$dir" "$files" "$round" "$commit_log"

    line="$(grep -m1 '^FIR2615 phase=commit' "$commit_log")"
    depth="$(sed -n 's/.* changes=\([0-9]*\).*/\1/p' <<<"$line")"
    open_ms="$(sed -n 's/.* open_ms=\([0-9]*\).*/\1/p' <<<"$line")"
    commit_ms="$(sed -n 's/.* commit_ms=\([0-9]*\).*/\1/p' <<<"$line")"
    after_open="$(sed -n 's/.* peak_after_open_bytes=\([0-9]*\).*/\1/p' <<<"$line")"
    bench_peak="$(sed -n 's/.* peak_rss_bytes=\([0-9]*\).*/\1/p' <<<"$line")"
    time_peak="$(awk '/maximum resident set size/ {print $1; exit}' "$commit_log")"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$files" "$depth" "$round" "$store_kib" "$open_ms" "$commit_ms" \
      "$after_open" "$bench_peak" "$time_peak"

    rm -rf "$dir" "$seed_log" "$commit_log"
  done
done
