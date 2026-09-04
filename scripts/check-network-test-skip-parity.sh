#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC
#
# Three tests in kin-db's lib suite (embed::tests::default_dimensions_match_default_model,
# embed::tests::process_embedding_queue_without_embeddings_is_noop,
# engine::graph::tests::test_vector_index_dimension_mismatch_auto_recovery) build a real
# embedder and need HF network plus nomic weights, which a CI runner cannot reliably fetch.
# ci.yml's `check` job carries the fix: HF_HUB_OFFLINE=1 plus a --skip for each of the three.
#
# Every OTHER place in this repo that runs kin-db's lib suite has to carry the identical guard,
# or it inherits the same flake the day its download hits a network hiccup or a stale lock file
# on the shared Hugging Face Hub cache -- which is exactly what happened on the 2026-09-04
# Kin Dependency Wave run: kin-dependency-wave.yml's test-command carried neither the offline
# flag nor the skip list, so the wave's own verification step failed with a download-lock
# IndexError unrelated to any pin it was validating.
#
# Nothing enforced that every surface stays in sync. Two, three, five files carrying the same
# guard drift the first time somebody edits one of them without the others, and the drift is
# invisible until a scheduled run or a release workflow hits the network on a bad day. So the
# parity is checked here, from the required CI job, where a divergence fails the pull request
# that introduced it.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "$root" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
CI = root / ".github/workflows/ci.yml"

SKIP_RE = re.compile(r"--skip\s+(\S+)")
# Two shapes appear in this repo: the YAML env-block form (`HF_HUB_OFFLINE: "1"`,
# value runs to end of line) and the inline shell-var form used inside a
# test-command string (`HF_HUB_OFFLINE=1 cargo test ...`, value runs to the next
# whitespace). Capture up to whitespace or a closing quote so both match.
OFFLINE_RE = re.compile(r"HF_HUB_OFFLINE\s*[:=]\s*['\"]?([^\s'\"]+)")


def read_lines(path):
    if not path.is_file():
        sys.exit(f"FAIL: missing file {path}")
    return path.read_text(encoding="utf-8").splitlines()


def block_after(lines, start, anchor_indent):
    """Lines strictly more indented than anchor_indent, starting after `start`,
    until a line at or below anchor_indent is hit (blank lines don't count)."""
    out = []
    for line in lines[start + 1:]:
        if not line.strip():
            out.append(line)
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= anchor_indent:
            break
        out.append(line)
    return out


def canonical_guard():
    """Extract the HF_HUB_OFFLINE value and --skip token set from ci.yml's
    `check` job, step `Test` -- the guard every other surface must match."""
    lines = read_lines(CI)
    try:
        job_at = next(i for i, l in enumerate(lines) if l.rstrip() == "  check:")
    except StopIteration:
        sys.exit(f"FAIL: no job 'check' in {CI}")
    job_indent = len(lines[job_at]) - len(lines[job_at].lstrip(" "))
    job_block = block_after(lines, job_at, job_indent)

    try:
        step_at = next(i for i, l in enumerate(job_block) if l.strip() == "- name: Test")
    except StopIteration:
        sys.exit(f"FAIL: no '- name: Test' step under job 'check' in {CI}")
    step_indent = len(job_block[step_at]) - len(job_block[step_at].lstrip(" "))
    step_block = block_after(job_block, step_at, step_indent)
    step_text = "\n".join(step_block)

    offline = OFFLINE_RE.search(step_text)
    skips = sorted(set(SKIP_RE.findall(step_text)))

    if not offline or offline.group(1) not in ("1", "true", "True"):
        sys.exit(f"FAIL: could not find a truthy HF_HUB_OFFLINE in ci.yml's Test step; "
                  f"the extraction is broken, not the file")
    if len(skips) < 3:
        sys.exit(f"FAIL: found only {len(skips)} --skip token(s) in ci.yml's Test step "
                  f"(expected 3); the extraction is broken, not the file")
    return skips


def find_step_block(path, job_name, step_name):
    lines = read_lines(path)
    try:
        job_at = next(i for i, l in enumerate(lines) if l.rstrip() == f"  {job_name}:")
    except StopIteration:
        sys.exit(f"FAIL: no job '{job_name}' in {path}")
    job_indent = len(lines[job_at]) - len(lines[job_at].lstrip(" "))
    job_block = block_after(lines, job_at, job_indent)
    try:
        step_at = next(i for i, l in enumerate(job_block) if l.strip() == f"- name: {step_name}")
    except StopIteration:
        sys.exit(f"FAIL: no '- name: {step_name}' step under job '{job_name}' in {path}")
    step_indent = len(job_block[step_at]) - len(job_block[step_at].lstrip(" "))
    return "\n".join(block_after(job_block, step_at, step_indent))


def find_input_block(path, input_name):
    lines = read_lines(path)
    pattern = re.compile(rf"^(\s*){re.escape(input_name)}:\s*(.*)$")
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            continue
        indent = len(m.group(1))
        inline = m.group(2).strip()
        if inline and inline not in (">-", "|", ">", "|-"):
            return inline
        block = block_after(lines, i, indent)
        return "\n".join(block)
    sys.exit(f"FAIL: no '{input_name}:' input in {path}")


# (file, how to find its guarded block) -- every surface in this repo that runs
# kin-db's lib test suite. Add a new pair here the day a new one appears.
TARGETS = [
    ("ci-linux.yml", lambda p: find_step_block(p, "linux-build-test", "Test")),
    ("windows-nightly.yml", lambda p: find_step_block(p, "windows", "Test")),
    ("registry-publish.yml", lambda p: find_input_block(p, "test-command")),
    ("kin-dependency-wave.yml", lambda p: find_input_block(p, "test-command")),
]

canon_skips = canonical_guard()
failures = []

for name, extractor in TARGETS:
    path = root / ".github/workflows" / name
    text = extractor(path)

    offline = OFFLINE_RE.search(text)
    offline_ok = bool(offline) and offline.group(1) in ("1", "true", "True")
    skips = sorted(set(SKIP_RE.findall(text)))

    missing_skips = sorted(set(canon_skips) - set(skips))
    if not offline_ok or missing_skips:
        failures.append((name, offline_ok, missing_skips))

if failures:
    print("FAIL: the following workflow surfaces run kin-db's lib test suite without the same")
    print("HF_HUB_OFFLINE + --skip guard ci.yml's Test step carries:")
    print("")
    for name, offline_ok, missing_skips in failures:
        print(f"  {name}:")
        if not offline_ok:
            print("    - missing (or falsy) HF_HUB_OFFLINE")
        for tok in missing_skips:
            print(f"    - missing --skip {tok}")
    print("")
    print("These three tests build a real embedder against the live Hugging Face Hub and")
    print("cannot run reliably in CI. Add the same guard ci.yml's Test step carries.")
    sys.exit(1)

print(f"OK: {len(TARGETS)} workflow surface(s) all carry the same HF_HUB_OFFLINE + "
      f"{len(canon_skips)}-test skip guard as ci.yml's Test step.")
PY
