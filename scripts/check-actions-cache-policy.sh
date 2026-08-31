#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Firelock, LLC

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workflow_root="${1:-${root}/.github/workflows}"

python3 - "${workflow_root}" <<'PY'
import re
import sys
from pathlib import Path

workflow_root = Path(sys.argv[1])
if not workflow_root.is_dir():
    sys.exit(f"FAIL: workflow directory does not exist: {workflow_root}")

expected_counts = {
    "ci.yml": (2, 1),
    "ci-linux.yml": (1, 0),
    "windows-nightly.yml": (1, 1),
}
allowed_paths = ["~/.cargo/registry", "~/.cargo/git"]
restore_key = "${{ runner.os }}-cargo-sources-v2"
restore_prefix = "${{ runner.os }}-cargo-sources-"
save_key = "${{ steps.cargo-sources.outputs.cache-primary-key }}"
save_condition = (
    "github.ref == 'refs/heads/main' && "
    "steps.cargo-sources.outputs.cache-hit != 'true'"
)


def step_block(lines, uses_at):
    uses_line = lines[uses_at]
    uses_indent = len(uses_line) - len(uses_line.lstrip())
    if uses_line.lstrip().startswith("- uses:"):
        start = uses_at
        step_indent = uses_indent
    else:
        start = None
        step_indent = None
        for index in range(uses_at - 1, -1, -1):
            line = lines[index]
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if stripped.startswith("- ") and indent < uses_indent:
                start = index
                step_indent = indent
                break
        if start is None:
            raise ValueError(f"could not find step start before line {uses_at + 1}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.startswith("- ") and indent <= step_indent:
            end = index
            break
        if stripped and indent < step_indent:
            end = index
            break
    return start, end, lines[start:end]


def scalar(block, field):
    match = None
    pattern = re.compile(rf"^\s*(?:-\s*)?{re.escape(field)}:\s*(.*?)\s*$")
    for line in block:
        candidate = pattern.match(line)
        if candidate:
            if match is not None:
                raise ValueError(f"step declares {field!r} more than once")
            match = candidate.group(1)
    return match


def literal_list(block, field):
    pattern = re.compile(rf"^(\s*)(?:-\s*)?{re.escape(field)}:\s*\|\s*$")
    for index, line in enumerate(block):
        match = pattern.match(line)
        if not match:
            continue
        field_indent = len(match.group(1))
        values = []
        for value_line in block[index + 1:]:
            stripped = value_line.strip()
            indent = len(value_line) - len(value_line.lstrip())
            if stripped and indent <= field_indent:
                break
            if stripped and not stripped.startswith("#"):
                values.append(stripped)
        return values
    return None


errors = []
counts = {}
workflows = sorted(workflow_root.glob("*.yml")) + sorted(workflow_root.glob("*.yaml"))
if not workflows:
    sys.exit(f"FAIL: no workflow files found under {workflow_root}")

for workflow in workflows:
    lines = workflow.read_text(encoding="utf-8").splitlines()
    seen_ranges = set()
    restore_count = 0
    save_count = 0
    for uses_at, line in enumerate(lines):
        if not re.match(r"^\s*(?:-\s*)?uses:\s*actions/cache(?:/[^@\s]+)?@", line):
            continue
        try:
            start, end, block = step_block(lines, uses_at)
            if (start, end) in seen_ranges:
                continue
            seen_ranges.add((start, end))
            action = scalar(block, "uses")
            paths = literal_list(block, "path")
            key = scalar(block, "key")
            condition = scalar(block, "if")
            block_text = "\n".join(block)

            if action == "actions/cache/restore@v6":
                restore_count += 1
                if scalar(block, "id") != "cargo-sources":
                    errors.append(f"{workflow.name}:{start + 1}: restore id must be cargo-sources")
                if key != restore_key:
                    errors.append(
                        f"{workflow.name}:{start + 1}: restore key must be the bounded epoch {restore_key}"
                    )
                if literal_list(block, "restore-keys") != [restore_prefix]:
                    errors.append(
                        f"{workflow.name}:{start + 1}: restore prefix must be {restore_prefix}"
                    )
            elif action == "actions/cache/save@v6":
                save_count += 1
                if condition != save_condition:
                    errors.append(
                        f"{workflow.name}:{start + 1}: cache save must be restricted to a main cache miss"
                    )
                if key != save_key:
                    errors.append(
                        f"{workflow.name}:{start + 1}: save key must come from the restore primary key"
                    )
            else:
                errors.append(
                    f"{workflow.name}:{start + 1}: use actions/cache/restore@v6 or actions/cache/save@v6, not {action}"
                )

            if paths != allowed_paths:
                errors.append(
                    f"{workflow.name}:{start + 1}: cache paths must be exactly {allowed_paths}; target output is forbidden"
                )
            if re.search(r"(^|[\s/])target([\s/]|$)", block_text):
                errors.append(f"{workflow.name}:{start + 1}: target output is forbidden in Actions caches")
            if "hashFiles(" in block_text or "github.sha" in block_text or "github.run_id" in block_text:
                errors.append(
                    f"{workflow.name}:{start + 1}: cache keys must not expand per dependency hash, SHA, or run"
                )
        except ValueError as error:
            errors.append(f"{workflow.name}:{uses_at + 1}: {error}")

    counts[workflow.name] = (restore_count, save_count)

for workflow_name, expected in expected_counts.items():
    actual = counts.get(workflow_name, (0, 0))
    if actual != expected:
        errors.append(
            f"{workflow_name}: expected {expected[0]} restore and {expected[1]} save steps; "
            f"found {actual[0]} restore and {actual[1]} save steps"
        )

for workflow_name, actual in counts.items():
    if workflow_name not in expected_counts and actual != (0, 0):
        errors.append(
            f"{workflow_name}: unexpected cache action; add it to the bounded policy deliberately"
        )

if errors:
    print("FAIL: GitHub Actions cache policy is not bounded:")
    for error in errors:
        print(f"  - {error}")
    sys.exit(1)

print(
    "OK: Actions caches are source-only, epoch-bounded, and save only from main "
    "(4 restores, 2 saves)."
)
PY
