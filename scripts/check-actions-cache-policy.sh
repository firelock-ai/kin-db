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
    if uses_line.lstrip().rstrip().startswith("-"):
        start = uses_at
        step_indent = uses_indent
    else:
        start = None
        step_indent = None
        for index in range(uses_at - 1, -1, -1):
            line = lines[index]
            stripped = line.lstrip().rstrip()
            indent = len(line) - len(line.lstrip())
            if (stripped == "-" or stripped.startswith("- ")) and indent < uses_indent:
                start = index
                step_indent = indent
                break
        if start is None:
            raise ValueError(f"could not find step start before line {uses_at + 1}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.lstrip().rstrip()
        indent = len(line) - len(line.lstrip())
        if (stripped == "-" or stripped.startswith("- ")) and indent <= step_indent:
            end = index
            break
        if stripped and indent < step_indent:
            end = index
            break
    return start, end, lines[start:end]


def scalar(block, field):
    match = None
    key = rf"(?:{re.escape(field)}|'{re.escape(field)}'|\"{re.escape(field)}\")"
    pattern = re.compile(rf"^\s*(?:-\s*)?{key}\s*:\s*(.*?)\s*$")
    for line in block:
        candidate = pattern.match(line)
        if candidate:
            if match is not None:
                raise ValueError(f"step declares {field!r} more than once")
            value = candidate.group(1)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            match = value
    return match


def literal_list(block, field):
    key = rf"(?:{re.escape(field)}|'{re.escape(field)}'|\"{re.escape(field)}\")"
    pattern = re.compile(rf"^(\s*)(?:-\s*)?{key}\s*:\s*\|\s*$")
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
    parsed_cache_steps = 0
    restore_count = 0
    save_count = 0
    for uses_at, line in enumerate(lines):
        line_action = scalar([line], "uses")
        if line_action is None or not line_action.startswith("actions/cache"):
            continue
        try:
            start, end, block = step_block(lines, uses_at)
            if (start, end) in seen_ranges:
                continue
            seen_ranges.add((start, end))
            parsed_cache_steps += 1
            action = scalar(block, "uses")
            paths = literal_list(block, "path")
            key = scalar(block, "key")
            condition = scalar(block, "if")
            block_text = "\n".join(block)

            if action == "actions/cache/restore@v6":
                restore_count += 1
                if scalar(block, "id") != "cargo-sources":
                    errors.append(f"{workflow.name}:{start + 1}: restore id must be cargo-sources")
                if condition is not None:
                    errors.append(
                        f"{workflow.name}:{start + 1}: cache restore must run on every workflow ref"
                    )
                step_indent = len(lines[start]) - len(lines[start].lstrip())
                job_start = 0
                for prior_at in range(start - 1, -1, -1):
                    prior_line = lines[prior_at]
                    stripped = prior_line.strip()
                    indent = len(prior_line) - len(prior_line.lstrip())
                    if stripped and indent < step_indent:
                        job_start = prior_at + 1
                        break
                prior_run = next(
                    (
                        prior_at + 1
                        for prior_at in range(job_start, start)
                        if scalar([lines[prior_at]], "run") is not None
                    ),
                    None,
                )
                if prior_run is not None:
                    errors.append(
                        f"{workflow.name}:{start + 1}: cache restore must precede every run step; "
                        f"found earlier work at line {prior_run}"
                    )
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
                later_step = None
                step_indent = len(lines[start]) - len(lines[start].lstrip())
                for later_at in range(end, len(lines)):
                    later_line = lines[later_at]
                    stripped = later_line.lstrip().rstrip()
                    indent = len(later_line) - len(later_line.lstrip())
                    if stripped and indent < step_indent:
                        break
                    if (stripped == "-" or stripped.startswith("- ")) and indent == step_indent:
                        later_step = later_at + 1
                        break
                if later_step is not None:
                    errors.append(
                        f"{workflow.name}:{start + 1}: cache save must be the last declared job step; "
                        f"found a later step at line {later_step}"
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

    raw_cache_mentions = "\n".join(lines).count("actions/cache")
    if parsed_cache_steps != raw_cache_mentions:
        errors.append(
            f"{workflow.name}: found {raw_cache_mentions} actions/cache mentions but parsed "
            f"{parsed_cache_steps} cache steps; unsupported YAML shape or comment"
        )

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
    "OK: repo-local Actions caches are source-only, epoch-bounded, and save only from main "
    "(4 restores, 2 saves)."
)
PY
