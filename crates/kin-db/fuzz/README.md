<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 Firelock, LLC -->

# kin-db fuzzing

Bounded `cargo-fuzz` (libFuzzer) coverage for kin-db's binary ingestion and
semantic-entity-graph build paths.

kin-db has no source-language parser of its own — tree-sitter parsing lives in
`kin`. kin-db's "parser/ingestion entry points that build the semantic entity
graph" are therefore the **binary snapshot/delta decoders**, the
**graph + index builder**, and the **entity-extraction read surfaces** that
walk the adjacency built during ingestion. These targets fuzz exactly those.

## Targets

| Target | Entry point | What it exercises |
| --- | --- | --- |
| `fuzz_snapshot_deser` | `GraphSnapshot::from_bytes` → `from_snapshot` → `to_bytes` | Snapshot header/version/checksum + MessagePack decode + round-trip |
| `fuzz_delta_deser` | `GraphSnapshotDelta::from_bytes` | Delta header/version/length-bound + MessagePack decode |
| `fuzz_delta_apply` | `from_bytes` → `apply_graph_delta` → `from_snapshot` | **Incremental ingestion**: merging a decoded delta into a snapshot, then rebuilding the graph |
| `fuzz_graph_ingest` | `from_bytes` → `from_snapshot` → `get_relations` / `get_all_relations_for_entity` | **Entity-graph build + extraction**: index/adjacency construction, then walking every ingested entity's adjacency (dangling/self/duplicate edges) |

## Prerequisites

```sh
rustup toolchain install nightly      # libFuzzer requires nightly
cargo install cargo-fuzz --locked
```

## Run (bounded / reproducible)

From `crates/kin-db`:

```sh
# Single target, 60s smoke (matches CI's bounded tier)
cargo +nightly fuzz run fuzz_graph_ingest -- -max_total_time=60 -rss_limit_mb=4096

# All targets
for t in fuzz_snapshot_deser fuzz_delta_deser fuzz_delta_apply fuzz_graph_ingest; do
  cargo +nightly fuzz run "$t" -- -max_total_time=60 -rss_limit_mb=4096
done

# Replay a single input (e.g. a crash artifact)
cargo +nightly fuzz run fuzz_graph_ingest fuzz/artifacts/fuzz_graph_ingest/<id>
```

The committed `corpus/<target>/` seeds drive the fuzzer past the magic/version
header into the decode → ingest → build → extract stages. The accumulated
corpus is intentionally **not** committed; only the hand-authored seeds are.

## CI

`.github/workflows/fuzz.yml` runs every target for a bounded `-max_total_time`
on PRs that touch the parser/ingestion/storage surface, on a weekly schedule,
and on manual dispatch. A crash fails the job and uploads the reproducing
artifact.

## Triage

A finding must be tied to a concrete decoder/ingestion fixture (commit the
reproducing input under `corpus/<target>/`), then either fixed in kin-db or
filed as a child of the parser/ingestion hardening epic (FIR-894) before that
epic closes.
