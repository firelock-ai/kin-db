# KinDB

KinDB is the graph and search substrate for the Kin ecosystem.

It exists as a separate repo because storage, snapshots, indexing, vector search, and retrieval-adjacent infrastructure are foundational below multiple Kin surfaces. The goal is to keep those concerns cleanly below the semantic VCS, editor, agent, and hosted product layers.

## What This Repo Owns

- graph storage and mutation primitives
- snapshot persistence and recovery-friendly serialization
- incremental index support
- full-text search substrate
- vector search substrate
- low-level traversal and query machinery

## Current State

Today this repo contains the `kin-db` crate under `crates/kin-db/` and provides the storage layer used by `kin`.

Current implementation themes:

- in-memory graph engine with snapshot persistence
- Tantivy-backed text search
- embedding and vector-search dependencies for semantic retrieval work
- concurrency and storage primitives meant to be reused by higher Kin layers

This repo is infrastructure, not the user-facing semantic product. If a behavior changes local repo truth, review semantics, or product UX directly, it probably belongs in `kin`, `kin-review`, `kin-search`, `kin-code`, or `kinhub`, not here.

## Validate

```bash
cargo test -p kin-db
```

Optional feature flags live in [`crates/kin-db/Cargo.toml`](/Users/troyfortinjr/GitHub/kin-ecosystem/kin-db/crates/kin-db/Cargo.toml) for Metal, CUDA, and Accelerate-backed embedding/runtime paths.

## Repo Layout

- `crates/kin-db/`
  the main Rust crate
- `docs/ARCHITECTURE.md`
  storage and engine design
- `docs/EVALUATION.md`
  evaluation notes and measurement direction
- `docs/ZERO_COPY_PLAN.md`
  lower-level performance and memory direction

## Relationship To Other Repos

- `kin`
  uses KinDB as its semantic graph storage and query engine
- `kin-search`
  should own ranking and retrieval policy above KinDB's raw indices
- `kin-code`, `kin-codex`, and `kinhub`
  should consume Kin semantics through stable boundaries built above KinDB, not by embedding product-specific logic into the database layer
- `kin-graph-service`
  depends on graph-backed data and projections, but should not redefine storage behavior

## Design Rule

Keep KinDB narrow and composable:

- put storage and low-level query capability here
- keep semantic repo rules in `kin`
- keep review/gate logic in `kin-review`
- keep ranking strategy in `kin-search`
- keep hosted/workbench behavior out of the database layer

For deeper design detail, start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
