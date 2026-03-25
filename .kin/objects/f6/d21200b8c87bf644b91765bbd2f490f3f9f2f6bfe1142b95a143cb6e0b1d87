# KinDB

**Embeddable graph engine for code-aware tools.**

KinDB is a purpose-built, embeddable graph database in Rust. It provides the storage, indexing, and retrieval substrate for the [Kin](https://github.com/anthropics/kin) semantic version control system and is designed to be usable independently by any tool that needs a fast, typed code graph.

> **Alpha** -- APIs will evolve. The core engine is proven: it powers Kin's 1,400+ test suite and validated benchmark sweeps.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Rust](https://img.shields.io/badge/Rust-2021_edition-orange.svg)](https://www.rust-lang.org/)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-yellow.svg)](#status)

---

## What KinDB Does

- **In-memory graph engine** with HashMap-based adjacency lists and compiled Rust queries (no query language, zero parsing overhead)
- **Snapshot persistence** with mmap and RCU snapshots (single writer, concurrent readers, zero-blocking reads)
- **Full-text search** via Tantivy
- **Vector similarity search** via HNSW index
- **Incremental indexing** for graph updates without full rebuilds
- **Static schema** -- Entity and Relation types known at compile time

---

## Quick Start

```bash
# Prerequisites: Rust 1.75+
git clone https://github.com/anthropics/kin-db.git
cd kin-db
cargo build --release

# Run tests
cargo test -p kin-db
```

---

## Repo Layout

```
crates/kin-db/       # Main Rust crate
  src/
    types.rs         # Core types: Entity, Relation, EntityKind, RelationKind
    store.rs         # GraphStore trait
    engine/          # In-memory graph, indexes, traversal
    storage/         # mmap persistence, RCU snapshots
    vector/          # HNSW vector similarity search
    search/          # Full-text search via Tantivy
docs/
  ARCHITECTURE.md    # Storage and engine design
  EVALUATION.md      # Database comparison that led to building KinDB
  ZERO_COPY_PLAN.md  # Performance and memory direction
```

Optional feature flags in `crates/kin-db/Cargo.toml` enable Metal, CUDA, and Accelerate-backed embedding/runtime paths.

---

## Design Principles

- **Batch write, continuous read** -- optimized for bulk indexing (like `kin commit`) followed by many reads.
- **No query language** -- all queries are compiled Rust functions. No parsing overhead, no runtime interpretation.
- **Static schema** -- Entity/Relation types known at compile time. No runtime schema discovery.
- **Narrow scope** -- storage and low-level query capability live here. Semantic rules, review logic, and ranking strategy belong in higher layers.

---

## Status

**What's solid:**
- In-memory graph with snapshot persistence
- Concurrent read access via RCU
- Tantivy-backed full-text search
- Vector similarity search
- Used as the storage engine for Kin's full test and benchmark suite

**What's evolving:**
- Embedding and vector-search tuning
- Zero-copy read path optimizations
- API surface for standalone use outside Kin

---

## Ecosystem

KinDB is the storage substrate for the Kin ecosystem:

| Component | Description |
|-----------|-------------|
| **[kin](https://github.com/anthropics/kin)** | Semantic VCS -- primary consumer of KinDB |
| **[kin-db](https://github.com/anthropics/kin-db)** | Embeddable graph engine (this repo) |
| **[kin-stack](https://github.com/anthropics/kin-stack)** | Orchestration, benchmarking, and proof tooling |
| **kin-code** | Editor shell |
| **kin-pilot** | Agent shell |
| **[KinLab](https://kinlab.ai)** | Hosted collaboration layer |

KinDB exists as a separate repo because storage, indexing, and retrieval are foundational concerns that sit below all product layers. Higher-level tools should consume Kin semantics through stable boundaries, not by embedding product logic into the database layer.

---

## Contributing

Contributions welcome. Please open an issue before submitting large changes.

## License

Apache-2.0.

---

Built by [Firelock, LLC](https://firelock.ai).
