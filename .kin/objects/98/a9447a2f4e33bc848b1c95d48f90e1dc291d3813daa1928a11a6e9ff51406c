# KinDB

Purpose-built, embeddable code graph database in Rust. Designed for kin — a semantic VCS that replaces file-based version control with a graph of entities and relationships.

## Build

```bash
git clone https://github.com/firelock-ai/kin-db.git
cd kin-db
cargo build
cargo test --workspace
```

## Architecture

See `docs/ARCHITECTURE.md` for the full design rationale and `docs/EVALUATION.md` for the database comparison that led to building KinDB.

### Crates

- `crates/kin-model` — Canonical semantic model crate owned by this repo: entities, relations, layout, and the `GraphStore` trait surface that KinDB implements.
- `crates/kin-db` — Graph engine crate.
  `types.rs` re-exports the canonical `kin-model` types for local compatibility.
  `store.rs` re-exports the local `GraphStore` trait surface.
  `engine/`, `storage/`, `vector/`, and `search/` implement the runtime behavior.

### Key Design Decisions

- **Static schema** — Entity/Relation types known at compile time. No runtime schema parsing.
- **Batch write / continuous read** — Optimized for `kin commit` (bulk write) then many reads.
- **parking_lot::RwLock** — Concurrent readers, exclusive writer. RCU snapshots for zero-blocking reads.
- **hashbrown::HashMap** — Faster than std HashMap for the access patterns here.
- **No query language** — All queries are compiled Rust functions. Zero parsing overhead.

## Testing

```bash
cargo test -p kin-db  # Current alpha test surface for the crate
```
