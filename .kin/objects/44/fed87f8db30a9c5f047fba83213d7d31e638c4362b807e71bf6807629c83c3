# KinDB

Purpose-built, embeddable code graph database in Rust. Designed for kin — a semantic VCS that replaces file-based version control with a graph of entities and relationships.

## Build

```bash
cargo build
cargo test
```

## Architecture

See `docs/ARCHITECTURE.md` for the full design rationale and `docs/EVALUATION.md` for the database comparison that led to building KinDB.

### Crate: `kin-db`

- `types.rs` — Core types: Entity, Relation, EntityKind, RelationKind, etc.
- `store.rs` — `GraphStore` trait (drop-in compatible with kin's KuzuDB backend)
- `engine/` — In-memory graph with HashMap-based adjacency lists, indexes, and traversal
- `storage/` — mmap persistence with RCU snapshots (single writer / multiple readers)
- `vector/` — HNSW vector similarity search
- `search/` — Full-text search via tantivy

### Key Design Decisions

- **Static schema** — Entity/Relation types known at compile time. No runtime schema parsing.
- **Batch write / continuous read** — Optimized for `kin commit` (bulk write) then many reads.
- **parking_lot::RwLock** — Concurrent readers, exclusive writer. RCU snapshots for zero-blocking reads.
- **hashbrown::HashMap** — Faster than std HashMap for the access patterns here.
- **No query language** — All queries are compiled Rust functions. Zero parsing overhead.

## Testing

```bash
cargo test --lib      # Unit tests (29+ tests)
```
