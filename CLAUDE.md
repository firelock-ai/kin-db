> **Umbrella guidance:** the workspace-root `AGENTS.md` is the source of truth for cross-repo thesis, boundaries, and rules. This file is the repo-specific authority for `kin-db`.

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

### Crates

- `kin-model` — External dependency consumed from the `kin` cargo registry (canonical repo: `firelock-ai/kin-model`): entities, relations, layout, and the canonical types KinDB builds on. Not part of this workspace.
- `crates/kin-db` — Graph engine crate (the only workspace member).
  `types.rs` re-exports the canonical `kin-model` types for local compatibility.
  `store.rs` re-exports `kin-model`'s `GraphStore` trait surface, which KinDB implements.
  `engine/`, `storage/`, `vector/`, and `search/` implement the runtime behavior.

### Key Design Decisions

- **Static schema** — Entity/Relation types known at compile time. No runtime schema parsing.
- **Batch write / continuous read** — Optimized for `kin commit` (bulk write) then many reads.
- **parking_lot::RwLock** — Concurrent readers, exclusive writer. RCU snapshots for zero-blocking reads.
- **hashbrown::HashMap** — Faster than std HashMap for the access patterns here.
- **No query language** — All queries are compiled Rust functions. Zero parsing overhead.

### File Paths Are Secondary Metadata

File paths in KinDB are strictly secondary metadata, not primary identity. Entities are identified by their semantic identity (name, kind, signature hash) and addressed by content hash. File paths exist only as projection hints for surfaces that need to map graph entities back to filesystem locations. No query, traversal, or storage operation uses file paths as a primary key. This is by design: the graph is the authority, and filesystems are derived views.

## Testing

```bash
cargo test -p kin-db  # Current alpha test surface for the crate
```
