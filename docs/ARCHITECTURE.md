# KinDB Architecture

> Historical note: this document started as the KuzuDB-to-KinDB migration design memo. KinDB is now Kin's current graph engine. References to KuzuDB below describe the prior prototype baseline and the rationale for replacing it, not the live backend shipped in today's alpha. The detached `kin-db` repo now also carries the local `crates/kin-model` crate that defines the canonical semantic model and trait surface used here.

## Why We're Building This

Kin is a sovereign semantic VCS that replaces file-based version control with a graph of
semantic entities and relationships. Before KinDB, the prototype stack used KuzuDB
(v0.11), an embedded C++ graph database with Rust bindings. This document captures why
that backend was replaced as Kin moved toward public-alpha scale.

### The Scale Problem

As Kin usage expands to larger public repos, the limits of the earlier KuzuDB prototype
become clear:

| Repo             | Entities   | KuzuDB Graph | Index Time | In-Memory  |
|------------------|------------|-------------|------------|------------|
| zod (today)      | 3,199      | 23 MB       | 38s        | 1 MB       |
| vscode           | ~213,000   | ~1.5 GB     | ~42 min    | ~43 MB     |
| kubernetes       | ~533,000   | ~3.8 GB     | ~1.8 hr    | ~107 MB    |
| chromium         | ~3,800,000 | ~27 GB      | ~13 hr     | ~768 MB    |
| linux kernel     | ~4,265,000 | ~30 GB      | ~14 hr     | ~853 MB    |
| large monorepo   | ~10M       | ~75 GB      | ~35 hr     | ~2.1 GB    |

The Linux kernel has 40 million lines of code (as of Jan 2025, kernel 6.14 rc1).
Chromium has 36 million. These are repos people will try kin on.

### KuzuDB's Breaking Points

1. **Exclusive write lock** — `open()` takes an OS-level file lock. No queries while
   indexing. A `kin commit` on kubernetes takes 1.8 hours during which the graph is
   completely unavailable.

2. **Disk-bound at scale** — A 30 GB graph file for the linux kernel means disk
   thrashing on every query. Random access patterns on spinning disks are catastrophic.

3. **No incremental updates** — Every `kin commit` re-indexes everything. On a large
   repo, changing one file triggers a full rebuild.

4. **No vector search** — Finding "code similar to X" is impossible. The graph stores
   exact structural relationships but has no semantic similarity capability.

5. **Per-invocation overhead** — Each `kin trace` / `kin search` CLI call opens the
   database, parses a Cypher query string, executes, and closes. Connection pooling
   is impossible with the embedded model.

6. **No full-text search** — Entity name lookup requires exact match or pattern glob.
   No fuzzy matching, no typo tolerance, no relevance ranking.

## Why Not a General-Purpose Database?

We evaluated five alternatives before deciding to build KinDB:

### SurrealDB 3.0

Rust-native, embeddable, multi-model (graph + vector + FTS). $23M raised, positioning
as "AI agent memory." Graph queries 8-22x faster in 3.0 vs 2.x.

**Why not:** General-purpose means runtime query parsing (SurrealQL). Schema flexibility
kin doesn't need. Graph model uses RELATE (document links), not native adjacency lists.
No code-graph-specific optimizations.

### CozoDB

Rust-native, embeddable, Datalog queries with built-in HNSW vector search and
MinHash-LSH. 250K QPS reads on commodity hardware.

**Why not:** Datalog is powerful but unfamiliar — contributor friction. Small community
(single maintainer). No concurrent write support documented. Query compilation overhead
for kin's simple lookup patterns.

### Neo4j

Mature, battle-tested, vector indexes (since 5.11), massive ecosystem.

**Why not:** JVM dependency. Not embeddable in Rust. Server architecture adds
network overhead. Overkill for an embedded tool.

### IndraDB

Rust-native, embeddable, RocksDB backend, concurrent writes.

**Why not:** No vector search. No full-text search. Limited graph traversal algorithms.
Small community, uncertain maintenance.

### petgraph + usearch

Pure Rust libraries: petgraph for in-memory graphs, usearch for vector search.

**Why not:** petgraph is a library, not a database. No persistence, no transactions,
no concurrent access management. Building on top of it means building a database anyway.
**However, petgraph's adjacency list design informs KinDB's architecture.**

## Design Principles

### 1. Schema Is Static

Entity types (Function, Class, Method, Interface, Constant, TypeAlias, EnumDef) and
relation types (Calls, Imports, References) are known at compile time. The database
schema never changes at runtime. This means:

- No schema migration code
- No type metadata tables
- Compile-time type safety on all queries
- Fixed-size records where possible

### 2. Batch Write, Continuous Read

Code changes arrive in batches (`kin commit`). Between commits, the graph is 100% reads.
This is the classic OLAP pattern and allows aggressive read optimization:

- Write path: bulk insert with sort-merge, rebuild indexes
- Read path: lock-free, zero-allocation, cache-friendly

### 3. Incremental Updates

Only changed files need re-indexing. KinDB tracks which files were modified since the
last commit and updates only their entities and relations. On a 500K-entity repo where
10 files changed, re-indexing should take seconds, not hours.

### 4. Code Graph Locality

Code entities cluster by file and module. A function's callers are usually in the same
package. KinDB exploits this:

- File-grouped storage (entities from the same file are contiguous in memory)
- Module-aware traversal (prefer same-module edges first)
- Cache-line-friendly node layout

### 5. Vector Search Is First-Class

"Find code similar to X" is a core capability, not a bolt-on:

- Every entity can have an optional embedding vector
- HNSW index for sub-millisecond similarity search on millions of vectors
- Hybrid queries: "find functions that call X AND are similar to Y"

### 6. Concurrent Reads During Writes

Inspired by Linux kernel's Read-Copy-Update (RCU) pattern:

- Readers access an immutable snapshot (zero locking)
- Writer creates a new snapshot in the background
- Atomic pointer swap makes the new snapshot visible to future readers
- Old snapshot is freed when the last reader releases it

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        KinDB API                            │
│   (current graph engine behind the repo-owned kin-model surface)   │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Graph      │  Vector      │  Text        │  Traversal      │
│  Engine     │  Index       │  Search      │  Algorithms     │
│             │  (HNSW)      │  (Tantivy)   │  (BFS/DFS)      │
├─────────────┴──────────────┴──────────────┴─────────────────┤
│                   Snapshot Manager (RCU)                     │
│         (concurrent readers + single background writer)     │
├─────────────────────────────────────────────────────────────┤
│                   Storage Layer (mmap)                       │
│   (MessagePack snapshots with mmap-backed loads today;      │
│            zero-copy/rkyv remains future work)              │
└─────────────────────────────────────────────────────────────┘
```

### Graph Engine

Custom adjacency list representation optimized for code graphs:

```rust
struct InMemoryGraph {
    entities: HashMap<EntityId, Entity>,
    outgoing: HashMap<EntityId, Vec<RelationId>>,  // entity → its dependencies
    incoming: HashMap<EntityId, Vec<RelationId>>,  // entity → its callers
    relations: HashMap<RelationId, Relation>,

    // Indexes for fast lookup
    name_index: HashMap<String, Vec<EntityId>>,    // name → entities
    file_index: HashMap<String, Vec<EntityId>>,    // file path → entities
    kind_index: HashMap<EntityKind, Vec<EntityId>>, // kind → entities
}
```

All read operations are O(1) lookups or O(k) scans where k is the result set size.
No query parsing. No B-tree traversal. Direct HashMap access.

### Vector Index

HNSW (Hierarchical Navigable Small World) index via the usearch crate:

- SIMD-optimized distance computation
- Optional GPU acceleration for batch operations
- Supports 10M+ vectors with sub-millisecond search
- Embeddings stored alongside entities

### Text Search

Tantivy (Rust-native Lucene alternative) for full-text search:

- Indexes entity names, signatures, file paths
- Fuzzy matching with edit distance
- Relevance-ranked results
- Incremental index updates

### Persistence

Memory-mapped snapshot files with mmap-backed loads:

- current snapshot files use MessagePack plus checksum validation
- Atomic save: write to `.tmp`, fsync, rename
- Snapshot-based: each save creates a complete, self-contained file
- Recovery focuses on atomic writes plus rebuild workflows; old snapshots are not retained automatically

### Concurrency

Read-Copy-Update (RCU) inspired by `cloudflare/mmap-sync`:

- Current snapshot behind `Arc<InMemoryGraph>`
- Readers clone the Arc (cheap) and work with immutable data
- Writer builds new graph, swaps the Arc atomically
- Zero reader-side locking

## Performance Targets

| Operation                         | KuzuDB (prototype baseline) | KinDB (current direction) |
|-----------------------------------|-------------------|-----------------|
| `kin trace <name>`                | 5-30ms            | <1ms            |
| `kin search <name>`               | 10-50ms           | <1ms            |
| `kin refs <name>`                 | 50-170ms          | <1ms            |
| `kin commit` (3K entities, fresh) | 38s               | <5s             |
| `kin commit` (500K, incremental)  | 1.8h              | <30s            |
| Vector similarity (1M entities)   | N/A               | <10ms           |
| Concurrent read during write      | BLOCKED           | Yes             |
| Memory (500K entities)            | 3.8 GB on disk    | ~200 MB in-mem  |

## Migration Path

KinDB implements the repo-owned `GraphStore` trait exposed by `crates/kin-model`.
That kept the backend swap surface stable for the earlier `KuzuGraphStore` migration:

```rust
// Before (kin-graph/KuzuDB)
let store = KuzuGraphStore::open_read_only(&layout.graph_dir())?;

// After (kin-db)
let store = SnapshotManager::open(&layout.graph_dir())?;
```

All 20+ trait methods are implemented. The kin CLI, MCP server, and benchmark harness
work unchanged — only the storage backend switches.

## Crate Structure

```
crates/
├── kin-model/
│   └── src/                # Canonical semantic types, layout, and GraphStore trait
└── kin-db/
    └── src/
        ├── lib.rs          # Public exports for graph, search, and snapshot APIs
        ├── types.rs        # Re-exports of canonical types from kin-model
        ├── store.rs        # Re-export of the local GraphStore trait surface
        ├── error.rs        # Error types
        ├── engine/
        │   ├── mod.rs      # In-memory graph module
        │   ├── graph.rs    # InMemoryGraph + GraphStore impl
        │   ├── index.rs    # Name, file, kind indexes
        │   └── traverse.rs # BFS, DFS, dead_code, impact analysis
        ├── vector/
        │   ├── mod.rs      # Vector index module
        │   └── hnsw.rs     # usearch HNSW wrapper
        ├── search/
        │   ├── mod.rs      # Text search module
        │   └── text.rs     # Tantivy full-text search
        └── storage/
            ├── mod.rs      # Persistence module
            ├── format.rs   # Current MessagePack snapshot envelope format
            ├── snapshot.rs # RCU snapshot management
            └── mmap.rs     # Memory-mapped file wrapper
```

## Future Work

- **GPU-accelerated embeddings** — ONNX Runtime for local code embedding models
- **Distributed mode** — Shard the graph across multiple machines for truly massive repos
- **Real-time indexing** — Watch filesystem events and update graph incrementally
- **Compression** — LZ4 compression on the mmap file for reduced disk footprint
- **WAL (Write-Ahead Log)** — For crash recovery with in-flight writes
