# Database Evaluation for Kin

> Historical note: this is the evaluation memo written before KinDB replaced KuzuDB. It remains useful background for the design tradeoffs, but KinDB is already the current engine in today's alpha.

## Context

Kin needed a graph database that could handle codebases from 3K entities (zod) to 10M+
entities (large monorepos). The earlier KuzuDB prototype worked at small scale but had
fundamental limitations that prevented scaling.

## Requirements

| Requirement              | Weight | Notes                                         |
|--------------------------|--------|-----------------------------------------------|
| Embeddable in Rust       | Must   | Kin is a CLI tool, not a server                |
| Graph traversal          | Must   | Call chains, dependency analysis, dead code    |
| Concurrent reads         | Must   | MCP server + CLI queries simultaneously        |
| Concurrent writes        | Must   | Index while querying during broader public alpha use |
| Vector similarity search | Must   | "Find code similar to X" is a core feature     |
| Full-text search         | High   | Fuzzy entity name matching                     |
| Incremental updates      | Must   | Only re-index changed files on commit          |
| < 1ms read latency       | High   | LLM agents make many sequential queries        |
| Scale to 10M entities    | Must   | Linux kernel, Chromium, monorepos              |
| GPU/SIMD acceleration    | Nice   | Batch embeddings, large-scale vector search    |

## Candidates Evaluated

### KuzuDB v0.11 (Prior Prototype Backend)

- **Type:** Embedded C++ graph database with Rust bindings
- **Query language:** Cypher
- **Strengths:** Purpose-built for graph OLAP, good multi-hop traversal,
  Cypher is expressive for path queries
- **Weaknesses:**
  - Exclusive write lock (no concurrent R/W)
  - No vector search
  - No full-text search
  - Per-invocation overhead (open DB → parse Cypher → execute → close)
  - Graph file scales ~7x entity count (23 MB for 3K entities → projected 75 GB for 10M)
  - No incremental updates
- **Verdict:** Adequate for prototyping, fails at production scale

### SurrealDB 3.0

- **Type:** Rust-native multi-model database (document + graph + vector + FTS)
- **Query language:** SurrealQL
- **Strengths:**
  - Embeddable in Rust as a library
  - Graph relations via RELATE
  - HNSW vector search (8x faster in 3.0)
  - Full-text search
  - MVCC concurrent R/W
  - $23M funding, active development
  - Positioning as "AI agent memory"
- **Weaknesses:**
  - General-purpose → pays for flexibility kin doesn't need
  - Runtime query parsing (SurrealQL)
  - Graph model uses document links, not native adjacency lists
  - Young (3.0 just launched Feb 2026)
  - No GPU acceleration
- **Verdict:** Strong candidate if we didn't want to build our own.
  Main concern is performance overhead from general-purpose abstractions.

### CozoDB v0.7

- **Type:** Rust-native embedded graph-vector database
- **Query language:** Datalog
- **Strengths:**
  - Embeddable (RocksDB/SQLite/in-memory backends)
  - Datalog recursion is ideal for graph traversal
  - HNSW vector search + MinHash-LSH + FTS
  - 250K QPS reads, 100K QPS mixed R/W
  - Transactional
- **Weaknesses:**
  - Datalog learning curve → contributor friction
  - Single maintainer, small community
  - No documented concurrent write support
  - No GPU acceleration
- **Verdict:** Architecturally elegant but community risk is high.
  Datalog is a barrier to contributors.

### Neo4j

- **Type:** JVM-based graph database server
- **Query language:** Cypher
- **Strengths:**
  - Mature, battle-tested
  - Vector indexes (since 5.11)
  - Massive ecosystem
  - Concurrent R/W
- **Weaknesses:**
  - JVM dependency (50+ MB runtime)
  - Not embeddable in Rust
  - Server architecture → network overhead
  - Overkill for a CLI tool
- **Verdict:** Wrong architecture entirely. Kin needs embedded, not client-server.

### IndraDB v3.0

- **Type:** Rust-native embeddable graph database
- **Query language:** Rust API
- **Strengths:**
  - Pure Rust
  - Embeddable as library
  - RocksDB/sled backends
  - Concurrent writes via backend
- **Weaknesses:**
  - No vector search
  - No full-text search
  - Limited traversal algorithms
  - Small community, uncertain maintenance
- **Verdict:** Too limited. Would need significant extension.

### petgraph + usearch + Tantivy

- **Type:** Library combination (not a database)
- **Strengths:**
  - petgraph: battle-tested graph data structures
  - usearch: SIMD-optimized HNSW vector search
  - Tantivy: Rust-native Lucene alternative
  - Maximum performance (no abstraction overhead)
- **Weaknesses:**
  - Not a database — no persistence, no transactions, no concurrency
  - Building on these means building a database from scratch
  - petgraph's generic design adds overhead kin doesn't need
- **Verdict:** These libraries inform KinDB's design but aren't sufficient alone.

## Decision Matrix

| Feature              | KuzuDB | SurrealDB | CozoDB | Neo4j | IndraDB | KinDB  |
|----------------------|--------|-----------|--------|-------|---------|--------|
| Embeddable in Rust   | ~      | Yes       | Yes    | No    | Yes     | Yes    |
| Graph traversal      | Yes    | Yes       | Yes    | Yes   | ~       | Yes    |
| Concurrent reads     | Yes    | Yes       | Yes    | Yes   | Yes     | Yes    |
| Concurrent R/W       | No     | Yes       | ~      | Yes   | Yes     | Yes    |
| Vector search        | No     | Yes       | Yes    | Yes   | No      | Yes    |
| Full-text search     | No     | Yes       | Yes    | Yes   | No      | Yes    |
| Incremental updates  | No     | ~         | ~      | Yes   | ~       | Yes    |
| < 1ms reads          | No     | ~         | ~      | No    | ~       | Yes    |
| 10M entity scale     | No     | ~         | ~      | Yes   | ~       | Yes    |
| GPU/SIMD             | No     | No        | No     | No    | No      | Yes    |
| Code-optimized       | No     | No        | No     | No    | No      | Yes    |
| No query parsing     | No     | No        | No     | No    | Yes     | Yes    |

Legend: Yes = fully supported, ~ = partial/uncertain, No = not supported

## Decision

**We built KinDB** — a purpose-built, embeddable code graph database in Rust.

### Rationale

1. **No existing DB checks all boxes.** SurrealDB comes closest but pays a
   general-purpose tax (query parsing, flexible schema, document model) that
   kin doesn't need.

2. **Code graphs are a specific domain.** The schema is static, access patterns
   are predictable (batch write / continuous read), and data has spatial locality
   (entities cluster by file). A purpose-built engine can exploit all of these.

3. **Sub-millisecond reads are critical.** LLM agents make 5-15 tool calls per
   task, each triggering a graph query. At 30ms per query (KuzuDB), that's
   150-450ms of latency. At <1ms (KinDB), it's <15ms. This directly impacts
   benchmark results.

4. **The GraphStore trait makes migration safe.** KinDB implements the same
   trait as KuzuDB. We can switch backends without changing any calling code.
   If KinDB doesn't work out, we can always fall back to KuzuDB or try SurrealDB.

5. **Vector search isn't optional.** The open-source community will expect
   "find similar code" capabilities. Bolting vector search onto KuzuDB would
   require a separate index anyway — might as well build it into the engine.
