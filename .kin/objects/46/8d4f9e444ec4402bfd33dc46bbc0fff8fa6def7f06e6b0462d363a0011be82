# KinDB

KinDB is the graph database engine for the Kin ecosystem.

It exists as a separate repo because graph storage, incremental indexing, vector search, full-text search, and snapshot management are foundational infrastructure below multiple Kin surfaces.

## Role In The Ecosystem

- `kin`
  uses KinDB as its semantic graph storage and query engine
- `kin-code`, `kin-codex`, and future hosted/private surfaces
  should consume Kin semantics through stable boundaries built on top of Kin and KinDB, not by embedding editor- or product-specific logic into the database layer

## Scope

- graph storage formats
- snapshot and concurrency model
- incremental indexing support
- vector similarity search
- full-text search
- code-graph traversal primitives

## Relationship To Other Repos

- `firelock-ai/kin` — semantic VCS and local-first substrate
- `firelock-ai/kin-code` — editor surface that ultimately depends on graph-backed projections
- `firelock-ai/kin-codex` — agent runtime surface that benefits from graph-native retrieval
- `firelock-ai/kin-graph-service` — graph-facing projection service for editor workflows

For deeper design detail, start with [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
