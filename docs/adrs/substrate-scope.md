# ADR: kin-db Substrate Scope

**Status:** Accepted
**Date:** 2026-03-26

## Context

kin-db has grown from a simple entity/relation store into a multi-domain graph engine covering entities, changes, work items, verification, provenance, and sessions. As the crate surface expands, the boundary between what kin-db should own and what consumers (kin-cli, kin-mcp, kin-daemon, kin-review) should own needs to be explicit.

Without a clear scope definition, business logic migrates into the storage layer, making kin-db harder to test, harder to reason about, and harder to evolve independently of consumer workflows.

## Decision

### kin-db IS: graph storage + retrieval substrate

kin-db provides:

1. **Typed CRUD** -- Create, read, update, delete for all graph-stored types: entities, relations, changes, branches, work items, annotations, verification data, provenance data, and sessions.
2. **Structural queries** -- Graph traversal (BFS neighborhood, downstream impact, incoming relation checks), filtered entity listing, and merge-base computation on the change DAG.
3. **Indexed retrieval** -- Full-text search (tantivy TextIndex), semantic similarity (usearch VectorIndex), and secondary indexes (name, file, kind).
4. **Persistence** -- Snapshot serialization (MessagePack), atomic writes with crash recovery, mmap-backed cold storage, incremental deltas, and pluggable StorageBackend (local filesystem, GCS).
5. **Integrity** -- Merkle DAG for tamper detection, generation-based compare-and-swap for multi-process coordination, flock-based exclusive access.
6. **Concurrency** -- 6 domain-sharded RwLocks within InMemoryGraph, RCU snapshot swaps via SnapshotManager, Arc-based zero-copy reader sharing.

### kin-db is NOT:

1. **Workflow engine** -- Review gates, approval policies, merge conflict resolution strategies, and session lifecycle management are consumer concerns. kin-db stores the data (reviews, approvals, sessions) but does not enforce workflows.
2. **Policy enforcer** -- Push authorization, branch protection rules, who-can-approve-what, and risk-level thresholds belong in kin-review, kin-daemon, or kin-mcp. kin-db never rejects a write based on policy.
3. **UI data model** -- Pagination, view-specific aggregations, dashboard metrics, and presentation transforms belong in kinlab or consumer crates. kin-db returns domain objects, not view models.
4. **Search ranker** -- kin-db provides raw retrieval signals (text scores, vector distances, graph proximity). Combining these signals into a ranked result set is kin-search's responsibility. kin-db exposes the dimensions; kin-search defines the ranking function.
5. **Sync protocol** -- Push/pull negotiation, conflict detection, and delta application sequencing belong in kin-remote. kin-db provides the delta computation and StorageBackend primitives that kin-remote builds on.

### The GraphStore trait surface

`GraphStore` (defined in kin-model) is the canonical trait that kin-db implements. Its methods should be **storage primitives**:

- **Reads:** get by ID, list with filter, structural traversal
- **Writes:** upsert, remove, create links
- **No side effects:** A `create_work_item` call stores the work item. It does not notify subscribers, trigger reconciliation, or enforce status transitions.

Sub-traits (`EntityStore`, `ChangeStore`, `WorkStore`, `VerificationStore`, `ProvenanceStore`) partition the surface by domain so consumers can depend on the narrowest interface they need.

### Where business logic lives

| Concern | Owner | kin-db role |
|---------|-------|-------------|
| Review gate enforcement | kin-review | Stores review data |
| Push authorization | kin-remote | Provides generation CAS |
| Session lifecycle | kin-daemon | Stores session/intent records |
| Search ranking | kin-search | Provides raw index signals |
| File reconciliation | kin-reconcile | Reads/writes entity graph |
| Contract discovery | kin-semantic-contracts | Reads entity/relation graph |
| Context budgeting | kin-context | Reads entity graph + metadata |

## Consequences

- New features that add business logic to kin-db must justify why the logic cannot live in a consumer crate.
- The GraphStore trait should only gain methods that are domain-agnostic storage primitives.
- Consumer crates should depend on the narrowest sub-trait possible, not the full GraphStore.
- kin-db tests verify storage correctness; consumer crates test workflow correctness.
