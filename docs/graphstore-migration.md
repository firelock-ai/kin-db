# GraphStore Sub-trait Migration Guide

## Background

The `GraphStore` trait in `kin-model` started as a single monolithic trait with 60+ methods spanning entities, changes, work items, verification, and provenance. Pass 1 of the substrate narrowing introduced five focused sub-traits that partition this surface by domain:

| Sub-trait | Domain | Methods |
|-----------|--------|---------|
| `EntityStore` | Entities, relations, graph traversal | 13 |
| `ChangeStore` | Semantic change DAG, branches | 10 |
| `WorkStore` | Work items, annotations, links | 16 |
| `VerificationStore` | Tests, coverage, contracts, runs | 20 |
| `ProvenanceStore` | Actors, delegations, approvals, audit | 9 |

All five sub-traits live in `kin-model::graph` alongside the original `GraphStore`.

## Why narrow?

1. **Smaller dependency surface.** A crate that only reads entities should not depend on the 20-method verification API. Narrower bounds make it clear what a consumer actually uses.
2. **Easier testing.** Mock implementations for 13 methods (EntityStore) are far simpler than mocking 60+ methods (GraphStore).
3. **Independent evolution.** Adding a new verification method does not force recompilation of crates that only use EntityStore.

## How to migrate

### Step 1: Identify which domains your crate uses

Audit your crate's usage of `GraphStore`:

```bash
grep -rn 'graph\.\|store\.\|GraphStore' crates/your-crate/src/
```

Categorize each call:
- Entity/relation CRUD, traversal, filtering -> `EntityStore`
- Change DAG, branches -> `ChangeStore`
- Work items, annotations, links -> `WorkStore`
- Tests, coverage, contracts -> `VerificationStore`
- Actors, delegations, approvals -> `ProvenanceStore`

### Step 2: Replace trait bounds

**Before:**
```rust
fn do_something<G: GraphStore>(graph: &G) -> Result<(), G::Error> {
    let entity = graph.get_entity(&id)?;
    let relations = graph.get_relations(&id, &[RelationKind::Calls])?;
    // ...
}
```

**After:**
```rust
use kin_model::EntityStore;

fn do_something<G: EntityStore>(graph: &G) -> Result<(), G::Error> {
    let entity = graph.get_entity(&id)?;
    let relations = graph.get_relations(&id, &[RelationKind::Calls])?;
    // ...
}
```

### Step 3: Use multiple bounds when needed

If a function touches multiple domains, use `+` bounds:

```rust
use kin_model::{EntityStore, ChangeStore};

fn trace_entity_history<G: EntityStore + ChangeStore>(
    graph: &G,
    id: &EntityId,
) -> Result<(), G::Error> {
    let entity = graph.get_entity(id)?;
    let history = graph.get_entity_history(id)?;
    // ...
}
```

### Step 4: Verify InMemoryGraph still satisfies your bounds

`InMemoryGraph` implements all five sub-traits (via its `GraphStore` implementation). No changes needed on the implementation side.

### Backward compatibility

The full `GraphStore` trait remains and is not deprecated. Code that needs the entire surface (e.g., snapshot serialization, full graph cloning) should continue using `GraphStore`. The sub-traits are for consumers that can narrow their dependency.

## Sub-trait design principles

All sub-trait methods follow these rules:

1. **Pure CRUD.** Methods store, retrieve, update, or delete data. No side effects, no policy enforcement, no notifications.
2. **ID-based access.** Primary key lookups return `Option<T>`. Missing items are `None`, not errors.
3. **Filter-based listing.** List methods accept a filter struct. Empty/default filter means "return all."
4. **No business logic.** `update_work_status` stores the new status. It does not validate transitions or enforce workflow rules.
5. **Consistent error type.** Each sub-trait has `type Error: std::error::Error + Send + Sync + 'static`.

## Mapping: consumer crate -> recommended sub-traits

| Consumer crate | Primary sub-traits | Notes |
|---------------|-------------------|-------|
| `kin-parser` / `kin-index` | `EntityStore` | Parse and index entities/relations |
| `kin-projection` | `EntityStore` | Read entities for file reconstruction |
| `kin-reconcile` | `EntityStore` | Bidirectional entity/file sync |
| `kin-review` | `EntityStore` + `ChangeStore` | Semantic diff + change history |
| `kin-context` | `EntityStore` | Build context packs from entity graph |
| `kin-mcp` | `GraphStore` (full) | Exposes all domains via MCP tools |
| `kin-daemon` | `GraphStore` (full) | Manages all graph state |
| `kin-remote` | `ChangeStore` + `EntityStore` | Push/pull sync |
| `kin-semantic-contracts` | `EntityStore` + `VerificationStore` | Contract discovery + coverage |
| `kin-bench` | `EntityStore` | Benchmark entity operations |
| `kin-spine` | `EntityStore` | Cross-repo entity federation |
