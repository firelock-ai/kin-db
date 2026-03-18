# Merkle DAG Tamper Detection Specification

## Overview

KinDB uses a Merkle DAG (Directed Acyclic Graph) structure for cryptographic integrity verification of the entity/relation graph. The Merkle DAG maps directly onto the existing graph topology, enabling efficient tamper detection without requiring a separate data structure.

## Design Rationale: DAG over Linear Chain

A linear Merkle chain (like git's commit chain) would only capture temporal ordering of changes. The entity/relation graph is inherently a DAG -- entities connect to other entities through typed relations. A Merkle DAG mirrors this structure:

- **Structural alignment**: Each graph node has a corresponding Merkle node. No mapping layer needed.
- **Sub-graph verification**: Verify any sub-tree without touching unrelated nodes.
- **Incremental updates**: Mutating one entity only requires rehashing the affected path to the root, not the entire history.
- **Parallel verification**: Independent sub-graphs can be verified concurrently.

## Hash Computation

### Entity Content Hash

```
SHA-256(
  "kin-entity-v1:"
  || Debug(entity.kind) || "|"
  || entity.name || "|"
  || Debug(entity.language) || "|"
  || entity.signature || "|"
  || Debug(entity.visibility) || "|"
  || entity.fingerprint.ast_hash (32 bytes)
  || entity.fingerprint.signature_hash (32 bytes)
  || entity.fingerprint.behavior_hash (32 bytes)
  || ["file:" || entity.file_origin] (if present)
  || "|"
  || ["doc:" || entity.doc_summary] (if present)
  || for each metadata key (sorted): "meta:" || key || "=" || value
)
```

The entity hash captures the semantic identity: kind, name, signature, fingerprint hashes, file origin, visibility, and metadata. It explicitly excludes the entity ID (which is a UUID, not content-derived) and graph-structural fields like `lineage_parent`, `created_in`, and `superseded_by`.

### Relation Hash

```
SHA-256(
  "kin-relation-v1:"
  || Debug(relation.kind) || "|"
  || source_entity_hash (32 bytes)
  || destination_entity_hash (32 bytes)
  || relation.confidence (4 bytes LE)
  || Debug(relation.origin)
)
```

The relation hash binds the relation kind and properties to the content hashes of both endpoint entities. This means tampering with either endpoint entity changes the relation hash.

### Sub-graph Hash

```
SHA-256(
  "kin-subgraph-v1:"
  || entity_content_hash (32 bytes)
  || sorted(outgoing_relation_hashes) (32 bytes each)
)
```

Each outgoing relation hash is computed using `compute_relation_hash`, where the destination hash is itself a recursive sub-graph hash. Sorting relation hashes ensures determinism regardless of relation insertion order.

**Cycle handling**: A cache with sentinel values breaks infinite recursion. When a cycle is detected, the cached sentinel (zero hash) is used for the back-edge, producing a consistent hash.

### Graph Root Hash

```
SHA-256(
  "kin-graph-root-v1:"
  || sorted(all_entity_subgraph_hashes) (32 bytes each)
)
```

The root hash is computed over all entity sub-graph hashes sorted lexicographically. This makes the root deterministic regardless of HashMap iteration order.

## Verification Algorithm

### Single Entity Verification

1. Look up the entity's stored content hash
2. Recompute the content hash from the entity's current fields
3. Compare: match = valid, mismatch = tampered

**Complexity**: O(1) per entity (single SHA-256 computation)

### Sub-graph Verification

1. Start at the root entity
2. Verify the root entity's content hash
3. For each outgoing relation, recursively verify the destination entity
4. Track visited nodes to handle cycles
5. Collect all verified and tampered nodes into a `VerificationReport`

**Complexity**: O(k) where k is the number of nodes in the sub-graph. For tree-shaped sub-graphs, this is O(log n) of the total graph when verifying a single path from root to leaf.

### Full Graph Verification

1. Build hash map from stored hashes
2. For each entity, verify content hash
3. Recompute graph root hash and compare

**Complexity**: O(n) where n is the total entity count

## Integration with Snapshot Format

### Entity Hash Storage

Entity content hashes are stored in a `HashMap<EntityId, [u8; 32]>` that can be built at snapshot save time using `build_entity_hash_map()`. This map is separate from the snapshot body to avoid circular dependencies (the hash depends on entity content, which is in the snapshot).

### Snapshot Save Flow

1. Serialize the graph to `GraphSnapshot`
2. Call `build_entity_hash_map(&snapshot)` to compute all entity hashes
3. Optionally compute `compute_graph_root_hash(&snapshot)` for the root hash
4. Store hashes alongside the snapshot (in the snapshot header or a sidecar)

### Snapshot Load + Verify Flow

1. Deserialize the snapshot (v3 format already verifies body SHA-256 checksum)
2. Load the stored entity hash map
3. Call `verify_subgraph()` or iterate `verify_entity()` for each entity
4. Report any tampered nodes

## Incremental Update Strategy

When an entity is mutated:

1. Call `update_entity_hash(&entity, &mut hash_map)` -- O(1) per entity
2. The sub-graph hashes of ancestors are lazily invalidated (recomputed on next verification)
3. The graph root hash is recomputed only when explicitly requested

This avoids the O(n) cost of rehashing the entire graph on every mutation. Only the directly affected entity hash is updated eagerly; structural hashes (sub-graph, root) are recomputed on demand.

## Security Properties

### What is detected

- **Content tampering**: Any modification to entity fields (name, signature, fingerprint, etc.) produces a different content hash.
- **Relation tampering**: Changing a relation's kind, confidence, or origin changes the relation hash. Changing either endpoint entity changes the relation hash (since endpoint content hashes are inputs).
- **Structural tampering**: Adding or removing relations changes the sub-graph hash of the source entity. Adding or removing entities changes the graph root hash.
- **Cascade detection**: Tampering with a leaf entity changes every ancestor's sub-graph hash up to the root.

### What is NOT detected

- **Reordering within sorted sets**: Relation hashes are sorted, so reordering identical relations is not detectable (by design -- order is not semantically meaningful).
- **ID substitution**: If an attacker replaces an entity's UUID while preserving all content fields, the content hash will match. However, the sub-graph structure would differ since relation endpoints reference specific IDs.
- **Metadata key collision**: If two metadata keys produce the same sorted serialization, they would hash identically. This is mitigated by the `key=value` format with explicit separators.

### Threat Model

The Merkle DAG protects against:

1. **Accidental corruption**: Bit flips, partial writes, filesystem errors
2. **Unauthorized modification**: Changes to entity content without updating hashes
3. **Rollback attacks**: Detected via graph root hash comparison against a trusted reference
4. **Selective tampering**: `verify_subgraph()` pinpoints exactly which nodes are tampered

It does NOT protect against an attacker who can modify both the entity content and its stored hash simultaneously. For that, external signing (e.g., a signed root hash published to an append-only log) would be needed.

## API Reference

| Function | Purpose |
|---|---|
| `compute_entity_hash(entity)` | SHA-256 content hash of a single entity |
| `compute_relation_hash(relation, src_hash, dst_hash)` | SHA-256 hash of a relation with endpoint hashes |
| `compute_subgraph_hash(entity_id, snapshot, cache)` | Recursive sub-graph hash rooted at entity |
| `compute_graph_root_hash(snapshot)` | Root hash of entire graph |
| `verify_entity(entity_id, snapshot, stored_hashes)` | Check one entity against stored hash |
| `verify_subgraph(entity_id, snapshot, stored_hashes)` | Verify sub-graph, return tampered nodes |
| `build_entity_hash_map(snapshot)` | Compute hashes for all entities |
| `update_entity_hash(entity, hash_map)` | Incrementally update one entity's hash |
| `remove_entity_hash(entity_id, hash_map)` | Remove a deleted entity's hash |
