# Zero-Copy Snapshot: rkyv + mmap

## Problem

Every CLI invocation deserializes a 73MB MessagePack snapshot into memory (~800ms).
This makes search slower than KuzuDB despite KinDB being faster at everything else.

## Solution

Replace MessagePack serialization with rkyv zero-copy archives.
mmap the archive file and access data directly — no deserialization step.

## How rkyv works

rkyv serializes Rust types into a binary format that can be accessed directly
from a byte buffer without any deserialization. The archived representation
(ArchivedEntity, ArchivedRelation, etc.) lives inside the mmap buffer and is
accessed via pointer casts:

```rust
// Current: 800ms (deserialize 73MB)
let snapshot: GraphSnapshot = msgpack::from_slice(&data)?;

// With rkyv: <1ms (mmap + pointer cast)
let mmap = Mmap::map(&file)?;
let snapshot = rkyv::access::<ArchivedGraphSnapshot>(&mmap)?;
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  graph.kndb (mmap'd)                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Entities  │  │  Relations   │  │  Name Index    │ │
│  │ (rkyv)    │  │  (rkyv)      │  │  (rkyv)        │ │
│  └──────────┘  └──────────────┘  └────────────────┘ │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │ Outgoing  │  │  Incoming    │  │  Branches      │ │
│  │ (rkyv)    │  │  (rkyv)      │  │  (rkyv)        │ │
│  └──────────┘  └──────────────┘  └────────────────┘ │
└──────────────────────────────────────────────────────┘
         ↑ mmap (OS page cache manages hot/cold)
         │
    CLI process (< 1ms to "load")
```

## Implementation Steps

### Phase 1: Add rkyv derives to all types (kin-model)
- Add `rkyv::Archive, rkyv::Serialize, rkyv::Deserialize` derives to:
  Entity, EntityId, EntityKind, Relation, RelationKind, RelationId,
  Branch, BranchName, SemanticChange, SemanticChangeId, and all nested types
- This is the bulk of the work — ~30 types need derives

### Phase 2: Create rkyv-based snapshot format (kin-db)
- New `storage/archive.rs` module
- `ArchiveSnapshot` — the rkyv-serializable top-level structure
- `save_archive(path, &InMemoryGraph)` — serialize to rkyv bytes, write to file
- `MmapArchive::open(path)` — mmap the file, validate, return typed reference

### Phase 3: Read-only access from mmap
- `MmapArchive` holds the mmap and provides `&ArchivedGraphSnapshot`
- Implement query methods on `ArchivedGraphSnapshot`:
  - `get_entity(id)` → O(1) hash lookup in archived HashMap
  - `query_entities(filter)` → scan archived entities
  - `get_relations(id)` → O(1) lookup in archived outgoing map
- These operate directly on mmapped data — zero allocation for reads

### Phase 4: Update CLI to use mmap archive
- Replace `SnapshotManager::open()` with `MmapArchive::open()`
- Read-only commands (search, trace, refs, overview) use the mmap path
- Write commands (commit) still deserialize fully, modify, then re-archive

## Expected Performance

| Operation | Current (msgpack) | With rkyv+mmap | Improvement |
|-----------|:-:|:-:|:-:|
| Snapshot "load" | ~800ms | <1ms | 800x |
| get_entity | <1ms | <1ms | same |
| query_entities | ~50ms | ~50ms | same |
| search total | ~826ms | ~50ms | 16x |
| trace total | ~780ms | ~50ms | 15x |

The startup cost drops from 800ms to <1ms. Query time stays the same.
Total per-command time goes from 800ms+ to 50ms.

## File size

rkyv archives are similar in size to MessagePack (sometimes larger due to
alignment padding, sometimes smaller due to no field names). Expect ~70-90MB
for the 119K entity vscode snapshot.

## Dependencies

- `rkyv = { version = "0.8", features = ["std", "bytecheck"] }`
- `bytecheck` for validation of untrusted archives
