# kin-db Concurrency Model

## Overview

kin-db supports three concurrency modes, each targeting a different deployment scenario:

1. **Single-process** -- InMemoryGraph with sharded RwLocks (CLI, single daemon)
2. **Multi-process with flock** -- SnapshotManager with OS-level file locks (multiple CLI invocations)
3. **Cloud with CAS** -- StorageBackend with generation-based compare-and-swap (GKE pods)

## Mode 1: Single-process (InMemoryGraph)

The most common mode. A single process (kin CLI, kin-daemon, or kin-mcp) owns an `InMemoryGraph` in memory.

### 6 Sharded RwLocks

`InMemoryGraph` partitions its data into six domain-specific sub-stores, each behind its own `parking_lot::RwLock`:

| Lock | Domain | Contents |
|------|--------|----------|
| `entities` | Core graph | entities, relations, outgoing/incoming edge maps, indexes, file hashes, shallow files |
| `changes` | Change DAG | semantic changes, change children, branches |
| `work` | Work graph | work items, annotations, work links |
| `verification` | Verification | test cases, assertions, runs, coverage edges, mock hints, contracts |
| `provenance` | Provenance | actors, delegations, approvals, audit events |
| `sessions` | Sessions | agent sessions, intents, downstream warnings |

### Why sharding matters

Independent domains can proceed without contention. For example:
- A session heartbeat (sessions lock) does not block an entity lookup (entities lock).
- Creating a work item (work lock) does not block a verification run query (verification lock).

### Lock ordering

When acquiring multiple locks, the order is fixed to prevent deadlocks:

```
entities -> changes -> work -> verification -> provenance -> sessions
```

In practice, most operations only touch one lock. The few operations that cross domains (e.g., `to_snapshot`, `from_snapshot`) acquire locks in this order.

### Reader/writer semantics

- **Readers:** `RwLock::read()` -- multiple concurrent readers allowed per shard.
- **Writers:** `RwLock::write()` -- exclusive access to the shard. Writers block readers and other writers on the same shard only.
- **No async:** All locks are synchronous `parking_lot::RwLock`. Do not hold them across `.await` points.

## Mode 2: Multi-process with flock (SnapshotManager)

When multiple processes might access the same `.kin/kindb/graph.kndb` file (e.g., concurrent CLI invocations, daemon + CLI), `SnapshotManager` provides process-level coordination.

### Exclusive file lock

On `SnapshotManager::open()`, an OS-level exclusive file lock (`fs2::FileExt::try_lock_exclusive`) is acquired on `graph.lock` adjacent to the snapshot file. This lock is held for the lifetime of the `SnapshotManager` and released when dropped.

```
.kin/kindb/
  graph.kndb       -- MessagePack snapshot
  graph.lock       -- flock sentinel (held by owning process)
  graph.tmp        -- recovery candidate (mid-write)
  graph.tmp.meta   -- recovery marker (SHA-256 + byte length)
```

If another process tries to `SnapshotManager::open()` while the lock is held, it receives a `LockError` immediately (non-blocking `try_lock_exclusive`).

### RCU snapshot swaps

Within the owning process, `SnapshotManager` provides read-copy-update semantics:

1. **Read:** `graph()` returns `Arc<InMemoryGraph>` -- cheap clone, no locking. Readers hold an `Arc` and can continue reading even after a swap.
2. **Write:** Build changes on the current graph (interior-mutable via RwLocks), then `save()` atomically writes to disk.
3. **Swap:** `swap(new_graph)` replaces the `Arc<InMemoryGraph>` under a brief `RwLock::write()`. Old readers continue with their existing `Arc`; new readers see the new graph.

### Atomic writes and crash recovery

`save()` uses a two-phase write protocol:

1. Write snapshot to `graph.tmp` with fsync.
2. Write recovery marker to `graph.tmp.meta` with fsync (SHA-256 checksum + byte length).
3. Rename `graph.tmp` to `graph.kndb` (atomic on POSIX).
4. Remove `graph.tmp.meta`.

If the process crashes mid-write:
- If `graph.kndb` exists and is valid: use it (crash was after rename).
- If `graph.kndb` is missing/corrupt and `graph.tmp` + `graph.tmp.meta` exist: validate the tmp against the marker and promote it.
- If the marker checksum does not match: reject the recovery candidate as incomplete.

### Generation marker

A generation counter (`u64`) is maintained alongside the snapshot. Each successful `save_snapshot` increments the generation. This enables:

- **Staleness detection:** A process can check if its view is outdated by comparing its generation against the current on-disk generation.
- **Compare-and-swap:** `StorageBackend::save_snapshot(expected_gen)` fails if another writer has advanced the generation.

## Mode 3: Cloud with CAS (GCS StorageBackend)

For cloud deployment (GKE), the `GcsBackend` implementation of `StorageBackend` uses GCS object generation numbers as the compare-and-swap primitive.

### How it works

- `load_snapshot` returns `(bytes, generation)` where `generation` is the GCS object generation.
- `save_snapshot(expected_gen)` uses GCS precondition `if_generation_match` to atomically reject writes if the generation has changed.
- No file locks needed -- GCS provides the coordination.

### Delta support

The `StorageBackend` trait supports incremental deltas:

- `save_delta(base_gen)` stores a `GraphSnapshotDelta` alongside the base snapshot.
- `load_deltas_since(since_gen)` returns all deltas since a given generation.
- `compact_deltas()` merges all deltas into the base snapshot and clears the delta journal.

This reduces write amplification: instead of saving the full snapshot on every change, only the delta is written. Periodic compaction merges accumulated deltas.

## Concurrency safety guarantees

| Scenario | Mechanism | Guarantee |
|----------|-----------|-----------|
| Single process, multiple threads reading | RwLock::read (multiple concurrent) | Safe |
| Single process, thread writing while others read | RwLock::write blocks readers on same shard only | Safe, minimal contention |
| Two processes, same snapshot file | flock exclusive on .lock file | Second open fails with LockError |
| Cloud, two pods writing same repo | GCS generation-match precondition | Second write fails with generation mismatch |
| Process crash mid-write | Recovery marker + atomic rename | No corruption, automatic recovery on next open |
| Process crash mid-write, no marker | tmp file ignored, primary used | Falls back to last good snapshot |
