# KinDB Disaster Recovery

> Current-state note: the recovery steps in this document describe today's KinDB
> snapshot format. The delegation-chain section near the end is a design note for a
> future `kin audit --verify-chain` feature and is not implemented in this alpha.

How to detect, diagnose, and recover from data corruption or loss in KinDB.

## Snapshot Storage Layout

KinDB persists the entire graph as a single snapshot file:

```
<repo>/.kin/
  kindb/
    graph.kndb          # Primary snapshot (the graph)
    graph.tmp           # Transient write file (renamed atomically to graph.kndb)
```

- **`graph.kndb`** — The current graph snapshot. Written atomically via tmp+rename so it is always in a consistent state on disk.
- **`graph.tmp`** — Exists only during a write operation. If present after a crash, KinDB now treats it as a recovery candidate: on open, a valid `graph.tmp` is promoted back to `graph.kndb` when the primary snapshot is missing or corrupted.

The snapshot path is computed by `KinLayout::kindb_snapshot_path()` — always `.kin/kindb/graph.kndb` relative to the repository root.

## On-Disk Format

The `.kndb` file uses a binary envelope:

```
[4 bytes]  Magic: "KNDB"
[4 bytes]  Version (little-endian u32): currently 3
[8 bytes]  Body length (little-endian u64)
[N bytes]  MessagePack-serialized GraphSnapshot body
[32 bytes] SHA-256 checksum of the body (v3+ only)
```

### Version History

| Version | Description |
|---------|-------------|
| 1 | Original format (entities, relations, changes, branches only) |
| 2 | Extended format (work items, annotations, tests, actors, sessions, etc.) |
| 3 | Same as v2 + SHA-256 integrity checksum appended after the body |

Older snapshots are loaded transparently — v1 and v2 files are deserialized and missing fields default to empty. Writing always produces v3.

## Detecting Corruption

### Automatic (v3 snapshots)

v3 snapshots include a SHA-256 checksum over the msgpack body. On load, KinDB recomputes the hash and compares it with the stored checksum. A mismatch produces:

```
storage error: snapshot checksum mismatch: file is corrupted
```

This means at least one byte in the snapshot body has been altered since it was written.

### Manual Verification

You can verify a snapshot file manually:

```bash
# Extract body length from the header (bytes 8-15, little-endian u64)
# Then compute SHA-256 of body bytes and compare to trailing 32 bytes
python3 -c "
import struct, hashlib, sys
data = open(sys.argv[1], 'rb').read()
magic, version = data[:4], struct.unpack('<I', data[4:8])[0]
body_len = struct.unpack('<Q', data[8:16])[0]
body = data[16:16+body_len]
stored = data[16+body_len:16+body_len+32]
computed = hashlib.sha256(body).digest()
ok = 'OK' if stored == computed else 'MISMATCH'
print(f'magic={magic} version={version} body_len={body_len} checksum={ok}')
" .kin/kindb/graph.kndb
```

For v1/v2 snapshots (no trailing checksum), basic validation is: the file starts with `KNDB`, version is 1 or 2, and `16 + body_len == file_size`.

## Recovery Procedures

### Scenario 1: Corrupted Snapshot (Checksum Mismatch)

**Symptoms:** `kin` commands fail with `snapshot checksum mismatch: file is corrupted`.

**Steps:**

1. **Let KinDB attempt automatic recovery first.** When `graph.kndb` is corrupted and `graph.tmp` is still valid, opening the repository will now promote the recovery snapshot back into place automatically.

2. **Check for a stale `.tmp` file manually if automatic recovery still fails.** If `graph.tmp` exists alongside `graph.kndb`, a crash may have interrupted an atomic write. The `.tmp` may contain a valid newer snapshot:

   ```bash
   # Verify the tmp file is valid
   python3 -c "..." .kin/kindb/graph.tmp   # use the script above

   # If valid, promote it
   cp .kin/kindb/graph.kndb .kin/kindb/graph.kndb.corrupt
   mv .kin/kindb/graph.tmp .kin/kindb/graph.kndb
   ```

3. **Rebuild from Git history** using `kin migrate`:

   ```bash
   # Back up the corrupted snapshot
   mv .kin/kindb/graph.kndb .kin/kindb/graph.kndb.corrupt

   # Re-initialize and migrate from Git
   kin migrate --depth shallow
   ```

   This scans the repository's source files and rebuilds entities and relations from scratch. Use `--depth deep` for full commit history traversal (slower but recovers change DAG).

4. **Rebuild from source files** (if Git is unavailable):

   ```bash
   # Remove corrupted snapshot
   rm .kin/kindb/graph.kndb

   # Re-initialize Kin in the repo
   kin init
   # Then re-index
   kin commit
   ```

   This creates a fresh graph from the current working tree. Historical changes and audit events will be lost.

### Scenario 2: Missing Snapshot File

**Symptoms:** `kin` commands fail with `failed to open .kin/kindb/graph.kndb: No such file or directory`.

**Steps:**

1. If `graph.tmp` exists, try opening the repo once before rebuilding. KinDB will promote a valid recovery snapshot automatically when `graph.kndb` is missing.

2. If the `.kin/` directory structure is intact and there is no recoverable `graph.tmp`, the snapshot was likely deleted or never created. Run:

   ```bash
   kin init    # creates a fresh empty graph
   kin commit  # indexes the working tree
   ```

3. If rebuilding from a Git repo:

   ```bash
   kin migrate --depth shallow
   ```

### Scenario 3: Entire `.kin/` Directory Lost

**Symptoms:** `kin` reports `not a Kin repository (no .kin/ found)`.

**Steps:**

1. Re-initialize:

   ```bash
   kin init
   kin commit
   ```

2. Or migrate from Git history:

   ```bash
   kin migrate --depth deep
   ```

   This recovers entity lineage and change history from Git commits.

### Scenario 4: Snapshot Loads but Data Seems Wrong

**Symptoms:** Entity counts are unexpectedly low, relations are missing, or queries return stale results.

**Steps:**

1. Check the snapshot version:

   ```bash
   python3 -c "
   import struct, sys
   data = open(sys.argv[1], 'rb').read()
   version = struct.unpack('<I', data[4:8])[0]
   body_len = struct.unpack('<Q', data[8:16])[0]
   print(f'version={version} body_len={body_len} file_size={len(data)}')
   " .kin/kindb/graph.kndb
   ```

2. If the snapshot is v1 or v2 (no checksum), re-save it to upgrade to v3 with integrity checking:

   ```bash
   # Any write operation (commit, etc.) will re-save as v3
   kin commit
   ```

3. If entity/relation counts are wrong, force a full re-index:

   ```bash
   rm .kin/kindb/graph.kndb
   kin init
   kin commit
   ```

## What Is Preserved vs. Lost on Recovery

| Data | `kin migrate --depth shallow` | `kin migrate --depth deep` | `kin init && kin commit` |
|------|-------------------------------|----------------------------|--------------------------|
| Entities (current) | Recovered | Recovered | Recovered |
| Relations | Recovered | Recovered | Recovered |
| Change DAG (history) | Lost | Recovered from Git | Lost |
| Branches | Lost | Recovered | Lost |
| Work items | Lost | Lost | Lost |
| Annotations | Lost | Lost | Lost |
| Test cases & coverage | Lost | Lost | Lost |
| Actors & delegations | Lost | Lost | Lost |
| Audit events | Lost | Lost | Lost |
| Sessions & intents | Lost (ephemeral) | Lost (ephemeral) | Lost (ephemeral) |

Work items, annotations, test metadata, actors, and audit events are graph-only state with no Git equivalent — they cannot be recovered from external sources.

## Standalone Vector Index Recovery

KinDB's persisted HNSW index also has a fail-closed recovery story when it is used directly outside the main snapshot flow.

Typical files:

```
<index-dir>/
  vectors.usearch     # saved HNSW structure
  vectors.keys        # digest-bound EntityId key-map sidecar
  vectors.tmp         # transient sidecar write file
```

- `vectors.keys` is not optional for a non-empty persisted index. Reload rejects a missing sidecar instead of guessing EntityId mappings.
- Reload also rejects stale, corrupted, or out-of-sync sidecars. KinDB compares the saved index digest with the sidecar digest and checks key counts against the loaded index size.
- A stale `vectors.tmp` from a prior interrupted sidecar write is overwritten on the next successful save.

**Symptoms:** vector reload fails with one of:

```text
vector key map ... is missing for non-empty index
vector key map ... does not match index ... (digest mismatch)
vector key map ... is out of sync with index ...
failed to deserialize vector key map ...
```

**Recovery:**

1. Keep the current index files as evidence instead of editing them in place.
2. Rebuild the vector index from the canonical embeddings/source graph.
3. Save the rebuilt index again so KinDB writes a fresh `vectors.keys` sidecar that matches the saved index bytes.

KinDB intentionally fails closed here: the safe outcome is an explicit reload failure, not silently returning the wrong EntityIds from similarity search.

## Delegation Chain Verification (Design Note)

### `kin audit --verify-chain <actor-id>`

**Status:** Not yet implemented. Below is the design for adding delegation chain replay to the audit command.

**Purpose:** Given an actor ID, traverse the full delegation chain and verify each link is valid (exists, not expired, scopes are consistent). Report the chain and any broken links.

### Implementation Plan

**Integration file to modify (in Kin's CLI repo):** `kin/crates/kin-cli/src/commands/audit.rs`

**Changes required:**

1. Add a `--verify-chain <actor-id>` option to the CLI argument parser in Kin's CLI surface (`kin/crates/kin-cli/src/main.rs` or equivalent).

2. Add a `verify_chain` function to `audit.rs`:

```rust
/// Traverse and verify the delegation chain for a given actor.
///
/// Algorithm:
/// 1. Look up the actor by ID.
/// 2. Find all delegations where this actor is the delegate.
/// 3. For each delegation, check:
///    a. The principal actor exists.
///    b. The delegation has not expired (ended_at is None or > now).
///    c. The scope is non-empty (optional strictness).
/// 4. Recurse: find delegations where the principal is a delegate.
/// 5. Continue until reaching an actor with no incoming delegations (the root).
/// 6. Report the full chain and flag any broken links.
pub async fn verify_chain(actor_id_str: String) -> Result<()> {
    let layout = kin_core::KinLayout::discover(&std::env::current_dir()?)
        .ok_or_else(|| anyhow::anyhow!("not a Kin repository"))?;
    let snap = kin_db::SnapshotManager::open(
        crate::backend::kindb_snapshot_path(&layout)
    )?;
    let graph = &*snap.graph();

    let target_id = parse_actor_id(&actor_id_str)?;
    let target = graph.get_actor(&target_id)?
        .ok_or_else(|| anyhow::anyhow!("actor not found: {}", actor_id_str))?;

    println!("Delegation chain for {} ({}):", target.display_name, target.kind);

    // Walk up the chain
    let mut current_id = target_id;
    let mut depth = 0;
    let mut visited = std::collections::HashSet::new();

    loop {
        if !visited.insert(current_id) {
            println!("  CYCLE DETECTED at depth {depth} — actor {current_id} already visited");
            break;
        }

        let actor = graph.get_actor(&current_id)?
            .ok_or_else(|| anyhow::anyhow!("broken chain: actor {} not found", current_id))?;

        // Find delegations where current actor is the delegate
        // (need to scan all actors' delegations — see note below)
        let all_actors = graph.list_actors()?;
        let mut found_principal = false;

        for potential_principal in &all_actors {
            let delegations = graph.get_delegations_for_actor(&potential_principal.actor_id)?;
            for d in &delegations {
                if d.delegate == current_id {
                    let status = if d.ended_at.is_some() { "EXPIRED" } else { "active" };
                    let indent = "  ".repeat(depth);
                    println!("{indent}  <- delegated by {} ({}) [{}]",
                        potential_principal.display_name,
                        potential_principal.kind,
                        status);

                    if d.ended_at.is_some() {
                        println!("{indent}     WARNING: delegation is expired");
                    }

                    current_id = potential_principal.actor_id;
                    found_principal = true;
                    break;
                }
            }
            if found_principal { break; }
        }

        if !found_principal {
            let indent = "  ".repeat(depth);
            println!("{indent}  ROOT: {} ({}) — no incoming delegation",
                actor.display_name, actor.kind);
            break;
        }

        depth += 1;
    }

    Ok(())
}
```

**Note on query efficiency:** The current `get_delegations_for_actor` returns delegations where the actor is the *principal*. For chain traversal, we also need to find delegations where the actor is the *delegate*. Two options:

1. **Quick (scan approach):** Iterate all actors, call `get_delegations_for_actor` for each, filter for `delegate == target`. Works for small actor sets.

2. **Better (add a new GraphStore method):** Add `get_delegations_by_delegate(&self, delegate_id: &ActorId) -> Vec<Delegation>` to the GraphStore trait and InMemoryGraph. This enables O(1) lookups.

The scan approach is sufficient for the first implementation since actor counts will be small (typically < 100 per repo).

## Prevention

1. **Back up `.kin/kindb/graph.kndb` regularly.** It is the single source of truth for the graph.
2. **Use v3 format.** Any snapshot written by the current version of KinDB includes SHA-256 integrity verification. Upgrade older snapshots by running any write operation (`kin commit`).
3. **Atomic writes protect against crash corruption.** KinDB writes to a `.tmp` file, calls `fsync`, then renames atomically. Corruption during write should not affect the existing snapshot.
4. **Do not modify `.kndb` files manually.** The binary format is not designed for external editing.
