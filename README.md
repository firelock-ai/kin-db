# kin-db — The Semantic Engine for Kin

> The semantic engine: graph storage, snapshots, indexing, text + vector search.

The graph is the canonical repository substrate in Kin — not a file index, not a metadata
overlay, but the primary source of truth for every entity, relation, and provenance record.
`kin-db` is that substrate: it owns graph storage, snapshot persistence, BM25 lexical
retrieval, and ANN vector search, and composes `kin-infer` for on-device embedding inference.
`kin` (the system of record) and `kin-vfs` (the transparent filesystem projection) both build
on top of it.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Part of Kin](https://img.shields.io/badge/part%20of-Kin-6E56CF.svg)](https://github.com/firelock-ai/kin)

## What is Kin?

Kin is the system of record for AI-written software — your code as a graph of
entities, relations, and intents, not a pile of files and diffs. AI agents and humans
navigate it semantically, with provenance, review, and governance built in. It coexists
with Git and projects graph truth back to a normal filesystem, so any tool works unchanged.

Start at **[firelock-ai/kin](https://github.com/firelock-ai/kin)** · **[kinlab.ai](https://kinlab.ai)**

## kin-db's role

`kin-db` owns the canonical graph substrate: entities, relations, provenance,
sessions, and their indexes. It is not a general-purpose graph database — it is
built specifically to support Kin's semantic repository model, where every
function, type, file, and relation is a first-class graph node with a stable
identity, content hash, and verifiable Merkle ancestry.

It is the lowest authoritative layer in the open Kin local substrate. `kin`
(the system of record) and `kin-vfs` (the filesystem projection) build on
top of it. It composes `kin-search` for BM25 lexical retrieval, `kin-vector`
for ANN/embedding retrieval, and `kin-infer` for on-device embedding inference.

## Build

```bash
cargo build
cargo test
```

Feature flags of note:

| Flag | Default | Purpose |
|------|---------|---------|
| `vector` | on | HNSW vector index via `kin-vector` |
| `embeddings` | on | GPU embedding via `kin-infer` |
| `metal` | off | Apple Metal GPU backend (requires `embeddings`) |
| `gcs` | off | Google Cloud Storage snapshot backend |

## Key types

- `InMemoryGraph` — the live, mutable graph. Implements `GraphStore`,
  `EntityStore`, `ChangeStore`, `SessionStore`, `ProvenanceStore`, and
  `VerificationStore`.
- `TieredGraph` — tiered storage wrapper: hot in-memory graph over a
  configurable cold backend with configurable memory limits.
- `SnapshotManager` — atomic snapshot persistence and swap.
- `GraphSnapshot` / `GraphSnapshotDelta` — serializable full and incremental
  graph states for persistence and sync.
- `RetrievalQuery` / `unified_retrieve` — unified BM25 + vector retrieval
  entry point with ranking policy applied above this layer.
- `MerkleHash` / `compute_repo_truth_hash` — content-hash verification for
  tamper detection and citable proof.
- `CodeEmbedder` — manages the background embedding worker and interfaces with
  `kin-infer` for on-device model inference.

## Legacy journal rebuilds

Pre-authority delta journals intentionally fail closed. To upgrade one, first
quiesce every old writer and preserve the base plus all journal artifacts. An
operator must reconcile those artifacts into a full graph, then call the
public recovery surface with the last observed authority or legacy generation:

```rust
let committed = backend.rebuild_legacy_journal(repo_id, &reconciled_bytes, expected_generation)?;
// Local SnapshotManager layout:
let (_, committed) = SnapshotManager::rebuild_legacy_journal(path, &reconciled_graph, expected_generation)?;
```

The operation captures the exact journal before a CAS promotion and deletes
only that capture after the new full authority is durable. A cleanup failure
still returns the committed generation; preserve it and retry the explicit
rebuild after rechecking that old writers remain stopped. GCS journal filenames
are historical timestamps, not replay authority, so GCS always requires
caller-reconciled full snapshot bytes.

## Ecosystem

| Repo | Role |
|------|------|
| [kin](https://github.com/firelock-ai/kin) | Semantic system of record — CLI, daemon, MCP server, projections |
| [kin-vfs](https://github.com/firelock-ai/kin-vfs) | Transparent filesystem projection |
| [kin-editor](https://github.com/firelock-ai/kin-editor) | VS Code extension |
| [kin-lsp](https://github.com/firelock-ai/kin-lsp) | Language-server enrichment boundary |
| [kinlab](https://kinlab.ai) | Hosted collaboration and control plane |

## License

[Apache-2.0](LICENSE). Part of the open Kin local substrate.
