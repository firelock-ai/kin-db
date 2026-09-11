# kin-db

kin-db is the graph storage engine behind Kin, a graph-native code repository for people and AI agents.

The graph storage engine behind Kin. It handles the entities, relationships,
snapshots, and change history that a Kin repository is made of.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Part of Kin](https://img.shields.io/badge/part%20of-Kin-6E56CF.svg)](https://github.com/firelock-ai/kin)

In Kin the graph is the repository. It is not an index over files, and not a
metadata layer sitting beside them. It is where every entity, relation, and
provenance record actually lives, and `kin-db` is what stores it. It owns graph
storage, snapshot persistence, BM25 lexical retrieval, and ANN vector search,
and it composes `kin-infer` for on-device embedding inference. `kin`, the
system of record, and `kin-vfs`, the transparent filesystem projection, both
build on top of it.

Apache-2.0, the same as `kin`.

## What Kin is

**AI writes code. Kin proves what changed.**

Kin is the system of record for AI-written software. Your code is a graph of
entities, relations, and change history rather than a pile of files and diffs,
so agents and humans can ask what a change touches instead of rebuilding that
picture on every run. Provenance and review are part of the record. Kin
coexists with Git, and filesystem projections let supported tools keep working
with ordinary files.

Start at **[firelock-ai/kin](https://github.com/firelock-ai/kin)** and
**[kinlab.ai](https://kinlab.ai)**.

## What kin-db does

`kin-db` owns the canonical graph: entities, relations, provenance, sessions,
and their indexes. It is not a general-purpose graph database. It is built for
Kin's semantic repository model, where every function, type, file, and relation
is a first-class node with a stable identity, a content hash, and verifiable
Merkle ancestry.

It is the lowest authoritative layer in the open Kin local stack. `kin` (the
system of record) and `kin-vfs` (the filesystem projection) build on top of it.
It composes `kin-search` for BM25 lexical retrieval, `kin-vector` for ANN and
embedding retrieval, and `kin-infer` for on-device embedding inference.

`kin-db` is not published on crates.io. It is built from source as part of the
Kin workspace, and the commands below are how you build and test it here.

## Build

```bash
cargo build
cargo test
```

Feature flags of note:

| Flag | Default | Purpose |
|------|---------|---------|
| `vector` | on | HNSW vector index via `kin-vector` |
| `embeddings` | on | On-device embedding inference via `kin-infer`, on CPU unless a GPU backend is also enabled |
| `metal` | off | Apple Metal GPU backend (requires `embeddings`) |
| `accelerate` | off | Apple Accelerate BLAS for the CPU embedding path (requires `embeddings`) |
| `gcs` | off | Google Cloud Storage snapshot backend |
| `sql` | off | SQLite snapshot backend |

## Key types

- `InMemoryGraph`: the live, mutable graph. Implements `GraphStore`,
  `EntityStore`, `ChangeStore`, `SessionStore`, `ProvenanceStore`, and
  `VerificationStore`.
- `TieredGraph`: a hot in-memory tier over a cold memory-mapped snapshot tier,
  with a configurable hot-tier byte budget. Implemented but not yet wired into
  the load/serve path: `SnapshotManager` always deserializes a snapshot fully
  into one `InMemoryGraph`, so the cold tier backs no production read today.
  Treat it as a candidate mechanism rather than a shipping capability.
- `SnapshotManager`: atomic snapshot persistence and swap.
- `GraphSnapshot` / `GraphSnapshotDelta`: serializable full and incremental
  graph states for persistence and sync.
- `RetrievalQuery` / `unified_retrieve`: unified BM25 + vector retrieval
  entry point with ranking policy applied above this layer.
- `MerkleHash` / `compute_graph_root_hash`: entity-integrity primitive over
  entity content and outgoing-relation topology.
- `compute_repo_truth_hash` / `RepoTruthHash`: whole-snapshot content-hash
  verification for tamper detection and citable proof. Covers the change DAG
  and every other truth domain by content, not cardinality, and is independent
  of map and insertion order. Persist `RepoTruthHash` rather than a bare digest:
  it carries `REPO_TRUTH_HASH_VERSION`, so a stored value from an older encoding
  reads as stale format instead of as truth drift.
- `CodeEmbedder`: manages the background embedding worker and interfaces with
  `kin-infer` for on-device model inference.

## Persistence contract

`SnapshotManager` accepts a logical snapshot namespace such as
`.kin/kindb/graph.kndb`. That path is not itself a snapshot payload. Current
local authority is the atomic `graph.kndb.authority.json` manifest, which binds
an immutable generation under `graph.kndb.snapshots/` plus every acknowledged
or retired delta by SHA-256. Raw bytes at `graph.kndb`, an authority manifest
from another version, or an unbound delta fail closed; KinDB does not migrate,
project, or silently repair them.

SQLite accepts only its current non-null snapshot-authority schema. GCS accepts
only the current full-authority envelope and does not expose unbound journal
objects as replayable graph truth.

Persisted vector indexes require both a complete in-index model/root descriptor
and a current metadata sidecar with provider, model, revision, pipeline, graph
root, dimensions, count, and embedder identity. Missing or mismatched identity
never becomes retrieval authority.

## Ecosystem

| Repo | Role |
|------|------|
| [kin](https://github.com/firelock-ai/kin) | Semantic system of record: CLI, daemon, MCP server, projections |
| [kin-vfs](https://github.com/firelock-ai/kin-vfs) | Transparent filesystem projection |
| [kin-editor](https://github.com/firelock-ai/kin-editor) | VS Code extension |
| [kin-lsp](https://github.com/firelock-ai/kin-lsp) | Language-server enrichment boundary |
| [kinlab](https://kinlab.ai) | Hosted collaboration and control plane |

## License

[Apache-2.0](LICENSE). Part of the open Kin local stack.
