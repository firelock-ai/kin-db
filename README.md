# kin-db

Semantic graph storage engine for the Kin stack.

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

## License

Apache-2.0. Part of the open Kin local substrate.
