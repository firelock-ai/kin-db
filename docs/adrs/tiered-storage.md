# ADR: Tiered Storage

**Status:** Accepted
**Date:** 2026-03-26

## Context

kin-db must handle graph sizes ranging from small repos (3K entities, ~1 MB) to massive monorepos (10M entities, ~2 GB in-memory). A single loading strategy cannot serve both ends of this spectrum without either wasting memory on small repos or crashing on large ones.

## Decision

### TieredGraph: adaptive hot/cold storage

`TieredGraph` automatically selects a loading strategy based on available system RAM:

| Strategy | Condition | Behavior |
|----------|-----------|----------|
| `FullLoad` | Estimated in-memory size <= 50% of available RAM | Full deserialization into `InMemoryGraph`. All queries are RAM-speed. |
| `MmapBacked` | Estimated in-memory size > 50% of available RAM | The snapshot file is memory-mapped. The OS page cache acts as the tiering engine -- recently accessed pages stay in RAM, cold pages are evicted to disk automatically. |

### How the decision is made

1. On `TieredGraph::open()`, the system detects available RAM via `sysinfo`.
2. The snapshot file size is multiplied by 4x (conservative estimate for MessagePack -> in-memory expansion).
3. If the estimated size fits within the hot tier budget (50% of free RAM, or explicit `max_hot_bytes`), `FullLoad` is used.
4. Otherwise, `MmapBacked` is used.

### Key insight: mmap IS tiered storage

The OS kernel already has a sophisticated page eviction policy (LRU or variants). Rather than building a custom eviction layer, kin-db delegates to the kernel:

- `mmap` maps the snapshot file into virtual memory.
- Frequently accessed entity pages stay resident in RAM (hot).
- Infrequently accessed pages are evicted to disk by the kernel (cold).
- Re-accessing cold data triggers a page fault -- transparent to the application.

This eliminates the need for explicit cache management, eviction policies, or cache coherence logic within kin-db.

### MmapBacked save reconciliation

When `MmapBacked` is active and the hot tier receives writes, saving requires reconciling the hot tier changes with the cold mmap data. `TieredGraph` tracks a `ManagedHotScope` -- the set of entity IDs, relation IDs, and branch names that have been written to the hot tier. On save:

1. Load the full cold snapshot (from mmap).
2. Overlay hot tier changes onto the cold snapshot.
3. Write the merged snapshot atomically.

### Configuration

```rust
TieredConfig {
    max_hot_bytes: Option<usize>,  // None = auto-detect (50% of free RAM)
    bytes_per_entity: usize,       // Default: 200 bytes (conservative estimate)
}
```

Consumers can override `max_hot_bytes` for testing or constrained environments. `bytes_per_entity` is used for capacity estimates only and does not affect correctness.

## Embedder lifecycle

### What CodeEmbedder does

`CodeEmbedder` generates vector embeddings for code entities using a local BERT model (BGE-small-en-v1.5 by default, 384 dimensions). These embeddings power the `VectorIndex` for semantic similarity search.

### Who owns it

| Component | Creates CodeEmbedder? | Owns model files? |
|-----------|----------------------|-------------------|
| kin-db | Exposes `CodeEmbedder` struct | No -- model cached by HuggingFace Hub |
| kin-daemon | Creates on startup if `embeddings` feature is enabled | Owns the lifecycle |
| kin-cli | May create for one-shot embedding operations | Owns the lifecycle |
| kin-mcp | Does not create -- uses daemon's index | N/A |

### Decision: embeddings are opt-in, consumer-managed

1. **Feature-gated.** `CodeEmbedder` is behind the `embeddings` feature flag. It pulls in `candle-core`, `candle-nn`, `candle-transformers`, `hf-hub`, and `tokenizers` -- heavy dependencies (~130 MB model download on first use).

2. **Consumer-created.** kin-db never auto-creates a `CodeEmbedder`. The consumer (daemon, CLI) creates one and feeds embeddings to `VectorIndex::upsert()`. kin-db stores and searches the vectors but does not generate them.

3. **GPU-optional.** GPU acceleration (Metal, CUDA, Accelerate) is behind additional feature flags (`metal`, `cuda`, `accelerate`). CPU fallback always works.

### VectorIndex lifecycle

`VectorIndex` (usearch HNSW) is separate from `CodeEmbedder`:

- `VectorIndex` is behind the `vector` feature (enabled by default).
- `CodeEmbedder` is behind the `embeddings` feature (not enabled by default in the workspace).
- You can have vector search without local embedding generation -- consumers can provide pre-computed embeddings from an external service.

```
CodeEmbedder (optional) --[generates embeddings]--> VectorIndex (default)
                                                        |
                                                   search_similar()
```

## Consequences

- Small repos get full RAM speed with zero configuration.
- Large repos degrade gracefully to mmap-backed storage without OOM.
- Embedding generation is never a surprise dependency -- consumers opt in explicitly.
- The `vector` feature (search) and `embeddings` feature (generation) are independently controllable.
- No custom eviction logic to maintain -- the OS kernel handles cache management.
