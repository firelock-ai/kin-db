# Changelog

All notable changes to kin-db will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.1] - 2026-07-26

### Changed

- Repository admission now derives an in-memory index of exact
  `(ArtifactId, GitObjectId)` pairs from previously persisted, raw-tree-verified
  Git history. Workspaces and native changes can restore or retarget those
  authenticated Gitlinks after ref movement, authority removal, and reopen,
  while raw objects, unseen targets, copied identities, and same-transaction
  history remain fail-closed.

## [0.5.0] - 2026-07-25

### Changed

- Local snapshot persistence now has one exact authority contract: an atomic
  manifest binds immutable generation payloads and acknowledged or retired
  deltas by digest. The requested `.kndb` path is a logical namespace, not a
  compatibility payload or projection.
- SQLite now requires the current non-null snapshot-authority schema, and GCS
  requires the current full-authority envelope. Neither backend migrates old
  formats or promotes unbound journal objects into graph truth.
- Persisted vector indexes now require a complete model/root descriptor and a
  complete current metadata sidecar, including a non-empty embedder identity.
- Artifact IDs are assigned from graph authority with collision retry, so a
  filesystem path does not seed identity and renames preserve the assigned ID.
- Regenerated the checked-in JSON schemas from the exact current `kin-model`
  schema generator.
- Replaced content-hash-only file tracking with an exact graph-owned working
  tree. Every tracked path now persists its blob identity and materialization
  kind, including regular versus executable files and symbolic links.
- Tree transitions are part of `TransactionDelta` and are validated before any
  graph mutation, so stale add, modify, mode-change, symlink, and removal
  transitions fail atomically instead of partially updating semantic facets.
- Repository-truth hashing now covers the exact working tree, including
  executable and symbolic-link identity.
- Snapshot v9 is one required schema rather than a partially defaulted
  compatibility layout. The zero-copy writer now persists the graph-assigned
  artifact index explicitly, preserving stable artifact identity across
  save/reopen and rename.

### Removed

- Removed the public local-authority repair and journal-rebuild APIs, old
  authority decoders, raw `graph.kndb` compatibility writes, and implicit
  vector-index acceptance paths.
- Removed snapshot formats v1 through v8 and their migration code. Pre-release
  repositories must be reinitialized because the missing file-kind information
  cannot be reconstructed from the old hash-only snapshots.
- Removed the superseded flat verification-link vectors, temporal tombstone
  maps, and persisted change-order cache. Verification edges live only as graph
  relations; revisions and ordering are derived from the change DAG.

## [0.4.0] - 2026-07-25

### Changed

- `compute_repo_truth_hash` now covers the change DAG and every other snapshot
  domain by content: each covered domain is reduced to a sorted multiset of
  per-element digests over a canonical JSON encoding (type tags, length
  prefixes, sorted object keys), so the digest is independent of map iteration
  order, insertion order, and the slice order of vectors materialized from
  maps. Previously the change DAG contributed only its cardinality and roughly
  thirty domains contributed nothing at all, so two graphs whose histories
  disagreed about every entity delta could hash identically. The digest value
  changes for every repository, which is why this release is 0.4.0: a `^0.3`
  consumer must opt into the new hash semantics deliberately.

### Added

- `RepoTruthHash` and `REPO_TRUTH_HASH_VERSION`: callers can persist the
  versioned wrapper so a stored digest from an older encoding reads as stale
  format rather than as truth drift. Exhaustive `GraphSnapshot` destructuring
  makes a newly added domain a compile error until it is either covered or
  explicitly excluded from the digest.

## [0.3.2] - 2026-07-25

### Changed

- Refreshed Kin registry dependency pins; release tags are now minted
  automatically once a version reaches main.

## [0.3.1] - 2026-07-22

### Added

- Bounded immutable source blob reads: `load_source_blob_bounded` rejects an
  object whose trusted metadata exceeds the caller's `max_bytes` budget before
  allocating its body, returning the typed `SourceBlobReadLimitExceeded` error
  (distinct from the backend-wide safety-limit `StorageError`). The local and
  GCS backends enforce the bound on their metadata path, and the trait default
  is fail-closed so a backend without bounded support cannot allocate an
  unbounded legacy result on a security-sensitive path.

### Changed

- `load_source_blob` now delegates to the bounded read with the backend-wide
  `MAX_SOURCE_BLOB_BYTES` safety limit, keeping the unbounded entry point within
  the same allocation boundary.

## [0.3.0] - 2026-07-21

### Added

- Immutable SHA-256-addressed exact-source object storage across local and GCS
  backends, with per-repository namespace isolation and bounded reads.

### Changed

- Raised the storage contract to 0.3.0 and adopted `kin-model` 0.4.0 exact
  source-entry modes and immutable semantic-change identity.
- Duplicate semantic change IDs are now idempotent only for structurally equal,
  IEEE-754-bit-exact payloads; non-finite immutable payloads fail before mutation.

### Security

- Local source objects use descriptor-relative, no-follow publication and reads,
  reject special files and ancestor substitution, preserve no-clobber semantics,
  and reconfirm file, directory, and trust-root durability before acknowledgement.

## [0.2.40] - 2026-07-21

### Changed

- Aligned the workspace and `crates/kin-db` crate manifest versions; no functional
  changes.

## [0.2.39] - 2026-07-20

### Added

- Documented evidence-bound graph authority recovery for operators restoring
  authority from verified legacy artifacts.

### Fixed

- Content-addressed authority recovery.
- Linux CI cache isolation, with tracked cache inputs now hashed.

### Changed

- Public positioning docs realigned to the "proves-the-change" tagline.

## [0.2.38] - 2026-07-16

### Added

- Evidence-bound graph authority recovery.

### Changed

- Polished the kin-db README and removed em dashes from package descriptions.

## [0.2.37] - 2026-07-13

### Changed

- Bumped the `kin-model` dependency to 0.2.5 via the registry dependency-wave
  automation.

## [0.2.36] - 2026-07-12

### Fixed

- Storage snapshot lock release hardening (#103).

### Changed

- Refreshed Kin registry dependency pins.

## [0.2.35] - 2026-07-11

Local storage authority hardened across concurrent writers.

### Fixed

- Snapshot locks are released explicitly, and snapshot authority transitions and
  restarts are made safe.
- Local journals are bound to snapshot authority; local storage authority races,
  mixed-version journal races, and backend journal cleanup are closed, with
  explicit legacy journal rebuilds added.
- A no-vector graph stats assertion.

## [0.2.34] - 2026-07-10

### Added

- Storage snapshots can now recover through delta replay.

### Fixed

- SQLite delta generation cutoffs are tested, and corrupt delta generations are
  rejected.

### Changed

- Docs: corrected kin-model ownership in the architecture description.

## [0.2.33] - 2026-07-10

### Fixed

- The vector sidecar now checkpoints on a throttle during bulk embed.
- Batched graph performance invariants; the deterministic embedding queue
  frontier is cached and batched relation embedding endpoints are invalidated
  correctly.
- Populated `SemanticFingerprint.equivalence_hash` for kin-model 0.2.4.

### Changed

- Docs: locked the public one-liner and category noun.

## [0.2.32] - 2026-07-09

### Added

- Batched semantic change registration.

### Changed

- Bumped `kin-model` to 0.2.3.
- Docs: aligned the category-noun tail with the locked one-liner, and described
  fingerprints by mechanism rather than marketing claims.

## [0.2.31] - 2026-07-08

### Added

- Registry metadata.

### Fixed

- A Hugging Face Hub cache-lock test race.

### Changed

- CI: bot-authored commits are now exempt from DCO sign-off enforcement.

## [0.2.30] - 2026-07-04

### Fixed

- Registry pin refresh; retired dimension-only `VectorIndex::load` callers.

## Early development (0.1.0 – 0.2.29) - 2026-03-14 to 2026-07-03

This range spans the initial KinDB scaffold through the alpha series and the
first thirty 0.2.x patch releases. It is collapsed here; per-commit detail lives
in git history.

### Added

- Initial KinDB scaffold: in-memory graph engine, mmap persistence, vector and
  text search (0.1.0).
- `kin-model` became a version-pinned dependency starting at 0.2.0/0.2.1.
- Registry release automation, throughput-profile embedding budget wiring, and
  adaptive GPU/CPU-twin batch dispatch (0.2.6–0.2.10).
- Canonical ordering for relation/neighborhood queries, and a graph Merkle root
  made independent of adjacency edge order (0.2.17–0.2.19).

### Changed

- Embedding backend hardening: per-load BERT backend selection, an LRU-bounded
  vector cache, tokenizer padding trimming, Metal dispatch memory guards, and an
  honest (non-degenerate) SeqFloor hybrid split (0.2.13–0.2.22).
- Portable static builds: dropped OpenSSL and C++ build dependencies (0.2.24);
  batched dependency bumps for rand/tokenizers/object_store/sysinfo/rustc-hash
  (0.2.25); off-by-default Accelerate feature forwarding for kin-infer BLAS
  (0.2.26).

### Fixed

- Storage/GCS conditional updates now thread object generation correctly, and
  the vector sidecar is gated on a persisted root (0.2.3–0.2.4).
- Embed config tolerance for duplicate `layer_norm_eps`, and an in-memory vector
  cache bound (0.2.2, 0.2.22).

[unreleased]: https://github.com/firelock-ai/kin-db/compare/9dfbe2da3c94...HEAD
[0.3.0]: https://github.com/firelock-ai/kin-db/compare/d46123fe1221...9dfbe2da3c94
[0.2.40]: https://github.com/firelock-ai/kin-db/compare/b8c6ce362afe...d46123fe1221
[0.2.39]: https://github.com/firelock-ai/kin-db/compare/daffd3c56a73...b8c6ce362afe
[0.2.38]: https://github.com/firelock-ai/kin-db/compare/fa8f53168850...daffd3c56a73
[0.2.37]: https://github.com/firelock-ai/kin-db/compare/92aedd0a672b...fa8f53168850
[0.2.36]: https://github.com/firelock-ai/kin-db/compare/ff2abb0479ae...92aedd0a672b
[0.2.35]: https://github.com/firelock-ai/kin-db/compare/2a6bcbd71e0b...ff2abb0479ae
[0.2.34]: https://github.com/firelock-ai/kin-db/compare/70a0367d1b72...2a6bcbd71e0b
[0.2.33]: https://github.com/firelock-ai/kin-db/compare/503ebe3509b1...70a0367d1b72
[0.2.32]: https://github.com/firelock-ai/kin-db/compare/2af3cad39d52...503ebe3509b1
[0.2.31]: https://github.com/firelock-ai/kin-db/compare/31b536592c49...2af3cad39d52
[0.2.30]: https://github.com/firelock-ai/kin-db/compare/40c3297cbcba...31b536592c49
