# Changelog

All notable changes to kin-db will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.13] - 2026-08-06

### Added

- `StorageBackend::with_source_blob_write_batch` writes many immutable bodies
  under one repository authority envelope. A local backend takes the
  repository lock once for the whole session instead of once per content
  address, and issues one set of durability barriers at the flush rather than
  seven per body: three confirming the `source-blobs/sha256/HH` chain, the
  directory barrier inside the no-clobber publication, the acknowledgement
  fsync, and a trailing directory barrier. The per-body fsync that precedes
  the `linkat` naming it does not move, so a directory entry still never
  becomes durable ahead of the bytes it names and a batch that loses power
  before its flush loses names rather than producing torn bodies. Bodies are
  durable when the session returns `Ok`, and readers cannot observe an
  unflushed session because the repository lock excludes them for its whole
  duration. `save_source_blob` keeps its per-body contract unchanged, and the
  default trait implementation writes through it.

## [0.7.11] - 2026-08-01

### Fixed

- Restored native Windows durability and its permanent `windows-latest` CI
  gate. Capability-relative directories are opened by cap-std through
  `NtCreateFile`, but `ReOpenFile` accepts only handles created by
  `CreateFile`; that mismatch made every repository namespace publication fail
  with `ERROR_ACCESS_DENIED`. Windows now opens the pinned ambient name with
  write access and no reparse traversal, binds the reopened handle back to the
  complete retained `FILE_ID_128`, and only then flushes it. Retained directory
  handles continue to omit delete sharing, so the identity-checked reopen
  cannot be displaced while it is selected. Randomized directory stages use
  one transient, identity-checked `CreateFile` handle that does share delete:
  cap-std deliberately strips `FILE_SHARE_DELETE` from every `maybe_dir` open,
  which otherwise makes the retained stage block its own publication rename.
  Path-based atomic writers now use the same Windows parent-directory barrier
  instead of silently skipping it, and no-follow regular-file reads open
  reparse points and directories only far enough to reject every reparse
  attribute and non-file object type, including non-name-surrogate file
  reparse points that Rust otherwise classifies as regular files.
- The derived read index now uses the shared unique-stage atomic writer instead
  of repeatedly truncating `graph.kidx.tmp`. A still-open or memory-mapped
  deterministic stage failed its file flush with `ERROR_ACCESS_DENIED` on
  Windows and cascaded through graph and vector tests; unique stages retain
  exact-byte post-install verification without that alias.

## [0.7.8] - 2026-07-31

### Fixed

- Restored compilation for musl targets. Publishing a retained local directory
  without replacing a competing target issues `renameat2` with
  `RENAME_NOREPLACE`, behind a `target_os = "linux"` gate, but `libc` declares
  that wrapper only for the environments whose C library exports one, which
  through 0.2.186 excluded musl. musl is `target_os = "linux"`, so a musl build
  entered the branch and then failed to resolve the function, and because
  release artifacts are the only musl builds of this crate the error was first
  seen by a downstream release rather than by any test. Environments without the
  wrapper now issue the identical kernel call through `syscall` with
  `SYS_renameat2`, so the no-replace guarantee that makes namespace publication
  a race one writer can win is enforced by the kernel on every Linux
  environment instead of by the presence of a C library wrapper. Behavior on
  glibc, Bionic, Apple platforms, and Windows is unchanged.

  The Linux workflow now cross-checks the musl target, and checks it a second
  time against the libc version consuming releases resolve, because a freshly
  resolved libc declares the wrapper for musl and would hide the difference the
  guard exists to catch.

## [0.7.5] - 2026-07-30

### Fixed

- Moved the durable history-validation record's validator version with what
  `open`'s full validation path accepts, not only with the persisted envelope
  schema. External-reference entry admission and Git projection tree replay were
  both added to the validation an exact record lets an open skip while the
  envelope schema stayed at 3, so records minted before either check existed
  still verified against the validator that added them and those stores reopened
  without ever running the new checks. The version now composes the envelope
  schema with an explicit coverage revision. Every record minted by an earlier
  validator is refused, each affected store pays one full validation on its next
  open, and that open rebinds a record at the current version.

  Opening a store alternately with binaries either side of this change pays a
  full validation every time, because the record version is compared for
  equality and each binary rebinds to its own value. That is inherent to
  equality-versioned records rather than new here, but this is the first change
  to move the version, so it is the first release where it is reachable.

## [0.7.4] - 2026-07-29

### Changed

- Repository admission and every unproven authority open now replay history in
  one first-parent forest pass instead of resolving the whole graph at each
  validated change. The per-change replay walked that change's first-parent chain
  to genesis, cloning every semantic change payload it touched, and rescanned the
  complete relation set after every change, so a linear history cost
  `O(history^2)` in walked changes. Each change is now validated once against the
  state its own lineage published, unwound through its own delta inverses while
  backtracking between branches, with the dangling-endpoint check narrowed to the
  relations a change asserted plus the endpoint nodes it dropped. Entity,
  relation, external-reference, and tree transitions keep the same fail-closed
  refusals.

### Added

- History and Git-projection replay now report `validated/total` and elapsed
  seconds periodically, so a long admission is observable instead of silent.

## [0.7.3] - 2026-07-29

### Fixed

- Updated the exact `kin-model` dependency to 0.7.1 so an unlabelled
  `MergeConflictEntry` retains its positional `label` slot and round-trips
  without decoding the following resolution into the wrong field.

## [0.7.2] - 2026-07-29

### Added

- Added `AuthorityPayloadStats` and
  `RepositoryAuthorityManager::open_with_payload_stats`, returning an
  immutable open receipt for the exact serialized snapshot and acknowledged
  delta bytes selected by coherent recovery. The receipt excludes staged,
  retired, superseded, source-CAS, index, overlay, and backend-metadata bytes
  and fails closed on inconsistent generations, counts, or arithmetic.

## [0.7.0] - 2026-07-28

### Added

- Persisted `kin-model` 0.7 external references as first-class graph endpoints.
  Transactions now add or remove immutable resolver-issued coordinates
  atomically with their relations, reject stale records and dangling endpoints,
  return the referenced records in mixed-node traversal, and preserve them
  across owned, zero-clone, locate-only, and delta-backed reopen paths.

### Changed

- Advanced the fail-closed graph snapshot format to v13 and incremental delta
  format to v5. The external-reference collections are append-only positional
  fields, and old snapshot/delta formats remain rejected instead of being
  reinterpreted.
- Retrieval-authority hashing is now v2 and repo-truth hashing is v5 so both
  bind the full immutable external-reference record, not only a relation
  endpoint ID.

## [0.6.7] - 2026-07-28

### Changed

- Repository-authority reopen can reuse a durable whole-history validation only
  when it binds the exact loaded snapshot bytes, repository, generation,
  validator version, and a journal-free authority. Every mismatch falls back
  to full replay, while structural and content-addressed body validation remain
  unconditional.
- Exact-open phase timings and the trusted-validation decision are observable,
  and the timed 100K-entity hydration guard runs in isolation instead of racing
  the parallel default suite.

## [0.6.6] - 2026-07-28

### Fixed

- `LocalFileBackend::list_repos` no longer hard-fails on a storage root that
  holds the engine's own files. A live `.kin/kindb/` keeps `graph.kndb`,
  `graph.kvec`, `graph.kidx`, `head-generation`, and `generation` beside the
  repository namespaces, and every one of those names is a valid repository id,
  so binding them as namespaces refused with `local directory namespace <path>
  is not a real directory` and the listing failed outright once any of them
  existed. Listing now skips regular files, which can never be a namespace. The
  bind is unchanged for everything else, so a symlink or any other non-directory
  claiming a namespace name is still refused rather than dropped from the
  listing.

### Added

- `LocalFileBackend::probe_pinned_repository_namespace`, a side-effect-free
  identity probe returning the typed `LocalNamespaceProbe`. Revalidating a
  retained binding through `load_snapshot_authority` acquires the repository
  lock, decodes the whole snapshot, and finalizes retired quarantines, so
  callers classifying on its error reported a truncated snapshot, a missing
  lock file, or a quarantined state on an intact namespace as a replaced
  repository. The probe answers the identity question alone: `Retained`,
  `Absent`, `IdentityLost` carrying whether the storage root or the namespace
  moved, and `Unavailable` for everything that says nothing about identity. The
  first probe on a fresh backend still takes the pin, so the ordering the
  authority reads rely on is unchanged.

### Changed

- The two fixtures guarding merge-carrying-second-parent replay now say what
  goes wrong when entity revisions are derived along first-parent lineage alone,
  and name the upstream commit that first reaches the shape. Documentation only,
  no behavior change.

## [0.6.4] - 2026-07-27

### Fixed

- Loading a snapshot whose entity-revision cache is empty now derives revisions
  by reading each change against its first declared parent, the same material
  lineage `resolve_graph_at` replays. The previous derivation replayed the whole
  change DAG as one flat topological sequence, which folded divergent siblings
  into a single state: a merge that restates its second parent's transition was
  reported as `change <id> has stale old payload for entity <id>` even though
  every lineage reaching it is consistent. Repository authority clears that
  cache on every admission, so history replay rebuilt it on each commit and
  refused any transaction carrying merge history. Preconditions are still
  enforced against the state each change was authored on, so an old payload no
  parent published still fails closed.

## [0.6.3] - 2026-07-27

### Added

- Added a public, read-only workspace admission snapshot that returns one
  coherent workspace binding, frozen case behavior, and exact compiled matcher
  from one repository-authority lease. Missing, tampered, or wrong-length CAS
  bodies remain fail-closed during reads.

### Security

- Local storage now retains one identity-pinned directory capability per
  repository namespace. Authority, snapshot, delta, immutable source, overlay,
  and lock operations target that retained capability and reject a repository
  descendant replaced at the same ambient path instead of rebinding to it.
- New repository and internal-surface directories publish from randomized
  stages with no-replace renames. Failed post-publication confirmation retains
  or poisons the exact epoch, and retries must durably re-confirm its containing
  directory before creating a lock, payload, or authority record.
- Windows retains the complete `FILE_ID_128` for directory identity, read from
  a retained handle through `GetFileInformationByHandleEx(FileIdInfo)` rather
  than the unstable `windows_by_handle` metadata accessors, so the crate builds
  on stable Rust and ReFS file IDs stay distinguishable. Pinned directory
  handles are flushed for repository, surface, lock, source-ancestor, and
  digest-entry publication. This replaces the 0.6.1 storage-root identity
  implementation, which read the legacy 64-bit file index by reopening the root
  path. The Windows arms remain unvalidated by CI: adding `windows-latest` to
  the matrix surfaced 99 pre-existing failures in modules this release does not
  touch, so enabling that platform is left to follow-up work rather than landed
  red or silenced.

### Fixed

- Reopen retained directory capabilities with a real access mode before
  fsyncing or locking them. `Dir::open_dir` is opened as `O_PATH` on the
  targets that have it, and an `O_PATH` descriptor rejects both `fsync` and
  `flock` with `EBADF`, so directory publication and repository lock
  acquisition failed on Linux while passing on macOS.

## [0.6.2] - 2026-07-27

### Changed

- Bumped the `kin-model` dependency to 0.6.1.

## [0.6.1] - 2026-07-26

### Fixed

- Read Windows storage-root identity from `BY_HANDLE_FILE_INFORMATION` through
  an open handle instead of the unstable `windows_by_handle` metadata
  accessors, so the crate builds on stable Rust. The root is reopened with the
  exact reach `symlink_metadata` had, and a volume serial that overflows its
  `DWORD` or a filesystem that reports no file ID now fails closed instead of
  pinning an identity that cannot tell two directories apart. Superseded in
  0.6.3 by retained-handle `FILE_ID_128` identity.

## [0.6.0] - 2026-07-26

### Changed

- Persist complete workspace semantic overlays alongside exact trees, validate
  them relative to the workspace base inside repository authority, and
  materialize the same entity/relation state after reopen.
- Reject pre-overlay repository authority and transaction schemas instead of
  silently rebuilding dirty workspace semantics from `base_target`.
- Advance the graph snapshot wire format to v12 and the local-state root
  domain to v2 so tree-only workspace snapshots and pre-overlay root
  identities fail closed rather than masquerading as current authority.

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
