> **Umbrella guidance:** the workspace-root `AGENTS.md` is the source of truth for cross-repo thesis, boundaries, and rules. This file is the repo-specific authority for `kin-db`.

# kin-db

The semantic engine: graph storage, snapshots, indexing, text search and vector search as one
embeddable Rust crate. It sits beneath `kin` and owns graph internals only, meaning storage, index
and snapshot persistence plus text and vector retrieval. Repo format, CLI, daemon, MCP, projections,
reconcile and provenance belong to `kin`, not here. `kin-model` is not in this workspace either; it
is an external dependency consumed from the `kin` registry.

`crates/kin-db` is the only workspace member. `types.rs` and `store.rs` re-export kin-model's
canonical types and its `GraphStore` trait, which this crate implements. `engine/`, `storage/`,
`vector/`, `search/` and `embed/` hold the runtime. File paths here are secondary metadata, never
primary identity: entities are addressed by semantic identity and content hash, and no query,
traversal or storage operation keys on a path.

## Gates

`bin/kin-precheck kin-db` from the umbrella root runs what this repo's `.github/workflows/ci.yml`
gates on, against the tree whose path, sha and branch it prints, and refuses with no tally when a
gate could not run. Run it before pushing. The same commands by hand, with `RUSTFLAGS=-Dwarnings`
as CI sets it:

```bash
cargo fmt -- --check
cargo clippy --all-targets -- -D warnings   # plus the -A allow-list; copy it from ci.yml
cargo build --all-targets
HF_HUB_OFFLINE=1 cargo test -- \
  --skip process_embedding_queue_without_embeddings_is_noop \
  --skip default_dimensions_match_default_model \
  --skip test_vector_index_dimension_mismatch_auto_recovery
HF_HUB_OFFLINE=1 cargo test --test scale_and_failure scale_100k_entities -- \
  --ignored --exact --test-threads=1
./scripts/check-windows-nightly-parity.sh
./scripts/check-schema-provenance.sh
```

The clippy allow-list in `ci.yml` is a burn-down list, not a style. A lint allowed there is allowed
only until somebody fixes it, so delete lines from it and never add. The three skipped tests build a
real embedder and need HF network plus nomic weights, which CI cannot fetch, so they run only
locally. The toolchain is pinned in `rust-toolchain.toml` at 1.96.0; bump it deliberately together
with the allow-list, never by `@stable` drift.

## Non-obvious behaviours

**Cargo.lock is gitignored** (`.gitignore:2`), because this is a library workspace. There is no
lockfile to update, and CI keys its cache on the manifests instead.

**Dependencies resolve from the private sparse registry.** `.cargo/config.toml` carries
`[registries.kin] index = "sparse+https://kinlab.ai/registry/cargo/"` and nothing else. Keep path
patches and credentials out of that tracked file so a fresh public clone still resolves. kin-model
is pinned exactly (`kin-model = { version = "=0.7.22", registry = "kin" }`), so needing an
unpublished kin-model change means publishing kin-model first, not patching this file.

**Snapshot format versions come from the contents, not from the binary.**
`crates/kin-db/src/storage/format.rs` holds `CURRENT_VERSION = 14`, `MIN_SUPPORTED_VERSION = 13`,
`MAX_SUPPORTED_VERSION = 14` and `MAGIC = *b"KNDB"`. `wire_version()` writes v13 when a snapshot
carries no materialized graph section and v14 when it does, so a store becomes v14 exactly when it
gains something a v13 reader could not represent, and a store that never gains one stays readable by
every shipped binary. Changing what a snapshot carries changes the version it writes at, which is
what an older binary refuses.

**Every landing on main publishes to the registry.** `.github/workflows/registry-publish.yml` runs
on push to main and calls kin-actions' `cargo-registry-release.yml` with `mint-release-tag: true`,
so a version bump in `Cargo.toml` becomes a published crate and a tag with nobody at a button. The
`release / ...` required contexts come from that workflow. Bump the version only when you mean to
publish, and publish bottom-up: primitives, kin-model, kin-db, then kin.

**`schemas/` is kin-model's schemars output and this repo cannot regenerate it.** No runtime path
reads it, which is how it drifted three minor versions behind the pin with CI green the whole way.
`scripts/check-schema-provenance.sh` compares the recorded generator version against the pin, which
is the part this repo can prove alone.

## Landing

kin-db lands through the hosted merge queue, ruleset `Merge queue on main`, active. The queue mints
the squash commit from the PR title and body with nobody at the merge button, so get both right
before arming. Required contexts on main are `Check & Test (ubuntu-latest)`,
`Check & Test (macos-latest)`, `cargo-deny`, `Linux Build & Test (no Metal)`, `DCO Sign-off`,
`release / Version bump gate`, `release / Registry-only build`, `release / Repo verification`,
`Schema Provenance` and `PR text hygiene`. Commit with `git commit -s`, since `DCO Sign-off` fails
on a missing or mismatched trailer. From the umbrella root, `bin/kin-lane merge enqueue kin-db
<lane> <pr>` records the row and `bin/kin-lane merge land kin-db <lane> <pr>` arms the queue.
