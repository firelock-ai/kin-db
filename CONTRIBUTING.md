# Contributing to KinDB

Thank you for your interest in contributing to KinDB. This document covers everything you need to get started.

## Building from Source

### Prerequisites

- **Rust stable** (2021 edition) -- install via [rustup](https://rustup.rs/)
- **C/C++ compiler** -- required for native dependencies
  - macOS: `xcode-select --install`
  - Ubuntu/Debian: `apt install build-essential`
  - Fedora: `dnf install gcc gcc-c++`

### Build

```bash
git clone https://github.com/firelock-ai/kin-db.git
cd kin-db
cargo build
```

This repo now builds standalone. The repo-owned `kin-model` crate lives in this workspace, so no sibling checkout is required.

### Run Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for the kin-db crate
cargo test -p kin-db
```

### Lint

```bash
# Check formatting
cargo fmt -- --check

# Run clippy
cargo clippy --workspace --all-targets --all-features -- -D warnings
```

## Developer Certificate of Origin (DCO)

All contributions to KinDB require a [Developer Certificate of Origin](https://developercertificate.org/) sign-off. By signing off a commit, you certify that you created the contribution or otherwise have the right to submit it under the project's open source license.

Add a `Signed-off-by` line to every commit message:

```
Add HNSW neighbor pruning optimization

Signed-off-by: Your Name <your.email@example.com>
```

You can do this automatically with `git commit -s`.

**Why DCO?** KinDB uses DCO as a lightweight contribution policy to maintain a clear provenance trail for all contributions and to document that they are submitted under the project's licensing terms. DCO is not required by Apache-2.0, but it helps keep contribution history clear for maintainers and future stewards.

## Making Changes

### Branch Strategy

1. Fork the repository and create a branch from `main`.
2. Name branches descriptively: `fix/snapshot-corruption`, `feat/vector-index-pruning`, `docs/architecture-update`.
3. Keep changes focused. One logical change per PR.

### Pull Request Process

1. Ensure all commits are signed off (`git commit -s`). PRs without DCO sign-off will not be merged.
2. Ensure `cargo test` passes locally before opening a PR.
3. Ensure `cargo clippy` produces no warnings.
4. Ensure `cargo fmt` has been applied.
5. Write a clear PR description explaining **what** changed and **why**.
6. Link related issues with `Closes #123` or `Fixes #123`.
7. A maintainer will review your PR. Expect feedback -- this is a complex codebase and we want to get the abstractions right.

### Commit Messages

Write clear, imperative-mood commit messages:

```
Add HNSW neighbor pruning optimization

Reduces vector search latency by 40% for graphs with >10k entities
by pruning distant neighbors during insertion.

Closes #15
```

## Code Style

- Follow standard Rust conventions. Run `cargo fmt` before committing.
- Use `clippy` as your lint guide -- treat warnings as errors.
- Prefer explicit types over complex inference chains in public APIs.
- Error handling: use `thiserror` for library errors.
- Add `#[cfg(test)]` unit tests in the same file as the code they test.

### Module Boundaries

Each module has a specific responsibility. Before adding code, make sure it belongs in the module you're modifying:

- **crates/kin-model/** -- Canonical semantic types, traits, layout, and shared contracts used by KinDB.
- **types.rs** -- Re-exports of the canonical types from `crates/kin-model`.
- **store.rs** -- Re-export of the local `GraphStore` trait surface.
- **engine/** -- In-memory graph, adjacency lists, indexes, traversal.
- **storage/** -- mmap persistence, RCU snapshots.
- **vector/** -- HNSW vector similarity search.
- **search/** -- Full-text search via Tantivy.

## Reporting Issues

### Bug Reports

Use the [bug report template](https://github.com/firelock-ai/kin-db/issues/new?template=bug_report.yml). Include:

- KinDB version
- OS and architecture
- Steps to reproduce
- Expected vs actual behavior
- Relevant logs

### Feature Requests

Use the [feature request template](https://github.com/firelock-ai/kin-db/issues/new?template=feature_request.yml). Describe the problem you're trying to solve, not just the solution you want.

## Good First Issues

Look for issues labeled [`good first issue`](https://github.com/firelock-ai/kin-db/labels/good%20first%20issue). These are scoped to a single module and include enough context to get started.

## Questions?

Open a [discussion](https://github.com/firelock-ai/kin-db/discussions) or ask in the issue tracker.
