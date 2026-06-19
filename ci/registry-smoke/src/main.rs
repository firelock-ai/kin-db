// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC
//
// Registry-only consumer smoke. Exercises a few kin-db public APIs
// so the published crate is actually compiled and LINKED — not merely resolved
// — when built against the registry. If the published kin-db references a
// dependency version that is not in the registry, this binary fails to build.

fn main() {
    // Construct + round-trip an empty snapshot through the public API, then
    // build the in-memory graph. These symbols span the storage + engine
    // surfaces, forcing a broad slice of the published crate to link.
    let snapshot = kin_db::GraphSnapshot::empty();
    let graph = kin_db::InMemoryGraph::from_snapshot(snapshot);
    std::hint::black_box(&graph);

    println!("kin-db registry-smoke OK — published crate is self-consistent");
}
