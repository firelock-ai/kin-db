// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Write a segment from a REAL persisted store and print what every column
//! cost, so the layout can be held to the sizes it registered before it was
//! written.
//!
//! Ignored by default: it needs a store on disk and it is a measurement rather
//! than an assertion. Run it as
//!
//! ```text
//! KIN_SEGMENT_SNAPSHOT=<path to a .kndb> KIN_SEGMENT_OUT=<empty dir> \
//!   cargo test -p kin-db --test segment_from_real_store -- --ignored --nocapture
//! ```
//!
//! It reads the snapshot's MATERIALIZED GRAPH SECTION rather than opening the
//! store through the authority. That is deliberate and it is not a shortcut: a
//! repository authority snapshot refuses to carry entity, relation or adjacency
//! domains at all, so the entity graph on a v14 or v16 store lives only in that
//! section. Reading it directly also means this measurement takes no daemon
//! lock, starts no daemon and mutates nothing.

use std::collections::HashMap;
use std::path::PathBuf;

use kin_db::storage::format::GraphSnapshot;
use kin_db::{write_segment, InMemoryGraph};

#[test]
#[ignore = "needs a real store on disk; see the module doc"]
fn write_a_segment_from_a_real_store_and_price_every_column() {
    let snapshot_path = PathBuf::from(
        std::env::var("KIN_SEGMENT_SNAPSHOT").expect("KIN_SEGMENT_SNAPSHOT must name a .kndb"),
    );
    let out = PathBuf::from(
        std::env::var("KIN_SEGMENT_OUT").expect("KIN_SEGMENT_OUT must name an output directory"),
    );

    let read_started = std::time::Instant::now();
    let bytes = std::fs::read(&snapshot_path).expect("the snapshot must be readable");
    let store_bytes = bytes.len();
    let mut snapshot = GraphSnapshot::from_bytes(&bytes).expect("the snapshot must decode");
    let decode_ms = read_started.elapsed().as_millis();
    // Free the file bytes, which is a real saving and a safe one.
    //
    // Dropping the CHANGE MAP here as well was tried and reverted. The graph
    // build refuses without it, `ChangeNotFound(0eabdc10...)`, because a
    // revision names the change that introduced it and the load resolves that
    // name. It also saved almost nothing: `from_bytes` has already decoded the
    // whole body by this line, so the peak is behind us and dropping a domain
    // afterwards only shortens how long it is held.
    drop(bytes);
    println!(
        "store          {} bytes, decoded in {decode_ms} ms, wire version {}",
        store_bytes, snapshot.version
    );

    let section = snapshot
        .materialized_graph
        .take()
        .expect("this measurement needs a v14 or v16 store, which carries a graph section");
    let state = std::sync::Arc::try_unwrap(section)
        .map(|owned| owned.state)
        .unwrap_or_else(|shared| shared.state.clone());

    // Captured before the move below, so the graph built from this same
    // state can be checked against the shape it was resolved with.
    let section_entity_count = state.entities.len();
    let section_relation_count = state.relations.len();
    println!(
        "section        {} entities, {} relations, {} revision keys",
        section_entity_count,
        section_relation_count,
        state.entity_revisions.len()
    );

    // The section carries no adjacency, so derive it exactly as a load would.
    let mut outgoing: HashMap<_, Vec<_>> = HashMap::new();
    let mut incoming: HashMap<_, Vec<_>> = HashMap::new();
    for relation in state.relations.values() {
        if let Some(src) = relation.src.as_entity() {
            outgoing.entry(src).or_default().push(relation.id);
        }
        if let Some(dst) = relation.dst.as_entity() {
            incoming.entry(dst).or_default().push(relation.id);
        }
    }

    snapshot.entities = state.entities;
    snapshot.relations = state.relations;
    snapshot.entity_revisions = state.entity_revisions;
    snapshot.resolved_tree = state.tree;
    snapshot.external_references = state.external_references;
    snapshot.outgoing = outgoing;
    snapshot.incoming = incoming;

    // A populated entity/relation view plus a live `repository_authority`
    // envelope is what `validate_unscoped_history_caches`
    // (kin-db/src/storage/repository.rs) refuses: a multi-ref repository has
    // no single current graph, so an authority snapshot is only allowed to
    // carry prepared state once it names the one change it was resolved at.
    // `MaterializedGraphSection::state` is exactly that: "Exactly what
    // `ChangeStore::resolve_graph_at(resolved_at)` returns" (format.rs), the
    // same resolved-state value `resolve_workspace_base_graph_snapshot`
    // (repository.rs) serves from this same section for a workspace's base.
    // That function's resolved base always sets `repository_authority: None`
    // before handing the snapshot to `InMemoryGraph`, "because a workspace
    // base is not a published authority snapshot: it carries no envelope, so
    // there is no change it can claim to be the resolution at" (same file).
    // `RepositoryAuthorityState::workspace_graph_snapshot` documents the same
    // contract publicly: "The result has no repository authority envelope
    // and is safe to hand to InMemoryGraph". This measurement already holds
    // a resolution at one named change, so it clears the field the same way
    // rather than asking the unscoped-view check to validate a view that is
    // not unscoped.
    snapshot.repository_authority = None;

    let build_started = std::time::Instant::now();
    let graph = InMemoryGraph::from_snapshot_without_text_index(snapshot)
        .expect("the materialized graph must load");
    let build_ms = build_started.elapsed().as_millis();
    println!(
        "graph          {} entities, {} relations, built in {build_ms} ms",
        graph.entity_count(),
        graph.relation_count()
    );
    // Clearing `repository_authority` above must not be allowed to paper over
    // a resolution that quietly came back empty or short: the graph this
    // measurement prices has to be the same shape as the section it was
    // resolved from, not merely "a graph that loaded".
    assert_ne!(
        graph.entity_count(),
        0,
        "the resolved graph is empty; the section it was built from held {section_entity_count} entities"
    );
    assert_eq!(
        graph.entity_count(),
        section_entity_count,
        "graph entity count disagrees with the resolved section it was built from"
    );
    assert_eq!(
        graph.relation_count(),
        section_relation_count,
        "graph relation count disagrees with the resolved section it was built from"
    );

    std::fs::create_dir_all(&out).expect("the output directory must be creatable");
    let write_started = std::time::Instant::now();
    let stats = write_segment(&graph, &out).expect("the segment must write");
    let write_ms = write_started.elapsed().as_millis();

    let entities = stats.shape.entity_count.max(1) as f64;
    println!("\nsegment written in {write_ms} ms");
    println!(
        "shape          entities {} relations {} entity-edges {} paths {} name-keys {} \
         revisions {} changes {}",
        stats.shape.entity_count,
        stats.shape.relation_count,
        stats.shape.entity_edge_count,
        stats.shape.path_count,
        stats.shape.name_key_count,
        stats.revision_count,
        stats.change_count
    );
    println!(
        "\n{:>4}  {:>6}  {:>12}  {:>14}  {:>12}",
        "col", "width", "count", "payload", "B/entity"
    );
    for column in &stats.columns {
        println!(
            "{:>4}  {:>6}  {:>12}  {:>14}  {:>12.2}",
            column.id,
            column.width,
            column.count,
            column.payload_len,
            column.on_disk as f64 / entities
        );
    }
    println!(
        "\nhot            {} bytes, {:.2} per entity",
        stats.hot_bytes,
        stats.hot_bytes as f64 / entities
    );
    println!(
        "segment total  {} bytes, {:.2} per entity, {:.4} of the store it came from",
        stats.total_bytes,
        stats.total_bytes as f64 / entities,
        stats.total_bytes as f64 / store_bytes as f64
    );
}
