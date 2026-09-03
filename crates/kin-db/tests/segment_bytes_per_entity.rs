// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! The segment layout costs the bytes per entity it was designed to cost.
//!
//! This guard has two halves, and the first is the one that cannot drift.
//!
//! **Every fixed-width column's payload is exactly its registered width times
//! its record count.** A column that gains a field, changes a width or loses
//! its `n + 1` offset entry fails here by name. The widths below are a second
//! copy of the on-disk contract on purpose: a layout change has to be made
//! twice, once in the format and once here, and the PR shows both.
//!
//! **The whole hot working set stays under a per-entity ceiling on a corpus
//! shaped like the real stores.** The corpus reproduces the distributions
//! measured by walking two real persisted stores field by field, rather than
//! numbers chosen to make the assertion pass:
//!
//! | quantity | Linux `fs`/`kernel`/`mm` subtree, 264,615 entities |
//! |---|---|
//! | `name` bytes, mean | 19.92 |
//! | `signature` bytes, mean | 77.17 |
//! | `doc_summary` present | 20.0% of entities, mean 222.58 bytes |
//! | entities per distinct path | 45.5 |
//! | out-degree, mean | 1.43 |
//!
//! The registered prediction for that shape, from the column widths alone, is
//! in the pull request body beside the measured result.

use kin_model::{
    Entity, EntityId, EntityKind, EntityMetadata, EntityRole, EntityStore, FilePathId,
    FingerprintAlgorithm, GraphNodeId, Hash256, LanguageId, Relation, RelationId, RelationKind,
    RelationOrigin, SemanticFingerprint, SourceSpan, Visibility,
};

use kin_db::storage::segment::format::column;
use kin_db::{write_segment, InMemoryGraph};

/// Entities in the corpus. Large enough that the fixed 64 bytes of header and
/// digest per column file is a rounding error rather than the measurement.
const ENTITIES: u32 = 8_192;

/// Measured on the Linux subtree store: 264,615 entities over 5,818 distinct
/// paths.
const ENTITIES_PER_PATH: u32 = 45;

/// Measured: `doc_summary` is present on 52,851 of 264,615 entities.
const DOC_EVERY: u32 = 5;

/// Measured: out-degree mean 1.43 over 264,615 entities.
const EDGES_PER_ENTITY: u32 = 3;
const EDGE_EVERY: u32 = 2;

/// Measured means, rounded to whole bytes.
const NAME_BYTES: usize = 20;
const SIGNATURE_BYTES: usize = 77;
const DOC_BYTES: usize = 223;

/// The registered width of every fixed-width column, and how many records it
/// holds relative to the entity and relation counts.
#[derive(Clone, Copy)]
enum Records {
    Entities,
    EntitiesPlusOne,
    Relations,
    EntityEdges,
}

const FIXED: &[(u32, u32, Records)] = &[
    (column::ENTITY_ID, 16, Records::Entities),
    (column::ENTITY_KIND, 1, Records::Entities),
    (column::ENTITY_LANGUAGE, 1, Records::Entities),
    (column::ENTITY_FLAGS, 1, Records::Entities),
    (column::ENTITY_AST_HASH, 32, Records::Entities),
    (column::ENTITY_PATH_ORD, 4, Records::Entities),
    (column::ENTITY_SPAN, 24, Records::Entities),
    (column::ENTITY_SPAN_PATH_ORD, 4, Records::Entities),
    (column::ENTITY_NAME_OFF, 4, Records::EntitiesPlusOne),
    (column::ENTITY_SIG_OFF, 4, Records::EntitiesPlusOne),
    (column::ENTITY_DOC_OFF, 4, Records::EntitiesPlusOne),
    (column::OUT_OFF, 4, Records::EntitiesPlusOne),
    (column::IN_OFF, 4, Records::EntitiesPlusOne),
    (column::OUT_DST, 4, Records::EntityEdges),
    (column::IN_SRC, 4, Records::EntityEdges),
    (column::IN_REL_ORD, 4, Records::EntityEdges),
    (column::REL_ID, 16, Records::Relations),
    (column::REL_KIND, 1, Records::Relations),
    (column::REL_CONFIDENCE, 4, Records::Relations),
    (column::REL_ORIGIN, 1, Records::Relations),
    (column::REL_SRC, 4, Records::Relations),
    (column::REL_FLAGS, 1, Records::Relations),
    (column::ENTITY_COLD_FLAGS, 1, Records::Entities),
    (column::ENTITY_SIGNATURE_HASH, 32, Records::Entities),
    (column::ENTITY_BEHAVIOR_HASH, 32, Records::Entities),
    (column::ENTITY_EQUIVALENCE_HASH, 32, Records::Entities),
    (column::ENTITY_STABILITY, 4, Records::Entities),
    (column::ENTITY_LINEAGE_PARENT, 16, Records::Entities),
    (column::ENTITY_CREATED_IN, 32, Records::Entities),
    (column::ENTITY_SUPERSEDED_BY, 16, Records::Entities),
    (column::REL_CREATED_IN, 32, Records::Relations),
];

/// Hot bytes per entity this layout is designed to cost on the corpus above.
///
/// Registered from the column widths before the writer existed: 95 bytes of
/// fixed entity columns, 8 of CSR offsets, plus the measured arenas. The
/// ceiling is that prediction with room for the name index and the per-column
/// framing, and it is deliberately tight enough that adding a per-entity
/// column of any real width breaks it.
const HOT_BYTES_PER_ENTITY_CEILING: f64 = 330.0;

fn corpus() -> InMemoryGraph {
    let graph = InMemoryGraph::new();
    let mut ids = Vec::with_capacity(ENTITIES as usize);

    for index in 0..ENTITIES {
        let id = EntityId(uuid::Uuid::from_u128(u128::from(index) + 1));
        ids.push(id);
        let path = format!("kernel/subsystem/module_{:04}.c", index / ENTITIES_PER_PATH);
        let name = pad(&format!("fn_{index}"), NAME_BYTES);
        graph
            .upsert_entity(&Entity {
                id,
                kind: EntityKind::Function,
                name,
                language: LanguageId::C,
                fingerprint: SemanticFingerprint {
                    algorithm: FingerprintAlgorithm::V1TreeSitter,
                    ast_hash: Hash256::from_bytes([1; 32]),
                    signature_hash: Hash256::from_bytes([2; 32]),
                    behavior_hash: Hash256::from_bytes([3; 32]),
                    equivalence_hash: Hash256::from_bytes([4; 32]),
                    stability_score: 0.9,
                },
                file_origin: Some(FilePathId(path.clone())),
                span: Some(SourceSpan {
                    file: FilePathId(path),
                    start_byte: 0,
                    end_byte: 64,
                    start_line: index % 4096,
                    start_col: 0,
                    end_line: index % 4096 + 8,
                    end_col: 1,
                }),
                signature: pad(&format!("static int fn_{index}"), SIGNATURE_BYTES),
                visibility: Visibility::Internal,
                role: EntityRole::Source,
                doc_summary: if index.is_multiple_of(DOC_EVERY) {
                    Some(pad(&format!("doc for fn_{index}"), DOC_BYTES))
                } else {
                    None
                },
                metadata: EntityMetadata::default(),
                lineage_parent: None,
                created_in: None,
                superseded_by: None,
            })
            .unwrap();
    }

    let mut serial = 0u128;
    for (index, source) in ids.iter().enumerate() {
        if !(index as u32).is_multiple_of(EDGE_EVERY) {
            continue;
        }
        for step in 1..=EDGES_PER_ENTITY {
            let target = ids[(index + step as usize) % ids.len()];
            serial += 1;
            graph
                .upsert_relation(&Relation {
                    id: RelationId(uuid::Uuid::from_u128(serial)),
                    kind: RelationKind::Calls,
                    src: GraphNodeId::Entity(*source),
                    dst: GraphNodeId::Entity(target),
                    confidence: 0.8,
                    origin: RelationOrigin::Parsed,
                    created_in: None,
                    import_source: None,
                    evidence: Vec::new(),
                })
                .unwrap();
        }
    }

    graph
}

fn pad(seed: &str, target: usize) -> String {
    let mut text = seed.to_string();
    while text.len() < target {
        text.push('x');
    }
    text.truncate(target.max(seed.len()));
    text
}

#[test]
fn every_fixed_width_column_is_exactly_its_registered_width_times_its_count() {
    let graph = corpus();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();

    let entities = stats.shape.entity_count;
    let relations = stats.shape.relation_count;
    let entity_edges = stats.shape.entity_edge_count;
    assert_eq!(entities, u64::from(ENTITIES));
    assert!(relations > 0, "the corpus must carry relations");

    let mut checked = 0usize;
    for (id, width, records) in FIXED {
        let column = stats
            .columns
            .iter()
            .find(|candidate| candidate.id == *id)
            .unwrap_or_else(|| panic!("segment is missing column {id}"));
        let expected_count = match records {
            Records::Entities => entities,
            Records::EntitiesPlusOne => entities + 1,
            Records::Relations => relations,
            Records::EntityEdges => entity_edges,
        };
        assert_eq!(
            column.width, *width,
            "column {id} declares width {} and the registered width is {width}",
            column.width
        );
        assert_eq!(
            column.count, expected_count,
            "column {id} holds {} records and the layout says {expected_count}",
            column.count
        );
        assert_eq!(
            column.payload_len,
            u64::from(*width) * expected_count,
            "column {id} payload is not its width times its count, so the column carries \
             something the layout does not describe"
        );
        checked += 1;
    }
    assert_eq!(
        checked,
        FIXED.len(),
        "every registered fixed-width column must be present"
    );
}

#[test]
fn the_hot_working_set_stays_under_its_registered_bytes_per_entity() {
    let graph = corpus();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();

    let entities = stats.shape.entity_count as f64;
    let per_entity = stats.hot_bytes as f64 / entities;

    // The positive control: the hot set is a real number rather than an empty
    // sum, and it is a strict subset of the whole segment.
    assert!(
        stats.hot_bytes > 0 && stats.hot_bytes < stats.total_bytes,
        "hot bytes {} must be a non-empty subset of total bytes {}",
        stats.hot_bytes,
        stats.total_bytes
    );
    assert!(
        per_entity < HOT_BYTES_PER_ENTITY_CEILING,
        "the hot working set is {per_entity:.2} bytes per entity and the registered ceiling is \
         {HOT_BYTES_PER_ENTITY_CEILING:.2}; {} hot bytes over {entities} entities",
        stats.hot_bytes
    );

    // And the whole point, stated as an assertion: the metadata side table and
    // the cold hashes are NOT in the hot set, so the hot set is smaller than
    // the segment by at least the cold columns.
    let cold: u64 = stats
        .columns
        .iter()
        .filter(|candidate| {
            matches!(
                candidate.id,
                column::ENTITY_SIGNATURE_HASH
                    | column::ENTITY_BEHAVIOR_HASH
                    | column::ENTITY_EQUIVALENCE_HASH
                    | column::ENTITY_METADATA_ARENA
            )
        })
        .map(|candidate| candidate.on_disk)
        .sum();
    assert!(
        stats.total_bytes - stats.hot_bytes >= cold,
        "the cold columns must be outside the hot working set"
    );
}
