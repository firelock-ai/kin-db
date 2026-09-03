// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Proofs for the mapped columnar segment.
//!
//! Each test names the property it holds and, where the property is a negative,
//! carries the positive control that must hit. The falsification arms that were
//! run against these are recorded in the pull request body: every arm removes
//! one property of the code and none of a test, and no arm is green across.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use kin_model::{
    CallArgShape, Entity, EntityId, EntityKind, EntityMetadata, EntityRole, EntityStore,
    FilePathId, FingerprintAlgorithm, GraphNodeId, Hash256, LanguageId, Relation, RelationEvidence,
    RelationId, RelationKind, RelationOrigin, SemanticChangeId, SemanticFingerprint, SourceSpan,
    Visibility,
};
use sha2::{Digest, Sha256};

use crate::engine::InMemoryGraph;
use crate::storage::segment::format::{
    column, column_file_name, entity_kind_code, entity_kind_of_code, language_code,
    language_of_code, relation_kind_code, relation_kind_of_code, relation_origin_code,
    relation_origin_of_code, role_code, role_of_code, visibility_code, visibility_of_code,
    DIGEST_LEN, HEADER_LEN, MANIFEST_FILE,
};
use crate::storage::segment::{write_segment, OpenProfile, SegmentReader};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

fn hash(seed: u8) -> Hash256 {
    Hash256::from_bytes([seed; 32])
}

fn entity(seed: u8, name: &str, path: &str) -> Entity {
    let mut raw = [0u8; 16];
    raw[0] = seed;
    raw[15] = 0xA5;
    Entity {
        id: EntityId(uuid::Uuid::from_bytes(raw)),
        kind: EntityKind::Function,
        name: name.to_string(),
        language: LanguageId::Rust,
        fingerprint: SemanticFingerprint {
            algorithm: FingerprintAlgorithm::V1TreeSitter,
            ast_hash: hash(seed),
            signature_hash: hash(seed.wrapping_add(1)),
            behavior_hash: hash(seed.wrapping_add(2)),
            equivalence_hash: hash(seed.wrapping_add(3)),
            stability_score: 0.5 + f32::from(seed) / 512.0,
        },
        file_origin: Some(FilePathId(path.to_string())),
        span: Some(SourceSpan {
            file: FilePathId(path.to_string()),
            start_byte: 10 * usize::from(seed),
            end_byte: 10 * usize::from(seed) + 40,
            start_line: u32::from(seed) + 1,
            start_col: 4,
            end_line: u32::from(seed) + 9,
            end_col: 1,
        }),
        signature: format!("fn {name}(argument: &str) -> usize"),
        visibility: Visibility::Public,
        role: EntityRole::Source,
        doc_summary: Some(format!("what {name} does")),
        metadata: EntityMetadata::default(),
        lineage_parent: None,
        created_in: None,
        superseded_by: None,
    }
}

/// A graph that exercises every optional field in both states, a duplicated
/// name, a repeated path, an entity with several outgoing edges, an entity with
/// none, evidence, and a relation whose endpoint has no ordinal in the segment.
fn fixture() -> (InMemoryGraph, Vec<Entity>, Vec<Relation>) {
    let graph = InMemoryGraph::new();

    let mut alpha = entity(1, "parse_header", "src/parse.rs");
    let mut beta = entity(2, "parse_header", "src/parse.rs");
    let mut gamma = entity(3, "emit", "src/emit.rs");
    let mut delta = entity(4, "Detached", "src/detached.rs");

    // Every optional entity field absent on one entity and present on another,
    // so a flag bit that is wired backwards cannot pass.
    beta.file_origin = None;
    beta.span = None;
    beta.doc_summary = None;
    beta.visibility = Visibility::Crate;
    beta.role = EntityRole::Vendored;
    beta.kind = EntityKind::Macro;
    beta.language = LanguageId::Hcl;

    // A span whose own file differs from `file_origin`, which the layout stores
    // as a second path ordinal rather than assuming the two agree.
    if let Some(span) = gamma.span.as_mut() {
        span.file = FilePathId("src/generated/emit.rs".to_string());
    }
    gamma.metadata.extra.insert(
        "embedding_body_preview".to_string(),
        serde_json::Value::String("fn emit() {}".to_string()),
    );
    gamma.metadata.extra.insert(
        "file_parsed_call_sites".to_string(),
        serde_json::Value::from(26u32),
    );

    delta.lineage_parent = Some(alpha.id);
    delta.created_in = Some(SemanticChangeId(hash(0x7f)));
    delta.superseded_by = Some(beta.id);
    alpha.doc_summary = Some(String::new());

    let entities = vec![alpha.clone(), beta.clone(), gamma.clone(), delta.clone()];
    for item in &entities {
        graph.upsert_entity(item).unwrap();
    }

    // A dangling entity endpoint: an entity id the graph never admitted, which
    // `relation_endpoint_may_be_written` accepts for any `GraphNodeId::Entity`.
    // The writer finds no ordinal for it, so the relation lands in the endpoint
    // side table. That is the same path a genuinely non-entity node takes; a
    // non-entity node is NOT covered here, because admitting one needs a
    // resolved tree this fixture does not build.
    let dangling = EntityId(uuid::Uuid::from_bytes([0xEE; 16]));

    let relations = vec![
        relation(0x11, RelationKind::Calls, alpha.id, gamma.id, 0.9, true),
        relation(
            0x12,
            RelationKind::References,
            alpha.id,
            beta.id,
            0.5,
            false,
        ),
        relation(0x13, RelationKind::Contains, alpha.id, delta.id, 1.0, false),
        relation(0x14, RelationKind::Imports, gamma.id, beta.id, 0.25, false),
        relation(0x15, RelationKind::Calls, gamma.id, dangling, 0.75, false),
    ];
    for item in &relations {
        graph.upsert_relation(item).unwrap();
    }

    (graph, entities, relations)
}

fn relation(
    seed: u8,
    kind: RelationKind,
    src: EntityId,
    dst: EntityId,
    confidence: f32,
    rich: bool,
) -> Relation {
    let mut raw = [0u8; 16];
    raw[0] = seed;
    raw[15] = 0x5A;
    Relation {
        id: RelationId(uuid::Uuid::from_bytes(raw)),
        kind,
        src: GraphNodeId::Entity(src),
        dst: GraphNodeId::Entity(dst),
        confidence,
        origin: if rich {
            RelationOrigin::Lsp
        } else {
            RelationOrigin::Parsed
        },
        created_in: if rich {
            Some(SemanticChangeId(hash(seed)))
        } else {
            None
        },
        import_source: if rich {
            Some("kin_db".to_string())
        } else {
            None
        },
        evidence: if rich {
            vec![RelationEvidence {
                source_span: Some(SourceSpan {
                    file: FilePathId("src/parse.rs".to_string()),
                    start_byte: 4,
                    end_byte: 24,
                    start_line: 2,
                    start_col: 0,
                    end_line: 2,
                    end_col: 20,
                }),
                parser_rule: Some("call_expression".to_string()),
                token: Some("emit".to_string()),
                source_path: Some("./emit".to_string()),
                resolved_path: Some("src/emit.rs".to_string()),
                occurrence_count: 3,
                call_shape: Some(CallArgShape::new(
                    2,
                    vec!["width".to_string(), "height".to_string()],
                    false,
                    true,
                )),
            }]
        } else {
            Vec::new()
        },
    }
}

fn written(dir: &Path) -> SegmentReader {
    SegmentReader::open_with_profile(dir, OpenProfile::Full).unwrap()
}

// ---------------------------------------------------------------------------
// Round trip
// ---------------------------------------------------------------------------

#[test]
fn every_entity_reads_back_byte_identical_through_the_mapped_views() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    assert_eq!(stats.shape.entity_count, entities.len() as u64);

    let reader = written(dir.path());
    let mut seen = 0usize;
    for expected in &entities {
        let ordinal = reader
            .ordinal_of_entity(&expected.id)
            .unwrap()
            .unwrap_or_else(|| panic!("entity {} has no ordinal", expected.id));
        let actual = reader.entity(ordinal).unwrap();
        assert_eq!(
            &actual, expected,
            "entity at ordinal {ordinal} did not read back equal"
        );
        // Byte-identical on the serialized form, not only structurally equal,
        // so a field that survives `PartialEq` by being defaulted on both sides
        // cannot pass.
        //
        // `metadata` is held to CONTENT rather than to bytes, and the reason is
        // a property of the type rather than of this format: `EntityMetadata`
        // wraps a `std::collections::HashMap`, whose iteration order is not
        // part of its value, so two equal maps can serialize to two byte
        // orders. Asserting bytes over it asserts something no writer can
        // promise. CI found this on an entity carrying two metadata keys, and
        // narrowing the claim is the fix rather than deleting it: the other
        // fourteen fields are still held to bytes, and metadata is held to a
        // key-ordered comparison that distinguishes value types, so a `26` that
        // came back as `26.0` still fails.
        let ordered = |entity: &Entity| -> BTreeMap<String, serde_json::Value> {
            entity
                .metadata
                .extra
                .iter()
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect()
        };
        assert_eq!(
            ordered(&actual),
            ordered(expected),
            "entity at ordinal {ordinal} did not carry back the same metadata"
        );

        let mut actual_without = actual.clone();
        let mut expected_without = expected.clone();
        actual_without.metadata = EntityMetadata::default();
        expected_without.metadata = EntityMetadata::default();
        assert_eq!(
            rmp_serde::to_vec(&actual_without).unwrap(),
            rmp_serde::to_vec(&expected_without).unwrap(),
            "entity at ordinal {ordinal} did not serialize identically outside its metadata"
        );
        seen += 1;
    }
    assert_eq!(seen, entities.len(), "every fixture entity must be checked");
}

#[test]
fn every_relation_reads_back_byte_identical_through_the_mapped_views() {
    let (graph, _, relations) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    assert_eq!(stats.shape.relation_count, relations.len() as u64);
    assert_eq!(
        stats.shape.entity_edge_count,
        relations.len() as u64 - 1,
        "exactly one fixture relation has an endpoint with no ordinal"
    );

    let reader = written(dir.path());
    let expected: BTreeMap<_, _> = relations
        .iter()
        .map(|item| (item.id.0.as_bytes().to_vec(), item.clone()))
        .collect();

    let mut seen = BTreeSet::new();
    for ordinal in 0..reader.relation_count() {
        let actual = reader.relation(ordinal).unwrap();
        let key = actual.id.0.as_bytes().to_vec();
        let want = expected
            .get(&key)
            .unwrap_or_else(|| panic!("segment invented relation {} ", actual.id));
        assert_eq!(&actual, want, "relation at ordinal {ordinal} differs");
        assert_eq!(
            rmp_serde::to_vec(&actual).unwrap(),
            rmp_serde::to_vec(want).unwrap(),
            "relation at ordinal {ordinal} did not serialize identically"
        );
        seen.insert(key);
    }
    assert_eq!(
        seen.len(),
        relations.len(),
        "every relation must appear once"
    );
}

// ---------------------------------------------------------------------------
// The queries ReadIndex answers, answered off the mapping
// ---------------------------------------------------------------------------

#[test]
fn the_ordinal_is_the_id_rank_and_the_id_column_is_sorted() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    let mut previous: Option<Vec<u8>> = None;
    for ordinal in 0..reader.entity_count() {
        let id = reader.entity_id(ordinal).unwrap();
        let raw = id.0.as_bytes().to_vec();
        if let Some(before) = &previous {
            assert!(
                before < &raw,
                "id column is not strictly ascending at ordinal {ordinal}, so a binary search \
                 over it cannot be the id index"
            );
        }
        assert_eq!(reader.ordinal_of_entity(&id).unwrap(), Some(ordinal));
        previous = Some(raw);
    }

    // The negative, with the positive control above it: an id the graph never
    // held resolves to no ordinal rather than to a neighbour.
    let absent = EntityId(uuid::Uuid::from_bytes([0xFF; 16]));
    assert!(!entities.iter().any(|item| item.id == absent));
    assert_eq!(reader.ordinal_of_entity(&absent).unwrap(), None);
}

#[test]
fn the_csr_answers_the_same_adjacency_the_graph_does() {
    let (graph, entities, relations) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    // The control that makes this test able to fail: at least one entity has
    // more than one outgoing edge. A fixture where every entity has at most one
    // reads green even against a CSR that puts each source's START where its
    // END belongs.
    let mut fan_out: BTreeMap<Vec<u8>, usize> = BTreeMap::new();
    for item in &relations {
        if let GraphNodeId::Entity(src) = item.src {
            *fan_out.entry(src.0.as_bytes().to_vec()).or_default() += 1;
        }
    }
    assert!(
        fan_out.values().any(|count| *count > 1),
        "the fixture must contain an entity with several outgoing edges"
    );

    for expected in &entities {
        let ordinal = reader.ordinal_of_entity(&expected.id).unwrap().unwrap();

        let mut want_out: BTreeSet<Vec<u8>> = BTreeSet::new();
        let mut want_in: BTreeSet<Vec<u8>> = BTreeSet::new();
        for item in &relations {
            if item.src == GraphNodeId::Entity(expected.id) {
                if let GraphNodeId::Entity(dst) = item.dst {
                    if entities.iter().any(|candidate| candidate.id == dst) {
                        want_out.insert(dst.0.as_bytes().to_vec());
                    }
                }
            }
            if item.dst == GraphNodeId::Entity(expected.id) {
                if let GraphNodeId::Entity(src) = item.src {
                    want_in.insert(src.0.as_bytes().to_vec());
                }
            }
        }

        let got_out: BTreeSet<Vec<u8>> = reader
            .outgoing(ordinal)
            .unwrap()
            .iter()
            .map(|target| reader.entity_id(target).unwrap().0.as_bytes().to_vec())
            .collect();
        assert_eq!(
            got_out, want_out,
            "outgoing set differs for {}",
            expected.name
        );

        let got_in: BTreeSet<Vec<u8>> = reader
            .incoming(ordinal)
            .unwrap()
            .iter()
            .map(|source| reader.entity_id(source).unwrap().0.as_bytes().to_vec())
            .collect();
        assert_eq!(
            got_in, want_in,
            "incoming set differs for {}",
            expected.name
        );

        // The relation ordinals an entity's outgoing range names must be
        // exactly the relations whose source is that entity.
        for relation_ordinal in reader.outgoing_relations(ordinal).unwrap() {
            assert_eq!(
                reader.relation_source_ordinal(relation_ordinal).unwrap(),
                Some(ordinal),
                "relation {relation_ordinal} is in the outgoing range of {ordinal} and does not \
                 name it as its source"
            );
        }
        let incoming_relations = reader.incoming_relations(ordinal).unwrap();
        for relation_ordinal in incoming_relations.iter() {
            let relation = reader.relation(relation_ordinal).unwrap();
            assert_eq!(
                relation.dst,
                GraphNodeId::Entity(expected.id),
                "relation {relation_ordinal} is in the incoming range of {ordinal} and does not \
                 point at it"
            );
        }
    }
}

#[test]
fn a_duplicated_name_returns_every_ordinal_that_carries_it() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    let duplicated = "parse_header";
    let expected = entities
        .iter()
        .filter(|item| item.name == duplicated)
        .count();
    assert!(expected > 1, "the fixture must carry a duplicated name");

    let ordinals: Vec<u32> = reader
        .entities_by_name(duplicated)
        .unwrap()
        .iter()
        .collect();
    assert_eq!(ordinals.len(), expected);
    for ordinal in &ordinals {
        assert_eq!(reader.entity_name(*ordinal).unwrap(), duplicated);
    }

    // Case folding is the same fold ReadIndex applies, because the key arena
    // holds what `to_lowercase` produced at write time.
    let mixed: Vec<u32> = reader
        .entities_by_name("PARSE_Header")
        .unwrap()
        .iter()
        .collect();
    assert_eq!(mixed, ordinals);

    assert!(reader
        .entities_by_name("no_such_symbol")
        .unwrap()
        .is_empty());
}

#[test]
fn the_counts_come_from_scanning_the_columns_rather_than_a_persisted_tally() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    let mut want_kinds: BTreeMap<u8, u32> = BTreeMap::new();
    let mut want_languages: BTreeMap<u8, u32> = BTreeMap::new();
    for item in &entities {
        *want_kinds.entry(entity_kind_code(item.kind)).or_default() += 1;
        *want_languages
            .entry(language_code(item.language))
            .or_default() += 1;
    }
    assert!(want_kinds.len() > 1 && want_languages.len() > 1);
    assert_eq!(reader.kind_counts().unwrap(), want_kinds);
    assert_eq!(reader.language_counts().unwrap(), want_languages);
}

#[test]
fn a_path_repeated_across_entities_is_stored_once() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();

    let distinct: BTreeSet<String> = entities
        .iter()
        .flat_map(|item| {
            item.file_origin
                .iter()
                .map(|path| path.0.clone())
                .chain(item.span.iter().map(|span| span.file.0.clone()))
        })
        .collect();
    assert_eq!(
        stats.shape.path_count,
        distinct.len() as u64,
        "the path table must hold each distinct path once, counting a span's own file"
    );

    let repeated = entities
        .iter()
        .filter(|item| item.file_origin.as_ref().map(|p| p.0.as_str()) == Some("src/parse.rs"))
        .count();
    assert!(repeated > 0, "the fixture must repeat one path");
}

// ---------------------------------------------------------------------------
// Wire codes
// ---------------------------------------------------------------------------

#[test]
fn every_enum_variant_round_trips_through_its_wire_code() {
    // These lists are exhaustive by construction: the `*_code` functions are
    // matches with no wildcard arm, so a variant added to kin-model fails to
    // compile there rather than silently reinterpreting persisted bytes here.
    let kinds = [
        EntityKind::Function,
        EntityKind::Class,
        EntityKind::Interface,
        EntityKind::TraitDef,
        EntityKind::TypeAlias,
        EntityKind::Module,
        EntityKind::Package,
        EntityKind::Test,
        EntityKind::Schema,
        EntityKind::ApiEndpoint,
        EntityKind::EventContract,
        EntityKind::File,
        EntityKind::DocumentNode,
        EntityKind::Method,
        EntityKind::EnumDef,
        EntityKind::EnumVariant,
        EntityKind::Constant,
        EntityKind::StaticVar,
        EntityKind::Macro,
    ];
    let mut codes = BTreeSet::new();
    for kind in kinds {
        let code = entity_kind_code(kind);
        assert!(codes.insert(code), "entity kind code {code} is reused");
        assert_eq!(entity_kind_of_code(code).unwrap(), kind);
    }
    assert_eq!(codes.len(), kinds.len());
    assert!(entity_kind_of_code(u8::MAX).is_err());

    let languages = [
        LanguageId::TypeScript,
        LanguageId::JavaScript,
        LanguageId::Python,
        LanguageId::Go,
        LanguageId::Java,
        LanguageId::Rust,
        LanguageId::C,
        LanguageId::Cpp,
        LanguageId::CSharp,
        LanguageId::Ruby,
        LanguageId::Php,
        LanguageId::Swift,
        LanguageId::Kotlin,
        LanguageId::Hcl,
    ];
    let mut codes = BTreeSet::new();
    for language in languages {
        let code = language_code(language);
        assert!(codes.insert(code), "language code {code} is reused");
        assert_eq!(language_of_code(code).unwrap(), language);
    }
    assert_eq!(codes.len(), languages.len());
    assert!(language_of_code(u8::MAX).is_err());

    for visibility in [
        Visibility::Public,
        Visibility::Private,
        Visibility::Internal,
        Visibility::Crate,
    ] {
        let code = visibility_code(visibility);
        assert!(code <= 0b11, "visibility must fit the two flag bits");
        assert_eq!(visibility_of_code(code).unwrap(), visibility);
    }

    for role in [
        EntityRole::Source,
        EntityRole::Test,
        EntityRole::External,
        EntityRole::Docs,
        EntityRole::Generated,
        EntityRole::Vendored,
    ] {
        let code = role_code(role);
        assert!(code <= 0b111, "role must fit the three flag bits");
        assert_eq!(role_of_code(code).unwrap(), role);
    }

    let relation_kinds = [
        RelationKind::Contains,
        RelationKind::Extends,
        RelationKind::Implements,
        RelationKind::Overrides,
        RelationKind::Calls,
        RelationKind::Instantiates,
        RelationKind::References,
        RelationKind::UsesMacro,
        RelationKind::UsesType,
        RelationKind::Imports,
        RelationKind::Includes,
        RelationKind::DependsOn,
        RelationKind::EmitsEvent,
        RelationKind::SubscribesTo,
        RelationKind::DefinesContract,
        RelationKind::ConsumesContract,
        RelationKind::SendsMessage,
        RelationKind::Spawns,
        RelationKind::Tests,
        RelationKind::Covers,
        RelationKind::CoChanges,
        RelationKind::DerivedFrom,
        RelationKind::DocumentedBy,
        RelationKind::OwnedBy,
        RelationKind::OwnedByFile,
    ];
    let mut codes = BTreeSet::new();
    for kind in relation_kinds {
        let code = relation_kind_code(kind);
        assert!(codes.insert(code), "relation kind code {code} is reused");
        assert_eq!(relation_kind_of_code(code).unwrap(), kind);
    }
    assert_eq!(codes.len(), relation_kinds.len());
    assert!(relation_kind_of_code(u8::MAX).is_err());

    for origin in [
        RelationOrigin::Parsed,
        RelationOrigin::Inferred,
        RelationOrigin::Manual,
        RelationOrigin::Lsp,
    ] {
        let code = relation_origin_code(origin);
        assert_eq!(relation_origin_of_code(code).unwrap(), origin);
    }
    assert!(relation_origin_of_code(u8::MAX).is_err());
}

// ---------------------------------------------------------------------------
// The version range, and the profile boundary
// ---------------------------------------------------------------------------

/// Rewrite the declared layout version of every file in a segment and re-digest
/// each one, so the only thing that changed is the version.
fn restamp_version(dir: &Path, version: u32) {
    let mut restamped = 0usize;
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        // Only the segment's own files. The atomic writer stages and promotes
        // through other names, and rewriting one of those would be corrupting
        // the harness rather than the segment.
        if path.extension().and_then(|extension| extension.to_str()) != Some("kseg") {
            continue;
        }
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[4..8].copy_from_slice(&version.to_le_bytes());
        let digest_start = bytes.len() - DIGEST_LEN;
        let digest = Sha256::digest(&bytes[..digest_start]);
        bytes[digest_start..].copy_from_slice(&digest);
        std::fs::write(&path, &bytes).unwrap();
        restamped += 1;
    }
    assert!(
        restamped > 1,
        "the restamp must reach the manifest and every column, and reached {restamped}"
    );
}

#[test]
fn a_segment_one_version_forward_still_opens_and_below_the_floor_is_refused_by_name() {
    let (graph, _, _) = fixture();

    // The control: at the version this binary writes, the segment opens.
    let current = tempfile::tempdir().unwrap();
    write_segment(&graph, current.path()).unwrap();
    assert!(SegmentReader::open(current.path()).is_ok());

    let forward = tempfile::tempdir().unwrap();
    write_segment(&graph, forward.path()).unwrap();
    restamp_version(forward.path(), super::MAX_SUPPORTED_SEGMENT_VERSION);
    SegmentReader::open(forward.path())
        .expect("a segment one additive version ahead must still open");

    let below = tempfile::tempdir().unwrap();
    write_segment(&graph, below.path()).unwrap();
    restamp_version(below.path(), super::MIN_SUPPORTED_SEGMENT_VERSION - 1);
    let error = SegmentReader::open(below.path())
        .expect_err("a segment below the floor must be refused")
        .to_string();
    assert!(
        error.contains("below the floor"),
        "the refusal must name the floor, and said: {error}"
    );

    let above = tempfile::tempdir().unwrap();
    write_segment(&graph, above.path()).unwrap();
    restamp_version(above.path(), super::MAX_SUPPORTED_SEGMENT_VERSION + 1);
    let error = SegmentReader::open(above.path())
        .expect_err("a segment above the ceiling must be refused")
        .to_string();
    assert!(
        error.contains("above the ceiling"),
        "the refusal must name the ceiling, and said: {error}"
    );
}

#[test]
fn the_hot_profile_answers_the_read_index_queries_and_leaves_the_cold_columns_unmapped() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = SegmentReader::open_with_profile(dir.path(), OpenProfile::Hot).unwrap();

    // Everything the ReadIndex query set needs is answerable.
    for ordinal in 0..reader.entity_count() {
        reader.entity_id(ordinal).unwrap();
        reader.entity_name(ordinal).unwrap();
        reader.entity_kind(ordinal).unwrap();
        reader.entity_language(ordinal).unwrap();
        reader.entity_start_line(ordinal).unwrap();
        reader.entity_path(ordinal).unwrap();
        reader.outgoing(ordinal).unwrap();
        reader.incoming(ordinal).unwrap();
    }
    reader.kind_counts().unwrap();
    reader.language_counts().unwrap();
    reader.entities_by_name(&entities[0].name).unwrap();

    // The cold columns are not merely unread, they are not mapped, so a caller
    // that wants them is told which profile carries them rather than handed a
    // zero.
    let error = reader
        .entity(0)
        .expect_err("a hot open must not reconstruct a whole entity")
        .to_string();
    assert!(
        error.contains("full profile"),
        "the refusal must name the profile that carries the column, and said: {error}"
    );
}

// ---------------------------------------------------------------------------
// Integrity: what open catches, and what only verify_all catches
// ---------------------------------------------------------------------------

#[test]
fn a_truncated_column_is_refused_at_open() {
    let (graph, _, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    assert!(SegmentReader::open(dir.path()).is_ok());

    let path = dir.path().join(column_file_name(column::ENTITY_ID));
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();

    let error = SegmentReader::open(dir.path())
        .expect_err("a truncated column must be refused at open")
        .to_string();
    assert!(
        error.contains("torn or truncated"),
        "the refusal must name the fault, and said: {error}"
    );
}

#[test]
fn a_flipped_byte_inside_a_column_is_caught_by_verify_all_and_not_by_open() {
    let (graph, _, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();

    // The control: before the flip, verify_all is green over every mapped
    // column and the count is non-zero, so a green result means something.
    let checked = written(dir.path()).verify_all().unwrap();
    assert!(checked > 0, "verify_all must actually hash columns");

    let path = dir.path().join(column_file_name(column::ENTITY_NAME_ARENA));
    let mut bytes = std::fs::read(&path).unwrap();
    let target = HEADER_LEN;
    assert!(
        bytes.len() > target + DIGEST_LEN,
        "the name arena must carry at least one byte to flip"
    );
    bytes[target] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    // This is the documented tradeoff, stated as a test rather than a comment:
    // the open is O(columns) and does not hash, so it accepts these bytes.
    let reader = written(dir.path());
    let error = reader
        .verify_all()
        .expect_err("verify_all must catch a flipped byte")
        .to_string();
    assert!(
        error.contains("does not match the digest"),
        "the refusal must name the digest, and said: {error}"
    );
}

#[test]
fn a_manifest_whose_digest_does_not_match_its_bytes_is_refused() {
    let (graph, _, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();

    let path = dir.path().join(MANIFEST_FILE);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[HEADER_LEN] ^= 0xFF;
    std::fs::write(&path, &bytes).unwrap();

    let error = SegmentReader::open(dir.path())
        .expect_err("a manifest that disagrees with its digest must be refused")
        .to_string();
    assert!(
        error.contains("manifest digest"),
        "the refusal must name the manifest digest, and said: {error}"
    );
}

#[test]
fn an_empty_graph_writes_and_opens() {
    let graph = InMemoryGraph::new();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    assert_eq!(stats.shape.entity_count, 0);
    assert_eq!(stats.shape.relation_count, 0);

    let reader = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(reader.entity_count(), 0);
    assert!(reader.entities_by_name("anything").unwrap().is_empty());
    assert_eq!(reader.unknown_columns(), 0);
}

// ---------------------------------------------------------------------------
// Entity revisions
// ---------------------------------------------------------------------------

/// A graph whose revisions were produced the way a real store produces them,
/// by admitting a `SemanticChange`, rather than by constructing
/// `EntityRevision` values by hand. The expected revisions are then read back
/// from the graph's own snapshot, so the proof compares the segment against the
/// graph rather than against the test's own idea of a revision.
fn revision_fixture(modify_head: bool) -> (InMemoryGraph, Vec<(EntityId, EntityRevision)>) {
    use kin_model::{
        AuthorId, ChangeOrigin, ChangeStore, EntityDelta, SemanticChange, SemanticChangeId,
        Timestamp, TreeEntry,
    };

    let graph = InMemoryGraph::new();
    let anchored = entity(9, "revised", "src/revised.rs");
    let mut floating = entity(10, "floating", "src/unused.rs");
    floating.file_origin = None;
    floating.span = None;

    graph.admit_artifact_for_test("src/revised.rs", TreeEntry::blob(hash(0x33), false));
    graph.upsert_entity(&anchored).unwrap();
    graph.upsert_entity(&floating).unwrap();

    let change = SemanticChange {
        id: SemanticChangeId::from_hash(hash(0)),
        parents: Vec::new(),
        timestamp: Timestamp::now(),
        author: AuthorId::new("memarena"),
        message: "record a revision for each entity".into(),
        entity_deltas: vec![
            EntityDelta::Added {
                new: anchored.clone(),
            },
            EntityDelta::Added {
                new: floating.clone(),
            },
        ],
        relation_deltas: Vec::new(),
        tree_deltas: Vec::new(),
        projected_files: Vec::new(),
        spec_link: None,
        evidence: Vec::new(),
        risk_summary: None,
        origin: ChangeOrigin::Native,
        admission_policy_delta: None,
        external_reference_deltas: Vec::new(),
    };
    let sealed = kin_model::compute_semantic_change_id(&change).expect("a valid change fixture");
    let change = SemanticChange {
        id: sealed,
        ..change
    };
    graph.create_change(&change).expect("the change must admit");

    if modify_head {
        // Move the head out from under its own revision, which is the only way
        // the differs bit is ever set. Same id, different content.
        let mut moved = anchored.clone();
        moved.signature = "fn revised(argument: &str, extra: u8) -> usize".to_string();
        moved.doc_summary = Some("the head moved after its revision".to_string());
        graph.upsert_entity(&moved).unwrap();
    }

    let expected: Vec<(EntityId, EntityRevision)> = graph
        .to_snapshot()
        .entity_revisions
        .into_iter()
        .flat_map(|(id, revisions)| revisions.into_iter().map(move |rev| (id, rev)))
        .collect();
    assert!(
        !expected.is_empty(),
        "the fixture must produce at least one revision, or every assertion below is vacuous"
    );
    (graph, expected)
}

#[test]
fn every_revision_reads_back_through_the_mapped_views() {
    let (graph, expected) = revision_fixture(false);
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    assert_eq!(stats.revision_count, expected.len() as u64);
    assert_eq!(
        stats.change_count, 1,
        "one commit means one distinct change id in the dictionary, which is what both real \
         stores measure"
    );

    let reader = written(dir.path());
    assert_eq!(reader.revision_count(), expected.len() as u64);

    let mut seen = 0usize;
    for (entity_id, want) in &expected {
        let ordinal = reader
            .ordinal_of_revision(&want.revision_id)
            .unwrap()
            .unwrap_or_else(|| panic!("revision {} has no ordinal", want.revision_id.0));

        assert_eq!(reader.revision_id(ordinal).unwrap(), want.revision_id);
        assert_eq!(
            reader
                .entity_id(reader.revision_entity_ordinal(ordinal).unwrap())
                .unwrap(),
            *entity_id
        );
        assert_eq!(
            reader.revision_introduced_by(ordinal).unwrap(),
            want.introduced_by
        );
        assert_eq!(
            reader.revision_previous(ordinal).unwrap(),
            want.previous_revision
        );
        assert_eq!(reader.revision_ended_by(ordinal).unwrap(), want.ended_by);

        let actual = reader.entity_revision(ordinal).unwrap();
        assert_eq!(actual.revision_id, want.revision_id);
        assert_eq!(actual.entity_id, want.entity_id);
        assert_eq!(actual.entity, want.entity, "the revision's entity differs");
        // Metadata is held to content rather than to bytes for the reason the
        // entity round trip records: a `HashMap`'s iteration order is not part
        // of its value.
        let mut actual_without = actual.entity.clone();
        let mut want_without = want.entity.clone();
        actual_without.metadata = EntityMetadata::default();
        want_without.metadata = EntityMetadata::default();
        assert_eq!(
            rmp_serde::to_vec(&actual_without).unwrap(),
            rmp_serde::to_vec(&want_without).unwrap(),
            "the revision's entity did not serialize identically outside its metadata"
        );
        seen += 1;
    }
    assert_eq!(seen, expected.len());
}

#[test]
fn a_revision_lookup_costs_a_binary_search_and_reads_no_second_entity() {
    let (graph, expected) = revision_fixture(false);
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    // The revision id column is sorted, which is what makes the ordinal the id
    // rank and the lookup an index-free binary search.
    let mut previous: Option<Vec<u8>> = None;
    for ordinal in 0..reader.revision_count() as u32 {
        let raw = reader.revision_id(ordinal).unwrap().0.as_bytes().to_vec();
        if let Some(before) = &previous {
            assert!(
                before < &raw,
                "the revision id column is not strictly ascending at ordinal {ordinal}, so a \
                 binary search over it cannot be the revision index"
            );
        }
        previous = Some(raw);
    }

    // This is the property the captain asked about: with the differs bit clear,
    // the revision's entity IS the head entity, so every field comes from the
    // head's own columns and there is no second entity in the segment at all.
    let delta_bytes = write_segment(&graph, tempfile::tempdir().unwrap().path())
        .unwrap()
        .columns
        .iter()
        .find(|column| column.id == crate::storage::segment::format::column::REV_DELTA_ARENA)
        .map(|column| column.payload_len)
        .expect("the delta side table column must exist");
    assert_eq!(
        delta_bytes, 0,
        "no revision in this fixture differs from its head, so the delta side table must be \
         empty, which is what both real stores measure over 294,007 revisions"
    );

    for (entity_id, want) in &expected {
        let ordinal = reader
            .ordinal_of_revision(&want.revision_id)
            .unwrap()
            .unwrap();
        assert!(
            !reader.revision_entity_differs_from_head(ordinal).unwrap(),
            "revision {} should match its head",
            want.revision_id.0
        );
        let entity_ordinal = reader.revision_entity_ordinal(ordinal).unwrap();
        assert_eq!(reader.entity_id(entity_ordinal).unwrap(), *entity_id);
        assert_eq!(
            reader.entity_name(entity_ordinal).unwrap(),
            want.entity.name,
            "the head's own name column must answer for the revision"
        );
    }

    // The negative, with the sorted-column control above it.
    let absent = kin_model::EntityRevisionId(Hash256::from_bytes([0xCD; 32]));
    assert!(!expected.iter().any(|(_, r)| r.revision_id == absent));
    assert_eq!(reader.ordinal_of_revision(&absent).unwrap(), None);
}

#[test]
fn a_revision_whose_head_moved_carries_its_own_entity_in_the_side_table() {
    let (graph, expected) = revision_fixture(true);
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    let reader = written(dir.path());

    // The control that makes this test able to fail: the delta side table is
    // NOT empty here, unlike the fixture above.
    let delta_bytes = stats
        .columns
        .iter()
        .find(|column| column.id == crate::storage::segment::format::column::REV_DELTA_ARENA)
        .map(|column| column.payload_len)
        .unwrap();
    assert!(
        delta_bytes > 0,
        "moving the head must put at least one revision's entity in the delta side table"
    );

    let mut differing = 0usize;
    for (_, want) in &expected {
        let ordinal = reader
            .ordinal_of_revision(&want.revision_id)
            .unwrap()
            .unwrap();
        let actual = reader.entity_revision(ordinal).unwrap();
        assert_eq!(
            actual.entity, want.entity,
            "a revision whose head moved must still read back its OWN entity, not the head's"
        );
        if reader.revision_entity_differs_from_head(ordinal).unwrap() {
            differing += 1;
            let head_ordinal = reader.revision_entity_ordinal(ordinal).unwrap();
            assert_ne!(
                reader.entity_signature(head_ordinal).unwrap(),
                want.entity.signature,
                "the head's signature must have moved, or this arm proves nothing"
            );
        }
    }
    assert_eq!(
        differing, 1,
        "exactly one fixture entity's head was moved out from under its revision"
    );
}

#[test]
fn a_segment_with_no_revisions_reports_none_and_still_opens() {
    let (graph, _, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    let stats = write_segment(&graph, dir.path()).unwrap();
    assert_eq!(stats.revision_count, 0);
    assert_eq!(stats.change_count, 0);

    let reader = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(reader.revision_count(), 0);
    assert_eq!(reader.change_count(), 0);
    let absent = kin_model::EntityRevisionId(Hash256::from_bytes([0x01; 32]));
    assert_eq!(reader.ordinal_of_revision(&absent).unwrap(), None);
}

// ---------------------------------------------------------------------------
// Forward compatibility, which is the property no fixture can prove alone
// ---------------------------------------------------------------------------

#[test]
fn a_manifest_row_this_binary_does_not_know_is_skipped_and_counted() {
    let (graph, entities, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();

    // The control: before the injection, nothing is unknown.
    assert_eq!(
        SegmentReader::open(dir.path()).unwrap().unknown_columns(),
        0
    );

    // Append one manifest row for a column id no version of this binary knows,
    // naming a file that does not exist. A reader that tried to map every row
    // rather than the rows its profile wants would fail on the missing file;
    // the additive contract says it must skip the row and count it.
    let path = dir.path().join(MANIFEST_FILE);
    let bytes = std::fs::read(&path).unwrap();
    let payload_len = u64::from_le_bytes(bytes[24..32].try_into().unwrap()) as usize;
    let count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

    let mut rebuilt = Vec::new();
    rebuilt.extend_from_slice(&bytes[..HEADER_LEN + payload_len]);
    rebuilt.extend_from_slice(&9_999u32.to_le_bytes()); // column id
    rebuilt.extend_from_slice(&4u32.to_le_bytes()); // width
    rebuilt.extend_from_slice(&1u64.to_le_bytes()); // count
    rebuilt.extend_from_slice(&4u64.to_le_bytes()); // payload_len
    rebuilt.extend_from_slice(&[0u8; 32]); // digest
    let new_payload = payload_len + 56;
    rebuilt[16..24].copy_from_slice(&(count + 1).to_le_bytes());
    rebuilt[24..32].copy_from_slice(&(new_payload as u64).to_le_bytes());
    let digest = Sha256::digest(&rebuilt[..HEADER_LEN + new_payload]);
    rebuilt.extend_from_slice(&digest);
    std::fs::write(&path, &rebuilt).unwrap();

    let reader = SegmentReader::open(dir.path()).unwrap();
    assert_eq!(
        reader.unknown_columns(),
        1,
        "the unknown row must be counted, so a reader can say a newer layout wrote this segment"
    );
    // And every entity query still answers, which is the whole point of the
    // additive rule.
    for ordinal in 0..reader.entity_count() {
        reader.entity_name(ordinal).unwrap();
        reader.outgoing(ordinal).unwrap();
    }
    assert!(!reader
        .entities_by_name(&entities[0].name)
        .unwrap()
        .is_empty());
}

#[test]
fn a_column_the_segment_does_not_carry_refuses_differently_from_one_the_profile_skips() {
    let (graph, _, _) = fixture();
    let dir = tempfile::tempdir().unwrap();
    write_segment(&graph, dir.path()).unwrap();
    let hot = SegmentReader::open_with_profile(dir.path(), OpenProfile::Hot).unwrap();

    // Present in the segment, not mapped by this profile.
    let present = hot
        .entity(0)
        .expect_err("a hot open must not reconstruct a whole entity")
        .to_string();
    assert!(
        present.contains("full profile"),
        "a present-but-unmapped column must name the profile, and said: {present}"
    );

    // Absent from the segment entirely. `fixture()` admits no change, so it
    // produces no revisions, but the columns exist with zero rows, so the
    // absent case is reached by asking for an ordinal past their end instead.
    let absent = hot
        .revision_id(0)
        .expect_err("a segment with no revisions has no revision 0")
        .to_string();
    assert!(
        absent.contains("past the end") || absent.contains("carries no column"),
        "the refusal must say which of the two faults it is, and said: {absent}"
    );
}
