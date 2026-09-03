// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Builds a segment from a live [`InMemoryGraph`].
//!
//! The writer is a one-time cost and is written for clarity rather than for a
//! small peak: it stages the string arenas once and copies them into the column
//! payloads in ordinal order, so its peak is about twice the arena bytes. What
//! this program is about is the SERVING cost, which is the reader's, and the
//! reader allocates nothing per entity.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use kin_model::{Entity, GraphNodeId, Relation};
use sha2::{Digest, Sha256};

use crate::engine::InMemoryGraph;
use crate::error::KinDbError;
use crate::storage::mmap::atomic_write_bytes_no_magic;
use crate::storage::segment::format::{
    column, column_file_name, encode_header, relation_kind_code, relation_origin_code, ColumnRecord,
    SegmentShape, COLD_HAS_CREATED_IN, COLD_HAS_LINEAGE, COLD_HAS_METADATA, COLD_HAS_SUPERSEDED,
    DIGEST_LEN, FLAG_HAS_DOC, FLAG_HAS_PATH, FLAG_HAS_SPAN, FLAG_ROLE_SHIFT, HEADER_LEN,
    MANIFEST_ENTRY_LEN, MANIFEST_FILE, MANIFEST_PREAMBLE_LEN, REL_ENTITY_ENDPOINTS,
    REL_HAS_CREATED_IN, REL_HAS_EVIDENCE, REL_HAS_IMPORT_SOURCE,
};
use crate::storage::segment::format::{entity_kind_code, language_code, role_code, visibility_code};

/// What one column cost on disk, so a caller can hold the layout to the
/// per-column sizes it registered before the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnStat {
    /// Which column. See [`crate::storage::segment::format::column`].
    pub id: u32,
    /// Bytes per record the header declares, or 1 for a byte arena.
    pub width: u32,
    /// Records in the column.
    pub count: u64,
    /// Payload bytes, which is `width * count` for a fixed-width column.
    pub payload_len: u64,
    /// Payload plus the 32-byte header and the 32-byte digest.
    pub on_disk: u64,
}

/// What a write produced, column by column, so a caller can hold the layout to
/// the per-column sizes it registered before the run.
#[derive(Debug, Clone, Default)]
pub struct SegmentWriteStats {
    /// Shape the manifest records.
    pub shape: SegmentShape,
    /// Every column written, in id order.
    pub columns: Vec<ColumnStat>,
    /// Bytes across every column file plus the manifest.
    pub total_bytes: u64,
    /// Bytes across the columns a hot read touches: the entity columns, the
    /// arenas they index, the path table, the name index and the CSR. Cold
    /// columns and the side tables are excluded because a hot read never maps
    /// them.
    pub hot_bytes: u64,
}

/// Columns a read that answers the `ReadIndex` query set actually maps.
const HOT_COLUMNS: &[u32] = &[
    column::ENTITY_ID,
    column::ENTITY_KIND,
    column::ENTITY_LANGUAGE,
    column::ENTITY_FLAGS,
    column::ENTITY_AST_HASH,
    column::ENTITY_PATH_ORD,
    column::ENTITY_SPAN,
    column::ENTITY_SPAN_PATH_ORD,
    column::ENTITY_NAME_OFF,
    column::ENTITY_NAME_ARENA,
    column::ENTITY_SIG_OFF,
    column::ENTITY_SIG_ARENA,
    column::ENTITY_DOC_OFF,
    column::ENTITY_DOC_ARENA,
    column::PATH_OFF,
    column::PATH_ARENA,
    column::NAME_KEY_OFF,
    column::NAME_KEY_ARENA,
    column::NAME_POSTING_OFF,
    column::NAME_POSTINGS,
    column::OUT_OFF,
    column::OUT_DST,
    column::IN_OFF,
    column::IN_SRC,
    column::IN_REL_ORD,
    column::REL_KIND,
    column::REL_CONFIDENCE,
    column::REL_ORIGIN,
    column::REL_SRC,
    column::REL_FLAGS,
];

/// A staged entity: fixed fields resolved, variable fields as ranges into the
/// staging arenas.
struct Staged {
    id: [u8; 16],
    kind: u8,
    language: u8,
    flags: u8,
    cold_flags: u8,
    ast_hash: [u8; 32],
    signature_hash: [u8; 32],
    behavior_hash: [u8; 32],
    equivalence_hash: [u8; 32],
    stability: f32,
    path_slot: u32,
    span_path_slot: u32,
    span: [u32; 6],
    name: (usize, usize),
    lower: (usize, usize),
    signature: (usize, usize),
    doc: (usize, usize),
    lineage_parent: [u8; 16],
    created_in: [u8; 32],
    superseded_by: [u8; 16],
    metadata: (usize, usize),
}

#[derive(Default)]
struct Arenas {
    names: Vec<u8>,
    lower: Vec<u8>,
    signatures: Vec<u8>,
    docs: Vec<u8>,
    metadata: Vec<u8>,
}

impl Arenas {
    fn push(target: &mut Vec<u8>, bytes: &[u8]) -> (usize, usize) {
        let start = target.len();
        target.extend_from_slice(bytes);
        (start, bytes.len())
    }
}

/// Write a segment for `graph` into `dir`, creating it if absent.
///
/// The directory is expected to be empty or to hold a previous segment; every
/// column file is replaced through the same atomic stage-and-rename the
/// snapshot writer uses, so a reader either sees the old column or the new one.
pub fn write_segment(graph: &InMemoryGraph, dir: &Path) -> Result<SegmentWriteStats, KinDbError> {
    let _span = tracing::info_span!(
        "kindb.segment.write",
        entities = graph.entity_count(),
        relations = graph.relation_count(),
        dir = %dir.display()
    )
    .entered();

    std::fs::create_dir_all(dir).map_err(|error| {
        KinDbError::StorageError(format!(
            "failed to create segment directory {}: {error}",
            dir.display()
        ))
    })?;

    let mut arenas = Arenas::default();
    let mut staged: Vec<Staged> = Vec::with_capacity(graph.entity_count());
    let mut path_slots: HashMap<String, u32> = HashMap::new();
    let mut path_list: Vec<String> = Vec::new();

    graph.for_each_entity(|entity| {
        staged.push(stage_entity(entity, &mut arenas, &mut path_slots, &mut path_list));
    });

    // The ordinal IS the id rank, so `id -> ordinal` is a binary search over
    // the id column and costs no separate index. Ids are 16 fixed bytes, which
    // is why this beats an FST here rather than merely avoiding a dependency.
    let mut order: Vec<u32> = (0..staged.len() as u32).collect();
    order.sort_unstable_by(|left, right| {
        staged[*left as usize].id.cmp(&staged[*right as usize].id)
    });
    let mut ordinal_of_slot = vec![0u32; staged.len()];
    for (ordinal, slot) in order.iter().enumerate() {
        ordinal_of_slot[*slot as usize] = ordinal as u32;
    }
    let entity_count = order.len();

    let mut ordinal_of_id: HashMap<[u8; 16], u32> = HashMap::with_capacity(entity_count);
    for (ordinal, slot) in order.iter().enumerate() {
        ordinal_of_id.insert(staged[*slot as usize].id, ordinal as u32);
    }

    // Paths are written in sorted order so a later reader can binary-search a
    // path to its ordinal without a second index.
    let mut path_order: Vec<u32> = (0..path_list.len() as u32).collect();
    path_order.sort_unstable_by(|left, right| {
        path_list[*left as usize].cmp(&path_list[*right as usize])
    });
    let mut path_ordinal_of_slot = vec![0u32; path_list.len()];
    for (ordinal, slot) in path_order.iter().enumerate() {
        path_ordinal_of_slot[*slot as usize] = ordinal as u32;
    }

    let mut written: Vec<ColumnRecord> = Vec::new();

    written.push(write_fixed(dir, column::ENTITY_ID, 16, entity_count, |ordinal| {
        staged[order[ordinal] as usize].id.to_vec()
    })?);
    written.push(write_fixed(dir, column::ENTITY_KIND, 1, entity_count, |ordinal| {
        vec![staged[order[ordinal] as usize].kind]
    })?);
    written.push(write_fixed(
        dir,
        column::ENTITY_LANGUAGE,
        1,
        entity_count,
        |ordinal| vec![staged[order[ordinal] as usize].language],
    )?);
    written.push(write_fixed(dir, column::ENTITY_FLAGS, 1, entity_count, |ordinal| {
        vec![staged[order[ordinal] as usize].flags]
    })?);
    written.push(write_fixed(
        dir,
        column::ENTITY_AST_HASH,
        32,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].ast_hash.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_PATH_ORD,
        4,
        entity_count,
        |ordinal| {
            let slot = staged[order[ordinal] as usize].path_slot;
            let resolved = path_ordinal_of_slot
                .get(slot as usize)
                .copied()
                .unwrap_or(u32::MAX);
            resolved.to_le_bytes().to_vec()
        },
    )?);
    written.push(write_fixed(dir, column::ENTITY_SPAN, 24, entity_count, |ordinal| {
        let span = staged[order[ordinal] as usize].span;
        let mut bytes = Vec::with_capacity(24);
        for value in span {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    })?);

    written.push(write_fixed(
        dir,
        column::ENTITY_SPAN_PATH_ORD,
        4,
        entity_count,
        |ordinal| {
            let slot = staged[order[ordinal] as usize].span_path_slot;
            let resolved = path_ordinal_of_slot
                .get(slot as usize)
                .copied()
                .unwrap_or(u32::MAX);
            resolved.to_le_bytes().to_vec()
        },
    )?);

    let (name_off, name_arena) = permute_arena(&arenas.names, &staged, &order, |entry| entry.name)?;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_NAME_OFF,
        4,
        entity_count as u64 + 1,
        name_off,
    )?);
    let name_arena_len = name_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_NAME_ARENA,
        1,
        name_arena_len,
        name_arena,
    )?);

    let (sig_off, sig_arena) =
        permute_arena(&arenas.signatures, &staged, &order, |entry| entry.signature)?;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_SIG_OFF,
        4,
        entity_count as u64 + 1,
        sig_off,
    )?);
    let sig_arena_len = sig_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_SIG_ARENA,
        1,
        sig_arena_len,
        sig_arena,
    )?);

    let (doc_off, doc_arena) = permute_arena(&arenas.docs, &staged, &order, |entry| entry.doc)?;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_DOC_OFF,
        4,
        entity_count as u64 + 1,
        doc_off,
    )?);
    let doc_arena_len = doc_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_DOC_ARENA,
        1,
        doc_arena_len,
        doc_arena,
    )?);

    // Path table, in sorted order.
    let mut path_off: Vec<u8> = Vec::with_capacity((path_order.len() + 1) * 4);
    let mut path_arena: Vec<u8> = Vec::new();
    path_off.extend_from_slice(&0u32.to_le_bytes());
    for slot in &path_order {
        path_arena.extend_from_slice(path_list[*slot as usize].as_bytes());
        path_off.extend_from_slice(&checked_u32(path_arena.len(), "path arena")?.to_le_bytes());
    }
    written.push(write_bytes_column(
        dir,
        column::PATH_OFF,
        4,
        path_order.len() as u64 + 1,
        path_off,
    )?);
    let path_arena_len = path_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::PATH_ARENA,
        1,
        path_arena_len,
        path_arena,
    )?);

    // Name index: lowercased keys in sorted order, CSR postings into ordinals.
    let mut by_lower: BTreeMap<&[u8], Vec<u32>> = BTreeMap::new();
    for (ordinal, slot) in order.iter().enumerate() {
        let entry = &staged[*slot as usize];
        let key = &arenas.lower[entry.lower.0..entry.lower.0 + entry.lower.1];
        by_lower.entry(key).or_default().push(ordinal as u32);
    }
    let name_key_count = by_lower.len();
    let mut key_off: Vec<u8> = Vec::with_capacity((name_key_count + 1) * 4);
    let mut key_arena: Vec<u8> = Vec::new();
    let mut posting_off: Vec<u8> = Vec::with_capacity((name_key_count + 1) * 4);
    let mut postings: Vec<u8> = Vec::with_capacity(entity_count * 4);
    key_off.extend_from_slice(&0u32.to_le_bytes());
    posting_off.extend_from_slice(&0u32.to_le_bytes());
    for (key, ordinals) in &by_lower {
        key_arena.extend_from_slice(key);
        key_off.extend_from_slice(&checked_u32(key_arena.len(), "name key arena")?.to_le_bytes());
        for ordinal in ordinals {
            postings.extend_from_slice(&ordinal.to_le_bytes());
        }
        posting_off
            .extend_from_slice(&checked_u32(postings.len() / 4, "name postings")?.to_le_bytes());
    }
    written.push(write_bytes_column(
        dir,
        column::NAME_KEY_OFF,
        4,
        name_key_count as u64 + 1,
        key_off,
    )?);
    let key_arena_len = key_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::NAME_KEY_ARENA,
        1,
        key_arena_len,
        key_arena,
    )?);
    written.push(write_bytes_column(
        dir,
        column::NAME_POSTING_OFF,
        4,
        name_key_count as u64 + 1,
        posting_off,
    )?);
    let postings_len = postings.len() as u64 / 4;
    written.push(write_bytes_column(
        dir,
        column::NAME_POSTINGS,
        4,
        postings_len,
        postings,
    )?);

    // Cold entity columns.
    written.push(write_fixed(
        dir,
        column::ENTITY_COLD_FLAGS,
        1,
        entity_count,
        |ordinal| vec![staged[order[ordinal] as usize].cold_flags],
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_SIGNATURE_HASH,
        32,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].signature_hash.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_BEHAVIOR_HASH,
        32,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].behavior_hash.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_EQUIVALENCE_HASH,
        32,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].equivalence_hash.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_STABILITY,
        4,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].stability.to_le_bytes().to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_LINEAGE_PARENT,
        16,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].lineage_parent.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_CREATED_IN,
        32,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].created_in.to_vec(),
    )?);
    written.push(write_fixed(
        dir,
        column::ENTITY_SUPERSEDED_BY,
        16,
        entity_count,
        |ordinal| staged[order[ordinal] as usize].superseded_by.to_vec(),
    )?);

    // Metadata side table, u64 offsets because this arena outgrows u32 at the
    // scale the design targets.
    let mut meta_off: Vec<u8> = Vec::with_capacity((entity_count + 1) * 8);
    let mut meta_arena: Vec<u8> = Vec::new();
    meta_off.extend_from_slice(&0u64.to_le_bytes());
    for slot in &order {
        let entry = &staged[*slot as usize];
        meta_arena.extend_from_slice(&arenas.metadata[entry.metadata.0..entry.metadata.0 + entry.metadata.1]);
        meta_off.extend_from_slice(&(meta_arena.len() as u64).to_le_bytes());
    }
    written.push(write_bytes_column(
        dir,
        column::ENTITY_METADATA_OFF,
        8,
        entity_count as u64 + 1,
        meta_off,
    )?);
    let meta_arena_len = meta_arena.len() as u64;
    written.push(write_bytes_column(
        dir,
        column::ENTITY_METADATA_ARENA,
        1,
        meta_arena_len,
        meta_arena,
    )?);

    let relation_records = write_relations(graph, dir, &ordinal_of_id, entity_count)?;
    let entity_edge_count = relation_records.entity_edge_count;
    let relation_count = relation_records.relation_count;
    written.extend(relation_records.columns);

    let shape = SegmentShape {
        entity_count: entity_count as u64,
        relation_count,
        entity_edge_count,
        path_count: path_order.len() as u64,
        name_key_count: name_key_count as u64,
    };

    written.sort_unstable_by_key(|record| record.id);
    let manifest_bytes = write_manifest(dir, &shape, &written)?;

    let mut columns: Vec<ColumnStat> = Vec::with_capacity(written.len());
    let mut total = manifest_bytes;
    let mut hot = 0u64;
    for record in &written {
        let on_disk = HEADER_LEN as u64 + record.payload_len + DIGEST_LEN as u64;
        columns.push(ColumnStat {
            id: record.id,
            width: record.width,
            count: record.count,
            payload_len: record.payload_len,
            on_disk,
        });
        total += on_disk;
        if HOT_COLUMNS.contains(&record.id) {
            hot += on_disk;
        }
    }

    Ok(SegmentWriteStats {
        shape,
        columns,
        total_bytes: total,
        hot_bytes: hot,
    })
}

struct RelationColumns {
    columns: Vec<ColumnRecord>,
    relation_count: u64,
    entity_edge_count: u64,
}

fn write_relations(
    graph: &InMemoryGraph,
    dir: &Path,
    ordinal_of_id: &HashMap<[u8; 16], u32>,
    entity_count: usize,
) -> Result<RelationColumns, KinDbError> {
    // Entity-endpoint relations occupy the low ordinals, because those are the
    // ones the CSR describes. Everything else keeps a full record in the
    // endpoint side table and is reachable by ordinal, so nothing is dropped.
    let mut entity_edges: Vec<Relation> = Vec::new();
    let mut other: Vec<Relation> = Vec::new();
    graph.for_each_relation(|relation| {
        let endpoints = match (relation.src, relation.dst) {
            (GraphNodeId::Entity(src), GraphNodeId::Entity(dst)) => {
                ordinal_of_id
                    .get(src.0.as_bytes())
                    .zip(ordinal_of_id.get(dst.0.as_bytes()))
                    .map(|(source, target)| (*source, *target))
            }
            _ => None,
        };
        if endpoints.is_some() {
            entity_edges.push(relation.clone());
        } else {
            other.push(relation.clone());
        }
    });

    // Sort by (src ordinal, dst ordinal) so the outgoing CSR is the relation
    // ordinal space itself: relation ordinal `r` is the `r`th entry of OUT_DST.
    entity_edges.sort_unstable_by_key(|relation| {
        let src = endpoint_ordinal(&relation.src, ordinal_of_id).unwrap_or(u32::MAX);
        let dst = endpoint_ordinal(&relation.dst, ordinal_of_id).unwrap_or(u32::MAX);
        (src, dst, *relation.id.0.as_bytes())
    });

    let entity_edge_count = entity_edges.len();
    let relation_count = entity_edge_count + other.len();

    let mut out_dst: Vec<u8> = Vec::with_capacity(entity_edge_count * 4);
    let mut rel_src: Vec<u8> = Vec::with_capacity(relation_count * 4);
    let mut rel_id: Vec<u8> = Vec::with_capacity(relation_count * 16);
    let mut rel_kind: Vec<u8> = Vec::with_capacity(relation_count);
    let mut rel_confidence: Vec<u8> = Vec::with_capacity(relation_count * 4);
    let mut rel_origin: Vec<u8> = Vec::with_capacity(relation_count);
    let mut rel_flags: Vec<u8> = Vec::with_capacity(relation_count);
    let mut rel_created_in: Vec<u8> = Vec::with_capacity(relation_count * 32);
    let mut import_off: Vec<u8> = Vec::with_capacity((relation_count + 1) * 4);
    let mut import_arena: Vec<u8> = Vec::new();
    let mut endpoints_off: Vec<u8> = Vec::with_capacity((relation_count + 1) * 8);
    let mut endpoints_arena: Vec<u8> = Vec::new();
    let mut evidence_off: Vec<u8> = Vec::with_capacity((relation_count + 1) * 8);
    let mut evidence_arena: Vec<u8> = Vec::new();

    import_off.extend_from_slice(&0u32.to_le_bytes());
    endpoints_off.extend_from_slice(&0u64.to_le_bytes());
    evidence_off.extend_from_slice(&0u64.to_le_bytes());

    let mut incoming: Vec<Vec<(u32, u32)>> = vec![Vec::new(); entity_count];

    // Outgoing CSR by counting sort rather than by pushing at the start of each
    // source's run. `entity_edges` is sorted by source ordinal, so relation
    // ordinal `r` is the `r`th entry of OUT_DST and the offsets are the prefix
    // sum of the per-source counts. Building it the other way put a source's
    // START where its END belongs, which is a class of CSR error that reads
    // green on a fixture where every entity has at most one outgoing edge.
    let mut out_counts = vec![0u32; entity_count + 1];
    for relation in &entity_edges {
        let src = endpoint_ordinal(&relation.src, ordinal_of_id).ok_or_else(|| {
            KinDbError::StorageError(
                "segment writer partitioned a relation as an entity edge whose source has no \
                 ordinal"
                    .to_string(),
            )
        })?;
        out_counts[src as usize + 1] += 1;
    }
    let mut running = 0u32;
    for value in out_counts.iter_mut() {
        running += *value;
        *value = running;
    }
    let mut out_off: Vec<u8> = Vec::with_capacity((entity_count + 1) * 4);
    for value in &out_counts {
        out_off.extend_from_slice(&value.to_le_bytes());
    }

    for (ordinal, relation) in entity_edges.iter().chain(other.iter()).enumerate() {
        let is_entity_edge = ordinal < entity_edge_count;
        if is_entity_edge {
            let src = endpoint_ordinal(&relation.src, ordinal_of_id).ok_or_else(|| {
                KinDbError::StorageError(
                    "segment writer partitioned a relation as an entity edge whose source has no \
                     ordinal"
                        .to_string(),
                )
            })?;
            let dst = endpoint_ordinal(&relation.dst, ordinal_of_id).ok_or_else(|| {
                KinDbError::StorageError(
                    "segment writer partitioned a relation as an entity edge whose destination \
                     has no ordinal"
                        .to_string(),
                )
            })?;
            out_dst.extend_from_slice(&dst.to_le_bytes());
            rel_src.extend_from_slice(&src.to_le_bytes());
            incoming[dst as usize].push((src, ordinal as u32));
        } else {
            rel_src.extend_from_slice(&u32::MAX.to_le_bytes());
        }

        rel_id.extend_from_slice(relation.id.0.as_bytes());
        rel_kind.push(relation_kind_code(relation.kind));
        rel_confidence.extend_from_slice(&relation.confidence.to_le_bytes());
        rel_origin.push(relation_origin_code(relation.origin));

        let mut flags = 0u8;
        if is_entity_edge {
            flags |= REL_ENTITY_ENDPOINTS;
        }
        match &relation.created_in {
            Some(change) => {
                flags |= REL_HAS_CREATED_IN;
                rel_created_in.extend_from_slice(change.0.as_bytes());
            }
            None => rel_created_in.extend_from_slice(&[0u8; 32]),
        }
        if let Some(source) = &relation.import_source {
            flags |= REL_HAS_IMPORT_SOURCE;
            import_arena.extend_from_slice(source.as_bytes());
        }
        import_off.extend_from_slice(&checked_u32(import_arena.len(), "import arena")?.to_le_bytes());

        if !is_entity_edge {
            let encoded = serde_json::to_vec(&(relation.src, relation.dst))?;
            endpoints_arena.extend_from_slice(&encoded);
        }
        endpoints_off.extend_from_slice(&(endpoints_arena.len() as u64).to_le_bytes());

        if !relation.evidence.is_empty() {
            flags |= REL_HAS_EVIDENCE;
            let encoded = serde_json::to_vec(&relation.evidence)?;
            evidence_arena.extend_from_slice(&encoded);
        }
        evidence_off.extend_from_slice(&(evidence_arena.len() as u64).to_le_bytes());

        rel_flags.push(flags);
    }

    let mut in_off: Vec<u8> = Vec::with_capacity((entity_count + 1) * 4);
    let mut in_src: Vec<u8> = Vec::with_capacity(entity_edge_count * 4);
    let mut in_rel: Vec<u8> = Vec::with_capacity(entity_edge_count * 4);
    let mut running = 0u32;
    in_off.extend_from_slice(&0u32.to_le_bytes());
    for bucket in &mut incoming {
        bucket.sort_unstable();
        for (source, relation_ordinal) in bucket.iter() {
            in_src.extend_from_slice(&source.to_le_bytes());
            in_rel.extend_from_slice(&relation_ordinal.to_le_bytes());
            running += 1;
        }
        in_off.extend_from_slice(&running.to_le_bytes());
    }

    let columns = vec![
        write_bytes_column(dir, column::OUT_OFF, 4, entity_count as u64 + 1, out_off)?,
        write_bytes_column(dir, column::OUT_DST, 4, entity_edge_count as u64, out_dst)?,
        write_bytes_column(dir, column::IN_OFF, 4, entity_count as u64 + 1, in_off)?,
        write_bytes_column(dir, column::IN_SRC, 4, entity_edge_count as u64, in_src)?,
        write_bytes_column(dir, column::IN_REL_ORD, 4, entity_edge_count as u64, in_rel)?,
        write_bytes_column(dir, column::REL_ID, 16, relation_count as u64, rel_id)?,
        write_bytes_column(dir, column::REL_KIND, 1, relation_count as u64, rel_kind)?,
        write_bytes_column(
            dir,
            column::REL_CONFIDENCE,
            4,
            relation_count as u64,
            rel_confidence,
        )?,
        write_bytes_column(dir, column::REL_ORIGIN, 1, relation_count as u64, rel_origin)?,
        write_bytes_column(dir, column::REL_SRC, 4, relation_count as u64, rel_src)?,
        write_bytes_column(dir, column::REL_FLAGS, 1, relation_count as u64, rel_flags)?,
        write_bytes_column(
            dir,
            column::REL_CREATED_IN,
            32,
            relation_count as u64,
            rel_created_in,
        )?,
        write_bytes_column(
            dir,
            column::REL_IMPORT_OFF,
            4,
            relation_count as u64 + 1,
            import_off,
        )?,
        {
            let len = import_arena.len() as u64;
            write_bytes_column(dir, column::REL_IMPORT_ARENA, 1, len, import_arena)?
        },
        write_bytes_column(
            dir,
            column::REL_ENDPOINTS_OFF,
            8,
            relation_count as u64 + 1,
            endpoints_off,
        )?,
        {
            let len = endpoints_arena.len() as u64;
            write_bytes_column(dir, column::REL_ENDPOINTS_ARENA, 1, len, endpoints_arena)?
        },
        write_bytes_column(
            dir,
            column::REL_EVIDENCE_OFF,
            8,
            relation_count as u64 + 1,
            evidence_off,
        )?,
        {
            let len = evidence_arena.len() as u64;
            write_bytes_column(dir, column::REL_EVIDENCE_ARENA, 1, len, evidence_arena)?
        },
    ];

    Ok(RelationColumns {
        columns,
        relation_count: relation_count as u64,
        entity_edge_count: entity_edge_count as u64,
    })
}

fn endpoint_ordinal(
    node: &GraphNodeId,
    ordinal_of_id: &HashMap<[u8; 16], u32>,
) -> Option<u32> {
    match node {
        GraphNodeId::Entity(id) => ordinal_of_id.get(id.0.as_bytes()).copied(),
        _ => None,
    }
}

fn stage_entity(
    entity: &Entity,
    arenas: &mut Arenas,
    path_slots: &mut HashMap<String, u32>,
    path_list: &mut Vec<String>,
) -> Staged {
    let mut flags = visibility_code(entity.visibility);
    flags |= role_code(entity.role) << FLAG_ROLE_SHIFT;

    let path_slot = match &entity.file_origin {
        Some(path) => {
            flags |= FLAG_HAS_PATH;
            intern_path(&path.0, path_slots, path_list)
        }
        None => u32::MAX,
    };

    // `span.file` is a separate `Option<FilePathId>` from `file_origin` and is
    // NOT assumed to hold the same path. Interning it costs four bytes per
    // entity and buys losslessness without an equality assertion over data
    // already on disk.
    let (span, span_path_slot) = match &entity.span {
        Some(span) => {
            flags |= FLAG_HAS_SPAN;
            (
                [
                    span.start_byte as u32,
                    span.end_byte as u32,
                    span.start_line,
                    span.start_col,
                    span.end_line,
                    span.end_col,
                ],
                intern_path(&span.file.0, path_slots, path_list),
            )
        }
        None => ([0u32; 6], u32::MAX),
    };

    let name = Arenas::push(&mut arenas.names, entity.name.as_bytes());
    let lower = Arenas::push(&mut arenas.lower, entity.name.to_lowercase().as_bytes());
    let signature = Arenas::push(&mut arenas.signatures, entity.signature.as_bytes());
    let doc = match &entity.doc_summary {
        Some(text) => {
            flags |= FLAG_HAS_DOC;
            Arenas::push(&mut arenas.docs, text.as_bytes())
        }
        None => (arenas.docs.len(), 0),
    };

    let mut cold_flags = 0u8;
    let lineage_parent = match &entity.lineage_parent {
        Some(id) => {
            cold_flags |= COLD_HAS_LINEAGE;
            *id.0.as_bytes()
        }
        None => [0u8; 16],
    };
    let created_in = match &entity.created_in {
        Some(change) => {
            cold_flags |= COLD_HAS_CREATED_IN;
            *change.0.as_bytes()
        }
        None => [0u8; 32],
    };
    let superseded_by = match &entity.superseded_by {
        Some(id) => {
            cold_flags |= COLD_HAS_SUPERSEDED;
            *id.0.as_bytes()
        }
        None => [0u8; 16],
    };

    // The census on the real VS Code store put metadata at 1,937 bytes per
    // entity, 79.3 percent of the persisted entities domain, and every key in
    // it is derived retrieval text rather than semantic identity. It is the
    // reason this is a side table and not a hot column.
    let metadata = if entity.metadata.extra.is_empty() {
        (arenas.metadata.len(), 0)
    } else {
        cold_flags |= COLD_HAS_METADATA;
        let encoded = serde_json::to_vec(&entity.metadata).unwrap_or_default();
        Arenas::push(&mut arenas.metadata, &encoded)
    };

    Staged {
        id: *entity.id.0.as_bytes(),
        kind: entity_kind_code(entity.kind),
        language: language_code(entity.language),
        flags,
        cold_flags,
        ast_hash: *entity.fingerprint.ast_hash.as_bytes(),
        signature_hash: *entity.fingerprint.signature_hash.as_bytes(),
        behavior_hash: *entity.fingerprint.behavior_hash.as_bytes(),
        equivalence_hash: *entity.fingerprint.equivalence_hash.as_bytes(),
        stability: entity.fingerprint.stability_score,
        path_slot,
        span_path_slot,
        span,
        name,
        lower,
        signature,
        doc,
        lineage_parent,
        created_in,
        superseded_by,
        metadata,
    }
}

fn intern_path(
    path: &str,
    path_slots: &mut HashMap<String, u32>,
    path_list: &mut Vec<String>,
) -> u32 {
    if let Some(slot) = path_slots.get(path) {
        return *slot;
    }
    let slot = path_list.len() as u32;
    path_list.push(path.to_string());
    path_slots.insert(path.to_string(), slot);
    slot
}

fn permute_arena(
    source: &[u8],
    staged: &[Staged],
    order: &[u32],
    pick: impl Fn(&Staged) -> (usize, usize),
) -> Result<(Vec<u8>, Vec<u8>), KinDbError> {
    let mut offsets: Vec<u8> = Vec::with_capacity((order.len() + 1) * 4);
    let mut arena: Vec<u8> = Vec::new();
    offsets.extend_from_slice(&0u32.to_le_bytes());
    for slot in order {
        let (start, len) = pick(&staged[*slot as usize]);
        arena.extend_from_slice(&source[start..start + len]);
        offsets.extend_from_slice(&checked_u32(arena.len(), "string arena")?.to_le_bytes());
    }
    Ok((offsets, arena))
}

fn checked_u32(value: usize, what: &str) -> Result<u32, KinDbError> {
    u32::try_from(value).map_err(|_| {
        KinDbError::StorageError(format!(
            "segment {what} reached {value} bytes, past the u32 offset ceiling this column uses; \
             the layout needs a wider offset column before a graph this size can be written"
        ))
    })
}

fn write_fixed(
    dir: &Path,
    id: u32,
    width: u32,
    count: usize,
    mut render: impl FnMut(usize) -> Vec<u8>,
) -> Result<ColumnRecord, KinDbError> {
    let mut payload = Vec::with_capacity(count * width as usize);
    for ordinal in 0..count {
        let bytes = render(ordinal);
        debug_assert_eq!(bytes.len(), width as usize);
        payload.extend_from_slice(&bytes);
    }
    write_bytes_column(dir, id, width, count as u64, payload)
}

fn write_bytes_column(
    dir: &Path,
    id: u32,
    width: u32,
    count: u64,
    payload: Vec<u8>,
) -> Result<ColumnRecord, KinDbError> {
    let payload_len = payload.len() as u64;
    let mut file = Vec::with_capacity(HEADER_LEN + payload.len() + DIGEST_LEN);
    file.extend_from_slice(&encode_header(id, width, count, payload_len));
    file.extend_from_slice(&payload);
    let digest = Sha256::digest(&file);
    let mut digest_bytes = [0u8; 32];
    digest_bytes.copy_from_slice(&digest);
    file.extend_from_slice(&digest_bytes);

    atomic_write_bytes_no_magic(&dir.join(column_file_name(id)), &file)?;

    Ok(ColumnRecord {
        id,
        width,
        count,
        payload_len,
        digest: digest_bytes,
    })
}

fn write_manifest(
    dir: &Path,
    shape: &SegmentShape,
    columns: &[ColumnRecord],
) -> Result<u64, KinDbError> {
    let mut payload =
        Vec::with_capacity(MANIFEST_PREAMBLE_LEN + columns.len() * MANIFEST_ENTRY_LEN);
    payload.extend_from_slice(&shape.entity_count.to_le_bytes());
    payload.extend_from_slice(&shape.relation_count.to_le_bytes());
    payload.extend_from_slice(&shape.entity_edge_count.to_le_bytes());
    payload.extend_from_slice(&shape.path_count.to_le_bytes());
    payload.extend_from_slice(&shape.name_key_count.to_le_bytes());
    for record in columns {
        payload.extend_from_slice(&record.id.to_le_bytes());
        payload.extend_from_slice(&record.width.to_le_bytes());
        payload.extend_from_slice(&record.count.to_le_bytes());
        payload.extend_from_slice(&record.payload_len.to_le_bytes());
        payload.extend_from_slice(&record.digest);
    }

    let payload_len = payload.len() as u64;
    let mut file = Vec::with_capacity(HEADER_LEN + payload.len() + DIGEST_LEN);
    file.extend_from_slice(&encode_header(
        column::MANIFEST,
        MANIFEST_ENTRY_LEN as u32,
        columns.len() as u64,
        payload_len,
    ));
    file.extend_from_slice(&payload);
    let digest = Sha256::digest(&file);
    file.extend_from_slice(&digest);

    let on_disk = file.len() as u64;
    atomic_write_bytes_no_magic(&dir.join(MANIFEST_FILE), &file)?;
    Ok(on_disk)
}
