// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Typed views over a mapped segment. Nothing is decoded into owned structs at
//! open, and every read below is a slice of a `memmap2::Mmap` plus a
//! `from_le_bytes` on a fixed-width window.
//!
//! Two deliberate choices are worth reading before the code.
//!
//! **Columns are mapped by profile, not lazily one at a time.** A lazy map
//! behind a lock could not hand out `&str` borrowed from the mapping, which is
//! the whole point, so [`OpenProfile`] decides the set up front.
//! [`OpenProfile::Hot`] maps what the `ReadIndex` query set reads and leaves
//! the cold hashes and the metadata side table out of the address space
//! entirely, so they cannot be read ahead into resident pages either.
//!
//! **Digests are checked against the manifest at open by LENGTH, and by
//! content only on request.** Hashing every column at open is exactly the
//! O(store) cost this design exists to remove. A torn or truncated column is
//! caught at open, because the manifest states each column's declared payload
//! length and its file length must agree. Bit rot inside an intact-looking
//! column is caught by [`SegmentReader::verify_all`], which is what a doctor
//! check and the round-trip proof call.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use sha2::{Digest, Sha256};

use kin_model::{
    Entity, EntityId, EntityKind, EntityMetadata, EntityRole, FilePathId, GraphNodeId, Hash256,
    LanguageId, Relation, RelationEvidence, RelationId, RelationKind, RelationOrigin,
    SemanticChangeId, SemanticFingerprint, SourceSpan, Visibility,
};

use crate::error::KinDbError;
use crate::storage::mmap::open_regular_nofollow;
use crate::storage::segment::format::{
    column, column_file_name, decode_header, entity_kind_of_code, language_of_code, read_u32,
    read_u64, relation_kind_of_code, relation_origin_of_code, role_of_code, visibility_of_code,
    ColumnRecord, SegmentShape, COLD_HAS_CREATED_IN, COLD_HAS_LINEAGE, COLD_HAS_METADATA,
    COLD_HAS_SUPERSEDED, DIGEST_LEN, FLAG_HAS_DOC, FLAG_HAS_PATH, FLAG_HAS_SPAN, FLAG_ROLE_MASK,
    FLAG_ROLE_SHIFT, FLAG_VISIBILITY_MASK, HEADER_LEN, MANIFEST_ENTRY_LEN, MANIFEST_FILE,
    MANIFEST_PREAMBLE_LEN, REL_ENTITY_ENDPOINTS, REL_HAS_CREATED_IN, REL_HAS_EVIDENCE,
    REL_HAS_IMPORT_SOURCE,
};

/// Which columns an open maps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenProfile {
    /// Only what an entity lookup, an adjacency walk, a name search and a
    /// count-by-kind read. The cold fingerprint hashes, the lineage refs and
    /// the metadata side table stay off the address space.
    Hot,
    /// Every column, which is what a full [`Entity`] or [`Relation`]
    /// reconstruction and [`SegmentReader::verify_all`] need.
    Full,
}

/// Columns [`OpenProfile::Hot`] maps, and which a hot open refuses without.
pub const HOT_REQUIRED: &[u32] = &[
    column::ENTITY_ID,
    column::ENTITY_KIND,
    column::ENTITY_LANGUAGE,
    column::ENTITY_FLAGS,
    column::ENTITY_PATH_ORD,
    column::ENTITY_SPAN,
    column::ENTITY_NAME_OFF,
    column::ENTITY_NAME_ARENA,
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
];

/// Columns beyond [`HOT_REQUIRED`] that a hot open also maps because they are
/// small and every hot answer wants them.
const HOT_ALSO: &[u32] = &[
    column::ENTITY_AST_HASH,
    column::ENTITY_SIG_OFF,
    column::ENTITY_SIG_ARENA,
    column::ENTITY_DOC_OFF,
    column::ENTITY_DOC_ARENA,
    column::ENTITY_SPAN_PATH_ORD,
    column::REL_ORIGIN,
    column::REL_SRC,
    column::REL_FLAGS,
];

/// A u32 ordinal list borrowed from a mapping.
#[derive(Debug, Clone, Copy)]
pub struct Ordinals<'a>(&'a [u8]);

impl<'a> Ordinals<'a> {
    /// How many ordinals the slice holds.
    pub fn len(&self) -> usize {
        self.0.len() / 4
    }

    /// True when the slice holds no ordinal.
    pub fn is_empty(&self) -> bool {
        self.0.len() < 4
    }

    /// The ordinal at `position`, or `None` past the end.
    pub fn get(&self, position: usize) -> Option<u32> {
        let start = position.checked_mul(4)?;
        let slice = self.0.get(start..start + 4)?;
        let mut buf = [0u8; 4];
        buf.copy_from_slice(slice);
        Some(u32::from_le_bytes(buf))
    }

    /// Every ordinal in order.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.0.chunks_exact(4).map(|chunk| {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(chunk);
            u32::from_le_bytes(buf)
        })
    }
}

struct MappedColumn {
    mapping: Mmap,
    width: u32,
    count: u64,
    payload_len: u64,
}

impl MappedColumn {
    fn payload(&self) -> &[u8] {
        let end = HEADER_LEN + self.payload_len as usize;
        &self.mapping[HEADER_LEN..end]
    }

    fn record(&self, index: usize) -> Option<&[u8]> {
        let width = self.width as usize;
        if width == 0 {
            return None;
        }
        let start = index.checked_mul(width)?;
        self.payload().get(start..start.checked_add(width)?)
    }
}

/// A mapped columnar segment, opened read-only.
pub struct SegmentReader {
    dir: PathBuf,
    shape: SegmentShape,
    manifest: Vec<ColumnRecord>,
    columns: BTreeMap<u32, MappedColumn>,
    profile: OpenProfile,
    unknown_columns: usize,
}

impl SegmentReader {
    /// Open the segment in `dir` with the hot profile.
    pub fn open(dir: &Path) -> Result<Self, KinDbError> {
        Self::open_with_profile(dir, OpenProfile::Hot)
    }

    /// Open the segment in `dir`, mapping the columns `profile` names.
    pub fn open_with_profile(dir: &Path, profile: OpenProfile) -> Result<Self, KinDbError> {
        let _span = tracing::info_span!(
            "kindb.segment.open",
            dir = %dir.display(),
            profile = ?profile
        )
        .entered();

        let manifest_path = dir.join(MANIFEST_FILE);
        let bytes = std::fs::read(&manifest_path).map_err(|error| {
            KinDbError::StorageError(format!(
                "failed to read segment manifest {}: {error}",
                manifest_path.display()
            ))
        })?;
        let header = decode_header(&bytes, "manifest")?;
        if header.id != column::MANIFEST {
            return Err(KinDbError::StorageError(format!(
                "segment manifest declares column id {}, expected {}",
                header.id,
                column::MANIFEST
            )));
        }
        let digest_start = HEADER_LEN + header.payload_len as usize;
        let computed = Sha256::digest(&bytes[..digest_start]);
        if computed.as_slice() != &bytes[digest_start..digest_start + DIGEST_LEN] {
            return Err(KinDbError::StorageError(
                "segment manifest digest does not match its own bytes; rebuild the segment".into(),
            ));
        }

        let payload = &bytes[HEADER_LEN..digest_start];
        if payload.len() < MANIFEST_PREAMBLE_LEN {
            return Err(KinDbError::StorageError(format!(
                "segment manifest payload is {} bytes, shorter than its {MANIFEST_PREAMBLE_LEN} \
                 byte preamble",
                payload.len()
            )));
        }
        let shape = SegmentShape {
            entity_count: read_u64(payload, 0)?,
            relation_count: read_u64(payload, 8)?,
            entity_edge_count: read_u64(payload, 16)?,
            path_count: read_u64(payload, 24)?,
            name_key_count: read_u64(payload, 32)?,
        };

        let mut manifest = Vec::with_capacity(header.count as usize);
        for index in 0..header.count as usize {
            let base = MANIFEST_PREAMBLE_LEN + index * MANIFEST_ENTRY_LEN;
            let digest_at = base + 24;
            let digest_bytes = payload.get(digest_at..digest_at + 32).ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment manifest declares {} columns and its table stops at entry {index}",
                    header.count
                ))
            })?;
            let mut digest = [0u8; 32];
            digest.copy_from_slice(digest_bytes);
            manifest.push(ColumnRecord {
                id: read_u32(payload, base)?,
                width: read_u32(payload, base + 4)?,
                count: read_u64(payload, base + 8)?,
                payload_len: read_u64(payload, base + 16)?,
                digest,
            });
        }

        let wanted: Vec<u32> = match profile {
            OpenProfile::Hot => HOT_REQUIRED
                .iter()
                .chain(HOT_ALSO.iter())
                .copied()
                .collect(),
            OpenProfile::Full => manifest.iter().map(|record| record.id).collect(),
        };

        let mut columns = BTreeMap::new();
        let mut unknown_columns = 0usize;
        for record in &manifest {
            if !wanted.contains(&record.id) {
                // A column this binary does not know, or one the profile does
                // not read. Skipping it is what makes reading one version
                // forward sound while the newer version is additive.
                if !is_known_column(record.id) {
                    unknown_columns += 1;
                }
                continue;
            }
            let mapped = map_column(dir, record)?;
            columns.insert(record.id, mapped);
        }

        if matches!(profile, OpenProfile::Hot) {
            for id in HOT_REQUIRED {
                if !columns.contains_key(id) {
                    return Err(KinDbError::StorageError(format!(
                        "segment is missing required column {id}; the segment was written by an \
                         older layout than this binary reads, so rebuild it"
                    )));
                }
            }
        }

        Ok(Self {
            dir: dir.to_path_buf(),
            shape,
            manifest,
            columns,
            profile,
            unknown_columns,
        })
    }

    /// What the manifest states about the segment.
    pub fn shape(&self) -> SegmentShape {
        self.shape
    }

    /// Which columns this open mapped.
    pub fn profile(&self) -> OpenProfile {
        self.profile
    }

    /// Columns in the manifest whose id this binary does not know. Non-zero
    /// means the segment came from a newer, additive layout.
    pub fn unknown_columns(&self) -> usize {
        self.unknown_columns
    }

    /// Entities in the segment, which is also the size of the ordinal space.
    pub fn entity_count(&self) -> u32 {
        self.shape.entity_count.min(u64::from(u32::MAX)) as u32
    }

    /// Relations in the segment.
    pub fn relation_count(&self) -> u32 {
        self.shape.relation_count.min(u64::from(u32::MAX)) as u32
    }

    /// Relations whose source and destination are both entities.
    pub fn entity_edge_count(&self) -> u32 {
        self.shape.entity_edge_count.min(u64::from(u32::MAX)) as u32
    }

    /// Hash every mapped column and compare it against the manifest.
    ///
    /// This is the O(bytes) check the open deliberately does not do. Call it
    /// from a doctor path or a proof, not from a serving open.
    pub fn verify_all(&self) -> Result<usize, KinDbError> {
        let _span = tracing::info_span!("kindb.segment.verify_all").entered();
        let mut checked = 0usize;
        for record in &self.manifest {
            let Some(mapped) = self.columns.get(&record.id) else {
                continue;
            };
            let end = HEADER_LEN + record.payload_len as usize;
            let computed = Sha256::digest(&mapped.mapping[..end]);
            if computed.as_slice() != record.digest {
                return Err(KinDbError::StorageError(format!(
                    "segment column {} does not match the digest the manifest records for it",
                    record.id
                )));
            }
            checked += 1;
        }
        Ok(checked)
    }

    /// The directory this segment was opened from.
    pub fn directory(&self) -> &Path {
        &self.dir
    }

    // -----------------------------------------------------------------------
    // Entity reads
    // -----------------------------------------------------------------------

    /// The ordinal of `id`, by binary search over the id column.
    ///
    /// Entities are written in id order, so the ordinal IS the id rank and this
    /// needs no separate index. That is why a sorted fixed-width column beats
    /// an FST here: the lookup structure is the data.
    pub fn ordinal_of_entity(&self, id: &EntityId) -> Result<Option<u32>, KinDbError> {
        let ids = self.column(column::ENTITY_ID)?;
        let needle = id.0.as_bytes();
        let payload = ids.payload();
        let count = ids.count as usize;
        let mut low = 0usize;
        let mut high = count;
        while low < high {
            let middle = low + (high - low) / 2;
            let start = middle * 16;
            let candidate = payload.get(start..start + 16).ok_or_else(|| {
                KinDbError::StorageError(
                    "segment id column is shorter than the count its header declares".into(),
                )
            })?;
            match candidate.cmp(needle.as_slice()) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return Ok(Some(middle as u32)),
            }
        }
        Ok(None)
    }

    /// The id at `ordinal`.
    pub fn entity_id(&self, ordinal: u32) -> Result<EntityId, KinDbError> {
        let bytes = self.fixed(column::ENTITY_ID, ordinal, "entity id")?;
        let mut raw = [0u8; 16];
        raw.copy_from_slice(bytes);
        Ok(EntityId(uuid::Uuid::from_bytes(raw)))
    }

    /// The kind at `ordinal`.
    pub fn entity_kind(&self, ordinal: u32) -> Result<EntityKind, KinDbError> {
        entity_kind_of_code(self.fixed(column::ENTITY_KIND, ordinal, "entity kind")?[0])
    }

    /// The language at `ordinal`.
    pub fn entity_language(&self, ordinal: u32) -> Result<LanguageId, KinDbError> {
        language_of_code(self.fixed(column::ENTITY_LANGUAGE, ordinal, "entity language")?[0])
    }

    /// The visibility at `ordinal`.
    pub fn entity_visibility(&self, ordinal: u32) -> Result<Visibility, KinDbError> {
        visibility_of_code(self.entity_flags(ordinal)? & FLAG_VISIBILITY_MASK)
    }

    /// The role at `ordinal`.
    pub fn entity_role(&self, ordinal: u32) -> Result<EntityRole, KinDbError> {
        role_of_code((self.entity_flags(ordinal)? >> FLAG_ROLE_SHIFT) & FLAG_ROLE_MASK)
    }

    /// The name at `ordinal`, borrowed from the mapping.
    pub fn entity_name(&self, ordinal: u32) -> Result<&str, KinDbError> {
        self.arena_slice_raw(
            column::ENTITY_NAME_OFF,
            column::ENTITY_NAME_ARENA,
            ordinal,
            "entity name",
        )
    }

    /// The signature at `ordinal`, borrowed from the mapping.
    pub fn entity_signature(&self, ordinal: u32) -> Result<&str, KinDbError> {
        self.arena_slice_raw(
            column::ENTITY_SIG_OFF,
            column::ENTITY_SIG_ARENA,
            ordinal,
            "entity signature",
        )
    }

    /// The doc summary at `ordinal`, or `None` when the entity carries none.
    pub fn entity_doc_summary(&self, ordinal: u32) -> Result<Option<&str>, KinDbError> {
        if self.entity_flags(ordinal)? & FLAG_HAS_DOC == 0 {
            return Ok(None);
        }
        self.arena_slice_raw(
            column::ENTITY_DOC_OFF,
            column::ENTITY_DOC_ARENA,
            ordinal,
            "entity doc summary",
        )
        .map(Some)
    }

    /// The file path at `ordinal`, or `None` when the entity carries none.
    pub fn entity_path(&self, ordinal: u32) -> Result<Option<&str>, KinDbError> {
        if self.entity_flags(ordinal)? & FLAG_HAS_PATH == 0 {
            return Ok(None);
        }
        let slot = read_u32(
            self.fixed(column::ENTITY_PATH_ORD, ordinal, "entity path")?,
            0,
        )?;
        self.path(slot).map(Some)
    }

    /// The path at `slot` in the path table.
    pub fn path(&self, slot: u32) -> Result<&str, KinDbError> {
        self.arena_slice_raw(column::PATH_OFF, column::PATH_ARENA, slot, "path table")
    }

    /// The `ast_hash` at `ordinal`, the one fingerprint hash the hot profile
    /// maps. The other three live in cold columns.
    pub fn entity_ast_hash(&self, ordinal: u32) -> Result<Hash256, KinDbError> {
        let bytes = self.fixed(column::ENTITY_AST_HASH, ordinal, "entity ast hash")?;
        let mut raw = [0u8; 32];
        raw.copy_from_slice(bytes);
        Ok(Hash256::from_bytes(raw))
    }

    /// The span at `ordinal`, or `None` when the entity carries none.
    pub fn entity_span(&self, ordinal: u32) -> Result<Option<SourceSpan>, KinDbError> {
        if self.entity_flags(ordinal)? & FLAG_HAS_SPAN == 0 {
            return Ok(None);
        }
        let bytes = self.fixed(column::ENTITY_SPAN, ordinal, "entity span")?;
        let slot = read_u32(
            self.fixed(column::ENTITY_SPAN_PATH_ORD, ordinal, "entity span path")?,
            0,
        )?;
        let file = if slot == u32::MAX {
            FilePathId(String::new())
        } else {
            FilePathId(self.path(slot)?.to_string())
        };
        Ok(Some(SourceSpan {
            file,
            start_byte: read_u32(bytes, 0)? as usize,
            end_byte: read_u32(bytes, 4)? as usize,
            start_line: read_u32(bytes, 8)?,
            start_col: read_u32(bytes, 12)?,
            end_line: read_u32(bytes, 16)?,
            end_col: read_u32(bytes, 20)?,
        }))
    }

    /// The first line of the span at `ordinal`, or zero, which is what
    /// `ReadIndex` exposes.
    pub fn entity_start_line(&self, ordinal: u32) -> Result<u32, KinDbError> {
        if self.entity_flags(ordinal)? & FLAG_HAS_SPAN == 0 {
            return Ok(0);
        }
        read_u32(self.fixed(column::ENTITY_SPAN, ordinal, "entity span")?, 8)
    }

    /// Ordinals whose lowercased name equals `name` lowercased.
    ///
    /// The key arena holds what `str::to_lowercase` produced at write time, so
    /// this is the same fold `ReadIndex` applies rather than an ASCII
    /// approximation of it.
    pub fn entities_by_name(&self, name: &str) -> Result<Ordinals<'_>, KinDbError> {
        let needle = name.to_lowercase();
        let keys = self.column(column::NAME_KEY_OFF)?;
        let count = keys.count.saturating_sub(1) as usize;
        let mut low = 0usize;
        let mut high = count;
        while low < high {
            let middle = low + (high - low) / 2;
            let candidate = self.arena_slice_raw(
                column::NAME_KEY_OFF,
                column::NAME_KEY_ARENA,
                middle as u32,
                "name key",
            )?;
            match candidate.cmp(needle.as_str()) {
                std::cmp::Ordering::Less => low = middle + 1,
                std::cmp::Ordering::Greater => high = middle,
                std::cmp::Ordering::Equal => return self.postings(middle as u32),
            }
        }
        Ok(Ordinals(&[]))
    }

    fn postings(&self, key: u32) -> Result<Ordinals<'_>, KinDbError> {
        let offsets = self.column(column::NAME_POSTING_OFF)?;
        let postings = self.column(column::NAME_POSTINGS)?;
        let start = read_u32(
            offsets.record(key as usize).ok_or_else(|| {
                KinDbError::StorageError("segment name posting offset is out of range".into())
            })?,
            0,
        )? as usize;
        let end = read_u32(
            offsets.record(key as usize + 1).ok_or_else(|| {
                KinDbError::StorageError("segment name posting offset is out of range".into())
            })?,
            0,
        )? as usize;
        let bytes = postings.payload().get(start * 4..end * 4).ok_or_else(|| {
            KinDbError::StorageError("segment name postings run past the column".into())
        })?;
        Ok(Ordinals(bytes))
    }

    /// Entity counts by kind wire code, computed by scanning the one-byte kind
    /// column. A four-million-entity graph is a four-megabyte scan, which is
    /// cheaper than persisting and trusting a count map.
    pub fn kind_counts(&self) -> Result<BTreeMap<u8, u32>, KinDbError> {
        Ok(tally(self.column(column::ENTITY_KIND)?.payload()))
    }

    /// Entity counts by language wire code.
    pub fn language_counts(&self) -> Result<BTreeMap<u8, u32>, KinDbError> {
        Ok(tally(self.column(column::ENTITY_LANGUAGE)?.payload()))
    }

    // -----------------------------------------------------------------------
    // Adjacency
    // -----------------------------------------------------------------------

    /// Destination ordinals of the entity at `ordinal`.
    pub fn outgoing(&self, ordinal: u32) -> Result<Ordinals<'_>, KinDbError> {
        let (start, end) = self.csr_range(column::OUT_OFF, ordinal)?;
        self.slice_u32(column::OUT_DST, start, end, "outgoing")
    }

    /// Relation ordinals of the entity at `ordinal`, which are exactly the
    /// positions of its outgoing edges. Sorting relations by source ordinal at
    /// write time is what makes the CSR the relation ordinal space itself.
    pub fn outgoing_relations(&self, ordinal: u32) -> Result<std::ops::Range<u32>, KinDbError> {
        let (start, end) = self.csr_range(column::OUT_OFF, ordinal)?;
        Ok(start..end)
    }

    /// Source ordinals pointing at the entity at `ordinal`.
    pub fn incoming(&self, ordinal: u32) -> Result<Ordinals<'_>, KinDbError> {
        let (start, end) = self.csr_range(column::IN_OFF, ordinal)?;
        self.slice_u32(column::IN_SRC, start, end, "incoming")
    }

    /// Relation ordinals pointing at the entity at `ordinal`.
    pub fn incoming_relations(&self, ordinal: u32) -> Result<Ordinals<'_>, KinDbError> {
        let (start, end) = self.csr_range(column::IN_OFF, ordinal)?;
        self.slice_u32(column::IN_REL_ORD, start, end, "incoming relations")
    }

    // -----------------------------------------------------------------------
    // Relation reads
    // -----------------------------------------------------------------------

    /// The relation kind at `relation_ordinal`.
    pub fn relation_kind(&self, relation_ordinal: u32) -> Result<RelationKind, KinDbError> {
        relation_kind_of_code(self.fixed(column::REL_KIND, relation_ordinal, "relation kind")?[0])
    }

    /// The confidence at `relation_ordinal`.
    pub fn relation_confidence(&self, relation_ordinal: u32) -> Result<f32, KinDbError> {
        let bytes = self.fixed(
            column::REL_CONFIDENCE,
            relation_ordinal,
            "relation confidence",
        )?;
        let mut raw = [0u8; 4];
        raw.copy_from_slice(bytes);
        Ok(f32::from_le_bytes(raw))
    }

    /// The origin at `relation_ordinal`.
    pub fn relation_origin(&self, relation_ordinal: u32) -> Result<RelationOrigin, KinDbError> {
        relation_origin_of_code(
            self.fixed(column::REL_ORIGIN, relation_ordinal, "relation origin")?[0],
        )
    }

    /// The source entity ordinal at `relation_ordinal`, or `None` when the
    /// relation's source is not an entity.
    pub fn relation_source_ordinal(
        &self,
        relation_ordinal: u32,
    ) -> Result<Option<u32>, KinDbError> {
        let value = read_u32(
            self.fixed(column::REL_SRC, relation_ordinal, "relation source")?,
            0,
        )?;
        Ok(if value == u32::MAX { None } else { Some(value) })
    }

    // -----------------------------------------------------------------------
    // Full reconstruction, which needs OpenProfile::Full
    // -----------------------------------------------------------------------

    /// Rebuild the whole [`Entity`] at `ordinal`. Needs [`OpenProfile::Full`].
    pub fn entity(&self, ordinal: u32) -> Result<Entity, KinDbError> {
        let cold = self.fixed(column::ENTITY_COLD_FLAGS, ordinal, "entity cold flags")?[0];
        let metadata = if cold & COLD_HAS_METADATA == 0 {
            EntityMetadata::default()
        } else {
            let bytes = self.side_table(
                column::ENTITY_METADATA_OFF,
                column::ENTITY_METADATA_ARENA,
                ordinal,
                "entity metadata",
            )?;
            serde_json::from_slice(bytes)?
        };
        Ok(Entity {
            id: self.entity_id(ordinal)?,
            kind: self.entity_kind(ordinal)?,
            name: self.entity_name(ordinal)?.to_string(),
            language: self.entity_language(ordinal)?,
            fingerprint: SemanticFingerprint {
                algorithm: kin_model::FingerprintAlgorithm::V1TreeSitter,
                ast_hash: self.entity_ast_hash(ordinal)?,
                signature_hash: self.cold_hash(column::ENTITY_SIGNATURE_HASH, ordinal)?,
                behavior_hash: self.cold_hash(column::ENTITY_BEHAVIOR_HASH, ordinal)?,
                equivalence_hash: self.cold_hash(column::ENTITY_EQUIVALENCE_HASH, ordinal)?,
                stability_score: {
                    let bytes =
                        self.fixed(column::ENTITY_STABILITY, ordinal, "entity stability")?;
                    let mut raw = [0u8; 4];
                    raw.copy_from_slice(bytes);
                    f32::from_le_bytes(raw)
                },
            },
            file_origin: self
                .entity_path(ordinal)?
                .map(|p| FilePathId(p.to_string())),
            span: self.entity_span(ordinal)?,
            signature: self.entity_signature(ordinal)?.to_string(),
            visibility: self.entity_visibility(ordinal)?,
            role: self.entity_role(ordinal)?,
            doc_summary: self.entity_doc_summary(ordinal)?.map(str::to_string),
            metadata,
            lineage_parent: if cold & COLD_HAS_LINEAGE == 0 {
                None
            } else {
                Some(self.cold_uuid(column::ENTITY_LINEAGE_PARENT, ordinal)?)
            },
            created_in: if cold & COLD_HAS_CREATED_IN == 0 {
                None
            } else {
                Some(SemanticChangeId(
                    self.cold_hash(column::ENTITY_CREATED_IN, ordinal)?,
                ))
            },
            superseded_by: if cold & COLD_HAS_SUPERSEDED == 0 {
                None
            } else {
                Some(self.cold_uuid(column::ENTITY_SUPERSEDED_BY, ordinal)?)
            },
        })
    }

    /// Rebuild the whole [`Relation`] at `relation_ordinal`. Needs
    /// [`OpenProfile::Full`].
    pub fn relation(&self, relation_ordinal: u32) -> Result<Relation, KinDbError> {
        let flags = self.fixed(column::REL_FLAGS, relation_ordinal, "relation flags")?[0];
        let id = {
            let bytes = self.fixed(column::REL_ID, relation_ordinal, "relation id")?;
            let mut raw = [0u8; 16];
            raw.copy_from_slice(bytes);
            RelationId(uuid::Uuid::from_bytes(raw))
        };

        let (src, dst) = if flags & REL_ENTITY_ENDPOINTS != 0 {
            let source = self
                .relation_source_ordinal(relation_ordinal)?
                .ok_or_else(|| {
                    KinDbError::StorageError(
                        "segment relation is flagged as an entity edge and carries no source \
                         ordinal"
                            .into(),
                    )
                })?;
            let target = self
                .column(column::OUT_DST)?
                .record(relation_ordinal as usize)
                .ok_or_else(|| {
                    KinDbError::StorageError(
                        "segment relation is flagged as an entity edge past the end of the \
                         outgoing column"
                            .into(),
                    )
                })?;
            (
                GraphNodeId::Entity(self.entity_id(source)?),
                GraphNodeId::Entity(self.entity_id(read_u32(target, 0)?)?),
            )
        } else {
            let bytes = self.side_table(
                column::REL_ENDPOINTS_OFF,
                column::REL_ENDPOINTS_ARENA,
                relation_ordinal,
                "relation endpoints",
            )?;
            serde_json::from_slice(bytes)?
        };

        let evidence: Vec<RelationEvidence> = if flags & REL_HAS_EVIDENCE == 0 {
            Vec::new()
        } else {
            let bytes = self.side_table(
                column::REL_EVIDENCE_OFF,
                column::REL_EVIDENCE_ARENA,
                relation_ordinal,
                "relation evidence",
            )?;
            serde_json::from_slice(bytes)?
        };

        Ok(Relation {
            id,
            kind: self.relation_kind(relation_ordinal)?,
            src,
            dst,
            confidence: self.relation_confidence(relation_ordinal)?,
            origin: self.relation_origin(relation_ordinal)?,
            created_in: if flags & REL_HAS_CREATED_IN == 0 {
                None
            } else {
                Some(SemanticChangeId(
                    self.cold_hash(column::REL_CREATED_IN, relation_ordinal)?,
                ))
            },
            import_source: if flags & REL_HAS_IMPORT_SOURCE == 0 {
                None
            } else {
                Some(
                    self.arena_slice_raw(
                        column::REL_IMPORT_OFF,
                        column::REL_IMPORT_ARENA,
                        relation_ordinal,
                        "relation import source",
                    )?
                    .to_string(),
                )
            },
            evidence,
        })
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    fn column(&self, id: u32) -> Result<&MappedColumn, KinDbError> {
        self.columns.get(&id).ok_or_else(|| {
            KinDbError::StorageError(format!(
                "segment column {id} is not mapped by this open; it belongs to the full profile \
                 rather than the hot one"
            ))
        })
    }

    fn entity_flags(&self, ordinal: u32) -> Result<u8, KinDbError> {
        Ok(self.fixed(column::ENTITY_FLAGS, ordinal, "entity flags")?[0])
    }

    fn fixed(&self, id: u32, index: u32, role: &str) -> Result<&[u8], KinDbError> {
        self.column(id)?.record(index as usize).ok_or_else(|| {
            KinDbError::StorageError(format!(
                "segment {role} at ordinal {index} is past the end of column {id}"
            ))
        })
    }

    fn cold_hash(&self, id: u32, ordinal: u32) -> Result<Hash256, KinDbError> {
        let bytes = self.fixed(id, ordinal, "cold hash")?;
        let mut raw = [0u8; 32];
        raw.copy_from_slice(bytes);
        Ok(Hash256::from_bytes(raw))
    }

    fn cold_uuid(&self, id: u32, ordinal: u32) -> Result<EntityId, KinDbError> {
        let bytes = self.fixed(id, ordinal, "cold id")?;
        let mut raw = [0u8; 16];
        raw.copy_from_slice(bytes);
        Ok(EntityId(uuid::Uuid::from_bytes(raw)))
    }

    fn csr_range(&self, offsets: u32, ordinal: u32) -> Result<(u32, u32), KinDbError> {
        let column = self.column(offsets)?;
        let start = read_u32(
            column.record(ordinal as usize).ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment adjacency offset for ordinal {ordinal} is past the end"
                ))
            })?,
            0,
        )?;
        let end = read_u32(
            column.record(ordinal as usize + 1).ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment adjacency offset for ordinal {ordinal} has no successor, so the \
                     offset column is one entry short of the entity count"
                ))
            })?,
            0,
        )?;
        if end < start {
            return Err(KinDbError::StorageError(format!(
                "segment adjacency offsets for ordinal {ordinal} run backwards, {start} to {end}"
            )));
        }
        Ok((start, end))
    }

    fn slice_u32(
        &self,
        id: u32,
        start: u32,
        end: u32,
        role: &str,
    ) -> Result<Ordinals<'_>, KinDbError> {
        let payload = self.column(id)?.payload();
        let bytes = payload
            .get(start as usize * 4..end as usize * 4)
            .ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment {role} range {start}..{end} runs past column {id}"
                ))
            })?;
        Ok(Ordinals(bytes))
    }

    fn arena_slice_raw(
        &self,
        offsets: u32,
        arena: u32,
        index: u32,
        role: &str,
    ) -> Result<&str, KinDbError> {
        let offset_column = self.column(offsets)?;
        let start = read_u32(
            offset_column.record(index as usize).ok_or_else(|| {
                KinDbError::StorageError(format!("segment {role} offset {index} is past the end"))
            })?,
            0,
        )? as usize;
        let end = read_u32(
            offset_column.record(index as usize + 1).ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment {role} offset {index} has no successor, so the offset column is one \
                     entry short"
                ))
            })?,
            0,
        )? as usize;
        let bytes = self
            .column(arena)?
            .payload()
            .get(start..end)
            .ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment {role} range {start}..{end} runs past its arena"
                ))
            })?;
        std::str::from_utf8(bytes).map_err(|error| {
            KinDbError::StorageError(format!("segment {role} is not valid UTF-8: {error}"))
        })
    }

    fn side_table(
        &self,
        offsets: u32,
        arena: u32,
        index: u32,
        role: &str,
    ) -> Result<&[u8], KinDbError> {
        let offset_column = self.column(offsets)?;
        let start = read_u64(
            offset_column.record(index as usize).ok_or_else(|| {
                KinDbError::StorageError(format!("segment {role} offset {index} is past the end"))
            })?,
            0,
        )? as usize;
        let end = read_u64(
            offset_column.record(index as usize + 1).ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment {role} offset {index} has no successor, so the offset column is one \
                     entry short"
                ))
            })?,
            0,
        )? as usize;
        self.column(arena)?
            .payload()
            .get(start..end)
            .ok_or_else(|| {
                KinDbError::StorageError(format!(
                    "segment {role} range {start}..{end} runs past its side table"
                ))
            })
    }
}

fn tally(bytes: &[u8]) -> BTreeMap<u8, u32> {
    let mut counts = BTreeMap::new();
    for byte in bytes {
        *counts.entry(*byte).or_insert(0u32) += 1;
    }
    counts
}

fn is_known_column(id: u32) -> bool {
    const KNOWN: &[u32] = &[
        column::MANIFEST,
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
        column::REL_ID,
        column::REL_KIND,
        column::REL_CONFIDENCE,
        column::REL_ORIGIN,
        column::REL_SRC,
        column::REL_FLAGS,
        column::ENTITY_COLD_FLAGS,
        column::ENTITY_SIGNATURE_HASH,
        column::ENTITY_BEHAVIOR_HASH,
        column::ENTITY_EQUIVALENCE_HASH,
        column::ENTITY_STABILITY,
        column::ENTITY_LINEAGE_PARENT,
        column::ENTITY_CREATED_IN,
        column::ENTITY_SUPERSEDED_BY,
        column::REL_CREATED_IN,
        column::REL_IMPORT_OFF,
        column::REL_IMPORT_ARENA,
        column::ENTITY_METADATA_OFF,
        column::ENTITY_METADATA_ARENA,
        column::REL_ENDPOINTS_OFF,
        column::REL_ENDPOINTS_ARENA,
        column::REL_EVIDENCE_OFF,
        column::REL_EVIDENCE_ARENA,
    ];
    KNOWN.contains(&id)
}

fn map_column(dir: &Path, record: &ColumnRecord) -> Result<MappedColumn, KinDbError> {
    let path = dir.join(column_file_name(record.id));
    let file = open_regular_nofollow(&path, "segment column")?;
    let metadata = file.metadata().map_err(|error| {
        KinDbError::StorageError(format!(
            "failed to stat segment column {}: {error}",
            path.display()
        ))
    })?;
    let expected = HEADER_LEN as u64 + record.payload_len + DIGEST_LEN as u64;
    if metadata.len() != expected {
        return Err(KinDbError::StorageError(format!(
            "segment column {} is {} bytes and the manifest declares {expected}; the column is \
             torn or truncated",
            record.id,
            metadata.len()
        )));
    }

    let mapping = unsafe {
        let _span = tracing::info_span!("kindb.segment.map_column", column = record.id).entered();
        Mmap::map(&file).map_err(|error| {
            KinDbError::StorageError(format!("failed to mmap {}: {error}", path.display()))
        })?
    };

    let header = decode_header(&mapping, "column")?;
    if header.id != record.id || header.width != record.width || header.count != record.count {
        return Err(KinDbError::StorageError(format!(
            "segment column {} disagrees with the manifest: header says id {} width {} count {}, \
             manifest says width {} count {}",
            record.id, header.id, header.width, header.count, record.width, record.count
        )));
    }

    Ok(MappedColumn {
        mapping,
        width: record.width,
        count: record.count,
        payload_len: record.payload_len,
    })
}
