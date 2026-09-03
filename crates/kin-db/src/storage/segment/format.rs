// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! On-disk contract for the mapped columnar entity-graph segment.
//!
//! A segment is a directory of immutable single-column files plus a manifest
//! naming every column, its width, its record count and its digest. Each file
//! carries its own magic, format version and SHA-256, so a torn write is
//! refused per column rather than mistaken for a short graph.
//!
//! Two properties are the reason the layout looks like this.
//!
//! **One column per file, not one file with a footer.** A reader maps only the
//! columns it reads. The cold fingerprint hashes and the metadata side table
//! never enter the address space of a process that answers entity lookups and
//! adjacency walks, so they cannot be read ahead into resident pages either.
//!
//! **The version check is a range, and a bump inside it may only ADD columns.**
//! [`ReadIndex::load`](crate::storage::index::ReadIndex::load) refuses on
//! `version != INDEX_VERSION`. That is the equality that made every store on
//! disk unopenable in the first draft of kin-db#271 while 1,033 tests stayed
//! green, because every fixture is written by the binary under test. Here the
//! reader accepts [`MIN_SUPPORTED_SEGMENT_VERSION`] through
//! [`MAX_SUPPORTED_SEGMENT_VERSION`] and refuses outside it by distinct named
//! errors. Reading one version forward is only sound while the newer version is
//! additive: a manifest entry whose column id this binary does not know is
//! skipped, and a column whose MEANING changes requires moving the floor rather
//! than adding to the ceiling.

use crate::error::KinDbError;
use kin_model::{EntityKind, EntityRole, LanguageId, RelationKind, RelationOrigin, Visibility};

/// Leading bytes of every file in a segment, manifest included.
pub const SEGMENT_MAGIC: [u8; 4] = *b"KSEG";

/// Oldest segment layout this binary can read.
pub const MIN_SUPPORTED_SEGMENT_VERSION: u32 = 1;

/// Layout this binary writes.
pub const CURRENT_SEGMENT_VERSION: u32 = 1;

/// Newest segment layout this binary will accept, one additive step ahead of
/// what it writes. See the module doc for what "additive" is allowed to mean.
pub const MAX_SUPPORTED_SEGMENT_VERSION: u32 = CURRENT_SEGMENT_VERSION + 1;

/// Bytes ahead of every column payload.
pub const HEADER_LEN: usize = 32;

/// Bytes of SHA-256 after every column payload.
pub const DIGEST_LEN: usize = 32;

/// File name of the manifest inside a segment directory.
pub const MANIFEST_FILE: &str = "manifest.kseg";

/// Bytes of one manifest entry: column id, width, count, payload length, digest.
pub const MANIFEST_ENTRY_LEN: usize = 4 + 4 + 8 + 8 + 32;

/// Bytes ahead of the manifest's entry table.
pub const MANIFEST_PREAMBLE_LEN: usize = 40;

/// Identity of one column. The numbers are the persisted contract and are
/// never reused for a different meaning; a retired column keeps its number.
pub mod column {
    /// The manifest itself, so its header is shaped like every other file.
    pub const MANIFEST: u32 = 0;

    // Hot entity columns, one entry per entity ordinal.
    pub const ENTITY_ID: u32 = 1;
    pub const ENTITY_KIND: u32 = 2;
    pub const ENTITY_LANGUAGE: u32 = 3;
    pub const ENTITY_FLAGS: u32 = 4;
    pub const ENTITY_AST_HASH: u32 = 5;
    pub const ENTITY_PATH_ORD: u32 = 6;
    pub const ENTITY_SPAN: u32 = 7;
    /// The path ordinal of `span.file`, which is a separate `Option` from
    /// `file_origin` and is NOT assumed to be the same path. Four bytes buys
    /// losslessness without an equality assertion over data already on disk.
    pub const ENTITY_SPAN_PATH_ORD: u32 = 31;
    pub const ENTITY_NAME_OFF: u32 = 8;
    pub const ENTITY_NAME_ARENA: u32 = 9;
    pub const ENTITY_SIG_OFF: u32 = 10;
    pub const ENTITY_SIG_ARENA: u32 = 11;
    pub const ENTITY_DOC_OFF: u32 = 12;
    pub const ENTITY_DOC_ARENA: u32 = 13;
    pub const PATH_OFF: u32 = 14;
    pub const PATH_ARENA: u32 = 15;
    pub const NAME_KEY_OFF: u32 = 16;
    pub const NAME_KEY_ARENA: u32 = 17;
    pub const NAME_POSTING_OFF: u32 = 18;
    pub const NAME_POSTINGS: u32 = 19;

    // Hot relation columns and the CSR adjacency over entity ordinals.
    pub const OUT_OFF: u32 = 20;
    pub const OUT_DST: u32 = 21;
    pub const IN_OFF: u32 = 22;
    pub const IN_SRC: u32 = 23;
    pub const IN_REL_ORD: u32 = 24;
    pub const REL_ID: u32 = 25;
    pub const REL_KIND: u32 = 26;
    pub const REL_CONFIDENCE: u32 = 27;
    pub const REL_ORIGIN: u32 = 28;
    pub const REL_SRC: u32 = 29;
    pub const REL_FLAGS: u32 = 30;

    // Cold entity columns. Reachable, never faulted by a hot read.
    pub const ENTITY_COLD_FLAGS: u32 = 40;
    pub const ENTITY_SIGNATURE_HASH: u32 = 41;
    pub const ENTITY_BEHAVIOR_HASH: u32 = 42;
    pub const ENTITY_EQUIVALENCE_HASH: u32 = 43;
    pub const ENTITY_STABILITY: u32 = 44;
    pub const ENTITY_LINEAGE_PARENT: u32 = 45;
    pub const ENTITY_CREATED_IN: u32 = 46;
    pub const ENTITY_SUPERSEDED_BY: u32 = 47;

    // Cold relation columns.
    pub const REL_CREATED_IN: u32 = 50;
    pub const REL_IMPORT_OFF: u32 = 51;
    pub const REL_IMPORT_ARENA: u32 = 52;

    // Side tables. Offsets are u64 because these arenas outgrow u32 at the
    // scale this design targets: metadata alone projects to 7.7 GB on a
    // 4M-entity graph.
    pub const ENTITY_METADATA_OFF: u32 = 60;
    pub const ENTITY_METADATA_ARENA: u32 = 61;
    pub const REL_ENDPOINTS_OFF: u32 = 62;
    pub const REL_ENDPOINTS_ARENA: u32 = 63;
    pub const REL_EVIDENCE_OFF: u32 = 64;
    pub const REL_EVIDENCE_ARENA: u32 = 65;

    // Entity revisions, addressed by revision ordinal.
    //
    // These are ADDITIVE: a reader that does not know them skips their manifest
    // rows and counts them in `unknown_columns`, and a hot open still answers
    // every entity query. That is why adding them does not move
    // `CURRENT_SEGMENT_VERSION`, and why the revision counts live in their
    // columns' own manifest rows rather than in the manifest preamble, which
    // could not grow without reinterpreting bytes a v1 reader already parses.
    pub const REV_ID: u32 = 70;
    pub const REV_ENTITY_ORD: u32 = 71;
    pub const REV_INTRODUCED_ORD: u32 = 72;
    pub const REV_FLAGS: u32 = 73;
    /// The previous revision's raw 32-byte id, not an ordinal.
    ///
    /// A previous-revision reference can point at a revision outside this
    /// segment, from a retired generation, and an ordinal cannot represent
    /// that. Deriving it instead is unsound: the id IS
    /// `sha256(entity_id || introduced_by)` for a revision built by
    /// `EntityRevision::new`, but a caller can construct the struct with any
    /// id, so a reader that recomputed one would silently substitute a
    /// different revision. Thirty-two raw bytes in a COLD column, measured
    /// absent on all 294,007 revisions across both real stores, is cheaper than
    /// a verify-then-compress path plus the side table its failure case needs.
    pub const REV_PREVIOUS_ID: u32 = 74;
    pub const REV_ENDED_ORD: u32 = 75;
    /// Distinct `SemanticChangeId` values, sorted. `introduced_by` and
    /// `ended_by` are u32 ordinals into this table rather than 32-byte hashes,
    /// because the distinct count is bounded by the commit count and measured
    /// ONE on both real stores against 264,615 and 29,392 revisions.
    pub const CHANGE_IDS: u32 = 76;
    /// Side table by revision ordinal, carrying the revision's whole `Entity`
    /// only when it differs from the head. Measured EMPTY on both real stores:
    /// all 294,007 revisions are byte-identical to their head entity.
    pub const REV_DELTA_OFF: u32 = 77;
    pub const REV_DELTA_ARENA: u32 = 78;
}

/// File name a column is stored under inside the segment directory.
pub fn column_file_name(id: u32) -> String {
    format!("c{id:03}.kseg")
}

// ---------------------------------------------------------------------------
// Entity flag bits
// ---------------------------------------------------------------------------

/// Visibility occupies the low two bits of the hot flags byte.
pub const FLAG_VISIBILITY_MASK: u8 = 0b0000_0011;
/// Role occupies the next three.
pub const FLAG_ROLE_SHIFT: u32 = 2;
/// Mask for the role field once shifted down.
pub const FLAG_ROLE_MASK: u8 = 0b0000_0111;
/// The entity carries a `span`.
pub const FLAG_HAS_SPAN: u8 = 0b0010_0000;
/// The entity carries a `doc_summary`.
pub const FLAG_HAS_DOC: u8 = 0b0100_0000;
/// The entity carries a `file_origin`.
pub const FLAG_HAS_PATH: u8 = 0b1000_0000;

/// The entity carries a `lineage_parent`.
pub const COLD_HAS_LINEAGE: u8 = 0b0000_0001;
/// The entity carries a `created_in`.
pub const COLD_HAS_CREATED_IN: u8 = 0b0000_0010;
/// The entity carries a `superseded_by`.
pub const COLD_HAS_SUPERSEDED: u8 = 0b0000_0100;
/// The entity carries non-empty `metadata`.
pub const COLD_HAS_METADATA: u8 = 0b0000_1000;

/// The relation carries a `created_in`.
pub const REL_HAS_CREATED_IN: u8 = 0b0000_0001;
/// The relation carries an `import_source`.
pub const REL_HAS_IMPORT_SOURCE: u8 = 0b0000_0010;
/// The relation carries at least one evidence record.
pub const REL_HAS_EVIDENCE: u8 = 0b0000_0100;
/// Both endpoints are entities, so `REL_SRC` and the CSR describe them and the
/// endpoint side table holds nothing for this ordinal.
pub const REL_ENTITY_ENDPOINTS: u8 = 0b0000_1000;

/// The revision carries a `previous_revision`.
pub const REV_HAS_PREVIOUS: u8 = 0b0000_0001;
/// The revision carries an `ended_by`.
pub const REV_HAS_ENDED: u8 = 0b0000_0010;
/// The revision's `entity` is NOT byte-identical to the head entity, so the
/// delta side table carries it. When this bit is CLEAR the head entity's own
/// columns answer every field and no second entity exists to decode.
///
/// A one-bit answer rather than a content hash on purpose: the comparison is
/// made once at write time, and a hash would add nothing the bit does not
/// already say. Detecting corruption is `verify_all`'s job, not this column's.
pub const REV_ENTITY_DIFFERS: u8 = 0b0000_0100;

// ---------------------------------------------------------------------------
// Wire codes for the enums the columns carry
// ---------------------------------------------------------------------------
//
// These are deliberately NOT `variant as u8`. `ReadIndex` stores `kind as u8`,
// which pins the persisted meaning of every `.kidx` byte to the DECLARATION
// ORDER of `EntityKind`, so inserting a variant silently reinterprets every
// store already on disk. Each match below is exhaustive with no wildcard arm,
// so adding a variant to kin-model is a compile error here rather than a silent
// reinterpretation there.

/// Wire code for an entity kind.
pub fn entity_kind_code(kind: EntityKind) -> u8 {
    match kind {
        EntityKind::Function => 0,
        EntityKind::Class => 1,
        EntityKind::Interface => 2,
        EntityKind::TraitDef => 3,
        EntityKind::TypeAlias => 4,
        EntityKind::Module => 5,
        EntityKind::Package => 6,
        EntityKind::Test => 7,
        EntityKind::Schema => 8,
        EntityKind::ApiEndpoint => 9,
        EntityKind::EventContract => 10,
        EntityKind::File => 11,
        EntityKind::DocumentNode => 12,
        EntityKind::Method => 13,
        EntityKind::EnumDef => 14,
        EntityKind::EnumVariant => 15,
        EntityKind::Constant => 16,
        EntityKind::StaticVar => 17,
        EntityKind::Macro => 18,
    }
}

/// Entity kind for a wire code, or a named refusal.
pub fn entity_kind_of_code(code: u8) -> Result<EntityKind, KinDbError> {
    Ok(match code {
        0 => EntityKind::Function,
        1 => EntityKind::Class,
        2 => EntityKind::Interface,
        3 => EntityKind::TraitDef,
        4 => EntityKind::TypeAlias,
        5 => EntityKind::Module,
        6 => EntityKind::Package,
        7 => EntityKind::Test,
        8 => EntityKind::Schema,
        9 => EntityKind::ApiEndpoint,
        10 => EntityKind::EventContract,
        11 => EntityKind::File,
        12 => EntityKind::DocumentNode,
        13 => EntityKind::Method,
        14 => EntityKind::EnumDef,
        15 => EntityKind::EnumVariant,
        16 => EntityKind::Constant,
        17 => EntityKind::StaticVar,
        18 => EntityKind::Macro,
        other => return Err(unknown_code("entity kind", u32::from(other))),
    })
}

/// Wire code for a language.
pub fn language_code(language: LanguageId) -> u8 {
    match language {
        LanguageId::TypeScript => 0,
        LanguageId::JavaScript => 1,
        LanguageId::Python => 2,
        LanguageId::Go => 3,
        LanguageId::Java => 4,
        LanguageId::Rust => 5,
        LanguageId::C => 6,
        LanguageId::Cpp => 7,
        LanguageId::CSharp => 8,
        LanguageId::Ruby => 9,
        LanguageId::Php => 10,
        LanguageId::Swift => 11,
        LanguageId::Kotlin => 12,
        LanguageId::Hcl => 13,
    }
}

/// Language for a wire code, or a named refusal.
pub fn language_of_code(code: u8) -> Result<LanguageId, KinDbError> {
    Ok(match code {
        0 => LanguageId::TypeScript,
        1 => LanguageId::JavaScript,
        2 => LanguageId::Python,
        3 => LanguageId::Go,
        4 => LanguageId::Java,
        5 => LanguageId::Rust,
        6 => LanguageId::C,
        7 => LanguageId::Cpp,
        8 => LanguageId::CSharp,
        9 => LanguageId::Ruby,
        10 => LanguageId::Php,
        11 => LanguageId::Swift,
        12 => LanguageId::Kotlin,
        13 => LanguageId::Hcl,
        other => return Err(unknown_code("language", u32::from(other))),
    })
}

/// Wire code for a visibility, two bits wide.
pub fn visibility_code(visibility: Visibility) -> u8 {
    match visibility {
        Visibility::Public => 0,
        Visibility::Private => 1,
        Visibility::Internal => 2,
        Visibility::Crate => 3,
    }
}

/// Visibility for a wire code, or a named refusal.
pub fn visibility_of_code(code: u8) -> Result<Visibility, KinDbError> {
    Ok(match code {
        0 => Visibility::Public,
        1 => Visibility::Private,
        2 => Visibility::Internal,
        3 => Visibility::Crate,
        other => return Err(unknown_code("visibility", u32::from(other))),
    })
}

/// Wire code for an entity role, three bits wide.
pub fn role_code(role: EntityRole) -> u8 {
    match role {
        EntityRole::Source => 0,
        EntityRole::Test => 1,
        EntityRole::External => 2,
        EntityRole::Docs => 3,
        EntityRole::Generated => 4,
        EntityRole::Vendored => 5,
    }
}

/// Entity role for a wire code, or a named refusal.
pub fn role_of_code(code: u8) -> Result<EntityRole, KinDbError> {
    Ok(match code {
        0 => EntityRole::Source,
        1 => EntityRole::Test,
        2 => EntityRole::External,
        3 => EntityRole::Docs,
        4 => EntityRole::Generated,
        5 => EntityRole::Vendored,
        other => return Err(unknown_code("entity role", u32::from(other))),
    })
}

/// Wire code for a relation kind.
pub fn relation_kind_code(kind: RelationKind) -> u8 {
    match kind {
        RelationKind::Contains => 0,
        RelationKind::Extends => 1,
        RelationKind::Implements => 2,
        RelationKind::Overrides => 3,
        RelationKind::Calls => 4,
        RelationKind::Instantiates => 5,
        RelationKind::References => 6,
        RelationKind::UsesMacro => 7,
        RelationKind::UsesType => 8,
        RelationKind::Imports => 9,
        RelationKind::Includes => 10,
        RelationKind::DependsOn => 11,
        RelationKind::EmitsEvent => 12,
        RelationKind::SubscribesTo => 13,
        RelationKind::DefinesContract => 14,
        RelationKind::ConsumesContract => 15,
        RelationKind::SendsMessage => 16,
        RelationKind::Spawns => 17,
        RelationKind::Tests => 18,
        RelationKind::Covers => 19,
        RelationKind::CoChanges => 20,
        RelationKind::DerivedFrom => 21,
        RelationKind::DocumentedBy => 22,
        RelationKind::OwnedBy => 23,
        RelationKind::OwnedByFile => 24,
    }
}

/// Relation kind for a wire code, or a named refusal.
pub fn relation_kind_of_code(code: u8) -> Result<RelationKind, KinDbError> {
    Ok(match code {
        0 => RelationKind::Contains,
        1 => RelationKind::Extends,
        2 => RelationKind::Implements,
        3 => RelationKind::Overrides,
        4 => RelationKind::Calls,
        5 => RelationKind::Instantiates,
        6 => RelationKind::References,
        7 => RelationKind::UsesMacro,
        8 => RelationKind::UsesType,
        9 => RelationKind::Imports,
        10 => RelationKind::Includes,
        11 => RelationKind::DependsOn,
        12 => RelationKind::EmitsEvent,
        13 => RelationKind::SubscribesTo,
        14 => RelationKind::DefinesContract,
        15 => RelationKind::ConsumesContract,
        16 => RelationKind::SendsMessage,
        17 => RelationKind::Spawns,
        18 => RelationKind::Tests,
        19 => RelationKind::Covers,
        20 => RelationKind::CoChanges,
        21 => RelationKind::DerivedFrom,
        22 => RelationKind::DocumentedBy,
        23 => RelationKind::OwnedBy,
        24 => RelationKind::OwnedByFile,
        other => return Err(unknown_code("relation kind", u32::from(other))),
    })
}

/// Wire code for a relation origin.
pub fn relation_origin_code(origin: RelationOrigin) -> u8 {
    match origin {
        RelationOrigin::Parsed => 0,
        RelationOrigin::Inferred => 1,
        RelationOrigin::Manual => 2,
        RelationOrigin::Lsp => 3,
    }
}

/// Relation origin for a wire code, or a named refusal.
pub fn relation_origin_of_code(code: u8) -> Result<RelationOrigin, KinDbError> {
    Ok(match code {
        0 => RelationOrigin::Parsed,
        1 => RelationOrigin::Inferred,
        2 => RelationOrigin::Manual,
        3 => RelationOrigin::Lsp,
        other => return Err(unknown_code("relation origin", u32::from(other))),
    })
}

fn unknown_code(what: &str, code: u32) -> KinDbError {
    KinDbError::StorageError(format!(
        "segment carries an unknown {what} wire code {code}; the segment was written by a newer \
         binary than this one, or a column is corrupt"
    ))
}

/// One row of the manifest's column table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColumnRecord {
    /// Which column this is. See [`column`].
    pub id: u32,
    /// Bytes per record, or 1 for a byte arena.
    pub width: u32,
    /// Records in the column.
    pub count: u64,
    /// Bytes of payload, which is `width * count` for a fixed-width column.
    pub payload_len: u64,
    /// SHA-256 over the column file's header and payload.
    pub digest: [u8; 32],
}

/// Everything the manifest states about a segment beyond its column table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SegmentShape {
    /// Entities in the segment, which is also the ordinal space.
    pub entity_count: u64,
    /// Relations in the segment, entity-endpoint ones first.
    pub relation_count: u64,
    /// Relations whose source and destination are both entities. These occupy
    /// ordinals `0..entity_edge_count` and are the ones the CSR describes.
    pub entity_edge_count: u64,
    /// Distinct file paths in the path table.
    pub path_count: u64,
    /// Distinct lowercased names in the name index.
    pub name_key_count: u64,
}

/// Assemble the 32-byte header every file in a segment leads with.
pub fn encode_header(id: u32, width: u32, count: u64, payload_len: u64) -> [u8; HEADER_LEN] {
    let mut header = [0u8; HEADER_LEN];
    header[0..4].copy_from_slice(&SEGMENT_MAGIC);
    header[4..8].copy_from_slice(&CURRENT_SEGMENT_VERSION.to_le_bytes());
    header[8..12].copy_from_slice(&id.to_le_bytes());
    header[12..16].copy_from_slice(&width.to_le_bytes());
    header[16..24].copy_from_slice(&count.to_le_bytes());
    header[24..32].copy_from_slice(&payload_len.to_le_bytes());
    header
}

/// What a file's header says about itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileHeader {
    /// Layout version the writer stamped.
    pub version: u32,
    /// Column id, 0 for the manifest.
    pub id: u32,
    /// Bytes per record.
    pub width: u32,
    /// Records.
    pub count: u64,
    /// Payload bytes.
    pub payload_len: u64,
}

/// Read and check a file header, refusing the magic and the version by name.
pub fn decode_header(bytes: &[u8], role: &str) -> Result<FileHeader, KinDbError> {
    if bytes.len() < HEADER_LEN + DIGEST_LEN {
        return Err(KinDbError::StorageError(format!(
            "segment {role} is {} bytes, shorter than the {} a header and digest need",
            bytes.len(),
            HEADER_LEN + DIGEST_LEN
        )));
    }
    if bytes[0..4] != SEGMENT_MAGIC {
        return Err(KinDbError::StorageError(format!(
            "segment {role} does not lead with KSEG"
        )));
    }
    let version = read_u32(bytes, 4)?;
    if version < MIN_SUPPORTED_SEGMENT_VERSION {
        return Err(KinDbError::StorageError(format!(
            "segment {role} declares layout version {version}, below the floor \
             {MIN_SUPPORTED_SEGMENT_VERSION} this binary reads; rebuild the segment"
        )));
    }
    if version > MAX_SUPPORTED_SEGMENT_VERSION {
        return Err(KinDbError::StorageError(format!(
            "segment {role} declares layout version {version}, above the ceiling \
             {MAX_SUPPORTED_SEGMENT_VERSION} this binary reads; upgrade the binary"
        )));
    }
    let id = read_u32(bytes, 8)?;
    let width = read_u32(bytes, 12)?;
    let count = read_u64(bytes, 16)?;
    let payload_len = read_u64(bytes, 24)?;
    let expected = HEADER_LEN as u64 + payload_len + DIGEST_LEN as u64;
    if bytes.len() as u64 != expected {
        return Err(KinDbError::StorageError(format!(
            "segment {role} is {} bytes, and its header declares {expected}",
            bytes.len()
        )));
    }
    Ok(FileHeader {
        version,
        id,
        width,
        count,
        payload_len,
    })
}

/// Read a little-endian u32 at `offset`, refusing a short slice by name.
pub fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, KinDbError> {
    let end = offset
        .checked_add(4)
        .ok_or_else(|| short_read("u32", offset, bytes.len()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| short_read("u32", offset, bytes.len()))?;
    let mut buf = [0u8; 4];
    buf.copy_from_slice(slice);
    Ok(u32::from_le_bytes(buf))
}

/// Read a little-endian u64 at `offset`, refusing a short slice by name.
pub fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, KinDbError> {
    let end = offset
        .checked_add(8)
        .ok_or_else(|| short_read("u64", offset, bytes.len()))?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| short_read("u64", offset, bytes.len()))?;
    let mut buf = [0u8; 8];
    buf.copy_from_slice(slice);
    Ok(u64::from_le_bytes(buf))
}

fn short_read(what: &str, offset: usize, len: usize) -> KinDbError {
    KinDbError::StorageError(format!(
        "segment read of a {what} at offset {offset} runs past the {len} bytes present"
    ))
}
