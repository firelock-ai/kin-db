// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! A memory-mapped columnar segment for the entity graph.
//!
//! # Why
//!
//! A serving daemon's resident set is decided today by a decoded object graph.
//! `GraphSnapshot` is one MessagePack positional array decoded whole into owned
//! heap, and the same entity facts then live in three or four owned shapes at
//! once: the live `InMemoryGraph`, the reconciler's last-known-good copy, the
//! cross-file linker's per-entity `String`, and [`ReadIndex`]. `MmapReader`
//! maps a snapshot only to decode it and drops the mapping at the end of the
//! function, so nothing served from the graph is a slice of a mapping.
//!
//! A segment inverts that. Every read below is a slice of a `memmap2::Mmap`
//! plus a `from_le_bytes` on a fixed-width window, so a serving process holds
//! page-table entries rather than heap, and the page cache decides what is
//! resident.
//!
//! # Shape, and where the bytes went
//!
//! Measured on the two real stores this was designed against, by walking their
//! persisted MessagePack bodies field by field rather than by estimating from
//! struct widths:
//!
//! | store | entities | persisted per entity | of which `metadata` |
//! |---|---|---|---|
//! | VS Code `src/vs` subtree | 29,392 | 2,443.7 B | 1,937.2 B, 79.3% |
//! | Linux `fs`/`kernel`/`mm` subtree | 264,615 | 1,345.4 B | 864.4 B, 64.2% |
//!
//! `metadata` is the largest field of an entity on both, and every key in it is
//! derived retrieval text: an embedding body preview, an import context, a
//! surface context, three parse counters. No entity lookup, adjacency walk or
//! count reads any of it. That is why it is a side table here and not a column,
//! and it is the single largest reason the hot working set is a small fraction
//! of the store.
//!
//! Beside it, `file_origin` repeats one of 1,356 distinct paths across 29,392
//! entities on the first store and one of 5,818 across 264,615 on the second,
//! a 20.3x and 46.4x repeat, which is why paths are an ordinal into a table.
//!
//! # The three properties that make it a format rather than a cache
//!
//! **The version check is a range, and a bump inside it may only add columns.**
//! See [`format`]. [`ReadIndex::load`] refuses on `version != INDEX_VERSION`,
//! and that equality is what made every store on disk unopenable in the first
//! draft of kin-db#271 while 1,033 tests stayed green, because every fixture is
//! written by the binary under test.
//!
//! **Enum discriminants are an explicit wire contract, not `variant as u8`.**
//! `ReadIndex` persists `entity.kind as u8`, which pins the meaning of every
//! `.kidx` byte to the declaration order of `EntityKind`. Each mapping in
//! [`format`] is an exhaustive match with no wildcard arm, so adding a variant
//! to kin-model is a compile error here rather than a silent reinterpretation
//! of bytes already written.
//!
//! **The ordinal is the id rank.** Entities are written sorted by id, so
//! `id -> ordinal` is a binary search over the id column and needs no separate
//! index at all. For 16-byte fixed-width keys that is smaller and faster than
//! an FST, which is why this carries no new dependency.
//!
//! [`ReadIndex`]: crate::storage::index::ReadIndex
//! [`ReadIndex::load`]: crate::storage::index::ReadIndex::load

pub mod format;
pub mod reader;
pub mod writer;

#[cfg(test)]
mod tests;

pub use format::{
    SegmentShape, CURRENT_SEGMENT_VERSION, MAX_SUPPORTED_SEGMENT_VERSION,
    MIN_SUPPORTED_SEGMENT_VERSION, SEGMENT_MAGIC,
};
pub use reader::{OpenProfile, Ordinals, SegmentReader};
pub use writer::{write_segment, ColumnStat, SegmentWriteStats};
