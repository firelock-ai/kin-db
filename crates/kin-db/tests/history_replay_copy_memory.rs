// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Whether reopening a store without a validation record copies the history it
//! replays.
//!
//! An open with no reopen proof runs whole-history replay. It used to hand the
//! replay an owned `Vec` of every `SemanticChange` in the repository, built with
//! `snapshot.changes.values().cloned().collect()`, so a second whole history
//! stood on the heap for the length of the replay. On a converted store the
//! change map is most of the body, 1.32 GiB decoded on a full VS Code tree.
//!
//! ## Why this compares two arms instead of asserting a ceiling
//!
//! A ratio ceiling has to be calibrated against a run, and there is no honest
//! way to pick one for a path that also builds a whole replay graph. So this
//! measures the same open twice over stores that differ in exactly one thing,
//! the number of bytes each change carries in its own message, and prices the
//! difference. A clone of the history carries those bytes. Nothing else on the
//! path reads a change's message: the topological pass reads parents, the
//! replay proof resolves entity deltas, and the first-parent walk reads ids.
//!
//! The payload lives in the message and not in the entity for that reason. An
//! entity heavy enough to matter would put the resolved graph and the derived
//! revisions on the same scale as the copy, and the difference between the arms
//! would stop being about the copy at all.
//!
//! Live heap rather than resident set, because resident set keeps counting
//! memory the allocator has freed and not returned to the operating system, so
//! it moves with the platform rather than with the code.
//!
//! This file is its own test binary and holds one test on purpose. The counters
//! below are process global, so a second test running beside this one would be
//! measured into it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use kin_db::{LocalFileBackend, RepositoryAuthorityManager, VersionedAuthorityState};
use kin_model::{
    compute_semantic_change_id, AdmissionPolicyDelta, AuthorId, ChangeOrigin, Entity, EntityDelta,
    EntityId, EntityKind, EntityMetadata, EntityRole, FilePathId, FingerprintAlgorithm, Hash256,
    LanguageId, OperationId, RepositoryId, RepositoryTransaction, SemanticChange, SemanticChangeId,
    SemanticFingerprint, SharedAdmissionPolicy, Timestamp, Visibility,
    REPOSITORY_TRANSACTION_SCHEMA_VERSION,
};
use uuid::Uuid;

// --- the instrument -------------------------------------------------------

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

fn record_allocation(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let pointer = unsafe { System.alloc_zeroed(layout) };
        if !pointer.is_null() {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let moved = unsafe { System.realloc(pointer, layout, new_size) };
        if !moved.is_null() {
            if new_size >= layout.size() {
                record_allocation(new_size - layout.size());
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Arm the peak at the current live heap and return that floor.
fn arm_peak() -> usize {
    let floor = LIVE.load(Ordering::Relaxed);
    PEAK.store(floor, Ordering::Relaxed);
    floor
}

fn peak_growth_since(floor: usize) -> usize {
    PEAK.load(Ordering::Relaxed).saturating_sub(floor)
}

// --- the fixture ----------------------------------------------------------

/// Enough history that a copy of it is worth measuring, and few enough that a
/// whole-history replay finishes inside an ordinary test run.
const COMMITS: usize = 300;

/// The one variable between the two arms.
const SMALL_MESSAGE_BYTES: usize = 64;
const LARGE_MESSAGE_BYTES: usize = 16_384;

fn fixed_timestamp() -> Timestamp {
    Timestamp(
        chrono::DateTime::parse_from_rfc3339("2026-09-03T00:00:00Z")
            .expect("fixed timestamp parses")
            .with_timezone(&chrono::Utc),
    )
}

/// One small distinct entity per commit. Deliberately small: the payload is in
/// the change's message, and an entity heavy enough to rival it would put the
/// resolved graph and the derived revisions on the same scale as the copy this
/// guard prices.
fn measurement_entity(index: usize) -> Entity {
    let path = format!("src/module_{index}.rs");
    let name = format!("kin_{index}");
    let byte = (index % 251) as u8;
    Entity {
        id: EntityId::from_content(&path, &name, "function", 1),
        kind: EntityKind::Function,
        name: name.clone(),
        language: LanguageId::Rust,
        fingerprint: SemanticFingerprint {
            algorithm: FingerprintAlgorithm::V1TreeSitter,
            ast_hash: Hash256::from_bytes([byte; 32]),
            signature_hash: Hash256::from_bytes([byte.wrapping_add(1); 32]),
            behavior_hash: Hash256::from_bytes([byte.wrapping_add(2); 32]),
            equivalence_hash: Hash256::from_bytes([byte.wrapping_add(3); 32]),
            stability_score: 1.0,
        },
        file_origin: Some(FilePathId::new(&path)),
        span: None,
        signature: format!("fn {name}()"),
        visibility: Visibility::Public,
        role: EntityRole::Source,
        doc_summary: None,
        metadata: EntityMetadata::default(),
        lineage_parent: None,
        created_in: None,
        superseded_by: None,
    }
}

fn history_chain(
    commits: usize,
    message_bytes: usize,
    shared: &SharedAdmissionPolicy,
) -> Vec<SemanticChange> {
    let mut chain: Vec<SemanticChange> = Vec::with_capacity(commits);
    let mut parent: Option<SemanticChangeId> = None;
    for index in 0..commits {
        let mut message = format!("synthetic converted commit {index} ");
        while message.len() < message_bytes {
            message.push('m');
        }
        let mut change = SemanticChange {
            id: SemanticChangeId::from_hash(Hash256::from_bytes([0; 32])),
            origin: ChangeOrigin::Native,
            parents: parent.into_iter().collect(),
            timestamp: fixed_timestamp(),
            author: AuthorId::new("memphase0-measurement"),
            message,
            entity_deltas: vec![EntityDelta::Added {
                new: measurement_entity(index),
            }],
            relation_deltas: Vec::new(),
            tree_deltas: Vec::new(),
            admission_policy_delta: (index == 0)
                .then(|| AdmissionPolicyDelta::initialize(shared.clone())),
            external_reference_deltas: Vec::new(),
            projected_files: Vec::new(),
            spec_link: None,
            evidence: Vec::new(),
            risk_summary: None,
        };
        change.id = compute_semantic_change_id(&change).expect("change id computes");
        parent = Some(change.id);
        chain.push(change);
    }
    chain
}

/// Publish one whole-history bootstrap into a fresh local store.
fn publish_bootstrap(directory: &std::path::Path, repository: &RepositoryId, message_bytes: usize) {
    let backend = Arc::new(LocalFileBackend::new(directory));
    let manager = RepositoryAuthorityManager::open(repository.clone(), backend)
        .expect("open fresh authority");
    let shared = SharedAdmissionPolicy::empty(0);
    let changes = history_chain(COMMITS, message_bytes, &shared);
    let lease = manager.read_authority();
    let transaction = RepositoryTransaction {
        schema_version: REPOSITORY_TRANSACTION_SCHEMA_VERSION,
        operation_id: OperationId::from_uuid(Uuid::from_u128(1)),
        repository_id: repository.clone(),
        expected_generation: lease.generation(),
        expected_roots: lease.roots().clone(),
        actor: AuthorId::new("memphase0-measurement"),
        reason: "synthetic whole-history bootstrap".to_string(),
        external_objects: Vec::new(),
        git_authority_delta: None,
        changes,
        aliases: Vec::new(),
        ref_mutations: Vec::new(),
        default_ref_mutation: None,
        workspace_mutation: None,
        local_overlay_delta: None,
        merge_transaction_delta: None,
        sealed_observation: None,
    };
    drop(lease);
    let receipt = manager
        .commit_repository_transaction(transaction)
        .expect("whole-history bootstrap commits");
    assert_eq!(receipt.generation, 1, "the bootstrap publishes generation 1");
}

/// Break the durable validation record so the next open cannot reuse it and has
/// to replay the whole history. The record names the exact snapshot bytes, so
/// naming different bytes retires it without touching the store itself.
fn retire_the_validation_record(directory: &std::path::Path, repository: &RepositoryId) {
    let path = directory.join(repository.as_str()).join("authority.json");
    let mut record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("the store carries an authority.json"))
            .expect("authority.json is json");
    assert!(
        !record["history_validation"].is_null(),
        "a committed store must carry a history validation record, or this test retires nothing"
    );
    record["history_validation"]["snapshot_sha256"] =
        serde_json::json!("0000000000000000000000000000000000000000000000000000000000000000");
    std::fs::write(&path, serde_json::to_vec(&record).expect("record serializes"))
        .expect("authority.json is writable");
}

/// Build a store whose changes each carry `message_bytes`, retire its
/// validation record, and return the transient bytes its reopen held on top of
/// what the reopen kept, beside the size of the store on disk.
fn transient_bytes_of_a_reopen_without_a_proof(message_bytes: usize) -> (usize, usize) {
    let directory = tempfile::tempdir().expect("scratch store");
    let repository = RepositoryId::new("memphase0-measurement").expect("repository id");
    publish_bootstrap(directory.path(), &repository, message_bytes);
    retire_the_validation_record(directory.path(), &repository);

    // A fresh backend, so the measured open recovers from the file rather than
    // from anything the publishing manager still held.
    let backend = Arc::new(LocalFileBackend::new(directory.path()));

    let floor = arm_peak();
    let (manager, payload) =
        RepositoryAuthorityManager::open_with_payload_stats(repository.clone(), backend)
            .expect("the published authority reopens");
    let growth = peak_growth_since(floor);
    let retained = live_bytes().saturating_sub(floor);

    // Controls. Both of these fail loudly rather than quietly making the
    // measurement about nothing.
    assert!(
        !manager.opened_by_history_validation(),
        "the retired record must send this open down the whole-history replay, which is the \
         path being priced"
    );
    assert_eq!(
        manager.read_authority().generation(),
        1,
        "the reopened authority must carry the published generation, or this measured an open \
         that did not happen"
    );

    let snapshot_bytes = payload
        .expect("a persisted authority reports its payload")
        .snapshot_bytes() as usize;

    drop(manager);
    (growth.saturating_sub(retained), snapshot_bytes)
}

// --- the guard ------------------------------------------------------------

#[test]
fn a_reopen_replay_does_not_copy_the_history_it_replays() {
    // Positive control on the instrument. A counter that never moves makes
    // every assertion below true for the wrong reason.
    let control_floor = live_bytes();
    let ballast = vec![0u8; 8 << 20];
    assert!(
        live_bytes() - control_floor >= ballast.len(),
        "the counting allocator did not observe an {} byte allocation, so it cannot observe an \
         open either",
        ballast.len()
    );
    drop(ballast);

    let (small_transient, small_store) =
        transient_bytes_of_a_reopen_without_a_proof(SMALL_MESSAGE_BYTES);
    let (large_transient, large_store) =
        transient_bytes_of_a_reopen_without_a_proof(LARGE_MESSAGE_BYTES);

    // Control on the fixture: the arms must actually differ on disk, or the
    // comparison below is between two identical stores.
    let message_difference = COMMITS * (LARGE_MESSAGE_BYTES - SMALL_MESSAGE_BYTES);
    assert!(
        large_store > small_store + message_difference / 2,
        "the two arms must differ by roughly the message bytes they carry: small store \
         {small_store} bytes, large store {large_store} bytes, messages differ by \
         {message_difference}"
    );

    let transient_difference = large_transient.saturating_sub(small_transient);
    println!(
        "small arm: store {small_store} bytes, transient {small_transient}\n\
         large arm: store {large_store} bytes, transient {large_transient}\n\
         messages differ by {message_difference} bytes, transients by {transient_difference}"
    );

    // A copy of the history carries every message, so before the change the
    // difference is at least one whole copy of them. A replay driven from ids
    // carries none, and what is left is the decoder's own slack on a bigger
    // body. Half is the split between those two, and it is deliberately loose:
    // the first hosted run prints both arms, and this constant is set from that
    // reading rather than from taste.
    let ceiling = message_difference / 2;
    assert!(
        transient_difference < ceiling,
        "reopening a store whose changes carry {} more bytes of message held {} more transient \
         bytes, over the {} byte ceiling; the replay is holding a copy of the history",
        message_difference,
        transient_difference,
        ceiling
    );
}
