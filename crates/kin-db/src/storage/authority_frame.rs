// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Authority frames: the O(change) persistence unit of repository authority.
//!
//! A repository-authority successor differs from its predecessor in exactly
//! the ways `prepare_successor` mutates it: new immutable changes, one appended
//! operation record and its receipt, and envelope collections that are
//! appended to, upserted, or replaced. An [`AuthorityFrame`] carries that
//! mutation as the writer already accumulated it, so it is drained from the
//! successor rather than computed by diffing two snapshots, and it carries the
//! results of the mutation (the successor workspace state, the successor ref
//! state) rather than the transaction, so recovery applies it without paying
//! for base-graph materialization again.
//!
//! Wire format, identical in layout to the KNDD graph delta so every reader
//! that does not know frames refuses at the first four bytes:
//!   [4B magic "KNAF"] [4B version LE] [8B body_len LE] [body ...] [32B SHA-256]
//!
//! The body is a MessagePack-serialized [`AuthorityFrame`]. The struct is
//! encoded positionally, so its fields are only ever appended, and any change
//! to what a field means bumps [`AuthorityFrame::CURRENT_VERSION`].

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use kin_model::{
    ExternalChangeAlias, ExternalObjectRecord, FrozenLocalOverlay, GitExternalAuthority,
    MergeTransactionRecord, RepositoryCommitOutcome, RepositoryCommitReceipt, RepositoryId,
    RepositoryOperationRecord, RepositoryRefState, SemanticChange, WorkspaceState,
};

use crate::error::KinDbError;
use crate::storage::backend::Generation;
use crate::storage::format::GraphSnapshot;
use crate::storage::repository::{
    derive_change_children, ChangeAdmissionPolicy, PersistedRepositoryAuthority,
};

/// One acknowledged successor of a repository-authority state, as a patch over
/// the state it extends.
///
/// Every sequence with set semantics is carried in canonical sorted order so
/// that encoding the same successor twice yields identical bytes, which is what
/// makes an exact retry of a frame append idempotent at the backend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthorityFrame {
    /// Envelope schema of the base and the successor.
    pub schema_version: u32,
    pub repository_id: RepositoryId,
    /// The one operation this successor appended. Its `roots_before` names
    /// the exact predecessor state and its `roots_after` names the successor.
    pub operation: RepositoryOperationRecord,
    /// Successor changes absent from the base, sorted by identity.
    pub changes: Vec<SemanticChange>,
    /// Admission policies for exactly the changes above, sorted by change id.
    pub admission_policies: Vec<ChangeAdmissionPolicy>,
    /// External object records absent from the base, sorted by object id.
    pub external_objects: Vec<ExternalObjectRecord>,
    /// External change aliases absent from the base, sorted by object id.
    pub aliases: Vec<ExternalChangeAlias>,
    /// `Some(replacement)` only when the successor moved the Git authority.
    pub git_external_authority: Option<Option<GitExternalAuthority>>,
    /// The successor's complete ref state; refs move and disappear.
    pub ref_state: RepositoryRefState,
    /// Successor workspaces that differ from the base, sorted by workspace id.
    pub workspaces: Vec<WorkspaceState>,
    /// The successor's complete local overlay list.
    pub local_overlays: Vec<FrozenLocalOverlay>,
    /// The successor's complete merge record list; records are removed too.
    pub merge_transactions: Vec<MergeTransactionRecord>,
}

impl AuthorityFrame {
    /// Magic bytes for the frame file header: "KNAF".
    pub const MAGIC: [u8; 4] = *b"KNAF";

    /// Current frame format version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Size of the SHA-256 checksum appended to the wire format.
    pub const CHECKSUM_LEN: usize = 32;

    const HEADER_LEN: usize = 16;

    /// Logical generation of the successor this frame produces.
    pub fn generation(&self) -> Generation {
        self.operation.roots_after.generation
    }

    /// Logical generation of the state this frame extends.
    pub fn base_generation(&self) -> Generation {
        self.operation.roots_before.generation
    }

    /// Whether `data` starts like a frame. Cheap enough to classify a journal
    /// entry before deciding which decoder owns it.
    pub fn is_frame_bytes(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == Self::MAGIC
    }

    /// Serialize the frame with header and SHA-256 checksum.
    pub fn to_bytes(&self) -> Result<Vec<u8>, KinDbError> {
        let body = rmp_serde::to_vec(self).map_err(|error| {
            KinDbError::StorageError(format!("authority frame serialization failed: {error}"))
        })?;
        let mut buf = Vec::with_capacity(Self::HEADER_LEN + body.len() + Self::CHECKSUM_LEN);
        buf.extend_from_slice(&Self::MAGIC);
        buf.extend_from_slice(&Self::CURRENT_VERSION.to_le_bytes());
        buf.extend_from_slice(&(body.len() as u64).to_le_bytes());
        buf.extend_from_slice(&body);
        let hash = Sha256::digest(&body);
        buf.extend_from_slice(&hash);
        Ok(buf)
    }

    /// Validate the header and body checksum and return the body slice.
    ///
    /// Storage uses this to refuse a frame it cannot vouch for without decoding
    /// its body; recovery decodes the returned body afterwards.
    pub fn verify_frame_bytes(data: &[u8]) -> Result<&[u8], KinDbError> {
        if data.len() < Self::HEADER_LEN {
            return Err(KinDbError::StorageError(
                "authority frame too small for header".to_string(),
            ));
        }
        let magic = &data[0..4];
        if magic != Self::MAGIC {
            return Err(KinDbError::StorageError(format!(
                "invalid authority frame magic bytes: expected KNAF, got {magic:?}"
            )));
        }
        let version = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| KinDbError::SliceConversionError("version bytes".to_string()))?,
        );
        if version != Self::CURRENT_VERSION {
            return Err(KinDbError::StorageError(format!(
                "unsupported authority frame version: {version} (expected {})",
                Self::CURRENT_VERSION
            )));
        }
        let body_len = u64::from_le_bytes(
            data[8..16]
                .try_into()
                .map_err(|_| KinDbError::SliceConversionError("body_len bytes".to_string()))?,
        );
        let body_len = usize::try_from(body_len).map_err(|_| {
            KinDbError::StorageError(
                "authority frame header body length overflows usize".to_string(),
            )
        })?;
        let body_end = Self::HEADER_LEN.checked_add(body_len).ok_or_else(|| {
            KinDbError::StorageError(
                "authority frame header body length overflows usize".to_string(),
            )
        })?;
        let checksum_end = body_end.checked_add(Self::CHECKSUM_LEN).ok_or_else(|| {
            KinDbError::StorageError(
                "authority frame header body length overflows usize".to_string(),
            )
        })?;
        if data.len() < checksum_end {
            return Err(KinDbError::StorageError(
                "authority frame truncated".to_string(),
            ));
        }
        if data.len() > checksum_end {
            return Err(KinDbError::StorageError(
                "authority frame carries trailing bytes past its checksum".to_string(),
            ));
        }
        let body = &data[Self::HEADER_LEN..body_end];
        let stored_hash = &data[body_end..checksum_end];
        let computed_hash = Sha256::digest(body);
        if stored_hash != computed_hash.as_slice() {
            return Err(KinDbError::StorageError(
                "authority frame checksum mismatch: file is corrupted".to_string(),
            ));
        }
        Ok(body)
    }

    /// Deserialize a frame from bytes with header and checksum validation.
    pub fn from_bytes(data: &[u8]) -> Result<Self, KinDbError> {
        let body = Self::verify_frame_bytes(data)?;
        let frame: Self = rmp_serde::from_slice(body).map_err(|error| {
            KinDbError::StorageError(format!("authority frame deserialization failed: {error}"))
        })?;
        frame.validate_shape()?;
        Ok(frame)
    }

    fn validate_shape(&self) -> Result<(), KinDbError> {
        self.operation.validate().map_err(|error| {
            KinDbError::StorageError(format!(
                "authority frame carries an invalid operation record: {error}"
            ))
        })?;
        if self.operation.repository_id != self.repository_id {
            return Err(KinDbError::StorageError(format!(
                "authority frame for repository {} carries an operation of repository {}",
                self.repository_id, self.operation.repository_id
            )));
        }
        let expected_generation = self.base_generation().checked_add(1).ok_or_else(|| {
            KinDbError::StorageError("authority frame generation exhausted".to_string())
        })?;
        if self.generation() != expected_generation {
            return Err(KinDbError::StorageError(format!(
                "authority frame moves generation {} to {}, not to its successor",
                self.base_generation(),
                self.generation()
            )));
        }
        require_sorted_unique_by(&self.changes, |change| change.id, "frame changes")?;
        require_sorted_unique_by(
            &self.admission_policies,
            |policy| policy.change_id,
            "frame admission policies",
        )?;
        require_sorted_unique_by(
            &self.external_objects,
            |record| record.object,
            "frame external objects",
        )?;
        require_sorted_unique_by(&self.aliases, |alias| alias.oid, "frame aliases")?;
        require_sorted_unique_by(
            &self.workspaces,
            |workspace| workspace.workspace_id,
            "frame workspaces",
        )?;
        Ok(())
    }

    /// Drain the mutation that carried `current` to `next` into one frame, and
    /// prove before returning it that applying the frame to `current`
    /// reproduces `next` exactly.
    ///
    /// Both snapshots must carry a repository authority envelope, and `next`
    /// must be the immediate successor of `current`. This computes no diff over
    /// the store: new changes are found by probing the successor's change ids
    /// against the base, and every sorted envelope sequence is walked once
    /// against its base counterpart.
    pub(crate) fn encode(
        current: &GraphSnapshot,
        next: &GraphSnapshot,
    ) -> Result<Self, KinDbError> {
        let base = current.repository_authority.as_ref().ok_or_else(|| {
            KinDbError::StorageError(
                "authority frame base carries no repository authority envelope".to_string(),
            )
        })?;
        let successor = next.repository_authority.as_ref().ok_or_else(|| {
            KinDbError::StorageError(
                "authority frame successor carries no repository authority envelope".to_string(),
            )
        })?;
        if successor.repository_id != base.repository_id {
            return Err(KinDbError::StorageError(format!(
                "authority frame successor belongs to repository {}, base to {}",
                successor.repository_id, base.repository_id
            )));
        }
        if successor.schema_version != base.schema_version {
            return Err(KinDbError::StorageError(format!(
                "authority frame successor envelope schema {} differs from base schema {}",
                successor.schema_version, base.schema_version
            )));
        }
        let operation = successor.operation_log.last().cloned().ok_or_else(|| {
            KinDbError::StorageError(
                "authority frame successor carries no operation record".to_string(),
            )
        })?;
        if operation.roots_before != base.roots {
            return Err(KinDbError::StorageError(
                "authority frame successor's last operation does not start from the base roots"
                    .to_string(),
            ));
        }
        if operation.roots_after != successor.roots {
            return Err(KinDbError::StorageError(
                "authority frame successor's last operation does not end at the successor roots"
                    .to_string(),
            ));
        }

        let mut changes: Vec<SemanticChange> = next
            .changes
            .values()
            .filter(|change| !current.changes.contains_key(&change.id))
            .cloned()
            .collect();
        changes.sort_by_key(|change| change.id);
        let mut new_change_ids: Vec<_> = changes.iter().map(|change| change.id).collect();
        new_change_ids.sort_unstable();
        let admission_policies = successor
            .admission_policies
            .iter()
            .filter(|policy| new_change_ids.binary_search(&policy.change_id).is_ok())
            .cloned()
            .collect();
        let external_objects = absent_from_base(
            &base.external_objects,
            &successor.external_objects,
            |record| record.object,
        );
        let aliases = absent_from_base(&base.aliases, &successor.aliases, |alias| alias.oid);
        let git_external_authority = (successor.git_external_authority
            != base.git_external_authority)
            .then(|| successor.git_external_authority.clone());
        let base_workspaces: BTreeMap<_, _> = base
            .workspaces
            .iter()
            .map(|workspace| (workspace.workspace_id, workspace))
            .collect();
        let workspaces = successor
            .workspaces
            .iter()
            .filter(|workspace| base_workspaces.get(&workspace.workspace_id) != Some(&workspace))
            .cloned()
            .collect();

        let frame = Self {
            schema_version: successor.schema_version,
            repository_id: successor.repository_id.clone(),
            operation,
            changes,
            admission_policies,
            external_objects,
            aliases,
            git_external_authority,
            ref_state: successor.ref_state.clone(),
            workspaces,
            local_overlays: successor.local_overlays.clone(),
            merge_transactions: successor.merge_transactions.clone(),
        };
        frame.validate_shape()?;
        frame.prove_reproduces(current, next)?;
        Ok(frame)
    }

    /// The writer's own check that the reader's `apply` reconstructs `next`
    /// from `current` and this frame.
    ///
    /// The envelope is reconstructed with the exact reader code path and
    /// compared whole. Changes are immutable and the frame carries only ids
    /// absent from the base, so key-set identity is value identity there.
    fn prove_reproduces(
        &self,
        current: &GraphSnapshot,
        next: &GraphSnapshot,
    ) -> Result<(), KinDbError> {
        let base = current
            .repository_authority
            .as_ref()
            .expect("checked by encode");
        let successor = next
            .repository_authority
            .as_ref()
            .expect("checked by encode");
        let reconstructed = self.apply_to_envelope(base.clone())?;
        if reconstructed != *successor {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {} does not reproduce the successor envelope; refusing to persist it",
                self.generation()
            )));
        }
        let expected_changes = current.changes.len().checked_add(self.changes.len());
        if expected_changes != Some(next.changes.len()) {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {} does not account for the successor's changes: base {}, frame {}, successor {}",
                self.generation(),
                current.changes.len(),
                self.changes.len(),
                next.changes.len()
            )));
        }
        for change in &self.changes {
            if current.changes.contains_key(&change.id) {
                return Err(KinDbError::StorageError(format!(
                    "authority frame for generation {} re-adds change {} the base already carries",
                    self.generation(),
                    change.id
                )));
            }
        }
        if !next.entity_revisions.is_empty() {
            return Err(KinDbError::StorageError(
                "authority frame successor carries entity revisions, which authority never persists"
                    .to_string(),
            ));
        }
        if next.change_children != derive_change_children(&next.changes) {
            return Err(KinDbError::StorageError(
                "authority frame successor change-child index does not derive from its history"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Apply this frame to the state it extends, in place.
    ///
    /// Every refusal here is a corruption or a wrong-base signal, and nothing
    /// is applied when any check fails: the envelope is rebuilt privately and
    /// swapped in only after every collection accepted its patch.
    pub(crate) fn apply(&self, base: &mut GraphSnapshot) -> Result<(), KinDbError> {
        let base_envelope = base.repository_authority.as_ref().ok_or_else(|| {
            KinDbError::StorageError(format!(
                "authority frame for generation {} was applied to a snapshot with no repository authority envelope",
                self.generation()
            ))
        })?;
        for change in &self.changes {
            if base.changes.contains_key(&change.id) {
                return Err(KinDbError::StorageError(format!(
                    "authority frame for generation {} re-adds change {} the base already carries",
                    self.generation(),
                    change.id
                )));
            }
        }
        let envelope = self.apply_to_envelope(base_envelope.clone())?;
        for change in &self.changes {
            base.changes.insert(change.id, change.clone());
        }
        base.change_children = derive_change_children(&base.changes);
        base.entity_revisions.clear();
        base.repository_authority = Some(envelope);
        Ok(())
    }

    fn apply_to_envelope(
        &self,
        mut envelope: PersistedRepositoryAuthority,
    ) -> Result<PersistedRepositoryAuthority, KinDbError> {
        let generation = self.generation();
        if envelope.repository_id != self.repository_id {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {generation} belongs to repository {}, not {}",
                self.repository_id, envelope.repository_id
            )));
        }
        if envelope.schema_version != self.schema_version {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {generation} carries envelope schema {}, base carries {}",
                self.schema_version, envelope.schema_version
            )));
        }
        if self.operation.roots_before != envelope.roots {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {generation} does not extend the base it was applied to: the base is at generation {}, and its roots differ from the frame's roots_before",
                envelope.roots.generation
            )));
        }
        merge_absent(
            &mut envelope.admission_policies,
            &self.admission_policies,
            |policy| policy.change_id,
            "admission policy",
            generation,
        )?;
        merge_absent(
            &mut envelope.external_objects,
            &self.external_objects,
            |record| record.object,
            "external object",
            generation,
        )?;
        merge_absent(
            &mut envelope.aliases,
            &self.aliases,
            |alias| alias.oid,
            "external alias",
            generation,
        )?;
        if let Some(git_external_authority) = &self.git_external_authority {
            envelope.git_external_authority = git_external_authority.clone();
        }
        envelope.ref_state = self.ref_state.clone();
        let mut workspaces: BTreeMap<_, _> = envelope
            .workspaces
            .drain(..)
            .map(|workspace| (workspace.workspace_id, workspace))
            .collect();
        for workspace in &self.workspaces {
            workspaces.insert(workspace.workspace_id, workspace.clone());
        }
        envelope.workspaces = workspaces.into_values().collect();
        envelope.local_overlays = self.local_overlays.clone();
        envelope.merge_transactions = self.merge_transactions.clone();

        let operation = self.operation.clone();
        if envelope
            .receipts
            .iter()
            .any(|receipt| receipt.operation_id == operation.operation_id)
        {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {generation} repeats operation {}, which the base already receipts",
                operation.operation_id
            )));
        }
        let receipt = RepositoryCommitReceipt {
            operation_id: operation.operation_id,
            repository_id: operation.repository_id.clone(),
            transaction_hash: operation.transaction_hash,
            outcome: RepositoryCommitOutcome::Committed,
            generation,
            roots_before: operation.roots_before.clone(),
            roots_after: operation.roots_after.clone(),
            operation: operation.clone(),
        };
        receipt.validate().map_err(|error| {
            KinDbError::StorageError(format!(
                "authority frame for generation {generation} derives an invalid receipt: {error}"
            ))
        })?;
        envelope.roots = operation.roots_after.clone();
        envelope.operation_log.push(operation);
        envelope.receipts.push(receipt);
        envelope
            .receipts
            .sort_by_key(|receipt| receipt.operation_id);
        Ok(envelope)
    }
}

/// Successor entries whose key is absent from the sorted, unique base
/// sequence, in the successor's order.
fn absent_from_base<T: Clone, K: Ord>(
    base: &[T],
    successor: &[T],
    key: impl Fn(&T) -> K,
) -> Vec<T> {
    let base_keys: Vec<K> = base.iter().map(&key).collect();
    successor
        .iter()
        .filter(|entry| base_keys.binary_search(&key(entry)).is_err())
        .cloned()
        .collect()
}

/// Merge sorted, unique `incoming` into sorted, unique `existing`, refusing any
/// key the base already carries.
fn merge_absent<T: Clone, K: Ord + std::fmt::Debug>(
    existing: &mut Vec<T>,
    incoming: &[T],
    key: impl Fn(&T) -> K,
    label: &str,
    generation: Generation,
) -> Result<(), KinDbError> {
    if incoming.is_empty() {
        return Ok(());
    }
    let mut merged: BTreeMap<K, T> = existing
        .drain(..)
        .map(|entry| (key(&entry), entry))
        .collect();
    for entry in incoming {
        let entry_key = key(entry);
        if merged.contains_key(&entry_key) {
            return Err(KinDbError::StorageError(format!(
                "authority frame for generation {generation} re-adds {label} {entry_key:?}, which the base already carries"
            )));
        }
        merged.insert(entry_key, entry.clone());
    }
    *existing = merged.into_values().collect();
    Ok(())
}

fn require_sorted_unique_by<T, K: Ord>(
    values: &[T],
    key: impl Fn(&T) -> K,
    label: &str,
) -> Result<(), KinDbError> {
    let mut previous: Option<K> = None;
    for value in values {
        let current = key(value);
        if previous.as_ref().is_some_and(|old| old >= &current) {
            return Err(KinDbError::StorageError(format!(
                "authority {label} are not in canonical unique order"
            )));
        }
        previous = Some(current);
    }
    Ok(())
}

/// Digest that names an authority head reconstructed from one full snapshot
/// and an ordered chain of acknowledged frames.
///
/// It is the SHA-256 over the base digest followed by every acknowledged frame
/// digest in generation order, each as its raw 32 bytes. Journal-free
/// authority is named by the base digest alone, so this is only ever computed
/// when at least one frame is acknowledged.
pub fn journal_sha256(
    snapshot_sha256: &str,
    frame_sha256s: impl IntoIterator<Item = impl AsRef<str>>,
) -> Result<String, KinDbError> {
    let mut hasher = Sha256::new();
    hasher.update(decode_digest(snapshot_sha256, "snapshot")?);
    for frame in frame_sha256s {
        hasher.update(decode_digest(frame.as_ref(), "authority frame")?);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn decode_digest(hex_digest: &str, label: &str) -> Result<[u8; 32], KinDbError> {
    let bytes = hex::decode(hex_digest).map_err(|error| {
        KinDbError::StorageError(format!("{label} digest {hex_digest} is not hex: {error}"))
    })?;
    <[u8; 32]>::try_from(bytes).map_err(|_| {
        KinDbError::StorageError(format!("{label} digest {hex_digest} is not 32 bytes"))
    })
}
