// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

//! Single-pass first-parent history replay for repository-authority admission.
//!
//! Repository admission and every unproven authority open must prove that the
//! persisted change DAG replays: each change's declared old payloads match the
//! state its first parent published, no change adds an identity that lineage
//! already carries, and no change leaves a relation dangling.
//!
//! Resolving that proof one change at a time walks from each change to genesis
//! and re-derives the whole graph, so a linear history costs `O(history^2)` in
//! walked changes and re-runs the whole-relation-set dangling scan once per
//! walked change. Material state has exactly one parent, the first ordered one,
//! so the changes form a first-parent forest: walking it once and carrying a
//! single state proves the same property at every node.
//!
//! Exactness is what makes this substitutable for the per-change replay rather
//! than an approximation of it:
//!
//! * Each change's state is uniquely determined, because its first-parent chain
//!   is unique. Validating a change once against that state is the same check
//!   the per-change replay performed once per target whose chain contained it.
//! * Rewinding a change applies the delta inverses in reverse order through the
//!   same validated transitions. Every inverse is fully determined by the delta
//!   itself: `Added` refuses unless the identity was absent, and `Modified` and
//!   `Removed` refuse unless the current payload equals the declared old one, so
//!   the pre-change value is exactly what the delta names.
//! * The dangling scan is narrowed to the endpoints a change can actually
//!   break. Given no relation dangled before the change, one can only dangle
//!   after it if the change asserted that relation or dropped one of its
//!   endpoint nodes, so checking those two sets is equivalent to rescanning
//!   every relation.
//!
//! The traversal ends by asserting the rewind restored the empty root state. A
//! state that does not come back to empty means the forward and inverse
//! transitions disagreed, and that fails the gate loudly instead of admitting
//! an unproven history.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::time::{Duration, Instant};

use kin_model::{
    Entity, EntityDelta, EntityId, ExternalReference, ExternalReferenceDelta, ExternalReferenceId,
    GraphNodeId, ModelError, Relation, RelationDelta, RelationId, ResolvedTree, SemanticChange,
    SemanticChangeId, TreeDelta,
};

use crate::error::KinDbError;

/// Wall-clock gap between progress reports for one long replay.
const PROGRESS_INTERVAL: Duration = Duration::from_secs(5);

/// Replays below this many changes finish fast enough that reporting progress
/// would only add noise to every ordinary command.
const PROGRESS_MIN_TOTAL: usize = 1_000;

/// Changes validated between clock reads, so the cadence check cannot become a
/// measurable share of a replay that is otherwise pure in-memory work.
const PROGRESS_CLOCK_STRIDE: usize = 256;

fn storage(message: String) -> KinDbError {
    KinDbError::StorageError(message)
}

/// Periodic `commits validated / total, elapsed` reporting for one replay.
///
/// Long admissions previously ran for hours with no output at all, which left
/// an operator unable to tell a slow phase from a wedged one.
pub(crate) struct ReplayProgress {
    phase: &'static str,
    total: usize,
    validated: usize,
    started: Instant,
    last_report: Instant,
}

impl ReplayProgress {
    pub(crate) fn new(phase: &'static str, total: usize) -> Self {
        let now = Instant::now();
        if total >= PROGRESS_MIN_TOTAL {
            tracing::info!(
                phase,
                total,
                "kindb: validating repository history, {total} changes"
            );
        }
        Self {
            phase,
            total,
            validated: 0,
            started: now,
            last_report: now,
        }
    }

    pub(crate) fn record(&mut self) {
        self.validated += 1;
        if self.total < PROGRESS_MIN_TOTAL || !self.validated.is_multiple_of(PROGRESS_CLOCK_STRIDE)
        {
            return;
        }
        let now = Instant::now();
        if now.duration_since(self.last_report) < PROGRESS_INTERVAL {
            return;
        }
        self.last_report = now;
        let elapsed_secs = self.started.elapsed().as_secs();
        let (phase, validated, total) = (self.phase, self.validated, self.total);
        tracing::info!(
            phase,
            validated,
            total,
            elapsed_secs,
            "kindb: validated {validated}/{total} changes in {elapsed_secs}s"
        );
    }

    pub(crate) fn finish(self) {
        if self.total < PROGRESS_MIN_TOTAL {
            return;
        }
        let elapsed_secs = self.started.elapsed().as_secs();
        let (phase, validated) = (self.phase, self.validated);
        tracing::info!(
            phase,
            validated,
            elapsed_secs,
            "kindb: validated {validated} changes in {elapsed_secs}s"
        );
    }
}

/// The exact state a first-parent replay carries between changes.
///
/// Entity revision timelines and tombstones are deliberately absent. Whole-graph
/// replay maintains both, but neither is ever read to decide a refusal, so
/// carrying them would add cost and an inverse problem without narrowing what
/// this gate accepts.
#[derive(Default)]
struct ReplayState {
    entities: HashMap<EntityId, Entity>,
    relations: HashMap<RelationId, Relation>,
    external_references: HashMap<ExternalReferenceId, ExternalReference>,
    tree: ResolvedTree,
    /// Relations naming each entity, so a removal checks its own referents
    /// instead of rescanning every relation in the graph.
    entity_referents: HashMap<EntityId, BTreeSet<RelationId>>,
    reference_referents: HashMap<ExternalReferenceId, BTreeSet<RelationId>>,
}

impl ReplayState {
    fn is_empty(&self) -> bool {
        self.entities.is_empty()
            && self.relations.is_empty()
            && self.external_references.is_empty()
            && self.tree.is_empty()
            && self.entity_referents.is_empty()
            && self.reference_referents.is_empty()
    }

    fn index_relation(&mut self, relation: &Relation) {
        for node in [relation.src, relation.dst] {
            match node {
                GraphNodeId::Entity(entity_id) => {
                    self.entity_referents
                        .entry(entity_id)
                        .or_default()
                        .insert(relation.id);
                }
                GraphNodeId::ExternalReference(reference_id) => {
                    self.reference_referents
                        .entry(reference_id)
                        .or_default()
                        .insert(relation.id);
                }
                GraphNodeId::Artifact(_)
                | GraphNodeId::Test(_)
                | GraphNodeId::Contract(_)
                | GraphNodeId::Work(_)
                | GraphNodeId::VerificationRun(_) => {}
            }
        }
    }

    fn unindex_relation(&mut self, relation: &Relation) {
        for node in [relation.src, relation.dst] {
            match node {
                GraphNodeId::Entity(entity_id) => {
                    if let Some(referents) = self.entity_referents.get_mut(&entity_id) {
                        referents.remove(&relation.id);
                        if referents.is_empty() {
                            self.entity_referents.remove(&entity_id);
                        }
                    }
                }
                GraphNodeId::ExternalReference(reference_id) => {
                    if let Some(referents) = self.reference_referents.get_mut(&reference_id) {
                        referents.remove(&relation.id);
                        if referents.is_empty() {
                            self.reference_referents.remove(&reference_id);
                        }
                    }
                }
                GraphNodeId::Artifact(_)
                | GraphNodeId::Test(_)
                | GraphNodeId::Contract(_)
                | GraphNodeId::Work(_)
                | GraphNodeId::VerificationRun(_) => {}
            }
        }
    }

    fn insert_relation(&mut self, relation: Relation) {
        if let Some(previous) = self.relations.insert(relation.id, relation.clone()) {
            self.unindex_relation(&previous);
        }
        self.index_relation(&relation);
    }

    fn remove_relation(&mut self, relation_id: RelationId) {
        if let Some(previous) = self.relations.remove(&relation_id) {
            self.unindex_relation(&previous);
        }
    }
}

/// Whether a change is being applied onto its first parent's state or unwound
/// back to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    Forward,
    Rewind,
}

/// Prove that every change reachable from `targets` along first-parent lineage
/// replays exactly, in one pass over that forest.
///
/// `targets` are the changes whose lineage must be proven. Their first-parent
/// ancestors are proven with them, because a target's own transition is only
/// meaningful against the state its lineage published.
pub(crate) fn validate_first_parent_history(
    changes: &HashMap<SemanticChangeId, SemanticChange>,
    targets: &[SemanticChangeId],
) -> Result<(), KinDbError> {
    let lineage = collect_first_parent_lineage(changes, targets)?;
    let mut children: BTreeMap<SemanticChangeId, BTreeSet<SemanticChangeId>> = BTreeMap::new();
    let mut roots = BTreeSet::new();
    for change_id in &lineage {
        let change = lineage_change(changes, change_id)?;
        match change.parents.first() {
            // Every first parent of a lineage member is itself a lineage
            // member, so this edge always stays inside the walked forest.
            Some(first_parent) => {
                children.entry(*first_parent).or_default().insert(*change_id);
            }
            None => {
                roots.insert(*change_id);
            }
        }
    }

    let mut frames = Vec::with_capacity(lineage.len().saturating_mul(2));
    for root in roots.iter().rev() {
        frames.push(Frame::Enter(*root));
    }
    let mut visited = HashSet::with_capacity(lineage.len());
    let mut state = ReplayState::default();
    let mut progress = ReplayProgress::new("history_replay", lineage.len());
    while let Some(frame) = frames.pop() {
        match frame {
            Frame::Enter(change_id) => {
                if !visited.insert(change_id) {
                    return Err(ModelError::Conflict(format!(
                        "cycle in first-parent history at change {change_id}"
                    ))
                    .into());
                }
                let change = lineage_change(changes, &change_id)?;
                apply_change(&mut state, change, Step::Forward)?;
                progress.record();
                frames.push(Frame::Exit(change_id));
                if let Some(next) = children.get(&change_id) {
                    for child in next.iter().rev() {
                        frames.push(Frame::Enter(*child));
                    }
                }
            }
            Frame::Exit(change_id) => {
                let change = lineage_change(changes, &change_id)?;
                apply_change(&mut state, change, Step::Rewind)?;
            }
        }
    }

    if visited.len() != lineage.len() {
        let change_id = lineage
            .iter()
            .find(|change_id| !visited.contains(change_id))
            .expect("an unvisited lineage member exists when the counts differ");
        return Err(ModelError::Conflict(format!(
            "cycle in first-parent history at change {change_id}"
        ))
        .into());
    }
    if !state.is_empty() {
        return Err(storage(
            "first-parent history replay did not restore the empty root state".to_string(),
        ));
    }
    progress.finish();
    Ok(())
}

enum Frame {
    Enter(SemanticChangeId),
    Exit(SemanticChangeId),
}

/// Every change reachable from `targets` by following first parents.
///
/// The upward walk stops at the first already-collected change, so a shared
/// trunk is walked once no matter how many targets name it. That is what keeps
/// collection linear in the lineage rather than in targets times history.
fn collect_first_parent_lineage(
    changes: &HashMap<SemanticChangeId, SemanticChange>,
    targets: &[SemanticChangeId],
) -> Result<HashSet<SemanticChangeId>, KinDbError> {
    let mut lineage = HashSet::new();
    for target in targets {
        let mut current = Some(*target);
        while let Some(change_id) = current {
            if !lineage.insert(change_id) {
                break;
            }
            let change = changes
                .get(&change_id)
                .ok_or_else(|| ModelError::ChangeNotFound(change_id.to_string()))?;
            current = change.parents.first().copied();
        }
    }
    Ok(lineage)
}

fn lineage_change<'a>(
    changes: &'a HashMap<SemanticChangeId, SemanticChange>,
    change_id: &SemanticChangeId,
) -> Result<&'a SemanticChange, KinDbError> {
    changes
        .get(change_id)
        .ok_or_else(|| ModelError::ChangeNotFound(change_id.to_string()).into())
}

/// Apply or unwind one change's complete transition.
///
/// Forward order is entities, external references, relations, then the
/// repository tree. Rewind is the exact reverse, which matters when one change
/// touches the same identity twice: unwinding in author order would check the
/// second delta's inverse against a state the first delta had already undone.
fn apply_change(
    state: &mut ReplayState,
    change: &SemanticChange,
    step: Step,
) -> Result<(), KinDbError> {
    let result = match step {
        Step::Forward => apply_forward(state, change),
        Step::Rewind => apply_rewind(state, change),
    };
    match step {
        Step::Forward => result,
        // A rewind failure is never a defect in the history being validated:
        // every inverse is determined by a transition this pass already
        // accepted. It means forward and inverse application disagreed, so the
        // gate fails loudly rather than reporting it as an invalid repository.
        Step::Rewind => result.map_err(|error| {
            storage(format!(
                "failed to rewind validated history replay at change {}: {error}",
                change.id
            ))
        }),
    }
}

fn apply_forward(state: &mut ReplayState, change: &SemanticChange) -> Result<(), KinDbError> {
    for delta in &change.entity_deltas {
        apply_entity_delta(state, change.id, delta)?;
    }
    for delta in &change.external_reference_deltas {
        apply_external_reference_delta(state, change.id, delta)?;
    }
    for delta in &change.relation_deltas {
        apply_relation_delta(state, change.id, delta)?;
    }
    apply_tree_deltas(state, change.id, &change.tree_deltas)?;
    validate_no_dangling_endpoints(state, change)
}

fn apply_rewind(state: &mut ReplayState, change: &SemanticChange) -> Result<(), KinDbError> {
    let inverse_tree: Vec<TreeDelta> = change
        .tree_deltas
        .iter()
        .rev()
        .map(TreeDelta::inverse)
        .collect();
    apply_tree_deltas(state, change.id, &inverse_tree)?;
    for delta in change.relation_deltas.iter().rev() {
        apply_relation_delta(state, change.id, &delta.inverse())?;
    }
    for delta in change.external_reference_deltas.iter().rev() {
        apply_external_reference_delta(state, change.id, &delta.inverse())?;
    }
    for delta in change.entity_deltas.iter().rev() {
        apply_entity_delta(state, change.id, &delta.inverse())?;
    }
    // The state being restored was proven when this change was entered, so
    // rescanning its endpoints would only re-derive an accepted conclusion.
    Ok(())
}

fn apply_entity_delta(
    state: &mut ReplayState,
    change_id: SemanticChangeId,
    delta: &EntityDelta,
) -> Result<(), KinDbError> {
    match delta {
        EntityDelta::Added { new } => {
            if state.entities.contains_key(&new.id) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} adds existing entity {}",
                    new.id
                ))
                .into());
            }
            state.entities.insert(new.id, new.clone());
        }
        EntityDelta::Modified { old, new } => {
            if old.id != new.id {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} modifies entity {} into different identity {}",
                    old.id, new.id
                ))
                .into());
            }
            if state.entities.get(&old.id) != Some(old) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} has stale old payload for entity {}",
                    old.id
                ))
                .into());
            }
            state.entities.insert(new.id, new.clone());
        }
        EntityDelta::Removed { old } => {
            if state.entities.get(&old.id) != Some(old) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} has stale old payload for removed entity {}",
                    old.id
                ))
                .into());
            }
            state.entities.remove(&old.id);
        }
    }
    Ok(())
}

fn apply_external_reference_delta(
    state: &mut ReplayState,
    change_id: SemanticChangeId,
    delta: &ExternalReferenceDelta,
) -> Result<(), KinDbError> {
    match delta {
        ExternalReferenceDelta::Added { new } => {
            new.validate()?;
            if state.external_references.contains_key(&new.id) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} adds existing external reference {}",
                    new.id
                ))
                .into());
            }
            state.external_references.insert(new.id, new.clone());
        }
        ExternalReferenceDelta::Removed { old } => {
            old.validate()?;
            if state.external_references.get(&old.id) != Some(old) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} has stale old payload for removed external reference {}",
                    old.id
                ))
                .into());
            }
            state.external_references.remove(&old.id);
        }
    }
    Ok(())
}

fn apply_relation_delta(
    state: &mut ReplayState,
    change_id: SemanticChangeId,
    delta: &RelationDelta,
) -> Result<(), KinDbError> {
    match delta {
        RelationDelta::Added { new } => {
            if state.relations.contains_key(&new.id) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} adds existing relation {}",
                    new.id
                ))
                .into());
            }
            state.insert_relation(new.clone());
        }
        RelationDelta::Modified { old, new } => {
            if old.id != new.id {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} modifies relation {} into different identity {}",
                    old.id, new.id
                ))
                .into());
            }
            if state.relations.get(&old.id) != Some(old) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} has stale old payload for relation {}",
                    old.id
                ))
                .into());
            }
            state.insert_relation(new.clone());
        }
        RelationDelta::Removed { old } => {
            if state.relations.get(&old.id) != Some(old) {
                return Err(ModelError::Conflict(format!(
                    "change {change_id} has stale old payload for removed relation {}",
                    old.id
                ))
                .into());
            }
            state.remove_relation(old.id);
        }
    }
    Ok(())
}

fn apply_tree_deltas(
    state: &mut ReplayState,
    change_id: SemanticChangeId,
    deltas: &[kin_model::TreeDelta],
) -> Result<(), KinDbError> {
    state.tree = state.tree.apply(deltas).map_err(|error| {
        ModelError::Conflict(format!(
            "invalid repository tree transition in change {change_id}: {error}"
        ))
    })?;
    Ok(())
}

/// Refuse a change that leaves a relation pointing at a node it dropped.
///
/// Whole-graph replay rescans every relation after every change. Given the
/// invariant held before this change, only the relations it asserted and the
/// endpoint nodes it dropped can have broken it, so those are the only ones
/// examined here.
fn validate_no_dangling_endpoints(
    state: &ReplayState,
    change: &SemanticChange,
) -> Result<(), KinDbError> {
    for delta in &change.relation_deltas {
        let Some(relation) = state.relations.get(&delta.target_id()) else {
            continue;
        };
        for node in [relation.src, relation.dst] {
            match node {
                GraphNodeId::Entity(entity_id) => {
                    if !state.entities.contains_key(&entity_id) {
                        return Err(dangling_entity(change.id, relation.id, entity_id));
                    }
                }
                GraphNodeId::ExternalReference(reference_id) => {
                    if !state.external_references.contains_key(&reference_id) {
                        return Err(dangling_reference(change.id, relation.id, reference_id));
                    }
                }
                GraphNodeId::Artifact(_)
                | GraphNodeId::Test(_)
                | GraphNodeId::Contract(_)
                | GraphNodeId::Work(_)
                | GraphNodeId::VerificationRun(_) => {}
            }
        }
    }

    for delta in &change.entity_deltas {
        let entity_id = delta.target_id();
        if state.entities.contains_key(&entity_id) {
            continue;
        }
        if let Some(relation_id) = state
            .entity_referents
            .get(&entity_id)
            .and_then(|referents| referents.iter().next())
        {
            return Err(dangling_entity(change.id, *relation_id, entity_id));
        }
    }

    for delta in &change.external_reference_deltas {
        let reference_id = delta.target_id();
        if state.external_references.contains_key(&reference_id) {
            continue;
        }
        if let Some(relation_id) = state
            .reference_referents
            .get(&reference_id)
            .and_then(|referents| referents.iter().next())
        {
            return Err(dangling_reference(change.id, *relation_id, reference_id));
        }
    }

    Ok(())
}

fn dangling_entity(
    change_id: SemanticChangeId,
    relation_id: RelationId,
    entity_id: EntityId,
) -> KinDbError {
    ModelError::Conflict(format!(
        "change {change_id} leaves relation {relation_id} dangling from entity {entity_id}; \
         relation removal must be explicit"
    ))
    .into()
}

fn dangling_reference(
    change_id: SemanticChangeId,
    relation_id: RelationId,
    reference_id: ExternalReferenceId,
) -> KinDbError {
    ModelError::Conflict(format!(
        "change {change_id} leaves relation {relation_id} dangling from external reference \
         {reference_id}; relation removal must be explicit"
    ))
    .into()
}
