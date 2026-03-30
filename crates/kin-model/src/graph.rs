// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

use crate::branch::Branch;
use crate::change::SemanticChange;
use crate::entity::{Entity, EntityKind};
use crate::ids::*;
use crate::relation::{GraphNodeId, Relation, RelationKind};
use crate::review::{
    Review, ReviewAssignment, ReviewComment, ReviewDecision, ReviewDecisionState, ReviewDiscussion,
    ReviewDiscussionId, ReviewDiscussionState, ReviewFilter, ReviewId, ReviewNote, ReviewNoteId,
};
use crate::verification::{ContractCoverageSummary, MockHint, VerificationRun, VerificationRunId};
use crate::work::{
    Annotation, AnnotationFilter, AnnotationId, WorkFilter, WorkId, WorkItem, WorkLink, WorkScope,
    WorkStatus,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// Domain sub-traits — narrower interfaces for consumers that only need a
// subset of GraphStore.
// ===========================================================================

/// Core entity and relation CRUD plus graph traversal operations.
pub trait EntityStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get_entity(&self, id: &EntityId) -> std::result::Result<Option<Entity>, Self::Error>;
    fn get_relations(
        &self,
        id: &EntityId,
        kinds: &[RelationKind],
    ) -> std::result::Result<Vec<Relation>, Self::Error>;
    fn get_all_relations_for_entity(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<Relation>, Self::Error>;
    fn get_downstream_impact(
        &self,
        id: &EntityId,
        max_depth: u32,
    ) -> std::result::Result<Vec<Entity>, Self::Error>;
    fn get_dependency_neighborhood(
        &self,
        id: &EntityId,
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error>;
    fn expand_neighborhood(
        &self,
        entity_ids: &[EntityId],
        edge_kinds: &[RelationKind],
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error>;
    fn traverse(
        &self,
        start: &GraphNodeId,
        edge_kinds: &[RelationKind],
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error>;
    fn find_dead_code(&self) -> std::result::Result<Vec<Entity>, Self::Error>;
    fn has_incoming_relation_kinds(
        &self,
        id: &EntityId,
        kinds: &[RelationKind],
        exclude_same_file: bool,
    ) -> std::result::Result<bool, Self::Error>;
    fn query_entities(
        &self,
        filter: &EntityFilter,
    ) -> std::result::Result<Vec<Entity>, Self::Error>;
    fn list_all_entities(&self) -> std::result::Result<Vec<Entity>, Self::Error>;
    fn upsert_entity(&self, entity: &Entity) -> std::result::Result<(), Self::Error>;
    fn upsert_relation(&self, relation: &Relation) -> std::result::Result<(), Self::Error>;
    fn remove_entity(&self, id: &EntityId) -> std::result::Result<(), Self::Error>;
    fn remove_relation(&self, id: &RelationId) -> std::result::Result<(), Self::Error>;

    // Shallow file tracking (C2 tier)
    fn upsert_shallow_file(
        &self,
        shallow: &crate::layout::ShallowTrackedFile,
    ) -> std::result::Result<(), Self::Error>;
    fn list_shallow_files(
        &self,
    ) -> std::result::Result<Vec<crate::layout::ShallowTrackedFile>, Self::Error>;
    fn upsert_structured_artifact(
        &self,
        artifact: &crate::layout::StructuredArtifact,
    ) -> std::result::Result<(), Self::Error>;
    fn list_structured_artifacts(
        &self,
    ) -> std::result::Result<Vec<crate::layout::StructuredArtifact>, Self::Error>;
    fn delete_structured_artifact(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<(), Self::Error>;
    fn upsert_opaque_artifact(
        &self,
        artifact: &crate::layout::OpaqueArtifact,
    ) -> std::result::Result<(), Self::Error>;
    fn list_opaque_artifacts(
        &self,
    ) -> std::result::Result<Vec<crate::layout::OpaqueArtifact>, Self::Error>;
    fn delete_opaque_artifact(&self, file_id: &FilePathId) -> std::result::Result<(), Self::Error>;
    fn upsert_file_layout(
        &self,
        layout: &crate::layout::FileLayout,
    ) -> std::result::Result<(), Self::Error>;
    fn get_file_layout(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<Option<crate::layout::FileLayout>, Self::Error>;
    fn list_file_layouts(&self)
        -> std::result::Result<Vec<crate::layout::FileLayout>, Self::Error>;
    fn get_file_hash(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<Option<Hash256>, Self::Error>;
    fn delete_file_layout(&self, file_id: &FilePathId) -> std::result::Result<(), Self::Error>;
}

/// Semantic change DAG and branch operations.
pub trait ChangeStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn get_entity_history(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<SemanticChange>, Self::Error>;
    fn find_merge_bases(
        &self,
        a: &SemanticChangeId,
        b: &SemanticChangeId,
    ) -> std::result::Result<Vec<SemanticChangeId>, Self::Error>;
    fn create_change(&self, change: &SemanticChange) -> std::result::Result<(), Self::Error>;
    fn get_change(
        &self,
        id: &SemanticChangeId,
    ) -> std::result::Result<Option<SemanticChange>, Self::Error>;
    fn get_changes_since(
        &self,
        base: &SemanticChangeId,
        head: &SemanticChangeId,
    ) -> std::result::Result<Vec<SemanticChange>, Self::Error>;
    fn get_branch(&self, name: &BranchName) -> std::result::Result<Option<Branch>, Self::Error>;
    fn create_branch(&self, branch: &Branch) -> std::result::Result<(), Self::Error>;
    fn update_branch_head(
        &self,
        name: &BranchName,
        new_head: &SemanticChangeId,
    ) -> std::result::Result<(), Self::Error>;
    fn delete_branch(&self, name: &BranchName) -> std::result::Result<(), Self::Error>;
    fn list_branches(&self) -> std::result::Result<Vec<Branch>, Self::Error>;
}

/// Work items, annotations, and work graph relationships.
pub trait WorkStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_work_item(&self, item: &WorkItem) -> std::result::Result<(), Self::Error>;
    fn get_work_item(&self, id: &WorkId) -> std::result::Result<Option<WorkItem>, Self::Error>;
    fn list_work_items(
        &self,
        filter: &WorkFilter,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn update_work_status(
        &self,
        id: &WorkId,
        status: WorkStatus,
    ) -> std::result::Result<(), Self::Error>;
    fn delete_work_item(&self, id: &WorkId) -> std::result::Result<(), Self::Error>;
    fn create_annotation(&self, ann: &Annotation) -> std::result::Result<(), Self::Error>;
    fn get_annotation(
        &self,
        id: &AnnotationId,
    ) -> std::result::Result<Option<Annotation>, Self::Error>;
    fn list_annotations(
        &self,
        filter: &AnnotationFilter,
    ) -> std::result::Result<Vec<Annotation>, Self::Error>;
    fn update_annotation_staleness(
        &self,
        id: &AnnotationId,
        staleness: crate::work::StalenessState,
    ) -> std::result::Result<(), Self::Error>;
    fn delete_annotation(&self, id: &AnnotationId) -> std::result::Result<(), Self::Error>;
    fn create_work_link(&self, link: &WorkLink) -> std::result::Result<(), Self::Error>;
    fn delete_work_link(&self, link: &WorkLink) -> std::result::Result<(), Self::Error>;
    fn get_work_for_scope(
        &self,
        scope: &WorkScope,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn get_annotations_for_scope(
        &self,
        scope: &WorkScope,
    ) -> std::result::Result<Vec<Annotation>, Self::Error>;
    fn get_child_work_items(
        &self,
        parent: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn get_parent_work_items(
        &self,
        child: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn get_blockers(&self, work_id: &WorkId) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn get_blocked_work_items(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error>;
    fn get_implementors(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<WorkScope>, Self::Error>;
    fn get_annotations_for_work_item(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<Annotation>, Self::Error>;
}

/// Test verification, coverage, contracts, and mock hints.
pub trait VerificationStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_test_case(
        &self,
        test: &crate::verification::TestCase,
    ) -> std::result::Result<(), Self::Error>;
    fn get_test_case(
        &self,
        id: &crate::verification::TestId,
    ) -> std::result::Result<Option<crate::verification::TestCase>, Self::Error>;
    fn get_tests_for_entity(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error>;
    fn delete_test_case(
        &self,
        id: &crate::verification::TestId,
    ) -> std::result::Result<(), Self::Error>;
    fn create_assertion(
        &self,
        assertion: &crate::verification::Assertion,
    ) -> std::result::Result<(), Self::Error>;
    fn get_assertion(
        &self,
        id: &crate::verification::AssertionId,
    ) -> std::result::Result<Option<crate::verification::Assertion>, Self::Error>;
    fn get_coverage_summary(
        &self,
    ) -> std::result::Result<crate::verification::CoverageSummary, Self::Error>;
    fn create_verification_run(
        &self,
        run: &VerificationRun,
    ) -> std::result::Result<(), Self::Error>;
    fn get_verification_run(
        &self,
        id: &VerificationRunId,
    ) -> std::result::Result<Option<VerificationRun>, Self::Error>;
    fn list_runs_for_test(
        &self,
        test_id: &crate::verification::TestId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error>;
    fn create_test_covers_entity(
        &self,
        test_id: &crate::verification::TestId,
        entity_id: &EntityId,
    ) -> std::result::Result<(), Self::Error>;
    fn create_test_covers_contract(
        &self,
        test_id: &crate::verification::TestId,
        contract_id: &ContractId,
    ) -> std::result::Result<(), Self::Error>;
    fn create_test_verifies_work(
        &self,
        test_id: &crate::verification::TestId,
        work_id: &WorkId,
    ) -> std::result::Result<(), Self::Error>;
    fn get_tests_covering_contract(
        &self,
        contract_id: &ContractId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error>;
    fn get_tests_verifying_work(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error>;
    fn create_mock_hint(&self, hint: &MockHint) -> std::result::Result<(), Self::Error>;
    fn get_mock_hints_for_test(
        &self,
        test_id: &crate::verification::TestId,
    ) -> std::result::Result<Vec<MockHint>, Self::Error>;
    fn link_run_proves_entity(
        &self,
        run_id: &VerificationRunId,
        entity_id: &EntityId,
    ) -> std::result::Result<(), Self::Error>;
    fn link_run_proves_work(
        &self,
        run_id: &VerificationRunId,
        work_id: &WorkId,
    ) -> std::result::Result<(), Self::Error>;
    fn list_runs_proving_entity(
        &self,
        entity_id: &EntityId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error>;
    fn list_runs_proving_work(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error>;
    fn create_contract(
        &self,
        contract: &crate::contract::Contract,
    ) -> std::result::Result<(), Self::Error>;
    fn get_contract(
        &self,
        id: &ContractId,
    ) -> std::result::Result<Option<crate::contract::Contract>, Self::Error>;
    fn list_contracts(&self) -> std::result::Result<Vec<crate::contract::Contract>, Self::Error>;
    fn get_contract_coverage_summary(
        &self,
    ) -> std::result::Result<ContractCoverageSummary, Self::Error>;
}

/// Actor provenance, delegations, approvals, and audit trail.
pub trait ProvenanceStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_actor(
        &self,
        actor: &crate::provenance::Actor,
    ) -> std::result::Result<(), Self::Error>;
    fn get_actor(
        &self,
        id: &crate::provenance::ActorId,
    ) -> std::result::Result<Option<crate::provenance::Actor>, Self::Error>;
    fn list_actors(&self) -> std::result::Result<Vec<crate::provenance::Actor>, Self::Error>;
    fn create_delegation(
        &self,
        delegation: &crate::provenance::Delegation,
    ) -> std::result::Result<(), Self::Error>;
    fn get_delegations_for_actor(
        &self,
        id: &crate::provenance::ActorId,
    ) -> std::result::Result<Vec<crate::provenance::Delegation>, Self::Error>;
    fn create_approval(
        &self,
        approval: &crate::provenance::Approval,
    ) -> std::result::Result<(), Self::Error>;
    fn get_approvals_for_change(
        &self,
        id: &SemanticChangeId,
    ) -> std::result::Result<Vec<crate::provenance::Approval>, Self::Error>;
    fn record_audit_event(
        &self,
        event: &crate::provenance::AuditEvent,
    ) -> std::result::Result<(), Self::Error>;
    fn query_audit_events(
        &self,
        actor_id: Option<&crate::provenance::ActorId>,
        limit: usize,
    ) -> std::result::Result<Vec<crate::provenance::AuditEvent>, Self::Error>;
}

/// Review decisions, notes, discussions, and assignments.
pub trait ReviewStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn create_review(&self, review: &Review) -> std::result::Result<(), Self::Error>;
    fn get_review(&self, id: &ReviewId) -> std::result::Result<Option<Review>, Self::Error>;
    fn list_reviews(&self, filter: &ReviewFilter) -> std::result::Result<Vec<Review>, Self::Error>;
    fn update_review_state(
        &self,
        id: &ReviewId,
        state: ReviewDecisionState,
    ) -> std::result::Result<(), Self::Error>;
    fn delete_review(&self, id: &ReviewId) -> std::result::Result<(), Self::Error>;

    fn add_review_decision(
        &self,
        id: &ReviewId,
        decision: &ReviewDecision,
    ) -> std::result::Result<(), Self::Error>;
    fn get_review_decisions(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewDecision>, Self::Error>;

    fn add_review_note(&self, note: &ReviewNote) -> std::result::Result<(), Self::Error>;
    fn get_review_notes(&self, id: &ReviewId) -> std::result::Result<Vec<ReviewNote>, Self::Error>;
    fn delete_review_note(&self, note_id: &ReviewNoteId) -> std::result::Result<(), Self::Error>;

    fn create_review_discussion(
        &self,
        discussion: &ReviewDiscussion,
    ) -> std::result::Result<(), Self::Error>;
    fn get_review_discussions(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewDiscussion>, Self::Error>;
    fn add_discussion_comment(
        &self,
        id: &ReviewDiscussionId,
        comment: &ReviewComment,
    ) -> std::result::Result<(), Self::Error>;
    fn set_discussion_state(
        &self,
        id: &ReviewDiscussionId,
        state: ReviewDiscussionState,
    ) -> std::result::Result<(), Self::Error>;

    fn assign_reviewer(
        &self,
        assignment: &ReviewAssignment,
    ) -> std::result::Result<(), Self::Error>;
    fn get_review_assignments(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewAssignment>, Self::Error>;
    fn remove_reviewer(
        &self,
        review_id: &ReviewId,
        reviewer: &str,
    ) -> std::result::Result<(), Self::Error>;
}

/// Session and intent management (daemon coordination).
pub trait SessionStore: Send + Sync {
    type Error: std::error::Error + Send + Sync + 'static;

    fn upsert_session(
        &self,
        session: &crate::session::AgentSession,
    ) -> std::result::Result<(), Self::Error>;
    fn get_session(
        &self,
        session_id: &SessionId,
    ) -> std::result::Result<Option<crate::session::AgentSession>, Self::Error>;
    fn delete_session(&self, session_id: &SessionId) -> std::result::Result<(), Self::Error>;
    fn list_sessions(&self) -> std::result::Result<Vec<crate::session::AgentSession>, Self::Error>;
    fn update_heartbeat(
        &self,
        session_id: &SessionId,
        heartbeat: &crate::timestamp::Timestamp,
    ) -> std::result::Result<(), Self::Error>;
    fn register_intent(
        &self,
        intent: &crate::session::Intent,
    ) -> std::result::Result<(), Self::Error>;
    fn get_intent(
        &self,
        intent_id: &IntentId,
    ) -> std::result::Result<Option<crate::session::Intent>, Self::Error>;
    fn delete_intent(&self, intent_id: &IntentId) -> std::result::Result<(), Self::Error>;
    fn list_intents_for_session(
        &self,
        session_id: &SessionId,
    ) -> std::result::Result<Vec<crate::session::Intent>, Self::Error>;
    fn list_all_intents(&self) -> std::result::Result<Vec<crate::session::Intent>, Self::Error>;
}

// ===========================================================================
// GraphStore — convenience supertrait
// ===========================================================================

/// Trait abstracting the graph database.
///
/// This is a convenience supertrait that combines all domain-specific store
/// traits. Consumers that only need a subset should bound on the narrower
/// sub-trait (e.g. `G: EntityStore`) instead.
///
/// Existing code using `G: GraphStore` continues to work unchanged — all
/// sub-trait methods are accessible through the supertrait bound.
pub trait GraphStore:
    EntityStore<Error = <Self as GraphStore>::Error>
    + ChangeStore<Error = <Self as GraphStore>::Error>
    + WorkStore<Error = <Self as GraphStore>::Error>
    + ReviewStore<Error = <Self as GraphStore>::Error>
    + VerificationStore<Error = <Self as GraphStore>::Error>
    + ProvenanceStore<Error = <Self as GraphStore>::Error>
    + SessionStore<Error = <Self as GraphStore>::Error>
    + Send
    + Sync
{
    type Error: std::error::Error + Send + Sync + 'static;
}

// ===========================================================================
// Supporting types
// ===========================================================================

/// A subgraph returned from neighborhood queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SubGraph {
    #[serde(default)]
    pub nodes: Vec<GraphNodeId>,
    pub entities: HashMap<EntityId, Entity>,
    pub relations: Vec<Relation>,
}

/// Filter for querying entities.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityFilter {
    pub kinds: Option<Vec<EntityKind>>,
    pub languages: Option<Vec<LanguageId>>,
    pub name_pattern: Option<String>,
    pub file_path: Option<FilePathId>,
}

// ===========================================================================
// Blanket impls — &G delegates to G for every trait
// ===========================================================================

impl<G: EntityStore> EntityStore for &G {
    type Error = G::Error;

    fn get_entity(&self, id: &EntityId) -> std::result::Result<Option<Entity>, Self::Error> {
        (**self).get_entity(id)
    }
    fn get_relations(
        &self,
        id: &EntityId,
        kinds: &[RelationKind],
    ) -> std::result::Result<Vec<Relation>, Self::Error> {
        (**self).get_relations(id, kinds)
    }
    fn get_all_relations_for_entity(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<Relation>, Self::Error> {
        (**self).get_all_relations_for_entity(id)
    }
    fn get_downstream_impact(
        &self,
        id: &EntityId,
        max_depth: u32,
    ) -> std::result::Result<Vec<Entity>, Self::Error> {
        (**self).get_downstream_impact(id, max_depth)
    }
    fn get_dependency_neighborhood(
        &self,
        id: &EntityId,
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error> {
        (**self).get_dependency_neighborhood(id, depth)
    }
    fn expand_neighborhood(
        &self,
        entity_ids: &[EntityId],
        edge_kinds: &[RelationKind],
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error> {
        (**self).expand_neighborhood(entity_ids, edge_kinds, depth)
    }
    fn traverse(
        &self,
        start: &GraphNodeId,
        edge_kinds: &[RelationKind],
        depth: u32,
    ) -> std::result::Result<SubGraph, Self::Error> {
        (**self).traverse(start, edge_kinds, depth)
    }
    fn find_dead_code(&self) -> std::result::Result<Vec<Entity>, Self::Error> {
        (**self).find_dead_code()
    }
    fn has_incoming_relation_kinds(
        &self,
        id: &EntityId,
        kinds: &[RelationKind],
        exclude_same_file: bool,
    ) -> std::result::Result<bool, Self::Error> {
        (**self).has_incoming_relation_kinds(id, kinds, exclude_same_file)
    }
    fn query_entities(
        &self,
        filter: &EntityFilter,
    ) -> std::result::Result<Vec<Entity>, Self::Error> {
        (**self).query_entities(filter)
    }
    fn list_all_entities(&self) -> std::result::Result<Vec<Entity>, Self::Error> {
        (**self).list_all_entities()
    }
    fn upsert_entity(&self, entity: &Entity) -> std::result::Result<(), Self::Error> {
        (**self).upsert_entity(entity)
    }
    fn upsert_relation(&self, relation: &Relation) -> std::result::Result<(), Self::Error> {
        (**self).upsert_relation(relation)
    }
    fn remove_entity(&self, id: &EntityId) -> std::result::Result<(), Self::Error> {
        (**self).remove_entity(id)
    }
    fn remove_relation(&self, id: &RelationId) -> std::result::Result<(), Self::Error> {
        (**self).remove_relation(id)
    }
    fn upsert_shallow_file(
        &self,
        shallow: &crate::layout::ShallowTrackedFile,
    ) -> std::result::Result<(), Self::Error> {
        (**self).upsert_shallow_file(shallow)
    }
    fn list_shallow_files(
        &self,
    ) -> std::result::Result<Vec<crate::layout::ShallowTrackedFile>, Self::Error> {
        (**self).list_shallow_files()
    }
    fn upsert_structured_artifact(
        &self,
        artifact: &crate::layout::StructuredArtifact,
    ) -> std::result::Result<(), Self::Error> {
        (**self).upsert_structured_artifact(artifact)
    }
    fn list_structured_artifacts(
        &self,
    ) -> std::result::Result<Vec<crate::layout::StructuredArtifact>, Self::Error> {
        (**self).list_structured_artifacts()
    }
    fn delete_structured_artifact(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).delete_structured_artifact(file_id)
    }
    fn upsert_opaque_artifact(
        &self,
        artifact: &crate::layout::OpaqueArtifact,
    ) -> std::result::Result<(), Self::Error> {
        (**self).upsert_opaque_artifact(artifact)
    }
    fn list_opaque_artifacts(
        &self,
    ) -> std::result::Result<Vec<crate::layout::OpaqueArtifact>, Self::Error> {
        (**self).list_opaque_artifacts()
    }
    fn delete_opaque_artifact(&self, file_id: &FilePathId) -> std::result::Result<(), Self::Error> {
        (**self).delete_opaque_artifact(file_id)
    }
    fn upsert_file_layout(
        &self,
        layout: &crate::layout::FileLayout,
    ) -> std::result::Result<(), Self::Error> {
        (**self).upsert_file_layout(layout)
    }
    fn get_file_layout(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<Option<crate::layout::FileLayout>, Self::Error> {
        (**self).get_file_layout(file_id)
    }
    fn list_file_layouts(
        &self,
    ) -> std::result::Result<Vec<crate::layout::FileLayout>, Self::Error> {
        (**self).list_file_layouts()
    }
    fn get_file_hash(
        &self,
        file_id: &FilePathId,
    ) -> std::result::Result<Option<Hash256>, Self::Error> {
        (**self).get_file_hash(file_id)
    }
    fn delete_file_layout(&self, file_id: &FilePathId) -> std::result::Result<(), Self::Error> {
        (**self).delete_file_layout(file_id)
    }
}

impl<G: ChangeStore> ChangeStore for &G {
    type Error = G::Error;

    fn get_entity_history(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<SemanticChange>, Self::Error> {
        (**self).get_entity_history(id)
    }
    fn find_merge_bases(
        &self,
        a: &SemanticChangeId,
        b: &SemanticChangeId,
    ) -> std::result::Result<Vec<SemanticChangeId>, Self::Error> {
        (**self).find_merge_bases(a, b)
    }
    fn create_change(&self, change: &SemanticChange) -> std::result::Result<(), Self::Error> {
        (**self).create_change(change)
    }
    fn get_change(
        &self,
        id: &SemanticChangeId,
    ) -> std::result::Result<Option<SemanticChange>, Self::Error> {
        (**self).get_change(id)
    }
    fn get_changes_since(
        &self,
        base: &SemanticChangeId,
        head: &SemanticChangeId,
    ) -> std::result::Result<Vec<SemanticChange>, Self::Error> {
        (**self).get_changes_since(base, head)
    }
    fn get_branch(&self, name: &BranchName) -> std::result::Result<Option<Branch>, Self::Error> {
        (**self).get_branch(name)
    }
    fn create_branch(&self, branch: &Branch) -> std::result::Result<(), Self::Error> {
        (**self).create_branch(branch)
    }
    fn update_branch_head(
        &self,
        name: &BranchName,
        new_head: &SemanticChangeId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).update_branch_head(name, new_head)
    }
    fn delete_branch(&self, name: &BranchName) -> std::result::Result<(), Self::Error> {
        (**self).delete_branch(name)
    }
    fn list_branches(&self) -> std::result::Result<Vec<Branch>, Self::Error> {
        (**self).list_branches()
    }
}

impl<G: WorkStore> WorkStore for &G {
    type Error = G::Error;

    fn create_work_item(&self, item: &WorkItem) -> std::result::Result<(), Self::Error> {
        (**self).create_work_item(item)
    }
    fn get_work_item(&self, id: &WorkId) -> std::result::Result<Option<WorkItem>, Self::Error> {
        (**self).get_work_item(id)
    }
    fn list_work_items(
        &self,
        filter: &WorkFilter,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).list_work_items(filter)
    }
    fn update_work_status(
        &self,
        id: &WorkId,
        status: WorkStatus,
    ) -> std::result::Result<(), Self::Error> {
        (**self).update_work_status(id, status)
    }
    fn delete_work_item(&self, id: &WorkId) -> std::result::Result<(), Self::Error> {
        (**self).delete_work_item(id)
    }
    fn create_annotation(&self, ann: &Annotation) -> std::result::Result<(), Self::Error> {
        (**self).create_annotation(ann)
    }
    fn get_annotation(
        &self,
        id: &AnnotationId,
    ) -> std::result::Result<Option<Annotation>, Self::Error> {
        (**self).get_annotation(id)
    }
    fn list_annotations(
        &self,
        filter: &AnnotationFilter,
    ) -> std::result::Result<Vec<Annotation>, Self::Error> {
        (**self).list_annotations(filter)
    }
    fn update_annotation_staleness(
        &self,
        id: &AnnotationId,
        staleness: crate::work::StalenessState,
    ) -> std::result::Result<(), Self::Error> {
        (**self).update_annotation_staleness(id, staleness)
    }
    fn delete_annotation(&self, id: &AnnotationId) -> std::result::Result<(), Self::Error> {
        (**self).delete_annotation(id)
    }
    fn create_work_link(&self, link: &WorkLink) -> std::result::Result<(), Self::Error> {
        (**self).create_work_link(link)
    }
    fn delete_work_link(&self, link: &WorkLink) -> std::result::Result<(), Self::Error> {
        (**self).delete_work_link(link)
    }
    fn get_work_for_scope(
        &self,
        scope: &WorkScope,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).get_work_for_scope(scope)
    }
    fn get_annotations_for_scope(
        &self,
        scope: &WorkScope,
    ) -> std::result::Result<Vec<Annotation>, Self::Error> {
        (**self).get_annotations_for_scope(scope)
    }
    fn get_child_work_items(
        &self,
        parent: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).get_child_work_items(parent)
    }
    fn get_parent_work_items(
        &self,
        child: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).get_parent_work_items(child)
    }
    fn get_blockers(&self, work_id: &WorkId) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).get_blockers(work_id)
    }
    fn get_blocked_work_items(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<WorkItem>, Self::Error> {
        (**self).get_blocked_work_items(work_id)
    }
    fn get_implementors(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<WorkScope>, Self::Error> {
        (**self).get_implementors(work_id)
    }
    fn get_annotations_for_work_item(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<Annotation>, Self::Error> {
        (**self).get_annotations_for_work_item(work_id)
    }
}

impl<G: ReviewStore> ReviewStore for &G {
    type Error = G::Error;

    fn create_review(&self, review: &Review) -> std::result::Result<(), Self::Error> {
        (**self).create_review(review)
    }
    fn get_review(&self, id: &ReviewId) -> std::result::Result<Option<Review>, Self::Error> {
        (**self).get_review(id)
    }
    fn list_reviews(&self, filter: &ReviewFilter) -> std::result::Result<Vec<Review>, Self::Error> {
        (**self).list_reviews(filter)
    }
    fn update_review_state(
        &self,
        id: &ReviewId,
        state: ReviewDecisionState,
    ) -> std::result::Result<(), Self::Error> {
        (**self).update_review_state(id, state)
    }
    fn delete_review(&self, id: &ReviewId) -> std::result::Result<(), Self::Error> {
        (**self).delete_review(id)
    }
    fn add_review_decision(
        &self,
        id: &ReviewId,
        decision: &ReviewDecision,
    ) -> std::result::Result<(), Self::Error> {
        (**self).add_review_decision(id, decision)
    }
    fn get_review_decisions(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewDecision>, Self::Error> {
        (**self).get_review_decisions(id)
    }
    fn add_review_note(&self, note: &ReviewNote) -> std::result::Result<(), Self::Error> {
        (**self).add_review_note(note)
    }
    fn get_review_notes(&self, id: &ReviewId) -> std::result::Result<Vec<ReviewNote>, Self::Error> {
        (**self).get_review_notes(id)
    }
    fn delete_review_note(&self, note_id: &ReviewNoteId) -> std::result::Result<(), Self::Error> {
        (**self).delete_review_note(note_id)
    }
    fn create_review_discussion(
        &self,
        discussion: &ReviewDiscussion,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_review_discussion(discussion)
    }
    fn get_review_discussions(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewDiscussion>, Self::Error> {
        (**self).get_review_discussions(id)
    }
    fn add_discussion_comment(
        &self,
        id: &ReviewDiscussionId,
        comment: &ReviewComment,
    ) -> std::result::Result<(), Self::Error> {
        (**self).add_discussion_comment(id, comment)
    }
    fn set_discussion_state(
        &self,
        id: &ReviewDiscussionId,
        state: ReviewDiscussionState,
    ) -> std::result::Result<(), Self::Error> {
        (**self).set_discussion_state(id, state)
    }
    fn assign_reviewer(
        &self,
        assignment: &ReviewAssignment,
    ) -> std::result::Result<(), Self::Error> {
        (**self).assign_reviewer(assignment)
    }
    fn get_review_assignments(
        &self,
        id: &ReviewId,
    ) -> std::result::Result<Vec<ReviewAssignment>, Self::Error> {
        (**self).get_review_assignments(id)
    }
    fn remove_reviewer(
        &self,
        review_id: &ReviewId,
        reviewer: &str,
    ) -> std::result::Result<(), Self::Error> {
        (**self).remove_reviewer(review_id, reviewer)
    }
}

impl<G: VerificationStore> VerificationStore for &G {
    type Error = G::Error;

    fn create_test_case(
        &self,
        test: &crate::verification::TestCase,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_test_case(test)
    }
    fn get_test_case(
        &self,
        id: &crate::verification::TestId,
    ) -> std::result::Result<Option<crate::verification::TestCase>, Self::Error> {
        (**self).get_test_case(id)
    }
    fn get_tests_for_entity(
        &self,
        id: &EntityId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error> {
        (**self).get_tests_for_entity(id)
    }
    fn delete_test_case(
        &self,
        id: &crate::verification::TestId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).delete_test_case(id)
    }
    fn create_assertion(
        &self,
        assertion: &crate::verification::Assertion,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_assertion(assertion)
    }
    fn get_assertion(
        &self,
        id: &crate::verification::AssertionId,
    ) -> std::result::Result<Option<crate::verification::Assertion>, Self::Error> {
        (**self).get_assertion(id)
    }
    fn get_coverage_summary(
        &self,
    ) -> std::result::Result<crate::verification::CoverageSummary, Self::Error> {
        (**self).get_coverage_summary()
    }
    fn create_verification_run(
        &self,
        run: &VerificationRun,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_verification_run(run)
    }
    fn get_verification_run(
        &self,
        id: &VerificationRunId,
    ) -> std::result::Result<Option<VerificationRun>, Self::Error> {
        (**self).get_verification_run(id)
    }
    fn list_runs_for_test(
        &self,
        test_id: &crate::verification::TestId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error> {
        (**self).list_runs_for_test(test_id)
    }
    fn create_test_covers_entity(
        &self,
        test_id: &crate::verification::TestId,
        entity_id: &EntityId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_test_covers_entity(test_id, entity_id)
    }
    fn create_test_covers_contract(
        &self,
        test_id: &crate::verification::TestId,
        contract_id: &ContractId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_test_covers_contract(test_id, contract_id)
    }
    fn create_test_verifies_work(
        &self,
        test_id: &crate::verification::TestId,
        work_id: &WorkId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_test_verifies_work(test_id, work_id)
    }
    fn get_tests_covering_contract(
        &self,
        contract_id: &ContractId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error> {
        (**self).get_tests_covering_contract(contract_id)
    }
    fn get_tests_verifying_work(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<crate::verification::TestCase>, Self::Error> {
        (**self).get_tests_verifying_work(work_id)
    }
    fn create_mock_hint(&self, hint: &MockHint) -> std::result::Result<(), Self::Error> {
        (**self).create_mock_hint(hint)
    }
    fn get_mock_hints_for_test(
        &self,
        test_id: &crate::verification::TestId,
    ) -> std::result::Result<Vec<MockHint>, Self::Error> {
        (**self).get_mock_hints_for_test(test_id)
    }
    fn link_run_proves_entity(
        &self,
        run_id: &VerificationRunId,
        entity_id: &EntityId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).link_run_proves_entity(run_id, entity_id)
    }
    fn link_run_proves_work(
        &self,
        run_id: &VerificationRunId,
        work_id: &WorkId,
    ) -> std::result::Result<(), Self::Error> {
        (**self).link_run_proves_work(run_id, work_id)
    }
    fn list_runs_proving_entity(
        &self,
        entity_id: &EntityId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error> {
        (**self).list_runs_proving_entity(entity_id)
    }
    fn list_runs_proving_work(
        &self,
        work_id: &WorkId,
    ) -> std::result::Result<Vec<VerificationRun>, Self::Error> {
        (**self).list_runs_proving_work(work_id)
    }
    fn create_contract(
        &self,
        contract: &crate::contract::Contract,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_contract(contract)
    }
    fn get_contract(
        &self,
        id: &ContractId,
    ) -> std::result::Result<Option<crate::contract::Contract>, Self::Error> {
        (**self).get_contract(id)
    }
    fn list_contracts(&self) -> std::result::Result<Vec<crate::contract::Contract>, Self::Error> {
        (**self).list_contracts()
    }
    fn get_contract_coverage_summary(
        &self,
    ) -> std::result::Result<ContractCoverageSummary, Self::Error> {
        (**self).get_contract_coverage_summary()
    }
}

impl<G: ProvenanceStore> ProvenanceStore for &G {
    type Error = G::Error;

    fn create_actor(
        &self,
        actor: &crate::provenance::Actor,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_actor(actor)
    }
    fn get_actor(
        &self,
        id: &crate::provenance::ActorId,
    ) -> std::result::Result<Option<crate::provenance::Actor>, Self::Error> {
        (**self).get_actor(id)
    }
    fn list_actors(&self) -> std::result::Result<Vec<crate::provenance::Actor>, Self::Error> {
        (**self).list_actors()
    }
    fn create_delegation(
        &self,
        delegation: &crate::provenance::Delegation,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_delegation(delegation)
    }
    fn get_delegations_for_actor(
        &self,
        id: &crate::provenance::ActorId,
    ) -> std::result::Result<Vec<crate::provenance::Delegation>, Self::Error> {
        (**self).get_delegations_for_actor(id)
    }
    fn create_approval(
        &self,
        approval: &crate::provenance::Approval,
    ) -> std::result::Result<(), Self::Error> {
        (**self).create_approval(approval)
    }
    fn get_approvals_for_change(
        &self,
        id: &SemanticChangeId,
    ) -> std::result::Result<Vec<crate::provenance::Approval>, Self::Error> {
        (**self).get_approvals_for_change(id)
    }
    fn record_audit_event(
        &self,
        event: &crate::provenance::AuditEvent,
    ) -> std::result::Result<(), Self::Error> {
        (**self).record_audit_event(event)
    }
    fn query_audit_events(
        &self,
        actor_id: Option<&crate::provenance::ActorId>,
        limit: usize,
    ) -> std::result::Result<Vec<crate::provenance::AuditEvent>, Self::Error> {
        (**self).query_audit_events(actor_id, limit)
    }
}

impl<G: SessionStore> SessionStore for &G {
    type Error = G::Error;

    fn upsert_session(
        &self,
        session: &crate::session::AgentSession,
    ) -> std::result::Result<(), Self::Error> {
        (**self).upsert_session(session)
    }
    fn get_session(
        &self,
        session_id: &SessionId,
    ) -> std::result::Result<Option<crate::session::AgentSession>, Self::Error> {
        (**self).get_session(session_id)
    }
    fn delete_session(&self, session_id: &SessionId) -> std::result::Result<(), Self::Error> {
        (**self).delete_session(session_id)
    }
    fn list_sessions(&self) -> std::result::Result<Vec<crate::session::AgentSession>, Self::Error> {
        (**self).list_sessions()
    }
    fn update_heartbeat(
        &self,
        session_id: &SessionId,
        heartbeat: &crate::timestamp::Timestamp,
    ) -> std::result::Result<(), Self::Error> {
        (**self).update_heartbeat(session_id, heartbeat)
    }
    fn register_intent(
        &self,
        intent: &crate::session::Intent,
    ) -> std::result::Result<(), Self::Error> {
        (**self).register_intent(intent)
    }
    fn get_intent(
        &self,
        intent_id: &IntentId,
    ) -> std::result::Result<Option<crate::session::Intent>, Self::Error> {
        (**self).get_intent(intent_id)
    }
    fn delete_intent(&self, intent_id: &IntentId) -> std::result::Result<(), Self::Error> {
        (**self).delete_intent(intent_id)
    }
    fn list_intents_for_session(
        &self,
        session_id: &SessionId,
    ) -> std::result::Result<Vec<crate::session::Intent>, Self::Error> {
        (**self).list_intents_for_session(session_id)
    }
    fn list_all_intents(&self) -> std::result::Result<Vec<crate::session::Intent>, Self::Error> {
        (**self).list_all_intents()
    }
}

/// Blanket impl: any shared reference to a GraphStore is also a GraphStore.
/// This allows `&InMemoryGraph` (from Arc::deref) to satisfy `G: GraphStore` bounds.
impl<G: GraphStore> GraphStore for &G {
    type Error = <G as GraphStore>::Error;
}
