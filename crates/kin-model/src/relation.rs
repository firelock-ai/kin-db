// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Firelock, LLC

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt;

use crate::ids::{ContractId, EntityId, RelationId, SemanticChangeId};
use crate::retrieval::ArtifactId;
use crate::verification::{TestId, VerificationRunId};
use crate::work::WorkId;

/// Typed graph node reference for first-class mixed-domain relations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum GraphNodeId {
    Entity(EntityId),
    Artifact(ArtifactId),
    Test(TestId),
    Contract(ContractId),
    Work(WorkId),
    VerificationRun(VerificationRunId),
}

impl GraphNodeId {
    pub fn as_entity(&self) -> Option<EntityId> {
        match self {
            Self::Entity(id) => Some(*id),
            _ => None,
        }
    }
}

impl fmt::Display for GraphNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entity(id) => write!(f, "entity:{id}"),
            Self::Artifact(id) => write!(f, "artifact:{}", id.0),
            Self::Test(id) => write!(f, "test:{id}"),
            Self::Contract(id) => write!(f, "contract:{id}"),
            Self::Work(id) => write!(f, "work:{id}"),
            Self::VerificationRun(id) => write!(f, "verification_run:{id}"),
        }
    }
}

impl From<EntityId> for GraphNodeId {
    fn from(value: EntityId) -> Self {
        Self::Entity(value)
    }
}

impl From<ArtifactId> for GraphNodeId {
    fn from(value: ArtifactId) -> Self {
        Self::Artifact(value)
    }
}

impl From<TestId> for GraphNodeId {
    fn from(value: TestId) -> Self {
        Self::Test(value)
    }
}

impl From<ContractId> for GraphNodeId {
    fn from(value: ContractId) -> Self {
        Self::Contract(value)
    }
}

impl From<WorkId> for GraphNodeId {
    fn from(value: WorkId) -> Self {
        Self::Work(value)
    }
}

impl From<VerificationRunId> for GraphNodeId {
    fn from(value: VerificationRunId) -> Self {
        Self::VerificationRun(value)
    }
}

/// A typed edge in the semantic graph.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Relation {
    pub id: RelationId,
    pub kind: RelationKind,
    pub src: GraphNodeId,
    pub dst: GraphNodeId,
    /// Confidence score (0.0 - 1.0).
    pub confidence: f32,
    pub origin: RelationOrigin,
    /// None while in overlay; set on kin commit.
    pub created_in: Option<SemanticChangeId>,
    /// For Calls/References edges, the module/package the target was imported from.
    /// Enables qualified cross-repo resolution in the spine.
    /// e.g., "requests" for `from requests import get`,
    ///        "kin_db" for `use kin_db::InMemoryGraph`
    #[serde(default)]
    pub import_source: Option<String>,
}

/// Classification of a relation edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum RelationKind {
    // ── Structural ──────────────────────────────
    Contains,   // parent encloses child (class→method, enum→variant)
    Extends,    // inherits implementation (class inheritance)
    Implements, // satisfies type contract (interface/trait/protocol)
    Overrides,  // method replaces parent method

    // ── Usage ───────────────────────────────────
    Calls,        // invokes at runtime
    Instantiates, // constructs an instance (new Foo(), Foo::new())
    References,   // non-call reference (field access, constant use)
    UsesType,     // type dependency in signature/body

    // ── Dependencies ────────────────────────────
    Imports,   // file-level import/use/require
    DependsOn, // package/crate-level dependency

    // ── Behavioral ──────────────────────────────
    EmitsEvent,       // publishes named event
    SubscribesTo,     // listens/subscribes to named event
    DefinesContract,  // defines API/schema contract
    ConsumesContract, // consumes API/schema contract

    // ── Concurrency ─────────────────────────────
    SendsMessage, // sends on typed channel/queue/mailbox
    Spawns,       // creates concurrent execution context

    // ── Lifecycle ───────────────────────────────
    Tests,       // test entity verifies target
    Covers,      // test provides runtime coverage
    CoChanges,   // entities change together in commits
    DerivedFrom, // generated/derived from another entity

    // ── Metadata ────────────────────────────────
    DocumentedBy, // entity has documentation
    OwnedBy,      // entity has responsible owner/team
    OwnedByFile,  // entity associated with file
}

/// How a relation was established.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
pub enum RelationOrigin {
    Parsed,
    Inferred,
    Manual,
    /// Discovered via Language Server Protocol (type-resolved).
    Lsp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relation_kind_roundtrip() {
        let kinds = vec![
            RelationKind::Calls,
            RelationKind::Imports,
            RelationKind::Contains,
            RelationKind::References,
            RelationKind::Implements,
            RelationKind::Extends,
            RelationKind::Tests,
            RelationKind::DependsOn,
            RelationKind::CoChanges,
            RelationKind::DefinesContract,
            RelationKind::ConsumesContract,
            RelationKind::EmitsEvent,
            RelationKind::OwnedBy,
            RelationKind::DocumentedBy,
            RelationKind::Covers,
            RelationKind::DerivedFrom,
            RelationKind::OwnedByFile,
        ];
        for k in kinds {
            let json = serde_json::to_string(&k).unwrap();
            let parsed: RelationKind = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, k);
        }
    }

    #[test]
    fn graph_node_id_roundtrips_through_json() {
        let node = GraphNodeId::Work(WorkId::new());
        let json = serde_json::to_string(&node).unwrap();
        let parsed: GraphNodeId = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed, node);
    }
}
