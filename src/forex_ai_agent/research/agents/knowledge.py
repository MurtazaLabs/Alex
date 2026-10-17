"""Knowledge Agent - Knowledge graph and concept management."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    Concept,
    KnowledgeEdge,
    Hypothesis,
    ResearchPhase,
    Claim,
    Source,
)


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge agent."""
    max_concepts: int = 10000
    min_edge_weight: float = 0.3
    auto_infer_relationships: bool = True
    relationship_types: List[str] = field(default_factory=lambda: [
        "implies",
        "contradicts",
        "supports",
        "is_a",
        "part_of",
        "parameter_of",
        "regime_of",
        "risk_factor_of",
    ])


class KnowledgeAgent(BaseResearchAgent):
    """
    Knowledge Agent - Manages knowledge graph and concept persistence.
    
    Capabilities:
    - Concept model persistence
    - Knowledge graph (relationships between concepts)
    - Relationship inference
    - Versioned concept evolution
    - Query interface
    """
    
    def __init__(
        self,
        config: KnowledgeConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("knowledge", event_bus)
        self.knowledge_config = config or KnowledgeConfig()
        
        # Knowledge stores
        self.concepts: Dict[str, Concept] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        # Indices
        self.name_to_id: Dict[str, str] = {}
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.category_index: Dict[str, Set[str]] = defaultdict(set)
    
    def _register_capabilities(self):
        self.capabilities = [
            "concept_persistence",
            "knowledge_graph",
            "relationship_inference",
            "concept_versioning",
            "concept_query",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute knowledge integration for a hypothesis."""
        phase = input_data.get("phase", "KNOWLEDGE_INTEGRATION")
        
        if phase == "KNOWLEDGE_INTEGRATION":
            return await self.integrate_knowledge(hypothesis, input_data)
        elif phase == "CONCEPT_QUERY":
            return await self.query_concepts(input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def integrate_knowledge(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Integrate hypothesis results into knowledge graph."""
        # Get all results from pipeline
        lit_review = input_data.get("literature_review_output", {})
        quant_output = input_data.get("quant_formalization_output", {})
        bt_output = input_data.get("backtesting_output", {})
        critique_output = input_data.get("critique_output", {})
        stat_output = input_data.get("statistical_validation_output", {})
        
        # Extract concepts from all sources
        concepts_added = 0
        edges_added = 0
        
        # 1. Add hypothesis as concept
        hyp_concept = self._hypothesis_to_concept(hypothesis)
        self._add_concept(hyp_concept)
        concepts_added += 1
        
        # 2. Add claims as concepts
        lit_review = input_data.get("literature_review_output", {})
        claims = lit_review.get("claims", [])
        for claim in claims:
            claim_concept = self._claim_to_concept(claim)
            self._add_concept(claim_concept)
            concepts_added += 1
            
            # Link claim to hypothesis
            self._add_edge(
                hyp_concept.id, claim_concept.id,
                "supports" if claim.confidence > 0.5 else "contradicts",
                weight=claim.confidence,
                evidence=[claim.source_id]
            )
            edges_added += 1
        
        # 3. Add rules as concepts
        quant_output = input_data.get("quant_formalization_output", {})
        for rule_type in ["entry_rules", "exit_rules", "risk_rules"]:
            for rule in quant_output.get(rule_type, []):
                rule_concept = self._rule_to_concept(rule, rule_type)
                self._add_concept(rule_concept)
                concepts_added += 1
                
                # Link to hypothesis
                self._add_edge(
                    hyp_concept.id, rule_concept.id,
                    "parameter_of",
                    weight=0.8,
                )
                edges_added += 1
        
        # 4. Add backtest results as concepts
        bt_output = input_data.get("backtesting_output", {})
        best_variant = bt_output.get("best_variant", {})
        if best_variant.get("valid", False):
            result_concept = self._result_to_concept(best_variant)
            self._add_concept(result_concept)
            concepts_added += 1
            
            # Link to hypothesis
            self._add_edge(
                hyp_concept.id, result_concept.id,
                "produced",
                weight=0.9,
                evidence=[best_variant.get("variant_name", "")],
            )
            edges_added += 1
        
        # 5. Add critique as concept
        critique_output = input_data.get("critique_output", {})
        critique = critique_output.get("critique", {})
        if critique:
            critique_concept = self._critique_to_concept(critique)
            self._add_concept(critique_concept)
            concepts_added += 1
            
            self._add_edge(
                hyp_concept.id, critique_concept.id,
                "evaluated_by",
                weight=0.9,
            )
            edges_added += 1
        
        # 6. Add statistical validation as concept
        stat_output = input_data.get("statistical_validation_output", {})
        validation = stat_output.get("validation", {})
        if validation:
            val_concept = self._validation_to_concept(validation)
            self._add_concept(val_concept)
            concepts_added += 1
            
            self._add_edge(
                hyp_concept.id, val_concept.id,
                "validated_by",
                weight=0.9,
            )
            edges_added += 1
        
        # Infer new relationships
        inferred = 0
        if self.knowledge_config.auto_infer_relationships:
            inferred = await self._infer_relationships()
        
        return {
            "hypothesis_id": hypothesis.id if hasattr(hypothesis, "id") else "",
            "concepts_added": concepts_added,
            "edges_added": edges_added,
            "relationships_inferred": inferred,
            "total_concepts": len(self.concepts),
            "total_edges": len(self.edges),
            "knowledge_graph_updated": True,
        }
    
    def _hypothesis_to_concept(self, hypothesis: Any) -> Concept:
        """Convert hypothesis to concept."""
        concept = Concept(
            id=str(uuid4())[:8],
            name=hypothesis.title if hasattr(hypothesis, "title") else "Hypothesis",
            description=hypothesis.description if hasattr(hypothesis, "description") else "",
            category="hypothesis",
            properties={
                "research_question": hypothesis.research_question if hasattr(hypothesis, "research_question") else "",
                "variables": hypothesis.variables if hasattr(hypothesis, "variables") else {},
            },
            confidence=0.5,
            tags=["hypothesis", "research"],
        )
        return concept
    
    def _claim_to_concept(self, claim: Claim) -> Concept:
        """Convert claim to concept."""
        concept = Concept(
            id=f"claim_{claim.id}",
            name=claim.text[:50],
            description=claim.text,
            category="claim",
            properties={
                "claim_type": claim.claim_type.value,
                "source_id": claim.source_id,
                "confidence": claim.confidence,
            },
            confidence=claim.confidence,
            tags=["claim", claim.claim_type.value.lower()],
        )
        return concept
    
    def _rule_to_concept(self, rule: Dict[str, Any], rule_type: str) -> Concept:
        """Convert rule to concept."""
        concept = Concept(
            id=f"rule_{rule['name']}",
            name=rule["name"],
            description=rule["description"],
            category="rule",
            properties={
                "rule_type": rule_type,
                "conditions": rule.get("conditions", []),
                "parameters": rule.get("parameters", {}),
            },
            tags=["rule", rule_type.lower()],
        )
        return concept
    
    def _result_to_concept(self, result: Dict[str, Any]) -> Concept:
        """Convert backtest result to concept."""
        concept = Concept(
            id=f"result_{result.get('variant_name', 'unknown')}",
            name=f"Backtest Result: {result.get('variant_name', 'unknown')}",
            description=f"Backtest result with {result.get('trades', 0)} trades",
            category="backtest_result",
            properties={
                "trades": result.get("trades", 0),
                "metrics": result.get("metrics", {}),
                "parameters": result.get("parameters", {}),
            },
            confidence=0.7,
            tags=["backtest", "result"],
        )
        return concept
    
    def _critique_to_concept(self, critique: Dict[str, Any]) -> Concept:
        """Convert critique to concept."""
        concept = Concept(
            id=f"critique_{critique.get('hypothesis_id', 'unknown')}",
            name=f"Critique: {critique.get('verdict', 'UNKNOWN')}",
            description=f"Critique score: {critique.get('overall_score', 0):.2f}",
            category="critique",
            properties={
                "overall_score": critique.get("overall_score", 0),
                "verdict": critique.get("verdict", ""),
                "critical_issues": critique.get("critical_issues", []),
                "warnings": critique.get("warnings", []),
            },
            confidence=0.8,
            tags=["critique", critique.get("verdict", "").lower()],
        )
        return concept
    
    def _validation_to_concept(self, validation: Dict[str, Any]) -> Concept:
        """Convert statistical validation to concept."""
        concept = Concept(
            id=f"validation_{validation.get('hypothesis_id', 'unknown')}",
            name=f"Statistical Validation: {'SIGNIFICANT' if validation.get('statistically_significant') else 'NOT SIGNIFICANT'}",
            description=f"p-value: {validation.get('p_value', 'N/A')}",
            category="statistical_validation",
            properties={
                "p_value": validation.get("p_value"),
                "statistically_significant": validation.get("statistically_significant"),
                "oos_sharpe": validation.get("oos_sharpe"),
                "walk_forward_consistency": validation.get("wf_consistency"),
            },
            confidence=0.85,
            tags=["validation", "statistics"],
        )
        return concept
    
    def _add_concept(self, concept: Concept):
        """Add or update a concept."""
        # Check if concept already exists
        existing_id = self.name_to_id.get(concept.name)
        if existing_id:
            existing = self.concepts[existing_id]
            # Update if new version has higher confidence
            if concept.confidence > existing.confidence:
                existing.description = concept.description
                existing.properties.update(concept.properties)
                existing.confidence = concept.confidence
                existing.version += 1
                existing.updated_at = datetime.utcnow()
            return existing_id
        
        # Add new concept
        concept_id = concept.id
        self.concepts[concept_id] = concept
        self.name_to_id[concept.name] = concept_id
        
        # Update indices
        self.category_index[concept.category].add(concept_id)
        for tag in concept.tags:
            self.tag_index[tag].add(concept_id)
        
        return concept_id
    
    def _add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        weight: float = 1.0,
        evidence: List[str] = None,
    ) -> KnowledgeEdge:
        """Add a relationship edge between concepts."""
        edge_id = f"{source_id}_{target_id}_{relationship_type}"
        
        if edge_id in self.edges:
            # Update weight if stronger
            if weight > self.edges[edge_id].weight:
                self.edges[edge_id].weight = weight
                self.edges[edge_id].evidence.extend(evidence or [])
            return self.edges[edge_id]
        
        edge = KnowledgeEdge(
            id=edge_id,
            source_concept_id=source_id,
            target_concept_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            evidence=evidence or [],
        )
        self.edges[edge_id] = edge
        return edge
    
    async def _infer_relationships(self) -> int:
        """Infer new relationships between concepts."""
        inferred = 0
        
        concepts = list(self.concepts.values())
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i+1:]:
                # Check for implied relationships
                rel_type, weight = self._infer_relationship(c1, c2)
                if rel_type and weight >= self.knowledge_config.min_edge_weight:
                    self._add_edge(c1.id, c2.id, rel_type, weight)
                    inferred += 1
        
        return inferred
    
    def _infer_relationship(
        self,
        c1: Concept,
        c2: Concept,
    ) -> Tuple[Optional[str], float]:
        """Infer relationship between two concepts."""
        # Simple heuristic-based inference
        
        # Same category
        if c1.category == c2.category:
            if c1.category == "claim":
                # Check for contradiction
                if "not" in c1.description.lower() and "not" not in c2.description.lower():
                    return "contradicts", 0.4
            return "related_to", 0.3
        
        # Hypothesis -> rule/claim
        if c1.category == "hypothesis" and c2.category in ["claim", "rule"]:
            return "implies", 0.5
        
        # Rule -> result
        if c1.category == "rule" and c2.category == "backtest_result":
            return "parameter_of", 0.7
        
        # Result -> critique
        if c1.category == "backtest_result" and c2.category == "critique":
            return "evaluated_by", 0.8
        
        # Result -> validation
        if c1.category == "backtest_result" and c2.category == "statistical_validation":
            return "validated_by", 0.8
        
        return None, 0.0
    
    async def query_concepts(
        self,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Query the knowledge graph."""
        query_type = input_data.get("query_type", "search")
        
        if query_type == "search":
            return await self._search_concepts(input_data)
        elif query_type == "neighbors":
            return await self._get_neighbors(input_data)
        elif query_type == "path":
            return await self._find_path(input_data)
        elif query_type == "subgraph":
            return await self._get_subgraph(input_data)
        else:
            return {"error": f"Unknown query type: {query_type}"}
    
    async def _search_concepts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Search concepts by text."""
        query = input_data.get("query", "").lower()
        category = input_data.get("category")
        tags = input_data.get("tags", [])
        limit = input_data.get("limit", 20)
        
        results = []
        for concept in self.concepts.values():
            # Text match
            if query and query not in concept.name.lower() and query not in concept.description.lower():
                continue
            
            # Category filter
            if category and concept.category != category:
                continue
            
            # Tag filter
            if tags and not any(t in concept.tags for t in tags):
                continue
            
            results.append({
                "id": concept.id,
                "name": concept.name,
                "description": concept.description[:200],
                "category": concept.category,
                "confidence": concept.confidence,
                "tags": concept.tags,
                "version": concept.version,
            })
            
            if len(results) >= limit:
                break
        
        return {"results": results, "count": len(results)}
    
    async def _get_neighbors(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get neighboring concepts."""
        concept_id = input_data.get("concept_id")
        depth = input_data.get("depth", 1)
        rel_types = input_data.get("relationship_types")
        
        if concept_id not in self.concepts:
            return {"error": "Concept not found"}
        
        neighbors = []
        visited = set([concept_id])
        current_level = [concept_id]
        
        for d in range(depth):
            next_level = []
            for cid in current_level:
                for edge in self.edges.values():
                    if edge.source_concept_id == cid:
                        if rel_types and edge.relationship_type not in rel_types:
                            continue
                        if edge.target_concept_id not in visited:
                            visited.add(edge.target_concept_id)
                            next_level.append(edge.target_concept_id)
                            target = self.concepts.get(edge.target_concept_id)
                            if target:
                                neighbors.append({
                                    "concept": target.__dict__,
                                    "relationship": edge.relationship_type,
                                    "weight": edge.weight,
                                    "depth": d + 1,
                                })
                    elif edge.target_concept_id == cid:
                        if rel_types and edge.relationship_type not in rel_types:
                            continue
                        if edge.source_concept_id not in visited:
                            visited.add(edge.source_concept_id)
                            next_level.append(edge.source_concept_id)
                            source = self.concepts.get(edge.source_concept_id)
                            if source:
                                neighbors.append({
                                    "concept": source.__dict__,
                                    "relationship": f"reverse_{edge.relationship_type}",
                                    "weight": edge.weight,
                                    "depth": d + 1,
                                })
            current_level = next_level
        
        return {"concept_id": concept_id, "neighbors": neighbors}
    
    async def _find_path(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Find path between two concepts."""
        source_id = input_data.get("source_id")
        target_id = input_data.get("target_id")
        max_depth = input_data.get("max_depth", 5)
        
        if source_id not in self.concepts or target_id not in self.concepts:
            return {"error": "Concept not found"}
        
        # BFS
        queue = [(source_id, [source_id])]
        visited = set([source_id])
        
        while queue:
            current, path = queue.pop(0)
            
            if len(path) > max_depth:
                continue
            
            if current == target_id:
                # Found path
                path_details = []
                for i, cid in enumerate(path):
                    concept = self.concepts.get(cid)
                    if concept:
                        path_details.append({
                            "id": cid,
                            "name": concept.name,
                            "category": concept.category,
                        })
                
                return {"path": path_details, "length": len(path) - 1}
            
            for edge in self.edges.values():
                if edge.source_concept_id == current and edge.target_concept_id not in visited:
                    visited.add(edge.target_concept_id)
                    queue.append((edge.target_concept_id, path + [edge.target_concept_id]))
        
        return {"path": None, "found": False}
    
    async def _get_subgraph(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get subgraph around concept."""
        concept_id = input_data.get("concept_id")
        radius = input_data.get("radius", 2)
        
        if concept_id not in self.concepts:
            return {"error": "Concept not found"}
        
        # Get neighbors up to radius
        result = await self._get_neighbors({"concept_id": concept_id, "depth": radius})
        
        # Build subgraph
        nodes = {concept_id: self.concepts[concept_id].__dict__}
        edges = []
        
        if "neighbors" in result:
            for n in result["neighbors"]:
                cid = n["concept"]["id"]
                nodes[cid] = n["concept"]
                edges.append({
                    "source": concept_id,
                    "target": cid,
                    "type": n["relationship"],
                    "weight": n["weight"],
                })
        
        return {"nodes": nodes, "edges": edges, "center": concept_id}
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.integrate_knowledge(hypothesis, input_data)


def create_knowledge_agent(
    config: KnowledgeConfig = None,
    event_bus: Optional[EventBus] = None,
) -> KnowledgeAgent:
    return KnowledgeAgent(config=config, event_bus=event_bus)