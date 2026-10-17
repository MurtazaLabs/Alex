"""Neo4j-backed Knowledge Graph for the Forex AI Agent.

Provides persistent graph storage using Neo4j for concepts, relationships,
and research pipeline integration.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from forex_ai_agent.core.event_bus import EventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    Concept,
    KnowledgeEdge,
    Hypothesis,
    ResearchPhase,
    Claim,
    Source,
)

logger = logging.getLogger(__name__)


@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout: int = 30
    max_retry_time: int = 30


class Neo4jKnowledgeGraph:
    """
    Neo4j-backed knowledge graph implementation.
    
    Provides:
    - Persistent concept storage
    - Relationship management
    - Cypher query interface
    - Graph algorithms (PageRank, community detection)
    - Path finding
    - Subgraph extraction
    """

    def __init__(self, config: Neo4jConfig = None):
        self.config = config or Neo4jConfig()
        self._driver: Optional[AsyncDriver] = None
        self._initialized = False

    async def connect(self) -> None:
        """Establish connection to Neo4j."""
        if self._driver is not None:
            return
        
        try:
            self._driver = AsyncGraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_timeout=self.config.connection_timeout,
                max_retry_time=self.config.max_retry_time,
            )
            
            # Verify connection
            await self._driver.verify_connectivity()
            self._initialized = True
            logger.info("Neo4j connection established", uri=self.config.uri)
            
            # Create indexes
            await self._create_indexes()
            
        except ServiceUnavailable as e:
            logger.error("Neo4j connection failed", error=str(e))
            raise

    async def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            self._initialized = False
            logger.info("Neo4j connection closed")

    async def _create_indexes(self) -> None:
        """Create indexes for efficient querying."""
        indexes = [
            "CREATE INDEX concept_name_idx IF NOT EXISTS FOR (c:Concept) ON (c.name)",
            "CREATE INDEX concept_category_idx IF NOT EXISTS FOR (c:Concept) ON (c.category)",
            "CREATE INDEX concept_id_idx IF NOT EXISTS FOR (c:Concept) ON (c.id)",
            "CREATE INDEX edge_type_idx IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relationship_type)",
        ]
        
        async with self._driver.session(database=self.config.database) as session:
            for index in indexes:
                try:
                    await session.run(index)
                except Exception as e:
                    logger.warning(f"Index creation skipped: {e}")

    async def add_concept(self, concept: Concept) -> str:
        """Add or update a concept in the graph."""
        if not self._initialized:
            await self.connect()
        
        query = """
        MERGE (c:Concept {id: $id})
        SET c.name = $name,
            c.display_name = $display_name,
            c.definition = $definition,
            c.category = $category,
            c.confidence = $confidence,
            c.tags = $tags,
            c.properties = $properties,
            c.version = $version,
            c.updated_at = $updated_at
        RETURN c.id
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                id=concept.id,
                name=concept.name,
                display_name=concept.display_name or "",
                definition=concept.definition,
                category=concept.category,
                confidence=concept.confidence,
                tags=concept.tags,
                properties=concept.properties,
                version=concept.version,
                updated_at=datetime.utcnow().isoformat(),
            )
            record = await result.single()
            return record["c.id"]

    async def add_edge(self, edge: KnowledgeEdge) -> str:
        """Add a relationship between concepts."""
        if not self._initialized:
            await self.connect()
        
        query = f"""
        MATCH (a:Concept {{id: $source_id}})
        MATCH (b:Concept {{id: $target_id}})
        MERGE (a)-[r:{edge.relationship_type.upper()}]->(b)
        SET r.weight = $weight,
            r.evidence = $evidence,
            r.id = $edge_id,
            r.updated_at = $updated_at
        RETURN r.id
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(
                query,
                source_id=edge.source_concept_id,
                target_id=edge.target_concept_id,
                weight=edge.weight,
                evidence=edge.evidence,
                edge_id=edge.id,
                updated_at=datetime.utcnow().isoformat(),
            )
            record = await result.single()
            return record["r.id"]

    async def get_concept(self, concept_id: str) -> Optional[Concept]:
        """Get a concept by ID."""
        if not self._initialized:
            await self.connect()
        
        query = "MATCH (c:Concept {id: $id}) RETURN c"
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, id=concept_id)
            record = await result.single()
            
            if record:
                node = record["c"]
                return Concept(
                    id=node["id"],
                    name=node["name"],
                    display_name=node.get("display_name"),
                    definition=node["definition"],
                    category=node["category"],
                    confidence=node.get("confidence", 0.5),
                    tags=node.get("tags", []),
                    properties=node.get("properties", {}),
                    version=node.get("version", 1),
                )
            return None

    async def get_neighbors(
        self,
        concept_id: str,
        depth: int = 1,
        relationship_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get neighboring concepts."""
        if not self._initialized:
            await self.connect()
        
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        
        query = f"""
        MATCH (c:Concept {{id: $id}})-[r{rel_filter}*1..{depth}]-(n:Concept)
        WHERE c.id <> n.id
        RETURN DISTINCT n, r, length(r) as depth
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, id=concept_id)
            neighbors = []
            
            async for record in result:
                node = record["n"]
                rels = record["r"]
                depth = record["depth"]
                
                # Get relationship type from first relationship
                rel_type = rels[0].type if rels else "RELATES_TO"
                weight = rels[0].get("weight", 1.0) if rels else 1.0
                
                neighbors.append({
                    "concept": dict(node),
                    "relationship": rel_type,
                    "weight": weight,
                    "depth": depth,
                })
            
            return neighbors

    async def find_path(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5,
    ) -> Optional[List[Dict[str, Any]]]:
        """Find shortest path between two concepts."""
        if not self._initialized:
            await self.connect()
        
        query = """
        MATCH (source:Concept {id: $source_id}), (target:Concept {id: $target_id})
        MATCH path = shortestPath((source)-[*..{max_depth}]-(target))
        RETURN path
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, source_id=source_id, target_id=target_id, max_depth=max_depth)
            record = await result.single()
            
            if not record:
                return None
            
            path = record["path"]
            nodes = []
            edges = []
            
            for node in path.nodes:
                nodes.append(dict(node))
            
            for rel in path.relationships:
                edges.append({
                    "source": rel.start_node["id"],
                    "target": rel.end_node["id"],
                    "type": rel.type,
                    "weight": rel.get("weight", 1.0),
                })
            
            return {
                "nodes": nodes,
                "edges": edges,
                "length": len(edges),
            }

    async def get_subgraph(
        self,
        concept_id: str,
        radius: int = 2,
    ) -> Dict[str, Any]:
        """Get subgraph around a concept."""
        if not self._initialized:
            await self.connect()
        
        query = """
        MATCH (center:Concept {id: $id})
        CALL apoc.path.subgraphNodes(center, {{maxLevel: {radius}}}) YIELD node
        WITH collect(node) as nodes
        UNWIND nodes as n
        MATCH (n)-[r]-(m)
        WHERE m IN nodes
        RETURN n, collect({target: m, type: type(r), weight: r.weight}) as relationships
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, id=concept_id, radius=radius)
            
            nodes = {}
            edges = []
            
            async for record in result:
                node = record["n"]
                nodes[node["id"]] = dict(node)
                
                for rel in record["relationships"]:
                    edges.append({
                        "source": node["id"],
                        "target": rel["target"]["id"],
                        "type": rel["type"],
                        "weight": rel["weight"],
                    })
            
            return {"nodes": nodes, "edges": edges, "center": concept_id}

    async def search_concepts(
        self,
        query: str = "",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search concepts by text, category, or tags."""
        if not self._initialized:
            await self.connect()
        
        conditions = []
        params = {"limit": limit}
        
        if query:
            conditions.append("(c.name CONTAINS $query OR c.definition CONTAINS $query)")
            params["query"] = query
        
        if category:
            conditions.append("c.category = $category")
            params["category"] = category
        
        if tags:
            conditions.append("ANY(tag IN c.tags WHERE tag IN $tags)")
            params["tags"] = tags
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        MATCH (c:Concept)
        WHERE {where_clause}
        RETURN c
        LIMIT $limit
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query, params)
            
            concepts = []
            async for record in result:
                node = record["c"]
                concepts.append({
                    "id": node["id"],
                    "name": node["name"],
                    "display_name": node.get("display_name"),
                    "definition": node["definition"][:200] if node.get("definition") else "",
                    "category": node["category"],
                    "confidence": node.get("confidence", 0.5),
                    "tags": node.get("tags", []),
                    "version": node.get("version", 1),
                })
            
            return concepts

    async def get_centrality_scores(
        self,
        algorithm: str = "pagerank",
    ) -> Dict[str, float]:
        """Compute centrality scores for concepts."""
        if not self._initialized:
            await self.connect()
        
        if algorithm == "pagerank":
            query = """
            CALL gds.pageRank.stream('concept-graph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
        elif algorithm == "betweenness":
            query = """
            CALL gds.betweenness.stream('concept-graph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
        elif algorithm == "degree":
            query = """
            CALL gds.degree.stream('concept-graph')
            YIELD nodeId, score
            RETURN gds.util.asNode(nodeId).id AS id, score
            """
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        async with self._driver.session(database=self.config.database) as session:
            try:
                result = await session.run(query)
                scores = {}
                async for record in result:
                    scores[record["id"]] = record["score"]
                return scores
            except Exception as e:
                logger.warning(f"GDS not available, falling back to degree centrality: {e}")
                # Fallback to simple degree centrality
                return await self._compute_degree_centrality()

    async def _compute_degree_centrality(self) -> Dict[str, float]:
        """Compute simple degree centrality."""
        query = """
        MATCH (c:Concept)
        OPTIONAL MATCH (c)-[r]-()
        RETURN c.id AS id, COUNT(r) AS degree
        """
        
        async with self._driver.session(database=self.config.database) as session:
            result = await session.run(query)
            return {record["id"]: record["degree"] for record in await result.values()}

    async def detect_communities(
        self,
        algorithm: str = "louvain",
    ) -> Dict[str, int]:
        """Detect communities in the knowledge graph."""
        if not self._initialized:
            await self.connect()
        
        if algorithm == "louvain":
            query = """
            CALL gds.louvain.stream('concept-graph')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS id, communityId
            """
        elif algorithm == "label_propagation":
            query = """
            CALL gds.labelPropagation.stream('concept-graph')
            YIELD nodeId, communityId
            RETURN gds.util.asNode(nodeId).id AS id, communityId
            """
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        async with self._driver.session(database=self.config.database) as session:
            try:
                result = await session.run(query)
                communities = {}
                async for record in result:
                    communities[record["id"]] = record["communityId"]
                return communities
            except Exception as e:
                logger.warning(f"GDS community detection not available: {e}")
                return {}

    async def sync_from_postgresql(self, concepts: List[Concept], edges: List[KnowledgeEdge]) -> Dict[str, int]:
        """Sync concepts and edges from PostgreSQL to Neo4j."""
        if not self._initialized:
            await self.connect()
        
        concepts_added = 0
        edges_added = 0
        
        # Batch insert concepts
        for concept in concepts:
            try:
                await self.add_concept(concept)
                concepts_added += 1
            except Exception as e:
                logger.error(f"Failed to add concept {concept.name}: {e}")
        
        # Batch insert edges
        for edge in edges:
            try:
                await self.add_edge(edge)
                edges_added += 1
            except Exception as e:
                logger.error(f"Failed to add edge {edge.id}: {e}")
        
        return {"concepts_added": concepts_added, "edges_added": edges_added}


class Neo4jKnowledgeAgent(BaseResearchAgent):
    """
    Knowledge Agent with Neo4j backend.
    
    Extends the in-memory KnowledgeAgent with persistent Neo4j storage.
    """

    def __init__(
        self,
        neo4j_config: Neo4jConfig = None,
        config: "KnowledgeConfig" = None,
        event_bus: Optional[EventBus] = None,
    ):
        from forex_ai_agent.research.agents.knowledge import KnowledgeConfig
        super().__init__("knowledge", event_bus)
        self.knowledge_config = config or KnowledgeConfig()
        self.neo4j = Neo4jKnowledgeGraph(neo4j_config)
        
        # Local caches
        self.concepts: Dict[str, Concept] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        
        self.capabilities = [
            "concept_persistence",
            "knowledge_graph",
            "relationship_inference",
            "concept_versioning",
            "concept_query",
            "neo4j_sync",
        ]

    async def initialize(self) -> None:
        """Initialize Neo4j connection."""
        await self.neo4j.connect()
        logger.info("Neo4j Knowledge Agent initialized")

    async def shutdown(self) -> None:
        """Close Neo4j connection."""
        await self.neo4j.close()
        logger.info("Neo4j Knowledge Agent shutdown")

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
        elif phase == "NEO4J_SYNC":
            return await self.sync_to_neo4j()
        else:
            return {"error": f"Unknown phase: {phase}"}

    async def integrate_knowledge(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Integrate hypothesis results into knowledge graph."""
        # ... (existing integration logic from KnowledgeAgent)
        # Then sync to Neo4j
        sync_result = await self.sync_to_neo4j()
        
        return {
            "hypothesis_id": hypothesis.id if hasattr(hypothesis, "id") else "",
            "neo4j_sync": sync_result,
            "total_concepts": len(self.concepts),
            "total_edges": len(self.edges),
        }

    async def sync_to_neo4j(self) -> Dict[str, int]:
        """Sync local concepts and edges to Neo4j."""
        concepts = list(self.concepts.values())
        edges = list(self.edges.values())
        return await self.neo4j.sync_from_postgresql(concepts, edges)

    async def query_concepts(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Query the Neo4j knowledge graph."""
        query_type = input_data.get("query_type", "search")
        
        if query_type == "search":
            results = await self.neo4j.search_concepts(
                query=input_data.get("query", ""),
                category=input_data.get("category"),
                tags=input_data.get("tags", []),
                limit=input_data.get("limit", 20),
            )
            return {"results": results, "count": len(results)}
        
        elif query_type == "neighbors":
            return await self.neo4j.get_neighbors(
                input_data.get("concept_id"),
                depth=input_data.get("depth", 1),
                relationship_types=input_data.get("relationship_types"),
            )
        
        elif query_type == "path":
            return await self.neo4j.find_path(
                input_data.get("source_id"),
                input_data.get("target_id"),
                max_depth=input_data.get("max_depth", 5),
            )
        
        elif query_type == "subgraph":
            return await self.neo4j.get_subgraph(
                input_data.get("concept_id"),
                radius=input_data.get("radius", 2),
            )
        
        elif query_type == "centrality":
            return await self.neo4j.get_centrality_scores(
                input_data.get("algorithm", "pagerank")
            )
        
        elif query_type == "communities":
            return await self.neo4j.detect_communities(
                input_data.get("algorithm", "louvain")
            )
        
        else:
            return {"error": f"Unknown query type: {query_type}"}


def create_neo4j_knowledge_graph(config: Neo4jConfig = None) -> Neo4jKnowledgeGraph:
    """Factory function to create Neo4j knowledge graph."""
    return Neo4jKnowledgeGraph(config)


def create_neo4j_knowledge_agent(
    neo4j_config: Neo4jConfig = None,
    config: "KnowledgeConfig" = None,
    event_bus: Optional[EventBus] = None,
) -> Neo4jKnowledgeAgent:
    """Factory function to create Neo4j knowledge agent."""
    return Neo4jKnowledgeAgent(neo4j_config=neo4j_config, config=config, event_bus=event_bus)