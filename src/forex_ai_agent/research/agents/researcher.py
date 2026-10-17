"""Researcher Agent - Literature review and source analysis."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    AgentType,
    Claim,
    ClaimType,
    Hypothesis,
    ResearchPhase,
    Source,
    SourceType,
)


@dataclass
class ResearcherConfig:
    """Configuration for researcher agent."""
    max_sources_per_query: int = 20
    min_credibility_threshold: float = 0.3
    enable_web_search: bool = False
    enable_academic_search: bool = True
    credibility_weights: Dict[str, float] = field(default_factory=lambda: {
        "peer_reviewed": 0.3,
        "citation_count": 0.2,
        "recency": 0.2,
        "author_reputation": 0.15,
        "venue_quality": 0.15,
    })


class ResearcherAgent(BaseResearchAgent):
    """
    Researcher Agent - Conducts literature review and extracts claims.
    
    Capabilities:
    - Source search (academic, web, internal)
    - Credibility scoring
    - Claim extraction with citations
    - Conflict detection between sources
    - Gap analysis
    """
    
    def __init__(
        self,
        config: ResearcherConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("researcher", event_bus)
        self.researcher_config = config or ResearcherConfig()
        self.source_cache: Dict[str, Source] = {}
        self.claim_cache: Dict[str, List[Claim]] = {}
    
    def _register_capabilities(self):
        self.capabilities = [
            "source_search",
            "credibility_scoring",
            "claim_extraction",
            "conflict_detection",
            "gap_analysis",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute literature review for a hypothesis."""
        phase = input_data.get("phase", "LITERATURE_REVIEW")
        
        if phase == "LITERATURE_REVIEW":
            return await self.execute_literature_review(hypothesis, input_data)
        elif phase == "HYPOTHESIS_FORMATION":
            return await self.execute_hypothesis_formation(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def execute_literature_review(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Conduct literature review for hypothesis."""
        query = self._generate_search_query(hypothesis)
        
        # Search for sources
        sources = await self._search_sources(query)
        
        # Score credibility
        scored_sources = await self._score_sources(sources)
        
        # Extract claims
        claims = await self._extract_claims(scored_sources, hypothesis)
        
        # Detect conflicts
        conflicts = self._detect_conflicts(claims)
        
        # Identify gaps
        gaps = self._identify_gaps(claims, hypothesis)
        
        return {
            "query": query,
            "sources_found": len(sources),
            "sources_after_filtering": len(scored_sources),
            "claims_extracted": len(claims),
            "conflicts_detected": len(conflicts),
            "gaps_identified": len(gaps),
            "sources": [s.__dict__ for s in scored_sources],
            "claims": claims,  # Keep as Claim objects
            "conflicts": conflicts,
            "gaps": gaps,
        }
    
    async def execute_hypothesis_formation(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assist in hypothesis formation based on literature."""
        lit_review = input_data.get("literature_review_output", {})
        claims = lit_review.get("claims", [])
        
        # Analyze claims for hypothesis formation
        patterns = self._identify_patterns(claims)
        opportunities = self._identify_opportunities(claims)
        contradictions = self._find_contradictions(claims)
        
        return {
            "patterns": patterns,
            "opportunities": opportunities,
            "contradictions": contradictions,
            "suggested_hypotheses": self._suggest_hypotheses(patterns, opportunities),
        }
    
    def _generate_search_query(self, hypothesis: Any) -> str:
        """Generate search query from hypothesis."""
        # Extract key terms from hypothesis
        terms = []
        if hasattr(hypothesis, "description"):
            terms.append(hypothesis.description)
        if hasattr(hypothesis, "research_question"):
            terms.append(hypothesis.research_question)
        if hasattr(hypothesis, "variables"):
            terms.extend(hypothesis.variables.keys())
        
        # Build query
        query = " ".join(terms[:5])  # Limit to top 5 terms
        return query
    
    async def _search_sources(self, query: str) -> List[Source]:
        """Search for relevant sources."""
        # In a real implementation, this would call APIs like:
        # - Google Scholar / Semantic Scholar
        # - arXiv
        # - PubMed
        # - Internal knowledge base
        # - Web search (if enabled)
        
        # For now, return mock sources
        return self._generate_mock_sources(query)
    
    def _generate_mock_sources(self, query: str) -> List[Source]:
        """Generate mock sources for testing."""
        sources = []
        
        # Academic papers
        topics = ["momentum", "mean reversion", "volatility", "liquidity", "market structure"]
        for i, topic in enumerate(topics):
            source = Source(
                id=f"src_{i}",
                type=SourceType.ACADEMIC_PAPER,
                title=f"An Empirical Study of {topic.title()} in Forex Markets",
                authors=["Smith, J.", "Doe, A."],
                publication_date=datetime(2023, 1, 15),
                url=f"https://example.com/paper_{i}",
                doi=f"10.1000/forex.{i}",
                abstract=f"This paper investigates {topic} in foreign exchange markets using high-frequency data...",
                credibility_score=0.8,
                tags=[topic, "forex", "empirical"],
            )
            sources.append(source)
        
        # Blog posts / practitioner articles
        for i, topic in enumerate(["risk management", "position sizing", "market regime"]):
            source = Source(
                id=f"src_blog_{i}",
                type=SourceType.BLOG_POST,
                title=f"Practical Guide to {topic.title()} in Trading",
                authors=["Trader, T."],
                publication_date=datetime(2024, 3, 1),
                url=f"https://tradingblog.example.com/{topic}",
                abstract=f"In this article, we explore practical approaches to {topic}...",
                credibility_score=0.4,
                tags=[topic, "practical", "trading"],
            )
            sources.append(source)
        
        return sources
    
    async def _score_sources(self, sources: List[Source]) -> List[Source]:
        """Score and filter sources by credibility."""
        scored = []
        
        for source in sources:
            # Calculate credibility score
            score = self._calculate_credibility(source)
            source.credibility_score = score
            
            if score >= self.researcher_config.min_credibility_threshold:
                scored.append(source)
        
        # Sort by credibility
        scored.sort(key=lambda s: s.credibility_score, reverse=True)
        
        return scored[:self.researcher_config.max_sources_per_query]
    
    def _calculate_credibility(self, source: Source) -> float:
        """Calculate credibility score for a source."""
        weights = self.researcher_config.credibility_weights
        score = 0.0
        
        # Peer reviewed
        if source.type == SourceType.ACADEMIC_PAPER:
            score += weights["peer_reviewed"] * 1.0
        elif source.type == SourceType.DOCUMENTATION:
            score += weights["peer_reviewed"] * 0.5
        else:
            score += weights["peer_reviewed"] * 0.1
        
        # Citation count (mock)
        citation_count = hash(source.doi or source.url or "") % 100
        score += weights["citation_count"] * min(citation_count / 100, 1.0)
        
        # Recency
        if source.publication_date:
            years_old = (datetime.utcnow() - source.publication_date).days / 365
            recency_score = max(0, 1 - years_old / 10)
            score += weights["recency"] * recency_score
        
        # Author reputation (mock)
        author_reputation = len(source.authors) * 0.1
        score += weights["author_reputation"] * min(author_reputation, 1.0)
        
        # Venue quality (mock)
        venue_quality = 0.7 if source.type == SourceType.ACADEMIC_PAPER else 0.3
        score += weights["venue_quality"] * venue_quality
        
        return min(max(score, 0.0), 1.0)
    
    async def _extract_claims(
        self,
        sources: List[Source],
        hypothesis: Any,
    ) -> List[Claim]:
        """Extract claims from sources."""
        claims = []
        
        for source in sources:
            # In a real implementation, this would use NLP to extract claims
            # For now, generate mock claims based on source content
            source_claims = self._extract_claims_from_source(source, hypothesis)
            claims.extend(source_claims)
        
        return claims
    
    def _extract_claims_from_source(self, source: Source, hypothesis: Any) -> List[Claim]:
        """Extract claims from a single source."""
        claims = []
        
        # Mock claim extraction based on source tags
        claim_templates = {
            "momentum": [
                ("Momentum strategies exhibit positive returns over 3-12 month horizons", 
                 ClaimType.EMPIRICAL_FINDING, 0.8),
                ("Momentum crashes occur during market reversals", 
                 ClaimType.RISK_FACTOR, 0.7),
            ],
            "mean reversion": [
                ("Mean reversion strategies work best in ranging markets",
                 ClaimType.MARKET_OBSERVATION, 0.75),
                ("Half-life of mean reversion is 5-20 days in forex",
                 ClaimType.PARAMETER_VALUE, 0.7),
            ],
            "volatility": [
                ("Volatility clustering is present in forex returns",
                 ClaimType.EMPIRICAL_FINDING, 0.85),
                ("GARCH models outperform constant volatility for risk management",
                 ClaimType.METHODOLOGY, 0.7),
            ],
            "liquidity": [
                ("Liquidity provision earns premium in forex markets",
                 ClaimType.STRATEGY_RULE, 0.65),
            ],
            "market structure": [
                ("Market structure breaks signal trend continuation",
                 ClaimType.STRATEGY_RULE, 0.8),
            ],
        }
        
        for tag in source.tags:
            if tag in claim_templates:
                for text, ctype, confidence in claim_templates[tag]:
                    claim = Claim(
                        id=str(uuid4())[:8],
                        text=text,
                        claim_type=ctype,
                        source_id=source.id,
                        confidence=confidence * source.credibility_score,
                    )
                    claims.append(claim)
        
        return claims
    
    def _detect_conflicts(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Detect conflicting claims."""
        conflicts = []
        
        # Group claims by type
        by_type = {}
        for claim in claims:
            by_type.setdefault(claim.claim_type, []).append(claim)
        
        # Check for contradictions within each type
        for ctype, type_claims in by_type.items():
            if len(type_claims) < 2:
                continue
            
            # Simple keyword-based conflict detection
            for i, c1 in enumerate(type_claims):
                for c2 in type_claims[i+1:]:
                    if self._claims_contradict(c1.text, c2.text):
                        conflicts.append({
                            "claim_1": c1.__dict__,
                            "claim_2": c2.__dict__,
                            "conflict_type": "direct_contradiction",
                            "severity": "high",
                        })
        
        return conflicts
    
    def _claims_contradict(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection."""
        # Look for negation patterns
        negations = ["not", "no", "never", "none", "neither", "nor"]
        pos1 = any(n in text1.lower() for n in negations)
        pos2 = any(n in text2.lower() for n in negations)
        
        # If one has negation and other doesn't, might be contradiction
        if pos1 != pos2:
            # Check if they share key terms
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            common = words1 & words2
            return len(common) > 3
        
        return False
    
    def _identify_gaps(self, claims: List[Claim], hypothesis: Any) -> List[Dict[str, Any]]:
        """Identify gaps in literature."""
        gaps = []
        
        # Check if key aspects of hypothesis are covered
        covered_topics = set()
        for claim in claims:
            covered_topics.update(claim.text.lower().split())
        
        # Expected topics from hypothesis
        if hasattr(hypothesis, "variables"):
            for var in hypothesis.variables.keys():
                if var.lower() not in " ".join(covered_topics):
                    gaps.append({
                        "topic": var,
                        "type": "missing_variable",
                        "description": f"No claims found covering variable: {var}",
                        "priority": "high",
                    })
        
        return gaps
    
    def _identify_patterns(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Identify patterns in claims."""
        patterns = []
        
        # Group by claim type
        by_type = {}
        for claim in claims:
            by_type.setdefault(claim.claim_type.value, []).append(claim)
        
        for ctype, type_claims in by_type.items():
            if len(type_claims) >= 2:
                avg_confidence = sum(c.confidence for c in type_claims) / len(type_claims)
                patterns.append({
                    "type": ctype,
                    "count": len(type_claims),
                    "avg_confidence": avg_confidence,
                    "summary": f"Found {len(type_claims)} claims of type {ctype}",
                })
        
        return patterns
    
    def _identify_opportunities(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Identify research opportunities."""
        opportunities = []
        
        # Look for high-confidence claims that haven't been fully explored
        for claim in claims:
            if claim.confidence > 0.7 and claim.claim_type in [
                ClaimType.EMPIRICAL_FINDING,
                ClaimType.MARKET_OBSERVATION,
            ]:
                opportunities.append({
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "type": "high_confidence_finding",
                    "potential": "Could form basis for new strategy",
                })
        
        return opportunities
    
    def _find_contradictions(self, claims: List[Claim]) -> List[Dict[str, Any]]:
        """Find contradictions in claims."""
        return self._detect_conflicts(claims)
    
    def _suggest_hypotheses(self, patterns: List[Dict], opportunities: List[Dict]) -> List[Dict[str, Any]]:
        """Suggest hypotheses based on patterns and opportunities."""
        hypotheses = []
        
        for opp in opportunities[:3]:  # Top 3
            hypotheses.append({
                "title": f"Investigate: {opp['claim_text'][:50]}...",
                "description": f"Based on high-confidence finding: {opp['claim_text']}",
                "type": "exploratory",
                "source_claim_id": opp.get("claim_id"),
            })
        
        return hypotheses


def create_researcher_agent(
    config: ResearcherConfig = None,
    event_bus: Optional[EventBus] = None,
) -> ResearcherAgent:
    """Create a researcher agent."""
    return ResearcherAgent(config=config, event_bus=event_bus)