"""Concept Discovery - Pattern mining, market condition clustering, and novel concept proposal."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.models import Concept, KnowledgeEdge, Claim
from forex_ai_agent.research.agents.knowledge import KnowledgeAgent


class PatternType(Enum):
    """Types of patterns that can be discovered."""
    PRICE_ACTION = "price_action"
    INDICATOR_CORRELATION = "indicator_correlation"
    REGIME_TRANSITION = "regime_transition"
    VOLATILITY_PATTERN = "volatility_pattern"
    VOLUME_PATTERN = "volume_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    CROSS_ASSET = "cross_asset"
    RISK_PATTERN = "risk_pattern"


@dataclass
class DiscoveredPattern:
    """A discovered pattern in market data."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    pattern_type: PatternType = PatternType.PRICE_ACTION
    name: str = ""
    description: str = ""
    
    # Pattern definition
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Statistics
    occurrences: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Validation
    in_sample_score: float = 0.0
    out_of_sample_score: float = 0.0
    statistical_significance: float = 0.0
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.utcnow)
    symbols_tested: List[str] = field(default_factory=list)
    timeframes_tested: List[str] = field(default_factory=list)
    regimes_tested: List[str] = field(default_factory=list)
    confidence: float = 0.0


class PatternMiner:
    """
    Mines patterns from market data and research results.
    
    Capabilities:
    - Price action pattern mining
    - Indicator correlation discovery
    - Regime transition detection
    - Volatility pattern recognition
    - Cross-asset relationship mining
    """

    def __init__(
        self,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        event_bus = None,
    ):
        self.knowledge_agent = knowledge_agent
        self.event_bus = event_bus
        self.discovered_patterns: Dict[str, DiscoveredPattern] = {}
        
    async def mine_patterns_from_signals(
        self,
        signals: List[Dict[str, Any]],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine patterns from trading signals and market data."""
        patterns = []
        
        # 1. Price action patterns
        pa_patterns = await self._mine_price_action_patterns(signals, market_data)
        patterns.extend(pa_patterns)
        
        # 2. Indicator correlations
        ic_patterns = await self._mine_indicator_correlations(signals, market_data)
        patterns.extend(ic_patterns)
        
        # 3. Regime transitions
        rt_patterns = await self._mine_regime_transitions(signals, market_data)
        patterns.extend(rt_patterns)
        
        # 4. Volatility patterns
        vol_patterns = await self._mine_volatility_patterns(signals, market_data)
        patterns.extend(vol_patterns)
        
        # 5. Temporal patterns
        temp_patterns = await self._mine_temporal_patterns(signals, market_data)
        patterns.extend(temp_patterns)
        
        # Store and return
        for pattern in patterns:
            self.discovered_patterns[pattern.id] = pattern
        
        return patterns

    async def _mine_price_action_patterns(
        self,
        signals: List[Dict],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine price action patterns from signals."""
        patterns = []
        
        # Group signals by symbol
        by_symbol = defaultdict(list)
        for s in signals:
            by_symbol[s.get("symbol", "UNKNOWN")].append(s)
        
        for symbol, sym_signals in by_symbol.items():
            if len(sym_signals) < 10:
                continue
            
            # Look for recurring entry patterns
            entry_patterns = self._cluster_entry_conditions(sym_signals)
            
            for cluster_id, cluster_signals in entry_patterns.items():
                if len(cluster_signals) < 5:
                    continue
                
                # Calculate statistics
                profits = [s.get("profit", 0) for s in cluster_signals]
                win_rate = sum(1 for p in profits if p > 0) / len(profits)
                avg_return = np.mean(profits) if profits else 0
                
                if win_rate > 0.5 and avg_return > 0:
                    pattern = DiscoveredPattern(
                        pattern_type=PatternType.PRICE_ACTION,
                        name=f"PriceAction_{symbol}_Cluster{cluster_id}",
                        description=f"Price action pattern for {symbol} with {len(cluster_signals)} occurrences",
                        conditions=self._extract_conditions(cluster_signals),
                        occurrences=len(cluster_signals),
                        win_rate=win_rate,
                        avg_return=avg_return,
                        symbols_tested=[symbol],
                        confidence=min(0.9, win_rate * 0.8 + 0.2),
                    )
                    patterns.append(pattern)
        
        return patterns

    def _cluster_entry_conditions(self, signals: List[Dict]) -> Dict[int, List[Dict]]:
        """Cluster signals by entry conditions."""
        # Simple clustering based on entry price relative to indicators
        clusters = defaultdict(list)
        
        for i, s in enumerate(signals):
            # Simple heuristic: cluster by RSI range at entry
            rsi = s.get("entry_rsi", 50)
            if rsi < 30:
                cluster = 0  # Oversold
            elif rsi > 70:
                cluster = 1  # Overbought
            else:
                cluster = 2  # Neutral
            clusters[cluster].append(s)
        
        return clusters

    def _extract_conditions(self, signals: List[Dict]) -> List[Dict[str, Any]]:
        """Extract common conditions from a cluster of signals."""
        conditions = []
        
        # Average indicator values
        numeric_fields = ["entry_rsi", "entry_macd", "entry_atr", "entry_adx"]
        for field in numeric_fields:
            values = [s.get(field, 0) for s in signals if field in s]
            if values:
                conditions.append({
                    "field": field,
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values) if len(values) > 1 else 0,
                })
        
        # Categorical fields
        cat_fields = ["direction", "setup_type", "session"]
        for field in cat_fields:
            values = [s.get(field, "") for s in signals if field in s]
            if values:
                from collections import Counter
                counts = Counter(values)
                most_common = counts.most_common(1)[0] if counts else ("", 0)
                conditions.append({
                    "field": field,
                    "most_common": most_common[0],
                    "frequency": most_common[1] / len(values),
                })
        
        return conditions

    async def _mine_indicator_correlations(
        self,
        signals: List[Dict],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine indicator correlation patterns."""
        patterns = []
        
        # Get indicator values from signals
        indicator_fields = ["entry_rsi", "entry_macd", "entry_atr", "entry_adx", "entry_cci"]
        
        # Calculate correlations
        for i, f1 in enumerate(indicator_fields):
            for f2 in indicator_fields[i+1:]:
                # Get paired values
                pairs = []
                for s in signals:
                    v1 = s.get(f1)
                    v2 = s.get(f2)
                    if v1 is not None and v2 is not None:
                        pairs.append((v1, v2))
                
                if len(pairs) >= 10:
                    v1_vals = [p[0] for p in pairs]
                    v2_vals = [p[1] for p in pairs]
                    corr = np.corrcoef(v1_vals, v2_vals)[0, 1]
                    
                    if abs(corr) > 0.7:
                        pattern = DiscoveredPattern(
                            pattern_type=PatternType.INDICATOR_CORRELATION,
                            name=f"Correlation_{f1}_{f2}",
                            description=f"Strong correlation ({corr:.2f}) between {f1} and {f2}",
                            conditions=[
                                {"field": f1, "field2": f2, "correlation": corr}
                            ],
                            parameters={"correlation": corr, "sample_size": len(pairs)},
                            occurrences=len(pairs),
                            confidence=min(0.9, abs(corr) + 0.1),
                        )
                        patterns.append(pattern)
        
        return patterns

    async def _mine_regime_transitions(
        self,
        signals: List[Dict],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine regime transition patterns."""
        patterns = []
        
        # Group signals by detected regime
        regime_signals = defaultdict(list)
        for s in signals:
            regime = s.get("regime", "unknown")
            regime_signals[regime].append(s)
        
        # Analyze transitions
        for regime, sigs in regime_signals.items():
            if len(sigs) < 10:
                continue
            
            profits = [s.get("profit", 0) for s in sigs]
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            
            if win_rate > 0.55:
                pattern = DiscoveredPattern(
                    pattern_type=PatternType.REGIME_TRANSITION,
                    name=f"Regime_{regime}",
                    description=f"Favorable regime {regime} with {len(sigs)} samples",
                    conditions=[{"regime": regime, "min_samples": len(sigs)}],
                    occurrences=len(sigs),
                    win_rate=win_rate,
                    avg_return=np.mean([s.get("profit", 0) for s in sigs]),
                    symbols_tested=list(set(s.get("symbol", "") for s in sigs)),
                    confidence=min(0.85, win_rate * 0.7 + 0.15),
                )
                patterns.append(pattern)
        
        return patterns

    async def _mine_volatility_patterns(
        self,
        signals: List[Dict],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine volatility-based patterns."""
        patterns = []
        
        # Analyze signals by ATR/volatility regime
        vol_signals = defaultdict(list)
        for s in signals:
            atr = s.get("entry_atr", 0)
            if atr > 0:
                # Categorize by ATR percentile
                vol_category = "high" if s.get("atr_percentile", 50) > 75 else "low" if s.get("atr_percentile", 50) < 25 else "normal"
                vol_signals[vol_category].append(s)
        
        for cat, sigs in vol_signals.items():
            if len(sigs) < 10:
                continue
            
            profits = [s.get("profit", 0) for s in sigs]
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            
            if win_rate != 0.5:
                pattern = DiscoveredPattern(
                    pattern_type=PatternType.VOLATILITY_PATTERN,
                    name=f"Volatility_{cat}",
                    description=f"Performance in {cat} volatility regime",
                    conditions=[{"volatility_regime": cat, "min_samples": len(sigs)}],
                    occurrences=len(sigs),
                    win_rate=win_rate,
                    avg_return=np.mean([s.get("profit", 0) for s in sigs]),
                    confidence=min(0.8, abs(win_rate - 0.5) * 2 + 0.3),
                )
                patterns.append(pattern)
        
        return patterns

    async def _mine_temporal_patterns(
        self,
        signals: List[Dict],
        market_data: Dict[str, Any],
    ) -> List[DiscoveredPattern]:
        """Mine temporal patterns (time-of-day, day-of-week, month)."""
        patterns = []
        
        # Time-of-day patterns
        hour_signals = defaultdict(list)
        for s in signals:
            entry_time = s.get("entry_time")
            if entry_time:
                hour = entry_time.hour if hasattr(entry_time, "hour") else int(str(entry_time).split(":")[0])
                hour_signals[hour].append(s)
        
        for hour, sigs in hour_signals.items():
            if len(sigs) < 10:
                continue
            
            profits = [s.get("profit", 0) for s in sigs]
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            
            if abs(win_rate - 0.5) > 0.1:
                pattern = DiscoveredPattern(
                    pattern_type=PatternType.TEMPORAL_PATTERN,
                    name=f"Hour_{hour}",
                    description=f"Performance during hour {hour}:00 UTC",
                    conditions=[{"hour": hour, "min_samples": len(sigs)}],
                    occurrences=len(sigs),
                    win_rate=win_rate,
                    avg_return=np.mean([s.get("profit", 0) for s in sigs]),
                    confidence=min(0.85, abs(win_rate - 0.5) * 3 + 0.2),
                )
                patterns.append(pattern)
        
        # Day of week
        dow_signals = defaultdict(list)
        for s in signals:
            entry_time = s.get("entry_time")
            if entry_time:
                dow = entry_time.weekday() if hasattr(entry_time, "weekday") else 0
                dow_signals[dow].append(s)
        
        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for dow, sigs in dow_signals.items():
            if len(sigs) < 10:
                continue
            
            profits = [s.get("profit", 0) for s in sigs]
            win_rate = sum(1 for p in profits if p > 0) / len(profits)
            
            if abs(win_rate - 0.5) > 0.05:
                pattern = DiscoveredPattern(
                    pattern_type=PatternType.TEMPORAL_PATTERN,
                    name=f"DOW_{dow_names[dow]}",
                    description=f"Performance on {dow_names[dow]}",
                    conditions=[{"day_of_week": dow, "min_samples": len(sigs)}],
                    occurrences=len(sigs),
                    win_rate=win_rate,
                    avg_return=np.mean([s.get("profit", 0) for s in sigs]),
                    confidence=min(0.8, abs(win_rate - 0.5) * 4 + 0.2),
                )
                patterns.append(pattern)
        
        return patterns

    def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        min_confidence: float = 0.0,
    ) -> List[DiscoveredPattern]:
        """Get discovered patterns with optional filtering."""
        patterns = list(self.discovered_patterns.values())
        
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type == pattern_type]
        
        if min_confidence > 0:
            patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        return sorted(patterns, key=lambda p: p.confidence, reverse=True)


class MarketConditionClusterer:
    """
    Clusters market conditions into regimes for analysis.
    """

    def __init__(self, n_clusters: int = 4, method: str = "kmeans"):
        self.n_clusters = n_clusters
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.cluster_centers = None

    def fit(self, features: np.ndarray) -> "MarketConditionClusterer":
        """Fit the clustering model."""
        if self.method == "kmeans":
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        elif self.method == "dbscan":
            self.model = DBSCAN(eps=0.5, min_samples=5)
        
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)
        
        if hasattr(self.model, 'cluster_centers_'):
            self.cluster_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
        
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        if self.model is None:
            raise ValueError("Model not fitted")
        scaled = self.scaler.transform(features)
        return self.model.predict(scaled)

    def get_cluster_profiles(self, features: np.ndarray, labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Get statistical profile of each cluster."""
        profiles = {}
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                cluster_features = features[mask]
                profiles[i] = {
                    "size": np.sum(mask),
                    "mean": np.mean(cluster_features, axis=0).tolist(),
                    "std": np.std(cluster_features, axis=0).tolist(),
                }
        return profiles


class ConceptProposer:
    """
    Proposes novel trading concepts based on discovered patterns and knowledge graph.
    """

    def __init__(
        self,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        pattern_miner: Optional[PatternMiner] = None,
    ):
        self.knowledge_agent = knowledge_agent
        self.pattern_miner = pattern_miner
        self.proposed_concepts: List[Dict[str, Any]] = []

    async def propose_concepts(
        self,
        patterns: List[Any],
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Propose new trading concepts from discovered patterns."""
        proposals = []
        
        # Filter high-confidence patterns
        strong_patterns = [p for p in patterns if p.confidence >= min_confidence]
        
        # 1. Combine complementary patterns
        combo_proposals = await self._combine_patterns(strong_patterns)
        proposals.extend(combo_proposals)
        
        # 2. Extrapolate to new symbols/timeframes
        extrapolation_proposals = await self._extrapolate_patterns(strong_patterns)
        proposals.extend(extrapolation_proposals)
        
        # 3. Invert losing patterns
        inversion_proposals = await self._invert_losing_patterns(strong_patterns)
        proposals.extend(inversion_proposals)
        
        # 4. Parameter optimization proposals
        param_proposals = await self._propose_parameter_optimizations(strong_patterns)
        proposals.extend(param_proposals)
        
        # 5. Regime-specific proposals
        regime_proposals = await self._propose_regime_specific(strong_patterns)
        proposals.extend(regime_proposals)
        
        # Score and rank proposals
        scored = self._score_proposals(proposals)
        
        # Store and return top proposals
        self.proposed_concepts = scored[:20]
        return self.proposed_concepts

    async def _combine_patterns(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Propose combinations of complementary patterns."""
        proposals = []
        
        # Group by type
        by_type = defaultdict(list)
        for p in patterns:
            by_type[p.pattern_type].append(p)
        
        # Combine different types
        type_pairs = [
            (PatternType.PRICE_ACTION, PatternType.VOLATILITY_PATTERN),
            (PatternType.PRICE_ACTION, PatternType.TEMPORAL_PATTERN),
            (PatternType.INDICATOR_CORRELATION, PatternType.REGIME_TRANSITION),
        ]
        
        for t1, t2 in type_pairs:
            if t1 in by_type and t2 in by_type:
                for p1 in by_type[t1][:3]:
                    for p2 in by_type[t2][:3]:
                        if p1.confidence > 0.7 and p2.confidence > 0.7:
                            proposals.append({
                                "type": "pattern_combination",
                                "name": f"Combo_{p1.name}_{p2.name}",
                                "description": f"Combine {p1.name} with {p2.name}",
                                "components": [p1.id, p2.id],
                                "expected_win_rate": (p1.win_rate + p2.win_rate) / 2,
                                "confidence": min(p1.confidence, p2.confidence) * 0.9,
                                "rationale": f"Complementary patterns: {p1.pattern_type.value} + {p2.pattern_type.value}",
                            })
        
        return proposals

    async def _extrapolate_patterns(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Extrapolate patterns to new symbols/timeframes."""
        proposals = []
        
        major_pairs = ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
        minor_pairs = ["EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "EUR/AUD"]
        timeframes = ["M15", "M30", "H1", "H4", "D1"]
        
        for pattern in patterns[:5]:
            if pattern.confidence < 0.7:
                continue
            
            # Extrapolate to other major pairs
            tested_symbols = getattr(pattern, 'symbols_tested', [])
            for symbol in major_pairs:
                if symbol not in tested_symbols:
                    proposals.append({
                        "type": "symbol_extrapolation",
                        "name": f"{pattern.name}_on_{symbol}",
                        "description": f"Apply {pattern.name} to {symbol}",
                        "source_pattern": pattern.id,
                        "target_symbol": symbol,
                        "expected_win_rate": pattern.win_rate * 0.9,
                        "confidence": pattern.confidence * 0.8,
                        "rationale": f"Pattern {pattern.name} showed {pattern.win_rate:.1%} win rate on {', '.join(pattern.symbols_tested)}",
                    })
            
            # Extrapolate to other timeframes
            tested_tfs = getattr(pattern, 'timeframes_tested', ['H1'])
            for tf in timeframes:
                if tf not in tested_tfs:
                    proposals.append({
                        "type": "timeframe_extrapolation",
                        "name": f"{pattern.name}_on_{tf}",
                        "description": f"Apply {pattern.name} on {tf} timeframe",
                        "source_pattern": pattern.id,
                        "target_timeframe": tf,
                        "expected_win_rate": pattern.win_rate * 0.95,
                        "confidence": pattern.confidence * 0.85,
                        "rationale": f"Test if pattern holds on {tf} timeframe",
                    })
        
        return proposals

    async def _invert_losing_patterns(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Propose inversions of losing patterns."""
        proposals = []
        
        # Find patterns with consistently low win rate
        for pattern in patterns:
            if pattern.win_rate < 0.4 and pattern.occurrences > 20:
                # Invert the pattern
                proposals.append({
                    "type": "pattern_inversion",
                    "name": f"Inverse_{pattern.name}",
                    "description": f"Invert {pattern.name} (original win rate: {pattern.win_rate:.1%})",
                    "source_pattern": pattern.id,
                    "inversion_type": "full_inversion",
                    "expected_win_rate": 1.0 - pattern.win_rate,
                    "confidence": (1.0 - pattern.win_rate) * pattern.confidence,
                    "rationale": f"Pattern has {pattern.win_rate:.1%} win rate over {pattern.occurrences} occurrences. Inverting may yield {(1-pattern.win_rate):.1%} win rate.",
                })
        
        return proposals

    async def _propose_parameter_optimizations(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Propose parameter optimizations for existing patterns."""
        proposals = []
        
        for pattern in patterns:
            if pattern.confidence > 0.6 and hasattr(pattern, 'parameters'):
                for param, config in pattern.parameters.items():
                    if isinstance(config, dict) and "min" in config and "max" in config:
                        current = config.get("default", (config["min"] + config["max"]) / 2)
                        min_v, max_v = config["min"], config["max"]
                        
                        # Propose testing extremes
                        for test_val in [min_v, (min_v + max_v) / 2, max_v]:
                            if test_val != current:
                                proposals.append({
                                    "type": "parameter_optimization",
                                    "name": f"Optimize_{pattern.name}_{pattern.parameters[param]}",
                                    "description": f"Test {param}={test_val} for {pattern.name}",
                                    "source_pattern": pattern.id,
                                    "parameter": param,
                                    "test_value": test_val,
                                    "current_value": current,
                                    "confidence": pattern.confidence * 0.8,
                                    "rationale": f"Test if {param}={test_val} improves {pattern.name}",
                                })
        
        return proposals

    async def _propose_regime_specific(self, patterns: List[Any]) -> List[Dict[str, Any]]:
        """Propose regime-specific variations of patterns."""
        proposals = []
        
        regimes = ["trending_up", "trending_down", "ranging", "high_vol", "low_vol"]
        
        for pattern in patterns:
            if pattern.confidence < 0.7:
                continue
            
            # Check if pattern already has regime info
            has_regime = any("regime" in c for c in getattr(pattern, 'conditions', []))
            
            if not has_regime:
                for regime in regimes:
                    proposals.append({
                        "type": "regime_specific",
                        "name": f"{pattern.name}_in_{regime}",
                        "description": f"Test {pattern.name} specifically in {regime} regime",
                        "source_pattern": pattern.id,
                        "target_regime": regime,
                        "expected_win_rate": pattern.win_rate,
                        "confidence": pattern.confidence * 0.8,
                        "rationale": f"Test if {pattern.name} performs differently in {regime} regime",
                    })
        
        return proposals

    def _score_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score and rank proposals."""
        for p in proposals:
            score = 0.0
            
            # Confidence weight
            score += p.get("confidence", 0) * 0.4
            
            # Expected win rate
            score += p.get("expected_win_rate", 0.5) * 0.3
            
            # Type bonus
            type_bonus = {
                "pattern_combination": 0.1,
                "symbol_extrapolation": 0.05,
                "timeframe_extrapolation": 0.05,
                "pattern_inversion": 0.1,
                "parameter_optimization": 0.05,
                "regime_specific": 0.1,
            }
            score += type_bonus.get(p.get("type", ""), 0)
            
            p["score"] = score
        
        # Sort by score descending
        proposals.sort(key=lambda p: p.get("score", 0), reverse=True)
        return proposals


# Factory function
async def create_concept_discovery(
    knowledge_agent: Optional[KnowledgeAgent] = None,
    event_bus=None,
) -> ConceptDiscovery:
    """Create a concept discovery system."""
    pattern_miner = PatternMiner(knowledge_agent=knowledge_agent, event_bus=event_bus)
    return ConceptDiscovery(
        knowledge_agent=knowledge_agent,
        pattern_miner=pattern_miner,
    )


class ConceptDiscovery:
    """Concept Discovery System - Discovers novel trading concepts from patterns and market data."""
    
    def __init__(
        self,
        knowledge_agent: Optional[KnowledgeAgent] = None,
        pattern_miner: Optional[PatternMiner] = None,
    ):
        self.knowledge_agent = knowledge_agent
        self.pattern_miner = pattern_miner or PatternMiner()
        self.discovered_concepts = []
    
    async def discover_concepts(
        self,
        signals: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        min_confidence: float = 0.7,
    ) -> List[DiscoveredPattern]:
        """Discover new trading concepts from signals and market data."""
        return await self.pattern_miner.mine_patterns_from_signals(signals, market_data)
    
    def get_discovered_concepts(self) -> List[DiscoveredPattern]:
        """Get all discovered concepts."""
        return self.discovered_concepts
    
    async def propose_new_concepts(
        self,
        patterns: List[DiscoveredPattern],
        min_confidence: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Propose new concepts based on discovered patterns."""
        proposer = ConceptProposer(pattern_miner=self.pattern_miner, knowledge_agent=self.knowledge_agent)
        return await proposer.propose_concepts(patterns, min_confidence)


# Factory function