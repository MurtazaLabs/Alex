"""Base classes for the signal rule engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum


class SignalStatus(str, Enum):
    """Signal lifecycle states."""

    NO_TRADE = "NO_TRADE"
    WATCH = "WATCH"
    SETUP_FORMING = "SETUP_FORMING"
    TRIGGERED = "TRIGGERED"
    FILLED = "FILLED"
    INVALIDATED = "INVALIDATED"
    EXPIRED = "EXPIRED"


class Direction(str, Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class SignalRule:
    """A single deterministic signal rule."""

    name: str
    condition: Callable[["MarketContext"], bool]
    weight: float = 1.0
    required: bool = False
    timeframe: str = "H1"
    description: str = ""
    category: str = "GENERAL"

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"Rule weight must be 0-1, got {self.weight}")


@dataclass
class EvidenceItem:
    """A piece of evidence for/against a signal."""

    category: str
    timeframe: str
    description: str
    value: Any
    strength: float
    supporting: bool = True


@dataclass
class SignalEvidence:
    """Collected evidence for a signal decision."""

    supporting: List[EvidenceItem] = field(default_factory=list)
    contradicting: List[EvidenceItem] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)

    def add_supporting(self, item: EvidenceItem) -> None:
        self.supporting.append(item)

    def add_contradicting(self, item: EvidenceItem) -> None:
        self.contradicting.append(item)

    def add_missing(self, description: str) -> None:
        self.missing.append(description)


@dataclass
class SignalScore:
    """Result of rule evaluation."""

    total_weight: float = 0.0
    passed_weight: float = 0.0
    required_passed: bool = True
    failed_rules: List[str] = field(default_factory=list)
    passed_rules: List[str] = field(default_factory=list)

    @property
    def score_pct(self) -> float:
        if self.total_weight == 0:
            return 0.0
        return self.passed_weight / self.total_weight


class RuleEngine:
    """Evaluates a set of rules against a market context."""

    def __init__(self, rules: List[SignalRule]):
        self.rules = rules
        self._rules_by_category: Dict[str, List[SignalRule]] = {}
        for rule in rules:
            self._rules_by_category.setdefault(rule.category, []).append(rule)

    def evaluate(
        self,
        context: "MarketContext",
        min_score_pct: float = 0.70,
        min_timeframes: int = 3,
    ) -> SignalScore:
        """Evaluate all rules and return a score."""
        score = SignalScore()

        for rule in self.rules:
            score.total_weight += rule.weight
            try:
                passed = rule.condition(context)
            except Exception as e:
                passed = False
                context.evidence.add_contradicting(EvidenceItem(
                    category=rule.category,
                    timeframe=rule.timeframe,
                    description=f"Rule evaluation error: {rule.name}: {e}",
                    value=str(e),
                    strength=0.0,
                    supporting=False,
                ))

            if passed:
                score.passed_weight += rule.weight
                score.passed_rules.append(rule.name)
                context.evidence.add_supporting(EvidenceItem(
                    category=rule.category,
                    timeframe=rule.timeframe,
                    description=rule.description or rule.name,
                    value=True,
                    strength=rule.weight,
                    supporting=True,
                ))
            else:
                score.failed_rules.append(rule.name)
                if rule.required:
                    score.required_passed = False
                context.evidence.add_contradicting(EvidenceItem(
                    category=rule.category,
                    timeframe=rule.timeframe,
                    description=f"FAILED: {rule.description or rule.name}",
                    value=False,
                    strength=rule.weight,
                    supporting=False,
                ))

        return score

    def get_rules_by_category(self, category: str) -> List[SignalRule]:
        return self._rules_by_category.get(category, [])


@dataclass
class MarketContext:
    """All data needed for signal evaluation at one moment."""

    pair: str
    timeframe: str
    current_price: float
    spread_pips: float
    atr_pips: float
    session: str
    news_state: str
    direction: "Direction" = Direction.NONE
    htf_context: Dict[str, "HTFContext"] = field(default_factory=dict)
    indicators: "IndicatorsSnapshot" = field(default_factory=lambda: IndicatorsSnapshot())
    structure: "StructureSnapshot" = field(default_factory=lambda: StructureSnapshot(trend="NEUTRAL"))

    evidence: SignalEvidence = field(default_factory=SignalEvidence)

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = SignalEvidence()


@dataclass
class HTFContext:
    """Higher-timeframe context snapshot."""

    timeframe: str
    trend: str
    structure: Optional["StructureSnapshot"] = None
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None


@dataclass
class IndicatorsSnapshot:
    """Technical indicator values for a timeframe."""

    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    atr: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None


@dataclass
class StructureSnapshot:
    """Market structure snapshot for a timeframe."""

    trend: str
    swing_highs: List[Dict] = field(default_factory=list)
    swing_lows: List[Dict] = field(default_factory=list)
    last_bos: Optional[Dict] = None
    last_choch: Optional[Dict] = None
    key_support: Optional[float] = None
    key_resistance: Optional[float] = None
    liquidity_sweeps: List[Dict] = field(default_factory=list)
    consolidation: Optional[Dict] = None
    retests: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)

    @property
    def last_swing_high(self) -> Optional[Dict]:
        return self.swing_highs[-1] if self.swing_highs else None

    @property
    def last_swing_low(self) -> Optional[Dict]:
        return self.swing_lows[-1] if self.swing_lows else None