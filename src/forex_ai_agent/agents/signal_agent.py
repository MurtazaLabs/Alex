"""Signal Agent - Evaluates deterministic conditions and generates trade setups.

Responsibilities:
- Rule engine evaluation (trend, structure, momentum, volatility, session)
- Signal state machine management
- Evidence collection
- Entry/SL/TP calculation when triggered
- Invalidation condition generation
- Full audit trail for every decision

The LLM never generates any of these numbers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..analysis.signal_rules.base_rule import (
    EvidenceItem,
    MarketContext,
    SignalEvidence,
    SignalStatus,
    SignalScore,
    Direction,
)
from ..analysis.signal_rules.engine import RuleEngine
from ..analysis.signal_rules.trend_rules import make_trend_rules
from ..analysis.signal_rules.structure_rules import make_structure_rules
from ..analysis.signal_rules.momentum_rules import make_momentum_rules
from ..analysis.signal_rules.volatility_rules import make_volatility_rules, make_session_rules
from ..config import get_settings
from ..core.event_bus import Event, EventBus
from ..core.permission_manager import PermissionManager

logger = logging.getLogger(__name__)


class SignalOutcome(str, Enum):
    """Result of signal evaluation."""

    NO_TRADE = "NO_TRADE"
    WATCH = "WATCH"
    SETUP_FORMING = "SETUP_FORMING"
    TRIGGERED = "TRIGGERED"
    INVALIDATED = "INVALIDATED"


@dataclass
class SignalCandidate:
    """A signal ready for risk validation."""

    pair: str
    direction: Direction
    timeframe: str
    setup_type: str
    entry_zone_low: float
    entry_zone_high: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    take_profit_3: Optional[float] = None
    risk_reward: float = 0.0
    strategy_version: str = "v1.0"
    detected_at: datetime = field(default_factory=datetime.utcnow)
    explanation: Optional[str] = None
    evidence: SignalEvidence = field(default_factory=SignalEvidence)
    invalidation_condition: str = ""
    score: SignalScore = field(default_factory=SignalScore)


class SignalAgent:
    """Evaluates deterministic conditions and generates trade setups.

    Pipeline:
    1. Evaluate all rule categories (trend, structure, momentum, volatility, session)
    2. Compute SignalScore and collect evidence
    3. If conditions met → create SignalCandidate with entry/SL/TP
    4. Emit events via the event bus
    """

    CATEGORY_ORDER = ["TREND", "STRUCTURE", "MOMENTUM", "VOLATILITY", "SESSION"]

    # Minimum weighted rule agreement for a setup to form.
    MIN_SCORE_PCT = 0.70
    # More than this many contradicting evidence items blocks a setup.
    MAX_CONTRADICTIONS = 2

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        permission_manager: Optional[PermissionManager] = None,
    ):
        self.event_bus = event_bus
        self.permission_manager = permission_manager
        self._rule_engine = self._build_rule_engine()
        self._signal_states: Dict[str, SignalStatus] = {}

    # ------------------------------------------------------------------
    # Rule Engine Setup
    # ------------------------------------------------------------------

    def _build_rule_engine(self) -> Dict[str, Any]:
        """Build the composite rule engine plus one engine per category."""
        builders = [
            make_trend_rules,
            make_structure_rules,
            make_momentum_rules,
            make_volatility_rules,
            make_session_rules,
        ]
        all_rules = []
        by_category: Dict[str, List[Any]] = {}
        for builder_fn in builders:
            for rule in builder_fn():
                all_rules.append(rule)
                by_category.setdefault(rule.category, []).append(rule)

        engines: Dict[str, RuleEngine] = {
            cat: RuleEngine(rules) for cat, rules in by_category.items()
        }
        return {"composite": RuleEngine(all_rules), "by_category": engines}

    # ------------------------------------------------------------------
    # Evaluation pipeline
    # ------------------------------------------------------------------

    async def evaluate_signal(
        self,
        pair: str,
        direction: Direction,
        timeframe: str,
        ctx: MarketContext,
    ) -> SignalCandidate:
        """Run the full signal evaluation pipeline and return a SignalCandidate."""

        signal_id = f"{pair}_{timeframe}_{direction.value}"

        # Step 1: Evaluate all rule categories (score + per-category report).
        score = self._rule_engine["composite"].evaluate(ctx, min_score_pct=self.MIN_SCORE_PCT)
        category_scores = self._category_scores(ctx)

        # Step 2: Failed rules are missing information, NOT contradictions.
        self._fill_missing_evidence(ctx.evidence, score)

        # Step 3: Determine signal state; build a candidate when a setup forms.
        state, candidate = self._determine_state(ctx, score)
        if candidate is not None:
            candidate.score = score

        # Step 4: Entry/SL/TP when a setup formed.
        if candidate is not None:
            candidate = self._calc_entry_sl_tp(candidate, ctx)

        # Step 5: Store state.
        self._signal_states[signal_id] = state

        # Step 6: Emit events (never crashes the pipeline).
        await self._emit_events(signal_id, state, candidate, ctx)

        if candidate is None:
            candidate = SignalCandidate(
                pair=pair,
                direction=direction,
                timeframe=timeframe,
                setup_type="NO_TRADE",
                entry_zone_low=0.0,
                entry_zone_high=0.0,
                stop_loss=0.0,
                take_profit_1=0.0,
                risk_reward=0.0,
                strategy_version=self._version(),
                invalidation_condition=self._gen_invalidation(state, ctx),
                evidence=ctx.evidence,
                score=score,
            )
        candidate.explanation = (
            f"Score {score.score_pct:.0%} "
            f"({score.passed_weight:.1f}/{score.total_weight:.1f}); "
            f"categories: "
            + ", ".join(f"{c} {category_scores.get(c, 0):.0%}" for c in self.CATEGORY_ORDER)
        )
        return candidate

    def _category_scores(self, ctx: MarketContext) -> Dict[str, float]:
        """Per-category agreement percentages for display (no evidence pollution)."""
        out: Dict[str, float] = {}
        for cat in self.CATEGORY_ORDER:
            engine = self._rule_engine["by_category"].get(cat)
            if engine is None:
                continue
            s = engine.evaluate(ctx, min_score_pct=self.MIN_SCORE_PCT)
            out[cat] = s.score_pct
        return out

    def _fill_missing_evidence(self, evidence: SignalEvidence, score: SignalScore) -> None:
        """Un-passed rules are recorded as missing information, not contradictions."""
        for rule_name in score.failed_rules:
            evidence.add_missing(f"Rule '{rule_name}' did not pass — check data quality or context")
        if score.score_pct < 1.0:
            evidence.add_missing(
                "Incomplete timeframe coverage — verify all required timeframes are available"
            )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _determine_state(
        self, ctx: MarketContext, score: SignalScore
    ) -> Tuple[SignalStatus, Optional[SignalCandidate]]:
        """Determine the signal state and optionally create a candidate."""

        if not score.required_passed:
            return SignalStatus.WATCH, None

        if score.score_pct < self.MIN_SCORE_PCT:
            return SignalStatus.WATCH, None

        # Only genuine contradictions block; rule failures live in `missing`.
        if len(ctx.evidence.contradicting) > self.MAX_CONTRADICTIONS:
            return SignalStatus.WATCH, None

        structure = ctx.structure
        # A structured setup needs a key level to anchor entry/SL/TP.
        if not (structure and (structure.key_support or structure.key_resistance)):
            return SignalStatus.WATCH, None

        candidate = SignalCandidate(
            pair=ctx.pair,
            direction=ctx.direction,
            timeframe=ctx.timeframe,
            setup_type="STRUCTURED_SETUP",
            entry_zone_low=0.0,
            entry_zone_high=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            risk_reward=0.0,
            invalidation_condition=self._gen_invalidation(SignalStatus.SETUP_FORMING, ctx),
            evidence=ctx.evidence,
        )
        return SignalStatus.SETUP_FORMING, candidate

    # ------------------------------------------------------------------
    # Entry / SL / TP calculation
    # ------------------------------------------------------------------

    def _calc_entry_sl_tp(
        self, candidate: SignalCandidate, ctx: MarketContext
    ) -> SignalCandidate:
        """Calculate entry zone, stop loss, and take profit levels.

        Levels anchor to structure when available and fall back to an ATR
        bracket otherwise; a final sanity pass guarantees a coherent bracket.
        """
        direction = ctx.direction
        price = ctx.current_price
        structure = ctx.structure
        atr_pips = ctx.atr_pips or 0.0

        # JPY-quoted pairs have a different pip size.
        pip = 0.01 if ctx.pair.upper().endswith("JPY") else 0.0001
        atr_price = atr_pips * pip
        # TP1 must clear the configured minimum R:R or RiskAgent will always reject.
        rr_floor = max(1.5, float(get_settings().min_risk_reward))

        if direction == Direction.LONG:
            support = structure.key_support if structure else None
            resistance = structure.key_resistance if structure else None

            if support is not None:
                entry_low = max(price * 0.999, support * 0.998)
                entry_high = max(price * 1.001, entry_low + 2 * pip)
            else:
                entry_low = price * 0.998
                entry_high = price * 1.002

            if support is not None:
                sl = support * 0.995
            else:
                sl = min(entry_low - 1.5 * atr_price, entry_low * 0.999)

            risk = max(entry_low - sl, pip)
            tp1 = entry_low + risk * rr_floor
            tp2 = resistance if (resistance and resistance > tp1) else entry_low + risk * (rr_floor + 0.5)
            tp3 = entry_low + risk * (rr_floor + 1.5) if resistance else None

            candidate.invalidation_condition = (
                f"H1 candle closes below {sl:.5f} (structural support broken)"
            )
        else:  # SHORT
            resistance = structure.key_resistance if structure else None
            support = structure.key_support if structure else None

            if resistance is not None:
                entry_high = min(price * 1.001, resistance * 1.002)
                entry_low = min(price * 0.999, entry_high - 2 * pip)
            else:
                entry_high = price * 1.002
                entry_low = price * 0.998

            if resistance is not None:
                sl = resistance * 1.005
            else:
                sl = max(entry_high + 1.5 * atr_price, entry_high * 1.001)

            risk = max(sl - entry_high, pip)
            tp1 = entry_high - risk * rr_floor
            tp2 = support if (support and support < tp1) else entry_high - risk * (rr_floor + 0.5)
            tp3 = entry_high - risk * (rr_floor + 1.5) if support else None

            candidate.invalidation_condition = (
                f"H1 candle closes above {sl:.5f} (structural resistance broken)"
            )

        candidate.entry_zone_low = round(entry_low, 5)
        candidate.entry_zone_high = round(entry_high, 5)
        candidate.stop_loss = round(sl, 5)
        candidate.take_profit_1 = round(tp1, 5)
        candidate.take_profit_2 = round(tp2, 5) if tp2 else None
        candidate.take_profit_3 = round(tp3, 5) if tp3 else None

        self._enforce_sane_bracket(candidate)

        entry_ref = candidate.entry_zone_low if direction == Direction.LONG else candidate.entry_zone_high
        risk = abs(entry_ref - candidate.stop_loss)
        if risk > 0:
            candidate.risk_reward = round(abs(candidate.take_profit_1 - entry_ref) / risk, 2)

        candidate.strategy_version = self._version()
        return candidate

    @staticmethod
    def _enforce_sane_bracket(candidate: SignalCandidate) -> None:
        """Guarantee LONG: sl < entry_low <= entry_high < tp1 (mirrored for SHORT)."""
        lo, hi = candidate.entry_zone_low, candidate.entry_zone_high
        sl, tp1 = candidate.stop_loss, candidate.take_profit_1
        if lo > hi:
            lo, hi = hi, lo
        if candidate.direction == Direction.LONG:
            if sl >= lo:
                sl = lo - abs(hi - lo) - abs(hi - lo) * 0.5  # 1.5x zone width
            if tp1 <= hi:
                tp1 = hi + (hi - sl) * 1.5
        else:
            if sl <= hi:
                sl = hi + abs(hi - lo) * 1.5
            if tp1 >= lo:
                tp1 = lo - (sl - lo) * 1.5
        candidate.entry_zone_low = round(lo, 5)
        candidate.entry_zone_high = round(hi, 5)
        candidate.stop_loss = round(sl, 5)
        candidate.take_profit_1 = round(tp1, 5)

    def _version(self) -> str:
        return "v1.0"

    # ------------------------------------------------------------------
    # Invalidation condition
    # ------------------------------------------------------------------

    def _gen_invalidation(self, state: SignalStatus, ctx: MarketContext) -> str:
        """Generate the invalidation condition string."""
        structure = ctx.structure
        if structure and structure.key_support and ctx.direction == Direction.LONG:
            return f"H1 candle closes below {structure.key_support:.5f} (structural support broken)"
        if structure and structure.key_resistance and ctx.direction == Direction.SHORT:
            return f"H1 candle closes above {structure.key_resistance:.5f} (structural resistance broken)"
        if structure and structure.last_bos:
            bos = structure.last_bos
            level = bos.get("level", "N/A") if isinstance(bos, dict) else getattr(bos, "level", "N/A")
            try:
                return f"H1 close beyond {float(level):.5f} invalidates the setup"
            except (TypeError, ValueError):
                return "Structural level invalidated"
        return "Structural level invalidated"

    # ------------------------------------------------------------------
    # State access & event emission
    # ------------------------------------------------------------------

    def get_state(self, signal_id: str) -> SignalStatus:
        """Return the current state of a signal."""
        return self._signal_states.get(signal_id, SignalStatus.NO_TRADE)

    async def _emit_events(
        self,
        signal_id: str,
        state: SignalStatus,
        candidate: Optional[SignalCandidate],
        ctx: MarketContext,
    ) -> None:
        """Emit events via the event bus if configured and a loop is running."""
        if self.event_bus is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # No running loop — skip fire-and-forget events silently.

        state_event = Event(
            event_type="signal.status.changed",
            aggregate_id=signal_id,
            aggregate_type="signal",
            payload={
                "signal_id": signal_id,
                "new_state": state.value,
                "pair": ctx.pair,
                "direction": ctx.direction.value,
                "timeframe": ctx.timeframe,
                "supporting_evidence": len(ctx.evidence.supporting),
                "missing_evidence": len(ctx.evidence.missing),
            },
            source_agent="signal_agent",
        )

        events = [state_event]
        if state in (SignalStatus.SETUP_FORMING, SignalStatus.TRIGGERED) and candidate is not None:
            events.append(
                Event(
                    event_type="signal.created",
                    aggregate_id=signal_id,
                    aggregate_type="signal",
                    payload={
                        "signal_id": signal_id,
                        "pair": ctx.pair,
                        "direction": ctx.direction.value,
                        "timeframe": ctx.timeframe,
                        "setup_type": candidate.setup_type,
                        "entry_zone": [candidate.entry_zone_low, candidate.entry_zone_high],
                        "stop_loss": candidate.stop_loss,
                        "take_profit_1": candidate.take_profit_1,
                        "risk_reward": candidate.risk_reward,
                        "status": state.value,
                        "strategy_version": candidate.strategy_version,
                        "invalidation": candidate.invalidation_condition,
                        "supporting_evidence_count": len(candidate.evidence.supporting),
                        "contradicting_evidence_count": len(candidate.evidence.contradicting),
                    },
                    source_agent="signal_agent",
                )
            )

        for event in events:
            try:
                await self.event_bus.publish(event)
            except Exception as exc:  # noqa: BLE001 — event bus failures never block signals
                logger.warning("Failed to publish %s: %s", event.event_type, exc)
