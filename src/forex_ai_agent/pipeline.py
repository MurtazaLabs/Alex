"""Main pipeline: real market data → structure → indicators → signal → risk → narration.

Data flows through the resilient feed chain (MT5 → Yahoo → Twelve Data → Mock).
Every signal returns its full evidence trail, risk checks, seven-section
explanation and an LLM narrative (with deterministic template fallback).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from forex_ai_agent.analysis.market_structure.engine import build_structure_snapshot
from forex_ai_agent.analysis.market_structure.models import StructureSnapshot
from forex_ai_agent.agents.reasoning_agent import ReasoningAgent
from forex_ai_agent.agents.technical_analysis_agent import compute_all_indicators
from forex_ai_agent.agents.market_agent import MarketAgent
from forex_ai_agent.agents.signal_agent import SignalAgent
from forex_ai_agent.agents.risk_agent import RiskAgent
from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.news import NewsAgent
from forex_ai_agent.core.identity import (
    IdentityLoader,
    get_identity_loader,
    get_alex_directives,
    get_no_trade_sessions,
    get_risk_limits,
)
from forex_ai_agent.config import get_settings
from forex_ai_agent.data.providers import (
    BaseMarketDataProvider,
    get_market_data_provider,
)
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

logger = logging.getLogger(__name__)


def session_for_utc_hour(hour_utc: int) -> str:
    """UTC-hour session mapping (mirrors the backtest engine)."""
    if 0 <= hour_utc < 8:
        return "Asian"
    if 8 <= hour_utc < 13:
        return "London"
    if 13 <= hour_utc < 17:
        return "Overlap"
    return "NewYork"


def _structure_to_signal_snapshot(
    snap: StructureSnapshot, timeframe: str
) -> "Dict[str, Any]":
    """Convert the market-structure engine snapshot into the rule-engine's dict shape."""

    def r5(value: Any) -> Any:
        return round(value, 5) if isinstance(value, float) else value

    return {
        "trend": snap.trend.value,
        "swing_highs": [{"index": s.index, "price": r5(s.price), "kind": s.kind} for s in snap.swing_highs],
        "swing_lows": [{"index": s.index, "price": r5(s.price), "kind": s.kind} for s in snap.swing_lows],
        "last_bos": (
            {"kind": snap.last_bos.kind, "direction": snap.last_bos.direction.value,
             "level": r5(snap.last_bos.level), "price": r5(snap.last_bos.price), "confidence": snap.last_bos.confidence}
            if snap.last_bos else None
        ),
        "last_choch": (
            {"kind": snap.last_choch.kind, "direction": snap.last_choch.direction.value,
             "level": r5(snap.last_choch.level), "price": r5(snap.last_choch.price), "confidence": snap.last_choch.confidence}
            if snap.last_choch else None
        ),
        "key_support": r5(snap.key_support),
        "key_resistance": r5(snap.key_resistance),
        "liquidity_sweeps": [
            {"direction": sw.direction.value, "swept_price": r5(sw.swept_price),
             "depth_atr": sw.depth_atr, "confidence": sw.confidence}
            for sw in snap.liquidity_sweeps
        ],
        "consolidation": (
            {"start_index": snap.consolidation.start_index, "end_index": snap.consolidation.end_index,
             "high": r5(snap.consolidation.high), "low": r5(snap.consolidation.low),
             "broken": snap.consolidation.broken,
             "breakout_direction": snap.consolidation.breakout_direction.value if snap.consolidation.breakout_direction else None}
            if snap.consolidation else None
        ),
        "retests": [
            {"index": r.index, "level": r5(r.level), "direction": r.direction.value,
             "success": r.success, "confidence": r.confidence}
            for r in snap.retests
        ],
        "events": [
            {"kind": e.kind, "direction": e.direction.value, "index": e.index,
             "level": r5(e.level), "price": r5(e.price), "confidence": e.confidence, "details": e.details}
            for e in snap.events
        ],
    }


class Pipeline:
    """Main orchestration pipeline for the Forex AI Agent."""

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        use_redis: bool = True,
        data_provider: Optional[BaseMarketDataProvider] = None,
    ):
        self.event_bus = event_bus or InMemoryEventBus()
        self.settings = get_settings()

        # Identity & constitution (Alex's soul, identity, user profile)
        self.identity_loader: IdentityLoader = get_identity_loader()
        self.soul = self.identity_loader.soul
        self.identity = self.identity_loader.identity
        self.user_profile = self.identity_loader.user
        self._log_identity()

        # Data feed chain (resilient: MT5 → Yahoo → TwelveData → Mock)
        self.data_provider = data_provider or get_market_data_provider()
        self.data_source = getattr(self.data_provider, "active_source", "unknown")

        # Agents
        self.market_agent = MarketAgent(event_bus=self.event_bus)
        self.signal_agent = SignalAgent(event_bus=self.event_bus)
        self.risk_agent = RiskAgent(event_bus=self.event_bus)
        self.news_agent = NewsAgent(event_bus=self.event_bus)
        self.reasoning_agent = ReasoningAgent()

        self._db_initialized = False

    def _log_identity(self) -> None:
        logger.info("=" * 60)
        logger.info("ALEX INITIALIZED — %s (%s)", self.identity.name, self.identity.codename)
        logger.info("Role: %s", self.identity.role)
        logger.info("Architecture: %s", self.identity.architecture_principle)
        logger.info("No-trade sessions: %s", get_no_trade_sessions())
        logger.info("Risk limits: %s", get_risk_limits())
        for directive in get_alex_directives():
            logger.info("Directive: %s", directive)
        logger.info("=" * 60)

    async def ensure_db(self) -> None:
        """Initialize database if not already done."""
        if not self._db_initialized:
            try:
                from forex_ai_agent.data.database import init_db
                await init_db(self.settings.database_url)
                self._db_initialized = True
            except Exception as exc:  # noqa: BLE001 — run without persistence if needed
                logger.warning("Database unavailable, running without persistence: %s", exc)

    async def initialize(self) -> None:
        """Initialize async components (news agent, db)."""
        await self.ensure_db()
        try:
            await self.news_agent.start()
        except Exception as exc:  # noqa: BLE001
            logger.warning("News agent failed to start: %s", exc)

    # ------------------------------------------------------------------
    # Data feed
    # ------------------------------------------------------------------

    async def fetch_multi_tf(
        self,
        pair: str,
        tfs: tuple[str, ...] = ("M15", "H1", "H4", "D1"),
        limit: int = 300,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch per-timeframe OHLC arrays from the resilient feed chain."""
        end = datetime.utcnow()
        out: Dict[str, Dict[str, Any]] = {}
        for tf in tfs:
            start = end - timedelta(minutes=_tf_minutes(tf) * limit * 1.5)
            bars = await self.data_provider.fetch_candles(pair, tf, start, end=end, limit=limit)
            highs = np.array([b.high for b in bars], dtype=float)
            lows = np.array([b.low for b in bars], dtype=float)
            closes = np.array([b.close for b in bars], dtype=float)
            out[tf] = {
                "highs": highs,
                "lows": lows,
                "closes": closes,
                "volume": np.array([b.volume for b in bars], dtype=float),
                "timestamps": [b.timestamp for b in bars],
                "bars": bars,
            }
        # Spread: real bid/ask when the feed provides a quote; ATR-estimated otherwise.
        spread_pips, spread_source = await self._estimate_spread(pair, out)
        out["_spread_pips"] = spread_pips
        out["_spread_source"] = spread_source
        self.data_source = getattr(self.data_provider, "active_source", "unknown")
        return out

    async def _estimate_spread(self, pair: str, multi_tf: Dict[str, Dict[str, Any]]) -> tuple[float, str]:
        pip = _pip_size(pair)
        try:
            quote = await self.data_provider.fetch_current_price(pair)
        except Exception as exc:  # noqa: BLE001
            logger.info("No quote available for spread (%s)", exc)
            quote = None
        if quote is not None and quote.bid and quote.ask:
            return (quote.ask - quote.bid) / pip, "real_quote"
        h1 = multi_tf.get("H1") or next(iter(multi_tf.values()))
        closes = h1.get("closes")
        highs = h1.get("highs")
        lows = h1.get("lows")
        if closes is not None and len(closes) >= 15:
            from forex_ai_agent.analysis.indicators.volatility import atr as atr_fn
            atr_val = float(atr_fn(highs, lows, closes, 14)[-1])
            return max(atr_val / pip * 0.08, 0.5), "estimated"
        return 1.0, "fallback"

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    async def generate_signal(
        self,
        pair: str,
        direction: str,
        timeframe: str = "H1",
        narrate: bool = True,
    ) -> Dict[str, Any]:
        """Full signal pipeline over real market data."""
        from forex_ai_agent.analysis.signal_rules.base_rule import (
            Direction as RuleDirection,
            HTFContext,
            IndicatorsSnapshot,
            MarketContext,
            StructureSnapshot as RuleStructure,
        )

        await self.ensure_db()
        dir_enum = (
            RuleDirection.LONG if str(direction).upper() == "LONG" else RuleDirection.SHORT
        )

        multi_tf = await self.fetch_multi_tf(pair)
        if timeframe not in multi_tf:
            return {"status": "insufficient_data", "message": f"No data for {timeframe}"}

        tf_data = multi_tf[timeframe]
        closes = tf_data["closes"]
        if len(closes) < 50:
            return {"status": "insufficient_data", "message": f"Only {len(closes)} bars"}

        # 1. Indicators per timeframe
        indicators = await compute_all_indicators(
            highs={tf: d["highs"] for tf, d in multi_tf.items() if not tf.startswith("_")},
            lows={tf: d["lows"] for tf, d in multi_tf.items() if not tf.startswith("_")},
            closes={tf: d["closes"] for tf, d in multi_tf.items() if not tf.startswith("_")},
        )

        # 2. Structure snapshot on the requested timeframe + HTF contexts
        structure_snap = build_structure_snapshot(
            timeframe=timeframe,
            highs=tf_data["highs"],
            lows=tf_data["lows"],
            closes=closes,
            timestamps=tf_data.get("timestamps"),
        )
        structure_dict = _structure_to_signal_snapshot(structure_snap, timeframe)

        htf_contexts: Dict[str, HTFContext] = {}
        for tf in ("D1", "H4", "H1", "M15"):
            if tf not in multi_tf:
                continue
            d = multi_tf[tf]
            snap_tf = build_structure_snapshot(
                timeframe=tf, highs=d["highs"], lows=d["lows"], closes=d["closes"],
                timestamps=d.get("timestamps"),
            )
            trend_val = snap_tf.trend.value
            htf_contexts[tf] = HTFContext(
                timeframe=tf,
                trend=trend_val,
                structure=_structure_to_signal_snapshot(snap_tf, tf),
                key_support=snap_tf.key_support,
                key_resistance=snap_tf.key_resistance,
            )

        # 3. Market context
        spread_pips = float(multi_tf.get("_spread_pips", 1.0))
        ind_tf = indicators.get(timeframe, {})
        atr_price = ind_tf.get("atr") or 0.0
        atr_pips = atr_price / _pip_size(pair) if atr_price else 10.0
        session = session_for_utc_hour(datetime.utcnow().hour)

        news_state = "LOW"
        try:
            base_ccy = pair.split("/")[0]
            impact = self.news_agent.get_risk_state(base_ccy)
            if impact is not None:
                news_state = impact.value
        except Exception:  # noqa: BLE001
            pass

        rule_structure = RuleStructure(
            trend=structure_dict["trend"],
            swing_highs=structure_dict["swing_highs"],
            swing_lows=structure_dict["swing_lows"],
            last_bos=structure_dict["last_bos"],
            last_choch=structure_dict["last_choch"],
            key_support=structure_dict["key_support"],
            key_resistance=structure_dict["key_resistance"],
            liquidity_sweeps=structure_dict["liquidity_sweeps"],
            consolidation=structure_dict["consolidation"],
            retests=structure_dict["retests"],
            events=structure_dict["events"],
        )

        market_ctx = MarketContext(
            pair=pair,
            timeframe=timeframe,
            current_price=float(closes[-1]),
            spread_pips=spread_pips,
            atr_pips=atr_pips,
            session=session,
            news_state=news_state,
            htf_context=htf_contexts,
            indicators=IndicatorsSnapshot(
                rsi=ind_tf.get("rsi"),
                macd=ind_tf.get("macd"),
                macd_signal=ind_tf.get("macd_signal"),
                macd_histogram=ind_tf.get("macd_histogram"),
                atr=ind_tf.get("atr"),
                bb_upper=ind_tf.get("bb_upper"),
                bb_middle=ind_tf.get("bb_middle"),
                bb_lower=ind_tf.get("bb_lower"),
                sma_20=ind_tf.get("sma_20"),
                sma_50=ind_tf.get("sma_50"),
                ema_20=ind_tf.get("ema_20"),
                ema_50=ind_tf.get("ema_50"),
            ),
            structure=rule_structure,
        )
        market_ctx.direction = dir_enum

        # 4. Signal evaluation
        candidate = await self.signal_agent.evaluate_signal(
            pair=pair, direction=dir_enum, timeframe=timeframe, ctx=market_ctx,
        )

        # 5. Risk validation (only for formed setups — zero levels are meaningless).
        from forex_ai_agent.agents.risk_agent import RiskValidationResult

        if candidate.setup_type != "NO_TRADE" and candidate.stop_loss and candidate.take_profit_1:
            equity = float(self.settings.account_equity)
            risk_pct = float(self.settings.default_risk_percent)
            entry_ref = (
                candidate.entry_zone_low
                if candidate.direction == RuleDirection.LONG
                else candidate.entry_zone_high
            )
            risk_result = self.risk_agent.validate(
                account_equity=equity,
                risk_percent=risk_pct,
                entry_price=entry_ref if entry_ref else float(closes[-1]),
                stop_loss=candidate.stop_loss,
                take_profit=candidate.take_profit_1,
                symbol=pair,
                current_spread_pips=spread_pips,
                current_atr_pips=atr_pips,
                current_session=session,
                currency=pair.split("/")[0],
                news_agent=self.news_agent,
            )
        else:
            risk_result = RiskValidationResult(
                passed=False, position_size_lots=0.0, risk_amount_usd=0.0, checks={}
            )

        # 6. Explanation (7 sections) + LLM narrative
        explanation = self._build_explanation(pair, dir_enum, timeframe, candidate, market_ctx, risk_result)
        narrative_context = self.reasoning_agent.build_context(
            pair=pair,
            direction=dir_enum.value,
            timeframe=timeframe,
            session=session,
            structure_dict=structure_dict,
            indicators_dict={
                "rsi": ind_tf.get("rsi"),
                "atr": atr_pips,
                "macd_histogram": ind_tf.get("macd_histogram"),
            },
            risk_checks={
                "passed": risk_result.passed,
                "checks": {
                    name: {"passed": c.passed, "message": c.message}
                    for name, c in risk_result.checks.items()
                },
            },
            signal_candidate={
                "setup_type": candidate.setup_type,
                "entry_zone_low": candidate.entry_zone_low,
                "entry_zone_high": candidate.entry_zone_high,
                "stop_loss": candidate.stop_loss,
                "take_profit_1": candidate.take_profit_1,
                "risk_reward": candidate.risk_reward,
                "invalidation": candidate.invalidation_condition,
            },
            data_source=self.data_source,
        )
        # ``narrate=False`` (agent chat): keep the deterministic 7-section
        # explanation, skip the slow LLM narration — the AgentCore adds its own
        # final answer on top of the evidence.
        prefer_llm = narrate and candidate.setup_type != "NO_TRADE"
        narrative = await self.reasoning_agent.narrate_signal(
            narrative_context, prefer_llm=prefer_llm
        )

        # 7. Assemble result
        result: Dict[str, Any] = {
            "status": "ok",
            "pair": pair,
            "data_source": self.data_source,
            "spread_pips": round(spread_pips, 2),
            "spread_source": multi_tf.get("_spread_source", "unknown"),
            "session": session,
            "atr_pips": round(atr_pips, 2),
            "signal_candidate": {
                "pair": candidate.pair,
                "direction": candidate.direction.value,
                "timeframe": candidate.timeframe,
                "setup_type": candidate.setup_type,
                "entry_zone_low": candidate.entry_zone_low,
                "entry_zone_high": candidate.entry_zone_high,
                "stop_loss": candidate.stop_loss,
                "take_profit_1": candidate.take_profit_1,
                "take_profit_2": candidate.take_profit_2,
                "take_profit_3": candidate.take_profit_3,
                "risk_reward": candidate.risk_reward,
                "strategy_version": candidate.strategy_version,
                "invalidation_condition": candidate.invalidation_condition,
                "explanation": candidate.explanation,
                "status": self.signal_agent.get_state(f"{pair}_{timeframe}_{dir_enum.value}").value,
                "score_pct": round(candidate.score.score_pct, 3) if candidate.score else None,
                "evidence": {
                    "supporting": [
                        {"category": e.category, "description": e.description, "strength": e.strength}
                        for e in candidate.evidence.supporting
                    ],
                    "contradicting": [
                        {"category": e.category, "description": e.description}
                        for e in candidate.evidence.contradicting
                    ],
                    "missing": list(candidate.evidence.missing)[:10],
                },
            },
            "risk_validation": {
                "passed": risk_result.passed,
                "position_size_lots": risk_result.position_size_lots,
                "risk_amount_usd": risk_result.risk_amount_usd,
                "checks": {
                    name: {"passed": c.passed, "message": c.message}
                    for name, c in risk_result.checks.items()
                },
            },
            "explanation": explanation,
            "narrative": narrative,
            "events_emitted": [
                "market.data.updated",
                "signal.status.changed",
                *(["signal.created"] if risk_result.passed else []),
                "risk.validation.updated",
            ],
        }
        return result

    def _build_explanation(
        self,
        pair: str,
        dir_enum: Any,
        timeframe: str,
        candidate: Any,
        market_ctx: Any,
        risk_result: Any,
    ) -> str:
        """Deterministic seven-section explanation (auditable, LLM-free)."""
        ev = candidate.evidence
        lines: List[str] = []
        formed = candidate.setup_type != "NO_TRADE"

        lines.append("WHY THIS SETUP EXISTS:")
        if ev.supporting:
            for i, e in enumerate(ev.supporting[:6], 1):
                lines.append(f"  {i}. [{e.category}] {e.description}")
        else:
            lines.append("  No supporting evidence collected — setup did not form.")

        if formed:
            lines.append("")
            lines.append("WHY ENTRY IS HERE:")
            lines.append(
                f"  Entry zone {candidate.entry_zone_low:.5f}-{candidate.entry_zone_high:.5f} "
                f"anchored to structure near current price {market_ctx.current_price:.5f}."
            )

            lines.append("")
            lines.append("WHY SL IS HERE:")
            lines.append(
                f"  Stop at {candidate.stop_loss:.5f} sits beyond the invalidating "
                f"{'support' if dir_enum.value == 'LONG' else 'resistance'} level."
            )

            lines.append("")
            lines.append("WHY TP IS HERE:")
            lines.append(
                f"  TP1 at {candidate.take_profit_1:.5f} gives {candidate.risk_reward:.1f}R; "
                f"targets {candidate.take_profit_2} then {candidate.take_profit_3}."
            )
        else:
            lines.append("")
            lines.append(
                "  No entry/SL/TP levels are shown because no setup formed — "
                "evidence does not meet the required thresholds yet."
            )

        lines.append("")
        lines.append("WHAT INVALIDATES THE TRADE:")
        lines.append(f"  {candidate.invalidation_condition}")

        lines.append("")
        lines.append("CONTRADICTING FACTORS:")
        if ev.contradicting:
            for i, e in enumerate(ev.contradicting[:3], 1):
                lines.append(f"  {i}. [{e.category}] {e.description}")
        else:
            lines.append("  None identified.")

        lines.append("")
        lines.append("MISSING INFORMATION:")
        if ev.missing:
            for i, m in enumerate(list(ev.missing)[:5], 1):
                lines.append(f"  {i}. {m}")
        else:
            lines.append("  None.")

        lines.append("")
        lines.append("RISK VALIDATION:")
        if risk_result.checks:
            lines.append(
                f"  {'PASSED' if risk_result.passed else 'FAILED'} — "
                f"{sum(1 for c in risk_result.checks.values() if c.passed)}/{len(risk_result.checks)} checks pass."
            )
        else:
            lines.append("  Not run — no setup to validate.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Return status of all pipeline components (feeds, LLM, settings)."""
        health: List[Dict[str, Any]] = []
        dp = self.data_provider
        if hasattr(dp, "health_report"):
            health = dp.health_report()
            self.data_source = getattr(dp, "active_source", self.data_source)
        try:
            llm_status = await self.reasoning_agent.llm.health()
        except Exception as exc:  # noqa: BLE001
            llm_status = {"error": str(exc)}
        return {
            "status": "ok",
            "data_source": self.data_source,
            "feed_health": health,
            "llm": llm_status,
            "agents": {
                "market_agent": getattr(self.market_agent, "name", "market"),
                "signal_agent": getattr(self.signal_agent, "name", "signal"),
                "risk_agent": getattr(self.risk_agent, "name", "risk"),
                "news_agent": getattr(self.news_agent, "name", "news"),
            },
            "db_initialized": self._db_initialized,
            "settings": {
                "risk_percent": f"{self.settings.default_risk_percent}%",
                "min_rr": f"{self.settings.min_risk_reward}",
                "max_exposure": f"{self.settings.max_total_exposure_pct}%",
                "max_permission_level": self.settings.max_permission_level,
            },
        }


def _tf_minutes(tf: str) -> int:
    return {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}.get(tf.upper(), 60)


def _pip_size(pair: str) -> float:
    try:
        return pip_size_for_symbol(pair)
    except ValueError:
        return 0.0001
