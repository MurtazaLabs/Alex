"""Agent tools — interface to the deterministic core.

Read-only tools (Phase 1): never mutate state.
Action tools (Phase 2+): mutate state via confirmation policy.
All tools run on the pipeline's event loop (CLI/bridge: loop.run_until_complete).
"""

from __future__ import annotations

import asyncio
import inspect
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from forex_ai_agent.agent.confirmations import (
    RiskClass,
    get_confirmation_policy,
    register_tool_risk,
    build_impact_summary,
    ActionProposal,
    ConfirmationToken,
)
from forex_ai_agent.config import get_settings
from forex_ai_agent.core.permission_manager import PermissionLevel


# ---------- Tool contracts (Pydantic models for args/returns) ----------

class MarketDataArgs(BaseModel):
    symbol: str = Field(description="Forex pair symbol, e.g. 'EUR/USD'")
    timeframe: str = Field(default="H1", description="Timeframe: M1, M5, M15, H1, H4, D1")
    limit: int = Field(default=200, ge=10, le=1000, description="Number of candles to fetch")


class MarketDataResult(BaseModel):
    symbol: str
    timeframe: str
    candles: List[Dict[str, Any]]  # {time, open, high, low, close, volume}
    count: int


class CurrentPriceArgs(BaseModel):
    symbol: str = Field(description="Forex pair symbol, e.g. 'EUR/USD'")


class CurrentPriceResult(BaseModel):
    symbol: str
    price: float
    bid: float
    ask: float
    spread_pips: float
    timestamp: str


class StructureArgs(BaseModel):
    symbol: str = Field(description="Forex pair symbol, e.g. 'EUR/USD'")
    timeframe: str = Field(default="H1", description="Timeframe: M1, M5, M15, H1, H4, D1")


class StructureResult(BaseModel):
    symbol: str
    timeframe: str
    current_price: float
    trend: str
    swing_highs: List[Dict[str, Any]]
    swing_lows: List[Dict[str, Any]]
    last_bos: Optional[Dict[str, Any]]
    last_choch: Optional[Dict[str, Any]]
    key_support: Optional[float]
    key_resistance: Optional[float]
    liquidity_sweeps: List[Dict[str, Any]]
    consolidation: Optional[Dict[str, Any]]
    retests: List[Dict[str, Any]]


class IndicatorsArgs(BaseModel):
    symbol: str = Field(description="Forex pair symbol, e.g. 'EUR/USD'")
    timeframe: str = Field(default="H1", description="Timeframe")
    indicators: List[str] = Field(default=["rsi", "ema", "macd", "atr"])


class IndicatorsResult(BaseModel):
    symbol: str
    timeframe: str
    rsi: Optional[float]
    ema_9: Optional[float]
    ema_21: Optional[float]
    macd: Optional[Dict[str, Optional[float]]]
    atr: Optional[float]


class ActiveSignalsArgs(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Filter by symbol")
    direction: Optional[str] = Field(default=None, description="LONG or SHORT")
    timeframe: Optional[str] = Field(default=None)


class ActiveSignalsResult(BaseModel):
    signals: List[Dict[str, Any]]


class RiskStateArgs(BaseModel):
    pass


class RiskStateResult(BaseModel):
    risk_level: str  # low/medium/high/critical/unknown
    daily_loss_pct: float = 0.0
    max_drawdown_pct: float = 0.0  # live-gateway figure pending execution layer
    open_positions: int = 0
    margin_level: float = 0.0  # live-gateway figure pending execution layer
    upcoming_events: List[Dict[str, Any]] = Field(default_factory=list)
    currencies: List[Dict[str, Any]] = Field(default_factory=list)
    note: Optional[str] = None


class NewsArgs(BaseModel):
    symbol: Optional[str] = Field(default=None)
    hours: int = Field(default=24)


class NewsResult(BaseModel):
    items: List[Dict[str, Any]]


class SearchStrategiesArgs(BaseModel):
    query: Optional[str] = Field(default=None)
    status: Optional[str] = Field(default=None)
    limit: int = Field(default=20)


class SearchStrategiesResult(BaseModel):
    strategies: List[Dict[str, Any]]


class GetStrategyArgs(BaseModel):
    version_id: Optional[str] = Field(default=None)
    version_string: Optional[str] = Field(default=None)


class GetStrategyResult(BaseModel):
    strategy: Optional[Dict[str, Any]]


class ListProbationArgs(BaseModel):
    pass


class ListProbationResult(BaseModel):
    records: List[Dict[str, Any]]


class ListResearchHistoryArgs(BaseModel):
    limit: int = Field(default=20)


class ListResearchHistoryResult(BaseModel):
    records: List[Dict[str, Any]]


# ---------- Action tool contracts (Phase 2) ----------

class StartResearchArgs(BaseModel):
    topic: str = Field(description="Research topic/question")
    pair: Optional[str] = Field(default=None, description="Forex pair to focus on")
    timeframe: Optional[str] = Field(default=None, description="Timeframe to analyze")
    depth: str = Field(default="standard", description="Research depth: quick/standard/deep")


class StartResearchResult(BaseModel):
    task_id: str
    status: str  # "started" | "error"
    proposal: Optional[Dict[str, Any]] = None  # B5: the plan card


class GenerateHypothesesArgs(BaseModel):
    pairs: Optional[List[str]] = Field(
        default=None, description="Universe to scan (default majors, max 4)"
    )
    timeframe: str = Field(default="H1")
    max_hypotheses: int = Field(default=3, ge=1, le=5)
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    use_llm: bool = Field(default=True, description="False → deterministic template hypotheses only")
    enqueue: bool = Field(default=True, description="False → propose only, touch nothing")


class GenerateHypothesesResult(BaseModel):
    report: Dict[str, Any]


class SelectExperimentsArgs(BaseModel):
    strategies: Optional[List[str]] = Field(
        default=None, description="Strategy ids to consider (default: all saved strategies)"
    )
    symbol: str = Field(default="EUR/USD", description="Test symbol for the runs")
    timeframe: str = Field(default="H1")
    start_date: str = Field(default="2025-01-01")
    end_date: str = Field(default="2025-12-31")
    max_experiments: int = Field(default=1, ge=1, le=3)
    max_variants: int = Field(default=4, ge=1, le=6)
    window_months: int = Field(default=6, ge=1, le=24)
    step_months: int = Field(default=2, ge=1, le=12)
    max_oos_windows: int = Field(default=3, ge=1, le=4)
    use_llm: bool = Field(default=True, description="False → deterministic baseline proposal only")
    dry_run: bool = Field(default=False, description="True → validate proposals, execute nothing")


class SelectExperimentsResult(BaseModel):
    report: Dict[str, Any]


class GenerateReportArgs(BaseModel):
    topic: str = Field(default="Research summary", description="Report topic")
    experiment_ids: Optional[List[str]] = Field(
        default=None, description="Specific experiment records to report on (default: recent)"
    )
    strategy_ids: Optional[List[str]] = Field(
        default=None, description="Filter evidence to these strategy ids"
    )
    limit_experiments: int = Field(default=10, ge=1, le=20)
    use_llm: bool = Field(default=True, description="False → deterministic sections only")
    persist: bool = Field(default=True, description="False → build the report, write nothing")


class GenerateReportResult(BaseModel):
    report: Dict[str, Any]


class ReviewPromotionArgs(BaseModel):
    limit: int = Field(default=5, ge=1, le=10)
    include_narration: bool = Field(
        default=True, description="False → skip LLM narration, deterministic one-liners only"
    )
    include_expired: bool = Field(
        default=False, description="Include expired probation rows (informational, never ready)"
    )


class ReviewPromotionResult(BaseModel):
    review: Dict[str, Any]


class ProbationMonitorArgs(BaseModel):
    version_id: Optional[str] = Field(
        default=None,
        description="Probation version id (prefix ok) — omit for all active probations",
    )
    limit: int = Field(default=20, ge=1, le=100)


class ProbationMonitorResult(BaseModel):
    summary: Dict[str, int]
    probations: List[Dict[str, Any]] = Field(default_factory=list)
    rendered: str = ""


class ResearchPlanArgs(BaseModel):
    task_id: Optional[str] = Field(
        default=None,
        description="Plan id (prefix ok) — omit to list recent structured plans",
    )
    limit: int = Field(default=10, ge=1, le=50)


class ResearchPlanResult(BaseModel):
    found: bool
    plans: List[Dict[str, Any]] = Field(default_factory=list)
    rendered: str = ""
    task_id: Optional[str] = None


class RemoveStrategyArgs(BaseModel):
    version_id: str = Field(description="Strategy version ID to remove")


class RemoveStrategyResult(BaseModel):
    success: bool
    version_id: str
    proposal: Optional[Dict[str, Any]] = None


class ArchiveStrategyArgs(BaseModel):
    version_id: str = Field(description="Strategy version ID to archive")


class ArchiveStrategyResult(BaseModel):
    success: bool
    version_id: str
    proposal: Optional[Dict[str, Any]] = None


class RejectStrategyArgs(BaseModel):
    version_id: str = Field(description="Strategy version ID to reject")
    reason: str = Field(description="Reason for rejection")


class RejectStrategyResult(BaseModel):
    success: bool
    version_id: str
    proposal: Optional[Dict[str, Any]] = None


class CreateStrategyDraftArgs(BaseModel):
    name: str = Field(description="Strategy name")
    description: str = Field(description="Strategy description")
    rules: Dict[str, Any] = Field(description="Entry/exit rules")
    parameters: Dict[str, Any] = Field(description="Strategy parameters")


class CreateStrategyDraftResult(BaseModel):
    draft_id: str
    status: str = "draft"


# ---------- Experiment tool contracts (Phase 3) ----------

class BacktestStrategyArgs(BaseModel):
    strategy_id: str = Field(description="Strategy version ID to backtest")
    symbol: str = Field(description="Forex pair symbol, e.g. 'EUR/USD'")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")
    timeframe: str = Field(default="H1", description="Timeframe")
    initial_balance: float = Field(default=100000.0, description="Starting balance")


class BacktestStrategyResult(BaseModel):
    strategy_id: str
    strategy_name: str
    metrics: Dict[str, Any]
    trades: int
    equity_curve: List[Dict[str, Any]]


class WalkForwardTestArgs(BaseModel):
    strategy_id: str = Field(description="Strategy version ID")
    symbol: str = Field(description="Forex pair symbol")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")
    timeframe: str = Field(default="H1", description="Timeframe")
    window_months: int = Field(default=6, description="IS window months")
    step_months: int = Field(default=2, description="Step forward months")
    run_parameter_grid: bool = Field(
        default=False,
        description="Also measure OOS parameter sensitivity (≤12 extra backtests, off by default)",
    )
    parameter_grid: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional explicit mutation list, e.g. [{'atr_sl_mult': 2.0}]",
    )


class WalkForwardTestResult(BaseModel):
    strategy_id: str
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    windows_tested: int
    windows: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-window rows: IS/OOS sharpe, PF, trades, drawdown",
    )
    aggregate_metrics: Dict[str, Any]
    stability_score: float
    verdict: str = ""  # STABLE | MIXED | UNSTABLE
    overfit_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-A2 component breakdown (degradation, stability, negative-window fraction, concentration, sensitivity)",
    )
    repro: Dict[str, Any] = Field(
        default_factory=dict,
        description="Stage-A1 reproducibility metadata — pass to replay_repro to re-run exactly",
    )


class CompareStrategiesArgs(BaseModel):
    strategy_ids: List[str] = Field(description="List of strategy version IDs to compare")
    symbol: str = Field(description="Forex pair symbol")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")
    timeframe: str = Field(default="H1", description="Timeframe")


class CompareStrategiesResult(BaseModel):
    comparison: List[Dict[str, Any]]
    best: Optional[Dict[str, Any]]
    worst: Optional[Dict[str, Any]]


class CreateExperimentArgs(BaseModel):
    strategy_id: str = Field(description="Base strategy version ID")
    param_mutations: Dict[str, Any] = Field(description="Parameter path -> new value, e.g. {'parameters.sl_atr_multiplier': 2.0}")
    symbol: str = Field(description="Forex pair symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    start_date: str = Field(description="Start date (ISO format)")
    end_date: str = Field(description="End date (ISO format)")
    hypothesis_id: Optional[str] = Field(default=None, description="Optional hypothesis ID")


class CreateExperimentResult(BaseModel):
    experiment_id: str
    verdict: str
    recommendation: str
    baseline_metrics: Dict[str, Any]
    variant_count: int


class ListExperimentsArgs(BaseModel):
    strategy_id: Optional[str] = Field(default=None, description="Filter by strategy ID")
    limit: int = Field(default=20, description="Max results")


class ListExperimentsResult(BaseModel):
    experiments: List[Dict[str, Any]]


class GetExperimentArgs(BaseModel):
    experiment_id: str = Field(description="Experiment ID")


class GetExperimentResult(BaseModel):
    experiment: Optional[Dict[str, Any]]


class ConcludeExperimentArgs(BaseModel):
    experiment_id: str = Field(description="Experiment ID")
    decision: str = Field(description="Decision: adopt/reject/defer")
    notes: str = Field(default="", description="Optional notes")


class ConcludeExperimentResult(BaseModel):
    experiment_id: str
    final_verdict: str
    concluded_at: Optional[str]


# ---------- B2 research graph tool contracts ----------

class HypothesisTestCountArgs(BaseModel):
    """B2: how many times has a hypothesis been tested?"""

    hypothesis_id: str = Field(description="Hypothesis ID (DB uuid or research id)")


class HypothesisTestCountResult(BaseModel):
    hypothesis_id: str
    times_tested: int = 0
    experiments: int = 0
    outcomes: Dict[str, Any] = Field(default_factory=dict)
    decisions: Dict[str, Any] = Field(default_factory=dict)
    strategy_versions: List[str] = Field(default_factory=list)
    parameters_varied: Dict[str, Any] = Field(default_factory=dict)
    first_tested: Optional[str] = None
    last_tested: Optional[str] = None
    note: str = ""


class AggregateByParameterArgs(BaseModel):
    """B2: aggregate stored outcomes by a parameter's value."""

    param_pattern: str = Field(
        description="Fuzzy parameter name to group by, e.g. 'displacement' or 'atr_sl_mult'"
    )


class AggregateByParameterResult(BaseModel):
    parameter_pattern: str
    observations_matched: int = 0
    buckets: List[Dict[str, Any]] = Field(default_factory=list)
    note: str = ""


class ResearchGraphSummaryArgs(BaseModel):
    """B2: cheap overview of the research graph."""

    pass


class ResearchGraphSummaryResult(BaseModel):
    edges: int = 0
    note: str = ""


# ---------- B4: Research Director tool contracts ----------

class DirectorEnqueueArgs(BaseModel):
    """Add a hypothesis to the Director's research queue (B4)."""

    pair: str = Field(description="Forex pair to research (e.g. GBP/JPY)")
    hypothesis: str = Field(default="", description="What to test on this pair")
    timeframe: str = Field(default="H1", description="Timeframe to research")
    priority: int = Field(
        default=2,
        description="1=HIGH runs first, 2=NORMAL, 3=LOW",
    )


class DirectorEnqueueResult(BaseModel):
    enqueued: bool
    reason: str = ""
    item_id: str = ""
    position: int = 0


class DirectorRunCycleArgs(BaseModel):
    """Run one budget-capped unattended research cycle (B4)."""

    max_items: int = Field(default=3, description="Max queue items to research this cycle")
    max_llm_calls: int = Field(default=3, description="LLM proposals allowed before deterministic fallback")
    use_llm: bool = Field(default=True, description="Allow LLM proposals (fallback spec is always available)")


class DirectorRunCycleResult(BaseModel):
    cycle_id: str
    items_run: int = 0
    validated: int = 0
    rejected: int = 0
    errors: int = 0
    llm_calls_used: int = 0
    budget_exhausted: bool = False
    results: List[Dict[str, Any]] = Field(default_factory=list)
    card: str = ""


class DirectorStatusArgs(BaseModel):
    """Research queue + recent audit trail (B4)."""

    pass


class DirectorStatusResult(BaseModel):
    queue: List[Dict[str, Any]] = Field(default_factory=list)
    open_items: int = 0
    by_status: Dict[str, int] = Field(default_factory=dict)
    recent_events: List[Dict[str, Any]] = Field(default_factory=list)
    card: str = ""


# ---------- Phase 5 tool contracts ----------

class CheckViabilityArgs(BaseModel):
    strategy_id: str = Field(description="Strategy version ID")
    symbol: str = Field(description="Forex pair symbol")
    timeframe: str = Field(default="H1", description="Timeframe")
    lookback_days: int = Field(default=90, description="Days of recent data to replay")


class CheckViabilityResult(BaseModel):
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    baseline_metrics: Dict[str, Any]
    current_metrics: Dict[str, Any]
    drift: Dict[str, float]
    oos_pf_drift: float
    dd_change_pct: float
    concentration_risk: float
    verdict: str  # STABLE/DEGRADED/CRITICAL
    concerns: List[str]
    disclaimer: str


class FindStrategiesArgs(BaseModel):
    pairs: List[str] = Field(description="List of forex pairs to search")
    timeframes: List[str] = Field(default=["H1", "H4"], description="Timeframes to test")
    days: int = Field(default=365, description="Historical days for research")
    min_profit_factor: float = Field(default=1.3, description="Minimum profit factor")
    min_trades: int = Field(default=20, description="Minimum trades")
    max_drawdown_pct: float = Field(default=0.20, description="Maximum drawdown")
    max_concurrent: int = Field(default=3, description="Max concurrent candidates")


class FindStrategiesResult(BaseModel):
    candidates_found: int
    ranked_strategies: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    disclaimer: str


# ---------- Phase 6 tool contracts: Explain / "Why?" ----------

class ExplainProbationArgs(BaseModel):
    version_id: str = Field(description="Strategy version ID or prefix")


class ExplainProbationResult(BaseModel):
    strategy_id: str
    strategy_name: str
    pair: str
    status: str
    evidence_cards: List[Dict[str, Any]]
    concerns: List[str]
    recommendation: str


class ExplainResearchArgs(BaseModel):
    task_id: str = Field(description="Research task ID or prefix")


class ExplainResearchResult(BaseModel):
    title: str
    subtitle: str
    evidence: List[Dict[str, Any]]
    concerns: List[str]
    recommendation: str
    verdict: str


class ExplainGateArgs(BaseModel):
    pair: str = Field(description="Forex pair symbol")
    days: int = Field(default=30, description="Days of history for gate")


class ExplainGateResult(BaseModel):
    title: str
    subtitle: str
    evidence: List[Dict[str, Any]]
    concerns: List[str]
    recommendation: str
    verdict: str


class AnswerWhyArgs(BaseModel):
    question: str = Field(description="Natural language 'why?' question")


class AnswerWhyResult(BaseModel):
    title: str
    subtitle: str
    evidence: List[Dict[str, Any]]
    concerns: List[str]
    recommendation: str
    verdict: str


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="Search query")
    limit: int = Field(default=10, description="Max results")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")


class SearchKnowledgeResult(BaseModel):
    results: List[Dict[str, Any]]


class GetStrategyHistoryArgs(BaseModel):
    strategy_name: str = Field(description="Strategy name to search for")


class GetStrategyHistoryResult(BaseModel):
    history: List[Dict[str, Any]]


class GetRejectionReasonsArgs(BaseModel):
    limit: int = Field(default=20, description="Max results")


class GetRejectionReasonsResult(BaseModel):
    rejections: List[Dict[str, Any]]


# ---------- Tool dataclass ----------

@dataclass(frozen=True)
class Tool:
    """Read-only tool definition for the agent registry."""
    name: str
    description: str
    args_model: Type[BaseModel]
    result_model: Type[BaseModel]
    handler: Callable[..., Awaitable[BaseModel]]
    min_permission_level: int = 0  # OBSERVE = 0
    is_async: bool = True

    def to_schema(self) -> Dict[str, Any]:
        """Return JSON schema for LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_model.model_json_schema(),
            "min_permission_level": self.min_permission_level,
        }


# ---------- Tool implementations ----------

async def _fetch_candles(symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
    """Fetch candles from the resilient feed chain (MT5 -> Yahoo -> ... -> Mock)."""
    from forex_ai_agent.data.providers import get_feed_chain
    from forex_ai_agent.utils.timeframes import timeframe_minutes

    provider = get_feed_chain()
    end = datetime.utcnow()
    start = end - timedelta(minutes=timeframe_minutes(timeframe) * limit * 2)
    bars = await provider.fetch_candles(symbol, timeframe, start, end=end, limit=limit)
    return [
        {
            "time": b.timestamp.isoformat(),
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]


async def _fetch_current_price(symbol: str) -> Dict[str, Any]:
    """Fetch the latest quote from the resilient feed chain."""
    from forex_ai_agent.data.providers import get_feed_chain

    provider = get_feed_chain()
    tick = await provider.fetch_current_price(symbol)
    if tick is None:
        raise ValueError(f"No price available for {symbol} right now")
    bid = tick.bid if tick.bid is not None else tick.close
    ask = tick.ask if tick.ask is not None else tick.close
    return {
        "symbol": symbol,
        "price": (bid + ask) / 2,
        "bid": bid,
        "ask": ask,
        "spread_pips": max((ask - bid) * 10000, 0.0),
        "timestamp": tick.timestamp.isoformat(),
    }


async def _build_structure_snapshot(symbol: str, timeframe: str) -> Dict[str, Any]:
    """Build a market-structure snapshot via the deterministic structure engine."""
    from forex_ai_agent.analysis.market_structure.engine import build_structure_snapshot

    candles = await _fetch_candles(symbol, timeframe, 500)
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]
    closes = [float(c["close"]) for c in candles]
    times = [datetime.fromisoformat(c["time"]) for c in candles]

    snap = build_structure_snapshot(timeframe, highs, lows, closes, timestamps=times)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "current_price": snap.current_price,
        "trend": getattr(snap.trend, "value", str(snap.trend)),
        "swing_highs": [
            {"index": s.index, "price": s.price, "strength": s.strength,
             "time": s.timestamp.isoformat() if s.timestamp else None}
            for s in snap.swing_highs[-12:]
        ],
        "swing_lows": [
            {"index": s.index, "price": s.price, "strength": s.strength,
             "time": s.timestamp.isoformat() if s.timestamp else None}
            for s in snap.swing_lows[-12:]
        ],
        "last_bos": (
            {"kind": snap.last_bos.kind,
             "direction": getattr(snap.last_bos.direction, "value", str(snap.last_bos.direction)),
             "level": snap.last_bos.level, "price": snap.last_bos.price,
             "index": snap.last_bos.index, "confidence": snap.last_bos.confidence}
            if snap.last_bos else None
        ),
        "last_choch": (
            {"kind": snap.last_choch.kind,
             "direction": getattr(snap.last_choch.direction, "value", str(snap.last_choch.direction)),
             "level": snap.last_choch.level, "price": snap.last_choch.price,
             "index": snap.last_choch.index, "confidence": snap.last_choch.confidence}
            if snap.last_choch else None
        ),
        "key_support": snap.key_support,
        "key_resistance": snap.key_resistance,
        "liquidity_sweeps": [
            {"index": s.index,
             "direction": getattr(s.direction, "value", str(s.direction)),
             "swept_price": s.swept_price, "depth_atr": s.depth_atr,
             "confidence": s.confidence}
            for s in snap.liquidity_sweeps[-8:]
        ],
        "consolidation": (
            {"top": snap.consolidation.top, "bottom": snap.consolidation.bottom}
            if snap.consolidation is not None and hasattr(snap.consolidation, "top")
            else ({"details": str(snap.consolidation)} if snap.consolidation is not None else None)
        ),
        "retests": [
            {"index": getattr(r, "index", None), "price": getattr(r, "price", None),
             "time": r.timestamp.isoformat() if getattr(r, "timestamp", None) else None}
            for r in snap.retests[-8:]
        ],
    }


async def _compute_indicators(symbol: str, timeframe: str, indicators: List[str]) -> Dict[str, Any]:
    """Compute technical indicators."""
    from forex_ai_agent.analysis.indicators import rsi, ema, macd, atr

    candles = await _fetch_candles(symbol, timeframe, 200)
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]

    result = {}
    if "rsi" in indicators:
        result["rsi"] = float(rsi(closes, period=14)[-1]) if len(closes) >= 14 else None
    if "ema" in indicators:
        result["ema_9"] = float(ema(closes, period=9)[-1]) if len(closes) >= 9 else None
        result["ema_21"] = float(ema(closes, period=21)[-1]) if len(closes) >= 21 else None
    if "macd" in indicators:
        # macd() returns {"macd", "signal", "histogram"} arrays (NaN warm-up).
        macd_out = macd(closes)
        result["macd"] = {
            k: (float(v[-1]) if len(v) and v[-1] == v[-1] else None)
            for k, v in macd_out.items()
        }
    if "atr" in indicators:
        result["atr"] = float(atr(highs, lows, closes, period=14)[-1]) if len(closes) >= 14 else None

    return result


async def _generate_signal(symbol: str, direction: Optional[str], timeframe: Optional[str]) -> List[Dict[str, Any]]:
    """Generate a live signal via the shared Pipeline (mirrors /api/alex/signals/generate)."""
    from forex_ai_agent.interfaces.api.routes.alex import _ensure_pipeline_init

    pipeline = await _ensure_pipeline_init()
    pair = (symbol or "EUR/USD").upper()
    tf = (timeframe or "H1").upper()
    dir_name = (direction or "LONG").upper()
    if dir_name not in {"LONG", "SHORT"}:
        dir_name = "LONG"
    try:
        result = await asyncio.wait_for(
            pipeline.generate_signal(pair, dir_name, tf, narrate=False), timeout=90.0
        )
    except Exception as exc:  # noqa: BLE001 - surface as an honest observation
        return [{"status": "error", "message": f"signal generation failed: {exc}"}]
    if result.get("status") != "ok":
        return [{
            "status": result.get("status", "no_signal"),
            "message": result.get("message", "no signal formed"),
        }]
    cand = result.get("signal_candidate") or {}
    ev = cand.get("evidence") or {}
    risk = cand.get("risk") or {}
    return [{
        "status": cand.get("status", "formed"),
        "pair": result.get("pair", pair),
        "direction": cand.get("direction"),
        "timeframe": cand.get("timeframe", tf),
        "setup": cand.get("setup_type"),
        "entry_zone": [cand.get("entry_zone_low"), cand.get("entry_zone_high")],
        "stop_loss": cand.get("stop_loss"),
        "take_profit_1": cand.get("take_profit_1"),
        "risk_reward": cand.get("risk_reward"),
        "score_pct": cand.get("score_pct"),
        "invalidation": cand.get("invalidation_condition"),
        "risk_passed": risk.get("passed"),
        "data_source": result.get("data_source"),
        "spread_pips": result.get("spread_pips"),
        "session": result.get("session"),
        "evidence_supporting": (ev.get("supporting") or [])[:6],
        "evidence_contradicting": (ev.get("contradicting") or [])[:6],
        "evidence_missing": (ev.get("missing") or [])[:6],
    }]


_news_agent_singleton = None


async def _get_news_agent():
    """Lazily create (once) the calendar NewsAgent."""
    global _news_agent_singleton
    if _news_agent_singleton is None:
        from forex_ai_agent.news.agent import create_news_agent

        _news_agent_singleton = await create_news_agent()
    return _news_agent_singleton


def _impact_str(v: Any) -> str:
    return getattr(v, "value", str(v))


async def _get_risk_state() -> Dict[str, Any]:
    """Current per-currency news risk + imminent calendar events (real NewsAgent)."""
    try:
        agent = await _get_news_agent()
        states = agent.get_all_risk_states()
        currencies = [
            {
                "currency": st.currency,
                "risk_level": _impact_str(st.risk_level).lower(),
                "next_event": st.next_event_title,
                "minutes_to_next_event": st.minutes_to_next_event,
            }
            for st in states.values()
        ]
        rank = {"none": 0, "low": 1, "medium": 2, "high": 3}
        overall = max(
            (c["risk_level"] for c in currencies),
            key=lambda lvl: rank.get(lvl, 0),
            default="none",
        )
        imminent = agent.get_imminent_events(minutes_ahead=60)
        events = [
            {
                "title": e.title,
                "currency": e.currency,
                "impact": _impact_str(e.impact),
                "scheduled": e.scheduled_time.isoformat(),
            }
            for e in imminent
        ]
        return {
            "risk_level": overall,
            "currencies": currencies,
            "upcoming_events": events,
            "daily_loss_pct": 0.0,   # live-gateway figures pending execution layer
            "open_positions": 0,
        }
    except Exception as exc:  # noqa: BLE001 — never crash the tool
        return {
            "risk_level": "unknown",
            "currencies": [],
            "upcoming_events": [],
            "daily_loss_pct": 0.0,
            "open_positions": 0,
            "note": f"news agent unavailable: {exc}",
        }


async def _get_news(symbol: Optional[str], hours: int) -> List[Dict[str, Any]]:
    """Upcoming calendar events (optionally filtered to a pair's currencies)."""
    try:
        agent = await _get_news_agent()
        events = agent.get_high_impact_upcoming(hours_ahead=max(1, hours))
        pair_curs = symbol.upper().replace("/", "") if symbol else ""
        out = []
        for e in events:
            if pair_curs and e.currency and e.currency.upper() not in pair_curs:
                continue
            out.append(
                {
                    "title": e.title,
                    "currency": e.currency,
                    "impact": _impact_str(e.impact),
                    "scheduled": e.scheduled_time.isoformat(),
                    "forecast": e.forecast,
                    "previous": e.previous,
                }
            )
        return out
    except Exception as exc:  # noqa: BLE001
        return [{"title": f"calendar unavailable: {exc}", "currency": "", "impact": "NONE", "scheduled": ""}]


async def _search_strategies(query: Optional[str], status: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """Search strategy versions."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    versions = vm.list_versions()

    filtered = []
    for v in versions:
        if status and v.status != status:
            continue
        if query and query.lower() not in (v.name + " " + (v.description or "")).lower():
            continue
        filtered.append({
            "id": v.id,
            "version": v.version,
            "name": v.name,
            "description": v.description,
            "status": v.status,
            "parameters": v.parameters,
            "created_at": v.created_at.isoformat() if v.created_at else None,
        })
    return filtered[:limit]


async def _get_strategy(version_id: Optional[str], version_string: Optional[str]) -> Optional[Dict[str, Any]]:
    """Get a specific strategy version."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    if version_id:
        v = vm.get_version(version_id)
    elif version_string:
        v = vm.get_version_by_string(version_string)
    else:
        return None

    if not v:
        return None

    return {
        "id": v.id,
        "version": v.version,
        "name": v.name,
        "description": v.description,
        "status": v.status,
        "rules": v.rules,
        "parameters": v.parameters,
        "created_at": v.created_at.isoformat() if v.created_at else None,
    }


async def _list_probation() -> List[Dict[str, Any]]:
    """List probation records."""
    from forex_ai_agent.autonomy.probation import ProbationStore

    store = ProbationStore()
    records = await store.list()
    return [
        {
            "version_id": r.version_id,
            "strategy_name": r.strategy_name,
            "pair": r.pair,
            "status": r.status,
            "observed_trades": r.observed_trades,
            "required_trades": r.required_trades,
            "eligible_for_approval": r.eligible_for_approval,
            "started_at": r.started_at,
        }
        for r in records
    ]


async def _list_research_history(limit: int) -> List[Dict[str, Any]]:
    """List recent research tasks."""
    from forex_ai_agent.autonomy.research_log import recent_research_tasks

    tasks = await recent_research_tasks(limit=limit)
    return [
        {
            "id": t.get("id"),
            "topic": t.get("topic"),
            "status": t.get("status"),
            "verdict": t.get("verdict"),
            "conclusion": t.get("conclusion"),
            "created_at": t.get("created"),
        }
        for t in tasks
    ]


# ---------- Tool handlers (async wrappers) ----------

async def get_market_data(args: MarketDataArgs) -> MarketDataResult:
    candles = await _fetch_candles(args.symbol, args.timeframe, args.limit)
    return MarketDataResult(
        symbol=args.symbol,
        timeframe=args.timeframe,
        candles=candles,
        count=len(candles),
    )


async def get_current_price(args: CurrentPriceArgs) -> CurrentPriceResult:
    data = await _fetch_current_price(args.symbol)
    return CurrentPriceResult(**data)


async def analyze_structure(args: StructureArgs) -> StructureResult:
    data = await _build_structure_snapshot(args.symbol, args.timeframe)
    return StructureResult(**data)


async def get_indicators(args: IndicatorsArgs) -> IndicatorsResult:
    data = await _compute_indicators(args.symbol, args.timeframe, args.indicators)
    return IndicatorsResult(
        symbol=args.symbol,
        timeframe=args.timeframe,
        **data,
    )


async def get_active_signals(args: ActiveSignalsArgs) -> ActiveSignalsResult:
    signals = await _generate_signal(args.symbol, args.direction, args.timeframe)
    return ActiveSignalsResult(signals=signals)


async def get_risk_state(args: RiskStateArgs) -> RiskStateResult:
    data = await _get_risk_state()
    return RiskStateResult(**data)


async def get_news(args: NewsArgs) -> NewsResult:
    items = await _get_news(args.symbol, args.hours)
    return NewsResult(items=items)


async def search_strategies(args: SearchStrategiesArgs) -> SearchStrategiesResult:
    strategies = await _search_strategies(args.query, args.status, args.limit)
    return SearchStrategiesResult(strategies=strategies)


async def get_strategy(args: GetStrategyArgs) -> GetStrategyResult:
    strategy = await _get_strategy(args.version_id, args.version_string)
    return GetStrategyResult(strategy=strategy)


async def list_probation(args: ListProbationArgs) -> ListProbationResult:
    records = await _list_probation()
    return ListProbationResult(records=records)


async def list_research_history(args: ListResearchHistoryArgs) -> ListResearchHistoryResult:
    records = await _list_research_history(args.limit)
    return ListResearchHistoryResult(records=records)


# ---------- Action tool handlers (Phase 2) ----------

async def _start_research(args: StartResearchArgs) -> StartResearchResult:
    """Start a structured research plan (B5) — persisted, Director-backed.

    Replaces the old in-memory orchestrator session: the plan row (with its
    experiment/validation checklists) is persisted, the pairs are enqueued
    with the Research Director, and progress self-reports from stored records
    as the Director works the plan. Accepts the SAME inputs as before, so
    existing callers keep working; ``proposal`` now carries the plan card.
    """
    from forex_ai_agent.agent.research_plan import create_research_plan

    pairs = [args.pair] if args.pair else ["EUR/USD"]
    try:
        created = await create_research_plan(
            topic=args.topic,
            pairs=pairs,
            timeframes=[args.timeframe] if args.timeframe else None,
            hypothesis_text=args.topic,
            enqueued=True,
        )
    except ValueError as exc:
        return StartResearchResult(task_id="", status="error", proposal={"error": str(exc)})
    return StartResearchResult(
        task_id=created["task_id"],
        status="started",
        proposal=created["card"],
    )


async def start_research(args: StartResearchArgs) -> StartResearchResult:
    """Start research - LOW risk, executes immediately."""
    return await _start_research(args)


async def _research_plan_query(task_id: str, limit: int) -> Dict[str, Any]:
    """One plan card by id, or the recent-plans index when no id is given."""
    from forex_ai_agent.agent.research_plan import (
        get_research_plan,
        list_research_plans,
        render_plan_card,
    )

    if task_id:
        card = await get_research_plan(task_id)
        if card is None:
            return {"found": False, "task_id": task_id}
        return {
            "found": True,
            "plans": [card],
            "rendered": render_plan_card(card),
        }
    plans = await list_research_plans(limit=max(1, min(int(limit), 50)))
    return {
        "found": bool(plans),
        "plans": plans,
        "rendered": "\n\n".join(render_plan_card(p) for p in plans),
    }


async def _remove_strategy(args: RemoveStrategyArgs) -> RemoveStrategyResult:
    """Remove a strategy version. Medium risk - requires confirmation."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    version = vm.get_version(args.version_id)
    if not version:
        return RemoveStrategyResult(
            success=False,
            version_id=args.version_id,
        )

    # Medium risk: return proposal for confirmation
    policy = get_confirmation_policy()
    token = policy.create_token(
        "remove_strategy",
        f"Remove strategy {args.version_id} ({version.name})"
    )
    proposal = {
        "token": token.token,
        "expires_in_seconds": token.ttl_seconds,
        "impact": f"Permanently removes strategy {version.name} v{version.version}",
        "confirm_url": f"/api/agent/confirm?token={token.token}",
    }
    return RemoveStrategyResult(
        success=False,  # Not executed yet
        version_id=args.version_id,
        proposal=proposal,
    )


async def remove_strategy(args: RemoveStrategyArgs) -> RemoveStrategyResult:
    """Remove strategy - MEDIUM risk, requires confirmation."""
    return await _remove_strategy(args)


async def _archive_strategy(args: ArchiveStrategyArgs) -> ArchiveStrategyResult:
    """Archive a strategy version. Medium risk - requires confirmation."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    version = vm.get_version(args.version_id)
    if not version:
        return ArchiveStrategyResult(
            success=False,
            version_id=args.version_id,
        )

    policy = get_confirmation_policy()
    token = policy.create_token(
        "archive_strategy",
        f"Archive strategy {args.version_id} ({version.name})"
    )
    proposal = {
        "token": token.token,
        "expires_in_seconds": token.ttl_seconds,
        "impact": f"Archives strategy {version.name} v{version.version} (reversible)",
        "confirm_url": f"/api/agent/confirm?token={token.token}",
    }
    return ArchiveStrategyResult(
        success=False,
        version_id=args.version_id,
        proposal=proposal,
    )


async def archive_strategy(args: ArchiveStrategyArgs) -> ArchiveStrategyResult:
    """Archive strategy - MEDIUM risk, requires confirmation."""
    return await _archive_strategy(args)


async def _reject_strategy(args: RejectStrategyArgs) -> RejectStrategyResult:
    """Reject a strategy version. Medium risk - requires confirmation."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    version = vm.get_version(args.version_id)
    if not version:
        return RejectStrategyResult(
            success=False,
            version_id=args.version_id,
        )

    policy = get_confirmation_policy()
    token = policy.create_token(
        "reject_strategy",
        f"Reject strategy {args.version_id} ({version.name}): {args.reason}"
    )
    proposal = {
        "token": token.token,
        "expires_in_seconds": token.ttl_seconds,
        "impact": f"Rejects strategy {version.name} v{version.version} with reason: {args.reason}",
        "confirm_url": f"/api/agent/confirm?token={token.token}",
    }
    return RejectStrategyResult(
        success=False,
        version_id=args.version_id,
        proposal=proposal,
    )


async def reject_strategy(args: RejectStrategyArgs) -> RejectStrategyResult:
    """Reject strategy - MEDIUM risk, requires confirmation."""
    return await _reject_strategy(args)


async def _create_strategy_draft(args: CreateStrategyDraftArgs) -> CreateStrategyDraftResult:
    """Create a strategy draft. Low risk - ASSIST level, executes immediately."""
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    # Create as draft (not active)
    draft = vm.create_version(
        name=args.name,
        description=args.description,
        rules=args.rules,
        parameters=args.parameters,
    )
    return CreateStrategyDraftResult(
        draft_id=draft.id,
        status="draft",
    )


async def create_strategy_draft(args: CreateStrategyDraftArgs) -> CreateStrategyDraftResult:
    """Create strategy draft - LOW risk, ASSIST level."""
    return await _create_strategy_draft(args)


# ---------- Experiment tool handlers (Phase 3) ----------

async def backtest_strategy_tool(args: BacktestStrategyArgs) -> BacktestStrategyResult:
    """Run backtest for a strategy - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import backtest_strategy

    result = await backtest_strategy(
        strategy_id=args.strategy_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        initial_balance=args.initial_balance,
    )
    return BacktestStrategyResult(**result)


async def walk_forward_test_tool(args: WalkForwardTestArgs) -> WalkForwardTestResult:
    """Run walk-forward test - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import walk_forward_test

    result = await walk_forward_test(
        strategy_id=args.strategy_id,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        window_months=args.window_months,
        step_months=args.step_months,
        run_parameter_grid=args.run_parameter_grid,
        parameter_grid=args.parameter_grid,
    )
    return WalkForwardTestResult(**result)


class GetStrategyHealthArgs(BaseModel):
    """Strategy Health card from stored state (Stage B1)."""

    strategy_id: str = Field(description="Strategy version ID")


class GetStrategyHealthResult(BaseModel):
    strategy_id: str
    strategy_name: str = ""
    pair: Optional[str] = None
    timeframe: Optional[str] = None
    badge: str = ""
    summary: str = ""
    health: Dict[str, Any] = Field(default_factory=dict)
    card: str = ""


class RunRobustnessSweepArgs(BaseModel):
    """Cross-pair / regime robustness sweep (Stage B3)."""

    strategy_id: str = Field(description="Strategy version ID")
    pairs: Optional[List[str]] = Field(
        default=None,
        description="Pairs to sweep (default: strategy pair + majors, max 4)",
    )
    start_date: Optional[str] = Field(default=None, description="Sweep start (ISO, default ~2y back)")
    end_date: Optional[str] = Field(default=None, description="Sweep end (ISO, default now)")
    timeframe: Optional[str] = Field(default=None, description="Override timeframe (default stored)")
    window_months: int = Field(default=6, description="IS window months per WF")
    step_months: int = Field(default=2, description="Step months per WF")
    save: bool = Field(default=True, description="Store the sweep on the strategy version")


class RunRobustnessSweepResult(BaseModel):
    strategy_id: str
    strategy_name: str = ""
    base_pair: str = ""
    timeframe: str = ""
    verdict: str = ""
    pairs: List[Dict[str, Any]] = Field(default_factory=list)
    pairs_summary: Dict[str, Any] = Field(default_factory=dict)
    per_regime: Dict[str, Any] = Field(default_factory=dict)
    regimes_summary: Dict[str, Any] = Field(default_factory=dict)
    budget: Dict[str, Any] = Field(default_factory=dict)
    saved: bool = False
    save_note: str = ""
    card: str = ""


class ExplainOverfitArgs(BaseModel):
    """Explainable overfit components (Stage A2)."""

    strategy_id: str = Field(description="Strategy version ID")


class ExplainOverfitResult(BaseModel):
    strategy_id: str
    strategy_name: str = ""
    title: str = "OVERFIT ANALYSIS"
    subtitle: str = ""
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    concerns: List[str] = Field(default_factory=list)
    recommendation: str = ""
    verdict: str = ""
    components: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class ReplayReproArgs(BaseModel):
    """Replay a stored reproducibility record (Stage A1)."""

    strategy_id: str = Field(description="Strategy version ID to replay")
    symbol: str = Field(description="Forex pair symbol (must match the original run)")
    start_date: str = Field(description="Original dataset start (ISO format)")
    end_date: str = Field(description="Original dataset end (ISO format)")
    timeframe: str = Field(default="H1", description="Original timeframe")
    include_walk_forward: bool = Field(
        default=False, description="Also replay the walk-forward analysis"
    )
    window_months: int = Field(default=6, description="IS window months (WF replay)")
    step_months: int = Field(default=2, description="Step forward months (WF replay)")


class ReplayReproResult(BaseModel):
    strategy_id: str
    replayed: bool = True
    metrics: Dict[str, Any]
    matches: Optional[bool] = Field(
        default=None,
        description="True/False vs stored gate numbers; None when nothing stored to compare",
    )
    matched_fields: Dict[str, bool] = Field(default_factory=dict)
    differences: List[str] = Field(default_factory=list)
    note: str = ""


async def explain_overfit_tool(args: "ExplainOverfitArgs") -> "ExplainOverfitResult":
    """Component-by-component overfit explanation from stored WF windows."""
    from forex_ai_agent.agent.explain import Explainer

    explainer = Explainer()
    card = await explainer.overfit_card(args.strategy_id)
    components: Dict[str, Any] = {}
    score: Optional[float] = None
    try:
        from forex_ai_agent.strategy import create_version_manager

        vm = await create_version_manager()
        version = vm.get_version(args.strategy_id) or (
            vm.get_version(args.strategy_id[:8]) if len(args.strategy_id) >= 8 else None
        )
        gate = dict(getattr(version, "backtest_results", None) or {}) if version else {}
        analysis = dict(((gate.get("walk_forward") or {}).get("overfit_analysis")) or {})
        if analysis:
            components = dict(analysis.get("components") or {})
            score = analysis.get("score")
    except Exception:  # noqa: BLE001 — card already carries the evidence
        pass
    return ExplainOverfitResult(
        strategy_id=args.strategy_id,
        strategy_name=(card.subtitle.split(" — ") or [""])[0],
        title=card.title,
        subtitle=card.subtitle,
        evidence=[
            {"check": e.check, "status": e.status, "detail": e.detail,
             "metric_value": e.metric_value, "threshold": e.threshold}
            for e in card.evidence
        ],
        concerns=card.concerns,
        recommendation=card.recommendation,
        verdict=card.verdict,
        components=components,
        score=score,
    )


async def replay_repro_tool(args: ReplayReproArgs) -> ReplayReproResult:
    """Re-run a stored repro and compare against the stored gate numbers."""
    from forex_ai_agent.agent.repro import replay_spec_run, rules_hash

    vm = await create_version_manager()
    strategy = vm.get_version(args.strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {args.strategy_id} not found")

    result = await replay_spec_run(
        strategy.rules,
        args.symbol,
        args.timeframe,
        args.start_date,
        args.end_date,
        want_walk_forward=args.include_walk_forward,
        window_months=args.window_months,
        step_months=args.step_months,
    )

    stored_gate: Dict[str, Any] = {}
    try:
        stored_gate = dict(getattr(strategy, "backtest_results", None) or {})
    except Exception:  # noqa: BLE001
        stored_gate = {}

    comparisons = [
        ("total_trades", "total_trades"),
        ("profit_factor", "profit_factor"),
        ("sharpe_ratio", "sharpe_ratio"),
        ("win_rate", "win_rate"),
        ("max_drawdown", "max_drawdown_pct"),
    ]
    matched_fields: Dict[str, bool] = {}
    differences: List[str] = []
    replay_metrics = dict(result.get("metrics") or {})
    for replay_key, stored_key in comparisons:
        if stored_key not in stored_gate:
            continue
        expected = float(stored_gate.get(stored_key) or 0.0)
        actual = float(replay_metrics.get(replay_key) or 0.0)
        ok = abs(expected - actual) <= 1e-6 * max(1.0, abs(expected))
        matched_fields[replay_key] = ok
        if not ok:
            differences.append(f"{replay_key}: replay {actual:.4f} vs stored {expected:.4f}")

    matches: Optional[bool] = None
    note = ""
    if matched_fields:
        matches = all(matched_fields.values())
        note = (
            "replay matches the stored gate numbers exactly"
            if matches
            else "replay DIVERGES from stored numbers — dataset or rules changed since the gate ran"
        )
    else:
        note = "no stored gate numbers on this strategy version — replay recorded, nothing to compare"

    repro_block = stored_gate.get("repro") if isinstance(stored_gate, dict) else None
    stored_hash = (
        (repro_block or {}).get("strategy", {}).get("rules_sha256")
        if isinstance(repro_block, dict)
        else None
    )
    current_hash = rules_hash(strategy.rules)
    if stored_hash and stored_hash != current_hash:
        differences.append(
            f"rules hash changed since storage ({stored_hash[:12]}… → {current_hash[:12]}…)"
        )
        matches = False

    return ReplayReproResult(
        strategy_id=args.strategy_id,
        replayed=True,
        metrics=replay_metrics,
        matches=matches,
        matched_fields=matched_fields,
        differences=differences,
        note=note,
    )


async def compare_strategies_tool(args: CompareStrategiesArgs) -> CompareStrategiesResult:
    """Compare multiple strategies - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import compare_strategies

    result = await compare_strategies(
        strategy_ids=args.strategy_ids,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
    )
    return CompareStrategiesResult(**result)


async def create_experiment_tool(args: CreateExperimentArgs) -> CreateExperimentResult:
    """Create and run parameter experiment - MEDIUM risk, ASSIST level."""
    from forex_ai_agent.agent.experiments import create_experiment

    result = await create_experiment(
        strategy_id=args.strategy_id,
        param_mutations=args.param_mutations,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        hypothesis_id=args.hypothesis_id,
    )
    return CreateExperimentResult(**result)


async def list_experiments_tool(args: ListExperimentsArgs) -> ListExperimentsResult:
    """List experiment records - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import list_experiments

    exps = await list_experiments(
        strategy_id=args.strategy_id,
        limit=args.limit,
    )
    return ListExperimentsResult(experiments=exps)


async def get_experiment_tool(args: GetExperimentArgs) -> GetExperimentResult:
    """Get experiment details - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import get_experiment

    exp = await get_experiment(args.experiment_id)
    return GetExperimentResult(experiment=exp)


async def conclude_experiment_tool(args: ConcludeExperimentArgs) -> ConcludeExperimentResult:
    """Record experiment decision - MEDIUM risk, ASSIST level."""
    from forex_ai_agent.agent.experiments import conclude_experiment

    result = await conclude_experiment(
        experiment_id=args.experiment_id,
        decision=args.decision,
        notes=args.notes,
    )
    return ConcludeExperimentResult(**result)


# ---------- Phase 5 tool handlers ----------

async def check_viability_tool(args: CheckViabilityArgs) -> CheckViabilityResult:
    """Check strategy viability - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.experiments import check_strategy_viability

    report = await check_strategy_viability(
        strategy_id=args.strategy_id,
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback_days=args.lookback_days,
    )
    return CheckViabilityResult(
        strategy_id=report.strategy_id,
        strategy_name=report.strategy_name,
        symbol=report.symbol,
        timeframe=report.timeframe,
        baseline_metrics=report.baseline_metrics,
        current_metrics=report.current_metrics,
        drift=report.drift,
        oos_pf_drift=report.oos_pf_drift,
        dd_change_pct=report.dd_change_pct,
        concentration_risk=report.concentration_risk,
        verdict=report.verdict,
        concerns=report.concerns,
        disclaimer=report.disclaimer,
    )


async def find_strategies_tool(args: FindStrategiesArgs) -> FindStrategiesResult:
    """Find promising strategies - MEDIUM risk, ASSIST level (bounded autonomous research)."""
    from forex_ai_agent.agent.experiments import find_promising_strategies

    result = await find_promising_strategies(
        pairs=args.pairs,
        timeframes=args.timeframes,
        days=args.days,
        min_profit_factor=args.min_profit_factor,
        min_trades=args.min_trades,
        max_drawdown_pct=args.max_drawdown_pct,
        max_concurrent=args.max_concurrent,
    )
    return FindStrategiesResult(
        candidates_found=result["candidates_found"],
        ranked_strategies=result["ranked_strategies"],
        errors=result["errors"],
        disclaimer=result["disclaimer"],
    )


# ---------- Phase 6 tool handlers: Explain / "Why?" ----------

async def explain_probation_tool(args: ExplainProbationArgs) -> ExplainProbationResult:
    """Generate EXPLAIN card for a probation record - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import explain_probation

    result = await explain_probation({"version_id": args.version_id})
    return ExplainProbationResult(**result)


async def explain_research_tool(args: ExplainResearchArgs) -> ExplainResearchResult:
    """Generate EXPLAIN card for a research task - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import explain_research

    result = await explain_research({"task_id": args.task_id})
    return ExplainResearchResult(**result)


async def explain_gate_tool(args: ExplainGateArgs) -> ExplainGateResult:
    """Explain gate rejection for a pair - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import explain_gate

    result = await explain_gate({"pair": args.pair, "days": args.days})
    return ExplainGateResult(**result)


async def get_strategy_health_tool(args: GetStrategyHealthArgs) -> GetStrategyHealthResult:
    """Strategy Health card from stored state - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.health import get_strategy_health, render_health_text

    result = await get_strategy_health(args.strategy_id)
    return GetStrategyHealthResult(
        strategy_id=result["strategy_id"],
        strategy_name=result.get("strategy_name", ""),
        pair=result.get("pair"),
        timeframe=result.get("timeframe"),
        badge=result["health"].get("badge", ""),
        summary=result.get("summary", ""),
        health=result["health"],
        card=render_health_text(result),
    )


async def run_robustness_sweep_tool(args: RunRobustnessSweepArgs) -> RunRobustnessSweepResult:
    """Run the cross-pair/regime sweep - MEDIUM risk, ASSIST level (heavy)."""
    from forex_ai_agent.agent.robustness import run_robustness_sweep, render_robustness_text

    result = await run_robustness_sweep(
        strategy_id=args.strategy_id,
        pairs=args.pairs,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        window_months=args.window_months,
        step_months=args.step_months,
        save=args.save,
    )
    saved = "save_error" not in result
    return RunRobustnessSweepResult(
        strategy_id=result["strategy_id"],
        strategy_name=result.get("strategy_name", ""),
        base_pair=result.get("base_pair", ""),
        timeframe=result.get("timeframe", ""),
        verdict=result.get("verdict", ""),
        pairs=result.get("pairs", []),
        pairs_summary=result.get("pairs_summary", {}),
        per_regime=result.get("per_regime", {}),
        regimes_summary=result.get("regimes_summary", {}),
        budget=result.get("budget", {}),
        saved=saved,
        save_note=result.get("save_error", ""),
        card=render_robustness_text(result),
    )


async def hypothesis_test_count_tool(
    args: HypothesisTestCountArgs,
) -> HypothesisTestCountResult:
    """Count stored tests of a hypothesis - LOW risk, OBSERVE level (B2)."""
    from forex_ai_agent.agent.research_graph import hypothesis_test_count

    data = await hypothesis_test_count(args.hypothesis_id)
    return HypothesisTestCountResult(**data)


async def aggregate_by_parameter_tool(
    args: AggregateByParameterArgs,
) -> AggregateByParameterResult:
    """Aggregate stored outcomes by parameter value - LOW risk, OBSERVE level (B2)."""
    from forex_ai_agent.agent.research_graph import aggregate_by_parameter

    data = await aggregate_by_parameter(args.param_pattern)
    return AggregateByParameterResult(**data)


async def research_graph_summary_tool(
    args: ResearchGraphSummaryArgs,
) -> ResearchGraphSummaryResult:
    """Research graph overview - LOW risk, OBSERVE level (B2)."""
    from forex_ai_agent.agent.research_graph import graph_summary

    data = await graph_summary()
    return ResearchGraphSummaryResult(edges=data["edges"], note="stored variant→outcome edges")


def _render_director_cycle(report: Dict[str, Any]) -> str:
    """Deterministic text card for a cycle report."""
    lines = [
        f"RESEARCH CYCLE {report.get('cycle_id', '?')}",
        (
            f"  Ran {report.get('items_run', 0)} item(s): "
            f"{report.get('validated', 0)} validated, {report.get('rejected', 0)} rejected, "
            f"{report.get('errors', 0)} errors"
        ),
        f"  LLM calls used: {report.get('llm_calls_used', 0)} (deterministic fallback thereafter)",
    ]
    for r in report.get("results", []):
        line = f"  {r.get('pair', '?')}: {r.get('status', '?')}"
        if r.get("version_id"):
            line += f" → version {r['version_id']} (probation started)"
        if r.get("error"):
            line += f" — {r['error']}"
        lines.append(line)
    if report.get("budget_exhausted"):
        lines.append("  Budget exhausted — remaining items stay queued for the next cycle.")
    lines.append("  No live execution. No risk changes. Human approval still required.")
    return "\n".join(lines)


def _render_director_status(snap: Dict[str, Any]) -> str:
    """Deterministic text card for the queue snapshot."""
    bs = snap.get("by_status", {})
    lines = [
        "RESEARCH DIRECTOR QUEUE",
        f"  Open: {snap.get('open_items', 0)}  "
        f"(queued {bs.get('QUEUED', 0)}, running {bs.get('RUNNING', 0)}, "
        f"validated {bs.get('DONE_VALIDATED', 0)}, rejected {bs.get('DONE_REJECTED', 0)}, "
        f"errors {bs.get('DONE_ERROR', 0)}, cancelled {bs.get('CANCELLED', 0)})",
    ]
    for item in snap.get("queue", [])[:8]:
        lines.append(
            f"  [{item.get('priority', '?')}] {item.get('pair')} {item.get('timeframe')} "
            f"— {item.get('status')}"
        )
    ev = snap.get("recent_events", [])[:3]
    if ev:
        lines.append("  Recent:")
        for e in ev:
            lines.append(f"    {e.get('created_at', '?')[:19]} {e.get('action')} {e.get('pair') or ''}")
    return "\n".join(lines)


async def research_plan_tool(args: ResearchPlanArgs) -> ResearchPlanResult:
    """Structured research plan card (RESEARCH #id) — LOW risk, OBSERVE (B5).

    Accepts the full id or an unambiguous prefix; progress comes only from
    persisted records (Director rows joined on plan_id).
    """
    from forex_ai_agent.agent.research_plan import (
        get_research_plan,
        list_research_plans,
        render_plan_card,
    )
    from forex_ai_agent.data.database import get_session_factory

    if args.task_id:
        from sqlalchemy import select as _select

        from forex_ai_agent.agent.research_plan import ResearchPlan

        prefix = str(args.task_id).strip().lower()
        factory = get_session_factory()
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        _select(ResearchPlan.id).order_by(ResearchPlan.created_at.desc()).limit(200)
                    )
                )
                .scalars()
                .all()
            )
        matches = [r for r in rows if str(r).lower().startswith(prefix)]
        if not matches:
            return ResearchPlanResult(found=False, task_id=args.task_id)
        if len(matches) > 1:
            return ResearchPlanResult(
                found=False,
                task_id=args.task_id,
                rendered=f"ambiguous plan prefix '{args.task_id}' matches {len(matches)} plans — use more characters",
            )
        card = await get_research_plan(str(matches[0]))
        if card is None:
            return ResearchPlanResult(found=False, task_id=args.task_id)
        return ResearchPlanResult(found=True, plans=[card], rendered=render_plan_card(card), task_id=str(matches[0]))

    plans = await list_research_plans(limit=max(1, min(int(args.limit), 50)))
    return ResearchPlanResult(
        found=bool(plans),
        plans=plans,
        rendered="\n\n".join(render_plan_card(p) for p in plans),
    )


async def generate_hypotheses_tool(args: GenerateHypothesesArgs) -> GenerateHypothesesResult:
    """LLM proposes what to research; deterministic code validates and enqueues
    into the Director queue - MEDIUM risk, ASSIST level (Stage C1)."""
    from forex_ai_agent.agent.hypothesis_gen import HypothesisGenConfig, generate_hypotheses

    config = HypothesisGenConfig(
        pairs=args.pairs,
        timeframe=args.timeframe,
        max_hypotheses=args.max_hypotheses,
        min_confidence=args.min_confidence,
        use_llm=args.use_llm,
        enqueue=args.enqueue,
    )
    report = await generate_hypotheses(config)
    return GenerateHypothesesResult(report=report)


async def review_promotion_tool(args: ReviewPromotionArgs) -> ReviewPromotionResult:
    """Deterministic probation promotion review: rank candidates whose gates are
    MET and package them for the human decision - LOW risk, OBSERVE (Stage C4).
    Never promotes; approve/reject stay human-only."""
    from forex_ai_agent.agent.probation_promotion import PromotionConfig, review_promotions

    config = PromotionConfig(
        limit=args.limit,
        include_narration=args.include_narration,
        include_expired=args.include_expired,
    )
    review = await review_promotions(config)
    return ReviewPromotionResult(review=review)


async def probation_monitor_tool(args: ProbationMonitorArgs) -> ProbationMonitorResult:
    """Deterministic probation health snapshot - LOW risk, OBSERVE (Stage D3).
    Read-only: flags and projections are arithmetic over stored records;
    never mutates probation state, never approves or rejects."""
    from forex_ai_agent.autonomy.probation_monitor import ProbationMonitor

    monitor = ProbationMonitor()
    reports = await monitor.monitor_once(publish=False)
    if args.version_id:
        reports = [
            r
            for r in reports
            if r["version_id"] == args.version_id
            or r["version_id"].startswith(args.version_id)
        ]
    shown = reports[: args.limit]
    return ProbationMonitorResult(
        summary=ProbationMonitor.summarize(reports),
        probations=shown,
        rendered=ProbationMonitor.render_reports(shown),
    )


async def select_experiments_tool(args: SelectExperimentsArgs) -> SelectExperimentsResult:
    """LLM chooses which experiment to run next; deterministic code validates and
    executes via the Phase-3 engines - MEDIUM risk, ASSIST level (Stage C2)."""
    from forex_ai_agent.agent.experiment_selector import ExperimentSelConfig, select_experiments

    config = ExperimentSelConfig(
        strategies=args.strategies,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        max_experiments=args.max_experiments,
        max_variants=args.max_variants,
        window_months=args.window_months,
        step_months=args.step_months,
        max_oos_windows=args.max_oos_windows,
        use_llm=args.use_llm,
        dry_run=args.dry_run,
    )
    report = await select_experiments(config)
    return SelectExperimentsResult(report=report)


async def generate_report_tool(args: GenerateReportArgs) -> GenerateReportResult:
    """LLM narrates a research report over stored evidence; code renders every
    number from the records and validates traceability - MEDIUM risk, ASSIST level (Stage C3)."""
    from forex_ai_agent.agent.report_gen import ReportConfig, generate_research_report

    config = ReportConfig(
        topic=args.topic,
        experiment_ids=args.experiment_ids,
        strategy_ids=args.strategy_ids,
        limit_experiments=args.limit_experiments,
        use_llm=args.use_llm,
        persist=args.persist,
    )
    doc = await generate_research_report(config)
    return GenerateReportResult(report=doc)


async def director_enqueue_tool(args: DirectorEnqueueArgs) -> DirectorEnqueueResult:
    """Enqueue a research hypothesis - MEDIUM risk, ASSIST level (B4)."""
    from forex_ai_agent.agent.director import enqueue_hypothesis

    data = await enqueue_hypothesis(
        pair=args.pair,
        hypothesis_text=args.hypothesis,
        timeframe=args.timeframe,
        priority=args.priority,
    )
    return DirectorEnqueueResult(**data)


async def director_run_cycle_tool(args: DirectorRunCycleArgs) -> DirectorRunCycleResult:
    """Run one unattended research cycle - MEDIUM risk, ASSIST level (B4)."""
    from forex_ai_agent.agent.director import DirectorConfig, run_cycle

    config = DirectorConfig(
        max_items_per_cycle=max(1, args.max_items),
        max_llm_calls_per_cycle=max(0, args.max_llm_calls),
        use_llm=args.use_llm,
    )
    report = await run_cycle(config)
    return DirectorRunCycleResult(
        cycle_id=report["cycle_id"],
        items_run=report["items_run"],
        validated=report["validated"],
        rejected=report["rejected"],
        errors=report["errors"],
        llm_calls_used=report["llm_calls_used"],
        budget_exhausted=report["budget_exhausted"],
        results=report["results"],
        card=_render_director_cycle(report),
    )


async def director_status_tool(args: DirectorStatusArgs) -> DirectorStatusResult:
    """Research queue status - LOW risk, OBSERVE level (B4)."""
    from forex_ai_agent.agent.director import queue_snapshot

    snap = await queue_snapshot()
    return DirectorStatusResult(
        queue=snap["queue"],
        open_items=snap["open_items"],
        by_status=snap["by_status"],
        recent_events=snap["recent_events"],
        card=_render_director_status(snap),
    )


async def answer_why_tool(args: AnswerWhyArgs) -> AnswerWhyResult:
    """Answer 'why?' question using knowledge retriever - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import answer_why

    result = await answer_why({"question": args.question})
    return AnswerWhyResult(**result)


async def search_knowledge_tool(args: SearchKnowledgeArgs) -> SearchKnowledgeResult:
    """Search research knowledge base - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import search_knowledge

    result = await search_knowledge({"query": args.query, "limit": args.limit, "filters": args.filters})
    return SearchKnowledgeResult(**result)


async def get_strategy_history_tool(args: GetStrategyHistoryArgs) -> GetStrategyHistoryResult:
    """Get all research/experiments for a strategy name - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import get_strategy_history

    result = await get_strategy_history({"strategy_name": args.strategy_name})
    return GetStrategyHistoryResult(**result)


async def get_rejection_reasons_tool(args: GetRejectionReasonsArgs) -> GetRejectionReasonsResult:
    """Get recent rejections with reasons - LOW risk, OBSERVE level."""
    from forex_ai_agent.agent.explain import get_rejection_reasons

    result = await get_rejection_reasons({"limit": args.limit})
    return GetRejectionReasonsResult(**result)


# ---------- Custom strategy backtest (Phase 7) ----------

class BacktestStrategySpecArgs(BaseModel):
    strategy_text: str = Field(description="Natural-language strategy rules given by the user")
    strategy_type: str = Field(
        default="auto",
        description="auto | session_sweep — 'session_sweep' is the implemented Asian-range London sweep",
    )
    symbol: str = Field(default="EUR/USD", description="Forex pair to backtest on")
    timeframe: str = Field(default="H1")
    days: int = Field(default=180, ge=30, le=1095, description="History span in days")
    rr_target: float = Field(default=2.0, gt=0, description="R-multiple target")
    sl_buffer_atr_frac: float = Field(default=0.10, ge=0, description="Stop buffer beyond the sweep (×ATR)")
    displacement_window_bars: int = Field(default=6, ge=1, le=48)
    displacement_atr_frac: float = Field(default=0.25, gt=0)
    save_if_passes: bool = Field(default=True, description="Version + probation the strategy if the gate passes")


class BacktestStrategySpecResult(BaseModel):
    strategy_name: str
    strategy_type: str
    symbol: str
    timeframe: str
    bars: int
    data_range: Dict[str, Any]
    metrics: Dict[str, Any]
    walk_forward: Dict[str, Any]
    passed: bool
    reasons: List[str]
    saved: bool = False
    version_id: Optional[str] = None
    probation_started: bool = False
    note: Optional[str] = None


async def backtest_strategy_spec(
    args: BacktestStrategySpecArgs,
) -> BacktestStrategySpecResult:
    """Backtest a user-described strategy through the SAME gate as autonomous
    research, then save + probation it only if it passes.

    Understood deterministic types:
    - ``session_sweep``: Asian range → London liquidity sweep → displacement →
      retracement entry, stop beyond the sweep, target = opposite range side
      or rr_target (whichever is closer).
    Any other description: honestly refused — the deterministic engine does not
    pretend to understand prose it cannot compile.
    """
    from forex_ai_agent.strategies.session_sweep import SessionSweepStrategy, SweepStrategyConfig
    from forex_ai_agent.autonomy.validator import BacktestValidator
    from forex_ai_agent.data.providers import get_feed_chain
    from forex_ai_agent.utils.timeframes import Timeframe
    from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest
    from forex_ai_agent.utils.math_helpers import pip_size_for_symbol
    from forex_ai_agent.strategy import create_version_manager
    from forex_ai_agent.strategy.versioning import VersionType
    import pandas as pd
    from datetime import timedelta

    stype = (args.strategy_type or "auto").strip().lower()
    text_l = args.strategy_text.lower()
    if stype == "auto":
        sweep_words = ("asian", "london", "sweep", "liquidity")
        stype = "session_sweep" if any(w in text_l for w in sweep_words) else "unsupported"
    if stype != "session_sweep":
        return BacktestStrategySpecResult(
            strategy_name=args.strategy_text[:80],
            strategy_type=stype,
            symbol=args.symbol,
            timeframe=args.timeframe,
            bars=0,
            data_range={},
            metrics={},
            walk_forward={},
            passed=False,
            reasons=[
                "This strategy cannot be backtested yet: no deterministic "
                "implementation exists for its rules. Implemented types: "
                "session_sweep (Asian range / London sweep / displacement / "
                "retracement). Ask Alex to run autonomous research on the pair "
                "instead, or request this strategy type be implemented."
            ],
            note="refused rather than guessed",
        )

    cfg = SweepStrategyConfig(
        rr_target=args.rr_target,
        sl_buffer_atr_frac=args.sl_buffer_atr_frac,
        displacement_window_bars=args.displacement_window_bars,
        displacement_atr_frac=args.displacement_atr_frac,
    )
    strategy = SessionSweepStrategy(config=cfg)

    # History through the feed chain (MT5 → yahoo → mock)
    provider = get_feed_chain()
    tf = Timeframe(args.timeframe)
    end = datetime.utcnow()
    start = end - timedelta(days=args.days)
    bars_per_day = {"M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
    limit = min(args.days * bars_per_day.get(args.timeframe.upper(), 24) + 50, 100_000)
    candles = await provider.fetch_candles(args.symbol, tf, start, end=end, limit=limit)
    if len(candles) < 250:
        raise ValueError(
            f"Insufficient history: {len(candles)} bars for {args.symbol} — need ≥250"
        )
    df = pd.DataFrame(
        {
            "open": [b.open for b in candles],
            "high": [b.high for b in candles],
            "low": [b.low for b in candles],
            "close": [b.close for b in candles],
            "volume": [float(getattr(b, "volume", 0) or 0) for b in candles],
        },
        index=[b.timestamp for b in candles],
    )
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    gate_validator = BacktestValidator()
    report = await gate_validator.validate_generator(
        strategy,
        df,
        symbol=args.symbol,
        timeframe=args.timeframe,
        run_walk_forward=True,
        generator_factory=lambda: SessionSweepStrategy(config=cfg),
        spec_name=strategy.name,
    )
    walk_forward: Dict[str, Any] = dict(report.walk_forward or {})

    saved = False
    version_id: Optional[str] = None
    probation_started = False
    note: Optional[str] = None
    if report.passed and args.save_if_passes:
        try:
            vm = await create_version_manager()
            version = vm.create_version(
                name=strategy.name[:80],
                description=f"User-described strategy backtested via backtest_strategy_spec ({args.symbol} {args.timeframe})",
                rules={
                    "strategy_type": "session_sweep",
                    "params": {
                        "rr_target": args.rr_target,
                        "sl_buffer_atr_frac": args.sl_buffer_atr_frac,
                        "displacement_window_bars": args.displacement_window_bars,
                        "displacement_atr_frac": args.displacement_atr_frac,
                    },
                },
                parameters={
                    "strategy_type": "session_sweep",
                    "symbol": args.symbol,
                    "timeframe": args.timeframe,
                    "rr_target": args.rr_target,
                },
                change_type=VersionType.MINOR,
            )
            # Store the full gate picture: full-backtest metrics, walk-forward
            # (with overfit_analysis) and the repro dict — so explain cards and
            # replay work later without re-running anything.
            version.backtest_results = {
                **dict(report.full_backtest or {}),
                "walk_forward": dict(report.walk_forward or {}),
                "repro": dict(report.repro or {}),
                "evaluated_at": report.evaluated_at,
            }
            version_id = str(version.id)
            saved = True
            try:
                from forex_ai_agent.autonomy.probation import ProbationStore
                from forex_ai_agent.autonomy.proposal import _adaptive_probation_quota

                req_trades, req_days = _adaptive_probation_quota(report)
                store = ProbationStore()
                prob = await store.start(
                    version_id=version_id,
                    strategy_name=strategy.name[:80],
                    pair=args.symbol,
                    spec_dict={"strategy_type": "session_sweep", "params": cfg.__dict__},
                    required_trades=req_trades,
                    required_days=req_days,
                )
                probation_started = True
                note = f"probation started: {prob.required_trades} trades / {prob.required_days} days"
            except Exception as exc:  # noqa: BLE001
                note = f"saved as {version_id} but probation failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            note = f"gate passed but saving failed: {type(exc).__name__}: {exc}"

    return BacktestStrategySpecResult(
        strategy_name=strategy.name,
        strategy_type="session_sweep",
        symbol=args.symbol,
        timeframe=args.timeframe,
        bars=len(df),
        data_range={
            "start": str(df.index.min()),
            "end": str(df.index.max()),
            "bars": str(len(df)),
        },
        metrics=dict(report.full_backtest or {}),
        walk_forward=walk_forward,
        passed=bool(report.passed),
        reasons=list(report.reasons),
        saved=saved,
        version_id=version_id,
        probation_started=probation_started,
        note=note,
    )


async def backtest_strategy_spec_tool(
    args: BacktestStrategySpecArgs,
) -> BacktestStrategySpecResult:
    return await backtest_strategy_spec(args)


# ---------- Tool Registry ----------

class ToolRegistry:
    """Registry of all agent tools with dispatch and permission checking."""

    def __init__(self, user_permission_level: int = 0):
        self._tools: Dict[str, Tool] = {}
        self.user_permission_level = user_permission_level
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all tools: 11 read-only + 5 action tools."""
        # Read-only tools (Phase 1) - OBSERVE level (0)
        read_only_tools = [
            Tool(
                name="get_market_data",
                description="Fetch OHLCV candles for a forex pair and timeframe",
                args_model=MarketDataArgs,
                result_model=MarketDataResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_market_data,
            ),
            Tool(
                name="get_current_price",
                description="Fetch current bid/ask/spread for a forex pair",
                args_model=CurrentPriceArgs,
                result_model=CurrentPriceResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_current_price,
            ),
            Tool(
                name="analyze_structure",
                description="Analyze market structure: trend, swings, BOS/CHoCH, support/resistance, liquidity sweeps, consolidation, retests",
                args_model=StructureArgs,
                result_model=StructureResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=analyze_structure,
            ),
            Tool(
                name="get_indicators",
                description="Compute technical indicators: RSI, EMA(9/21), MACD, ATR",
                args_model=IndicatorsArgs,
                result_model=IndicatorsResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_indicators,
            ),
            Tool(
                name="get_active_signals",
                description="Get active trading signals from the pipeline (optional filter by symbol/direction/timeframe)",
                args_model=ActiveSignalsArgs,
                result_model=ActiveSignalsResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_active_signals,
            ),
            Tool(
                name="get_risk_state",
                description="Get current risk state: daily loss, drawdown, positions, margin, upcoming events",
                args_model=RiskStateArgs,
                result_model=RiskStateResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_risk_state,
            ),
            Tool(
                name="get_news",
                description="Get recent forex news (optional filter by pair, hours back)",
                args_model=NewsArgs,
                result_model=NewsResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_news,
            ),
            Tool(
                name="search_strategies",
                description="Search strategy versions (optional query/status filter)",
                args_model=SearchStrategiesArgs,
                result_model=SearchStrategiesResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=search_strategies,
            ),
            Tool(
                name="get_strategy",
                description="Get a specific strategy version by ID or version string",
                args_model=GetStrategyArgs,
                result_model=GetStrategyResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_strategy,
            ),
            Tool(
                name="list_probation",
                description="List all probation records with status and trade counts",
                args_model=ListProbationArgs,
                result_model=ListProbationResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=list_probation,
            ),
            Tool(
                name="list_research_history",
                description="List recent research tasks with verdicts and conclusions",
                args_model=ListResearchHistoryArgs,
                result_model=ListResearchHistoryResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=list_research_history,
            ),
        ]

        # Action tools (Phase 2) - with risk classes and permission levels
        action_tools = [
            Tool(
                name="start_research",
                description=(
                    "Start an autonomous research task in the BACKGROUND "
                    "(LOW risk, executes immediately). It returns only a task "
                    "id — results appear later on the Research page. Do NOT "
                    "call it repeatedly for results; if the user described "
                    "explicit strategy rules and wants metrics NOW, use "
                    "backtest_strategy_spec instead."
                ),
                args_model=StartResearchArgs,
                result_model=StartResearchResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=start_research,
            ),
            Tool(
                name="remove_strategy",
                description="Remove a strategy version (MEDIUM risk, requires confirmation)",
                args_model=RemoveStrategyArgs,
                result_model=RemoveStrategyResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=remove_strategy,
            ),
            Tool(
                name="archive_strategy",
                description="Archive a strategy version (MEDIUM risk, requires confirmation)",
                args_model=ArchiveStrategyArgs,
                result_model=ArchiveStrategyResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=archive_strategy,
            ),
            Tool(
                name="reject_strategy",
                description="Reject a strategy version with reason (MEDIUM risk, requires confirmation)",
                args_model=RejectStrategyArgs,
                result_model=RejectStrategyResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=reject_strategy,
            ),
            Tool(
                name="create_strategy_draft",
                description="Create a strategy draft (LOW risk, ASSIST level)",
                args_model=CreateStrategyDraftArgs,
                result_model=CreateStrategyDraftResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=create_strategy_draft,
            ),
        ]

        # Experiment tools (Phase 3) - with risk classes and permission levels
        experiment_tools = [
            Tool(
                name="backtest_strategy",
                description="Run backtest for a strategy version (LOW risk, OBSERVE level)",
                args_model=BacktestStrategyArgs,
                result_model=BacktestStrategyResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=backtest_strategy_tool,
            ),
            Tool(
                name="walk_forward_test",
                description=(
                    "Run REAL walk-forward analysis on a saved strategy: rolling in-sample vs "
                    "out-of-sample backtests, per-window metrics, overfitting score and a "
                    "STABLE/MIXED/UNSTABLE verdict (LOW risk, OBSERVE level). Needs ~1 year of "
                    "history for reliable windows."
                ),
                args_model=WalkForwardTestArgs,
                result_model=WalkForwardTestResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=walk_forward_test_tool,
            ),
            Tool(
                name="replay_repro",
                description=(
                    "Deterministically re-run a stored backtest reproducibility record and "
                    "verify the metrics still match (LOW risk, OBSERVE level). Answers 'why "
                    "did this pass?' with exact numbers instead of memory."
                ),
                args_model=ReplayReproArgs,
                result_model=ReplayReproResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=replay_repro_tool,
            ),
            Tool(
                name="compare_strategies",
                description="Compare multiple strategies side by side (LOW risk, OBSERVE level)",
                args_model=CompareStrategiesArgs,
                result_model=CompareStrategiesResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=compare_strategies_tool,
            ),
            Tool(
                name="create_experiment",
                description="Create and run a parameter mutation experiment (MEDIUM risk, ASSIST level)",
                args_model=CreateExperimentArgs,
                result_model=CreateExperimentResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=create_experiment_tool,
            ),
            Tool(
                name="list_experiments",
                description="List experiment records (LOW risk, OBSERVE level)",
                args_model=ListExperimentsArgs,
                result_model=ListExperimentsResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=list_experiments_tool,
            ),
            Tool(
                name="get_experiment",
                description="Get experiment details with full metrics (LOW risk, OBSERVE level)",
                args_model=GetExperimentArgs,
                result_model=GetExperimentResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_experiment_tool,
            ),
            Tool(
                name="conclude_experiment",
                description="Record final decision on an experiment (MEDIUM risk, ASSIST level)",
                args_model=ConcludeExperimentArgs,
                result_model=ConcludeExperimentResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=conclude_experiment_tool,
            ),
        ]

        # Phase 5 tools: Viability check & Autonomous research
        phase5_tools = [
            Tool(
                name="check_viability",
                description="Check strategy viability vs baseline (LOW risk, OBSERVE level)",
                args_model=CheckViabilityArgs,
                result_model=CheckViabilityResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=check_viability_tool,
            ),
            Tool(
                name="backtest_strategy_spec",
                description=(
                    "Backtest a strategy the USER DESCRIBED in words (e.g. an "
                    "Asian-range London liquidity sweep) through the validation "
                    "gate; if it passes, it is versioned and enters paper "
                    "probation automatically. Always prefer this tool over "
                    "start_research when the user supplies explicit strategy rules."
                ),
                args_model=BacktestStrategySpecArgs,
                result_model=BacktestStrategySpecResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=backtest_strategy_spec_tool,
            ),
            Tool(
                name="find_strategies",
                description="Bounded autonomous research: find promising strategies (MEDIUM risk, ASSIST level)",
                args_model=FindStrategiesArgs,
                result_model=FindStrategiesResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=find_strategies_tool,
            ),
        ]

        # Phase 6 tools: Explain / "Why?" everywhere
        phase6_tools = [
            Tool(
                name="explain_probation",
                description="Generate EXPLAIN card for probation record with evidence (LOW risk, OBSERVE level)",
                args_model=ExplainProbationArgs,
                result_model=ExplainProbationResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=explain_probation_tool,
            ),
            Tool(
                name="explain_research",
                description="Generate EXPLAIN card for research task (LOW risk, OBSERVE level)",
                args_model=ExplainResearchArgs,
                result_model=ExplainResearchResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=explain_research_tool,
            ),
            Tool(
                name="explain_gate",
                description="Explain gate rejection for a pair (LOW risk, OBSERVE level)",
                args_model=ExplainGateArgs,
                result_model=ExplainGateResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=explain_gate_tool,
            ),
            Tool(
                name="answer_why",
                description="Answer 'why?' question using knowledge retriever (LOW risk, OBSERVE level)",
                args_model=AnswerWhyArgs,
                result_model=AnswerWhyResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=answer_why_tool,
            ),
            Tool(
                name="explain_overfit",
                description=(
                    "Component-by-component overfit analysis for a strategy — degradation, "
                    "stability, negative windows, concentration, sensitivity — every value "
                    "traced to stored walk-forward windows (LOW risk, OBSERVE level)."
                ),
                args_model=ExplainOverfitArgs,
                result_model=ExplainOverfitResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=explain_overfit_tool,
            ),
            Tool(
                name="get_strategy_health",
                description=(
                    "Five-light health card for a strategy — profitability, robustness, "
                    "sample size, OOS consistency, drawdown — plus an overall badge and "
                    "staleness, all read from stored state in under a second, no re-run "
                    "(LOW risk, OBSERVE level)."
                ),
                args_model=GetStrategyHealthArgs,
                result_model=GetStrategyHealthResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_strategy_health_tool,
            ),
            Tool(
                name="run_robustness_sweep",
                description=(
                    "Walk-forward the strategy across its pair plus the majors and tag "
                    "every OOS window with its market regime — produces the cross-pair / "
                    "regime evidence behind the health card's last two lights. Heavy: "
                    "up to 4 pairs x 8 windows of backtests, stored on the version "
                    "(MEDIUM risk, ASSIST level — comparable budget to an experiment)."
                ),
                args_model=RunRobustnessSweepArgs,
                result_model=RunRobustnessSweepResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=run_robustness_sweep_tool,
            ),
            Tool(
                name="hypothesis_test_count",
                description=(
                    "How many times has a hypothesis been tested, with what outcomes and "
                    "human decisions — a join over stored research-graph records, no re-run "
                    "(LOW risk, OBSERVE level)."
                ),
                args_model=HypothesisTestCountArgs,
                result_model=HypothesisTestCountResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=hypothesis_test_count_tool,
            ),
            Tool(
                name="aggregate_by_parameter",
                description=(
                    "Group stored experiment outcomes by a parameter's value — answers 'what "
                    "happens OOS when displacement > 0.25?' from real records (LOW risk, "
                    "OBSERVE level)."
                ),
                args_model=AggregateByParameterArgs,
                result_model=AggregateByParameterResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=aggregate_by_parameter_tool,
            ),
            Tool(
                name="research_graph_summary",
                description=(
                    "Count of stored variant→outcome edges in the research graph (LOW risk, "
                    "OBSERVE level)."
                ),
                args_model=ResearchGraphSummaryArgs,
                result_model=ResearchGraphSummaryResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=research_graph_summary_tool,
            ),
            Tool(
                name="search_knowledge",
                description="Search research knowledge base (LOW risk, OBSERVE level)",
                args_model=SearchKnowledgeArgs,
                result_model=SearchKnowledgeResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=search_knowledge_tool,
            ),
            Tool(
                name="research_plan",
                description=(
                    "Structured research plan card (RESEARCH #id): hypothesis, pairs, "
                    "checklists and self-reported progress from persisted records. "
                    "Pass task_id for one plan or omit it to list recent plans "
                    "(LOW risk, OBSERVE level)."
                ),
                args_model=ResearchPlanArgs,
                result_model=ResearchPlanResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=research_plan_tool,
            ),
            Tool(
                name="generate_hypotheses",
                description=(
                    "LLM-driven research direction: propose testable hypotheses from "
                    "deterministic market-structure briefs, validate them "
                    "deterministically, and enqueue the survivors with the Research "
                    "Director. The backtest gate still owns every outcome (MEDIUM "
                    "risk, ASSIST level)."
                ),
                args_model=GenerateHypothesesArgs,
                result_model=GenerateHypothesesResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=generate_hypotheses_tool,
            ),
            Tool(
                name="review_probation_promotions",
                description=(
                    "Deterministic review of the probation queue: ranks candidates "
                    "whose objective gates are MET and packages stored numbers for "
                    "the human approval decision. LLM narration is prose only; "
                    "nothing is promoted, approved or rejected here (LOW risk, "
                    "OBSERVE level)."
                ),
                args_model=ReviewPromotionArgs,
                result_model=ReviewPromotionResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=review_promotion_tool,
            ),
            Tool(
                name="monitor_probation_health",
                description=(
                    "Continuous probation health snapshot (Stage D3): deterministic "
                    "flags over accrual progress vs quota and window, current standing "
                    "vs each row's own gate floors, event-stream freshness / replay "
                    "fallback and feed integrity. Projections are arithmetic over "
                    "stored records; read-only — never mutates probation state "
                    "(LOW risk, OBSERVE level)."
                ),
                args_model=ProbationMonitorArgs,
                result_model=ProbationMonitorResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=probation_monitor_tool,
            ),
            Tool(
                name="select_experiments",
                description=(
                    "LLM chooses which experiment to run next from a deterministic "
                    "menu of strategies, parameter ranges and prior tests; "
                    "deterministic code validates every proposal and the Phase-3 "
                    "backtest/walk-forward engines own every number. Research only "
                    "(MEDIUM risk, ASSIST level)."
                ),
                args_model=SelectExperimentsArgs,
                result_model=SelectExperimentsResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=select_experiments_tool,
            ),
            Tool(
                name="generate_research_report",
                description=(
                    "Written research report from stored records: deterministic "
                    "evidence catalog (experiments, research tasks, probation, "
                    "health) plus LLM narration whose every number must trace to "
                    "the cited evidence; the numbers ledger is code-rendered and "
                    "the report is persisted (MEDIUM risk, ASSIST level)."
                ),
                args_model=GenerateReportArgs,
                result_model=GenerateReportResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=generate_report_tool,
            ),
            Tool(
                name="director_enqueue",
                description=(
                    "Add a pair/hypothesis to the Research Director's persisted queue "
                    "(deduplicated per pair; budget-capped) (MEDIUM risk, ASSIST level)."
                ),
                args_model=DirectorEnqueueArgs,
                result_model=DirectorEnqueueResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=director_enqueue_tool,
            ),
            Tool(
                name="director_run_cycle",
                description=(
                    "Run one budget-capped unattended research cycle over the queue: "
                    "research brief → spec → backtest gate → version + probation for "
                    "survivors. Research tools only; no live execution, no risk changes, "
                    "every action audited (MEDIUM risk, ASSIST level)."
                ),
                args_model=DirectorRunCycleArgs,
                result_model=DirectorRunCycleResult,
                min_permission_level=PermissionLevel.ASSIST,
                handler=director_run_cycle_tool,
            ),
            Tool(
                name="director_status",
                description=(
                    "Research Director queue snapshot + recent audit events, from stored "
                    "records (LOW risk, OBSERVE level)."
                ),
                args_model=DirectorStatusArgs,
                result_model=DirectorStatusResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=director_status_tool,
            ),
            Tool(
                name="get_strategy_history",
                description="Get all research/experiments for a strategy name (LOW risk, OBSERVE level)",
                args_model=GetStrategyHistoryArgs,
                result_model=GetStrategyHistoryResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_strategy_history_tool,
            ),
            Tool(
                name="get_rejection_reasons",
                description="Get recent rejections with reasons (LOW risk, OBSERVE level)",
                args_model=GetRejectionReasonsArgs,
                result_model=GetRejectionReasonsResult,
                min_permission_level=PermissionLevel.OBSERVE,
                handler=get_rejection_reasons_tool,
            ),
        ]

        # Register all tools
        for tool in read_only_tools + action_tools + experiment_tools + phase5_tools + phase6_tools:
            self._tools[tool.name] = tool

        # Register risk classes for action tools
        register_tool_risk("start_research", RiskClass.LOW)
        register_tool_risk("research_plan", RiskClass.LOW)
        register_tool_risk("review_probation_promotions", RiskClass.LOW)
        register_tool_risk("monitor_probation_health", RiskClass.LOW)
        register_tool_risk("generate_hypotheses", RiskClass.MEDIUM)
        register_tool_risk("select_experiments", RiskClass.MEDIUM)
        register_tool_risk("generate_research_report", RiskClass.MEDIUM)
        register_tool_risk("remove_strategy", RiskClass.MEDIUM)
        register_tool_risk("archive_strategy", RiskClass.MEDIUM)
        register_tool_risk("reject_strategy", RiskClass.MEDIUM)
        register_tool_risk("create_strategy_draft", RiskClass.LOW)

        # Register risk classes for experiment tools
        register_tool_risk("backtest_strategy", RiskClass.LOW)
        register_tool_risk("backtest_strategy_spec", RiskClass.LOW)
        register_tool_risk("walk_forward_test", RiskClass.LOW)
        register_tool_risk("compare_strategies", RiskClass.LOW)
        register_tool_risk("create_experiment", RiskClass.MEDIUM)
        register_tool_risk("list_experiments", RiskClass.LOW)
        register_tool_risk("get_experiment", RiskClass.LOW)
        register_tool_risk("conclude_experiment", RiskClass.MEDIUM)

        # Register risk classes for Phase 5 tools
        register_tool_risk("check_viability", RiskClass.LOW)
        register_tool_risk("find_strategies", RiskClass.MEDIUM)

        # Register risk classes for Phase 6 tools
        register_tool_risk("explain_probation", RiskClass.LOW)
        register_tool_risk("explain_research", RiskClass.LOW)
        register_tool_risk("explain_gate", RiskClass.LOW)
        register_tool_risk("answer_why", RiskClass.LOW)
        register_tool_risk("search_knowledge", RiskClass.LOW)
        register_tool_risk("get_strategy_health", RiskClass.LOW)
        register_tool_risk("hypothesis_test_count", RiskClass.LOW)
        register_tool_risk("aggregate_by_parameter", RiskClass.LOW)
        register_tool_risk("research_graph_summary", RiskClass.LOW)
        register_tool_risk("run_robustness_sweep", RiskClass.MEDIUM)

        # Register risk classes for B4 Director tools
        register_tool_risk("director_enqueue", RiskClass.MEDIUM)
        register_tool_risk("director_run_cycle", RiskClass.MEDIUM)
        register_tool_risk("director_status", RiskClass.LOW)
        register_tool_risk("get_strategy_history", RiskClass.LOW)
        register_tool_risk("get_rejection_reasons", RiskClass.LOW)

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self, min_level: Optional[int] = None) -> List[Tool]:
        """List tools, optionally filtered by minimum permission level."""
        tools = list(self._tools.values())
        if min_level is not None:
            tools = [t for t in tools if t.min_permission_level <= min_level]
        return tools

    def get_schemas_for_prompt(self, min_level: Optional[int] = None) -> str:
        """Generate tool documentation for the system prompt."""
        tools = self.list_tools(min_level)
        lines = []
        for tool in tools:
            schema = tool.args_model.model_json_schema()
            props = schema.get("properties", {})
            args_desc = ", ".join(
                f"{k}: {v.get('type', 'any')}" + (f" ({v.get('description', '')})" if v.get('description') else "")
                for k, v in props.items()
            )
            lines.append(f"- {tool.name}({args_desc}) — {tool.description}")
        return "\n".join(lines)

    async def dispatch(self, name: str, args: Dict[str, Any], confirm_token: Optional[str] = None) -> BaseModel:
        """Dispatch a tool call with permission checking and confirmation handling."""
        tool = self._tools.get(name)
        if not tool:
            raise ValueError(f"Unknown tool: {name}")

        if self.user_permission_level < tool.min_permission_level:
            raise PermissionError(
                f"Tool '{name}' requires permission level {tool.min_permission_level}, "
                f"user has {self.user_permission_level}"
            )

        # Check if this is a confirmation request
        if confirm_token:
            policy = get_confirmation_policy()
            token_obj = policy.consume_token(confirm_token)
            if not token_obj:
                raise PermissionError("Invalid or expired confirmation token")
            if token_obj.tool_name != name:
                raise PermissionError(f"Token is for tool '{token_obj.tool_name}', not '{name}'")

        # Validate args
        validated_args = tool.args_model(**args)
        # Call handler
        result = await tool.handler(validated_args)

        # For medium-risk tools without token, they return a proposal
        # The caller should check result for 'proposal' field
        return result


# ---------- Global registry instance ----------

_registry: Optional[ToolRegistry] = None


def get_tool_registry(user_permission_level: int = 0) -> ToolRegistry:
    """Get or create the global tool registry."""
    global _registry
    if _registry is None or _registry.user_permission_level != user_permission_level:
        _registry = ToolRegistry(user_permission_level=user_permission_level)
    return _registry