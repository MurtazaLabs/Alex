"""Research briefs: deterministic per-pair context for strategy proposals.

The brief is the LLM's entire view of a pair — built exclusively from the
same deterministic machinery the live pipeline uses (``fetch_multi_tf``,
``compute_all_indicators``, ``build_structure_snapshot``), plus the news
agent's risk state. It contains numbers (prices, ATR, RSI, trend labels)
but the LLM still never turns them into trade numbers: its only output is
a :class:`~forex_ai_agent.strategy.spec.StrategySpec`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from ..agents.technical_analysis_agent import compute_all_indicators
from ..analysis.market_structure.engine import build_structure_snapshot
from ..utils.math_helpers import pip_size_for_symbol

logger = logging.getLogger(__name__)


@dataclass
class PairBrief:
    """Everything the proposal step may know about one pair."""

    pair: str
    generated_at: str
    data_source: str
    session: str
    regime: str
    trend_h1: str
    trend_h4: str
    rsi_h1: float
    atr_pips_h1: float
    price: float
    sma20_h1: float
    sma50_h1: float
    ema20_h1: float
    structure_events_h1: List[Dict[str, Any]] = field(default_factory=list)
    support_h1: Optional[float] = None
    resistance_h1: Optional[float] = None
    news_state: str = "unknown"
    upcoming_high_impact: int = 0

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact dict for the LLM prompt (JSON-safe, no raw arrays)."""
        return {
            "pair": self.pair,
            "data_source": self.data_source,
            "session": self.session,
            "regime": self.regime,
            "trend_h1": self.trend_h1,
            "trend_h4": self.trend_h4,
            "price": round(self.price, 5),
            "sma20_h1": round(self.sma20_h1, 5),
            "sma50_h1": round(self.sma50_h1, 5),
            "ema20_h1": round(self.ema20_h1, 5),
            "rsi_h1": round(self.rsi_h1, 1),
            "atr_pips_h1": round(self.atr_pips_h1, 2),
            "support_h1": round(self.support_h1, 5) if self.support_h1 else None,
            "resistance_h1": round(self.resistance_h1, 5) if self.resistance_h1 else None,
            "recent_structure_events": self.structure_events_h1[:5],
            "news_state": self.news_state,
            "upcoming_high_impact_events_24h": self.upcoming_high_impact,
        }


def classify_regime(closes: np.ndarray, atr_pips: float, pip_size: float) -> str:
    """Coarse deterministic regime label from ADX-proxy + range behavior."""
    if len(closes) < 60:
        return "unknown"
    window = closes[-60:]
    # ADX proxy: net directional movement vs total path length.
    path = np.abs(np.diff(window)).sum()
    net = abs(window[-1] - window[0])
    efficiency = float(net / path) if path > 0 else 0.0
    atr_price = max(atr_pips * pip_size, 1e-9)
    range_width = float(window.max() - window.min())
    if efficiency >= 0.35:
        return "trending"
    if range_width < 6 * atr_price:
        return "ranging"
    return "mixed"


def session_for_utc_hour(hour_utc: int) -> str:
    """Session label (mirrors pipeline.session_for_utc_hour)."""
    if 7 <= hour_utc < 12:
        return "London"
    if 12 <= hour_utc < 16:
        return "London/NY overlap"
    if 16 <= hour_utc < 21:
        return "New York"
    return "Asia"


async def build_pair_brief(
    pipeline: Any,
    pair: str,
    timeframe: str = "H1",
    limit: int = 300,
) -> PairBrief:
    """Build a brief from the live resilient feed chain (deterministic)."""
    multi_tf = await pipeline.fetch_multi_tf(pair, tfs=(timeframe, "H4"), limit=limit)
    tf_data = multi_tf[timeframe]
    h4_data = multi_tf["H4"]

    indicators = await compute_all_indicators(
        highs={timeframe: tf_data["highs"], "H4": h4_data["highs"]},
        lows={timeframe: tf_data["lows"], "H4": h4_data["lows"]},
        closes={timeframe: tf_data["closes"], "H4": h4_data["closes"]},
    )
    ind = indicators.get(timeframe, {})

    def _last(key: str) -> Optional[float]:
        v = ind.get(key)
        if v is None:
            return None
        arr = np.asarray(v, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return float(arr[-1]) if arr.size else None

    pip = pip_size_for_symbol(pair)
    atr_val = _last("atr") or 0.0
    atr_pips = atr_val / pip

    snapshot = build_structure_snapshot(
        timeframe=timeframe,
        highs=tf_data["highs"],
        lows=tf_data["lows"],
        closes=tf_data["closes"],
        atr_pips=atr_pips,
        pip_size=pip,
        timestamps=tf_data.get("timestamps"),
    )

    def _trend_label(snap: Any) -> str:
        trend = getattr(snap, "trend", None)
        if trend is None:
            return "NEUTRAL"
        return str(getattr(trend, "name", trend))

    h1_label = _trend_label(snapshot)
    h4_snapshot = build_structure_snapshot(
        timeframe="H4",
        highs=h4_data["highs"],
        lows=h4_data["lows"],
        closes=h4_data["closes"],
        atr_pips=atr_pips,
        pip_size=pip,
        timestamps=h4_data.get("timestamps"),
    )

    news_state, upcoming = "unknown", 0
    news_agent = getattr(pipeline, "news_agent", None)
    if news_agent is not None:
        try:
            base = pair.split("/")[0]
            state = news_agent.get_risk_state(base)
            news_state = getattr(state, "name", str(state))
            upcoming = len(news_agent.get_high_impact_upcoming(hours_ahead=24))
        except Exception as exc:  # noqa: BLE001 — news is enrichment, never a gate
            logger.info("News enrichment unavailable for %s: %s", pair, exc)

    closes = tf_data["closes"]
    return PairBrief(
        pair=pair,
        generated_at=datetime.utcnow().isoformat(),
        data_source=getattr(pipeline.data_provider, "active_source", "unknown"),
        session=session_for_utc_hour(datetime.utcnow().hour),
        regime=classify_regime(closes, atr_pips, pip),
        trend_h1=h1_label,
        trend_h4=_trend_label(h4_snapshot),
        rsi_h1=_last("rsi") or 50.0,
        atr_pips_h1=atr_pips,
        price=float(closes[-1]),
        sma20_h1=_last("sma_20") or float(closes[-1]),
        sma50_h1=_last("sma_50") or float(closes[-1]),
        ema20_h1=_last("ema_20") or float(closes[-1]),
        structure_events_h1=[
            e for e in (getattr(snapshot, "events", None) or [])
            if isinstance(e, dict)
        ][:8],
        support_h1=getattr(snapshot, "key_support", None),
        resistance_h1=getattr(snapshot, "key_resistance", None),
        news_state=news_state,
        upcoming_high_impact=upcoming,
    )
