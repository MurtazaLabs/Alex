"""Technical Analysis Agent - Multi-timeframe indicator pipeline with caching.

Calculates deterministic technical indicators across required timeframes:
M1, M5, M15, M30, H1, H4, D1

Indicators: SMA, EMA, MACD, RSI, Stochastic, ATR, Bollinger Bands
State: Cached per pair+timeframe with TTL for freshness
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..analysis.indicators.trend import sma, ema, macd
from ..analysis.indicators.momentum import rsi, stochastic
from ..analysis.indicators.volatility import atr, true_range, bollinger_bands
from ..analysis.indicators.price_action import candle_direction, detect_swing_points
from ..utils.timeframes import Timeframe, is_valid_timeframe, TIMEFRAME_ORDER


# ---------------------------------------------------------------------------
# Per-timeframe cache: LRU with TTL
# ---------------------------------------------------------------------------

class TechnicalAnalysisAgent:
    """Multi-timeframe technical indicator pipeline with caching.

    Calculates deterministic technical indicators across required timeframes:
    M1, M5, M15, M30, H1, H4, D1

    Indicators: SMA, EMA, MACD, RSI, Stochastic, ATR, Bollinger Bands
    State: Cached per pair+timeframe with TTL for freshness
    """

    def __init__(self, max_size: int = 128, ttl_seconds: int = 30):
        self._max_size = max_size
        self._ttl = timedelta(seconds=ttl_seconds)
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.utcnow() - timestamp < self._ttl:
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                    return value
                else:
                    # Expired
                    del self._cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = (value, datetime.utcnow())

    def invalidate(self, key: str) -> None:
        """Force invalidate a specific key."""
        self._cache.pop(key, None)


# Global agent instance (shared across modules)
_technical_analysis_agent: Optional[TechnicalAnalysisAgent] = None


def get_technical_analysis_agent() -> TechnicalAnalysisAgent:
    """Get the global technical analysis agent instance."""
    global _technical_analysis_agent
    if _technical_analysis_agent is None:
        _technical_analysis_agent = TechnicalAnalysisAgent(max_size=256, ttl_seconds=60)
    return _technical_analysis_agent


def get_indicator_cache() -> TechnicalAnalysisAgent:
    """Alias for the indicator cache (the TA agent doubles as the TTL cache)."""
    return get_technical_analysis_agent()


# ---------------------------------------------------------------------------
# Multi-timeframe Indicator Computation
# ---------------------------------------------------------------------------

async def compute_all_indicators(
    highs: Dict[str, np.ndarray],
    lows: Dict[str, np.ndarray],
    closes: Dict[str, np.ndarray],
    volumes: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute all technical indicators across all required timeframes.

    Returns nested dict: {timeframe: {indicator_name: value_or_array}}
    """
    cache = get_indicator_cache()
    results: Dict[str, Dict[str, Any]] = {}

    for tf in TIMEFRAME_ORDER:
        if tf not in closes or closes[tf] is None or len(closes[tf]) == 0:
            results[tf] = {}
            continue

        h = highs.get(tf, np.array([]))
        l = lows.get(tf, np.array([]))
        c = closes[tf]

        # Warmup: ensure enough data for all indicators
        min_len = max(50, 26 + 9)  # MACD slow+signal
        if len(c) < min_len:
            results[tf] = {}
            continue

        # Compute indicators
        indicator_result: Dict[str, Any] = {}

        # Trend
        indicator_result["sma_20"] = float(sma(c, 20)[-1])
        indicator_result["ema_20"] = float(ema(c, 20)[-1])
        indicator_result["sma_50"] = float(sma(c, 50)[-1]) if len(c) >= 50 else None
        indicator_result["ema_50"] = float(ema(c, 50)[-1]) if len(c) >= 50 else None

        # MACD
        macd_result = macd(c, fast=12, slow=26, signal=9)
        indicator_result["macd"] = float(macd_result["macd"][-1])
        indicator_result["macd_signal"] = float(macd_result["signal"][-1])
        indicator_result["macd_histogram"] = float(macd_result["histogram"][-1])

        # Momentum
        rsi_val = rsi(c, 14)
        indicator_result["rsi"] = float(rsi_val[-1])

        # Stochastic
        h_tf = h if h.size > 0 else c
        l_tf = l if l.size > 0 else c
        stoch = stochastic(h_tf, l_tf, c, 14)
        indicator_result["stochastic"] = float(stoch[-1])

        # Volatility
        atr_val = atr(h, l, c, 14)
        indicator_result["atr"] = float(atr_val[-1])

        bb = bollinger_bands(c, 20, 2.0)
        indicator_result["bb_upper"] = float(bb["upper"][-1])
        indicator_result["bb_middle"] = float(bb["middle"][-1])
        indicator_result["bb_lower"] = float(bb["lower"][-1])

        # Price action
        direction = candle_direction(float(c[0]), float(c[-1]))
        indicator_result["candle_direction"] = direction

        # Cache the result
        cache_key = f"indicators_{tf}"
        await cache.set(cache_key, indicator_result)

        results[tf] = indicator_result

    return results


async def compute_single_indicator(
    indicator: str,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    **params: Any,
) -> Optional[float]:
    """Compute a single indicator value with caching."""
    cache = get_indicator_cache()
    cache_key = f"{indicator}_{len(closes)}" if "closes" in params else indicator

    # Try cache first
    cached = await cache.get(cache_key)
    if cached is not None:
        return cached

    # Compute
    if indicator == "rsi":
        val = rsi(closes, params.get("period", 14))
        result = float(val[-1]) if val.size > 0 and not np.isnan(val[-1]) else None
    elif indicator == "sma":
        val = sma(closes, params.get("period", 20))
        result = float(val[-1]) if val.size > 0 and not np.isnan(val[-1]) else None
    elif indicator == "ema":
        val = ema(closes, params.get("period", 20))
        result = float(val[-1]) if val.size > 0 and not np.isnan(val[-1]) else None
    elif indicator == "macd":
        m = macd(closes, fast=12, slow=26, signal=9)
        result = float(m["macd"][-1]) if m["macd"].size > 0 and not np.isnan(m["macd"][-1]) else None
    elif indicator == "rsi":
        val = rsi(closes, params.get("period", 14))
        result = float(val[-1]) if val.size > 0 and not np.isnan(val[-1]) else None
    elif indicator == "atr":
        # Need highs, lows, closes
        result = None  # Requires full context
    elif indicator == "bb_upper":
        # Need closes
        result = None
    else:
        result = None

    await cache.set(cache_key, result)
    return result


# ---------------------------------------------------------------------------
# High-level helper: build MarketContext for SignalAgent
# ---------------------------------------------------------------------------

async def build_market_context(
    pair: str,
    timeframe: str,
    candle_data: Dict[str, Dict[str, np.ndarray]],
    structure_data: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Build a MarketContext object for the SignalAgent from raw candle data.

    Expected candle_data format:
    {
        "M1": {"high": np.array, "low": np.array, "close": np.array, "volume": np.array},
        "M5": {...},
        "H1": {...},
        "H4": {...},
        "D1": {...},
    }
    """
    from ..signal_rules.base_rule import (
        MarketContext,
        HTFContext,
        IndicatorsSnapshot,
        Direction,
        StructureSnapshot,
    )

    direction = Direction.LONG  # Default; will be set by signal evaluation

    # 1. Collect indicators across timeframes
    all_indicators: Dict[str, Dict[str, Any]] = await compute_all_indicators(
        highs=candle_data.get("highs", {}),
        lows=candle_data.get("lows", {}),
        closes=candle_data.get("closes", {}),
    )

    # 2. Build per-timeframe HTF context
    htf_contexts: Dict[str, HTFContext] = {}
    for tf in ["D1", "H4", "H1", "M15", "M5", "M1"]:
        if tf not in candle_data.get("closes", {}):
            continue
        c = candle_data["closes"][tf]
        h = candle_data.get("highs", {}).get(tf, np.array([c[0]] * len(c)))
        l = candle_data["lows", {}].get(tf, np.array([c[0]] * len(c)))

        # Trend from swing structure
        trend = "NEUTRAL"
        if len(c) > 10:
            # Simple: compare first vs last close
            if c[-1] > c[0]:
                trend = "BULLISH"
            elif c[-1] < c[0]:
                trend = "BEARISH"

        htf_contexts[tf] = HTFContext(
            timeframe=tf,
            trend=trend,
            key_support=None,  # Would come from structure agent
            key_resistance=None,
        )

    # 2. Build indicators snapshot for the requested timeframe
    requested_tf = timeframe or "H1"
    ind_1h = all_indicators.get(requested_tf, {})

    indicators_snapshot = IndicatorsSnapshot(
        rsi=ind_1h.get("rsi"),
        macd=ind_1h.get("macd"),
        macd_signal=ind_1h.get("macd_signal"),
        macd_histogram=ind_1h.get("macd_histogram"),
        atr=ind_1h.get("atr"),
        bb_upper=ind_1h.get("bb_upper"),
        bb_middle=ind_1h.get("bb_middle"),
        bb_lower=ind_1h.get("bb_lower"),
        sma_20=ind_1h.get("sma_20"),
        sma_50=ind_1h.get("sma_50"),
        ema_20=ind_1h.get("ema_20"),
        ema_50=ind_1h.get("ema_50"),
    )

    # 3. Build structure snapshot (placeholder - would come from MarketStructureAgent)
    structure_snapshot = StructureSnapshot(
        trend="NEUTRAL",
        swing_highs=structure_data.get("swing_highs", []) if structure_data else [],
        swing_lows=structure_data.get("swing_lows", []) if structure_data else [],
        last_bos=structure_data.get("last_bos") if structure_data else None,
        last_choch=structure_data.get("last_choch") if structure_data else None,
        key_support=structure_data.get("key_support") if structure_data else None,
        key_resistance=structure_data.get("key_resistance") if structure_data else None,
        liquidity_sweeps=structure_data.get("liquidity_sweeps", []) if structure_data else [],
        consolidation=structure_data.get("consolidation") if structure_data else None,
        retests=structure_data.get("retests", []) if structure_data else [],
        events=structure_data.get("events", []) if structure_data else [],
    )

    # 4. Current price from latest candle
    latest_candle = candle_data.get("closes", {}).get(requested_tf)
    current_price = float(latest_candle[-1]) if latest_candle is not None and len(latest_candle) > 0 else 0.0

    # 5. Build and return MarketContext
    ctx = MarketContext(
        pair=pair,
        timeframe=requested_tf,
        current_price=current_price,
        spread_pips=0.0,  # Would be fetched from market data
        atr_pips=ind_1h.get("atr", 0.0) or 0.0,
        session="London",  # Would come from market data
        news_state="LOW",  # Would come from NewsAgent
        htf_context=htf_contexts,
        indicators=indicators_snapshot,
        structure=structure_snapshot,
        evidence=None,  # Populated by SignalAgent
    )

    return ctx