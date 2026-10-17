"""Session sweep strategy — Asian range → London liquidity sweep → displacement → retracement.

A deterministic, first-class implementation of the classic London sweep playbook
(it cannot be expressed by the indicator-condition ``StrategySpec``):

1. Build the **Asian range** from each UTC day's 00:00–06:59 H1 bars
   (high = max high, low = min low).
2. Wait for the **London window** (07:00–10:59 UTC by default). Do NOT enter
   immediately.
3. A **sweep** = a bar pierces the Asian low (or high) but CLOSES back inside
   the range. A clean break-and-close-beyond is NOT a sweep — it is a breakout
   and is skipped (``no valid sweep → do nothing``).
4. **Displacement** = within ``displacement_window_bars`` after the sweep, an
   opposite-direction bar CLOSES beyond the swept extreme by ≥
   ``displacement_atr_frac`` × ATR(14) AND breaks the most recent opposite
   swing (a structure break).
5. **Entry on the retracement**: once displaced, a limit-style entry is armed
   at the displacement bar's 50% level; it is taken when price pulls back to
   within ``retrace_tol_atr_frac`` × ATR of that level (market order at the
   current close, the engine's execution model).
6. **Stops beyond the sweep** (sweep extreme ± ``sl_buffer_atr_frac`` × ATR).
   **Targets**: opposite side of the Asian range (primary) OR ``rr_target``
   (default 2R) if farther — one target, whichever is closer per the rule
   ``target_mode``.
7. If no valid sweep + displacement forms, the day does nothing.

Everything is rule-based: the LLM is never in the trade-calculation path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class SweepStrategyConfig:
    """Tunable knobs — every value deterministic, none LLM-chosen."""

    asia_start_hour_utc: int = 0          # Asian session window (UTC)
    asia_end_hour_utc: int = 7            # exclusive end: 00:00–06:59
    london_open_hour_utc: int = 7         # sweep search begins
    london_close_hour_utc: int = 11       # exclusive end: no fresh sweeps after
    displacement_window_bars: int = 6     # bars after sweep to displace
    displacement_atr_frac: float = 0.25   # close beyond swept extreme ≥ 0.25×ATR
    retrace_tol_atr_frac: float = 0.15    # pullback tolerance around the 50% level
    sl_buffer_atr_frac: float = 0.10      # stop beyond the sweep extreme
    rr_target: float = 2.0                # R multiple target
    target_mode: str = "range_or_rr"      # "range_or_rr" | "rr" | "range"
    session_utc_offset_hours: int = 0     # set to broker offset if bars aren't UTC
    risk_per_trade_pct: float = 0.01
    account_balance: float = 100_000.0
    max_one_trade_per_day: bool = True


@dataclass
class _DayState:
    """Per-UTC-day bookkeeping."""

    day: str
    asia_high: Optional[float] = None
    asia_low: Optional[float] = None
    sweep_dir: Optional[str] = None       # "low" | "high"
    sweep_extreme: Optional[float] = None
    sweep_bar_ts: Optional[datetime] = None
    bars_since_sweep: int = 0
    displaced: bool = False
    disp_dir: Optional[str] = None        # "long" | "short"
    retrace_level: Optional[float] = None
    structure_level: Optional[float] = None
    swing_highs: List[float] = field(default_factory=list)
    swing_lows: List[float] = field(default_factory=list)
    traded: bool = False


def _atr14(highs: List[float], lows: List[float], closes: List[float]) -> float:
    """ATR(14) over the trailing window (rolling mean of true range).

    Uses whatever bars exist once ≥2 closes are available — a warm-up that
    returned 0 for the first 14 bars would blind the sweep detector, which
    needs no ATR at all.
    """
    if len(closes) < 2:
        return 0.0
    trs = [
        max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        for i in range(1, len(closes))
    ]
    return float(np.mean(trs[-14:]))


def _update_swings(state: _DayState, bar: Dict[str, Any], lookback: int = 10) -> None:
    """Track recent swing highs/lows for the structure-break requirement."""
    state.swing_highs.append(float(bar["high"]))
    state.swing_lows.append(float(bar["low"]))
    for seq in (state.swing_highs, state.swing_lows):
        if len(seq) > lookback * 2:
            del seq[: len(seq) - lookback * 2]


def _structure_break(state: _DayState, direction: str, price: float) -> bool:
    """True if ``price`` broke the most recent opposite swing (10-bar window)."""
    if direction == "long":
        window = state.swing_highs[-10:]
        if len(window) < 4:
            return False
        recent_high = max(window[:-1])
        return price > recent_high
    window = state.swing_lows[-10:]
    if len(window) < 4:
        return False
    recent_low = min(window[:-1])
    return price < recent_low


class SessionSweepStrategy:
    """Bar-by-bar signal generator for the backtest engine.

    ``generate_signals(timestamp, data, indicators)`` matches the engine's
    SignalGenerator protocol (same signature the spec evaluator uses).
    """

    def __init__(self, config: Optional[SweepStrategyConfig] = None, name: str = "Asian Range London Sweep") -> None:
        self.cfg = config or SweepStrategyConfig()
        self.name = name
        self._days: Dict[str, _DayState] = {}
        self._history: Dict[str, Dict[str, List[float]]] = {}

    # ------------------------------------------------------------------
    def _local(self, ts: datetime) -> datetime:
        return ts + timedelta(hours=self.cfg.session_utc_offset_hours)

    def _state_for(self, ts: datetime) -> _DayState:
        day = self._local(ts).date().isoformat()
        if day not in self._days:
            self._days[day] = _DayState(day=day)
        return self._days[day]

    def _build_ranges(self, symbol: str, ts: datetime, bar: Dict[str, Any]) -> None:
        """Build the Asian range from bars inside the window; outside the
        window no updates happen, so the range is frozen automatically
        (never freeze on the FIRST bar — that reduces the range to one bar
        and triggers false sweeps)."""
        local = self._local(ts)
        hour = local.hour
        state = self._state_for(ts)
        if self.cfg.asia_start_hour_utc <= hour < self.cfg.asia_end_hour_utc:
            if state.asia_high is None:
                state.asia_high = float(bar["high"])
                state.asia_low = float(bar["low"])
            else:
                state.asia_high = max(state.asia_high, float(bar["high"]))
                state.asia_low = min(state.asia_low, float(bar["low"]))

    # ------------------------------------------------------------------
    @staticmethod
    def _bar_dict(bar: Any) -> Dict[str, Any]:
        """Accept both engine BarData objects and plain dicts."""
        if hasattr(bar, "open") and not isinstance(bar, dict):
            return {
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(getattr(bar, "volume", 0) or 0),
            }
        return bar

    def generate_signals(
        self,
        timestamp: datetime,
        data: Dict[str, Any],
        indicators: Dict[str, Dict[str, float]],
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        for symbol, raw_bar in data.items():
            if raw_bar is None:
                continue
            bar = self._bar_dict(raw_bar)
            hist = self._history.setdefault(
                symbol, {"high": [], "low": [], "close": []}
            )
            hist["high"].append(float(bar["high"]))
            hist["low"].append(float(bar["low"]))
            hist["close"].append(float(bar["close"]))

            self._build_ranges(symbol, timestamp, bar)
            state = self._state_for(timestamp)
            _update_swings(state, bar)  # structure context must never skip a bar
            atr = _atr14(hist["high"], hist["low"], hist["close"])
            if atr <= 0:
                continue

            local = self._local(timestamp)
            sig = self._step(symbol, state, bar, timestamp, local, atr)
            if sig:
                signals.append(sig)
        return signals

    # ------------------------------------------------------------------
    def _step(
        self,
        symbol: str,
        state: _DayState,
        bar: Dict[str, Any],
        ts: datetime,
        local: datetime,
        atr: float,
    ) -> Optional[Dict[str, Any]]:
        cfg = self.cfg
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])

        # --- 1. detect the sweep (London window only) -------------------
        in_london = cfg.london_open_hour_utc <= local.hour < cfg.london_close_hour_utc
        if (
            state.asia_high is not None
            and state.sweep_dir is None
            and in_london
        ):
            swept_low = l < state.asia_low and c > state.asia_low      # wick below, close back inside
            swept_high = h > state.asia_high and c < state.asia_high   # wick above, close back inside
            if swept_low:
                state.sweep_dir, state.sweep_extreme = "low", l
                state.sweep_bar_ts = ts
            elif swept_high:
                state.sweep_dir, state.sweep_extreme = "high", h
                state.sweep_bar_ts = ts
            # clean breaks (close beyond the range) are NOT sweeps → ignored

        # --- 2. displacement after the sweep ----------------------------
        if state.sweep_dir is not None and not state.displaced:
            state.bars_since_sweep += 1
            if state.bars_since_sweep > cfg.displacement_window_bars:
                state.sweep_dir = None  # window expired → stand down for the day
                return None
            if state.sweep_dir == "low":
                needed = state.sweep_extreme + cfg.displacement_atr_frac * atr
                if c > needed and _structure_break(state, "long", c):
                    state.displaced = True
                    state.disp_dir = "long"
                    state.retrace_level = (l + c) / 2.0  # displacement bar 50%
                    state.structure_level = needed
            else:
                needed = state.sweep_extreme - cfg.displacement_atr_frac * atr
                if c < needed and _structure_break(state, "short", c):
                    state.displaced = True
                    state.disp_dir = "short"
                    state.retrace_level = (h + c) / 2.0
                    state.structure_level = needed
            return None

        # --- 3. entry on the retracement --------------------------------
        if state.displaced and not state.traded:
            if cfg.max_one_trade_per_day and state.traded:
                return None
            tol = cfg.retrace_tol_atr_frac * atr
            level = state.retrace_level
            if state.disp_dir == "long" and l <= level + tol:
                sl = state.sweep_extreme - cfg.sl_buffer_atr_frac * atr
                risk = c - sl
                if risk <= 0:
                    return None
                tp = self._target(state, c, risk)
                state.traded = True
                return self._signal(symbol, "LONG", c, sl, tp, ts, state)
            if state.disp_dir == "short" and h >= level - tol:
                sl = state.sweep_extreme + cfg.sl_buffer_atr_frac * atr
                risk = sl - c
                if risk <= 0:
                    return None
                tp = self._target(state, c, risk)
                state.traded = True
                return self._signal(symbol, "SHORT", c, sl, tp, ts, state)
        return None

    # ------------------------------------------------------------------
    def _target(self, state: _DayState, entry: float, risk: float) -> float:
        cfg = self.cfg
        rr_tp = entry + cfg.rr_target * risk if state.disp_dir == "long" else entry - cfg.rr_target * risk
        if cfg.target_mode == "rr":
            return rr_tp
        range_tp = (
            state.asia_high if state.disp_dir == "long" else state.asia_low
        )
        if cfg.target_mode == "range":
            return range_tp
        # range_or_rr: take whichever is CLOSER (conservative)
        if state.disp_dir == "long":
            return min(range_tp, rr_tp) if range_tp > entry else rr_tp
        return max(range_tp, rr_tp) if range_tp < entry else rr_tp

    def _signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        sl: float,
        tp: float,
        ts: datetime,
        state: _DayState,
    ) -> Dict[str, Any]:
        sl_dist = abs(entry - sl)
        volume = 1.0
        if sl_dist > 0:
            volume = (self.cfg.account_balance * self.cfg.risk_per_trade_pct) / (
                sl_dist * 100_000.0
            )
        volume = max(0.01, min(5.0, volume))
        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry,
            "sl": sl,
            "tp": tp,
            "volume": round(volume, 2),
            "comment": f"sweep:{self.name}:{state.sweep_dir}@{state.sweep_extreme:.5f}",
        }
