"""Tests for the deterministic Asian-range / London sweep strategy.

Synthetic H1 bars construct: Asian range → London sweep → displacement →
retracement, and assert the emitted signal's SL/TP follow the stated rules.
No network, no database.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from forex_ai_agent.strategies.session_sweep import (
    SessionSweepStrategy,
    SweepStrategyConfig,
)


DAY0 = datetime(2026, 9, 14, 0, 0)  # a Monday, UTC


def _bar(i: int, o: float, h: float, l: float, c: float):
    return DAY0 + timedelta(hours=i), {
        "open": o, "high": h, "low": l, "close": c, "volume": 100.0,
    }


def _drive(strategy: SessionSweepStrategy, bars):
    """Feed (ts, ohlc) pairs; return the list of emitted signals."""
    out = []
    for ts, b in bars:
        sigs = strategy.generate_signals(ts, {"EUR/USD": b}, {"EUR/USD": {}})
        out.extend(sigs)
    return out


def _scenario_bars():
    """One scripted day: 1.0000 base, Asian range 0.9980–1.0020.

    Bars 0–6: Asian range. 7: drift. 8 (08:00 UTC, London): sweep of the low
    (wick 0.9960, close 0.9985 inside). 9–10: nothing. 11: displacement up
    (close 1.0060, breaks structure). 12–14: pullback to ~1.0010 (50% of the
    displacement bar) → entry expected on 12–14. 15+: rally away.
    """
    bars = []
    o = 1.0000
    # Asian range (bars 0-6)
    for i in range(7):
        bars.append(_bar(i, o, 1.0020 if i else 1.0005, 0.9980 if i else 0.9995, 1.0000))
    # pre-London drift (7)
    bars.append(_bar(7, 1.0000, 1.0010, 0.9990, 1.0000))
    # London: sweep of the Asian low (bar 8)
    bars.append(_bar(8, 0.9990, 0.9995, 0.9960, 0.9985))
    # hesitation (9-10)
    bars.append(_bar(9, 0.9985, 0.9990, 0.9975, 0.9980))
    bars.append(_bar(10, 0.9980, 0.9995, 0.9980, 0.9990))
    # displacement up (11): close far above sweep extreme + structure break
    bars.append(_bar(11, 0.9990, 1.0065, 0.9990, 1.0060))
    # retracement (12-14): touches ~ (0.9960+1.0060)/2 = 1.0010
    bars.append(_bar(12, 1.0055, 1.0058, 1.0020, 1.0030))
    bars.append(_bar(13, 1.0030, 1.0035, 1.0005, 1.0012))
    bars.append(_bar(14, 1.0012, 1.0040, 1.0010, 1.0038))
    # continuation (15-18)
    for i in range(15, 19):
        bars.append(_bar(i, 1.0040, 1.0060, 1.0035, 1.0055))
    return bars


def test_long_sweep_emits_one_signal_with_rules():
    s = SessionSweepStrategy()
    sigs = _drive(s, _scenario_bars())
    assert len(sigs) == 1, f"expected exactly 1 signal, got {len(sigs)}"
    sig = sigs[0]
    assert sig["direction"] == "LONG"
    entry = sig["entry_price"]
    sl = sig["sl"]
    tp = sig["tp"]
    # stop beyond the sweep extreme (0.9960) with ATR buffer
    assert sl < 0.9960
    # risk is positive and TP = 2R (range target 1.0020 is below entry+2R → rr used)
    risk = entry - sl
    assert risk > 0
    assert tp == pytest.approx(entry + 2.0 * risk, rel=1e-6)
    # volume sized to 1% risk
    assert 0.01 <= sig["volume"] <= 5.0


def test_clean_break_is_not_a_sweep():
    """Close BEYOND the range = breakout, not sweep → no signal all day."""
    s = SessionSweepStrategy()
    bars = []
    for i in range(7):
        bars.append(_bar(i, 1.0000, 1.0020, 0.9980, 1.0000))
    for i in range(7, 11):
        bars.append(_bar(i, 1.0000, 1.0010, 0.9990, 1.0000))
    # clean break below the Asian low (close 0.9950 < 0.9980)
    bars.append(_bar(8, 0.9990, 0.9990, 0.9945, 0.9950))
    for i in range(9, 19):
        bars.append(_bar(i, 0.9950, 0.9960, 0.9935, 0.9945))
    assert _drive(s, bars) == []


def test_no_sweep_no_trade():
    s = SessionSweepStrategy()
    bars = []
    for i in range(7):
        bars.append(_bar(i, 1.0000, 1.0020, 0.9980, 1.0000))
    for i in range(11, 22):
        bars.append(_bar(i, 1.0000, 1.0010, 0.9990, 1.0000))
    assert _drive(s, bars) == []


def test_short_sweep_mirrors_rules():
    """Sweep of the Asian HIGH then bearish displacement → SHORT."""
    s = SessionSweepStrategy()
    bars = []
    for i in range(7):
        bars.append(_bar(i, 1.0000, 1.0020, 0.9980, 1.0000))
    for i in range(7, 11):
        bars.append(_bar(i, 1.0000, 1.0010, 0.9990, 1.0000))
    # sweep of the high (bar 8): wick above, close back inside
    bars.append(_bar(8, 1.0010, 1.0040, 1.0005, 1.0015))
    bars.append(_bar(9, 1.0015, 1.0025, 1.0010, 1.0020))
    bars.append(_bar(10, 1.0020, 1.0025, 1.0005, 1.0010))
    # bearish displacement (11): close below sweep extreme - buffer, structure break
    bars.append(_bar(11, 1.0010, 1.0015, 0.9935, 0.9940))
    # pullback up to ~ (1.0040+0.9940)/2 = 0.9990
    bars.append(_bar(12, 0.9940, 0.9985, 0.9935, 0.9980))
    bars.append(_bar(13, 0.9980, 0.9995, 0.9965, 0.9970))
    bars.append(_bar(14, 0.9970, 0.9975, 0.9930, 0.9935))
    for i in range(15, 19):
        bars.append(_bar(i, 0.9935, 0.9945, 0.9900, 0.9910))
    sigs = _drive(s, bars)
    assert len(sigs) == 1
    sig = sigs[0]
    assert sig["direction"] == "SHORT"
    assert sig["sl"] > 1.0040  # beyond the sweep high
    risk = sig["sl"] - sig["entry_price"]
    assert sig["tp"] == pytest.approx(sig["entry_price"] - 2.0 * risk, rel=1e-6)


def test_displacement_window_expiry_stands_down():
    """No displacement within the window → no trade that day."""
    s = SessionSweepStrategy(
        config=SweepStrategyConfig(displacement_window_bars=2)
    )
    bars = []
    for i in range(7):
        bars.append(_bar(i, 1.0000, 1.0020, 0.9980, 1.0000))
    for i in range(7, 11):
        bars.append(_bar(i, 1.0000, 1.0010, 0.9990, 1.0000))
    bars.append(_bar(8, 0.9990, 0.9995, 0.9960, 0.9985))
    # drift with no displacement for the rest of the day
    for i in range(9, 19):
        bars.append(_bar(i, 0.9985, 0.9995, 0.9980, 0.9990))
    assert _drive(s, bars) == []


def test_range_target_used_when_closer():
    """target_mode='range' uses the opposite Asian side as TP."""
    s = SessionSweepStrategy(
        config=SweepStrategyConfig(target_mode="range")
    )
    sigs = _drive(s, _scenario_bars())
    assert len(sigs) == 1
    # long entry ~1.001, opposite side = Asian HIGH 1.0020
    assert sigs[0]["tp"] == pytest.approx(1.0020, abs=3e-4)
