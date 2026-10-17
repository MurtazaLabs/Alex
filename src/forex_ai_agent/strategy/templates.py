"""Reference strategy templates — deterministic, human-confirmed specs.

These are NOT LLM proposals. Each template encodes a strategy shape a human
confirmed for Alex (see ``ALEX_FULL_AUTONOMY_PLAN.md`` "Confirmed trading
parameters", 2026-10-16), expressed entirely in the whitelisted spec
vocabulary. They flow through the SAME deterministic gate, registry and
paper probation as LLM candidates (``autonomy.proposal.propose_spec``) —
the gate still decides their fate on real data.

Mean reversion (Bollinger Bands + RSI, H1) — the confirmed primary strategy:

- LONG:  price closes below the lower BB(20, 2.0) AND RSI(14) < 30
- SHORT: price closes above the upper BB(20, 2.0) AND RSI(14) > 70
- SL 1.5 x ATR(14), TP 2.0 x ATR(14)  (R:R ~ 1.33)
- min-ATR filter (avoid dead markets); the max-spread and session filters
  live in the engine's cost model and the E4 session filter, not the spec.

One pinned spec per direction: the DSL ANDs its conditions and the ``both``
direction flag flips on the last condition's bias (a trend-following idiom),
so the two mean-reversion sides must be separate specs.
"""

from __future__ import annotations

from typing import List

from .spec import SpecCondition, StrategySpec

#: Minimum ATR(14) in price units (~2 pips on EUR/USD) — dead-market filter
MIN_ATR_THRESHOLD = 0.0002

#: The confirmed ATR bracket multipliers (R:R ~ 1.33)
_ATR_SL_MULT = 1.5
_ATR_TP_MULT = 2.0


def mean_reversion_specs(pair: str) -> List[StrategySpec]:
    """The confirmed BB+RSI mean-reversion strategy as two pinned specs.

    Returns ``[long_spec, short_spec]`` — same family, one per side, so each
    flows through the gate and probation as its own version.
    """
    long_spec = StrategySpec(
        name=f"MeanRevBB Long {pair}",
        description=(
            "Confirmed mean reversion (BB+RSI, H1): LONG when price closes "
            "below the lower Bollinger Band (20, 2.0) and RSI(14) < 30, "
            "SL 1.5 x ATR, TP 2.0 x ATR."
        ),
        direction="long",
        timeframe="H1",
        entry_conditions=[
            SpecCondition(left="price", verb="below", right="bb_lower"),
            SpecCondition(left="rsi", verb="below", right=30),
            SpecCondition(left="atr", verb="above", right=MIN_ATR_THRESHOLD),
        ],
        atr_sl_mult=_ATR_SL_MULT,
        atr_tp_mult=_ATR_TP_MULT,
    )
    short_spec = StrategySpec(
        name=f"MeanRevBB Short {pair}",
        description=(
            "Confirmed mean reversion (BB+RSI, H1): SHORT when price closes "
            "above the upper Bollinger Band (20, 2.0) and RSI(14) > 70, "
            "SL 1.5 x ATR, TP 2.0 x ATR."
        ),
        direction="short",
        timeframe="H1",
        entry_conditions=[
            SpecCondition(left="price", verb="above", right="bb_upper"),
            SpecCondition(left="rsi", verb="above", right=70),
            SpecCondition(left="atr", verb="above", right=MIN_ATR_THRESHOLD),
        ],
        atr_sl_mult=_ATR_SL_MULT,
        atr_tp_mult=_ATR_TP_MULT,
    )
    return [long_spec, short_spec]
