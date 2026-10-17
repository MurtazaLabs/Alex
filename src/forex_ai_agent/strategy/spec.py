"""Strategy specs: the ONLY format in which the LLM may propose strategies.

A spec is pure structure — indicator/price conditions with whitelisted verbs.
Compiling a spec to ``SignalRule`` objects is deterministic: Alex's code owns
every computation. The spec carries NO trade numbers (no entries, stops,
targets, sizes) — those always come from SignalAgent/RiskAgent math, in live
trading from the existing pipeline, and in backtests from ATR brackets
computed by the deterministic ``SpecSignalGenerator`` (autonomy/validator.py).

Verb support:
    - ``above`` / ``below``: work everywhere (live rules are stateless).
    - ``crosses_above`` / ``crosses_below`` / ``rising`` / ``falling`` need
      the previous bar, so they are backtest-only. ``compile()`` raises on
      them; the backtest adapter (SpecSignalGenerator) evaluates them
      statefully. The proposal prompt tells the LLM this.

Constitution mapping (docs/Forex docmentation/ALEX_REASONING_PLAN.md §1):
    - LLM proposes structure → this module validates + compiles it.
    - Anything outside the whitelist is rejected, never guessed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..analysis.signal_rules.base_rule import Direction, MarketContext, SignalRule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Whitelisted vocabulary (the LLM's entire universe)
# ---------------------------------------------------------------------------

# Indicators available inside the backtest engine's indicator snapshot:
# name -> (engine key, sane threshold low, sane threshold high)
INDICATORS: Dict[str, Tuple[str, float, float]] = {
    "ema": ("ema_20", 2.0, 500.0),
    "sma": ("sma_20", 2.0, 500.0),
    "sma_slow": ("sma_50", 2.0, 500.0),
    "rsi": ("rsi_14", 1.0, 99.0),
    "atr": ("atr_14", 0.0, 1e9),
    "price": ("close", 0.0, 1e9),
    # Bollinger Bands — FIXED 20-period / 2.0 sigma (the human-confirmed
    # mean-reversion parameters, 2026-10-16). Period and sigma are NOT
    # tunable: the spec carries structure only, the engine owns the math.
    "bb_upper": ("bb_upper", 0.0, 1e9),
    "bb_middle": ("bb_middle", 0.0, 1e9),
    "bb_lower": ("bb_lower", 0.0, 1e9),
}

VERBS = {
    "above",
    "below",
    "crosses_above",  # backtest-only (stateful)
    "crosses_below",  # backtest-only (stateful)
    "rising",         # backtest-only (stateful)
    "falling",        # backtest-only (stateful)
}

STATEFUL_VERBS = {"crosses_above", "crosses_below", "rising", "falling"}

# Common LLM spellings that map unambiguously onto whitelist entries.
_INDICATOR_ALIASES = {
    "close": "price",
    "c": "price",
    "ema20": "ema",
    "ema_20": "ema",
    "sma20": "sma",
    "sma_20": "sma",
    "sma50": "sma_slow",
    "sma_50": "sma_slow",
    "rsi14": "rsi",
    "rsi_14": "rsi",
    "atr14": "atr",
    "atr_14": "atr",
    "bbupper": "bb_upper",
    "upper_band": "bb_upper",
    "bollinger_upper": "bb_upper",
    "bbmiddle": "bb_middle",
    "bbmid": "bb_middle",
    "middle_band": "bb_middle",
    "bollinger_middle": "bb_middle",
    "bblower": "bb_lower",
    "lower_band": "bb_lower",
    "bollinger_lower": "bb_lower",
}


def _normalize_indicator(name: str) -> str:
    """Map common aliases onto the canonical whitelist names."""
    n = str(name).strip().lower()
    return _INDICATOR_ALIASES.get(n, n)

# Backtest SL/TP bracket knobs (the ONLY numeric params a spec may carry,
# constrained here — never entries/targets/sizes themselves).
ATR_SL_MULT = (0.5, 5.0)
ATR_TP_MULT = (1.0, 10.0)

MAX_CONDITIONS = 8


class SpecCondition(BaseModel):
    """One rule condition: ``left verb right`` over whitelisted operands."""

    model_config = ConfigDict(extra="forbid")

    left: str = Field(..., description="Indicator name from the whitelist")
    verb: str = Field(..., description=f"One of {sorted(VERBS)}")
    right: Union[str, float, int] = Field(
        ..., description="Indicator name or numeric literal"
    )

    @field_validator("left", mode="before")
    @classmethod
    def _left_known(cls, v: Any) -> Any:
        n = _normalize_indicator(v)
        if n not in INDICATORS:
            raise ValueError(f"unknown indicator {v!r}; allowed: {sorted(INDICATORS)}")
        return n

    @field_validator("verb")
    @classmethod
    def _verb_known(cls, v: str) -> str:
        if v not in VERBS:
            raise ValueError(f"unknown verb {v!r}; allowed: {sorted(VERBS)}")
        return v

    @field_validator("right", mode="before")
    @classmethod
    def _right_scalar(cls, v: Any) -> Any:
        # LLMs emit numbers as JSON numbers; normalize to str for storage.
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise ValueError("right must be an indicator name or a number")
        return str(v) if isinstance(v, (int, float)) else v


class StrategySpec(BaseModel):
    """A full strategy specification.

    ``extra="forbid"`` is a constitution control: any field outside the
    schema (e.g. a sneaky ``stop_loss`` or ``lot_size``) invalidates the
    whole proposal instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=3, max_length=80)
    description: str = Field(default="", max_length=500)
    direction: str = Field(default="both", description="long | short | both")
    timeframe: str = Field(default="H1")
    entry_conditions: List[SpecCondition] = Field(..., min_length=1, max_length=MAX_CONDITIONS)
    atr_sl_mult: float = Field(default=1.5)
    atr_tp_mult: float = Field(default=3.0)
    min_score_pct: float = Field(default=0.6)

    @field_validator("direction")
    @classmethod
    def _direction_known(cls, v: str) -> str:
        v = v.lower()
        if v not in {"long", "short", "both"}:
            raise ValueError("direction must be long|short|both")
        return v

    @field_validator("atr_sl_mult")
    @classmethod
    def _sl_mult_range(cls, v: float) -> float:
        if not ATR_SL_MULT[0] <= v <= ATR_SL_MULT[1]:
            raise ValueError(f"atr_sl_mult must be in {ATR_SL_MULT}")
        return v

    @field_validator("atr_tp_mult")
    @classmethod
    def _tp_mult_range(cls, v: float) -> float:
        if not ATR_TP_MULT[0] <= v <= ATR_TP_MULT[1]:
            raise ValueError(f"atr_tp_mult must be in {ATR_TP_MULT}")
        return v

    # ------------------------------------------------------------------
    # Compilation (deterministic)
    # ------------------------------------------------------------------

    def compile(self, direction: Optional[str] = None) -> List[SignalRule]:
        """Compile entry conditions into one ANDed SignalRule (deterministic).

        Live rules are stateless: only ``above``/``below`` are supported
        here — stateful verbs raise ``ValueError`` with a clear message
        (backtests evaluate them statefully via SpecSignalGenerator).
        The rule's ``condition`` closure reads only ``MarketContext``
        fields. No eval, no globals.
        """
        stateful = [c.verb for c in self.entry_conditions if c.verb in STATEFUL_VERBS]
        if stateful:
            raise ValueError(
                f"stateful verbs {stateful} need previous-bar history; live "
                "compilation supports above/below only (backtests handle all verbs)"
            )
        cond = [c for c in self.entry_conditions]
        dir_word = (direction or self.direction).lower()

        def _condition(ctx: MarketContext) -> bool:
            if dir_word == "long" and ctx.direction == Direction.SHORT:
                return False
            if dir_word == "short" and ctx.direction == Direction.LONG:
                return False
            ind = {
                "price": ctx.current_price,
                "ema": ctx.indicators.ema_20,
                "sma": ctx.indicators.sma_20,
                "sma_slow": ctx.indicators.sma_50,
                "rsi": ctx.indicators.rsi,
                "atr": ctx.indicators.atr,
                "bb_upper": ctx.indicators.bb_upper,
                "bb_middle": ctx.indicators.bb_middle,
                "bb_lower": ctx.indicators.bb_lower,
            }
            for c in cond:
                left = ind.get(c.left)
                right_key = _normalize_indicator(c.right)
                right = float(c.right) if _is_number(c.right) else ind.get(right_key)
                if left is None or right is None:
                    return False  # missing data → condition not satisfied
                ok = left > right if c.verb == "above" else left < right
                if not ok:
                    return False
            return True

        weight = round(min(1.0, 0.4 + 0.1 * len(cond)), 2)
        return [
            SignalRule(
                name=f"{self.name}:entry",
                condition=_condition,
                weight=weight,
                required=True,
                timeframe=self.timeframe,
                description=self.description or f"{len(cond)} compiled conditions",
                category="PROPOSED",
            )
        ]

    def has_stateful_verbs(self) -> bool:
        return any(c.verb in STATEFUL_VERBS for c in self.entry_conditions)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Parsing / validation entry points
# ---------------------------------------------------------------------------

def parse_spec(raw: Any) -> Tuple[Optional[StrategySpec], List[str]]:
    """Parse + validate an LLM-proposed object. Returns (spec|None, errors)."""
    if not isinstance(raw, dict):
        return None, ["proposal is not a JSON object"]
    try:
        spec = StrategySpec(**raw)
    except ValidationError as exc:
        errors = []
        for e in exc.errors():
            loc = ".".join(str(p) for p in e.get("loc", []))
            errors.append(f"{loc or 'spec'}: {e.get('msg', 'invalid')}")
        return None, errors

    errors: List[str] = []
    for i, c in enumerate(spec.entry_conditions):
        if _is_number(c.right):
            _key, lo, hi = INDICATORS[c.left]
            val = float(c.right)
            if not lo <= val <= hi:
                errors.append(
                    f"entry_conditions[{i}]: threshold {val} out of range "
                    f"for {c.left} ({lo}-{hi})"
                )
        elif _normalize_indicator(c.right) not in INDICATORS:
            errors.append(
                f"entry_conditions[{i}]: right operand {c.right!r} is neither "
                f"a number nor a known indicator ({sorted(INDICATORS)})"
            )
    if errors:
        return None, errors
    return spec, []


def fallback_spec(pair: str) -> StrategySpec:
    """Deterministic default proposal (used when no LLM is available).

    A conservative trend-follow starter: fast average above slow average and
    price above the fast average — the classic trend filter. The backtest gate
    still decides its fate; the fallback exists so the loop never blocks.
    """
    return StrategySpec(
        name=f"FallbackTrend {pair}",
        description="Deterministic fallback: SMA trend filter (sma above sma_slow, price above sma).",
        direction="both",
        timeframe="H1",
        entry_conditions=[
            SpecCondition(left="sma", verb="above", right="sma_slow"),
            SpecCondition(left="price", verb="above", right="sma"),
        ],
        atr_sl_mult=1.5,
        atr_tp_mult=3.0,
        min_score_pct=0.6,
    )
