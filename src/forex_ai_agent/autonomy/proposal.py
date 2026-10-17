"""Proposal loop: research brief → LLM spec → validation → version.

The LLM's ONLY job is to emit a strategy-spec JSON object given a research
brief. Its output is parsed with the whitelist schema (invalid vocabulary →
rejected with reasons), validated against history by the BacktestValidator,
and only surviving specs are versioned by the StrategyVersionManager for
paper probation. Without an LLM, a deterministic fallback spec keeps the
loop runnable (still gated by the same backtest).

Constitution: the LLM never sees account state and never emits numbers that
matter — the only numerics it may propose are the range-limited ATR
multipliers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..research.briefs import PairBrief, build_pair_brief
from ..strategy.spec import StrategySpec, fallback_spec, parse_spec
from ..strategy.versioning import StrategyVersion, StrategyVersionManager, VersionType
from .validator import BacktestValidator, ValidationReport

logger = logging.getLogger(__name__)


# Stage C4: a shape needs this many stored gate rejections before the
# deterministic weak-hypothesis veto refuses to re-backtest it.
_WEAK_SHAPE_MIN_ATTEMPTS = 3


async def _load_veto_conclusions() -> List[Dict[str, Any]]:
    """Veto's data seam: raw rejected-run conclusions (tests patch this)."""
    from forex_ai_agent.agent.research_lessons import load_rejected_conclusions

    return await load_rejected_conclusions()

_PROPOSAL_SYSTEM_PROMPT = """You are Alex, a forex strategy research agent. You receive a \
deterministic research brief for one currency pair and must propose ONE strategy as a JSON \
object. HARD RULES:
- Output ONLY a JSON object — no prose, no markdown, no thinking process.
- Use ONLY this exact schema:
  {"name": str (3-80 chars), "description": str, "direction": "long"|"short"|"both",
   "timeframe": "H1", "min_score_pct": 0.6,
   "entry_conditions": [{"left": indicator, "verb": verb, "right": indicator-or-number}, ...],
   "atr_sl_mult": float, "atr_tp_mult": float}
- Indicators allowed: price, ema, sma, sma_slow, rsi, atr, bb_upper, bb_middle, bb_lower (Bollinger Bands are FIXED 20-period / 2.0 sigma — never propose periods or sigmas).
- Verbs allowed: above, below, crosses_above, crosses_below, rising, falling.
- Numeric thresholds: rsi 1-99, ema/sma/sma_slow are PRICE levels (use 2-500 band or another \
indicator), price/atr accept any positive value. Never invent other indicators or fields.
- Combine 1-4 conditions that AND together into a coherent entry logic for the observed regime.
- atr_sl_mult 0.5-5.0 (stop distance in ATRs), atr_tp_mult 1.0-10.0 (target distance in ATRs).
- You are proposing RULES, not trades: no entries, stops, targets, sizes, or money figures.
- If the brief shows a strong trend, propose a trend-following entry; if ranging, propose \
mean-reversion at the extremes. Explain your reasoning inside "description"."""


@dataclass
class ProposalResult:
    """Outcome of one propose→validate→version cycle."""

    pair: str
    status: str  # "validated" | "rejected" | "error"
    spec: Optional[StrategySpec] = None
    spec_dict: Optional[Dict[str, Any]] = None
    used_llm: bool = False
    llm_error: Optional[str] = None
    validation: Optional[ValidationReport] = None
    version: Optional[StrategyVersion] = None
    brief: Optional[Dict[str, Any]] = None
    messages: List[str] = field(default_factory=list)
    proposed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _schema_hint() -> str:
    return (
        '{"name": "...", "description": "...", "direction": "long|short|both", '
        '"timeframe": "H1", "min_score_pct": 0.6, '
        '"entry_conditions": [{"left": "sma", "verb": "above", "right": "sma_slow"}], '
        '"atr_sl_mult": 1.5, "atr_tp_mult": 3.0}'
    )


async def fetch_history_for(
    pipeline: Any, pair: str, days: int = 180, timeframe: str = "H1"
) -> Any:
    """H1 candle history as an indexed DataFrame (for the backtest gate)."""
    import pandas as pd

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    # Ask for the full span: without an explicit limit the provider default
    # (1000 bars) silently truncates deep history on capable feeds like MT5.
    bars_per_day = {"M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
    limit = min(days * bars_per_day.get(timeframe.upper(), 24) + 50, 100_000)
    bars = await pipeline.data_provider.fetch_candles(
        pair, timeframe, start, end=end, limit=limit
    )
    if not bars:
        raise RuntimeError(f"no history returned for {pair}")
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
        },
        index=[b.timestamp for b in bars],
    )
    df.index = pd.to_datetime(df.index)
    return df[~df.index.duplicated(keep="last")].sort_index()


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first parseable JSON object out of arbitrary LLM output.

    Tolerates markdown fences, preamble prose, and trailing commentary.
    Two repair layers, in order:
    1. ``strict=False`` — allows raw control characters (newlines/tabs)
       inside strings, the most common near-JSON model output.
    2. Candidate scan — try every balanced ``{...}`` block; a later one
       may parse even if an earlier (e.g. reasoning) object is malformed.
    """
    import json

    if not isinstance(text, str):
        return None
    cleaned = text.strip()

    def _scan_from(start: int) -> Optional[Dict[str, Any]]:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    for kwargs in ({}, {"strict": False}):
                        try:
                            parsed = json.loads(candidate, **kwargs)
                            return parsed if isinstance(parsed, dict) else None
                        except json.JSONDecodeError:
                            continue
                    return None
        return None

    pos = cleaned.find("{")
    while pos != -1:
        parsed = _scan_from(pos)
        if parsed is not None:
            return parsed
        pos = cleaned.find("{", pos + 1)
    return None


async def _llm_propose(
    llm: Any, brief: PairBrief
) -> tuple[Optional[StrategySpec], List[str], Optional[str]]:
    """Ask the reasoning tier for a spec. Returns (spec, errors, llm_error)."""
    system = _PROPOSAL_SYSTEM_PROMPT + (
        f"\n\nRespond with ONLY a JSON object matching this shape:\n{_schema_hint()}"
    )
    try:
        raw = await llm.reason(
            system,
            {
                "task": "propose one strategy for the pair described below",
                "brief": brief.to_prompt_dict(),
            },
        )
    except Exception as exc:  # noqa: BLE001 — LLM problems must never stop the loop
        return None, [], f"{type(exc).__name__}: {exc}"
    parsed = _extract_json_object(raw) if isinstance(raw, str) else raw
    if parsed is None:
        preview = (raw[:120] + "…") if isinstance(raw, str) and len(raw) > 120 else raw
        return None, [], f"no JSON object found in LLM output: {preview!r}"
    spec, errors = parse_spec(parsed)
    return spec, errors, None


def _adaptive_probation_quota(
    validation: Optional[ValidationReport],
    default_trades: int = 20,
    default_days: int = 14,
) -> tuple[int, int]:
    """Size probation to the strategy's VALIDATED trade frequency.

    A 20-trades/14-days quota is impossible for a system that trades ~30x a
    year (it would auto-expire with ~2 trades). Instead: expect 1.5x the
    validated rate within the window, keep a 5-trade statistical floor, and
    stretch the window (capped at 90 days) until that quota is reachable.
    """
    import math
    from datetime import datetime as _dt

    try:
        full = (validation.full_backtest or {}) if validation else {}
        rng = (validation.data_range or {}) if validation else {}
        total = int(full.get("total_trades", 0))
        start = _dt.fromisoformat(str(rng.get("start"))[:19])
        end = _dt.fromisoformat(str(rng.get("end"))[:19])
        span_days = max((end - start).days, 1)
        rate = total / span_days  # trades per day
        if rate <= 0:
            return default_trades, default_days
        expected_14d = rate * default_days
        trades = max(5, min(default_trades, round(expected_14d * 1.5)))
        need_days = math.ceil(trades / rate)
        window = max(default_days, min(90, need_days))
        return trades, window
    except Exception:  # noqa: BLE001 — sizing must never block a proposal
        return default_trades, default_days


async def _log_result(result: ProposalResult, enabled: bool) -> None:
    """Persist the run to the research log (best-effort, never raises)."""
    if not enabled:
        return
    from .research_log import log_research_run

    await log_research_run(result)


async def propose_for_pair(
    pipeline: Any,
    pair: str,
    candles_df: Any,
    *,
    use_llm: bool = True,
    validator: Optional[BacktestValidator] = None,
    version_manager: Optional[StrategyVersionManager] = None,
    version_manager_dir: str = "strategies/versions",
    timeframe: str = "H1",
    log_to_db: bool = True,
    enter_probation: bool = True,
) -> ProposalResult:
    """One full autonomy-loop iteration for ``pair``.

    Steps: research brief → LLM (or fallback) spec → backtest gate →
    version (only survivors). ``candles_df`` must be an indexed DataFrame
    of H1 bars (the same frame the validator backtests against).
    Every terminal outcome is persisted to the research log unless
    ``log_to_db=False`` (tests / dry runs).
    """
    result = ProposalResult(pair=pair, status="error")
    validator = validator or BacktestValidator()

    # 1. Research brief (deterministic).
    try:
        brief = await build_pair_brief(pipeline, pair, timeframe=timeframe)
        result.brief = brief.to_prompt_dict()
    except Exception as exc:  # noqa: BLE001
        result.messages.append(f"brief failed: {type(exc).__name__}: {exc}")
        await _log_result(result, log_to_db)
        return result

    # 2. LLM proposal (or deterministic fallback).
    spec: Optional[StrategySpec] = None
    if use_llm:
        spec, errors, llm_error = await _llm_propose(pipeline.reasoning_agent.llm, brief)
        result.used_llm = llm_error is None
        result.llm_error = llm_error
        if llm_error:
            result.messages.append(f"LLM unavailable ({llm_error}); using fallback spec")
        elif spec is None:
            result.messages.append(f"LLM spec rejected by whitelist: {errors}")
    if spec is None:
        spec = fallback_spec(pair)
        result.messages.append("using deterministic fallback spec")

    result.spec = spec
    result.spec_dict = spec.model_dump()

    # 2b. Stage C4 weak-hypothesis veto — deterministic, BEFORE the gate.
    # A spec shape with enough stored gate REJECTIONS is not re-backtested;
    # the honest rejection reason (with the stored sample) is persisted to the
    # research log so the loop learns instead of burning compute. Loader
    # failure → no veto, the gate still runs (never blocks a proposal).
    try:
        from forex_ai_agent.agent.research_lessons import (
            spec_shape,
            veto_reason,
            weak_shapes,
        )

        conclusions = await _load_veto_conclusions()
        w_shapes = weak_shapes(conclusions, min_attempts=_WEAK_SHAPE_MIN_ATTEMPTS)
        shape = spec_shape(spec.model_dump())
        vetoed = next(
            (l for l in w_shapes if shape[0] == l.direction and shape[1] == l.timeframe
             and list(shape[2]) == list(l.entry_verbs)),
            None,
        )
        if vetoed is not None:
            reason = veto_reason(vetoed, attempts_seen=vetoed.count)
            result.status = "rejected"
            result.validation = ValidationReport(
                spec_name=spec.name,
                passed=False,
                reasons=[reason],
            )
            result.messages.append(reason)
            await _log_result(result, log_to_db)
            return result
    except Exception as exc:  # noqa: BLE001 — the veto must never break the loop
        logger.warning("weak-shape veto check failed (proceeding to gate): %s: %s",
                       type(exc).__name__, exc)

    # 3-5. Deterministic gate → version → probation (shared with propose_spec).
    return await _gate_version_probation(
        result,
        pair,
        spec,
        candles_df,
        validator=validator,
        version_manager=version_manager,
        version_manager_dir=version_manager_dir,
        enter_probation=enter_probation,
        log_to_db=log_to_db,
    )


async def propose_spec(
    pair: str,
    spec: StrategySpec,
    candles_df: Any,
    *,
    validator: Optional[BacktestValidator] = None,
    version_manager: Optional[StrategyVersionManager] = None,
    version_manager_dir: str = "strategies/versions",
    enter_probation: bool = True,
    log_to_db: bool = True,
) -> ProposalResult:
    """Deterministic entry point for a ready spec (gate → version → probation).

    Used by the human-confirmed reference strategies (``strategy/templates``):
    they flow through the SAME gate, registry and probation as LLM candidates,
    but skip the speculative machinery (research brief, LLM proposal,
    weak-shape veto) — a reference spec is authored, not guessed. The gate
    still decides its fate on real data.
    """
    result = ProposalResult(pair=pair, status="error")
    result.spec = spec
    result.spec_dict = spec.model_dump()
    result.messages.append("reference spec supplied deterministically (no LLM proposal)")
    return await _gate_version_probation(
        result,
        pair,
        spec,
        candles_df,
        validator=validator or BacktestValidator(),
        version_manager=version_manager,
        version_manager_dir=version_manager_dir,
        enter_probation=enter_probation,
        log_to_db=log_to_db,
    )


async def _gate_version_probation(
    result: ProposalResult,
    pair: str,
    spec: StrategySpec,
    candles_df: Any,
    *,
    validator: BacktestValidator,
    version_manager: Optional[StrategyVersionManager],
    version_manager_dir: str,
    enter_probation: bool,
    log_to_db: bool,
) -> ProposalResult:
    """Steps 3-5 of the loop, shared by the LLM path and ``propose_spec``."""
    report = await validator.validate_spec(spec, candles_df, symbol=pair)
    result.validation = report
    if not report.passed:
        result.status = "rejected"
        await _log_result(result, log_to_db)
        return result

    # 4. Version the survivor.
    try:
        vm = version_manager or StrategyVersionManager(version_manager_dir)
        version = vm.create_version(
            name=spec.name,
            description=spec.description,
            rules=spec.model_dump(),
            parameters={
                "atr_sl_mult": spec.atr_sl_mult,
                "atr_tp_mult": spec.atr_tp_mult,
                "direction": spec.direction,
                "timeframe": spec.timeframe,
            },
            change_type=VersionType.MINOR,
        )
        result.version = version
        result.status = "validated"
        result.messages.append(f"versioned as {version.id}")
    except Exception as exc:  # noqa: BLE001
        result.messages.append(f"versioning failed: {type(exc).__name__}: {exc}")
        result.status = "error"
        await _log_result(result, log_to_db)
        return result

    # 5. Enter paper probation (the arena where the human later judges it).
    if not enter_probation:
        result.messages.append("probation skipped (enter_probation=False)")
        return result
    try:
        from .probation import ProbationStore

        store = ProbationStore()
        req_trades, req_days = _adaptive_probation_quota(result.validation)
        prob = await store.start(
            version_id=str(version.id),
            strategy_name=spec.name,
            pair=pair,
            spec_dict=spec.model_dump(),
            required_trades=req_trades,
            required_days=req_days,
        )
        result.messages.append(
            f"probation active: {prob.required_trades} trades / {prob.required_days} days"
        )
    except Exception as exc:  # noqa: BLE001 — probation DB is optional to run
        result.messages.append(
            f"probation not started ({type(exc).__name__}: {exc}); "
            "version exists but needs manual probation"
        )
    await _log_result(result, log_to_db)
    return result
