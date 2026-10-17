"""Reproducibility metadata for every persisted backtest/WF/gate result (Stage A1).

Alex's constitution: **deterministic code owns every number**. The backtest
engine has no RNG, so any run can be replayed exactly *if* we record what
shaped it. ``build_repro`` captures that context once, at run time; every
persisted result (gate ``ValidationReport``, experiment ``dataset``,
walk-forward payloads) carries one.

A repro dict is intentionally JSON-safe (booleans/numbers/strings/lists of
primitives) so it can live in any ``JSONType`` column without a migration.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Bump when the set of fields (or their meaning) changes so stored repros
# can be interpreted correctly years later. Semantic, not code, versioning.
REPRO_SCHEMA_VERSION = 1

# Package version — engine identity for replay auditing.
try:
    from forex_ai_agent import __version__ as ENGINE_VERSION
except Exception:  # pragma: no cover — defensive; package always defines it
    ENGINE_VERSION = "unknown"

# Thresholds understood by replay (import target kept lazy where needed).
_THRESHOLD_FIELDS = (
    "min_trades_full",
    "min_oos_trades",
    "min_oos_sharpe",
    "max_oos_drawdown",
    "max_overfitting_score",
    "min_win_rate",
    "min_profit_factor",
)

# Walk-forward shape fields replay needs to slice identically.
_WF_FIELDS = ("window_months", "step_months")

# Signal-params that define a spec strategy's behavior (A2 sensitivity grid).
SIGNAL_PARAM_NAMES = ("atr_sl_mult", "atr_tp_mult", "min_score_pct")


def _jsonable(value: Any) -> Any:
    """Coerce a value to something JSONType-safe (dict/list/str/num/bool/None)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    return str(value)


def rules_hash(rules: Any) -> str:
    """Stable SHA-256 over a strategy's rules, canonically.

    Input is first canonicalized through ``parse_spec`` when possible, so two
    dicts that express the same strategy (e.g. with vs without explicit
    defaults) hash identically. Unparseable input falls back to a stable
    hash of its normalized JSON form.
    """
    import hashlib
    import json

    if rules is None:
        return ""
    if isinstance(rules, str):
        try:
            rules = json.loads(rules)
        except (TypeError, ValueError):
            return hashlib.sha256(rules.encode("utf-8")).hexdigest()
    if isinstance(rules, dict) and "entry_conditions" in rules:
        try:
            from forex_ai_agent.strategy.spec import parse_spec

            spec, errors = parse_spec(rules)
            if spec is not None:
                rules = spec.model_dump()
        except Exception:  # noqa: BLE001 — fall back to raw hashing
            pass
    try:
        payload = json.dumps(rules, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        payload = str(rules)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def gate_thresholds_dict(thresholds: Any) -> Dict[str, Any]:
    """Snapshot a ``GateThresholds``-like object into a plain dict."""
    if thresholds is None:
        return {}
    if isinstance(thresholds, dict):
        return {k: _jsonable(thresholds.get(k)) for k in _THRESHOLD_FIELDS if k in thresholds}
    return {name: _jsonable(getattr(thresholds, name, None)) for name in _THRESHOLD_FIELDS}


def cost_models_dict(config: Any) -> Dict[str, Any]:
    """Snapshot the cost/execution assumptions of a ``BacktestConfig``.

    Model objects are recorded as class name + constructor-relevant state so
    the replay can rebuild (or at least identify) the exact model used.
    """
    def _model(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return _jsonable(obj)
        state = getattr(obj, "__dict__", None)
        return {
            "type": type(obj).__name__,
            "state": _jsonable(state) if state else {},
        }

    return {
        "initial_balance": _jsonable(getattr(config, "initial_balance", None)),
        "leverage": _jsonable(getattr(config, "leverage", None)),
        "margin_rate": _jsonable(getattr(config, "margin_rate", None)),
        "spread_model": _model(getattr(config, "spread_model", None)),
        "commission_model": _model(getattr(config, "commission_model", None)),
        "swap_model": _model(getattr(config, "swap_model", None)),
        "slippage_model": _model(getattr(config, "slippage_model", None)),
        "fill_policy": str(getattr(config, "fill_policy", "")),
        "warmup_bars": _jsonable(getattr(config, "warmup_bars", None)),
    }


def build_repro(
    *,
    strategy_id: Optional[str] = None,
    strategy_name: Optional[str] = None,
    rules: Any = None,
    symbol: str = "",
    timeframe: str = "",
    bar_count: Optional[int] = None,
    data_start: Any = None,
    data_end: Any = None,
    feed_provider: Optional[str] = None,
    feed_chain: Optional[Sequence[str]] = None,
    thresholds: Any = None,
    backtest_config: Any = None,
    walk_forward: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the canonical repro dict for a run.

    Every parameter maps to a roadmap A1 field; ``None``/unknown values are
    recorded as ``None`` honestly (e.g. ``random_seed`` — the engine has no
    RNG, and we say so rather than inventing one).
    """
    repro: Dict[str, Any] = {
        "repro_schema_version": REPRO_SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "recorded_at": datetime.utcnow().isoformat(),
        "random_seed": None,  # engine is deterministic by design; no RNG exists
        "deterministic": True,
    }
    strategy: Dict[str, Any] = {
        "strategy_id": strategy_id,
        "strategy_name": strategy_name,
        "rules_sha256": rules_hash(rules),
    }
    dataset: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "bar_count": bar_count,
        "start": data_start.isoformat() if isinstance(data_start, datetime) else data_start,
        "end": data_end.isoformat() if isinstance(data_end, datetime) else data_end,
        "provider": feed_provider,
        "feed_chain": list(feed_chain) if feed_chain else None,
    }
    if thresholds is not None:
        repro["gate_thresholds"] = gate_thresholds_dict(thresholds)
    if backtest_config is not None:
        repro["cost_models"] = cost_models_dict(backtest_config)
    if walk_forward is not None:
        repro["walk_forward"] = _jsonable(dict(walk_forward))
    repro["strategy"] = strategy
    repro["dataset"] = dataset
    if extra:
        repro.update(_jsonable(dict(extra)))
    return repro


def feed_identity() -> Dict[str, Any]:
    """Best-effort snapshot of which feed served the bars.

    Never raises: a repro is metadata and must not break the run it describes.
    """
    provider: Optional[str] = None
    chain: List[str] = []
    try:
        from forex_ai_agent.data.providers.factory import get_feed_chain

        fc = get_feed_chain()
        provider = getattr(fc, "active_source", None)
        chain = [
            p.name
            for p in (getattr(fc, "providers", None) or [])
            if getattr(p, "name", None)
        ]
    except Exception as exc:  # pragma: no cover — metadata must never crash runs
        logger.debug("feed identity unavailable for repro: %s", exc)
    return {"provider": provider, "chain": chain or None}


async def replay_spec_run(
    strategy_rules: Any,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    candles_fetcher=None,
    want_walk_forward: bool = False,
    window_months: int = 6,
    step_months: int = 2,
) -> Dict[str, Any]:
    """Re-run a stored repro for a spec strategy and return its metrics.

    Deterministic replay: same rules + same dataset identity ⇒ same numbers
    (the engine has no RNG). Returns the same metric blocks the original run
    produced so callers can compare field-by-field. ``candles_fetcher`` lets
    tests inject synthetic bars instead of hitting the feed chain.
    """
    from datetime import datetime as _dt

    import pandas as pd

    from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest
    from forex_ai_agent.autonomy.validator import SpecSignalGenerator
    from forex_ai_agent.strategy.spec import parse_spec
    from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

    spec, errors = parse_spec(strategy_rules)
    if spec is None:
        raise ValueError(f"stored rules are not a valid spec: {'; '.join(errors[:3])}")

    start = _dt.fromisoformat(start_date)
    end = _dt.fromisoformat(end_date)
    if candles_fetcher is not None:
        candles = candles_fetcher(symbol, timeframe, start, end)
        if hasattr(candles, "__await__"):
            candles = await candles
    else:
        from forex_ai_agent.data.providers.factory import get_feed_chain

        candles = await get_feed_chain().fetch_candles(
            symbol, Timeframe(timeframe), start, end, limit=20000
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

    pip = pip_size_for_symbol(symbol)
    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=100000.0,
        symbols=[symbol],
        timeframe=spec.timeframe or timeframe,
        warmup_bars=60,
    )
    result = run_backtest({symbol: df}, SpecSignalGenerator(spec, pip_size=pip), config)

    raw = dict(result.metrics or {})
    summary = dict(raw.get("summary") or {})
    portfolio = dict(raw.get("portfolio") or {})
    out: Dict[str, Any] = {
        "symbol": symbol,
        "timeframe": timeframe,
        "start": start_date,
        "end": end_date,
        "bars": len(df),
        "metrics": {
            "total_trades": int(summary.get("total_trades", 0)),
            "win_rate": float(summary.get("win_rate", 0.0)),
            "profit_factor": float(summary.get("profit_factor", 0.0)),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0.0)),
            "total_return_pct": float(summary.get("total_return_pct", 0.0)),
            "max_drawdown": float(portfolio.get("max_drawdown_pct", 0.0)) / 100.0,
        },
    }
    if want_walk_forward:
        from forex_ai_agent.agent.experiments import run_walk_forward_on_df

        out["walk_forward"] = run_walk_forward_on_df(
            spec,
            df,
            symbol,
            timeframe,
            window_months=window_months,
            step_months=step_months,
        )
    return out
