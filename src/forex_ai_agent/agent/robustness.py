"""Cross-pair / regime robustness (Stage B3).

Runs a strategy's walk-forward across its intended pair AND the major pairs,
tagging every OOS window with a deterministic market regime (trending /
ranging / volatile), then aggregates per-pair and per-regime. This feeds the
last two Health lights (PAIR GENERALITY, REGIME ROBUSTNESS) — the roadmap's
B3 acceptance: "health shows all seven lights; generality/robustness backed by
per-pair, per-regime WF rows".

Constitution: every number here is deterministic code over stored/fetched
bars. The LLM may narrate the card; it never decides what passes.

The result is stored on the ``StrategyVersion.backtest_results["robustness"]``
so health, explain cards and tools read it back without re-running anything.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Budget: a sweep costs (pairs × ≤8 windows) backtests. Four pairs keeps a
# full sweep inside ~32 backtests — same order as the A2b grid budget.
_MAX_PAIRS = 4
_DEFAULT_EXTRA_PAIRS = ("EUR/USD", "GBP/USD", "USD/JPY")

# A regime bucket needs at least this many OOS windows before its mean Sharpe
# is evidence rather than noise (4-window WF × 4 pairs ⇒ thin buckets happen).
_MIN_REGIME_WINDOWS = 3

ROBUSTNESS_SCHEMA_VERSION = 1


def classify_regime(close: Any, frame_vol: Optional[float] = None) -> str:
    """Deterministic regime tag for one OOS window of closes.

    - ``volatile``: window return volatility clearly exceeds the pair's own
      full-period volatility (ratio > 1.5) — chaos dominates whatever the
      direction.
    - ``trending``: net drift is statistically significant over the window —
      a t-statistic ``|mean| / std * sqrt(n)`` above 2.0 (scale-free: judged
      the same at any window length, relative to the pair's OWN noise).
    - ``ranging``: everything else.
    """
    try:
        import pandas as pd

        ret = pd.Series(close).pct_change().dropna()
        if len(ret) < 8:
            return "ranging"
        vol = float(ret.std())
        if vol <= 0:
            return "ranging"
        if frame_vol and frame_vol > 0 and vol > 1.5 * frame_vol:
            return "volatile"
        t_stat = abs(float(ret.mean())) / vol * (len(ret) ** 0.5)
        return "trending" if t_stat > 2.0 else "ranging"
    except Exception:  # noqa: BLE001 — a tag must never fail a window
        return "ranging"


def _regime_bucket() -> Dict[str, Any]:
    return {
        "windows": 0,
        "trades": 0,
        "oos_sharpes": [],
        "pairs": set(),
    }


def _finalize_regimes(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in sorted(buckets):
        b = buckets[name]
        sharpes = b.pop("oos_sharpes")
        pairs = b.pop("pairs")
        if not sharpes:
            continue
        out[name] = {
            "windows": b["windows"],
            "trades": b["trades"],
            "distinct_pairs": len(pairs),
            "mean_oos_sharpe": round(sum(sharpes) / len(sharpes), 4),
            "positive_windows": sum(1 for s in sharpes if s > 0),
            "worst_oos_sharpe": round(min(sharpes), 4),
        }
    return out


def _summarize_pairs(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    usable = [r for r in rows if r.get("windows_tested", 0) > 0]
    profitable = [r for r in usable if float((r.get("aggregate") or {}).get("oos_sharpe_mean", 0)) > 0]
    failing = [r for r in usable if r not in profitable]
    return {
        "tested": len(rows),
        "usable": len(usable),
        "profitable": len(profitable),
        "failing": len(failing),
        "profitable_pairs": [r["symbol"] for r in profitable],
        "failing_pairs": [r["symbol"] for r in failing],
    }


def _sweep_verdict(pairs_summary: Dict[str, Any], per_regime: Dict[str, Dict[str, Any]]) -> str:
    """Deterministic overall verdict for the sweep card and health lights."""
    if pairs_summary["usable"] == 0:
        return "NO_DATA"
    regimes_positive = sum(1 for r in per_regime.values() if r["mean_oos_sharpe"] > 0)
    regimes_measured = sum(1 for r in per_regime.values() if r["windows"] >= _MIN_REGIME_WINDOWS)
    worst_regime = min((r["worst_oos_sharpe"] for r in per_regime.values()), default=0.0)
    if worst_regime < -1.0:
        return "FRAGILE"
    if pairs_summary["failing"] == 0 and pairs_summary["profitable"] >= 2 and regimes_positive >= 2:
        return "ROBUST"
    if pairs_summary["profitable"] >= 1 or regimes_positive >= 1:
        return "MIXED"
    return "FRAGILE"


def _resolve_base_pair(strategy: Any) -> str:
    """The pair a strategy is 'about', mirroring the health card's ladder:
    parameters → rules → stored gate-report metrics (probation record checked
    separately by the async caller).
    """
    return (
        (strategy.parameters or {}).get("symbol")
        or (strategy.rules or {}).get("symbol")
        or (strategy.backtest_results or {}).get("symbol")
        or ""
    )


async def run_robustness_sweep(
    strategy_id: str,
    pairs: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = None,
    window_months: int = 6,
    step_months: int = 2,
    initial_balance: float = 100_000.0,
    save: bool = True,
) -> Dict[str, Any]:
    """WF the strategy across its pair + majors, tag regimes, store the sweep.

    Defaults: pairs = the strategy's own pair plus up to three majors; dates =
    trailing ~2 years; timeframe = the strategy's stored timeframe. The result
    is stored under ``backtest_results["robustness"]`` (when ``save``) so
    health and explain cards read stored state only.
    """
    import pandas as pd

    from forex_ai_agent.agent.experiments import run_walk_forward_on_df
    from forex_ai_agent.agent.repro import build_repro, feed_identity
    from forex_ai_agent.data.providers import get_feed_chain
    from forex_ai_agent.utils.timeframes import Timeframe
    from forex_ai_agent.strategy import create_version_manager
    from forex_ai_agent.strategy.spec import parse_spec

    vm = await create_version_manager()
    strategy = vm.get_version(strategy_id) or (
        vm.get_version(strategy_id[:8]) if len(strategy_id) >= 8 else None
    )
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")
    spec, errors = parse_spec(strategy.rules)
    if spec is None:
        raise ValueError(
            f"strategy {strategy_id} rules are not a valid spec: {'; '.join(errors[:3])}"
        )

    base_pair = _resolve_base_pair(strategy)
    if not base_pair:
        # Last resort (the f39b8d40 pattern): the pair lives only on the
        # probation record the strategy started from.
        try:
            from forex_ai_agent.autonomy.probation import ProbationStore

            prob = await ProbationStore().get(strategy.id)
            base_pair = str((prob.pair if prob else "") or "")
        except Exception:  # noqa: BLE001 — probation unavailable is not fatal
            base_pair = ""
    timeframe = timeframe or (strategy.parameters or {}).get("timeframe") or "H1"

    requested = [p for p in (pairs or []) if p]
    if base_pair and base_pair not in requested:
        requested.insert(0, base_pair)
    if not pairs:
        # Default universe: base pair + the majors, up to the budget cap.
        # An explicit caller-supplied pair list is respected verbatim.
        for p in _DEFAULT_EXTRA_PAIRS:
            if len(requested) >= _MAX_PAIRS:
                break
            if p not in requested:
                requested.append(p)
    requested = requested[:_MAX_PAIRS]

    end_dt = datetime.fromisoformat(end_date) if end_date else datetime.utcnow()
    start_dt = datetime.fromisoformat(start_date) if start_date else end_dt - timedelta(days=730)

    provider = get_feed_chain()
    tf = Timeframe(timeframe)
    pair_rows: List[Dict[str, Any]] = []
    regime_buckets: Dict[str, Dict[str, Any]] = {}
    backtests_run = 0

    for symbol in requested:
        is_base = symbol == base_pair
        try:
            candles = await provider.fetch_candles(symbol, tf, start_dt, end_dt, limit=20000)
            if len(candles) < 250:
                pair_rows.append(
                    {
                        "symbol": symbol,
                        "base": is_base,
                        "status": "SKIPPED",
                        "note": f"only {len(candles)} bars — need ≥250",
                        "windows_tested": 0,
                        "windows": [],
                        "aggregate": {},
                    }
                )
                continue
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

            wf = run_walk_forward_on_df(
                spec,
                df,
                symbol,
                timeframe,
                window_months=window_months,
                step_months=step_months,
                initial_balance=initial_balance,
            )
            backtests_run += wf.get("windows_tested", 0) * 2

            frame_vol = float(df["close"].pct_change().dropna().std() or 0.0)
            windows: List[Dict[str, Any]] = []
            for w in wf.get("windows", []):
                row = dict(w)
                regime = "ranging"
                if w.get("oos_start") is not None and w.get("oos_end") is not None:
                    mask = (df.index >= pd.Timestamp(w["oos_start"])) & (
                        df.index <= pd.Timestamp(w["oos_end"])
                    )
                    regime = classify_regime(df.loc[mask, "close"], frame_vol=frame_vol)
                row["regime"] = regime
                windows.append(row)
                bucket = regime_buckets.setdefault(regime, _regime_bucket())
                bucket["windows"] += 1
                bucket["trades"] += int(w.get("oos_trades", 0) or 0)
                bucket["oos_sharpes"].append(float(w.get("oos_sharpe", 0.0) or 0.0))
                bucket["pairs"].add(symbol)

            pair_rows.append(
                {
                    "symbol": symbol,
                    "base": is_base,
                    "status": "OK",
                    "windows_tested": wf.get("windows_tested", 0),
                    "windows": windows,
                    "aggregate": dict(wf.get("aggregate_metrics") or {}),
                    "stability_score": wf.get("stability_score"),
                    "verdict": wf.get("verdict"),
                }
            )
        except Exception as exc:  # noqa: BLE001 — one bad pair must not kill the sweep
            logger.warning(
                "robustness_pair_failed %s: %s", symbol, f"{type(exc).__name__}: {exc}"
            )
            pair_rows.append(
                {
                    "symbol": symbol,
                    "base": is_base,
                    "status": "ERROR",
                    "note": f"{type(exc).__name__}: {exc}",
                    "windows_tested": 0,
                    "windows": [],
                    "aggregate": {},
                }
            )

    per_regime = _finalize_regimes(regime_buckets)
    pairs_summary = _summarize_pairs(pair_rows)
    feed = feed_identity()
    evaluated_at = datetime.utcnow().isoformat()
    result: Dict[str, Any] = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "evaluated_at": evaluated_at,
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "base_pair": base_pair,
        "timeframe": timeframe,
        "pairs": pair_rows,
        "pairs_summary": pairs_summary,
        "per_regime": per_regime,
        "regimes_summary": {
            "tested": len(per_regime),
            "measured": sum(1 for r in per_regime.values() if r["windows"] >= _MIN_REGIME_WINDOWS),
            "positive": sum(1 for r in per_regime.values() if r["mean_oos_sharpe"] > 0),
            "worst_oos_sharpe": min(
                (r["worst_oos_sharpe"] for r in per_regime.values()), default=None
            ),
        },
        "verdict": _sweep_verdict(pairs_summary, per_regime),
        "budget": {
            "pairs_requested": len(requested),
            "pairs_run": len(requested),
            "backtests_run": backtests_run,
            "max_pairs": _MAX_PAIRS,
        },
        "repro": build_repro(
            strategy_id=strategy_id,
            strategy_name=strategy.name,
            rules=strategy.rules,
            symbol=base_pair or ",".join(requested),
            timeframe=timeframe,
            bar_count=None,
            data_start=start_dt,
            data_end=end_dt,
            feed_provider=feed["provider"],
            feed_chain=feed["chain"],
            extra={
                "sweep": {
                    "pairs": requested,
                    "window_months": window_months,
                    "step_months": step_months,
                }
            },
        ),
    }

    if save:
        try:
            backtest_results = dict(strategy.backtest_results or {})
            backtest_results["robustness"] = result
            strategy.backtest_results = backtest_results
            vm.update_version(strategy)
        except Exception as exc:  # noqa: BLE001 — storage failure must not lose the report
            result["save_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("robustness_save_failed: %s", result["save_error"])

    return result


def render_robustness_text(result: Dict[str, Any]) -> str:
    """Deterministic text card for tool narration (no invented numbers)."""
    lines = [
        f"ROBUSTNESS SWEEP — {result.get('strategy_name', result.get('strategy_id', '?'))}",
        f"Verdict: {result.get('verdict', '?')}  (base pair {result.get('base_pair') or '?'} {result.get('timeframe', '')})",
    ]
    s = result.get("pairs_summary") or {}
    lines.append(
        f"  Pairs: {s.get('profitable', 0)}/{s.get('usable', 0)} OOS-profitable "
        f"({', '.join(s.get('profitable_pairs') or []) or 'none'})"
    )
    for name, r in (result.get("per_regime") or {}).items():
        lines.append(
            f"  Regime {name}: {r['positive_windows']}/{r['windows']} windows positive, "
            f"mean OOS Sharpe {r['mean_oos_sharpe']}, worst {r['worst_oos_sharpe']} "
            f"across {r['distinct_pairs']} pair(s)"
    )
    lines.append(
        f"  Budget: {result.get('budget', {}).get('backtests_run', 0)} backtests across "
        f"{result.get('budget', {}).get('pairs_run', 0)} pair(s)"
    )
    return "\n".join(lines)
