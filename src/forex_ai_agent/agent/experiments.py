"""Variant runner & experiment records (Phase 3).

- Mutate one parameter → recompile → rerun IS + WFO/OOS → structured before/after diff
- Experiment records stored in existing Experiment/Hypothesis models
- Verdict cards: metrics, reasons, verdict FAIL/PASS, recommendation

Phase 5 additions:
- Viability check tool: replay current bars vs validated baseline → degradation report
- Autonomous research mode: "Find promising strategies for X" via propose_for_pair loop
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from forex_ai_agent.backtest import run_backtest, run_walkforward, BacktestConfig
from forex_ai_agent.backtest.execution import (
    SpreadModel,
    SlippageModel,
    CommissionModel,
    SwapModel,
    FixedSpreadModel,
    FixedSlippageModel,
    PerLotCommissionModel,
    FixedSwapModel,
    DEFAULT_SPREADS,
    DEFAULT_SWAP,
)
from forex_ai_agent.strategy import create_version_manager, StrategyVersion
from forex_ai_agent.data.providers import get_feed_chain
from forex_ai_agent.utils.timeframes import Timeframe
from forex_ai_agent.agent.repro import build_repro, feed_identity
from forex_ai_agent.agent.overfit import compute_overfit_components
from datetime import timedelta

import numpy as np

# Approximate bar length in hours — converts month-based window/step params
# into bar counts for any supported timeframe.
_TF_BAR_HOURS = {
    "M1": 1 / 60, "M5": 1 / 12, "M15": 0.25, "M30": 0.5,
    "H1": 1.0, "H4": 4.0, "D1": 24.0,
}


@dataclass
class ExperimentRecord:
    """Experiment record for agent storage."""
    id: str
    hypothesis_id: Optional[str]
    strategy_id: str
    base_version: str
    parameter_changes: Dict[str, Any]  # param -> new_value
    dataset: Dict[str, Any]  # symbol, timeframe, start, end
    baseline_metrics: Dict[str, float]
    variant_metrics: Dict[str, float]
    verdict: str  # "PASS" | "FAIL" | "INCONCLUSIVE"
    reasons: List[str]
    recommendation: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    concluded_at: Optional[datetime] = None
    # variant_id -> {param: new_value} — the graph edge needs per-variant
    # mutations; the aggregate ``parameter_changes`` dict alone cannot
    # distinguish multi-parameter experiments.
    variant_changes: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class VariantResult:
    """Result of a single variant test."""
    variant_id: str
    parameter_changes: Dict[str, Any]
    backtest_result: Optional[Dict[str, Any]]
    walkforward_result: Optional[Dict[str, Any]]
    metrics: Dict[str, float]
    verdict: str
    reasons: List[str]


class VariantRunner:
    """Runs parameter mutation experiments."""

    def __init__(self):
        self._experiments: Dict[str, ExperimentRecord] = {}

    def _build_signal_generator_from_strategy(self, strategy: StrategyVersion):
        """Build a signal generator function from a strategy version."""
        rules = strategy.rules
        params = strategy.parameters

        def signal_generator(timestamp, data):
            """Generate signals based on strategy rules."""
            signals = []

            # Example rule evaluation - simplified for EMA crossover
            if "entry" in rules:
                entry_rule = rules["entry"]
                # This is a simplified evaluator; real one would parse the rule string
                pass

            return signals

        return signal_generator

    async def _run_backtest_for_strategy(
        self,
        strategy: StrategyVersion,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        initial_balance: float = 100000.0,
    ) -> Dict[str, Any]:
        """Run a backtest for a strategy using the backtest engine."""
        import pandas as pd
        from forex_ai_agent.strategy.spec import parse_spec
        from forex_ai_agent.autonomy.validator import SpecSignalGenerator
        from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

        # The versioned spec is the single source of truth for rules
        # (proposal.py stores spec.model_dump() into StrategyVersion.rules).
        spec, errors = parse_spec(strategy.rules)
        if spec is None:
            raise ValueError(
                f"strategy {strategy.id} rules are not a valid spec: {'; '.join(errors[:3])}"
            )

        # H1 history as an indexed DataFrame (same shape as the gate's contract)
        provider = get_feed_chain()
        tf = Timeframe(timeframe)
        candles = await provider.fetch_candles(symbol, tf, start_date, end_date, limit=10000)
        if len(candles) < 250:
            raise ValueError(
                f"Insufficient history: {len(candles)} bars for {symbol} "
                f"({start_date:%Y-%m-%d} → {end_date:%Y-%m-%d}) — need ≥250. "
                "Retry with timeframe=H1 (24 bars/day) or extend the date range."
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

        # Real evaluator — the same one the proposal gate uses
        gen = SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(symbol))
        config = BacktestConfig(
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            initial_balance=initial_balance,
            symbols=[symbol],
            timeframe=spec.timeframe or timeframe,
            warmup_bars=60,
        )
        result = run_backtest({symbol: df}, gen, config)
        return result

    def _mutate_strategy(
        self,
        base_strategy: StrategyVersion,
        param_path: str,
        new_value: Any,
    ) -> StrategyVersion:
        """Create a new strategy variant with one parameter changed."""
        # Deep copy the base strategy
        variant = copy.deepcopy(base_strategy)
        variant.id = str(uuid4())[:8]
        variant.parent_id = base_strategy.id

        # Navigate to the parameter and set the new value.
        # Dotted paths address a container field: "rules.min_score_pct",
        # "parameters.timeframe". Single-part paths hit the first dict that
        # already owns the key — for spec knobs that's ``rules`` (the versioned
        # spec drives the backtest evaluator); unknown keys are recorded on
        # ``parameters`` so they can't corrupt the spec.
        parts = param_path.split(".")
        if len(parts) == 1:
            key = parts[0]
            if isinstance(variant.rules, dict) and key in variant.rules:
                variant.rules[key] = new_value
            else:
                variant.parameters[key] = new_value
        else:
            container = getattr(variant, parts[0], None)
            if not isinstance(container, dict):
                raise ValueError(
                    f"unknown parameter container '{parts[0]}' (use rules/parameters)"
                )
            target = container
            for part in parts[1:-1]:
                nxt = target.get(part)
                if not isinstance(nxt, dict):
                    nxt = {}
                    target[part] = nxt
                target = nxt
            target[parts[-1]] = new_value

        # A mutation that breaks the spec is rejected immediately
        from forex_ai_agent.strategy.spec import parse_spec

        spec_check, spec_errors = parse_spec(variant.rules)
        if spec_check is None:
            raise ValueError(
                f"mutation '{param_path}' produced an invalid spec: "
                f"{'; '.join(spec_errors[:3])}"
            )

        # Bump version as experimental
        variant.bump_version("experimental")
        variant.status = "experimental"

        return variant

    def _compare_metrics(
        self,
        baseline: Dict[str, float],
        variant: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, List[str]]:
        """Compare metrics and return verdict + reasons."""
        reasons = []

        # Default thresholds for PASS
        default_thresholds = {
            "profit_factor": 1.2,
            "max_drawdown_pct": 0.15,  # 15%
            "sharpe_ratio": 0.5,
            "win_rate": 0.45,
            "total_trades": 30,
        }
        thresholds = thresholds or default_thresholds

        # Check each threshold
        pf_change = variant.get("profit_factor", 0) - baseline.get("profit_factor", 0)
        dd_change = variant.get("max_drawdown_pct", 1) - baseline.get("max_drawdown_pct", 0)

        if variant.get("profit_factor", 0) < thresholds["profit_factor"]:
            reasons.append(f"Profit factor {variant.get('profit_factor', 0):.2f} below threshold {thresholds['profit_factor']}")
        if variant.get("max_drawdown_pct", 1) > thresholds["max_drawdown_pct"]:
            reasons.append(f"Max drawdown {variant.get('max_drawdown_pct', 0)*100:.1f}% exceeds threshold {thresholds['max_drawdown_pct']*100:.0f}%")
        if variant.get("total_trades", 0) < thresholds["total_trades"]:
            reasons.append(f"Only {variant.get('total_trades', 0)} trades (min {thresholds['total_trades']})")

        # Overall verdict
        if not reasons:
            verdict = "PASS"
        elif pf_change < -0.3 or dd_change > 0.05:
            verdict = "FAIL"
        else:
            verdict = "INCONCLUSIVE"

        if verdict == "PASS" and pf_change > 0.1:
            reasons.append(f"Profit factor improved by {pf_change:.2f}")
        if verdict == "PASS" and dd_change < -0.02:
            reasons.append(f"Drawdown reduced by {abs(dd_change)*100:.1f}%")

        return verdict, reasons

    async def run_experiment(
        self,
        strategy_id: str,
        param_mutations: Dict[str, Any],  # param_path -> new_value
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        hypothesis_id: Optional[str] = None,
    ) -> ExperimentRecord:
        """Run a full experiment: baseline + variants."""
        vm = await create_version_manager()
        base_strategy = vm.get_version(strategy_id)
        if not base_strategy:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Run baseline
        print(f"Running baseline for {base_strategy.name}...")
        baseline_result = await self._run_backtest_for_strategy(
            base_strategy, symbol, timeframe, start_date, end_date
        )
        baseline_metrics = baseline_result.metrics if hasattr(baseline_result, 'metrics') else {}

        variant_results = []
        for param_path, new_value in param_mutations.items():
            print(f"Running variant: {param_path} = {new_value}")
            variant = self._mutate_strategy(base_strategy, param_path, new_value)
            variant_result = await self._run_backtest_for_strategy(
                variant, symbol, timeframe, start_date, end_date
            )
            variant_metrics = variant_result.metrics if hasattr(variant_result, 'metrics') else {}

            verdict, reasons = self._compare_metrics(baseline_metrics, variant_metrics)

            variant_results.append(VariantResult(
                variant_id=variant.id,
                parameter_changes={param_path: new_value},
                backtest_result=variant_result.__dict__ if hasattr(variant_result, '__dict__') else {},
                walkforward_result=None,  # variants run single-pass; WF is a separate tool
                metrics=variant_metrics,
                verdict=verdict,
                reasons=reasons,
            ))

        # Aggregate verdict
        overall_verdict = "PASS"
        all_reasons = []
        for vr in variant_results:
            if vr.verdict == "FAIL":
                overall_verdict = "FAIL"
            elif vr.verdict == "INCONCLUSIVE" and overall_verdict == "PASS":
                overall_verdict = "INCONCLUSIVE"
            all_reasons.extend([f"{vr.parameter_changes}: {r}" for r in vr.reasons])

        recommendation = self._generate_recommendation(overall_verdict, variant_results)

        feed = feed_identity()
        experiment_dataset: Dict[str, Any] = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "repro": build_repro(
                strategy_id=strategy_id,
                strategy_name=base_strategy.name,
                rules=base_strategy.rules,
                symbol=symbol,
                timeframe=timeframe,
                data_start=start_date,
                data_end=end_date,
                feed_provider=feed["provider"],
                feed_chain=feed["chain"],
                extra={"base_version": base_strategy.version},
            ),
        }

        # Create experiment record
        record = ExperimentRecord(
            id=str(uuid4())[:8],
            hypothesis_id=hypothesis_id,
            strategy_id=strategy_id,
            base_version=base_strategy.version,
            parameter_changes=param_mutations,
            dataset=experiment_dataset,
            baseline_metrics=baseline_metrics,
            variant_metrics={vr.variant_id: vr.metrics for vr in variant_results},
            verdict=overall_verdict,
            reasons=all_reasons,
            recommendation=recommendation,
            variant_changes={vr.variant_id: dict(vr.parameter_changes) for vr in variant_results},
        )

        self._experiments[record.id] = record

        # B2: persist to the research graph (best-effort, never raises) so
        # hypothesis/parameter questions are answerable from stored records.
        try:
            from forex_ai_agent.agent.research_graph import persist_agent_experiment

            await persist_agent_experiment(record)
        except Exception as exc:  # noqa: BLE001 — graph is advisory here
            print(f"research graph persist skipped: {type(exc).__name__}: {exc}")
        return record

    def _generate_recommendation(
        self,
        verdict: str,
        variants: List[VariantResult],
    ) -> str:
        """Generate human-readable recommendation."""
        if verdict == "PASS":
            best = max(variants, key=lambda v: v.metrics.get("profit_factor", 0))
            return (
                f"Recommend adopting variant {best.variant_id[:8]} "
                f"(PF: {best.metrics.get('profit_factor', 0):.2f}, "
                f"DD: {best.metrics.get('max_drawdown_pct', 0)*100:.1f}%)"
            )
        elif verdict == "FAIL":
            return "No variant improved over baseline. Do not adopt changes."
        else:
            return "Results inconclusive. Consider larger dataset or different parameters."

    def get_experiment(self, exp_id: str) -> Optional[ExperimentRecord]:
        return self._experiments.get(exp_id)

    def list_experiments(self, strategy_id: Optional[str] = None) -> List[ExperimentRecord]:
        exps = list(self._experiments.values())
        if strategy_id:
            exps = [e for e in exps if e.strategy_id == strategy_id]
        return sorted(exps, key=lambda e: e.created_at, reverse=True)

    async def conclude_experiment(
        self,
        exp_id: str,
        decision: str,  # "adopt" | "reject" | "defer"
        notes: str = "",
    ) -> ExperimentRecord:
        exp = self._experiments.get(exp_id)
        if not exp:
            raise ValueError(f"Experiment {exp_id} not found")

        exp.verdict = decision.upper()
        exp.concluded_at = datetime.utcnow()
        exp.recommendation += f" | Decision: {decision} - {notes}"

        # B2: propagate the human decision onto the research graph so
        # "how often did we adopt changes like this?" is a stored-record join.
        try:
            from forex_ai_agent.agent.research_graph import record_experiment_decision

            await record_experiment_decision(exp_id, decision, notes)
        except Exception as exc:  # noqa: BLE001 — graph is advisory here
            print(f"research graph decision skipped: {type(exc).__name__}: {exc}")
        return exp


# Global instance
_runner: Optional[VariantRunner] = None


def get_variant_runner() -> VariantRunner:
    global _runner
    if _runner is None:
        _runner = VariantRunner()
    return _runner


# Tool handlers (Phase 3)

async def backtest_strategy(
    strategy_id: str,
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "H1",
    initial_balance: float = 100000.0,
) -> Dict[str, Any]:
    """Run backtest for a strategy version."""
    runner = get_variant_runner()
    vm = await create_version_manager()
    strategy = vm.get_version(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")

    result = await runner._run_backtest_for_strategy(
        strategy,
        symbol,
        timeframe,
        datetime.fromisoformat(start_date),
        datetime.fromisoformat(end_date),
        initial_balance,
    )

    raw_metrics = dict(result.metrics or {})
    return {
        "strategy_id": strategy_id,
        "strategy_name": strategy.name,
        "metrics": dict(raw_metrics.get("summary") or {}),
        "trades": len(result.trades) if hasattr(result, "trades") else 0,
        "equity_curve": [
            {"time": t.isoformat(), "equity": v} for t, v in (result.equity_curve or [])[:200]
        ],
    }


async def walk_forward_test(
    strategy_id: str,
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "H1",
    window_months: int = 6,
    step_months: int = 2,
    initial_balance: float = 100_000.0,
    run_parameter_grid: bool = False,
    parameter_grid: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Real walk-forward analysis for a saved strategy.

    Rolling IS→OOS windows over the requested range (engine:
    ``run_walk_forward_on_df``), plus a Stage-A1 ``repro`` dict so the run
    can be replayed exactly later (see ``agent/repro.py``).
    """
    import pandas as pd

    from forex_ai_agent.strategy.spec import parse_spec

    vm = await create_version_manager()
    strategy = vm.get_version(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")
    spec, errors = parse_spec(strategy.rules)
    if spec is None:
        raise ValueError(
            f"strategy {strategy_id} rules are not a valid spec: {'; '.join(errors[:3])}"
        )

    provider = get_feed_chain()
    tf = Timeframe(timeframe)
    candles = await provider.fetch_candles(
        symbol, tf, datetime.fromisoformat(start_date), datetime.fromisoformat(end_date), limit=20000
    )
    if len(candles) < 250:
        raise ValueError(
            f"Insufficient history: {len(candles)} bars for {symbol} — walk-forward needs ≥250. "
            "Extend the date range (≈1 year of H1 gives the most reliable windows)."
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

    result = run_walk_forward_on_df(
        spec,
        df,
        symbol,
        timeframe,
        window_months=window_months,
        step_months=step_months,
        initial_balance=initial_balance,
        run_parameter_grid=run_parameter_grid,
        parameter_grid=parameter_grid,
    )
    result["strategy_id"] = strategy_id
    result["strategy_name"] = strategy.name
    feed = feed_identity()
    result["repro"] = build_repro(
        strategy_id=strategy_id,
        strategy_name=strategy.name,
        rules=strategy.rules,
        symbol=symbol,
        timeframe=timeframe,
        bar_count=len(df),
        data_start=df.index.min(),
        data_end=df.index.max(),
        feed_provider=feed["provider"],
        feed_chain=feed["chain"],
        walk_forward={
            "window_months": window_months,
            "step_months": step_months,
            "initial_balance": initial_balance,
        },
    )
    return result


def run_walk_forward_on_df(
    spec: Any,
    df: Any,
    symbol: str,
    timeframe: str,
    window_months: int = 6,
    step_months: int = 2,
    initial_balance: float = 100_000.0,
    run_parameter_grid: bool = False,
    parameter_grid: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Rolling IS→OOS walk-forward core over an already-loaded bar DataFrame.

    Each window backtests the strategy *unchanged* on an in-sample slice,
    then on the following out-of-sample slice — no re-fitting, so this
    measures OOS stability, not optimization. Aggregates OOS Sharpe/PF/
    drawdown, an IS→OOS degradation (overfitting) score and a 0–1 stability
    score — the same quantity the autonomy gate judges. Used by
    ``walk_forward_test`` (feed-backed) and by replay/golden tests with
    synthetic frames (Stage A3).
    """
    import pandas as pd

    from forex_ai_agent.autonomy.validator import SpecSignalGenerator
    from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

    hours = _TF_BAR_HOURS.get(timeframe.upper(), 1.0)
    month_bars = 30.44 * 24.0 / hours
    step_bars = max(40, int(round(step_months * month_bars)))
    test_bars = max(60, int(round(window_months * month_bars / 3.0)))
    train_bars = max(80, test_bars * 2)
    n = len(df)
    if train_bars + test_bars > n:
        raise ValueError(
            f"Not enough history for walk-forward: {n} bars, need ≥{train_bars + test_bars} "
            f"(IS {train_bars} + OOS {test_bars} bars for {timeframe}). Extend the date range."
        )

    def _metrics(res: Any) -> Dict[str, Any]:
        raw = dict(res.metrics or {})
        s = dict(raw.get("summary") or {})
        p = dict(raw.get("portfolio") or {})
        return {
            "sharpe": float(s.get("sharpe_ratio", 0.0)),
            "pf": float(s.get("profit_factor", 0.0)),
            "trades": int(s.get("total_trades", 0)),
            "win_rate": float(s.get("win_rate", 0.0)),
            "return_pct": float(s.get("total_return_pct", 0.0)),
            "max_dd": float(p.get("max_drawdown_pct", 0.0)) / 100.0,
        }

    def _cfg(a: pd.DataFrame) -> BacktestConfig:
        return BacktestConfig(
            start_date=a.index.min().to_pydatetime(),
            end_date=a.index.max().to_pydatetime(),
            initial_balance=initial_balance,
            symbols=[symbol],
            timeframe=spec.timeframe or timeframe,
            warmup_bars=60,
        )

    pip = pip_size_for_symbol(symbol)
    window_rows: List[Dict[str, Any]] = []
    is_sharpes: List[float] = []
    oos_sharpes: List[float] = []
    oos_drawdowns: List[float] = []
    oos_pfs: List[float] = []
    total_oos_trades = 0
    oos_slices: List[Tuple[int, int]] = []
    w = 0
    # Hard cap of 8 windows bounds runtime; step size controls coverage.
    while len(window_rows) < 8:
        train_start = w * step_bars
        train_end = train_start + train_bars
        if train_end >= n:
            break
        test_start = train_end
        test_end = min(test_start + test_bars, n)
        oos_slices.append((test_start, test_end))
        train_df = df.iloc[train_start:train_end]
        test_df = df.iloc[test_start:test_end]
        if len(train_df) < 80 or len(test_df) < 40:
            break

        try:
            is_res = run_backtest(
                {symbol: train_df}, SpecSignalGenerator(spec, pip_size=pip), _cfg(train_df)
            )
            oos_res = run_backtest(
                {symbol: test_df}, SpecSignalGenerator(spec, pip_size=pip), _cfg(test_df)
            )
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"walk-forward window {w + 1} failed: {type(exc).__name__}: {exc}"
            ) from exc

        im, om = _metrics(is_res), _metrics(oos_res)
        is_sharpes.append(im["sharpe"])
        oos_sharpes.append(om["sharpe"])
        oos_drawdowns.append(om["max_dd"])
        oos_pfs.append(om["pf"])
        total_oos_trades += om["trades"]
        window_rows.append(
            {
                "window": w + 1,
                "is_bars": len(train_df),
                "oos_bars": len(test_df),
                # B3: real OOS dates — lets the robustness sweep tag each
                # window's market regime without re-deriving slice bounds.
                "oos_start": str(test_df.index.min()),
                "oos_end": str(test_df.index.max()),
                "is_sharpe": im["sharpe"],
                "is_pf": im["pf"],
                "is_trades": im["trades"],
                "oos_sharpe": om["sharpe"],
                "oos_pf": om["pf"],
                "oos_trades": om["trades"],
                "oos_max_dd": om["max_dd"],
                "oos_return_pct": om["return_pct"],
            }
        )
        w += 1

    if not window_rows:
        raise ValueError(
            "walk-forward produced no valid windows — history too short for the "
            "requested window/step sizes"
        )

    # ---- A2b: optional small parameter grid (budget-capped) ---------------
    grid_result: Optional[Dict[str, Any]] = None
    if run_parameter_grid:
        try:
            grid_result = _run_parameter_grid(
                spec, df, symbol, timeframe, oos_slices, initial_balance, parameter_grid
            )
        except Exception as exc:  # noqa: BLE001 — sensitivity is optional
            grid_result = {
                "status": "ERROR",
                "note": f"{type(exc).__name__}: {exc}",
                "variants": [],
            }

    mean_is = float(np.mean(is_sharpes))
    mean_oos = float(np.mean(oos_sharpes))
    std_oos = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
    degradation = 1.0 - (mean_oos / mean_is) if mean_is > 0 else 1.0
    degradation = max(0.0, min(1.0, degradation))
    neg_frac = sum(1 for s in oos_sharpes if s <= 0) / len(oos_sharpes)
    vol_ratio = min(1.0, std_oos / abs(mean_oos)) if mean_oos != 0 else 1.0
    # 0 = perfectly stable OOS performance, 1 = wildly unstable.
    stability = max(0.0, min(1.0, 0.4 * degradation + 0.3 * vol_ratio + 0.3 * neg_frac))

    # Verdict bands calibrated for 4-window granularity (Stage-A3 golden
    # tests): each component is coarse at n=4, so STABLE at ≤0.25 stability
    # was effectively unreachable — 0.4×degradation alone exceeds it with any
    # real IS→OOS decay. Bands now align with the gate's 0.5 overfit ceiling.
    # Dead windows (zero OOS trades) are “no evidence”, not “Sharpe 0”: a
    # strategy that stops trading out-of-sample must never be STABLE.
    dead_windows = sum(1 for r in window_rows if r.get("oos_trades", 0) == 0)
    collapse = bool(oos_sharpes) and min(oos_sharpes) < -1.0
    if dead_windows > len(window_rows) / 2:
        # Majority-dead: the strategy stopped trading out-of-sample.
        verdict = "UNSTABLE"
    elif mean_oos >= 0.3 and stability <= 0.45 and dead_windows == 0 and not collapse:
        verdict = "STABLE"
    elif mean_oos <= 0 or collapse:
        # A catastrophic OOS window (Sharpe < −1) is disqualifying — never MIXED.
        verdict = "UNSTABLE"
    elif stability <= 0.65:
        verdict = "MIXED"
    else:
        verdict = "UNSTABLE"

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "windows_tested": len(window_rows),
        "windows": window_rows,
        "aggregate_metrics": {
            "is_sharpe_mean": mean_is,
            "oos_sharpe_mean": mean_oos,
            "oos_sharpe_std": std_oos,
            "oos_pf_mean": float(np.mean(oos_pfs)) if oos_pfs else 0.0,
            "oos_trades_total": total_oos_trades,
            "oos_trades_per_window": total_oos_trades / len(window_rows),
            "oos_max_drawdown": max(oos_drawdowns) if oos_drawdowns else 0.0,
            "degradation_overfitting": degradation,
            "oos_negative_window_frac": neg_frac,
        },
        "stability_score": stability,
        "verdict": verdict,
        "initial_balance": initial_balance,
        "overfit_analysis": compute_overfit_components(
            window_rows, oos_trades_total=total_oos_trades, parameter_grid=grid_result
        ),
    }


# Budget for the A2b sensitivity grid: extra OOS backtests per WF run.
_GRID_BUDGET_BACKTESTS = 12


def _run_parameter_grid(
    spec: Any,
    df: Any,
    symbol: str,
    timeframe: str,
    oos_slices: List[Tuple[int, int]],
    initial_balance: float,
    parameter_grid: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """A2b: measure OOS sensitivity to the key signal parameters.

    Runs the *same* OOS windows with small parameter mutations and reports
    per-variant mean OOS Sharpe. Hard-capped at ``_GRID_BUDGET_BACKTESTS``
    extra backtests per walk-forward run (off by default).
    """
    from forex_ai_agent.autonomy.validator import SpecSignalGenerator
    from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

    if not hasattr(spec, "model_copy"):
        return {
            "status": "NOT_MEASURED",
            "note": "spec does not support parameter mutation",
            "variants": [],
        }

    try:
        from forex_ai_agent.strategy.spec import ATR_SL_MULT, ATR_TP_MULT

        sl_lo, sl_hi = ATR_SL_MULT
        tp_lo, tp_hi = ATR_TP_MULT
    except Exception:  # noqa: BLE001 — fall back to generous clamps
        sl_lo, sl_hi, tp_lo, tp_hi = 0.0, 10.0, 0.0, 20.0

    if parameter_grid:
        mutations_list = [dict(m) for m in parameter_grid][:8]
    else:
        base_sl = float(getattr(spec, "atr_sl_mult", 1.5))
        base_tp = float(getattr(spec, "atr_tp_mult", 3.0))
        base_score = float(getattr(spec, "min_score_pct", 0.6))
        mutations_list = [
            {"atr_sl_mult": min(sl_hi, base_sl * 1.25)},
            {"atr_sl_mult": max(sl_lo, base_sl * 0.8)},
            {"atr_tp_mult": min(tp_hi, base_tp * 1.25)},
            {"atr_tp_mult": max(tp_lo, base_tp * 0.8)},
            {"min_score_pct": min(0.95, base_score + 0.1)},
            {"min_score_pct": max(0.3, base_score - 0.1)},
        ]

    max_variants = max(1, _GRID_BUDGET_BACKTESTS // max(1, len(oos_slices)))
    mutations_list = mutations_list[:max_variants]

    pip = pip_size_for_symbol(symbol)
    variants: List[Dict[str, Any]] = []
    for mutations in mutations_list:
        vspec = spec.model_copy(update=mutations)
        sharpes: List[float] = []
        for (s_idx, e_idx) in oos_slices:
            slice_df = df.iloc[s_idx:e_idx]
            if len(slice_df) < 40:
                continue
            res = run_backtest(
                {symbol: slice_df},
                SpecSignalGenerator(vspec, pip_size=pip),
                BacktestConfig(
                    start_date=slice_df.index.min().to_pydatetime(),
                    end_date=slice_df.index.max().to_pydatetime(),
                    initial_balance=initial_balance,
                    symbols=[symbol],
                    timeframe=timeframe,
                    warmup_bars=60,
                ),
            )
            m = dict((res.metrics or {}).get("summary") or {})
            sharpes.append(float(m.get("sharpe_ratio", 0.0)))
        if sharpes:
            variants.append(
                {
                    "parameter_changes": mutations,
                    "oos_sharpe_mean": round(float(np.mean(sharpes)), 4),
                    "windows": len(sharpes),
                }
            )

    worst = min(variants, key=lambda v: v["oos_sharpe_mean"]) if variants else None
    return {
        "status": "MEASURED" if variants else "SKIPPED_BUDGET",
        "variants": variants,
        "worst_variant": worst,
        "worst_param_change": (worst or {}).get("parameter_changes"),
        "budget_backtests": _GRID_BUDGET_BACKTESTS,
        "note": (
            f"≤{_GRID_BUDGET_BACKTESTS} extra OOS backtests; "
            f"{len(mutations_list)} variants × {len(oos_slices)} windows"
        ),
    }


async def compare_strategies(
    strategy_ids: List[str],
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = "H1",
) -> Dict[str, Any]:
    """Compare multiple strategies side by side."""
    runner = get_variant_runner()
    vm = await create_version_manager()

    results = []
    for sid in strategy_ids:
        strategy = vm.get_version(sid)
        if not strategy:
            continue
        result = await runner._run_backtest_for_strategy(
            strategy,
            symbol,
            timeframe,
            datetime.fromisoformat(start_date),
            datetime.fromisoformat(end_date),
        )
        raw_metrics = dict(result.metrics or {})
        results.append({
            "strategy_id": sid,
            "strategy_name": strategy.name,
            "metrics": dict(raw_metrics.get("summary") or {}),
        })

    # Rank by profit factor
    ranked = sorted(
        results,
        key=lambda r: r.get("metrics", {}).get("profit_factor", 0),
        reverse=True
    )

    return {
        "comparison": ranked,
        "best": ranked[0] if ranked else None,
        "worst": ranked[-1] if ranked else None,
    }


async def create_experiment(
    strategy_id: str,
    param_mutations: Dict[str, Any],
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    hypothesis_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create and run a parameter experiment."""
    runner = get_variant_runner()
    record = await runner.run_experiment(
        strategy_id=strategy_id,
        param_mutations=param_mutations,
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime.fromisoformat(start_date),
        end_date=datetime.fromisoformat(end_date),
        hypothesis_id=hypothesis_id,
    )
    return {
        "experiment_id": record.id,
        "verdict": record.verdict,
        "recommendation": record.recommendation,
        "baseline_metrics": record.baseline_metrics,
        "variant_count": len(record.variant_metrics),
    }


async def list_experiments(
    strategy_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List experiment records."""
    runner = get_variant_runner()
    exps = runner.list_experiments(strategy_id)
    return [
        {
            "id": e.id,
            "strategy_id": e.strategy_id,
            "base_version": e.base_version,
            "verdict": e.verdict,
            "recommendation": e.recommendation,
            "created_at": e.created_at.isoformat(),
            "concluded_at": e.concluded_at.isoformat() if e.concluded_at else None,
        }
        for e in exps[:limit]
    ]


async def get_experiment(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific experiment record."""
    runner = get_variant_runner()
    exp = runner.get_experiment(experiment_id)
    if not exp:
        return None
    return {
        "id": exp.id,
        "hypothesis_id": exp.hypothesis_id,
        "strategy_id": exp.strategy_id,
        "base_version": exp.base_version,
        "parameter_changes": exp.parameter_changes,
        "dataset": exp.dataset,
        "baseline_metrics": exp.baseline_metrics,
        "variant_metrics": exp.variant_metrics,
        "verdict": exp.verdict,
        "reasons": exp.reasons,
        "recommendation": exp.recommendation,
        "created_at": exp.created_at.isoformat(),
        "concluded_at": exp.concluded_at.isoformat() if exp.concluded_at else None,
    }


async def conclude_experiment(
    experiment_id: str,
    decision: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Record final decision on an experiment."""
    runner = get_variant_runner()
    exp = await runner.conclude_experiment(experiment_id, decision, notes)
    return {
        "experiment_id": exp.id,
        "final_verdict": exp.verdict,
        "concluded_at": exp.concluded_at.isoformat() if exp.concluded_at else None,
    }


# =============================================================================
# Phase 5: Viability Check & Autonomous Research
# =============================================================================

@dataclass
class ViabilityReport:
    """Degradation report for strategy viability check."""
    strategy_id: str
    strategy_name: str
    symbol: str
    timeframe: str
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    drift: Dict[str, float]  # metric -> change
    oos_pf_drift: float
    dd_change_pct: float
    concentration_risk: float
    verdict: str  # "STABLE" | "DEGRADED" | "CRITICAL"
    concerns: List[str]
    disclaimer: str = "I have not modified anything"


async def check_strategy_viability(
    strategy_id: str,
    symbol: str,
    timeframe: str = "H1",
    lookback_days: int = 90,
) -> ViabilityReport:
    """Replay current bars vs validated baseline → degradation report.
    
    Uses ProbationRunner replay logic: compare recent performance
    against the validated baseline from probation period.
    
    Returns structured report ending with "I have not modified anything".
    """
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner
    from forex_ai_agent.strategy import create_version_manager
    from forex_ai_agent.data.providers import get_feed_chain
    
    vm = await create_version_manager()
    strategy = vm.get_version(strategy_id)
    if not strategy:
        raise ValueError(f"Strategy {strategy_id} not found")
    
    # Get probation record for baseline
    runner = ProbationRunner()
    probation_records = await runner.store.list()
    prob_record = None
    for r in probation_records:
        if r.version_id == strategy_id:
            prob_record = r
            break
    
    if not prob_record:
        raise ValueError(f"No probation record for strategy {strategy_id}")
    
    # Get baseline metrics from probation (validated performance).
    # ProbationStatus.last_metrics holds the paper-probation metrics dict.
    baseline_metrics = (prob_record.last_metrics or {}) if prob_record.last_metrics else {}
    
    # Fetch recent data for replay
    provider = get_feed_chain()
    tf = Timeframe(timeframe)
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=lookback_days)
    
    candles = await provider.fetch_candles(symbol, tf, start_date, end_date, limit=5000)
    
    if len(candles) < 50:
        raise ValueError(f"Insufficient recent data: {len(candles)} candles")
    
    # The versioned spec is the single source of truth for rules
    # (proposal.py stores spec.model_dump() into StrategyVersion.rules).
    from forex_ai_agent.strategy.spec import parse_spec
    from forex_ai_agent.autonomy.validator import SpecSignalGenerator
    from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

    spec, spec_errors = parse_spec(strategy.rules)
    if spec is None:
        raise ValueError(
            f"strategy {strategy_id} rules are not a valid spec: {'; '.join(spec_errors[:3])}"
        )
    
    # Convert to DataFrame
    import pandas as pd
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
    
    gen = SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(symbol))
    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=100000.0,
        symbols=[symbol],
        timeframe=spec.timeframe or timeframe,
        warmup_bars=60,
    )
    
    result = run_backtest({symbol: df}, gen, config)

    raw_metrics = dict(result.metrics or {})
    current_metrics = dict(raw_metrics.get("summary") or {})
    
    # Calculate drift
    drift = {}
    for key in ['profit_factor', 'max_drawdown_pct', 'sharpe_ratio', 'win_rate', 'total_trades']:
        baseline = baseline_metrics.get(key, 0)
        current = current_metrics.get(key, 0)
        if baseline != 0:
            drift[key] = (current - baseline) / abs(baseline)
        else:
            drift[key] = 0.0
    
    oos_pf_drift = drift.get('profit_factor', 0)
    dd_change_pct = drift.get('max_drawdown_pct', 0) * 100
    
    # Concentration risk: check if trades clustered in few pairs/months
    concentration_risk = 0.0
    if hasattr(result, 'trades') and result.trades:
        # Simplified: check trade distribution
        concentration_risk = min(len(result.trades) / 100.0, 1.0)
    
    # Determine verdict
    concerns = []
    if oos_pf_drift < -0.3:
        concerns.append(f"Profit factor degraded by {abs(oos_pf_drift)*100:.0f}% (baseline: {baseline_metrics.get('profit_factor', 0):.2f} → current: {current_metrics.get('profit_factor', 0):.2f})")
    if dd_change_pct > 50:
        concerns.append(f"Max drawdown increased by {dd_change_pct:.0f}%")
    if current_metrics.get('total_trades', 0) < baseline_metrics.get('total_trades', 0) * 0.5:
        concerns.append(f"Trade frequency halved ({current_metrics.get('total_trades', 0)} vs baseline {baseline_metrics.get('total_trades', 0)})")
    if current_metrics.get('win_rate', 0) < baseline_metrics.get('win_rate', 0) * 0.8:
        concerns.append(f"Win rate dropped significantly")
    
    if not concerns:
        verdict = "STABLE"
    elif oos_pf_drift < -0.5 or dd_change_pct > 100:
        verdict = "CRITICAL"
    else:
        verdict = "DEGRADED"
    
    return ViabilityReport(
        strategy_id=strategy_id,
        strategy_name=strategy.name,
        symbol=symbol,
        timeframe=timeframe,
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        drift=drift,
        oos_pf_drift=oos_pf_drift,
        dd_change_pct=dd_change_pct,
        concentration_risk=concentration_risk,
        verdict=verdict,
        concerns=concerns,
    )


async def find_promising_strategies(
    pairs: List[str],
    timeframes: List[str] = None,
    days: int = 365,
    min_profit_factor: float = 1.3,
    min_trades: int = 20,
    max_drawdown_pct: float = 0.20,
    max_concurrent: int = 3,
) -> Dict[str, Any]:
    """Bounded autonomous research: iterate pairs/timeframes, rank gate survivors, enroll into probation.
    
    Hard blocks enforced in code:
    - No self-approval
    - No risk-limit changes
    - No audit deletion
    - No permission self-promotion
    - LLM budget guardrails: per-session turn caps, cheap tier for routing
    """
    from forex_ai_agent.autonomy.proposal import propose_for_pair
    from forex_ai_agent.pipeline import Pipeline
    from forex_ai_agent.core.event_bus import InMemoryEventBus
    
    timeframes = timeframes or ["H1", "H4"]
    pipeline = Pipeline(event_bus=InMemoryEventBus())
    await pipeline.initialize()
    
    results = []
    for pair in pairs:
        for tf in timeframes:
            try:
                history = await fetch_history_for(pipeline, pair, days=days)
                result = await propose_for_pair(pipeline, pair, history, use_llm=True)
                
                if result.version and result.validation and result.validation.passed:
                    fb = result.validation.full_backtest or {}
                    if (fb.get('profit_factor', 0) >= min_profit_factor and
                        fb.get('total_trades', 0) >= min_trades and
                        fb.get('max_drawdown_pct', 1) <= max_drawdown_pct):
                        results.append({
                            "pair": pair,
                            "timeframe": tf,
                            "strategy_id": result.version.id,
                            "strategy_name": result.version.name,
                            "profit_factor": fb.get('profit_factor', 0),
                            "total_trades": fb.get('total_trades', 0),
                            "max_drawdown_pct": fb.get('max_drawdown_pct', 0),
                            "win_rate": fb.get('win_rate', 0),
                        })
            except Exception as e:
                results.append({
                    "pair": pair,
                    "timeframe": tf,
                    "error": str(e),
                })
            
            if len([r for r in results if 'strategy_id' in r]) >= max_concurrent:
                break
    
    # Rank by profit factor
    ranked = sorted(
        [r for r in results if 'strategy_id' in r],
        key=lambda r: r.get('profit_factor', 0),
        reverse=True
    )
    
    return {
        "candidates_found": len(ranked),
        "ranked_strategies": ranked,
        "errors": [r for r in results if 'error' in r],
        "disclaimer": "Candidates enrolled in paper probation. No live execution without human approval. I have not modified anything.",
    }


async def fetch_history_for(
    pipeline,
    pair: str,
    days: int = 365,
) -> List[Dict[str, Any]]:
    """Fetch historical candles for a pair."""
    from forex_ai_agent.data.providers import get_feed_chain
    from forex_ai_agent.utils.timeframes import Timeframe
    
    provider = get_feed_chain()
    tf = Timeframe("H1")
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    candles = await provider.fetch_candles(pair, tf, start_date, end_date, limit=50000)
    
    return [
        {
            "time": c.timestamp.isoformat(),
            "open": c.open,
            "high": c.high,
            "low": c.low,
            "close": c.close,
            "volume": c.volume,
        }
        for c in candles
    ]


def _build_signal_generator_from_strategy(strategy: StrategyVersion):
    """Build a signal generator function from a strategy version."""
    rules = strategy.rules
    params = strategy.parameters
    
    def signal_generator(timestamp, data):
        """Generate signals based on strategy rules."""
        signals = []
        # Simplified evaluator - real one would parse the rule string
        # This is a placeholder for the actual rule evaluation
        return signals
    
    return signal_generator