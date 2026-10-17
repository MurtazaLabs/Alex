"""BacktestValidator: the deterministic gate between proposal and probation.

Pipeline (all deterministic — the LLM touches nothing here):

    spec → parse/validate → backtest (full history)
         → walk-forward with parameter grid (OOS windows)
         → gate on OOS sharpe, drawdown, trade count, overfitting

A strategy that fails any gate is rejected with explicit reasons; nothing
proceeds to paper probation without surviving real historical friction
(spread, slippage, commission via BacktestConfig defaults).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..backtest.walkforward import WalkForwardAnalyzer, WalkForwardConfig
from ..strategy.spec import StrategySpec, parse_spec
from ..utils.math_helpers import pip_size_for_symbol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backtest adapter: evaluates a spec statefully over historical bars
# ---------------------------------------------------------------------------


class SpecSignalGenerator:
    """SignalGenerator over historical bars, driven purely by a StrategySpec.

    Stateful verbs (crosses/rising/falling) compare the current indicator
    values with those of the previous bar. SL/TP are deterministic ATR
    brackets from the spec (never LLM numbers — the spec only carries the
    multipliers, which the schema range-limits).
    """

    def __init__(
        self,
        spec: StrategySpec,
        pip_size: float = 0.0001,
        cooldown_bars: int = 12,
        risk_per_trade_pct: float = 0.01,
        account_balance: float = 100_000.0,
    ) -> None:
        self.spec = spec
        self.pip_size = pip_size
        self.cooldown_bars = max(1, cooldown_bars)
        self.risk_per_trade_pct = risk_per_trade_pct
        self.account_balance = account_balance
        self._prev: Dict[str, Dict[str, float]] = {}
        self._bars_since_signal: Dict[str, int] = {}

    def _values(self, indicators: Dict[str, float], close: float) -> Dict[str, float]:
        return {
            "price": close,
            "ema": indicators.get("ema_20"),
            "sma": indicators.get("sma_20"),
            "sma_slow": indicators.get("sma_50"),
            "rsi": indicators.get("rsi_14"),
            "atr": indicators.get("atr_14"),
            "bb_upper": indicators.get("bb_upper"),
            "bb_middle": indicators.get("bb_middle"),
            "bb_lower": indicators.get("bb_lower"),
        }

    def _eval_condition(
        self, left: float, verb: str, right: float, prev: Dict[str, float]
    ) -> bool:
        if verb == "above":
            return left > right
        if verb == "below":
            return left < right
        prev_left = prev.get("_left")
        prev_right = prev.get("_right")
        if prev_left is None or prev_right is None:
            return False  # first bar: no previous state → no crosses
        if verb == "crosses_above":
            return prev_left <= prev_right and left > right
        if verb == "crosses_below":
            return prev_left >= prev_right and left < right
        if verb == "rising":
            return left > prev_left
        if verb == "falling":
            return left < prev_left
        return False

    def generate_signals(
        self,
        timestamp: datetime,
        data: Dict[str, Any],
        indicators: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        signals: List[Dict] = []
        for symbol, bar in data.items():
            if bar is None:
                continue
            vals = self._values(indicators.get(symbol, {}), float(bar.close))
            # Only require the values this spec actually reads — an unrelated
            # indicator missing from a partial snapshot must not skip the bar
            # (mirrors compile(): a missing operand fails the condition, not
            # the whole bar).
            required = {"price", "atr"}
            for c in self.spec.entry_conditions:
                required.add(c.left)
                if not _is_number(c.right):
                    rk = str(c.right).strip().lower()
                    required.add(_ALIAS_RIGHT.get(rk, rk))
            if any(vals.get(k) is None for k in required):
                continue
            # Cooldown: at most one entry per symbol per cooldown window —
            # conditions can stay true for weeks; without this the engine
            # would pyramid a new position every bar.
            since = self._bars_since_signal.get(symbol, self.cooldown_bars)
            self._bars_since_signal[symbol] = since + 1
            prev = self._prev.get(symbol, {})
            ok = True
            for c in self.spec.entry_conditions:
                left = vals[c.left]
                right_key = str(c.right).strip().lower()
                right_key = _ALIAS_RIGHT.get(right_key, right_key)
                right = (
                    float(c.right)
                    if _is_number(c.right)
                    else vals.get(right_key)
                )
                pl = prev.get(c.left, left)
                pr = prev.get(right_key, right) if right_key in vals else right
                if not self._eval_condition(left, c.verb, right, {"_left": pl, "_right": pr}):
                    ok = False
                    break
            # Roll previous values forward for stateful verbs.
            self._prev[symbol] = dict(vals)
            if not ok or since < self.cooldown_bars:
                continue
            self._bars_since_signal[symbol] = 0

            # Direction: pinned specs trade one side; 'both' flips with the
            # conditions' bias (trend rules are symmetric). Engine expects
            # the string values 'LONG'/'SHORT'.
            dir_word = self.spec.direction
            if dir_word == "both":
                dir_word = "long" if left >= right else "short"
            atr = vals["atr"] or 0.0
            entry = float(bar.close)
            if atr <= 0:
                continue
            if dir_word == "long":
                sl = entry - self.spec.atr_sl_mult * atr
                tp = entry + self.spec.atr_tp_mult * atr
                direction = "LONG"
            else:
                sl = entry + self.spec.atr_sl_mult * atr
                tp = entry - self.spec.atr_tp_mult * atr
                direction = "SHORT"
            # Risk-normalized size: SL hit ≈ risk_per_trade_pct of equity,
            # independent of pair/quote currency (engine P&L uses fixed
            # 100k contract units).
            sl_dist = abs(entry - sl)
            volume = 1.0
            if sl_dist > 0:
                volume = (self.account_balance * self.risk_per_trade_pct) / (
                    sl_dist * 100_000.0
                )
            volume = max(0.01, min(5.0, volume))
            signals.append(
                {
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry,
                    "sl": sl,
                    "tp": tp,
                    "volume": round(volume, 2),
                    "comment": f"spec:{self.spec.name}",
                }
            )
        return signals


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


_ALIAS_RIGHT = {
    "close": "price",
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


def _overfit_analysis(window_rows: List[Dict[str, Any]], oos_trades: int) -> Dict[str, Any]:
    """Stage-A2 component breakdown over WF windows (lazy import: autonomy
    must not import the agent package at module load)."""
    try:
        from forex_ai_agent.agent.overfit import compute_overfit_components

        return compute_overfit_components(window_rows, oos_trades_total=oos_trades)
    except Exception:  # noqa: BLE001 — analysis must never break the gate
        return {}


def _repro_for(
    *,
    symbol: str,
    timeframe: str,
    candles_df: pd.DataFrame,
    thresholds: Any,
    backtest_config: Any,
    rules: Any = None,
) -> Dict[str, Any]:
    """Stage-A1 repro dict for this gate run (lazy import, never fatal)."""
    try:
        from forex_ai_agent.agent.repro import build_repro, feed_identity

        feed = feed_identity()
        return build_repro(
            rules=rules,
            symbol=symbol,
            timeframe=timeframe,
            bar_count=len(candles_df),
            data_start=candles_df.index.min(),
            data_end=candles_df.index.max(),
            feed_provider=feed["provider"],
            feed_chain=feed["chain"],
            thresholds=thresholds,
            backtest_config=backtest_config,
        )
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """Everything the gate measured, plus the pass/fail decision and reasons."""

    spec_name: str
    passed: bool
    reasons: List[str] = field(default_factory=list)
    full_backtest: Dict[str, Any] = field(default_factory=dict)
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    data_range: Dict[str, str] = field(default_factory=dict)
    repro: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_name": self.spec_name,
            "passed": self.passed,
            "reasons": self.reasons,
            "full_backtest": self.full_backtest,
            "walk_forward": self.walk_forward,
            "data_range": self.data_range,
            "repro": self.repro,
            "evaluated_at": self.evaluated_at,
        }


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@dataclass
class GateThresholds:
    """Objective pass criteria — a proposal must clear ALL of them.

    Defaults sized for Yahoo's ~1000-bar H1 history window; with a 12-bar
    entry cooldown the realistic ceiling is a few dozen trades.
    """

    min_trades_full: int = 20
    min_oos_trades: int = 6           # summed over walk-forward test windows
    min_oos_sharpe: float = 0.3
    max_oos_drawdown: float = 0.35    # fraction of equity
    max_overfitting_score: float = 0.5
    min_win_rate: float = 0.25        # loose: trend systems live on R:R
    min_profit_factor: float = 1.0


class BacktestValidator:
    """Compiles a spec and judges it against real historical data."""

    def __init__(
        self,
        thresholds: Optional[GateThresholds] = None,
        initial_balance: float = 100_000.0,
        walkforward: Optional[WalkForwardConfig] = None,
    ) -> None:
        self.thresholds = thresholds or GateThresholds()
        self.initial_balance = initial_balance
        self._wf_config = walkforward

    # ------------------------------------------------------------------

    async def validate(
        self,
        raw_spec: Dict[str, Any],
        candles_df: pd.DataFrame,
        symbol: str = "EUR/USD",
        run_walk_forward: bool = True,
    ) -> ValidationReport:
        """Validate a raw LLM proposal dict against ``candles_df`` (H1 bars)."""
        spec, errors = parse_spec(raw_spec)
        if spec is None:
            return ValidationReport(
                spec_name=str(raw_spec.get("name", "unknown")) if isinstance(raw_spec, dict) else "unknown",
                passed=False,
                reasons=[f"spec invalid: {e}" for e in errors],
            )
        return await self.validate_spec(
            spec, candles_df, symbol=symbol, run_walk_forward=run_walk_forward
        )

    async def validate_spec(
        self,
        spec: StrategySpec,
        candles_df: pd.DataFrame,
        symbol: str = "EUR/USD",
        run_walk_forward: bool = True,
        generator_factory: Optional[Callable[[], Any]] = None,
    ) -> ValidationReport:
        """Validate a parsed spec against ``candles_df`` (indexed H1 bars)."""
        report = ValidationReport(spec_name=spec.name, passed=False)
        if candles_df is None or len(candles_df) < 250:
            report.reasons.append(
                f"insufficient history: {0 if candles_df is None else len(candles_df)} bars (need ≥250)"
            )
            return report
        report.data_range = {
            "start": str(candles_df.index.min()),
            "end": str(candles_df.index.max()),
            "bars": str(len(candles_df)),
        }

        pip = pip_size_for_symbol(symbol)
        gen = generator_factory() if generator_factory else SpecSignalGenerator(spec, pip_size=pip)
        config = BacktestConfig(
            start_date=candles_df.index.min().to_pydatetime(),
            end_date=candles_df.index.max().to_pydatetime(),
            initial_balance=self.initial_balance,
            symbols=[symbol],
            timeframe=spec.timeframe,
            warmup_bars=60,
        )
        report.repro = _repro_for(
            symbol=symbol,
            timeframe=spec.timeframe,
            candles_df=candles_df,
            thresholds=self.thresholds,
            backtest_config=config,
            rules=spec.model_dump(),
        )
        try:
            result = run_backtest({symbol: candles_df}, gen, config)
        except Exception as exc:  # noqa: BLE001 — a crashing strategy is a rejection
            report.reasons.append(f"backtest crashed: {type(exc).__name__}: {exc}")
            return report

        raw_metrics = dict(result.metrics or {})
        # Engine returns nested metric groups; flatten the ones we gate on.
        summary = dict(raw_metrics.get("summary") or {})
        portfolio = dict(raw_metrics.get("portfolio") or {})
        metrics = {
            **summary,
            "max_drawdown": portfolio.get("max_drawdown_pct", 0.0) / 100.0,
            "final_equity": getattr(result, "stats", {}).get("final_equity"),
        }
        report.full_backtest = {
            "total_trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct", 0.0) / 100.0,
            "total_return_pct": summary.get("total_return_pct", 0.0),
            "final_equity": getattr(result, "stats", {}).get("final_equity"),
        }
        trades = summary.get("total_trades", 0)
        if trades < self.thresholds.min_trades_full:
            report.reasons.append(
                f"too few trades in full backtest: {trades} < {self.thresholds.min_trades_full}"
            )
        wr = summary.get("win_rate", 0.0)
        if wr < self.thresholds.min_win_rate:
            report.reasons.append(
                f"win rate {wr:.0%} below floor {self.thresholds.min_win_rate:.0%}"
            )
        pf = summary.get("profit_factor", 0.0)
        if pf < self.thresholds.min_profit_factor:
            report.reasons.append(
                f"profit factor {pf:.2f} below floor {self.thresholds.min_profit_factor:.2f}"
            )

        wf_summary: Dict[str, Any] = {}
        if run_walk_forward and trades >= self.thresholds.min_trades_full:
            try:
                wf_summary = self._walk_forward(
                    spec, candles_df, symbol, pip, generator_factory=generator_factory
                )
                report.walk_forward = wf_summary
            except Exception as exc:  # noqa: BLE001
                report.reasons.append(f"walk-forward failed: {type(exc).__name__}: {exc}")

        if wf_summary:
            if wf_summary.get("oos_trades", 0) < self.thresholds.min_oos_trades:
                report.reasons.append(
                    f"OOS trades {wf_summary.get('oos_trades', 0)} < {self.thresholds.min_oos_trades}"
                )
            if wf_summary.get("oos_sharpe", -9) < self.thresholds.min_oos_sharpe:
                report.reasons.append(
                    f"OOS sharpe {wf_summary.get('oos_sharpe', 0):.2f} < {self.thresholds.min_oos_sharpe}"
                )
            if wf_summary.get("oos_max_drawdown", 1.0) > self.thresholds.max_oos_drawdown:
                report.reasons.append(
                    f"OOS drawdown {wf_summary.get('oos_max_drawdown', 1.0):.0%} > "
                    f"{self.thresholds.max_oos_drawdown:.0%}"
                )
            if wf_summary.get("overfitting_score", 1.0) > self.thresholds.max_overfitting_score:
                report.reasons.append(
                    f"overfitting score {wf_summary.get('overfitting_score', 1.0):.2f} > "
                    f"{self.thresholds.max_overfitting_score}"
                )
            # Stage-A2: the composite component score catches failure modes the
            # scalar degradation misses (e.g. edge death: dead windows and
            # negative-window fraction don't move the IS→OOS ratio).
            analysis = wf_summary.get("overfit_analysis") or {}
            composite = analysis.get("score")
            if composite is not None and composite > self.thresholds.max_overfitting_score:
                highs = [
                    k.replace("_", " ")
                    for k, v in (analysis.get("components") or {}).items()
                    if isinstance(v, dict) and v.get("label") == "HIGH"
                ]
                suffix = f" (HIGH: {', '.join(highs)})" if highs else ""
                report.reasons.append(
                    f"overfit composite {composite:.2f} > "
                    f"{self.thresholds.max_overfitting_score}{suffix}"
                )

        report.passed = not report.reasons
        if report.passed:
            logger.info("Spec %r PASSED the backtest gate", spec.name)
        else:
            logger.info("Spec %r rejected: %s", spec.name, "; ".join(report.reasons))
        return report

    async def validate_generator(
        self,
        generator: Any,
        candles_df: pd.DataFrame,
        symbol: str = "EUR/USD",
        timeframe: str = "H1",
        run_walk_forward: bool = True,
        generator_factory: Optional[Callable[[], Any]] = None,
        spec_name: str = "custom",
    ) -> ValidationReport:
        """Run the SAME gate as validate_spec, but driven by a custom signal
        generator (a strategy that cannot be expressed as a StrategySpec).

        ``generator_factory`` must build a FRESH generator per backtest —
        stateful generators must never be shared across windows.
        """
        report = ValidationReport(spec_name=spec_name, passed=False)
        if candles_df is None or len(candles_df) < 250:
            report.reasons.append(
                f"insufficient history: {0 if candles_df is None else len(candles_df)} bars (need ≥250)"
            )
            return report
        report.data_range = {
            "start": str(candles_df.index.min()),
            "end": str(candles_df.index.max()),
            "bars": str(len(candles_df)),
        }

        config = BacktestConfig(
            start_date=candles_df.index.min().to_pydatetime(),
            end_date=candles_df.index.max().to_pydatetime(),
            initial_balance=self.initial_balance,
            symbols=[symbol],
            timeframe=timeframe,
            warmup_bars=60,
        )
        report.repro = _repro_for(
            symbol=symbol,
            timeframe=timeframe,
            candles_df=candles_df,
            thresholds=self.thresholds,
            backtest_config=config,
        )
        try:
            result = run_backtest({symbol: candles_df}, generator, config)
        except Exception as exc:  # noqa: BLE001 — a crashing strategy is a rejection
            report.reasons.append(f"backtest crashed: {type(exc).__name__}: {exc}")
            return report

        raw_metrics = dict(result.metrics or {})
        summary = dict(raw_metrics.get("summary") or {})
        portfolio = dict(raw_metrics.get("portfolio") or {})
        report.full_backtest = {
            "total_trades": summary.get("total_trades", 0),
            "win_rate": summary.get("win_rate", 0.0),
            "profit_factor": summary.get("profit_factor", 0.0),
            "sharpe_ratio": summary.get("sharpe_ratio", 0.0),
            "max_drawdown_pct": portfolio.get("max_drawdown_pct", 0.0) / 100.0,
            "total_return_pct": summary.get("total_return_pct", 0.0),
            "final_equity": getattr(result, "stats", {}).get("final_equity"),
        }
        trades = summary.get("total_trades", 0)
        if trades < self.thresholds.min_trades_full:
            report.reasons.append(
                f"too few trades in full backtest: {trades} < {self.thresholds.min_trades_full}"
            )
        wr = summary.get("win_rate", 0.0)
        if wr < self.thresholds.min_win_rate:
            report.reasons.append(
                f"win rate {wr:.0%} below floor {self.thresholds.min_win_rate:.0%}"
            )
        pf = summary.get("profit_factor", 0.0)
        if pf < self.thresholds.min_profit_factor:
            report.reasons.append(
                f"profit factor {pf:.2f} below floor {self.thresholds.min_profit_factor:.2f}"
            )

        wf_summary: Dict[str, Any] = {}
        if run_walk_forward and trades >= self.thresholds.min_trades_full:
            if generator_factory is None:
                report.walk_forward = {
                    "note": "walk-forward skipped: no generator_factory provided"
                }
            else:
                try:
                    wf_summary = self._walk_forward_generic(
                        generator_factory, candles_df, symbol, timeframe
                    )
                    report.walk_forward = wf_summary
                except Exception as exc:  # noqa: BLE001
                    report.reasons.append(f"walk-forward failed: {type(exc).__name__}: {exc}")

        if wf_summary:
            if wf_summary.get("oos_trades", 0) < self.thresholds.min_oos_trades:
                report.reasons.append(
                    f"OOS trades {wf_summary.get('oos_trades', 0)} < {self.thresholds.min_oos_trades}"
                )
            if wf_summary.get("oos_sharpe", -9) < self.thresholds.min_oos_sharpe:
                report.reasons.append(
                    f"OOS sharpe {wf_summary.get('oos_sharpe', 0):.2f} < {self.thresholds.min_oos_sharpe}"
                )
            if wf_summary.get("oos_max_drawdown", 1.0) > self.thresholds.max_oos_drawdown:
                report.reasons.append(
                    f"OOS drawdown {wf_summary.get('oos_max_drawdown', 1.0):.0%} > "
                    f"{self.thresholds.max_oos_drawdown:.0%}"
                )
            if wf_summary.get("overfitting_score", 1.0) > self.thresholds.max_overfitting_score:
                report.reasons.append(
                    f"overfitting score {wf_summary.get('overfitting_score', 1.0):.2f} > "
                    f"{self.thresholds.max_overfitting_score}"
                )
            # Stage-A2: the composite component score catches failure modes the
            # scalar degradation misses (e.g. edge death: dead windows and
            # negative-window fraction don't move the IS→OOS ratio).
            analysis = wf_summary.get("overfit_analysis") or {}
            composite = analysis.get("score")
            if composite is not None and composite > self.thresholds.max_overfitting_score:
                highs = [
                    k.replace("_", " ")
                    for k, v in (analysis.get("components") or {}).items()
                    if isinstance(v, dict) and v.get("label") == "HIGH"
                ]
                suffix = f" (HIGH: {', '.join(highs)})" if highs else ""
                report.reasons.append(
                    f"overfit composite {composite:.2f} > "
                    f"{self.thresholds.max_overfitting_score}{suffix}"
                )

        report.passed = not report.reasons
        if report.passed:
            logger.info("Custom strategy %r PASSED the backtest gate", spec_name)
        else:
            logger.info("Custom strategy %r rejected: %s", spec_name, "; ".join(report.reasons))
        return report

    def _walk_forward_generic(
        self,
        generator_factory: Callable[[], Any],
        candles_df: pd.DataFrame,
        symbol: str,
        timeframe: str,
    ) -> Dict[str, Any]:
        """Walk-forward for custom generators (same windows as _walk_forward)."""
        n = len(candles_df)
        n_windows = 4
        window_bars = max(60, n // 6)
        oos_sharpes: List[float] = []
        oos_trades = 0
        oos_drawdowns: List[float] = []
        is_sharpes: List[float] = []
        window_rows: List[Dict[str, Any]] = []

        for w in range(n_windows):
            train_end = int(n * (w + 1) / (n_windows + 1))
            train_start = 0 if w == 0 else int(n * w / (n_windows + 1))
            test_start, test_end = train_end, min(train_end + window_bars, n)
            if test_end - test_start < 40:
                continue
            train_df = candles_df.iloc[train_start:train_end]
            test_df = candles_df.iloc[test_start:test_end]
            if len(train_df) < 80 or len(test_df) < 40:
                continue

            is_result = run_backtest(
                {symbol: train_df},
                generator_factory(),
                BacktestConfig(
                    start_date=train_df.index.min().to_pydatetime(),
                    end_date=train_df.index.max().to_pydatetime(),
                    initial_balance=self.initial_balance,
                    symbols=[symbol],
                    timeframe=timeframe,
                    warmup_bars=60,
                ),
            )
            oos_result = run_backtest(
                {symbol: test_df},
                generator_factory(),
                BacktestConfig(
                    start_date=test_df.index.min().to_pydatetime(),
                    end_date=test_df.index.max().to_pydatetime(),
                    initial_balance=self.initial_balance,
                    symbols=[symbol],
                    timeframe=timeframe,
                    warmup_bars=60,
                ),
            )
            is_summary = dict(is_result.metrics.get("summary") or {})
            oos_summary = dict(oos_result.metrics.get("summary") or {})
            oos_portfolio = dict(oos_result.metrics.get("portfolio") or {})
            is_sharpes.append(float(is_summary.get("sharpe_ratio", 0.0)))
            oos_sharpes.append(float(oos_summary.get("sharpe_ratio", 0.0)))
            oos_trades += int(oos_summary.get("total_trades", 0))
            oos_drawdowns.append(float(oos_portfolio.get("max_drawdown_pct", 0.0)) / 100.0)
            window_rows.append(
                {
                    "window": w + 1,
                    "train_bars": len(train_df),
                    "test_bars": len(test_df),
                    "is_sharpe": float(is_summary.get("sharpe_ratio", 0.0)),
                    "oos_sharpe": float(oos_summary.get("sharpe_ratio", 0.0)),
                    "oos_trades": int(oos_summary.get("total_trades", 0)),
                    "oos_pf": float(oos_summary.get("profit_factor", 0.0)),
                }
            )

        if not oos_sharpes:
            return {}
        mean_oos = float(np.mean(oos_sharpes)) if oos_sharpes else 0.0
        std_oos = float(np.std(oos_sharpes)) if len(oos_sharpes) > 1 else 0.0
        mean_is = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        degradation = 1.0 - (mean_oos / mean_is) if mean_is > 0 else 1.0
        return {
            "windows": window_rows,
            "oos_trades": oos_trades,
            "oos_sharpe": mean_oos,
            "oos_sharpe_std": std_oos,
            "oos_max_drawdown": max(oos_drawdowns) if oos_drawdowns else 0.0,
            "overfitting_score": max(0.0, min(1.0, degradation)),
            "is_sharpe_mean": mean_is,
            "overfit_analysis": _overfit_analysis(window_rows, oos_trades),
        }

    # ------------------------------------------------------------------

    def _walk_forward(
        self,
        spec: StrategySpec,
        candles_df: pd.DataFrame,
        symbol: str,
        pip: float,
        generator_factory: Optional[Callable[[], Any]] = None,
    ) -> Dict[str, Any]:
        """Compact walk-forward: split history into expanding OOS windows."""
        n = len(candles_df)
        n_windows = 4
        window_bars = max(60, n // 6)
        oos_sharpes: List[float] = []
        oos_trades = 0
        oos_drawdowns: List[float] = []
        is_sharpes: List[float] = []
        window_rows: List[Dict[str, Any]] = []

        for w in range(n_windows):
            train_end = int(n * (w + 1) / (n_windows + 1))
            train_start = 0 if w == 0 else int(n * w / (n_windows + 1))
            test_start, test_end = train_end, min(train_end + window_bars, n)
            if test_end - test_start < 40:
                continue
            train_df = candles_df.iloc[train_start:train_end]
            test_df = candles_df.iloc[test_start:test_end]
            if len(train_df) < 80 or len(test_df) < 40:
                continue

            pip = pip_size_for_symbol(symbol)
            is_gen = generator_factory() if generator_factory else SpecSignalGenerator(spec, pip_size=pip)
            is_result = run_backtest(
                {symbol: train_df},
                is_gen,
                BacktestConfig(
                    start_date=train_df.index.min().to_pydatetime(),
                    end_date=train_df.index.max().to_pydatetime(),
                    initial_balance=self.initial_balance,
                    symbols=[symbol],
                    timeframe=spec.timeframe,
                    warmup_bars=60,
                ),
            )
            oos_gen = generator_factory() if generator_factory else SpecSignalGenerator(spec, pip_size=pip)
            oos_result = run_backtest(
                {symbol: test_df},
                oos_gen,
                BacktestConfig(
                    start_date=test_df.index.min().to_pydatetime(),
                    end_date=test_df.index.max().to_pydatetime(),
                    initial_balance=self.initial_balance,
                    symbols=[symbol],
                    timeframe=spec.timeframe,
                    warmup_bars=60,
                ),
            )
            is_summary = dict(is_result.metrics.get("summary") or {})
            oos_summary = dict(oos_result.metrics.get("summary") or {})
            oos_portfolio = dict(oos_result.metrics.get("portfolio") or {})
            is_sh = float(is_summary.get("sharpe_ratio", 0.0))
            oos_sh = float(oos_summary.get("sharpe_ratio", 0.0))
            oos_dd = float(oos_portfolio.get("max_drawdown_pct", 0.0)) / 100.0
            oos_n = int(oos_summary.get("total_trades", 0))
            is_sharpes.append(is_sh)
            oos_sharpes.append(oos_sh)
            oos_trades += oos_n
            oos_drawdowns.append(oos_dd)
            window_rows.append(
                {
                    "window": w + 1,
                    "train_bars": len(train_df),
                    "test_bars": len(test_df),
                    "is_sharpe": round(is_sh, 3),
                    "oos_sharpe": round(oos_sh, 3),
                    "oos_trades": oos_n,
                    "oos_max_drawdown": round(oos_dd, 4),
                }
            )

        oos_mean = float(np.mean(oos_sharpes)) if oos_sharpes else -9.0
        is_mean = float(np.mean(is_sharpes)) if is_sharpes else 0.0
        # Overfitting proxy: how much better the past looks than the future.
        overfit = float(np.clip(1.0 - (oos_mean / is_mean), 0.0, 1.0)) if is_mean > 0 else 1.0
        return {
            "windows": window_rows,
            "oos_sharpe": round(oos_mean, 3),
            "is_sharpe": round(is_mean, 3),
            "oos_trades": oos_trades,
            "oos_max_drawdown": max(oos_drawdowns) if oos_drawdowns else 1.0,
            "overfitting_score": round(overfit, 3),
            "overfit_analysis": _overfit_analysis(window_rows, oos_trades),
        }
