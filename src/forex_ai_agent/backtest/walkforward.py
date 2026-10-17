"""Walk-Forward Analysis Framework.

Implements expanding and anchored window validation for robust strategy testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from forex_ai_agent.backtest.engine import BacktestConfig, DataFeed, BacktestResult, run_backtest
from forex_ai_agent.backtest.metrics import calculate_metrics, monte_carlo_simulation


@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration."""
    train_window_days: int = 252  # ~1 year
    test_window_days: int = 63    # ~1 quarter
    step_days: int = 21           # ~1 month step
    min_train_days: int = 100
    anchored: bool = False        # If True, train window expands from fixed start
    metric: str = "sharpe_ratio"  # Metric to optimize
    min_trades: int = 30          # Minimum trades in test period
    optimization_metric: str = "sharpe_ratio"
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    regime_aware: bool = False    # If True, analyze by market regime
    confidence_level: float = 0.95  # Confidence level for parameter stability
    n_bootstrap: int = 1000       # Bootstrap iterations for confidence intervals


@dataclass
class WalkForwardWindow:
    """Single walk-forward window result."""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: Optional[Dict] = None
    test_result: Optional[Dict] = None
    best_params: Optional[Dict] = None
    is_valid: bool = True
    validation_notes: List[str] = field(default_factory=list)
    regime_metrics: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis result."""
    config: WalkForwardConfig
    windows: List[WalkForwardWindow]
    aggregate_metrics: Dict[str, float]
    parameter_stability: Dict[str, Dict[str, float]]
    parameter_confidence_intervals: Dict[str, Dict[str, float]]
    regime_stability: Dict[str, Dict[str, float]]
    overfitting_score: float
    statistical_significance: Dict[str, float]
    combined_equity_curve: List[Tuple[datetime, float]]
    window_equity_curves: List[List[Tuple[datetime, float]]]
    summary: str = ""


class WalkForwardAnalyzer:
    """Walk-forward analysis engine."""

    def __init__(
        self,
        config: WalkForwardConfig,
        data: Dict[str, pd.DataFrame],
        signal_generator_factory: Callable[[Dict], Any],
        base_config: BacktestConfig,
    ):
        self.config = config
        self.data = data
        self.signal_generator_factory = signal_generator_factory
        self.base_config = base_config
        self.windows: List[WalkForwardWindow] = []

    def run(self) -> WalkForwardResult:
        """Run complete walk-forward analysis."""
        print(f"Starting walk-forward analysis...")
        print(f"  Train window: {self.config.train_window_days} days")
        print(f"  Test window: {self.config.test_window_days} days")
        print(f"  Step: {self.config.step_days} days")
        print(f"  Anchored: {self.config.anchored}")
        print(f"  Regime-aware: {self.config.regime_aware}")

        # Get full data range
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)

        if not all_dates:
            return WalkForwardResult(
                config=self.config,
                windows=[],
                aggregate_metrics={},
                parameter_stability={},
                parameter_confidence_intervals={},
                regime_stability={},
                overfitting_score=0,
                statistical_significance={},
                combined_equity_curve=[],
                window_equity_curves=[],
                summary="No data available",
            )

        min_date = min(all_dates)
        max_date = max(all_dates)

        # Generate windows
        windows = self._generate_windows(min_date, max_date)
        print(f"Generated {len(windows)} windows")

        # Run each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}: Train {train_start.date()} to {train_end.date()}, Test {test_start.date()} to {test_end.date()}")

            window = WalkForwardWindow(
                window_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            # Optimize on train
            best_params = self._optimize_parameters(train_start, train_end)
            if best_params:
                window.best_params = best_params
                print(f"  Best params: {best_params}")

            # Test on out-of-sample
            test_result = self._run_test(test_start, test_end, best_params)
            window.test_result = test_result

            if test_result:
                window.train_result = self._run_test(train_start, train_end, best_params)

            # Regime-aware analysis
            if self.config.regime_aware and test_result:
                window.regime_metrics = self._analyze_by_regime(test_start, test_end, best_params)

            self.windows.append(window)

        # Analyze results
        result = self._analyze_results()
        result.summary = self._generate_summary(result)
        return result

    def _generate_windows(
        self,
        min_date: datetime,
        max_date: datetime,
    ) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """Generate walk-forward windows."""
        windows = []

        if self.config.anchored:
            # Anchored: train always starts at min_date
            train_start = min_date
            current = min_date + timedelta(days=self.config.train_window_days)

            while current + timedelta(days=self.config.test_window_days) <= max_date:
                train_end = current
                test_start = current
                test_end = current + timedelta(days=self.config.test_window_days)

                if (train_end - train_start).days >= self.config.min_train_days:
                    windows.append((train_start, train_end, test_start, test_end))

                current += timedelta(days=self.config.step_days)
        else:
            # Rolling: train window moves forward
            train_start = min_date
            current = min_date + timedelta(days=self.config.train_window_days)

            while current + timedelta(days=self.config.test_window_days) <= max_date:
                train_end = current
                test_start = current
                test_end = current + timedelta(days=self.config.test_window_days)

                if (train_end - train_start).days >= self.config.min_train_days:
                    windows.append((train_start, train_end, test_start, test_end))

                train_start += timedelta(days=self.config.step_days)
                current += timedelta(days=self.config.step_days)

        return windows

    def _optimize_parameters(
        self,
        train_start: datetime,
        train_end: datetime,
    ) -> Optional[Dict]:
        """Optimize parameters on training window."""
        if not self.config.parameter_ranges:
            return {}

        best_params = None
        best_score = -np.inf

        # Grid search (can be replaced with Bayesian optimization)
        param_names = list(self.config.parameter_ranges.keys())
        param_values = [np.linspace(r[0], r[1], 5) for r in self.config.parameter_ranges.values()]

        from itertools import product
        for combo in product(*param_values):
            params = dict(zip(param_names, combo))

            # Create signal generator with params
            signal_gen = self.signal_generator_factory(params)

            # Run backtest on train window
            config = BacktestConfig(
                start_date=train_start,
                end_date=train_end,
                symbols=self.base_config.symbols,
                timeframe=self.base_config.timeframe,
            )

            # Filter data for train window
            train_data = {}
            for symbol, df in self.data.items():
                mask = (df.index >= train_start) & (df.index <= train_end)
                train_data[symbol] = df[mask]

            if not train_data or all(len(df) == 0 for df in train_data.values()):
                continue

            try:
                result = run_backtest(train_data, signal_gen, config)
                score = result.metrics.get(self.config.optimization_metric, 0)

                if score > best_score and result.metrics.get("total_trades", 0) >= self.config.min_trades:
                    best_score = score
                    best_params = params
            except Exception as e:
                print(f"    Optimization error: {e}")
                continue

        return best_params

    def _run_test(
        self,
        test_start: datetime,
        test_end: datetime,
        params: Optional[Dict],
    ) -> Optional[Dict]:
        """Run backtest on test window with given parameters."""
        if not params:
            params = {}

        signal_gen = self.signal_generator_factory(params)

        config = BacktestConfig(
            start_date=test_start,
            end_date=test_end,
            symbols=self.base_config.symbols,
            timeframe=self.base_config.timeframe,
        )

        test_data = {}
        for symbol, df in self.data.items():
            mask = (df.index >= test_start) & (df.index <= test_end)
            test_data[symbol] = df[mask]

        if not test_data or all(len(df) == 0 for df in test_data.values()):
            return None

        try:
            result = run_backtest(test_data, signal_gen, config)
            return {
                "metrics": result.metrics,
                "trades": result.trades,
                "equity_curve": result.equity_curve,
                "stats": result.stats,
            }
        except Exception as e:
            print(f"    Test error: {e}")
            return None

    def _analyze_by_regime(
        self,
        test_start: datetime,
        test_end: datetime,
        params: Optional[Dict],
    ) -> Dict[str, Dict]:
        """Analyze performance by market regime."""
        if not params:
            params = {}

        signal_gen = self.signal_generator_factory(params)

        config = BacktestConfig(
            start_date=test_start,
            end_date=test_end,
            symbols=self.base_config.symbols,
            timeframe=self.base_config.timeframe,
        )

        test_data = {}
        for symbol, df in self.data.items():
            mask = (df.index >= test_start) & (df.index <= test_end)
            test_data[symbol] = df[mask]

        if not test_data or all(len(df) == 0 for df in test_data.values()):
            return {}

        try:
            result = run_backtest(test_data, signal_gen, config)
            
            # Simple regime classification based on volatility and trend
            regime_metrics = {}
            
            # Calculate market regime indicators for each symbol
            for symbol, df in test_data.items():
                if len(df) < 20:
                    continue
                    
                returns = df['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                trend = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 252 / len(df)
                
                # Classify regime
                if volatility > returns.std() * 1.5:
                    regime = "volatile"
                elif abs(trend) > volatility * 0.5:
                    regime = "trending"
                else:
                    regime = "ranging"
                
                if regime not in regime_metrics:
                    regime_metrics[regime] = {"trades": 0, "returns": []}
                
                if result.trades:
                    symbol_trades = [t for t in result.trades if t.get('symbol') == symbol]
                    regime_metrics[regime]["trades"] += len(symbol_trades)
                    regime_metrics[regime]["returns"].extend([t.get('pnl', 0) for t in symbol_trades])
            
            # Summarize regime metrics
            summary = {}
            for regime, data in regime_metrics.items():
                if data["returns"]:
                    summary[regime] = {
                        "mean_return": np.mean(data["returns"]),
                        "std_return": np.std(data["returns"]),
                        "trades": data["trades"],
                        "sharpe": np.mean(data["returns"]) / np.std(data["returns"]) if np.std(data["returns"]) > 0 else 0,
                    }
            
            return summary
        except Exception as e:
            print(f"    Regime analysis error: {e}")
            return {}

    def _compute_parameter_confidence_intervals(self) -> Dict[str, Dict[str, float]]:
        """Compute bootstrap confidence intervals for parameter stability."""
        if not self.windows or not self.windows[0].best_params:
            return {}
        
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_level
        
        for param_name in self.windows[0].best_params:
            values = [w.best_params.get(param_name, 0) for w in self.windows if w.best_params]
            if len(values) < 2:
                continue
            
            # Bootstrap confidence intervals
            n_bootstrap = self.config.n_bootstrap
            bootstrap_means = []
            
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                bootstrap_means.append(np.mean(sample))
            
            lower = np.percentile(bootstrap_means, alpha / 2 * 100)
            upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)
            
            confidence_intervals[param_name] = {
                "mean": np.mean(values),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "ci_width": float(upper - lower),
            }
        
        return confidence_intervals

    def _compute_statistical_significance(self) -> Dict[str, float]:
        """Compute statistical significance of walk-forward results."""
        if not self.windows:
            return {}
        
        test_metrics = [w.test_result["metrics"] for w in self.windows if w.test_result]
        if not test_metrics:
            return {}
        
        sharpe_values = [m.get("summary", {}).get("sharpe_ratio", 0) for m in test_metrics]
        
        if len(sharpe_values) < 2:
            return {}
        
        # T-test: H0: mean sharpe = 0
        t_stat, p_value = stats.ttest_1samp(sharpe_values, 0)
        
        # One-sided test: H0: mean sharpe <= 0
        p_value_one_sided = p_value / 2 if t_stat > 0 else 1 - p_value / 2
        
        # Probability of positive Sharpe (binomial test)
        n_positive = sum(1 for s in sharpe_values if s > 0)
        n_total = len(sharpe_values)
        p_binomial = stats.binomtest(n_positive, n_total, 0.5, alternative='greater').pvalue
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "p_value_one_sided": float(p_value_one_sided),
            "p_binomial": float(p_binomial),
            "prob_positive_sharpe": float(n_positive / n_total),
        }

    def _compute_regime_stability(self) -> Dict[str, Dict[str, float]]:
        """Compute parameter stability across market regimes."""
        if not self.config.regime_aware or not self.windows:
            return {}
        
        regime_params = {}
        
        for window in self.windows:
            if not window.regime_metrics or not window.best_params:
                continue
            
            for regime, metrics in window.regime_metrics.items():
                if regime not in regime_params:
                    regime_params[regime] = {}
                
                for param_name, value in window.best_params.items():
                    if param_name not in regime_params[regime]:
                        regime_params[regime][param_name] = []
                    regime_params[regime][param_name].append(value)
        
        stability = {}
        for regime, params in regime_params.items():
            stability[regime] = {}
            for param_name, values in params.items():
                if len(values) >= 2:
                    stability[regime][param_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "cv": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                    }
        
        return stability

    def _analyze_results(self) -> WalkForwardResult:
        """Analyze walk-forward results."""
        if not self.windows:
            return WalkForwardResult(
                config=self.config,
                windows=[],
                aggregate_metrics={},
                parameter_stability={},
                parameter_confidence_intervals={},
                regime_stability={},
                overfitting_score=0,
                statistical_significance={},
                combined_equity_curve=[],
                window_equity_curves=[],
                summary="No valid windows",
            )

        # Aggregate metrics across windows
        test_metrics = [w.test_result["metrics"] for w in self.windows if w.test_result]

        if not test_metrics:
            return WalkForwardResult(
                config=self.config,
                windows=self.windows,
                aggregate_metrics={},
                parameter_stability={},
                parameter_confidence_intervals={},
                regime_stability={},
                overfitting_score=0,
                statistical_significance={},
                combined_equity_curve=[],
                window_equity_curves=[],
                summary="No valid test results",
            )

        # Aggregate key metrics
        aggregate = {}
        for key in ["sharpe_ratio", "profit_factor", "win_rate", "expectancy", "total_return_pct", "max_drawdown_pct"]:
            values = [m.get("summary", {}).get(key, 0) for m in test_metrics]
            if values:
                aggregate[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                }

        # Parameter stability
        param_stability = {}
        if self.windows[0].best_params:
            for param_name in self.windows[0].best_params:
                values = [w.best_params.get(param_name, 0) for w in self.windows if w.best_params]
                if values:
                    param_stability[param_name] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "cv": np.std(values) / np.mean(values) if np.mean(values) != 0 else 0,
                        "min": np.min(values),
                        "max": np.max(values),
                    }

        # Parameter confidence intervals
        param_confidence_intervals = self._compute_parameter_confidence_intervals()

        # Regime stability
        regime_stability = self._compute_regime_stability()

        # Overfitting score (train vs test performance gap)
        overfitting_score = 0
        if len(self.windows) > 1:
            train_scores = [w.train_result["metrics"]["summary"].get("sharpe_ratio", 0) for w in self.windows if w.train_result]
            test_scores = [w.test_result["metrics"]["summary"].get("sharpe_ratio", 0) for w in self.windows if w.test_result]

            if train_scores and test_scores:
                avg_train = np.mean(train_scores)
                avg_test = np.mean(test_scores)
                overfitting_score = max(0, (avg_train - avg_test) / avg_train) if avg_train != 0 else 0

        # Statistical significance
        statistical_significance = self._compute_statistical_significance()

        # Combine equity curves
        combined_equity = []
        window_curves = []
        for w in self.windows:
            if w.test_result and w.test_result.get("equity_curve"):
                curve = w.test_result["equity_curve"]
                window_curves.append(curve)
                combined_equity.extend(curve)

        # Sort by timestamp
        combined_equity.sort(key=lambda x: x[0])

        return WalkForwardResult(
            config=self.config,
            windows=self.windows,
            aggregate_metrics=aggregate,
            parameter_stability=param_stability,
            parameter_confidence_intervals=param_confidence_intervals,
            regime_stability=regime_stability,
            overfitting_score=overfitting_score,
            statistical_significance=statistical_significance,
            combined_equity_curve=combined_equity,
            window_equity_curves=window_curves,
        )

    def _generate_summary(self, result: WalkForwardResult) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Walk-Forward Analysis Summary",
            f"=" * 40,
            f"Windows: {len(result.windows)}",
            f"Overfitting Score: {result.overfitting_score:.2%}",
        ]
        
        if result.aggregate_metrics:
            lines.append(f"\nAggregate Metrics:")
            for metric, stats in result.aggregate_metrics.items():
                if isinstance(stats, dict):
                    lines.append(f"  {metric}: mean={stats.get('mean', 0):.3f}, std={stats.get('std', 0):.3f}")
        
        if result.parameter_stability:
            lines.append(f"\nParameter Stability:")
            for param, stats in result.parameter_stability.items():
                lines.append(f"  {param}: mean={stats.get('mean', 0):.4f}, CV={stats.get('cv', 0):.3f}")
        
        if result.statistical_significance:
            lines.append(f"\nStatistical Significance:")
            lines.append(f"  p-value (two-sided): {result.statistical_significance.get('p_value', 1):.4f}")
            lines.append(f"  p-value (one-sided): {result.statistical_significance.get('p_value_one_sided', 1):.4f}")
            lines.append(f"  Prob positive Sharpe: {result.statistical_significance.get('prob_positive_sharpe', 0):.2%}")
        
        return "\n".join(lines)


def run_walkforward(
    data: Dict[str, pd.DataFrame],
    signal_generator_factory: Callable[[Dict], Any],
    config: WalkForwardConfig,
    base_config: BacktestConfig,
) -> WalkForwardResult:
    """Run walk-forward analysis."""
    analyzer = WalkForwardAnalyzer(config, data, signal_generator_factory, base_config)
    return analyzer.run()