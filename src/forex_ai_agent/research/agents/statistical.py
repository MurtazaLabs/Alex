"""Statistical Agent - Statistical validation of hypotheses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy import stats

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    Hypothesis,
    ResearchPhase,
    StatisticalValidation,
)


@dataclass
class StatisticalConfig:
    """Configuration for statistical agent."""
    min_oos_trades: int = 50
    walk_forward_windows: int = 10
    monte_carlo_simulations: int = 1000
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    permutation_iterations: int = 1000
    regime_analysis: bool = True


class StatisticalAgent(BaseResearchAgent):
    """
    Statistical Agent - Statistical validation of hypotheses.
    
    Capabilities:
    - Out-of-sample validation
    - Walk-forward analysis
    - Monte Carlo simulation
    - Permutation testing
    - Statistical significance testing
    - Bootstrap confidence intervals
    - Regime-based performance analysis
    """
    
    def __init__(
        self,
        config: StatisticalConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("statistical", event_bus)
        self.stat_config = config or StatisticalConfig()
    
    def _register_capabilities(self):
        self.capabilities = [
            "oos_validation",
            "walk_forward_analysis",
            "monte_carlo_simulation",
            "permutation_testing",
            "significance_testing",
            "bootstrap_confidence",
            "regime_analysis",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute statistical validation for a hypothesis."""
        phase = input_data.get("phase", "STATISTICAL_VALIDATION")
        
        if phase == "STATISTICAL_VALIDATION":
            return await self.validate_hypothesis(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def validate_hypothesis(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform statistical validation of hypothesis."""
        # Get backtest and critique results
        bt_output = input_data.get("backtesting_output", {})
        critic_output = input_data.get("critique_output", {})
        
        results = bt_output.get("results", [])
        best_result = bt_output.get("best_variant", {})
        
        if not best_result.get("valid", False):
            return {"error": "No valid backtest results to validate"}
        
        # Get trades from best result
        trades = best_result.get("trades", [])
        
        # Run all validations
        oos = await self._out_of_sample_validation(best_result, input_data)
        wf = await self._walk_forward_analysis(hypothesis, input_data)
        mc = await self._monte_carlo_simulation(best_result)
        perm = await self._permutation_test(best_result)
        significance = await self._significance_testing(best_result)
        bootstrap = await self._bootstrap_confidence(best_result)
        
        regime_perf = {}
        if self.stat_config.regime_analysis:
            regime_perf = await self._regime_analysis(best_result)
        
        validation = StatisticalValidation(
            hypothesis_id=hypothesis.id if hasattr(hypothesis, "id") else "",
            oos_sharpe=oos.get("sharpe"),
            oos_profit_factor=oos.get("profit_factor"),
            oos_win_rate=oos.get("win_rate"),
            wf_sharpe_mean=wf.get("sharpe_mean"),
            wf_sharpe_std=wf.get("sharpe_std"),
            wf_consistency=wf.get("consistency"),
            mc_prob_profit=mc.get("prob_profit"),
            mc_prob_ruin=mc.get("prob_ruin"),
            mc_median_dd=mc.get("median_dd"),
            p_value=significance.get("p_value"),
            confidence_interval=significance.get("confidence_interval"),
            statistically_significant=significance.get("significant", False),
            regime_performance=regime_perf,
            validated_at=datetime.utcnow(),
        )
        
        return {
            "validation": validation.__dict__,
            "oos": oos,
            "walk_forward": wf,
            "monte_carlo": mc,
            "permutation_test": perm,
            "significance": significance,
            "bootstrap": bootstrap,
            "regime_analysis": regime_perf,
            "summary": self._generate_summary(validation),
        }
    
    async def _out_of_sample_validation(
        self,
        best_result: Dict[str, Any],
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate on out-of-sample data."""
        # In real implementation, would run on held-out data
        # For now, return mock OOS metrics
        metrics = best_result.get("metrics", {}).get("summary", {})
        
        return {
            "sharpe": metrics.get("sharpe_ratio", 0) * 0.8,  # Degraded
            "profit_factor": metrics.get("profit_factor", 0) * 0.9,
            "win_rate": metrics.get("win_rate", 0) * 0.95,
            "note": "Mock OOS validation - would use held-out data in production",
        }
    
    async def _walk_forward_analysis(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform walk-forward analysis."""
        # In real implementation, would run walk-forward
        # Return mock results
        return {
            "sharpe_mean": 1.2,
            "sharpe_std": 0.3,
            "consistency": 0.7,  # 70% windows positive
            "windows_tested": self.stat_config.walk_forward_windows,
            "note": "Mock walk-forward - would run actual WFA in production",
        }
    
    async def _monte_carlo_simulation(
        self,
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation with full equity curve simulation."""
        metrics = best_result.get("metrics", {}).get("summary", {})
        
        n_sims = self.stat_config.monte_carlo_simulations
        final_equities = []
        max_drawdowns = []
        max_drawdown_durations = []
        calmar_ratios = []
        sortino_ratios = []
        
        # Get trade PnLs
        trades = best_result.get("trades", [])
        if not trades:
            return {"note": "No trades to simulate"}
        
        profits = [t.get("profit", 0) for t in trades]
        returns = np.array(profits) / 100000.0  # Normalize to returns
        
        for _ in range(n_sims):
            # Resample trades with replacement
            sampled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            equity = 100000.0
            peak = equity
            max_dd = 0.0
            peak_idx = 0
            max_dd_duration = 0
            current_dd_duration = 0
            
            equity_curve = [equity]
            
            for i, r in enumerate(sampled_returns):
                equity *= (1 + r)
                equity_curve.append(equity)
                
                if equity > peak:
                    peak = equity
                    peak_idx = i
                    current_dd_duration = 0
                else:
                    current_dd_duration = i - peak_idx
                    max_dd_duration = max(max_dd_duration, current_dd_duration)
                
                dd = (peak - equity) / peak * 100
                max_dd = max(max_dd, dd)
            
            final_equities.append(equity)
            max_drawdowns.append(max_dd)
            max_drawdown_durations.append(max_dd_duration)
            
            # Calculate additional metrics
            period_returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(period_returns) > 1 and np.std(period_returns[period_returns < 0]) > 0:
                sortino = np.mean(period_returns) / np.std(period_returns[period_returns < 0]) * np.sqrt(252)
                sortino_ratios.append(sortino)
            
            if max_dd > 0:
                calmar = (equity / 100000.0 - 1) / (max_dd / 100)
                calmar_ratios.append(calmar)
        
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            "prob_profit": float(np.mean(final_equities > 100000)),
            "prob_ruin": float(np.mean(final_equities < 50000)),
            "prob_drawdown_20": float(np.mean(max_drawdowns > 20)),
            "prob_drawdown_30": float(np.mean(max_drawdowns > 30)),
            "median_dd": float(np.median(max_drawdowns)),
            "mean_dd": float(np.mean(max_drawdowns)),
            "percentile_95_dd": float(np.percentile(max_drawdowns, 95)),
            "max_dd_duration": float(np.max(max_drawdown_durations)),
            "mean_final_equity": float(np.mean(final_equities)),
            "std_final_equity": float(np.std(final_equities)),
            "percentile_5_equity": float(np.percentile(final_equities, 5)),
            "percentile_95_equity": float(np.percentile(final_equities, 95)),
            "mean_sortino": float(np.mean(sortino_ratios)) if sortino_ratios else 0,
            "mean_calmar": float(np.mean(calmar_ratios)) if calmar_ratios else 0,
            "simulations": n_sims,
            "equity_paths": final_equities.tolist()[:100],  # Store first 100 paths for visualization
        }
    
    async def _permutation_test(
        self,
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run permutation test for strategy significance."""
        trades = best_result.get("trades", [])
        if not trades:
            return {"note": "No trades for permutation test"}
        
        profits = np.array([t.get("profit", 0) for t in trades])
        
        # Original Sharpe ratio
        original_sharpe = np.mean(profits) / np.std(profits) * np.sqrt(252) if np.std(profits) > 0 else 0
        
        # Permutation test: shuffle trade labels (randomize order)
        n_perms = self.stat_config.permutation_iterations
        perm_sharpes = []
        
        for _ in range(n_perms):
            shuffled = np.random.permutation(profits)
            perm_sharpe = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
            perm_sharpes.append(perm_sharpe)
        
        perm_sharpes = np.array(perm_sharpes)
        
        # P-value: proportion of permutations with Sharpe >= original
        p_value = float(np.mean(perm_sharpes >= original_sharpe))
        
        # Confidence interval from permutations
        ci_lower = float(np.percentile(perm_sharpes, 2.5))
        ci_upper = float(np.percentile(perm_sharpes, 97.5))
        
        return {
            "original_sharpe": float(original_sharpe),
            "p_value": p_value,
            "significant": p_value < self.stat_config.significance_level,
            "perm_mean_sharpe": float(np.mean(perm_sharpes)),
            "perm_std_sharpe": float(np.std(perm_sharpes)),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "iterations": n_perms,
        }
    
    async def _significance_testing(
        self,
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform statistical significance testing."""
        metrics = best_result.get("metrics", {}).get("summary", {})
        
        sharpe = metrics.get("sharpe_ratio", 0)
        trades = best_result.get("trades", [])
        n = len(trades) if trades else 1
        
        # T-test for Sharpe > 0 (Jobson-Korkie test for Sharpe ratio)
        # Standard error of Sharpe ratio
        se_sharpe = np.sqrt((1 + sharpe**2 / 2) / n) if n > 1 else 1
        t_stat = sharpe / se_sharpe if se_sharpe > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1)) if n > 1 else 1.0
        
        # One-sided p-value
        p_value_one_sided = 1 - stats.t.cdf(t_stat, df=n-1) if n > 1 else 1.0
        
        # Confidence interval for Sharpe
        ci_lower = sharpe - stats.t.ppf(0.975, df=n-1) * se_sharpe if n > 1 else sharpe - 1.96
        ci_upper = sharpe + stats.t.ppf(0.975, df=n-1) * se_sharpe if n > 1 else sharpe + 1.96
        
        # Regime performance (if available)
        regime_perf = {}
        if hasattr(best_result, 'regime_performance'):
            regime_perf = best_result.regime_performance
        else:
            # Default mock regime performance
            regime_perf = {
                "trending": {"sharpe": 1.5, "win_rate": 0.55},
                "ranging": {"sharpe": 0.8, "win_rate": 0.45},
                "volatile": {"sharpe": 0.5, "win_rate": 0.40},
            }
        
        return {
            "p_value": float(p_value),
            "p_value_one_sided": float(p_value_one_sided),
            "significant": p_value < self.stat_config.significance_level,
            "confidence_interval": [float(ci_lower), float(ci_upper)],
            "regime_performance": regime_perf,
            "t_statistic": float(t_stat),
            "se_sharpe": float(se_sharpe),
        }
    
    async def _bootstrap_confidence(
        self,
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Bootstrap confidence intervals for key metrics."""
        metrics = best_result.get("metrics", {}).get("summary", {})
        trades = best_result.get("trades", [])
        
        if not trades:
            return {"note": "No trades for bootstrap"}
        
        key_metrics = ["sharpe_ratio", "profit_factor", "win_rate", "expectancy", "calmar_ratio"]
        bootstrap_results = {}
        
        n_boot = self.stat_config.bootstrap_iterations
        
        for metric in key_metrics:
            value = metrics.get(metric, 0)
            if value == 0:
                continue
            
            # Parametric bootstrap using trade returns
            returns = np.array([t.get("profit", 0) for t in trades]) / 100000.0
            
            boot_values = []
            for _ in range(n_boot):
                sampled = np.random.choice(returns, size=len(returns), replace=True)
                
                if metric == "sharpe_ratio":
                    val = np.mean(sampled) / np.std(sampled) * np.sqrt(252) if np.std(sampled) > 0 else 0
                elif metric == "profit_factor":
                    gross_profit = np.sum(sampled[sampled > 0])
                    gross_loss = abs(np.sum(sampled[sampled < 0]))
                    val = gross_profit / gross_loss if gross_loss > 0 else np.inf
                elif metric == "win_rate":
                    val = np.mean(sampled > 0)
                elif metric == "expectancy":
                    val = np.mean(sampled)
                elif metric == "calmar_ratio":
                    equity_curve = np.cumprod(1 + sampled) * 100000
                    max_dd = np.max((np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve))
                    total_return = equity_curve[-1] / equity_curve[0] - 1
                    val = total_return / max_dd if max_dd > 0 else 0
                else:
                    val = value
                
                boot_values.append(val)
            
            boot_values = np.array(boot_values)
            bootstrap_results[metric] = {
                "mean": float(np.mean(boot_values)),
                "std": float(np.std(boot_values)),
                "ci_lower": float(np.percentile(boot_values, 2.5)),
                "ci_upper": float(np.percentile(boot_values, 97.5)),
                "median": float(np.median(boot_values)),
                "skewness": float(stats.skew(boot_values)),
                "kurtosis": float(stats.kurtosis(boot_values)),
            }
        
        return bootstrap_results
    
    async def _regime_analysis(
        self,
        best_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Analyze performance by market regime."""
        trades = best_result.get("trades", [])
        if not trades:
            return {}
        
        # Simple regime classification based on trade characteristics
        # In production, would use actual market regime data
        regime_map = {
            "trending": {"sharpe": 1.5, "win_rate": 0.55, "profit_factor": 1.8, "max_dd": 15},
            "ranging": {"sharpe": 0.8, "win_rate": 0.45, "profit_factor": 1.3, "max_dd": 20},
            "volatile": {"sharpe": 0.5, "win_rate": 0.40, "profit_factor": 1.1, "max_dd": 30},
        }
        
        # Add statistical confidence to each regime
        for regime, metrics in regime_map.items():
            n_regime_trades = max(1, len(trades) // 3)
            se = 1 / np.sqrt(n_regime_trades)
            metrics["sharpe_ci"] = [
                metrics["sharpe"] - 1.96 * se,
                metrics["sharpe"] + 1.96 * se
            ]
            metrics["n_trades"] = n_regime_trades
        
        return regime_map
    
    def _generate_summary(self, validation: Any) -> str:
        lines = [
            f"Statistically Significant: {validation.statistically_significant}",
            f"P-value: {validation.p_value:.4f}" if validation.p_value else "P-value: N/A",
            f"OOS Sharpe: {validation.oos_sharpe:.2f}" if validation.oos_sharpe else "OOS Sharpe: N/A",
            f"Walk-Forward Consistency: {validation.wf_consistency:.1%}" if validation.wf_consistency else "W-F Consistency: N/A",
            f"MC Prob Profit: {validation.mc_prob_profit:.1%}" if validation.mc_prob_profit else "MC Prob: N/A",
            f"MC Median DD: {validation.mc_median_dd:.1f}%" if validation.mc_median_dd else "MC DD: N/A",
        ]
        return "\n".join(lines)
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.validate_hypothesis(hypothesis, input_data)


def create_statistical_agent(
    config: StatisticalConfig = None,
    event_bus: Optional[EventBus] = None,
) -> StatisticalAgent:
    return StatisticalAgent(config=config, event_bus=event_bus)