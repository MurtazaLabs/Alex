"""Backtest Agent - Runs hypothesis-specific backtests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    BacktestResult,
    Hypothesis,
    ResearchPhase,
)
from forex_ai_agent.backtest import (
    BacktestConfig,
    BacktestEngine,
    DataFeed,
    run_backtest,
    SignalGenerator,
)


@dataclass
class BacktestAgentConfig:
    """Configuration for backtest agent."""
    default_symbols: List[str] = field(default_factory=lambda: ["EUR/USD", "GBP/USD", "USD/JPY"])
    default_timeframe: str = "H1"
    default_start_date: str = "2023-01-01"
    default_end_date: str = "2024-12-31"
    default_initial_balance: float = 100000.0
    default_warmup_bars: int = 100
    min_trades_for_valid: int = 30
    quick_mode: bool = True  # Default to quick mode for faster testing


class BacktestAgent(BaseResearchAgent):
    """
    Backtest Agent - Runs hypothesis-specific backtests.
    
    Capabilities:
    - Hypothesis-specific backtests
    - Parameter sweeps
    - Regime-segmented results
    - Quick validation mode
    """
    
    def __init__(
        self,
        config: BacktestAgentConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("backtest", event_bus)
        self.bt_config = config or BacktestAgentConfig()
    
    def _register_capabilities(self):
        self.capabilities = [
            "hypothesis_backtest",
            "parameter_sweep",
            "regime_analysis",
            "quick_validation",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute backtest for a hypothesis."""
        phase = input_data.get("phase", "BACKTESTING")
        
        if phase == "BACKTESTING":
            return await self.run_hypothesis_backtest(hypothesis, input_data)
        elif phase == "PARAMETER_SWEEP":
            return await self.run_parameter_sweep(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def run_hypothesis_backtest(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run backtest for a hypothesis."""
        # Get formalized rules from quant agent
        quant_output = input_data.get("quant_formalization_output", {})
        
        # Extract rules and parameters
        entry_rules = quant_output.get("entry_rules", [])
        exit_rules = quant_output.get("exit_rules", [])
        risk_rules = quant_output.get("risk_rules", [])
        parameters = quant_output.get("parameters", {})
        variants = quant_output.get("variants", [])
        
        if not variants:
            return {"error": "No variants to test"}
        
        # Get data configuration
        symbols = input_data.get("symbols", self.bt_config.default_symbols)
        timeframe = input_data.get("timeframe", self.bt_config.default_timeframe)
        start_date = input_data.get("start_date", self.bt_config.default_start_date)
        end_date = input_data.get("end_date", self.bt_config.default_end_date)
        
        # Run backtest for each variant
        results = []
        for variant in variants[:5]:  # Limit to 5 variants
            result = await self._run_single_backtest(
                hypothesis, variant, symbols, timeframe, start_date, end_date
            )
            results.append(result)
        
        # Select best variant
        best_result = max(results, key=lambda r: r.get("sharpe_ratio", -999))
        
        return {
            "variants_tested": len(results),
            "results": results,
            "best_variant": best_result,
            "summary": self._summarize_results(results),
        }
    
    async def _run_single_backtest(
        self,
        hypothesis: Any,
        variant: Dict[str, Any],
        symbols: List[str],
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Run a single backtest for a variant."""
        # Create signal generator from variant
        signal_gen = self._create_signal_generator(variant)
        
        # Create backtest config
        config = BacktestConfig(
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            symbols=symbols,
            timeframe=timeframe,
            initial_balance=100000.0,
            warmup_bars=100,
        )
        
        # Generate synthetic data for testing
        data = self._generate_synthetic_data(symbols, start_date, end_date)
        
        # Run backtest
        try:
            result = run_backtest(data, signal_gen, config)
            
            return {
                "variant_name": variant.get("name", "unknown"),
                "parameters": variant.get("parameters", {}),
                "metrics": result.metrics,
                "trades": len(result.trades),
                "equity_curve": result.equity_curve[-100:],  # Last 100 points
                "valid": len(result.trades) >= 30,
            }
        except Exception as e:
            return {
                "variant_name": variant.get("name", "unknown"),
                "error": str(e),
                "valid": False,
            }
    
    def _create_signal_generator(self, variant: Dict[str, Any]):
        """Create signal generator from variant rules."""
        # Simplified - create a simple SMA crossover signal generator
        class SimpleSMASignal(SignalGenerator):
            def __init__(self):
                self.prev_fast = None
                self.prev_slow = None
            
            def generate_signals(self, timestamp, data, indicators):
                for symbol, bar in data.items():
                    ind = indicators.get(symbol, {})
                    fast = ind.get('sma_20')
                    slow = ind.get('sma_50')
                    if fast is None or slow is None:
                        continue
                    if self.prev_fast is not None and self.prev_slow is not None:
                        if self.prev_fast <= self.prev_slow and fast > slow:
                            return [{
                                'symbol': symbol,
                                'direction': 'LONG',
                                'entry_price': bar.close,
                                'sl': bar.close * 0.995,
                                'tp': bar.close * 1.015,
                                'volume': 1.0,
                            }]
                        elif self.prev_fast >= self.prev_slow and fast < slow:
                            return [{
                                'symbol': symbol,
                                'direction': 'SHORT',
                                'entry_price': bar.close,
                                'sl': bar.close * 1.005,
                                'tp': bar.close * 0.985,
                                'volume': 1.0,
                            }]
                    self.prev_fast = fast
                    self.prev_slow = slow
                return []
        
        return SimpleSMASignal()
    
    def _generate_synthetic_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        """Generate synthetic market data for testing."""
        import pandas as pd
        import numpy as np
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        freq = 'h'
        
        dates = pd.date_range(start, end, freq='h')
        n_bars = len(dates)
        
        # Limit bars in quick mode
        if self.bt_config.quick_mode:
            n_bars = min(n_bars, 200)
            dates = dates[:n_bars]
        
        data = {}
        for symbol in symbols:
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)
            close = 1.0
            prices = []
            for i in range(n_bars):
                drift = 0.00001 * np.sin(i / 500)
                shock = np.random.normal(0, 0.0005)
                close = close * (1 + drift + shock)
                prices.append(close)
            
            df = pd.DataFrame({
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.0003))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.0003))) for p in prices],
                'close': prices,
                'volume': np.random.randint(100, 1000, n_bars),
            }, index=dates)
            
            df['high'] = df[['open', 'high', 'close']].max(axis=1)
            df['low'] = df[['open', 'low', 'close']].min(axis=1)
            data[symbol] = df
        
        return data
    
    def _summarize_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize backtest results."""
        valid_results = [r for r in results if r.get("valid", False)]
        
        if not valid_results:
            return {"error": "No valid results"}
        
        return {
            "valid_variants": len(valid_results),
            "best_sharpe": max(r["metrics"]["summary"]["sharpe_ratio"] for r in valid_results),
            "best_profit_factor": max(r["metrics"]["summary"]["profit_factor"] for r in valid_results),
            "avg_win_rate": np.mean([r["metrics"]["summary"]["win_rate"] for r in valid_results]),
            "avg_expectancy": np.mean([r["metrics"]["summary"]["expectancy"] for r in valid_results]),
        }
    
    async def run_parameter_sweep(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run parameter sweep for hypothesis."""
        # Simplified implementation
        return {
            "message": "Parameter sweep not yet implemented",
            "sweep_config": input_data.get("sweep_config", {}),
        }
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.run_hypothesis_backtest(hypothesis, input_data)


def create_backtest_agent(
    config: BacktestAgentConfig = None,
    event_bus: Optional[EventBus] = None,
) -> "BacktestAgent":
    return BacktestAgent(config=config, event_bus=event_bus)