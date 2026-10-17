"""Critic Agent - Analyzes hypotheses for flaws and weaknesses."""

from __future__ import annotations

import asyncio
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    CritiqueResult,
    Hypothesis,
    ResearchPhase,
)


@dataclass
class CriticConfig:
    """Configuration for critic agent."""
    min_sample_size: int = 100
    permutation_iterations: int = 1000
    sensitivity_threshold: float = 0.1
    overfitting_threshold: float = 0.7
    cherry_picking_threshold: float = 0.8


class CriticAgent(BaseResearchAgent):
    """
    Critic Agent - Analyzes hypotheses for flaws and weaknesses.
    
    Capabilities:
    - Parameter sensitivity analysis
    - Permutation testing
    - Regime dependence testing
    - Confounder control
    - Look-ahead audit
    - Overfitting detection
    - Cherry-picking checks
    """
    
    def __init__(
        self,
        config: CriticConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("critic", event_bus)
        self.critic_config = config or CriticConfig()
    
    def _register_capabilities(self):
        self.capabilities = [
            "parameter_sensitivity",
            "permutation_testing",
            "regime_dependence",
            "confounder_control",
            "lookahead_audit",
            "overfitting_detection",
            "cherry_picking_detection",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute critique of a hypothesis."""
        phase = input_data.get("phase", "CRITIQUE")
        
        if phase == "CRITIQUE":
            return await self.critique_hypothesis(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def critique_hypothesis(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform comprehensive critique of hypothesis."""
        # Get backtest results
        bt_output = input_data.get("backtesting_output", {})
        results = bt_output.get("results", [])
        
        if not results:
            return {"error": "No backtest results to critique"}
        
        best_result = bt_output.get("best_variant", {})
        
        # Run all critiques
        sensitivity = await self._parameter_sensitivity_analysis(results)
        permutation = await self._permutation_test(results)
        regime_dep = await self._regime_dependence_test(results)
        confounder = await self._confounder_check(hypothesis, input_data)
        lookahead = await self._lookahead_audit(hypothesis, input_data)
        overfitting = await self._overfitting_detection(results)
        cherry_picking = await self._cherry_picking_check(results)
        
        # Compile critique result
        critique = CritiqueResult(
            hypothesis_id=hypothesis.id if hasattr(hypothesis, "id") else "",
            overall_score=self._calculate_overall_score(
                sensitivity, permutation, regime_dep, confounder,
                lookahead, overfitting, cherry_picking
            ),
            parameter_sensitivity=sensitivity,
            permutation_test_p_value=permutation.get("p_value"),
            regime_dependence_score=regime_dep.get("score", 0),
            confounder_risk=confounder.get("risk", 0),
            look_ahead_bias_risk=lookahead.get("risk", 0),
            overfitting_score=overfitting.get("score", 0),
            cherry_picking_score=cherry_picking.get("score", 0),
            critical_issues=self._compile_critical_issues(
                sensitivity, permutation, regime_dep, confounder,
                lookahead, overfitting, cherry_picking
            ),
            warnings=self._compile_warnings(
                sensitivity, permutation, regime_dep, confounder,
                lookahead, overfitting, cherry_picking
            ),
            suggestions=self._compile_suggestions(
                sensitivity, permutation, regime_dep, confounder,
                lookahead, overfitting, cherry_picking
            ),
            verdict=self._determine_verdict(
                sensitivity, permutation, regime_dep, confounder,
                lookahead, overfitting, cherry_picking
            ),
            reviewed_at=datetime.utcnow(),
        )
        
        return {
            "critique": critique.__dict__,
            "summary": self._generate_summary(critique),
        }
    
    async def _parameter_sensitivity_analysis(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Analyze parameter sensitivity."""
        sensitivity = {}
        
        # Group results by parameter
        param_performance = {}
        for r in results:
            if not r.get("valid", False):
                continue
            params = r.get("parameters", {})
            for pname, pval in params.items():
                if pname not in param_performance:
                    param_performance[pname] = []
                param_performance[pname].append({
                    "value": pval,
                    "sharpe": r.get("metrics", {}).get("summary", {}).get("sharpe_ratio", 0),
                })
        
        # Calculate sensitivity for each parameter
        for pname, performances in param_performance.items():
            if len(performances) < 2:
                sensitivity[pname] = 0.0
                continue
            
            values = [p["value"] for p in performances]
            sharpes = [p["sharpe"] for p in performances]
            
            # Correlation between parameter value and Sharpe
            if len(set(values)) > 1:
                corr = np.corrcoef(values, sharpes)[0, 1]
                sensitivity[pname] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                sensitivity[pname] = 0.0
        
        return sensitivity
    
    async def _permutation_test(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run permutation test to check if results are significant."""
        # Collect all trade PnLs from valid results
        all_pnls = []
        for r in results:
            if not r.get("valid", False):
                continue
            # In real implementation, would get actual trade PnLs
            # For now, simulate based on metrics
            sharpe = r.get("metrics", {}).get("summary", {}).get("sharpe_ratio", 0)
            if sharpe > 0:
                all_pnls.append(sharpe)
        
        if len(all_pnls) < 10:
            return {"p_value": 1.0, "significant": False}
        
        # Permutation test: shuffle labels and recompute statistic
        observed_stat = np.mean(all_pnls)
        n_permutations = self.critic_config.permutation_iterations
        permuted_stats = []
        
        for _ in range(min(100, self.critic_config.permutation_iterations)):
            shuffled = np.random.permutation(all_pnls)
            perm_stat = np.mean(shuffled[:len(shuffled)//2]) - np.mean(shuffled[len(shuffled)//2:])
            permuted_stats.append(abs(perm_stat))
        
        p_value = np.mean(np.array(permuted_stats) >= abs(observed_stat))
        
        return {
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "observed_statistic": float(observed_stat),
        }
    
    async def _regime_dependence_test(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Test if strategy performance depends on market regime."""
        # Simplified: check if performance varies by time period
        regime_performance = {}
        
        for r in results:
            if not r.get("valid", False):
                continue
            # In real implementation, would analyze performance by regime
            pass
        
        return {
            "score": 0.3,  # Mock score
            "regime_performance": regime_performance,
            "regime_dependent": False,
        }
    
    async def _confounder_check(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check for confounding variables."""
        confounders = []
        
        # Check for common confounders
        lit_review = input_data.get("literature_review_output", {})
        claims = lit_review.get("claims", [])
        
        # Check for market regime confounder
        regime_claims = [c for c in claims if "regime" in c.get("text", "").lower()]
        if regime_claims:
            confounders.append("market_regime")
        
        # Check for volatility confounder
        vol_claims = [c for c in claims if "volatil" in c.get("text", "").lower()]
        if vol_claims:
            confounders.append("volatility")
        
        risk = len(confounders) * 0.1
        
        return {
            "risk": min(risk, 1.0),
            "confounders": confounders,
        }
    
    async def _lookahead_audit(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Audit for look-ahead bias."""
        issues = []
        
        # Check if rules use future information
        quant_output = input_data.get("quant_formalization_output", {})
        entry_rules = quant_output.get("entry_rules", [])
        
        for rule in entry_rules:
            conditions = rule.get("conditions", [])
            for cond in conditions:
                if any(word in cond.lower() for word in ["future", "next", "will be", "predict"]):
                    issues.append(f"Potential look-ahead in rule: {cond}")
        
        risk = len(issues) * 0.2
        
        return {
            "risk": min(risk, 1.0),
            "issues": issues,
        }
    
    async def _overfitting_detection(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Detect overfitting in backtest results."""
        if not results:
            return {"score": 1.0, "indicators": []}
        
        indicators = []
        score = 0.0
        
        # Check 1: Too many parameters for trades
        for r in results:
            if not r.get("valid", False):
                continue
            trades = r.get("trades", 0)
            # Mock parameter count
            param_count = 10
            if trades > 0 and param_count / trades > 0.5:
                indicators.append("Too many parameters relative to trades")
                score += 0.2
        
        # Check 2: In-sample vs out-of-sample (mock)
        score += 0.1
        indicators.append("No out-of-sample validation")
        
        # Check 3: Perfect performance (suspicious)
        for r in results:
            if not r.get("valid", False):
                continue
            metrics = r.get("metrics", {}).get("summary", {})
            if metrics.get("win_rate", 0) > 0.8 and metrics.get("profit_factor", 0) > 5:
                indicators.append("Suspiciously high performance")
                score += 0.3
        
        return {
            "score": min(score, 1.0),
            "indicators": indicators,
        }
    
    async def _cherry_picking_check(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Check for cherry-picking in results."""
        indicators = []
        score = 0.0
        
        # Check if only best variant reported
        valid_results = [r for r in results if r.get("valid", False)]
        
        if len(results) > 1 and len(valid_results) == 1:
            indicators.append("Only 1 variant reported as valid out of many tested")
            score += 0.3
        
        # Check for selective time period
        # (would need access to actual data)
        
        return {
            "score": min(score, 1.0),
            "indicators": indicators,
        }
    
    def _calculate_overall_score(
        self,
        sensitivity: Dict,
        permutation: Dict,
        regime_dep: Dict,
        confounder: Dict,
        lookahead: Dict,
        overfitting: Dict,
        cherry_picking: Dict,
    ) -> float:
        """Calculate overall critique score (0-1, higher is better)."""
        penalties = 0.0
        
        # Parameter sensitivity (high sensitivity = bad)
        avg_sensitivity = np.mean(list(sensitivity.values())) if sensitivity else 0
        penalties += avg_sensitivity * 0.1
        
        # Permutation test
        if permutation.get("p_value", 1) > 0.05:
            penalties += 0.2
        
        # Regime dependence
        penalties += regime_dep.get("score", 0) * 0.15
        
        # Confounder
        penalties += confounder.get("risk", 0) * 0.1
        
        # Lookahead
        penalties += lookahead.get("risk", 0) * 0.2
        
        # Overfitting
        penalties += overfitting.get("score", 0) * 0.2
        
        # Cherry picking
        penalties += cherry_picking.get("score", 0) * 0.1
        
        return max(0.0, 1.0 - penalties)
    
    def _compile_critical_issues(
        self,
        sensitivity: Dict,
        permutation: Dict,
        regime_dep: Dict,
        confounder: Dict,
        lookahead: Dict,
        overfitting: Dict,
        cherry_picking: Dict,
    ) -> List[str]:
        issues = []
        
        if permutation.get("p_value", 1) > 0.05:
            issues.append("Results not statistically significant (p > 0.05)")
        
        for issue in overfitting.get("indicators", []):
            issues.append(f"Overfitting: {issue}")
        
        for issue in lookahead.get("issues", []):
            issues.append(f"Look-ahead bias: {issue}")
        
        for issue in cherry_picking.get("indicators", []):
            issues.append(f"Cherry-picking: {issue}")
        
        for c in confounder.get("confounders", []):
            issues.append(f"Potential confounder: {c}")
        
        return issues
    
    def _compile_warnings(
        self,
        sensitivity: Dict,
        permutation: Dict,
        regime_dep: Dict,
        confounder: Dict,
        lookahead: Dict,
        overfitting: Dict,
        cherry_picking: Dict,
    ) -> List[str]:
        warnings = []
        
        high_sens = [k for k, v in sensitivity.items() if v > self.critic_config.sensitivity_threshold]
        if high_sens:
            warnings.append(f"High parameter sensitivity: {', '.join(high_sens)}")
        
        if regime_dep.get("regime_dependent", False):
            warnings.append("Strategy performance depends on market regime")
        
        if overfitting.get("score", 0) > self.critic_config.overfitting_threshold:
            warnings.append("Potential overfitting detected")
        
        return warnings
    
    def _compile_suggestions(
        self,
        sensitivity: Dict,
        permutation: Dict,
        regime_dep: Dict,
        confounder: Dict,
        lookahead: Dict,
        overfitting: Dict,
        cherry_picking: Dict,
    ) -> List[str]:
        suggestions = []
        
        if permutation.get("p_value", 1) > 0.05:
            suggestions.append("Collect more data or simplify strategy")
        
        high_sens = [k for k, v in sensitivity.items() if v > 0.1]
        if high_sens:
            suggestions.append(f"Fix sensitive parameters: {', '.join(high_sens)}")
        
        if overfitting.get("score", 0) > 0.3:
            suggestions.append("Reduce parameter count or add regularization")
        
        if lookahead.get("issues"):
            suggestions.append("Remove look-ahead bias from rules")
        
        suggestions.append("Run walk-forward analysis")
        suggestions.append("Test on multiple market regimes")
        
        return suggestions
    
    def _determine_verdict(
        self,
        sensitivity: Dict,
        permutation: Dict,
        regime_dep: Dict,
        confounder: Dict,
        lookahead: Dict,
        overfitting: Dict,
        cherry_picking: Dict,
    ) -> str:
        # FAIL conditions
        if permutation.get("p_value", 1) > 0.05:
            return "FAIL"
        if overfitting.get("score", 0) > self.critic_config.overfitting_threshold:
            return "FAIL"
        if lookahead.get("risk", 0) > 0.3:
            return "FAIL"
        if cherry_picking.get("score", 0) > self.critic_config.cherry_picking_threshold:
            return "FAIL"
        
        # CONDITIONAL_PASS conditions
        if (regime_dep.get("score", 0) > 0.5 or 
            confounder.get("risk", 0) > 0.3 or
            overfitting.get("score", 0) > 0.3):
            return "CONDITIONAL_PASS"
        
        return "PASS"
    
    def _generate_summary(self, critique: Any) -> str:
        lines = [
            f"Overall Score: {critique.overall_score:.2f}",
            f"Verdict: {critique.verdict}",
            f"Critical Issues: {len(critique.critical_issues)}",
            f"Warnings: {len(critique.warnings)}",
        ]
        if critique.critical_issues:
            lines.append("Issues:")
            for issue in critique.critical_issues[:3]:
                lines.append(f"  - {issue}")
        return "\n".join(lines)
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.critique_hypothesis(hypothesis, input_data)


def create_critic_agent(
    config: CriticConfig = None,
    event_bus: Optional[EventBus] = None,
) -> CriticAgent:
    return CriticAgent(config=config, event_bus=event_bus)