"""Quant Agent - Converts concepts to testable trading rules."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus
from forex_ai_agent.research.agents.base import BaseResearchAgent
from forex_ai_agent.research.models import (
    Hypothesis,
    ResearchPhase,
    Claim,
    ClaimType,
)


@dataclass
class QuantConfig:
    """Configuration for quant agent."""
    max_rules_per_hypothesis: int = 10
    parameter_grid_size: int = 5
    min_rule_complexity: int = 1
    max_rule_complexity: int = 5


class QuantAgent(BaseResearchAgent):
    """
    Quant Agent - Converts concepts to testable trading rules.
    
    Capabilities:
    - Concept to testable rule conversion
    - Parameter space definition
    - Multiple rule variants generation
    - Rule validation and sanity checks
    """
    
    def __init__(
        self,
        config: QuantConfig = None,
        event_bus: Optional[EventBus] = None,
    ):
        super().__init__("quant", event_bus)
        self.quant_config = config or QuantConfig()
    
    def _register_capabilities(self):
        self.capabilities = [
            "concept_to_rules",
            "parameter_space_definition",
            "rule_variant_generation",
            "rule_validation",
        ]
    
    async def execute(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute quant formalization for a hypothesis."""
        phase = input_data.get("phase", "QUANT_FORMALIZATION")
        
        if phase == "QUANT_FORMALIZATION":
            return await self.formalize_hypothesis(hypothesis, input_data)
        else:
            return {"error": f"Unknown phase: {phase}"}
    
    async def formalize_hypothesis(
        self,
        hypothesis: Any,
        input_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Convert hypothesis to testable rules."""
        # Get literature review claims
        lit_review = input_data.get("literature_review_output", {})
        claims = lit_review.get("claims", [])
        
        # Extract relevant claims
        relevant_claims = self._filter_relevant_claims(claims, hypothesis)
        
        # Generate entry rules
        entry_rules = self._generate_entry_rules(relevant_claims, hypothesis)
        
        # Generate exit rules
        exit_rules = self._generate_exit_rules(relevant_claims, hypothesis)
        
        # Generate risk rules
        risk_rules = self._generate_risk_rules(hypothesis)
        
        # Define parameter space
        param_space = self._define_parameter_space(hypothesis, entry_rules, exit_rules, risk_rules)
        
        # Generate rule variants
        variants = self._generate_variants(entry_rules, exit_rules, risk_rules, param_space)
        
        return {
            "entry_rules": entry_rules,
            "exit_rules": exit_rules,
            "risk_rules": risk_rules,
            "parameter_space": param_space,
            "variants": variants,
            "total_variants": len(variants),
            "formalization_complete": True,
        }
    
    def _filter_relevant_claims(
        self,
        claims: List[Claim],
        hypothesis: Any,
    ) -> List[Claim]:
        """Filter claims relevant to hypothesis."""
        relevant = []
        
        for claim in claims:
            # Check if claim relates to hypothesis variables
            if hasattr(hypothesis, "variables"):
                for var in hypothesis.variables.keys():
                    if var.lower() in claim.text.lower():
                        relevant.append(claim)
                        break
            else:
                # If no variables defined, use high-confidence claims
                if claim.confidence > 0.6:
                    relevant.append(claim)
        
        return relevant
    
    def _generate_entry_rules(
        self,
        claims: List[Claim],
        hypothesis: Any,
    ) -> List[Dict[str, Any]]:
        """Generate entry rules from claims."""
        rules = []
        
        # Trend-following entry
        rules.append({
            "name": "trend_follow_entry",
            "type": "ENTRY",
            "description": "Enter in direction of trend on pullback",
            "conditions": [
                "SMA(20) > SMA(50) for LONG, SMA(20) < SMA(50) for SHORT",
                "Price pulls back to SMA(20) zone",
                "RSI(14) between 30-70 (not overbought/oversold)",
            ],
            "parameters": {
                "fast_ma": {"default": 20, "min": 10, "max": 50},
                "slow_ma": {"default": 50, "min": 30, "max": 100},
                "rsi_period": {"default": 14, "min": 10, "max": 20},
                "rsi_min": {"default": 30, "min": 20, "max": 40},
                "rsi_max": {"default": 70, "min": 60, "max": 80},
            },
        })
        
        # Structure break entry
        rules.append({
            "name": "structure_break_entry",
            "type": "ENTRY",
            "description": "Enter on break of market structure",
            "conditions": [
                "Higher high / lower low break",
                "Retest of broken level",
                "Volume confirmation",
            ],
            "parameters": {
                "lookback": {"default": 20, "min": 10, "max": 50},
                "retest_tolerance": {"default": 0.0005, "min": 0.0001, "max": 0.001},
            },
        })
        
        return rules[:self.quant_config.max_rules_per_hypothesis]
    
    def _generate_exit_rules(
        self,
        claims: List[Claim],
        hypothesis: Any,
    ) -> List[Dict[str, Any]]:
        """Generate exit rules from claims."""
        rules = []
        
        # Take profit rules
        rules.append({
            "name": "fixed_r_tp",
            "type": "TAKE_PROFIT",
            "description": "Fixed R-multiple take profit",
            "conditions": [
                "Take profit at fixed R-multiple from entry",
            ],
            "parameters": {
                "r_multiple": {"default": 2.0, "min": 1.0, "max": 5.0},
            },
        })
        
        # Trailing stop
        rules.append({
            "name": "trailing_stop",
            "type": "TRAILING_STOP",
            "description": "Trailing stop to lock in profits",
            "conditions": [
                "Trail stop after 1R profit",
                "Trail distance based on ATR",
            ],
            "parameters": {
                "activation_r": {"default": 1.0, "min": 0.5, "max": 2.0},
                "trail_atr_multiple": {"default": 1.5, "min": 1.0, "max": 3.0},
            },
        })
        
        return rules
    
    def _generate_risk_rules(self, hypothesis: Any) -> List[Dict[str, Any]]:
        """Generate risk management rules."""
        rules = []
        
        # Position sizing
        rules.append({
            "name": "fixed_fractional_sizing",
            "type": "POSITION_SIZING",
            "description": "Fixed fractional position sizing based on account risk",
            "conditions": [
                "Risk per trade = account_equity * risk_percent",
                "Position size = risk_amount / (entry - stop_loss) * pip_value",
            ],
            "parameters": {
                "risk_percent": {"default": 0.01, "min": 0.0025, "max": 0.02},
            },
        })
        
        # Stop loss
        rules.append({
            "name": "structure_sl",
            "type": "STOP_LOSS",
            "description": "Stop loss at structure invalidation",
            "conditions": [
                "SL at recent swing high/low",
                "Minimum distance: 10 pips",
                "Maximum distance: 100 pips",
            ],
            "parameters": {
                "min_pips": {"default": 10, "min": 5, "max": 20},
                "max_pips": {"default": 100, "min": 50, "max": 200},
            },
        })
        
        # Correlation limit
        rules.append({
            "name": "correlation_limit",
            "type": "CORRELATION_LIMIT",
            "description": "Limit correlated position exposure",
            "conditions": [
                "Max 5% total risk in correlated pairs",
            ],
            "parameters": {
                "max_correlated_risk_pct": {"default": 0.05, "min": 0.02, "max": 0.10},
            },
        })
        
        return rules
    
    def _define_parameter_space(
        self,
        hypothesis: Any,
        entry_rules: List[Dict],
        exit_rules: List[Dict],
        risk_rules: List[Dict],
    ) -> Dict[str, Any]:
        """Define the parameter space for optimization."""
        param_space = {}
        
        for rule in entry_rules + exit_rules + risk_rules:
            for param_name, param_config in rule.get("parameters", {}).items():
                full_name = f"{rule['name']}.{param_name}"
                param_space[full_name] = {
                    "type": "float" if isinstance(param_config.get("default"), float) else "int",
                    "default": param_config.get("default"),
                    "min": param_config.get("min"),
                    "max": param_config.get("max"),
                    "rule": rule["name"],
                }
        
        return param_space
    
    def _generate_variants(
        self,
        entry_rules: List[Dict],
        exit_rules: List[Dict],
        risk_rules: List[Dict],
        param_space: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate rule variants for testing."""
        variants = []
        
        # Base variant (all default)
        base = {
            "name": "base",
            "entry_rules": [r["name"] for r in entry_rules],
            "exit_rules": [r["name"] for r in exit_rules],
            "risk_rules": [r["name"] for r in risk_rules],
            "parameters": {k: v["default"] for k, v in param_space.items()},
        }
        variants.append(base)
        
        # Generate parameter variants
        for param_name, param_config in param_space.items():
            if param_config["type"] == "float":
                for val in [param_config["min"], (param_config["min"] + param_config["max"]) / 2, param_config["max"]]:
                    variant = base.copy()
                    variant["name"] = f"{base['name']}_{param_name}_{val}"
                    variant["parameters"] = base["parameters"].copy()
                    variant["parameters"][param_name] = val
                    variants.append(variant)
        
        return variants[:20]  # Limit variants
    
    async def execute(self, hypothesis: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
        return await self.formalize_hypothesis(hypothesis, input_data)


def create_quant_agent(
    config: QuantConfig = None,
    event_bus: Optional[EventBus] = None,
) -> "QuantAgent":
    return QuantAgent(config=config, event_bus=event_bus)