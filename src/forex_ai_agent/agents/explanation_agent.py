"""Explanation Agent - Explains WHY a signal exists based on structured evidence.

Receives structured evidence (not raw interpretation) and generates:
- Why the setup exists
- Why entry is here
- Why SL is here
- Why TP is here
- What invalidates the trade
- What evidence contradicts the setup
- What information is missing

Never describes confidence as guaranteed probability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ConfidenceLevel(Enum):
    """Confidence calibration - NOT probability."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ExplanationContext:
    """Context for explanation generation."""
    pair: str
    direction: str
    timeframe: str
    setup_type: str
    entry_zone: Tuple[float, float]
    stop_loss: float
    take_profit: float
    risk_reward: float
    market_state: Dict[str, any]
    supporting_evidence: List[Dict[str, any]]
    contradicting_evidence: List[Dict[str, any]]
    missing_information: List[str]
    invalidation_condition: str
    strategy_version: str


class ExplanationAgent:
    """
    The Explanation Agent explains why a signal exists based on structured evidence.

    It receives evidence collections (not raw unrestricted market interpretation)
    and produces auditable, traceable explanations.
    """

    # Templates for different explanation sections
    TEMPLATES = {
        "why_setup": (
            "WHY THIS SETUP EXISTS:\n"
            "{points}\n\n"
            "WHY ENTRY IS HERE:\n"
            "{entry_reason}\n\n"
            "WHY SL IS HERE:\n"
            "{sl_reason}\n\n"
            "WHY TP IS HERE:\n"
            "{tp_reason}\n\n"
            "WHAT INVALIDATES THE TRADE:\n"
            "{invalid_reason}\n\n"
            "CONTRADICTING FACTORS:\n"
            "{contradictions}\n\n"
            "MISSING INFORMATION:\n"
            "{missing}"
        ),
        "bullet_plant": "{num}. {text}",
    }

    def explain(self, context: ExplanationContext) -> str:
        """Generate a full explanation from the given context."""

        sections = []

        # 1. Why the setup exists
        why_setup = self._why_setup(context)
        sections.append(why_setup)

        # 2. Why entry is here
        why_entry = self._why_entry(context)
        sections.append(why_entry)

        # 3. Why SL is here
        why_sl = self._why_sl(context)
        sections.append(why_sl)

        # 4. Why TP is here
        why_tp = self._why_tp(context)
        sections.append(why_tp)

        # 5. What invalidates the trade
        why_invalidates = self._why_invalidates(context)
        sections.append(why_invalidates)

        # 6. Contradicting factors
        contradictions = self._contradicting_factors(context)
        sections.append(contradictions)

        # 7. Missing information
        missing = self._missing_information(context)
        sections.append(missing)

        return "\n\n".join(sections)

    def _why_setup(self, context: ExplanationContext) -> str:
        """Generate 'Why this setup exists' section."""
        points = []
        
        # Market state/trend
        if "trend" in context.market_state:
            trend = context.market_state["trend"]
            points.append(f"1. {context.pair} {context.direction} trend is {trend} on {context.timeframe}")
        
        # Structure
        if "structure" in context.market_state:
            structure = context.market_state["structure"]
            if structure.get("last_bos"):
                points.append(
                    f"2. {context.pair} formed a {context.direction} break of structure "
                    f"at {structure['last_bos'].get('level')} on {context.timeframe}"
                )
        
        # Pullback/retest
        if "structure" in context.market_state:
            structure = context.market_state["structure"]
            if structure.get("retests"):
                points.append(
                    f"3. Price is retracing to {structure.get('key_support', structure.get('key_resistance'))} "
                    f"(now {context.direction} {'support' if context.direction == 'LONG' else 'resistance'})"
                )
        
        # Momentum
        if "indicators" in context.market_state:
            ind = context.market_state["indicators"]
            rsi = ind.get("rsi")
            if rsi is not None:
                points.append(
                    f"4. RSI(14) = {rsi} - {'momentum supportive' if 40 <= rsi <= 70 else 'momentum weak'}"
                )
        
        # Volatility
        if "indicators" in context.market_state:
            ind = context.market_state["indicators"]
            atr = ind.get("atr")
            if atr is not None:
                points.append(
                    f"5. ATR(14) = {atr} pips - SL distance appropriate for current volatility"
                )
        
        bullet_points = "\n".join(
            self.TEMPLATES["bullet_plant"].format(num=i+1, text=p) 
            for i, p in enumerate(points)
        )
        return f"WHY THIS SETUP EXISTS:\n{bullet_points}"

    def _why_entry(self, context: ExplanationContext) -> str:
        """Generate 'Why entry is here' section."""
        entry_low, entry_high = context.entry_zone
        reason_parts = []
        
        # Entry zone based on structure
        if context.setup_type == "structure_break_retest":
            reason_parts.append(
                f"Entry zone {entry_low:.5f}-{entry_high:.5f} is the retest of broken "
                f"{'resistance' if context.direction == 'LONG' else 'support'} level"
            )
        elif context.setup_type == "liquidity_sweep_reversal":
            reason_parts.append(
                f"Entry zone captures the reversal after liquidity sweep of prior "
                f"{'lows' if context.direction == 'LONG' else 'highs'}"
            )
        else:
            reason_parts.append(
                f"Entry zone at current price with proximity to {context.setup_type}"
            )
        
        # Also check if entering at pullback
        if context.risk_reward and context.risk_reward >= 2.0:
            reason_parts.append(
                f"Risk/reward of {context.risk_reward:.1f}:1 supports entry at this level"
            )
        
        return f"WHY ENTRY IS HERE:\n{chr(10).join(reason_parts)}"

    def _why_sl(self, context: ExplanationContext) -> str:
        """Generate 'Why SL is here' section."""
        sl = context.stop_loss
        direction = context.direction
        
        reason_parts = []
        
        # Structural invalidation point
        if context.invalidation_condition:
            reason_parts.append(
                f"Stop loss at {sl:.5f} is the structural invalidation level - "
                f"if {context.pair} {direction} closes beyond this level, "
                f"the setup premise is broken"
            )
        
        # ATR-based reasoning
        if "indicators" in context.market_state:
            ind = context.market_state["indicators"]
            atr = ind.get("atr")
            if atr is not None:
                sl_pips = abs(context.entry_zone[0] - sl) / 0.0001  # rough pips
                reason_parts.append(
                    f"SL provides {sl_pips:.1f} pips of room, roughly {atr:.1f} pips "
                    f"ATR, allowing normal volatility fluctuation"
                )
        
        # Key level
        if "structure" in context.market_state:
            structure = context.market_state["structure"]
            sl_level = structure.get("key_support") if direction == "LONG" else structure.get("key_resistance")
            if sl_level:
                reason_parts.append(
                    f"SL at {sl:.5f} is just below {sl_level:.5f} "
                    f"{'support' if direction == 'LONG' else 'resistance'} level"
                )
        
        return f"WHY SL IS HERE:\n{chr(10).join(reason_parts)}"

    def _why_tp(self, context: ExplanationContext) -> str:
        """Generate 'Why TP is here' section."""
        tp = context.take_profit
        rr = context.risk_reward
        direction = context.direction
        entry_low, entry_high = context.entry_zone
        
        reason_parts = []
        
        # RR justification
        reason_parts.append(
            f"Risk/reward of {rr:.1f}:1 means taking {tp:.5f} targets "
            f"{(tp - entry_low if direction == 'LONG' else entry_high - tp):.1f} pips "
            f"risk"
        )
        
        # Structural target
        if "structure" in context.market_state:
            structure = context.market_state["structure"]
            if direction == "LONG":
                next_res = structure.get("key_resistance")
                if next_res:
                    reason_parts.append(
                        f"TP at {tp:.5f} targets next key resistance level at {next_res:.5f}"
                    )
            else:
                next_sup = structure.get("key_support")
                if next_sup:
                    reason_parts.append(
                        f"TP at {tp:.5f} targets next key support level at {next_sup:.5f}"
                    )
        
        # Fixed R multiple
        if rr:
            risk = abs(entry_low - context.stop_loss)
            target_move = risk * rr
            reason_parts.append(
                f"TP {rr:.1f}R above entry captures {target_move:.1f} pips of movement"
            )
        
        return f"WHY TP IS HERE:\n{chr(10).join(reason_parts)}"

    def _why_invalidates(self, context: ExplanationContext) -> str:
        """Generate 'What invalidates the trade' section."""
        return f"WHAT INVALIDATES THE TRADE:\n{context.invalidation_condition}"

    def _contradicting_factors(self, context: ExplanationContext) -> str:
        """Generate 'Contradicting factors' section."""
        contradictions = context.contradicting_evidence
        if not contradictions:
            return "WHAT INVALIDATES THE TRADE:\nNo significant contradicting factors identified."
        
        lines = []
        for i, c in enumerate(contradictions[:3], 1):  # Limit to top 3
            desc = c.get("description", c.get("value", "unknown"))
            lines.append(f"{i}. {desc}")
        
        if len(contradictions) > 3:
            lines.append(f"{len(contradictions) - 3}+ additional contradicting factors")
        
        return f"CONTRADICTING FACTORS:\n{chr(10).join(lines)}"

    def _missing_information(self, context: ExplanationContext) -> str:
        """Generate 'What information is missing' section."""
        missing = context.missing_information
        if not missing:
            return "MISSING INFORMATION:\nNo critical information missing."
        
        lines = [f"{i+1}. {m}" for i, m in enumerate(missing[:5])]  # Limit to top 5
        if len(missing) > 5:
            lines.append(f"{len(missing) - 5}+ additional missing data points")
        
        return f"MISSING INFORMATION:\n{chr(10).join(lines)}"


# Convenience function
def generate_signal_explanation(
    pair: str,
    direction: str,
    timeframe: str,
    setup_type: str,
    entry_zone: Tuple[float, float],
    stop_loss: float,
    take_profit: float,
    risk_reward: float,
    market_state: Dict[str, any],
    supporting_evidence: List[Dict[str, any]],
    contradicting_evidence: List[Dict[str, any]],
    missing_information: List[str],
    invalidation_condition: str,
    strategy_version: str = "v1.0",
) -> str:
    """Generate a full signal explanation from individual components."""
    
    context = ExplanationContext(
        pair=pair,
        direction=direction,
        timeframe=timeframe,
        setup_type=setup_type,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_reward=risk_reward,
        market_state=market_state,
        supporting_evidence=supporting_evidence,
        contradicting_evidence=contradicting_evidence,
        missing_information=missing_information,
        invalidation_condition=invalidation_condition,
        strategy_version=strategy_version,
    )
    
    agent = ExplanationAgent()
    return agent.explain(context)