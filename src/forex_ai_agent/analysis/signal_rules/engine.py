"""Rule engine for signal evaluation.

Evaluates SignalRule instances against a MarketContext and produces
a SignalScore (total/passed weight, required pass/fail, failed/passed rule names).
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..signal_rules.base_rule import SignalRule, MarketContext, SignalScore


class RuleEngine:
    """Evaluates a collection of SignalRule instances against MarketContext."""

    def __init__(self, rules: List[SignalRule]):
        self.rules = rules
        # Group by category for reporting
        self._by_category: Dict[str, List[SignalRule]] = {}
        for rule in rules:
            cat = getattr(rule, "category", "GENERAL")
            self._by_category.setdefault(cat, []).append(rule)

    def evaluate(
        self,
        ctx: MarketContext,
        min_score_pct: float = 0.70,
    ) -> SignalScore:
        """Run all rules and return an aggregated SignalScore."""
        score = SignalScore()

        for rule in self.rules:
            score.total_weight += rule.weight
            try:
                passed = rule.condition(ctx)
            except Exception:
                passed = False

            if passed:
                score.passed_weight += rule.weight
                score.passed_rules.append(rule.name)
            else:
                score.failed_rules.append(rule.name)
                if rule.required:
                    score.required_passed = False

        # Compute percentage
        if score.total_weight > 0:
            score._score_pct = round(score.passed_weight / score.total_weight, 2)
        else:
            score._score_pct = 0.0

        return score

    def get_rules_by_category(self, category: str) -> List[SignalRule]:
        return self._by_category.get(category, [])