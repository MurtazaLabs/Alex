"""Explanation / "Why?" layer (Phase 6).

- EXPLAIN buttons on Strategies/probation cards → evidence list with ✓/⚠ items
- Explain on Research page rows and gate rejections
- Research knowledge retriever: search over research_log, experiments, rejections
- Reuses explanation agent's evidence architecture
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from forex_ai_agent.autonomy.probation import ProbationStore
from forex_ai_agent.autonomy.research_log import recent_research_tasks
from forex_ai_agent.strategy import create_version_manager
from forex_ai_agent.data.database import get_session_factory
from forex_ai_agent.data.models.research import Experiment, Hypothesis, ResearchTask
from forex_ai_agent.data.models.autonomy import StrategyProbation
from sqlalchemy import select, or_, desc, String

# B2: words that must never become parameter-pattern guesses when routing
# "what happens when displacement > 0.25?"-style questions to the graph.
_GRAPH_STOPWORDS = {
    "what", "when", "which", "does", "happen", "happens", "this", "that",
    "with", "have", "been", "many", "times", "tested", "test", "strategy",
    "oos", "out", "sample", "above", "below", "over", "under", "than",
    "more", "less", "then", "were", "results", "result", "value", "values",
}


@dataclass
class EvidenceItem:
    """Single evidence item for an explanation."""
    check: str
    status: str  # "pass" | "fail" | "warn" | "info"
    detail: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class ExplanationCard:
    """Structured explanation card for UI rendering."""
    title: str
    subtitle: str
    evidence: List[EvidenceItem]
    concerns: List[str]
    recommendation: str
    verdict: str  # "PASS" | "FAIL" | "CONDITIONAL" | "UNKNOWN"
    generated_at: datetime = field(default_factory=datetime.utcnow)
    disclaimer: str = "This is research information, not trading advice."


@dataclass
class StrategyExplanation:
    """Full explanation for a strategy/probation item."""
    strategy_id: str
    strategy_name: str
    pair: str
    timeframe: str
    status: str
    evidence_cards: List[ExplanationCard]
    concerns: List[str]
    recommendation: str


class KnowledgeRetriever:
    """Search over research_log, experiments, rejections for 'why?' questions."""

    def __init__(self):
        self._session_factory = get_session_factory

    async def search(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Full-text search across research knowledge base."""
        results = []

        async with self._session_factory()() as session:
            # Search ResearchTasks
            task_query = select(ResearchTask).where(
                or_(
                    ResearchTask.concept_name.ilike(f"%{query}%"),
                    ResearchTask.conclusion_json.cast(String).ilike(f"%{query}%"),
                )
            ).order_by(desc(ResearchTask.created_at)).limit(limit)
            tasks = (await session.execute(task_query)).scalars().all()
            for t in tasks:
                concl = ""
                if t.conclusion_json:
                    concl = t.conclusion_json.get("summary") or t.conclusion_json.get("conclusion") or ""
                results.append({
                    "type": "research_task",
                    "id": str(t.id),
                    "topic": t.concept_name,
                    "verdict": t.status,
                    "conclusion": concl,
                    "recommendation": t.recommendation,
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                    "score": self._score_match(query, t.concept_name + " " + (t.conclusion_json.get("conclusion", "") if t.conclusion_json else "")),
                })

            # Search Experiments
            exp_query = select(Experiment).where(
                Experiment.parameters_json.cast(String).ilike(f"%{query}%")
            ).limit(limit)
            experiments = (await session.execute(exp_query)).scalars().all()
            for e in experiments:
                results.append({
                    "type": "experiment",
                    "id": str(e.id),
                    "hypothesis_id": str(e.hypothesis_id),
                    "parameters": e.parameters_json,
                    "results": e.results_json,
                    "conclusion": e.conclusion,
                    "created_at": None,
                    "score": self._score_match(query, str(e.parameters_json)),
                })

            # Search Hypotheses
            hyp_query = select(Hypothesis).where(
                or_(
                    Hypothesis.hypothesis_text.ilike(f"%{query}%"),
                    Hypothesis.null_hypothesis_text.ilike(f"%{query}%"),
                )
            ).order_by(desc(Hypothesis.created_at)).limit(limit)
            hypotheses = (await session.execute(hyp_query)).scalars().all()
            for h in hypotheses:
                results.append({
                    "type": "hypothesis",
                    "id": str(h.id),
                    "hypothesis_text": h.hypothesis_text,
                    "null_hypothesis_text": h.null_hypothesis_text,
                    "status": h.status,
                    "created_at": h.created_at.isoformat() if h.created_at else None,
                    "score": self._score_match(query, h.hypothesis_text or ""),
                })

        # Sort by relevance score
        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        return results[:limit]

    def _score_match(self, query: str, text: str) -> float:
        """Simple relevance scoring."""
        query_lower = query.lower()
        text_lower = text.lower()
        if query_lower in text_lower:
            # Position-based scoring
            pos = text_lower.find(query_lower)
            return 1.0 - (pos / max(len(text), 1)) * 0.5
        return 0.0

    async def get_strategy_history(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Get all research/experiments related to a strategy name."""
        results = []

        async with self._session_factory()() as session:
            # Find research tasks mentioning this strategy
            task_query = select(ResearchTask).where(
                ResearchTask.concept_name.ilike(f"%{strategy_name}%")
            ).order_by(desc(ResearchTask.created_at))
            tasks = (await session.execute(task_query)).scalars().all()
            for t in tasks:
                results.append({
                    "type": "research_task",
                    "topic": t.concept_name,
                    "verdict": t.status,
                    "conclusion": t.conclusion_json.get("conclusion", "") if t.conclusion_json else "",
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                })

            # Find experiments with this strategy in parameters
            exp_query = select(Experiment).where(
                Experiment.parameters_json.cast(String).ilike(f"%{strategy_name}%")
            )
            experiments = (await session.execute(exp_query)).scalars().all()
            for e in experiments:
                results.append({
                    "type": "experiment",
                    "id": str(e.id),
                    "parameters": e.parameters_json,
                    "results": e.results_json,
                    "conclusion": e.conclusion,
                    "created_at": None,
                })

        return results

    async def get_rejection_reasons(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent rejections with reasons."""
        results = []

        async with self._session_factory()() as session:
            # Rejected research tasks
            task_query = select(ResearchTask).where(
                ResearchTask.status == "FAILED"
            ).order_by(desc(ResearchTask.created_at)).limit(limit)
            tasks = (await session.execute(task_query)).scalars().all()
            for t in tasks:
                results.append({
                    "type": "rejected_research",
                    "topic": t.concept_name,
                    "reason": t.conclusion_json.get("reasons", ["No reason recorded"]) if t.conclusion_json else ["No reason recorded"],
                    "created_at": t.created_at.isoformat() if t.created_at else None,
                })

            # Rejected probation records
            store = ProbationStore()
            prob_records = await store.list()
            for r in prob_records:
                if r.status in {"failed", "rejected"}:
                    results.append({
                        "type": "rejected_strategy",
                        "strategy_name": r.strategy_name,
                        "version_id": r.version_id,
                        "pair": r.pair,
                        "reason": r.notes or "No reason recorded",
                        "status": r.status,
                        "created_at": r.started_at,
                    })

        return results[:limit]


class Explainer:
    """Generates EXPLAIN cards for strategies, probation, research."""

    def __init__(self):
        self.knowledge = KnowledgeRetriever()

    async def explain_probation(self, version_id: str) -> StrategyExplanation:
        """Generate EXPLAIN card for a probation record."""
        store = ProbationStore()
        records = await store.list()
        record = None
        for r in records:
            if r.version_id == version_id or r.version_id.startswith(version_id):
                record = r
                break

        if not record:
            raise ValueError(f"No probation record for {version_id}")

        # Build evidence cards
        evidence_cards = []

        # ProbationStatus exposes computed probation metrics (last_metrics),
        # not raw backtest gate results. Gate numbers live on the StrategyVersion.
        from forex_ai_agent.strategy import create_version_manager

        vm = await create_version_manager()
        version = vm.get_version(record.version_id) or (
            vm.get_version(record.version_id[:8]) if len(record.version_id) >= 8 else None
        )
        gate = (version.backtest_results if version else None) or {}
        paper = record.last_metrics or {}

        # Card 1: Gate validation
        gate_items = [
            EvidenceItem("OOS Profit Factor > floor", "pass" if (gate or {}).get("profit_factor", 0) > 1.2 else "fail",
                        f"OOS PF: {(gate or {}).get('profit_factor', 0):.2f} (floor: 1.2)",
                        metric_value=(gate or {}).get('profit_factor', 0), threshold=1.2),
            EvidenceItem("Trade count ≥ minimum", "pass" if (gate or {}).get("total_trades", 0) >= 20 else "fail",
                        f"Trades: {(gate or {}).get('total_trades', 0)} (min: 20)",
                        metric_value=(gate or {}).get('total_trades', 0), threshold=20),
            EvidenceItem("Max drawdown ≤ limit", "pass" if (gate or {}).get("max_drawdown_pct", 1) <= 0.20 else "fail",
                        f"Max DD: {(gate or {}).get('max_drawdown_pct', 0)*100:.1f}% (limit: 20%)",
                        metric_value=(gate or {}).get('max_drawdown_pct', 0), threshold=0.20),
            EvidenceItem("Win rate ≥ threshold", "pass" if (gate or {}).get("win_rate", 0) >= 0.45 else "warn",
                        f"Win rate: {(gate or {}).get('win_rate', 0)*100:.1f}% (target: 45%)",
                        metric_value=(gate or {}).get('win_rate', 0), threshold=0.45),
            EvidenceItem("Sharpe ratio ≥ 0.5", "pass" if (gate or {}).get("sharpe_ratio", 0) >= 0.5 else "warn",
                        f"Sharpe: {(gate or {}).get('sharpe_ratio', 0):.2f} (target: 0.5)",
                        metric_value=(gate or {}).get('sharpe_ratio', 0), threshold=0.5),
        ]
        evidence_cards.append(ExplanationCard(
            title="Backtest Gate Validation",
            subtitle=f"Probation: {record.strategy_name} on {record.pair}",
            evidence=gate_items,
            concerns=[],
            recommendation="Gate passed → entered paper probation" if record.status == "active" else "Gate failed → strategy rejected",
            verdict="PASS" if record.eligible_for_approval else "FAIL",
        ))

        # Card 2: Paper probation progress
        progress_items = [
            EvidenceItem("Observed trades", "info",
                        f"{record.observed_trades}/{record.required_trades} trades ({record.observed_trades/record.required_trades*100:.0f}%)",
                        metric_value=record.observed_trades, threshold=record.required_trades),
            EvidenceItem("Elapsed time", "info",
                        f"Started: {(record.started_at or 'Unknown')[:10]}",
                        metric_value=None),
        ]
        evidence_cards.append(ExplanationCard(
            title="Paper Probation Progress",
            subtitle=f"Status: {record.status}",
            evidence=progress_items,
            concerns=[],
            recommendation="Complete paper trades → await YOUR approval" if record.status == "active" else "Review required",
            verdict="PASS" if record.status == "active" else "UNKNOWN",
        ))

        # Card 3: Walk-forward / OOS stability (if available)
        wf = (gate or {}).get("walk_forward", {})
        if wf:
            wf_items = [
                EvidenceItem("OOS Sharpe", "pass" if wf.get("oos_sharpe", 0) >= 0.5 else "fail",
                            f"OOS Sharpe: {wf.get('oos_sharpe', 0):.2f} (target: 0.5)",
                            metric_value=wf.get("oos_sharpe", 0), threshold=0.5),
            EvidenceItem("Overfitting score", "pass" if wf.get("overfitting_score", 1) <= 0.3 else "warn",
                        f"Overfitting: {wf.get('overfitting_score', 0):.2f} (max: 0.3)",
                        metric_value=wf.get("overfitting_score", 0), threshold=0.3),
            *([
                EvidenceItem(
                    "Overfit components",
                    "info",
                    ", ".join(
                        f"{k.replace('_', ' ')} {v.get('label', '?')}" if isinstance(v, dict) else f"{k} {v}"
                        for k, v in (wf.get("overfit_analysis") or {}).get("components", {}).items()
                        if k != "parameter_sensitivity"
                    ) or "components unavailable",
                    metric_value=(wf.get("overfit_analysis") or {}).get("score"),
                ),
            ] if wf.get("overfit_analysis") else []),
                EvidenceItem("OOS Max DD", "pass" if wf.get("oos_max_drawdown", 1) <= 0.20 else "fail",
                            f"OOS Max DD: {wf.get('oos_max_drawdown', 0)*100:.1f}% (limit: 20%)",
                            metric_value=wf.get("oos_max_drawdown", 0), threshold=0.20),
            ]
            evidence_cards.append(ExplanationCard(
                title="Walk-Forward / OOS Stability",
                subtitle="Robustness check across time windows",
                evidence=wf_items,
                concerns=[],
                recommendation="Stable across windows" if all(i.status in {"pass", "info"} for i in wf_items) else "Instability detected",
                verdict="PASS" if all(i.status in {"pass", "info"} for i in wf_items) else "FAIL",
            ))

        # Overall concerns
        concerns = []
        if record.observed_trades < record.required_trades * 0.5:
            concerns.append(f"⚠ Only {record.observed_trades}/{record.required_trades} trades — low trade frequency")
        if (gate or {}).get("profit_factor", 0) < 1.3:
            concerns.append("⚠ Profit factor near minimum threshold")
        if (gate or {}).get("max_drawdown_pct", 0) > 0.15:
            concerns.append("⚠ Drawdown approaching limit")

        return StrategyExplanation(
            strategy_id=record.version_id,
            strategy_name=record.strategy_name,
            pair=record.pair,
            timeframe="H1",  # Would come from strategy
            status=record.status,
            evidence_cards=evidence_cards,
            concerns=concerns,
            recommendation="Approve for live" if record.eligible_for_approval else "Continue paper probation",
        )

    async def explain_research_task(self, task_id: str) -> ExplanationCard:
        """Generate EXPLAIN card for a research task."""
        from forex_ai_agent.autonomy.research_log import recent_research_tasks

        tasks = await recent_research_tasks(limit=100)
        task = None
        for t in tasks:
            if str(t.get("id", "")).startswith(task_id):
                task = t
                break

        if not task:
            raise ValueError(f"No research task for {task_id}")

        items = [
            EvidenceItem("Research verdict", "pass" if task.get("verdict") == "PASS" else "fail",
                        f"Verdict: {task.get('verdict') or 'Unknown'}"),
            EvidenceItem("Conclusion", "info", task.get("conclusion") or "No conclusion recorded"),
        ]

        concerns = []
        if task.get("verdict") == "FAIL":
            concerns.append("Research rejected — see rejection reasons below")

        return ExplanationCard(
            title=f"Research: {task.get('topic')}",
            subtitle=f"Status: {task.get('status')} | Verdict: {task.get('verdict') or 'Pending'}",
            evidence=items,
            concerns=concerns,
            recommendation=task.get("recommendation") or task.get("conclusion") or "No recommendation",
            verdict=task.get("verdict") or "UNKNOWN",
        )

    async def explain_gate_rejection(self, pair: str, days: int = 30) -> ExplanationCard:
        """Explain why a pair was rejected at the gate."""
        from forex_ai_agent.autonomy.proposal import fetch_history_for, propose_for_pair
        from forex_ai_agent.pipeline import Pipeline
        from forex_ai_agent.core.event_bus import InMemoryEventBus

        pipeline = Pipeline(event_bus=InMemoryEventBus())
        await pipeline.initialize()

        history = await fetch_history_for(pipeline, pair, days=days)
        result = await propose_for_pair(pipeline, pair, history, use_llm=True)

        if result.validation and not result.validation.passed:
            items = [
                EvidenceItem("Gate result", "fail", "REJECTED"),
            ]
            for reason in result.validation.reasons:
                items.append(EvidenceItem("Rejection reason", "fail", reason))
            fb = result.validation.full_backtest or {}
            if fb:
                items.append(EvidenceItem("Full backtest PF", "info",
                            f"PF: {fb.get('profit_factor', 0):.2f} (floor: 1.2)",
                            metric_value=fb.get("profit_factor", 0), threshold=1.2))
                items.append(EvidenceItem("Total trades", "info",
                            f"Trades: {fb.get('total_trades', 0)} (min: 20)",
                            metric_value=fb.get("total_trades", 0), threshold=20))
                items.append(EvidenceItem("Max DD", "info",
                            f"Max DD: {fb.get('max_drawdown_pct', 0)*100:.1f}% (limit: 20%)",
                            metric_value=fb.get("max_drawdown_pct", 0), threshold=0.20))

            return ExplanationCard(
                title=f"Gate Rejection: {pair}",
                subtitle="Strategy proposal failed validation",
                evidence=items,
                concerns=[r for r in result.validation.reasons],
                recommendation="Do not pursue — evidence does not meet thresholds",
                verdict="FAIL",
            )
        else:
            return ExplanationCard(
                title=f"Gate Passed: {pair}",
                subtitle="Strategy proposal passed validation",
                evidence=[EvidenceItem("Gate result", "pass", "PASSED")],
                concerns=[],
                recommendation="Strategy enrolled in paper probation",
                verdict="PASS",
            )

    async def overfit_card(self, version_id: str) -> ExplanationCard:
        """Stage-A2: component-by-component overfit analysis from stored data.

        Every value traces to a stored walk-forward window — no re-run needed.
        """
        import re

        from forex_ai_agent.agent.overfit import compute_overfit_components
        from forex_ai_agent.strategy import create_version_manager

        vm = await create_version_manager()
        version = vm.get_version(version_id) or (
            vm.get_version(version_id[:8]) if len(version_id) >= 8 else None
        )
        if version is None:
            raise ValueError(f"Strategy version {version_id!r} not found")

        gate = dict(getattr(version, "backtest_results", None) or {})
        wf = dict(gate.get("walk_forward") or {})
        analysis = dict(wf.get("overfit_analysis") or {})
        if not analysis:
            windows = wf.get("windows") or []
            if windows:
                analysis = compute_overfit_components(
                    windows, oos_trades_total=wf.get("oos_trades")
                )
        if not analysis or analysis.get("score") is None:
            raise ValueError(
                f"no walk-forward analysis stored for {version.name!r} — run "
                "walk_forward_test or the gate first"
            )

        score = float(analysis["score"])
        threshold = 0.5  # GateThresholds.max_overfitting_score
        components = dict(analysis.get("components") or {})

        def _item(name: str) -> "EvidenceItem":
            comp = dict(components.get(name) or {})
            label = str(comp.get("label", "UNKNOWN"))
            value = comp.get("value")
            return EvidenceItem(
                name.replace("_", " ").title(),
                "pass" if label == "LOW" else ("warn" if label == "MEDIUM" else "fail"),
                f"{label} ({comp.get('detail', '')})",
                metric_value=value if isinstance(value, (int, float)) else None,
            )

        items = [
            _item("degradation"),
            _item("stability"),
            _item("negative_window_fraction"),
            _item("trade_concentration"),
        ]
        sens = dict(components.get("parameter_sensitivity") or {})
        items.append(
            EvidenceItem(
                "Parameter sensitivity",
                "info" if sens.get("status") != "MEASURED" else (
                    "pass" if str(sens.get("status")) == "MEASURED" and float(sens.get("oos_sharpe_range") or 0) < 1.0 else "warn"
                ),
                (
                    f"{sens.get('status')}: {sens.get('note', 'not measured')}"
                    if sens.get("status") != "MEASURED"
                    else f"OOS Sharpe range {sens.get('oos_sharpe_min')}…{sens.get('oos_sharpe_max')} across variants"
                ),
                metric_value=sens.get("oos_sharpe_range") if isinstance(sens.get("oos_sharpe_range"), (int, float)) else None,
            )
        )
        items.append(
            EvidenceItem(
                "Composite score",
                "pass" if score <= threshold else "fail",
                f"Score: {score:.2f}   Threshold: ≤ {threshold:.2f}   "
                f"Result: {'PASS' if score <= threshold else 'FAIL'}",
                metric_value=score,
                threshold=threshold,
            )
        )

        concerns: List[str] = []
        for name, comp in components.items():
            if isinstance(comp, dict) and comp.get("label") == "HIGH":
                concerns.append(f"HIGH {name.replace('_', ' ')}: {comp.get('detail', '')}")
        if analysis.get("note"):
            concerns.append(str(analysis["note"]))

        return ExplanationCard(
            title="OVERFIT ANALYSIS",
            subtitle=f"{version.name} — from stored walk-forward windows (as of {str(gate.get('evaluated_at', 'unknown'))[:10]})",
            evidence=items,
            concerns=concerns or ["No HIGH components"],
            recommendation=(
                "Overfitting within tolerance — gate may accept"
                if score <= threshold
                else "Overfitting above threshold — reject or investigate components above"
            ),
            verdict="PASS" if score <= threshold else "FAIL",
        )

    async def research_graph_card(
        self,
        hypothesis_id: Optional[str] = None,
        param_pattern: Optional[str] = None,
    ) -> ExplanationCard:
        """B2 EXPLAIN card: graph answers from stored records, no re-runs."""
        from forex_ai_agent.agent.research_graph import (
            aggregate_by_parameter,
            hypothesis_test_count,
        )

        if hypothesis_id:
            data = await hypothesis_test_count(hypothesis_id)
            title = f"RESEARCH GRAPH — hypothesis {hypothesis_id}"
            items = [
                EvidenceItem(
                    "Times tested",
                    "info" if data["times_tested"] else "warn",
                    f"{data['times_tested']} experiment edges across "
                    f"{data['experiments']} experiment(s)",
                ),
                EvidenceItem(
                    "Outcomes",
                    "pass" if data["outcomes"].get("PASS") else "info",
                    f"PASS {data['outcomes']['PASS']} · FAIL {data['outcomes']['FAIL']} "
                    f"· INCONCLUSIVE {data['outcomes']['INCONCLUSIVE']}",
                ),
            ]
            if data["decisions"].get("adopt") or data["decisions"].get("reject"):
                items.append(
                    EvidenceItem(
                        "Human decisions",
                        "info",
                        f"adopt {data['decisions']['adopt']} · reject {data['decisions']['reject']} "
                        f"· defer {data['decisions']['defer']}",
                    )
                )
            for param, n in list(data["parameters_varied"].items())[:4]:
                items.append(EvidenceItem(f"Parameter varied: {param}", "info", f"{n} observations"))
            concerns = [] if data["times_tested"] else [
                "no stored tests reference this hypothesis id — nothing to conclude"
            ]
            return ExplanationCard(
                title=title,
                subtitle=f"from stored research-graph records (as of {datetime.now():%Y-%m-%d %H:%M})",
                evidence=items,
                concerns=concerns,
                recommendation=(
                    f"This hypothesis has {data['times_tested']} stored test(s); "
                    "see the experiments page for the individual verdicts."
                    if data["times_tested"]
                    else "Run a parameter experiment against it to produce the first edge."
                ),
                verdict="ANSWERED" if data["times_tested"] else "NO_DATA",
            )

        data = await aggregate_by_parameter(param_pattern or "")
        items = [
            EvidenceItem(
                f"{b['parameter']} = {b['value']}",
                "fail" if b["fail_rate"] and b["fail_rate"] >= 0.5 else "info",
                f"{b['n']} obs · fail rate {b['fail_rate']} · mean OOS Sharpe {b['mean_oos_sharpe']}",
            )
            for b in data["buckets"][:6]
        ]
        worst = None
        if data["buckets"]:
            scored = [b for b in data["buckets"] if b["mean_oos_sharpe"] is not None]
            if scored:
                worst = min(scored, key=lambda b: b["mean_oos_sharpe"])
        return ExplanationCard(
            title=f"RESEARCH GRAPH — parameter '{param_pattern}'",
            subtitle="grouped by stored parameter-change observations",
            evidence=items,
            concerns=[] if data["observations_matched"] else [
                "no stored observations match this parameter"
            ],
            recommendation=(
                f"Highest-risk setting observed: {worst['parameter']} = {worst['value']} "
                f"(mean OOS Sharpe {worst['mean_oos_sharpe']})"
                if worst
                else "Run parameter experiments to populate the graph."
            ),
            verdict="ANSWERED" if data["observations_matched"] else "NO_DATA",
        )

    async def answer_why(self, question: str) -> ExplanationCard:
        """Answer 'why?' questions using knowledge retriever + explanation agent."""
        # Route overfit questions to the Stage-A2 component card when a
        # strategy id is present in the question.
        import re as _re

        q_lower = question.lower()
        id_match = _re.search(r"\b[0-9a-f]{8}(?:[0-9a-f]{4})?\b", q_lower)

        if "overfit" in q_lower:
            if id_match:
                try:
                    return await self.overfit_card(id_match.group(0))
                except ValueError:
                    pass  # fall through to knowledge search

        # B2: route graph questions ("how many times have we tested this
        # hypothesis?" / "what happens OOS when displacement > 0.25?") to the
        # research graph — deterministic joins over stored records. The graph
        # only answers when it actually has data (verdict ANSWERED); anything
        # else falls through to knowledge search, so trigger words like "when"
        # never hijack unrelated questions with an empty card.
        if any(k in q_lower for k in ("tested", "how many times", "displacement", "when ")):
            card = None
            if id_match:
                try:
                    card = await self.research_graph_card(hypothesis_id=id_match.group(0))
                except Exception:  # noqa: BLE001 — fall through to knowledge search
                    card = None
            if card is None or card.verdict != "ANSWERED":
                words = [w for w in _re.findall(r"[a-z_]{4,}", q_lower) if w not in _GRAPH_STOPWORDS]
                if words:
                    try:
                        card = await self.research_graph_card(param_pattern=words[0])
                    except Exception:  # noqa: BLE001 — fall through
                        card = None
            if card is not None and card.verdict == "ANSWERED":
                return card

        # Search knowledge base
        results = await self.knowledge.search(question, limit=5)

        # Build context
        context = {
            "question": question,
            "search_results": results,
        }

        # Compose deterministically from knowledge-search results (the
        # ExplanationAgent is a setup-evidence template engine, not a Q&A brain).
        if results:
            evidence_items = []
            concerns = []
            for r in results[:5]:
                label = str(r.get("type", "record"))
                summary = str(
                    r.get("topic")
                    or r.get("hypothesis_text")
                    or r.get("conclusion")
                    or r.get("name")
                    or "N/A"
                )[:140]
                evidence_items.append(
                    EvidenceItem(f"{label}: {summary}", "info",
                                 str(r.get("verdict") or r.get("status") or ""))
                )
                reason = r.get("reasons") or r.get("rejection_reasons")
                if reason:
                    concerns.extend([str(x) for x in reason][:3])
            top = results[0]
            top_summary = str(
                top.get("topic") or top.get("hypothesis_text")
                or top.get("conclusion") or top.get("name") or "N/A"
            )
            return ExplanationCard(
                title="Answer (from knowledge base)",
                subtitle=f"Question: {question}",
                evidence=evidence_items,
                concerns=concerns[:5],
                recommendation=f"Most relevant: {top.get('type', 'record')} — {top_summary}",
                verdict="UNKNOWN",
            )
        else:
            return ExplanationCard(
                title="No Answer Found",
                subtitle=f"Question: {question}",
                evidence=[EvidenceItem("Result", "info", "No matching records in knowledge base")],
                concerns=["No research, experiments, or hypotheses match this query"],
                recommendation="Run research or check strategy name spelling",
                    verdict="UNKNOWN",
                )


# Tool handlers (Phase 6)

async def explain_probation(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate EXPLAIN card for a probation record."""
    explainer = Explainer()
    explanation = await explainer.explain_probation(args["version_id"])
    return {
        "strategy_id": explanation.strategy_id,
        "strategy_name": explanation.strategy_name,
        "pair": explanation.pair,
        "status": explanation.status,
        "evidence_cards": [
            {
                "title": c.title,
                "subtitle": c.subtitle,
                "evidence": [
                    {"check": e.check, "status": e.status, "detail": e.detail,
                     "metric_value": e.metric_value, "threshold": e.threshold}
                    for e in c.evidence
                ],
                "concerns": c.concerns,
                "recommendation": c.recommendation,
                "verdict": c.verdict,
            }
            for c in explanation.evidence_cards
        ],
        "concerns": explanation.concerns,
        "recommendation": explanation.recommendation,
    }


async def explain_research(args: Dict[str, Any]) -> Dict[str, Any]:
    """Generate EXPLAIN card for a research task."""
    explainer = Explainer()
    explanation = await explainer.explain_research_task(args["task_id"])
    return {
        "title": explanation.title,
        "subtitle": explanation.subtitle,
        "evidence": [
            {"check": e.check, "status": e.status, "detail": e.detail,
             "metric_value": e.metric_value, "threshold": e.threshold}
            for e in explanation.evidence
        ],
        "concerns": explanation.concerns,
        "recommendation": explanation.recommendation,
        "verdict": explanation.verdict,
    }


async def explain_gate(args: Dict[str, Any]) -> Dict[str, Any]:
    """Explain gate rejection for a pair."""
    explainer = Explainer()
    explanation = await explainer.explain_gate_rejection(
        args["pair"],
        args.get("days", 30)
    )
    return {
        "title": explanation.title,
        "subtitle": explanation.subtitle,
        "evidence": [
            {"check": e.check, "status": e.status, "detail": e.detail,
             "metric_value": e.metric_value, "threshold": e.threshold}
            for e in explanation.evidence
        ],
        "concerns": explanation.concerns,
        "recommendation": explanation.recommendation,
        "verdict": explanation.verdict,
    }


async def answer_why(args: Dict[str, Any]) -> Dict[str, Any]:
    """Answer 'why?' question using knowledge retriever."""
    explainer = Explainer()
    explanation = await explainer.answer_why(args["question"])
    return {
        "title": explanation.title,
        "subtitle": explanation.subtitle,
        "evidence": [
            {"check": e.check, "status": e.status, "detail": e.detail,
             "metric_value": e.metric_value, "threshold": e.threshold}
            for e in explanation.evidence
        ],
        "concerns": explanation.concerns,
        "recommendation": explanation.recommendation,
        "verdict": explanation.verdict,
    }


async def search_knowledge(args: Dict[str, Any]) -> Dict[str, Any]:
    """Search research knowledge base."""
    retriever = KnowledgeRetriever()
    results = await retriever.search(
        args["query"],
        limit=args.get("limit", 10),
        filters=args.get("filters"),
    )
    return {"results": results}


async def get_strategy_history(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get all research/experiments for a strategy name."""
    retriever = KnowledgeRetriever()
    results = await retriever.get_strategy_history(args["strategy_name"])
    return {"history": results}


async def get_rejection_reasons(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get recent rejections with reasons."""
    retriever = KnowledgeRetriever()
    results = await retriever.get_rejection_reasons(args.get("limit", 20))
    return {"rejections": results}


# Global instances
_explainer: Optional[Explainer] = None
_knowledge: Optional[KnowledgeRetriever] = None


def get_explainer() -> Explainer:
    global _explainer
    if _explainer is None:
        _explainer = Explainer()
    return _explainer


def get_knowledge_retriever() -> KnowledgeRetriever:
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeRetriever()
    return _knowledge