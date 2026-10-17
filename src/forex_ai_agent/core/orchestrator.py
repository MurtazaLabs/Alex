from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from ..config import get_settings
from ..config.logging import get_logger
from .event_bus import Event, EventBus
from .permission_manager import PermissionLevel, PermissionManager

logger = get_logger(__name__)
settings = get_settings()

# --- Intent keyword maps (deterministic, delegation starts here) -------------

_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "RESEARCH": [
        "research",
        "learn",
        "study",
        "investigate",
        "deepen",
        "understand",
        "find out",
        "hypothes",
        "literature",
    ],
    "ANALYSIS": [
        "analyz",
        "analyse",
        "trend",
        "signal",
        "setup",
        "structure",
        "indicator",
        "market",
        "pair",
        "price",
        "chart",
        "candle",
        "momentum",
        "volatility",
        "liquidity",
    ],
    "EXECUTION": [
        "execute",
        "buy",
        "sell",
        "open trade",
        "place order",
        "paper trade",
        "live trade",
        "enter position",
        "trade now",
    ],
    "STRATEGY": ["strategy", "rule", "backtest", "optimize", "versioning"],
    "STATUS": [
        "status",
        "position",
        "portfolio",
        "open trades",
        "watchlist",
        "what is happening",
        "summary",
    ],
}

_INTENT_CLASSIFICATION: Dict[str, Dict[str, Any]] = {
    "RESEARCH": {"category": "RESEARCH", "domains": ["research"], "required_level": 0},
    "ANALYSIS": {"category": "ANALYSIS", "domains": ["market"], "required_level": 1},
    "EXECUTION": {
        "category": "EXECUTION",
        "domains": ["risk", "execution"],
        "required_level": 3,
    },
    "STRATEGY": {
        "category": "STRATEGY",
        "domains": ["strategy", "research"],
        "required_level": 0,
    },
    "STATUS": {"category": "STATUS", "domains": [], "required_level": 0},
    "GENERAL": {"category": "GENERAL", "domains": [], "required_level": 0},
}

_DOMAIN_TO_SPECIALIST: Dict[str, str] = {
    "market": "market_agent",
    "research": "research_agent",
    "risk": "risk_agent",
    "strategy": "strategy_agent",
    "execution": "execution_agent",
    "news": "news_agent",
}


class Orchestrator:
    """
    Master orchestrator for the Forex AI Suite.

    Responsibilities:
    - Identify user intent
    - Classify the task
    - Select the minimum necessary specialist(s)
    - Coordinate their execution (parallel + sequential)
    - Resolve conflicts between specialist outputs
    - Validate the final result
    - Return one coherent, actionable answer
    """

    def __init__(
        self,
        permission_manager: PermissionManager,
        event_bus: Optional[EventBus] = None,
        specialists: Optional[List[Any]] = None,
    ):
        self.permission_manager = permission_manager
        self.event_bus = event_bus
        self._specialists: Dict[str, Any] = {}
        for specialist in specialists or []:
            self.register_specialist(specialist)

    def register_specialist(self, specialist: Any) -> None:
        """Register a specialist for routing."""
        self._specialists[specialist.name] = specialist
        logger.debug("specialist_registered", name=specialist.name)

    async def process_request(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process a user request through the orchestration loop.

        Loop: Intent → Decompose → Route → Execute → Validate → Synthesize
        """
        context = context or {}

        # Step 1: Identify user intent
        intent = self._identify_intent(user_input)

        # Step 2: Classify the task
        classification = self._classify_task(intent, user_input)

        # Step 3: Identify required domains
        domains = self._identify_domains(classification)

        # Step 4: Map domains to specialists
        specialists = self._map_to_specialists(domains)

        # Step 5: Check permissions
        if not self._check_permission_for_task(classification):
            return {
                "status": "permission_denied",
                "message": (
                    f"Current permission level: "
                    f"{PermissionLevel.label(self.permission_manager.current_level)}. "
                    f"Required: {classification.get('required_level', 'N/A')}"
                ),
            }

        # Step 6: Build execution plan
        execution_plan = self._build_execution_plan(
            specialists, classification, context
        )

        # Step 7: Run independent specialists in parallel
        specialist_results = await self._run_specialists_parallel(
            specialists, execution_plan
        )

        # Step 8: Run dependent specialists sequentially
        if execution_plan.get("dependent"):
            specialist_results = await self._run_specialists_sequential(
                specialists, execution_plan, specialist_results
            )

        # Step 9: Collect specialist results
        collected_results = self._collect_results(specialist_results)

        # Step 10: Check for conflicts
        conflict_resolution = self._check_conflicts(collected_results)

        # Step 11: Apply safety and risk gates
        safety_result = self._apply_safety_gates(conflict_resolution, classification)

        # Step 12: Synthesize final result
        final_result = self._synthesize_result(safety_result, classification, context)

        # Step 13: Classify final state
        final_state = self._classify_final_state(final_result)

        # Step 14: Store relevant knowledge
        await self._store_knowledge(final_result, classification, context)

        # Step 15: Return result to user
        return {
            "status": "completed",
            "final_state": final_state,
            "result": final_result,
            "intent": intent,
            "classification": classification,
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _identify_intent(self, user_input: str) -> str:
        """Classify the request into one of the intent buckets (keyword based)."""
        text = user_input.lower()
        for intent, keywords in _INTENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return intent
        return "GENERAL"

    def _classify_task(self, intent: str, user_input: str) -> Dict[str, Any]:
        """Build the structured classification for the identified intent."""
        base = dict(
            _INTENT_CLASSIFICATION.get(intent, _INTENT_CLASSIFICATION["GENERAL"])
        )
        base["summary"] = user_input
        base["involves_execution"] = base.get("category") == "EXECUTION"
        return base

    def _identify_domains(self, classification: Dict[str, Any]) -> List[str]:
        """Return the ordered list of domain capabilities required."""
        return list(classification.get("domains", []))

    def _map_to_specialists(self, domains: List[str]) -> List[Any]:
        """Map required domains onto registered specialists."""
        mapped: List[Any] = []
        seen: set = set()
        for domain in domains:
            specialist_name = _DOMAIN_TO_SPECIALIST.get(domain)
            specialist = self._specialists.get(specialist_name or "")
            if specialist is not None and specialist.name not in seen:
                mapped.append(specialist)
                seen.add(specialist.name)
        # Fallback: when no domain matches (e.g. a generic request) run every
        # registered specialist so their findings still feed the synthesis step.
        if not mapped:
            for specialist in self._specialists.values():
                if specialist.name not in seen:
                    mapped.append(specialist)
                    seen.add(specialist.name)
        return mapped

    def _check_permission_for_task(self, classification: Dict[str, Any]) -> bool:
        """True when the current permission level clears the task requirement."""
        required = int(classification.get("required_level", 0))
        return self.permission_manager.check_permission(required)

    def _build_execution_plan(
        self,
        specialists: List[Any],
        classification: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Decide which specialists run in parallel and which depend on each other.

        Explicit pipelines (e.g. ``market -> risk``) run sequentially; everything
        else is independent and runs concurrently.
        """
        names = [spec.name for spec in specialists]
        context = context or {}
        # A request may force specific specialists regardless of intent routing.
        forced = [n for n in context.get("force", []) if n in names]
        pipeline = classification.get("pipeline") or []
        dependent = [s for s in pipeline if _DOMAIN_TO_SPECIALIST.get(s) in names]
        parallel = [s for s in names if s not in dependent]
        parallel += forced
        return {
            "parallel": parallel,
            "dependent": dependent,
            "tasks": {name: {"domain": self._domain_for(name)} for name in names},
        }

    async def _run_specialists_parallel(
        self,
        specialists: List[Any],
        execution_plan: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute independent specialists concurrently and collect their output."""
        results: Dict[str, Any] = {}
        targets = execution_plan.get("parallel", [])
        tasks: Dict[str, asyncio.Task] = {}

        for spec in specialists:
            if spec.name not in targets:
                continue
            tasks[spec.name] = asyncio.create_task(
                self._run_one(spec, execution_plan["tasks"].get(spec.name, {}))
            )

        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception:  # noqa: BLE001 - record and continue
                logger.exception("specialist_failed", specialist=name)
                results[name] = {"specialist": name, "status": "FAILED"}
        return results

    async def _run_specialists_sequential(
        self,
        specialists: List[Any],
        execution_plan: Dict[str, Any],
        current_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute dependent specialists in order, chaining their results."""
        by_name = {spec.name: spec for spec in specialists}
        for domain in execution_plan.get("dependent", []):
            name = _DOMAIN_TO_SPECIALIST.get(domain)
            spec = by_name.get(name)
            if spec is None:
                continue
            context = {"upstream": current_results}
            try:
                current_results[name] = await self._run_one(
                    spec, execution_plan["tasks"].get(name, {}), context
                )
            except Exception:  # noqa: BLE001
                logger.exception("specialist_failed_sequential", specialist=name)
                current_results[name] = {"specialist": name, "status": "FAILED"}
        return current_results

    @staticmethod
    async def _run_one(
        specialist: Any, task: Dict[str, Any], context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Invoke a single specialist and normalize its output."""
        raw = await specialist.run(task=task, context=context or {})
        if isinstance(raw, dict) and "specialist" not in raw:
            raw = {"specialist": specialist.name, **raw}
        return raw

    def _collect_results(
        self, specialist_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Normalize specialist results into a list of dicts."""
        return [
            dict(v) if isinstance(v, dict) else {"specialist": k, "value": v}
            for k, v in specialist_results.items()
        ]

    def _check_conflicts(
        self, collected_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect opposing directional claims between specialists."""
        directions = [
            r.get("direction")
            for r in collected_results
            if isinstance(r, dict) and r.get("direction") in ("LONG", "SHORT")
        ]
        conflict = len(set(directions)) > 1
        return {
            "conflict_detected": conflict,
            "directions": directions,
            "resolved": not conflict,
            "collective": collected_results,
        }

    def _apply_safety_gates(
        self,
        conflict_resolution: Dict[str, Any],
        classification: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Enforce permissions and conflict gates before synthesis.

        Execution is blocked when the permission level is insufficient or when
        the specialists produce a live-trade recommendation without authorization.
        """
        gates: List[Dict[str, Any]] = []
        passed = True

        if classification.get("involves_execution"):
            if not self.permission_manager.can_live_trade:
                passed = False
                gates.append(
                    {
                        "gate": "live_execution_permission",
                        "passed": False,
                        "reason": "Live trading requires permission level 4",
                    }
                )
        if conflict_resolution.get("conflict_detected"):
            passed = False
            gates.append(
                {
                    "gate": "specialist_conflict",
                    "passed": False,
                    "reason": "Specialists disagreed on direction",
                }
            )

        gates.append(
            {
                "gate": "permission_level",
                "passed": self._check_permission_for_task(classification),
                "reason": f"required={classification.get('required_level')}",
            }
        )

        return {
            "passed": passed and gates[-1]["passed"],
            "gates": gates,
            "collective": conflict_resolution.get("collective", []),
        }

    def _synthesize_result(
        self,
        safety_result: Dict[str, Any],
        classification: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine valid specialist findings into one coherent result."""
        collective = safety_result.get("collective", [])
        findings = [r for r in collective if r.get("status") in (None, "SUCCESS")]
        summary_parts = []
        for finding in findings:
            summary_parts.append(finding.get("summary") or finding.get("verdict"))
        summary = (
            " | ".join([p for p in summary_parts if p])
            or "No specialist output available."
        )

        direction = None
        directions = [f.get("direction") for f in findings if f.get("direction")]
        if directions and len(set(directions)) == 1:
            direction = directions[0]

        return {
            "summary": summary,
            "direction": direction,
            "findings": findings,
            "gates": safety_result.get("gates", []),
            "data_complete": len(directions) > 0 or not findings,
        }

    def _classify_final_state(self, final_result: Dict[str, Any]) -> str:
        """Label the outcome of the orchestration run."""
        if final_result.get("gates") and any(
            not g["passed"] for g in final_result["gates"]
        ):
            return "BLOCKED"
        if not final_result.get("findings"):
            return "NO_SPECIALISTS_AVAILABLE"
        if final_result.get("direction"):
            return "COMPLETED"
        return "PARTIAL"

    async def _store_knowledge(
        self,
        final_result: Dict[str, Any],
        classification: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Emit a knowledge-storage event when an event bus is configured."""
        if self.event_bus is None:
            return
        event = Event(
            event_type="orchestrator.result.stored",
            aggregate_id="orchestrator",
            aggregate_type="request",
            payload={
                "category": classification.get("category"),
                "summary": final_result.get("summary"),
                "findings": final_result.get("findings"),
            },
            source_agent="orchestrator",
        )
        try:
            await self.event_bus.publish(event)
        except Exception:  # noqa: BLE001 - never let knowledge storage fail a request
            logger.exception("knowledge_store_failed")

    def _domain_for(self, specialist_name: str) -> str:
        """Map a specialist name back to its routing domain."""
        for domain, name in _DOMAIN_TO_SPECIALIST.items():
            if name == specialist_name:
                return domain
        return "general"
