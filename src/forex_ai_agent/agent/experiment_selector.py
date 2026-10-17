"""LLM experiment selection (Stage C2) — the LLM chooses WHAT TO TEST.

Roadmap Stage C: "Director gains LLM-driven hypothesis generation and
experiment selection over the B4 deterministic skeleton." C1 chose what to
research (hypotheses into the Director queue); C2 chooses which EXPERIMENT to
run next on an existing strategy — over the deterministic machinery that
already exists (Phase 3 ``backtest_strategy`` / ``walk_forward_test``).

Constitution applied here (mirrors ``agent/hypothesis_gen.py``):
- The LLM only ever sees deterministic candidate menus (strategy names,
  parameter ranges, what has been tested before — no opinions) and only ever
  produces CANDIDATE experiment proposals (strategy id + parameter
  mutations). It computes no number anyone acts on.
- Deterministic validation before anything runs: strategy must exist,
  every mutation must hit the strategy's declared parameter ranges,
  the run's own deterministic baseline candidate is always included, budget
  is capped (max_experiments ≤ 3, mutations ≤ 6, OOS windows ≤ 4).
- The deterministic fallback proposes a baseline experiment from the same
  menu — the loop never stalls on credits (B4's promise, carried into C).
- Hard blocks by construction: this module only *runs the existing research
  tools* — no orders, no risk mutation, no permission changes, no version
  creation (source-scanned by a test, like B4/C1).
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

EXPERIMENT_SELECT_SCHEMA_VERSION = 1

_MAX_EXPERIMENTS = 3
_MAX_MUTATIONS = 6
_MAX_OOS_WINDOWS = 4
_MAX_STRATEGIES_PER_RUN = 8

_LLM_SYSTEM_PROMPT = (
    "You are Alex's experiment selector. You receive a deterministic menu of "
    "forex strategies (names, declared parameter ranges, research history) and "
    "the parameters that have been tested before. Choose which ONE experiment "
    "is most informative to run NEXT — avoid repeating tested parameters; "
    "prefer filling gaps in parameter space or the least-recently-tested "
    "strategy. You choose WHAT TO TEST ONLY: every experiment is executed by "
    "deterministic code (real backtests + walk-forward), every number comes "
    "from the engine. Never invent metrics. Respond with the JSON described."
)

_LLM_SCHEMA_HINT = (
    '{"experiments": [{"strategy_id": "id from the menu", '
    '"rationale": "why this experiment, now (<=300 chars)", '
    '"parameter_mutations": {"param": numeric_value}, '
    '"confidence": 0.75}]}'
)


@dataclass
class ExperimentSelConfig:
    """Budget + scope caps for one selection run. All deterministic."""

    strategies: Optional[List[str]] = None   # ids; None → all saved strategies
    timeframe: str = "H1"
    start_date: str = "2025-01-01"
    end_date: str = "2025-12-31"
    symbol: str = "EUR/USD"                  # run-scoped test symbol
    max_experiments: int = 1                 # actually executed this run
    max_variants: int = 4                    # accepted proposals per strategy
    window_months: int = 6
    step_months: int = 2
    max_oos_windows: int = 3                 # budget cap applied to WF runs
    use_llm: bool = True
    dry_run: bool = False                    # True → validate proposals, execute nothing


@dataclass
class ExperimentSelReport:
    """What one selection run did — every number traceable to a record."""

    generated_at: str
    timeframe: str
    period: str
    strategies_seen: int = 0
    llm_used: bool = False
    llm_error: Optional[str] = None
    fallback_used: bool = False
    raw_proposals: int = 0
    accepted: List[Dict[str, Any]] = field(default_factory=list)
    rejected: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def disclaimer(self) -> str:
        return (
            "The LLM chose what to test; deterministic code executed every "
            "experiment and owns every number. Research only — no live "
            "execution, no risk changes."
        )

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["disclaimer"] = self.disclaimer
        return data


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def _spec_error(strategy: Any) -> Optional[str]:
    """None if the strategy's rules parse into a runnable spec, else the reason."""
    try:
        from forex_ai_agent.strategy.spec import parse_spec

        _, errors = parse_spec(getattr(strategy, "rules", None))
        return "; ".join(errors[:3]) if errors else None
    except Exception as exc:  # noqa: BLE001 — a broken spec module must not kill the menu
        return f"{type(exc).__name__}: {exc}"


def _clamp_param(
    name: str, raw: Any, ranges: Dict[str, Tuple[float, float]]
) -> Optional[float]:
    """Deterministic clamp of one LLM mutation to the strategy's declared range."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if name not in ranges:
        return None
    lo, hi = ranges[name]
    return round(max(lo, min(hi, value)), 6)


# ---------------------------------------------------------------------------
# Deterministic menu builder — the ONLY thing the LLM sees
# ---------------------------------------------------------------------------


def build_menu_entry(
    strategy: Any,
    experiments: List[Dict[str, Any]],
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Dict[str, Any]]:
    """One menu entry: identity, declared ranges, tested-before — facts only."""
    try:
        sid = str(getattr(strategy, "id", "") or "")
        name = str(getattr(strategy, "name", "") or "")
    except Exception:  # noqa: BLE001 — defensive: menu must never crash
        return None
    if not sid or not name:
        return None
    params = dict(getattr(strategy, "parameters", None) or {})
    ranges = dict(param_ranges or {})
    if not ranges:
        ranges = {
            str(k): (float(v) * 0.5, float(v) * 1.5)
            for k, v in params.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
    tested = [
        {
            "parameter_changes": e.get("parameter_changes"),
            "verdict": e.get("verdict"),
            "created_at": e.get("created_at"),
        }
        for e in (experiments or [])[-5:]
    ]
    return {
        "strategy_id": sid,
        "name": name,
        "params": {str(k): v for k, v in params.items()},
        "param_ranges": {k: list(v) for k, v in ranges.items()},
        "tested_before": tested,
    }


def build_menu(
    strategies: List[Any],
    experiments_by_strategy: Dict[str, List[Dict[str, Any]]],
    param_ranges_by_strategy: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> List[Dict[str, Any]]:
    """Menu for many strategies; drops unusable entries."""
    ranges = param_ranges_by_strategy or {}
    out: List[Dict[str, Any]] = []
    for s in strategies:
        entry = build_menu_entry(s, experiments_by_strategy.get(str(getattr(s, "id", "")), []), ranges.get(str(getattr(s, "id", ""))))
        if entry is not None:
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Deterministic fallback — the run's own baseline candidate
# ---------------------------------------------------------------------------


def _fallback_proposals(menu: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Baseline experiment per strategy: run the strategy as-is once.

    Clearly marked ``source="baseline_fallback"`` — no LLM was involved. This
    is the run's own deterministic candidate (mirrors C1's template fallback).
    """
    out: List[Dict[str, Any]] = []
    for entry in menu:
        out.append(
            {
                "strategy_id": entry["strategy_id"],
                "rationale": (
                    "deterministic baseline from the menu (LLM unavailable): "
                    "establish the strategy's reference numbers before any mutation"
                ),
                "parameter_mutations": {},
                "confidence": None,
                "source": "baseline_fallback",
            }
        )
    return out


# ---------------------------------------------------------------------------
# Deterministic validation of one LLM proposal
# ---------------------------------------------------------------------------


def _validate_proposal(
    cand: Any,
    menu_by_id: Dict[str, Dict[str, Any]],
    config: ExperimentSelConfig,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate one LLM proposal → (normalized, reject_reason)."""
    if not isinstance(cand, dict):
        return None, "not a JSON object"
    sid = str(cand.get("strategy_id") or "").strip()
    if not sid:
        return None, "missing strategy_id"
    entry = menu_by_id.get(sid)
    if entry is None:
        return None, "strategy_id not in this run's menu"
    mutations = cand.get("parameter_mutations")
    if mutations is None:
        mutations = {}
    if not isinstance(mutations, dict):
        return None, "parameter_mutations must be an object"
    if len(mutations) > _MAX_MUTATIONS:
        return None, f"more than {_MAX_MUTATIONS} parameter mutations"
    clean: Dict[str, float] = {}
    for name, raw in mutations.items():
        clamped = _clamp_param(str(name), raw, entry["param_ranges"])
        if clamped is None:
            return None, f"mutation '{name}' not in the strategy's declared ranges"
        clean[str(name)] = clamped
    confidence = cand.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            return None, "confidence not a number"
        if not 0.0 <= confidence <= 1.0:
            return None, f"confidence {confidence} outside 0..1"
    rationale = str(cand.get("rationale") or "").strip()
    if len(rationale) > 300:
        rationale = rationale[:300]
    if len(rationale) < 10:
        return None, "rationale too short (<10 chars) — say why this experiment"
    # Provenance is honest: LLM proposals carry no source (→ "llm"), while
    # deterministic baseline fallbacks keep their tag (mirrors C1).
    source = str(cand.get("source") or "llm")
    if source not in ("llm", "baseline_fallback"):
        source = "llm"
    return {
        "strategy_id": sid,
        "rationale": rationale,
        "parameter_mutations": clean,
        "confidence": confidence,
        "source": source,
    }, None


# ---------------------------------------------------------------------------
# Executor — the existing deterministic Phase-3 machinery, verbatim
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Engine metrics carry numpy scalars — convert them to plain JSON types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj
    if hasattr(obj, "item"):
        try:
            return obj.item()  # numpy scalars → python scalars
        except Exception:  # noqa: BLE001
            return str(obj)
    return obj


async def _run_one(
    entry: Dict[str, Any],
    proposal: Dict[str, Any],
    config: ExperimentSelConfig,
) -> Dict[str, Any]:
    """Run one accepted proposal through the real deterministic engines.

    Mutations → ``create_experiment`` (baseline vs. mutated variant through
    the same backtest/walk-forward engines); no mutations → baseline
    walk-forward of the strategy as-is. Every number comes from the engine.
    """
    from forex_ai_agent.agent.experiments import create_experiment, walk_forward_test

    sid = proposal["strategy_id"]
    mutations = dict(proposal["parameter_mutations"])
    out: Dict[str, Any] = {
        "source": proposal.get("source") or "llm",
        "strategy_id": sid,
        "parameter_mutations": mutations,
    }
    try:
        if mutations:
            exp = await create_experiment(
                strategy_id=sid,
                param_mutations=mutations,
                symbol=config.symbol,
                timeframe=config.timeframe,
                start_date=config.start_date,
                end_date=config.end_date,
            )
            out["experiment"] = {
                "experiment_id": exp.get("experiment_id"),
                "verdict": exp.get("verdict"),
                "recommendation": exp.get("recommendation"),
                "baseline_metrics": exp.get("baseline_metrics"),
                "variant_count": exp.get("variant_count"),
            }
        else:
            wf = await walk_forward_test(
                sid,
                config.symbol,
                config.start_date,
                config.end_date,
                timeframe=config.timeframe,
                window_months=config.window_months,
                step_months=config.step_months,
                run_parameter_grid=False,
            )
            out["walk_forward"] = {
                "windows_tested": wf.get("windows_tested"),
                "windows": list(wf.get("windows") or [])[: config.max_oos_windows],
                "aggregate_metrics": wf.get("aggregate_metrics"),
                "stability_score": wf.get("stability_score"),
                "verdict": wf.get("verdict"),
                "budget_capped_to": config.max_oos_windows,
            }
        out["status"] = "RUN"
    except Exception as exc:  # noqa: BLE001 — one bad run must not stop the others
        out["status"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return _to_json_safe(out)


# ---------------------------------------------------------------------------
# One selection run
# ---------------------------------------------------------------------------


async def select_experiments(
    config: Optional[ExperimentSelConfig] = None,
    *,
    menu_loader: Optional[Callable[..., Any]] = None,
    llm: Optional[Any] = None,
    runner: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """One selection run: menu → LLM candidates → validation → execution.

    ``menu_loader()`` → (strategies, experiments_by_strategy) and ``runner``
    are injection seams (tests substitute fakes; production lazy-imports the
    real ones). Returns a JSON-safe report dict; never raises for LLM
    failures — an empty menu raises ``ValueError``. ``dry_run=True`` stops
    after validation: proposals are reported, nothing is executed.
    """
    config = config or ExperimentSelConfig()
    menu_loader = menu_loader or _default_menu_loader

    report = ExperimentSelReport(
        generated_at=_now_iso(), timeframe=config.timeframe, period=f"{config.start_date}..{config.end_date}"
    )

    # 1. Deterministic menu (facts only, no opinions).
    loaded = menu_loader(config)
    if inspect.isawaitable(loaded):
        loaded = await loaded
    strategies, experiments_by_strategy = loaded

    # Spec check FIRST (mirrors C1's dead-pair rejection): only strategies
    # whose rules actually parse can be selected — offering unrunnable ones
    # to the LLM would waste a proposal on a guaranteed error.
    runnable: List[Any] = []
    for s in strategies:
        if _spec_error(s) is None:
            runnable.append(s)
        else:
            report.rejected.append(
                {"strategy_id": str(getattr(s, "id", "?")), "reason": "strategy rules are not a runnable spec"}
            )

    menu = build_menu(runnable, experiments_by_strategy)[:_MAX_STRATEGIES_PER_RUN]
    report.strategies_seen = len(menu)
    if not menu:
        raise ValueError("no runnable strategies in the selection menu (all rules fail the spec check)")
    menu_by_id = {e["strategy_id"]: e for e in menu}

    # 2. Proposals: LLM first, deterministic fallback on any failure.
    proposals: List[Dict[str, Any]] = []
    if config.use_llm:
        if llm is None:
            from forex_ai_agent.llm.client import LLMClient

            llm = LLMClient()
        try:
            raw = await llm.reason_json(
                _LLM_SYSTEM_PROMPT,
                {"menu": menu, "period": report.period, "timeframe": config.timeframe},
                schema_hint=_LLM_SCHEMA_HINT,
            )
            raw_list = raw.get("experiments") if isinstance(raw, dict) else None
            if not isinstance(raw_list, list):
                raise ValueError("LLM response missing 'experiments' list")
            report.llm_used = True
            proposals = list(raw_list)
        except Exception as exc:  # noqa: BLE001 — credits/rate-limit/garbage JSON
            report.llm_error = f"{type(exc).__name__}: {exc}"
            logger.warning("experiment selection LLM failed: %s: %s", type(exc).__name__, exc)
    if not proposals:
        proposals = _fallback_proposals(menu)
        report.fallback_used = not report.llm_used
    report.raw_proposals = len(proposals)

    # 3. Deterministic validation, dedup, cap.
    validated: List[Dict[str, Any]] = []
    seen: set = set()
    for cand in proposals:
        normalized, reason = _validate_proposal(cand, menu_by_id, config)
        if reason is not None:
            report.rejected.append(
                {"strategy_id": str((cand or {}).get("strategy_id", "?")) if isinstance(cand, dict) else "?", "reason": reason}
            )
            continue
        key = (normalized["strategy_id"], tuple(sorted(normalized["parameter_mutations"].items())))
        if key in seen:
            report.rejected.append({"strategy_id": normalized["strategy_id"], "reason": "duplicate experiment proposal"})
            continue
        seen.add(key)
        validated.append(normalized)
    if len(validated) > config.max_experiments:
        for extra in validated[config.max_experiments:]:
            report.rejected.append({"strategy_id": extra["strategy_id"], "reason": "beyond max_experiments for this run"})
        validated = validated[: config.max_experiments]

    if config.dry_run:
        # Propose-only mode (mirrors C1's enqueue=False): validate and report
        # WITHOUT executing anything — no engines run, nothing is touched.
        for prop in validated:
            report.accepted.append(
                {
                    "source": prop["source"],
                    "strategy_id": prop["strategy_id"],
                    "parameter_mutations": prop["parameter_mutations"],
                    "rationale": prop["rationale"],
                    "confidence": prop["confidence"],
                    "status": "DRY_RUN_NOT_EXECUTED",
                }
            )
        return report.to_dict()

    # 4. Execution — existing deterministic tools; no version creation.
    results: List[Dict[str, Any]] = []
    for prop in validated:
        entry = menu_by_id[prop["strategy_id"]]
        try:
            result = (
                await runner(entry, prop, config)
                if runner is not None
                else await _run_one(entry, prop, config)
            )
        except Exception as exc:  # noqa: BLE001 — one bad run must not stop the report
            result = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}
        result = dict(result)
        result.setdefault("source", prop.get("source") or "llm")
        result["rationale"] = prop["rationale"]
        result["confidence"] = prop["confidence"]
        results.append(result)
        report.accepted.append(result)

    return report.to_dict()


async def _default_menu_loader(
    config: ExperimentSelConfig,
) -> Tuple[List[Any], Dict[str, List[Dict[str, Any]]]]:
    """The real menu source: saved strategies + their experiment history."""
    from forex_ai_agent.agent.experiments import list_experiments
    from forex_ai_agent.strategy.versioning import create_version_manager

    vm = await create_version_manager()
    strategies = [
        s
        for s in vm.list_versions()
        if getattr(s, "status", "draft") not in ("archived", "deprecated")
    ]
    experiments_by_strategy: Dict[str, List[Dict[str, Any]]] = {}
    for s in strategies:
        try:
            experiments_by_strategy[s.id] = await list_experiments(strategy_id=s.id, limit=10)
        except Exception:  # noqa: BLE001 — a broken history must not kill the menu
            experiments_by_strategy[s.id] = []
    return strategies, experiments_by_strategy
