"""Research lessons (learning loop) — deterministic facts from past rejections.

The unattended session exposed the loop's weakest link: C1 proposes, the B4
Director's backtest gate rejects, and the next C1 run proposes the same
shapes again. This module closes that loop WITHOUT giving the LLM any new
powers: it deterministically MINES stored ``ResearchTask`` rows (written by
``autonomy/research_log.py`` for every Director run) into compact lesson
facts — what the gate rejected, how often, and the spec shape that failed —
and C1 feeds them into its context as facts only.

Constitution mapping:
- The miner is pure deterministic code over stored records; the LLM cannot
  invent, edit or delete a lesson. Lessons carry no advice — counts, verbatim
  gate reasons, and spec shapes, exactly as stored.
- The backtest gate itself is untouched: lessons only bias WHAT C1 proposes
  next; every future proposal is still validated and gated identically.
Positive lessons extend the same loop to gate-PASSED runs: rows whose
``verdict == "validated"`` are mined into success facts — how often a spec
shape PASSED the gate and which strategy version it produced — so C1 can
lean toward shapes that worked, not just away from failures.

Constitution mapping:
- The miner is pure deterministic code over stored records; the LLM cannot
  invent, edit or delete a lesson. Lessons carry no advice — counts, verbatim
  gate reasons, and spec shapes, exactly as stored.
- The backtest gate itself is untouched: lessons only bias WHAT C1 proposes
  next; every future proposal is still validated and gated identically.
- Hard blocks by construction: read-only over ``research_tasks`` (no writes,
  no orders, no risk changes — source-scanned by a test).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from forex_ai_agent.data.database import get_session_factory

logger = logging.getLogger(__name__)

LESSONS_SCHEMA_VERSION = 2
WEAK_SHAPES_SCHEMA_VERSION = 1
_MAX_LESSONS = 8          # lessons fed to the LLM per run
_LOOKBACK_ROWS = 60       # recent ResearchTask rows scanned
_REJECT_VERDICT = "rejected"        # conclusion_json.verdict set by the research log
_VALIDATED_VERDICT = "validated"    # conclusion_json.verdict for gate-passed runs

# Canonical reason buckets — prefix-classified from the gate's verbatim
# reasons (ValidationReport.reasons, autonomy/validator.py).
_REASON_BUCKETS: Tuple[Tuple[str, str], ...] = (
    ("spec invalid", "spec_invalid"),
    ("backtest crashed", "backtest_crashed"),
)


@dataclass
class GateLesson:
    """One recurring rejected pattern, mined from stored runs."""

    count: int                    # how many rejected runs match this pattern
    reason_bucket: str            # spec_invalid | backtest_crashed | gate_threshold
    sample_reason: str            # one verbatim gate reason (≤200 chars)
    direction: str                # spec direction observed on those runs
    timeframe: str                # spec timeframe observed
    entry_verbs: List[str]        # sorted verbs used in entry_conditions


@dataclass
class SuccessLesson:
    """One recurring gate-PASSED pattern, mined from stored runs."""

    times_passed: int             # how many validated runs match this pattern
    direction: str                # spec direction observed on those runs
    timeframe: str                # spec timeframe observed
    entry_verbs: List[str]        # sorted verbs used in entry_conditions
    version_ids: List[str]        # strategy versions those runs produced


def _classify(reason: str) -> str:
    low = str(reason or "").lower()
    for prefix, bucket in _REASON_BUCKETS:
        if low.startswith(prefix):
            return bucket
    return "gate_threshold"


def _spec_shape(spec: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    direction = str((spec or {}).get("direction") or "?").lower()
    timeframe = str((spec or {}).get("timeframe") or "?").upper()
    verbs: List[str] = []
    for cond in (spec or {}).get("entry_conditions") or []:
        if isinstance(cond, dict) and cond.get("verb"):
            verbs.append(str(cond["verb"]))
    return direction, timeframe, sorted(set(verbs))


def spec_shape(spec: Dict[str, Any]) -> Tuple[str, str, List[str]]:
    """Public spec-shape key: (direction, timeframe, sorted unique verbs).

    The canonical grouping used by every C4 decision (weak-shape vetoes,
    shape-diff C1 nudges) and the success-lesson grouping — one definition,
    everywhere."""
    return _spec_shape(spec)


async def load_rejected_conclusions(
    lookback_rows: int = _LOOKBACK_ROWS,
    session_factory: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Raw conclusion dicts of the most recent REJECTED runs (for the C4
    weak-hypothesis veto). Read-only; never raises — failures yield []."""
    try:
        from forex_ai_agent.data.models.research import ResearchTask

        factory = session_factory or get_session_factory()

        async def _query() -> List[Dict[str, Any]]:
            async with factory() as session:
                rows = (
                    (
                        await session.execute(
                            select(ResearchTask)
                            .order_by(ResearchTask.created_at.desc())
                            .limit(max(1, min(int(lookback_rows), 200)))
                        )
                    )
                    .scalars()
                    .all()
                )
            return [
                c
                for c in (_row_conclusion(r) for r in rows)
                if c is not None and str(c.get("verdict") or "") == _REJECT_VERDICT
            ]

        return await _query()
    except Exception as exc:  # noqa: BLE001 — the veto must never break proposals
        logger.warning("rejected-conclusion loading failed: %s: %s", type(exc).__name__, exc)
        return []


def mine_gate_lessons(rows: List[Any]) -> List[GateLesson]:
    """Aggregate rejected runs into lessons. Deterministic; tolerant of junk.

    ``rows`` are ``ResearchTask``-like objects (or dicts with the same keys):
    only rows whose ``conclusion_json.verdict == "rejected"`` are mined; the
    pattern key is (reason_bucket, direction, timeframe, entry_verbs).
    """
    agg: Dict[Tuple[str, str, str, Tuple[str, ...]], GateLesson] = {}
    for row in rows:
        conclusion = getattr(row, "conclusion_json", None)
        if conclusion is None and isinstance(row, dict):
            conclusion = row.get("conclusion_json")
        if not isinstance(conclusion, dict):
            continue
        if str(conclusion.get("verdict") or "") != _REJECT_VERDICT:
            continue
        validation = conclusion.get("validation") or {}
        reasons = validation.get("reasons") if isinstance(validation, dict) else None
        reasons = [str(r) for r in reasons if str(r or "").strip()] if isinstance(reasons, list) else []
        if not reasons:
            continue  # a rejection without a stored reason teaches nothing
        direction, timeframe, verbs = _spec_shape(conclusion.get("spec") or {})
        for reason in reasons:
            bucket = _classify(reason)
            key = (bucket, direction, timeframe, tuple(verbs))
            lesson = agg.get(key)
            if lesson is None:
                lesson = GateLesson(
                    count=0,
                    reason_bucket=bucket,
                    sample_reason=str(reason)[:200],
                    direction=direction,
                    timeframe=timeframe,
                    entry_verbs=list(verbs),
                )
                agg[key] = lesson
            lesson.count += 1

    lessons = sorted(
        agg.values(),
        key=lambda l: (-l.count, l.reason_bucket, l.direction, l.timeframe, tuple(l.entry_verbs)),
    )
    return lessons[:_MAX_LESSONS]


def render_lessons_for_context(lessons: List[GateLesson]) -> Dict[str, Any]:
    """The LLM-facing lesson facts. Counts and stored values only — no advice
    beyond the deterministic instruction that failing shapes recur."""
    return {
        "note": (
            "Deterministic lessons mined from recently REJECTED runs (stored "
            "records). 'times_rejected' counts how often the backtest gate "
            "rejected this spec shape, 'reason_sample' is a verbatim gate "
            "reason. Prefer hypotheses unlikely to reproduce these exact "
            "failing shapes; never fabricate numbers about them."
        ),
        "lessons": [
            {
                "times_rejected": l.count,
                "reason_bucket": l.reason_bucket,
                "reason_sample": l.sample_reason,
                "spec_shape": {
                    "direction": l.direction,
                    "timeframe": l.timeframe,
                    "entry_verbs": l.entry_verbs,
                },
            }
            for l in lessons
        ],
    }


def _row_conclusion(row: Any) -> Optional[Dict[str, Any]]:
    """Extract a dict ``conclusion_json`` from a row-like object or dict."""
    conclusion = getattr(row, "conclusion_json", None)
    if conclusion is None and isinstance(row, dict):
        conclusion = row.get("conclusion_json")
    return conclusion if isinstance(conclusion, dict) else None


def _version_ids(conclusion: Dict[str, Any]) -> List[str]:
    """Strategy-version ids a validated run produced, newest first (≤3)."""
    ids: List[str] = []
    for key in ("version_id", "version_ids"):
        value = conclusion.get(key)
        if isinstance(value, str) and value.strip():
            ids.append(value.strip())
        elif isinstance(value, (list, tuple)):
            ids.extend(str(v).strip() for v in value if str(v or "").strip())
    return ids[:3]


def mine_success_lessons(rows: List[Any]) -> List[SuccessLesson]:
    """Aggregate gate-PASSED runs into success lessons. Deterministic; tolerant
    of junk.

    ``rows`` are ``ResearchTask``-like objects (or dicts with the same keys):
    only rows whose ``conclusion_json.verdict == "validated"`` are mined; the
    pattern key is (direction, timeframe, entry_verbs), mirroring the
    rejection grouping. ``times_passed`` counts validated RUNS (one per row —
    unlike rejections, a pass has no per-reason multiplicity); ``version_ids``
    lists the strategy versions those runs produced.
    """
    agg: Dict[Tuple[str, str, Tuple[str, ...]], SuccessLesson] = {}
    for row in rows:
        conclusion = _row_conclusion(row)
        if conclusion is None:
            continue
        if str(conclusion.get("verdict") or "") != _VALIDATED_VERDICT:
            continue
        direction, timeframe, verbs = _spec_shape(conclusion.get("spec") or {})
        key = (direction, timeframe, tuple(verbs))
        lesson = agg.get(key)
        if lesson is None:
            lesson = SuccessLesson(
                times_passed=0,
                direction=direction,
                timeframe=timeframe,
                entry_verbs=list(verbs),
                version_ids=[],
            )
            agg[key] = lesson
        lesson.times_passed += 1
        for vid in _version_ids(conclusion):
            if vid not in lesson.version_ids and len(lesson.version_ids) < 3:
                lesson.version_ids.append(vid)

    lessons = sorted(
        agg.values(),
        key=lambda l: (-l.times_passed, l.direction, l.timeframe, tuple(l.entry_verbs)),
    )
    return lessons[:_MAX_LESSONS]


def render_successes_for_context(lessons: List[SuccessLesson]) -> Dict[str, Any]:
    """The LLM-facing success facts. Stored values only — version ids and
    counts exactly as recorded, with a deterministic lean-toward note."""
    return {
        "note": (
            "Deterministic lessons mined from recently VALIDATED runs (stored "
            "records). 'times_passed' counts how often this spec shape PASSED "
            "the backtest gate; 'version_ids' lists the strategy versions it "
            "produced. Lean toward promising these shapes where the current "
            "briefs make them plausible — never copy numbers from them, and "
            "never fabricate backtest results."
        ),
        "lessons": [
            {
                "times_passed": l.times_passed,
                "spec_shape": {
                    "direction": l.direction,
                    "timeframe": l.timeframe,
                    "entry_verbs": l.entry_verbs,
                },
                "version_ids": list(l.version_ids),
            }
            for l in lessons
        ],
    }


async def load_rejected_run_lessons(
    lookback_rows: int = _LOOKBACK_ROWS,
    session_factory: Optional[Any] = None,
) -> List[GateLesson]:
    """Back-compat wrapper: rejection lessons only (the combined loader's
    rejection half). Never raises."""
    rejections, _ = await load_run_lessons(
        lookback_rows=lookback_rows, session_factory=session_factory
    )
    return rejections


async def load_run_lessons(
    lookback_rows: int = _LOOKBACK_ROWS,
    session_factory: Optional[Any] = None,
) -> Tuple[List[GateLesson], List[SuccessLesson]]:
    """Mine the most recent stored Director runs for BOTH polarities:
    (rejection lessons, success lessons). One row scan feeds both miners.
    Never raises — an empty or broken log yields ([], []) and C1 proceeds
    without lessons."""
    try:
        from forex_ai_agent.data.models.research import ResearchTask

        factory = session_factory or get_session_factory()
        async with factory() as session:
            rows = (
                (
                    await session.execute(
                        select(ResearchTask)
                        .order_by(ResearchTask.created_at.desc())
                        .limit(max(1, min(int(lookback_rows), 200)))
                    )
                )
                .scalars()
                .all()
            )
        rows = list(rows)
        return mine_gate_lessons(rows), mine_success_lessons(rows)
    except Exception as exc:  # noqa: BLE001 — lessons must never break C1
        logger.warning("research-lesson mining failed: %s: %s", type(exc).__name__, exc)
        return [], []


# ---------------------------------------------------------------------------
# Stage C4 — weak-shape policy (deterministic decisions over stored verdicts)
# ---------------------------------------------------------------------------


def _sample_reason(conclusions: List[Dict[str, Any]], shape: Tuple[str, str, Tuple[str, ...]]) -> str:
    """One verbatim gate reason for a shape, newest first (≤200 chars)."""
    probe = (shape[0], shape[1], list(shape[2]))  # spec_shape returns a verb LIST
    for c in conclusions:
        if spec_shape(c.get("spec") or {}) == probe:
            validation = c.get("validation") or {}
            reasons = validation.get("reasons") if isinstance(validation, dict) else None
            for r in reasons or []:
                text = str(r or "").strip()
                if text:
                    return text[:200]
            break
    return "gate criteria not met"


def weak_shapes(
    conclusions: List[Dict[str, Any]],
    *,
    min_attempts: int = 3,
    max_lessons: int = _MAX_LESSONS,
) -> List[GateLesson]:
    """C4 weak-hypothesis policy: shapes with ≥``min_attempts`` gate REJECTIONS
    in the stored record. Deterministic: grouped by the shared ``spec_shape``
    key, ordered by (count desc, shape), capped; verbatim sample reason per
    shape. These are the veto sources — numbers only, no advice.
    """
    counts: Dict[Tuple[str, str, Tuple[str, ...]], int] = {}
    for c in conclusions:
        direction, timeframe, verbs = spec_shape(c.get("spec") or {})
        key = (direction, timeframe, tuple(verbs))  # verbs must be hashable
        counts[key] = counts.get(key, 0) + 1
    lessons = [
        GateLesson(
            count=n,
            reason_bucket="gate_threshold",
            sample_reason=_sample_reason(conclusions, key),
            direction=key[0],
            timeframe=key[1],
            entry_verbs=list(key[2]),
        )
        for key, n in counts.items()
        if n >= min_attempts
    ]
    lessons.sort(key=lambda l: (-l.count, l.direction, l.timeframe, tuple(l.entry_verbs)))
    return lessons[:max(1, int(max_lessons))]


def veto_reason(lesson: GateLesson, attempts_seen: Optional[int] = None) -> str:
    """The verbatim-style reason stored when the veto fires (an honest "why"
    for the research log, the audit trail and the UI — not a silent skip)."""
    n = int(attempts_seen if attempts_seen is not None else lesson.count)
    return (
        f"weak-hypothesis veto: shape {lesson.direction}/{lesson.timeframe}/"
        f"{'+'.join(lesson.entry_verbs) or 'no-verbs'} rejected {n}x in the "
        f"stored record; sample gate reason: {lesson.sample_reason}"
    )


def shape_nudges(spec: Dict[str, Any], lessons: List[GateLesson], max_lessons: int = 2) -> List[str]:
    """Deterministic shape-diff nudges for C1's proposal context: which vetoed
    shapes the candidate most resembles (shared direction or timeframe or any
    shared verb), with the reason family that keeps failing. Stored facts
    only — never a forbidden list, never fabricated numbers."""
    d, tf, verbs = spec_shape(spec)
    verb_set = set(verbs)
    nudges: List[Tuple[int, str]] = []
    for l in lessons:
        shared_verbs = verb_set & set(l.entry_verbs)
        score = int(d == l.direction) + int(tf == l.timeframe) + len(shared_verbs)
        if score <= 0:
            continue
        nudges.append(
            (
                score,
                f"resembles weak shape {l.direction}/{l.timeframe}/"
                f"{'+'.join(l.entry_verbs) or 'no-verbs'} ({l.count} stored "
                f"rejections; sample reason: {l.sample_reason})",
            )
        )
    nudges.sort(key=lambda t: (-t[0], t[1]))
    return [text for _, text in nudges[:max(0, int(max_lessons))]]
