"""LLM research reports (Stage C3) — the LLM NARRATES, records own every number.

Roadmap Stage C accept: "end-to-end unattended research session produces a
written report whose every number traces to a stored, reproducible record
(A1 pays off here)."

Constitution applied here (mirrors C1/C2):
- The LLM only ever sees a deterministic EVIDENCE CATALOG — every entry is an
  id plus facts and numbers read straight from stored records (experiments,
  research history, probation, strategy health). It produces narration only:
  a title, a summary and sections that REFERENCE evidence ids.
- Deterministic code renders the report document: the numbers ledger IS the
  evidence catalog, never LLM text. Every narrative section is traced: any
  number in the prose that cannot be found in the cited evidence (directly or
  via the evidence it names) is untraceable — the section is replaced with a
  deterministic summary and the violation is recorded honestly.
- The deterministic fallback writes every section from the same ledger — the
  report never stalls on credits (B4's promise, carried into C).
- Hard blocks by construction: this module only READS stored records and
  writes its own report row — no orders, no risk mutation, no permission
  changes, no experiment execution (source-scanned by a test).
"""

from __future__ import annotations

import inspect
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from sqlalchemy import String, select
from sqlalchemy.orm import Mapped, mapped_column

from forex_ai_agent.data.database import get_session_factory
from forex_ai_agent.data.models.base import (
    Base,
    JSONType,
    TimestampMixin,
    UUIDPkMixin,
)

logger = logging.getLogger(__name__)

REPORT_SCHEMA_VERSION = 1
_MAX_EVIDENCE = 40
_SMALL_INT = 99  # prose ints ≤ this (counts, list lengths) are not policed

_LLM_SYSTEM_PROMPT = (
    "You are Alex's research report writer. You receive a deterministic "
    "EVIDENCE CATALOG: entries with an id, a kind, a one-line summary and "
    "numbers read directly from stored records. Write the report: a title, a "
    "short summary, and sections that CATALOGUE evidence ids and narrate what "
    "they show. RULES: cite only ids from the catalog; never invent numbers — "
    "the only numbers allowed in your prose are ones visible in the cited "
    "evidence; no advice, no trading recommendations beyond what the records "
    "state (verdicts, recommendations stored on the records). Respond with "
    "the JSON described."
)

_LLM_SCHEMA_HINT = (
    '{"title": "short report title", "summary": "3-4 sentence overview", '
    '"sections": [{"heading": "section heading", "narrative": "what the cited '
    'evidence shows, prose", "evidence_ids": ["exp:...", "prob:..."]}]}'
)


# ---------------------------------------------------------------------------
# Model — research_reports (create_all-managed, like research_plans)
# ---------------------------------------------------------------------------


class ResearchReport(UUIDPkMixin, TimestampMixin, Base):
    """One generated research report (the written-findings record)."""

    __tablename__ = "research_reports"

    topic: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="COMPLETE", index=True)
    evidence_count: Mapped[int] = mapped_column(nullable=False, default=0)
    llm_used: Mapped[bool] = mapped_column(nullable=False, default=False)
    report_json: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    schema_version: Mapped[int] = mapped_column(nullable=False, default=REPORT_SCHEMA_VERSION)


async def ensure_report_schema(session_factory: Optional[Any] = None) -> Dict[str, Any]:
    """Create ``research_reports`` if the DB predates C3. Idempotent."""
    factory = session_factory or get_session_factory()
    created = False
    async with factory() as session:
        conn = await session.connection()

        def _ensure(sync_conn: Any) -> None:
            nonlocal created
            before = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            Base.metadata.create_all(sync_conn, tables=[ResearchReport.__table__])
            after = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            created = bool(after - before)

        await conn.run_sync(_ensure)
    return {"created": created, "table": ResearchReport.__tablename__}


# ---------------------------------------------------------------------------
# Config / report dataclass
# ---------------------------------------------------------------------------


@dataclass
class ReportConfig:
    """Scope + budget caps for one report run. All deterministic."""

    topic: str = "Research summary"
    experiment_ids: Optional[List[str]] = None   # None → recent experiments
    strategy_ids: Optional[List[str]] = None     # filter experiments/health
    limit_experiments: int = 10                  # recency cap
    use_llm: bool = True
    persist: bool = True                         # False → build only, write nothing


# ---------------------------------------------------------------------------
# Evidence: the ONLY thing the LLM (and the reader) can number-cite
# ---------------------------------------------------------------------------


@dataclass
class EvidenceItem:
    """One stored record, trimmed to citable facts and numbers."""

    id: str                 # e.g. "exp:ab12cd34", "prob:9a8b...", "health:f39b..."
    kind: str               # experiment | research_task | probation | health
    summary: str            # one line, facts only
    metrics: Dict[str, float] = field(default_factory=dict)
    facts: Dict[str, Any] = field(default_factory=dict)


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if f == f else None  # NaN guard
    except (TypeError, ValueError):
        return None


def _extract_metrics(prefix: str, payload: Dict[str, Any], limit: int = 12) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in (payload or {}).items():
        n = _num(v)
        if n is not None and not isinstance(v, bool):
            out[f"{prefix}{k}"] = n
        if len(out) >= limit:
            break
    return out


def build_evidence_pack(
    experiments: List[Dict[str, Any]],
    research_tasks: List[Dict[str, Any]],
    probation: List[Dict[str, Any]],
    health_by_strategy: Dict[str, Dict[str, Any]],
) -> Dict[str, EvidenceItem]:
    """Trim stored records into citable evidence items, deterministically."""
    pack: Dict[str, EvidenceItem] = {}

    for exp in experiments[:_MAX_EVIDENCE]:
        eid = str(exp.get("id") or "").strip()
        if not eid:
            continue
        metrics = {}
        metrics.update(_extract_metrics("baseline_", exp.get("baseline_metrics") or {}, 8))
        metrics.update(_extract_metrics("variant_", exp.get("variant_metrics") or {}, 8))
        changes = exp.get("parameter_changes") or {}
        pack[f"exp:{eid}"] = EvidenceItem(
            id=f"exp:{eid}",
            kind="experiment",
            summary=(
                f"experiment {eid} on strategy {exp.get('strategy_id', '?')}: "
                f"verdict {exp.get('verdict', '?')}"
            ),
            metrics=metrics,
            facts={
                "strategy_id": str(exp.get("strategy_id") or ""),
                "verdict": str(exp.get("verdict") or ""),
                "recommendation": str(exp.get("recommendation") or "")[:200],
                "parameter_changes": changes,
            },
        )

    for task in research_tasks[:_MAX_EVIDENCE]:
        tid = str(task.get("id") or "").strip()
        if not tid:
            continue
        pack[f"task:{tid}"] = EvidenceItem(
            id=f"task:{tid}",
            kind="research_task",
            summary=f"research task {tid}: {task.get('topic', '?')} — {task.get('status', '?')}",
            metrics={},
            facts={
                "topic": str(task.get("topic") or ""),
                "status": str(task.get("status") or ""),
                "verdict": str(task.get("verdict") or ""),
                "conclusion": str(task.get("conclusion") or "")[:300],
            },
        )

    for rec in probation[:_MAX_EVIDENCE]:
        vid = str(rec.get("version_id") or "").strip()
        if not vid:
            continue
        observed = _num(rec.get("observed_trades")) or 0.0
        required = _num(rec.get("required_trades")) or 0.0
        pack[f"prob:{vid}"] = EvidenceItem(
            id=f"prob:{vid}",
            kind="probation",
            summary=(
                f"probation {vid} ({rec.get('pair', '?')}): {rec.get('status', '?')}, "
                f"{rec.get('observed_trades', 0)}/{rec.get('required_trades', 0)} trades"
            ),
            metrics={
                "observed_trades": observed,
                "required_trades": required,
                "progress": (observed / required) if required else 0.0,
            },
            facts={
                "pair": str(rec.get("pair") or ""),
                "status": str(rec.get("status") or ""),
                "eligible_for_approval": bool(rec.get("eligible_for_approval")),
            },
        )

    for sid, card in list(health_by_strategy.items())[:_MAX_EVIDENCE]:
        if not card:
            continue
        badge = str(card.get("badge") or card.get("overall") or "")
        pack[f"health:{sid}"] = EvidenceItem(
            id=f"health:{sid}",
            kind="health",
            summary=f"health card for strategy {sid}" + (f": {badge}" if badge else ""),
            metrics=_extract_metrics("", card, 8),
            facts={"badge": badge},
        )
    return pack


def render_evidence_catalog(pack: Dict[str, EvidenceItem]) -> List[Dict[str, Any]]:
    """The LLM-facing catalog: ids, kinds, summaries, numbers. Facts only."""
    return [
        {"id": it.id, "kind": it.kind, "summary": it.summary, "metrics": dict(it.metrics)}
        for it in pack.values()
    ]


# ---------------------------------------------------------------------------
# Traceability: every prose number must come from the evidence
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _matches_evidence(num: float, evidence: List[EvidenceItem]) -> bool:
    """A number traces if some evidence metric/fact equals it (2dp) or the
    prose value is a prefix of the stored value (0.76 vs 0.7612…)."""
    for it in evidence:
        candidates = list(it.metrics.values())
        candidates.extend(
            v for v in (it.facts.get("parameter_changes") or {}).values()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        )
        for stored in candidates:
            if abs(stored - num) < 0.005:
                return True
            if num != 0 and f"{num:.6f}".rstrip("0").rstrip(".") in f"{stored:.6f}":
                return True
    return False


def trace_narrative(
    narrative: str, evidence: List[EvidenceItem]
) -> Tuple[bool, List[str]]:
    """Check every non-trivial number in the prose against the cited evidence.

    Small ints (≤ _SMALL_INT) and years are allowed (list counts, dates);
    anything bigger or decimal must trace to stored numbers.
    """
    issues: List[str] = []
    for raw in _NUM_RE.findall(narrative or ""):
        try:
            num = float(raw)
        except ValueError:  # pragma: no cover — regex guarantees floats
            continue
        if num.is_integer() and abs(num) <= _SMALL_INT:
            continue
        if 1900 <= num <= 2100:
            continue  # years
        if not _matches_evidence(num, evidence):
            issues.append(raw)
    return (not issues), issues


# ---------------------------------------------------------------------------
# Deterministic fallback narration
# ---------------------------------------------------------------------------


def _fallback_section(kind: str, items: List[EvidenceItem]) -> Tuple[str, List[str]]:
    """Deterministic summary from the ledger — no LLM, no invented numbers."""
    if not items:
        return f"No {kind.replace('_', ' ')} records in scope.", []
    lines = [f"{len(items)} {kind.replace('_', ' ')} record(s) in scope:"]
    for it in items[:6]:
        metrics = ", ".join(f"{k}={v:g}" for k, v in sorted(it.metrics.items())[:4])
        lines.append(f"- {it.summary}" + (f" [{metrics}]" if metrics else ""))
    return "\n".join(lines), [it.id for it in items[:6]]


def _fallback_document(pack: Dict[str, EvidenceItem], config: ReportConfig) -> Dict[str, Any]:
    by_kind: Dict[str, List[EvidenceItem]] = {}
    for it in pack.values():
        by_kind.setdefault(it.kind, []).append(it)
    sections = []
    for kind in ("experiment", "research_task", "probation", "health"):
        items = by_kind.get(kind) or []
        narrative, ids = _fallback_section(kind, items)
        sections.append(
            {"heading": kind.replace("_", " ").capitalize(), "narrative": narrative,
             "source": "fallback", "evidence_ids": ids, "trace_issues": []}
        )
    return {"title": config.topic, "summary": "Deterministic report (LLM unavailable).", "sections": sections}


# ---------------------------------------------------------------------------
# Report document assembly — code owns the numbers ledger
# ---------------------------------------------------------------------------


def _resolve_citations(
    raw_ids: Any, pack: Dict[str, EvidenceItem]
) -> Tuple[List[str], List[str]]:
    """Map LLM-cited ids onto pack ids: exact, then unique prefix (≥3 chars).

    Live ledger ids can be long UUIDs (``task:20fcd565-7400-…``) — the LLM
    reasonably cites the short form; anything unresolvable is reported back.
    """
    resolved: List[str] = []
    unknown: List[str] = []
    for raw in raw_ids if isinstance(raw_ids, list) else []:
        rid = str(raw or "").strip()
        if not rid:
            continue
        if rid in pack:
            if rid not in resolved:
                resolved.append(rid)
            continue
        matches = [k for k in pack if k.startswith(rid)] if len(rid) >= 3 else []
        if len(matches) == 1:
            if matches[0] not in resolved:
                resolved.append(matches[0])
        else:
            unknown.append(rid)
    return resolved, unknown


def build_report_document(
    pack: Dict[str, EvidenceItem],
    narrative: Dict[str, Any],
    llm_used: bool,
    llm_error: Optional[str],
    config: ReportConfig,
) -> Dict[str, Any]:
    by_kind: Dict[str, List[EvidenceItem]] = {}
    for it in pack.values():
        by_kind.setdefault(it.kind, []).append(it)

    sections_out: List[Dict[str, Any]] = []
    replaced = 0
    llm_sections = narrative.get("sections") if isinstance(narrative, dict) else None
    llm_sections = llm_sections if isinstance(llm_sections, list) else []

    known_ids = set(pack.keys())
    discarded: List[Dict[str, str]] = []

    # Resolve LLM citations ONCE (exact or unique prefix), then hand sections
    # out to kinds: a section belongs to the first canonical kind its resolved
    # evidence points at; it is consumed exactly once.
    resolved_sections: List[Dict[str, Any]] = []
    for s in llm_sections:
        if not isinstance(s, dict):
            continue
        resolved, unknown = _resolve_citations(s.get("evidence_ids"), pack)
        if not resolved:
            discarded.append(
                {
                    "heading": str(s.get("heading") or "")[:120],
                    "reason": (
                        "no known evidence ids"
                        + (f" (unresolved: {unknown})" if unknown else "")
                    ),
                }
            )
            continue
        resolved_sections.append(
            {"sec": s, "ids": resolved, "kinds": {pack[i].kind for i in resolved}}
        )
    consumed: set = set()

    for kind in ("experiment", "research_task", "probation", "health"):
        items = by_kind.get(kind) or []
        idx = next(
            (i for i, e in enumerate(resolved_sections) if i not in consumed and kind in e["kinds"]),
            None,
        )
        if idx is not None:
            consumed.add(idx)
            sec = resolved_sections[idx]["sec"]
            ids = resolved_sections[idx]["ids"]
            heading = str(sec.get("heading") or kind)[:120]
            text = str(sec.get("narrative") or "").strip()
            cited = [pack[i] for i in ids]
            # Same-kind records in scope may also appear in the prose: they
            # are stored values, not inventions (the traceability rule says so).
            trace_pool = cited + [it for it in items if it not in cited]
            # Identifier fragments are not numeric claims: mask cited/in-scope
            # ids (and their hex fragments) out of the prose before scanning.
            mask = set()
            for it in trace_pool:
                mask.add(it.id)
                if ":" in it.id:
                    mask.add(it.id.split(":", 1)[1])
            for raw in (sec.get("evidence_ids") or []):
                r = str(raw or "").strip()
                if r:
                    mask.add(r)
                    if ":" in r:
                        mask.add(r.split(":", 1)[1])
            masked = text
            for token in mask:
                if token:
                    masked = masked.replace(token, "")
            ok, issues = trace_narrative(masked, trace_pool)
            if not text or len(text) < 20:
                ok, issues = False, ["narrative too short"]
            if not ok:
                replaced += 1
                fb_text, fb_ids = _fallback_section(kind, items)
                sections_out.append(
                    {"heading": heading, "narrative": fb_text, "source": "fallback",
                     "evidence_ids": fb_ids, "trace_issues": issues}
                )
            else:
                # Provenance stays honest: sections arriving from the fallback
                # document keep their tag even when they match by evidence ids.
                src = "llm" if (llm_used and str(sec.get("source") or "") != "fallback") else "fallback"
                sections_out.append(
                    {"heading": heading, "narrative": text, "source": src,
                     "evidence_ids": ids, "trace_issues": []}
                )
        else:
            fb_text, fb_ids = _fallback_section(kind, items)
            sections_out.append(
                {"heading": kind.replace("_", " ").capitalize(), "narrative": fb_text,
                 "source": "fallback", "evidence_ids": fb_ids, "trace_issues": []}
            )

    doc = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "topic": config.topic,
        "generated_at": datetime.utcnow().isoformat(),
        "scope": {k: len(v) for k, v in by_kind.items()},
        "evidence": render_evidence_catalog(pack),
        "title": str(narrative.get("title") or config.topic)[:200] if isinstance(narrative, dict) else config.topic,
        "summary": str(narrative.get("summary") or "")[:1000] if isinstance(narrative, dict) else "",
        "llm_used": llm_used,
        "llm_error": llm_error,
        "sections": sections_out,
        "traceability": {
            "rule": (
                "every number in an [llm] section traces to the cited evidence "
                "(or other in-scope records of the same kind); untraceable "
                "sections are replaced by deterministic summaries"
            ),
            "sections_replaced": replaced,
            "discarded_sections": discarded,
        },
    }
    doc["disclaimer"] = (
        "The LLM only narrates; the numbers ledger above is rendered directly "
        "from stored records. Research only — no live execution, no risk changes."
    )
    return doc


def render_report_text(doc: Dict[str, Any]) -> str:
    """Deterministic text rendering of the report document."""
    lines = [f"RESEARCH REPORT — {doc.get('topic', '')}", doc.get("title", "")]
    if doc.get("summary"):
        lines.append(doc["summary"])
    for sec in doc.get("sections") or []:
        lines.append(f"\n== {sec.get('heading', '')} [{sec.get('source', '?')}] ==")
        lines.append(sec.get("narrative", ""))
    ledger = doc.get("evidence") or []
    if ledger:
        lines.append("\n== Evidence ledger (stored records) ==")
        for e in ledger:
            nums = ", ".join(f"{k}={v:g}" for k, v in sorted((e.get("metrics") or {}).items())[:4])
            lines.append(f"- {e.get('id')} · {e.get('summary')}" + (f" [{nums}]" if nums else ""))
    lines.append(f"\n({doc.get('disclaimer', '')})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# One report run
# ---------------------------------------------------------------------------


async def generate_research_report(
    config: Optional[ReportConfig] = None,
    *,
    evidence_loader: Optional[Callable[..., Any]] = None,
    llm: Optional[Any] = None,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """One report run: evidence pack → LLM narration → trace → persist.

    ``evidence_loader(config)`` and ``llm`` are injection seams (tests
    substitute fakes; production lazy-imports the real ones). Returns the
    JSON-safe report document; raises ``ValueError`` only when no evidence
    exists at all.
    """
    config = config or ReportConfig()
    evidence_loader = evidence_loader or _default_evidence_loader

    pack: Dict[str, EvidenceItem] = evidence_loader(config)
    if inspect.isawaitable(pack):
        pack = await pack
    if not pack:
        raise ValueError("no evidence records found — run experiments or research first")

    narrative: Dict[str, Any] = {}
    llm_used = False
    llm_error: Optional[str] = None
    if config.use_llm:
        if llm is None:
            from forex_ai_agent.llm.client import LLMClient

            llm = LLMClient()
        try:
            raw = await llm.reason_json(
                _LLM_SYSTEM_PROMPT,
                {
                    "topic": config.topic,
                    "evidence_catalog": render_evidence_catalog(pack),
                },
                schema_hint=_LLM_SCHEMA_HINT,
            )
            if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
                raise ValueError("LLM response missing 'sections' list")
            narrative = raw
            llm_used = True
        except Exception as exc:  # noqa: BLE001 — credits/rate-limit/garbage JSON
            llm_error = f"{type(exc).__name__}: {exc}"
            logger.warning("report generation LLM failed: %s: %s", type(exc).__name__, exc)
    if not llm_used:
        narrative = _fallback_document(pack, config)

    doc = build_report_document(pack, narrative, llm_used, llm_error, config)

    if config.persist:
        factory = session_factory or get_session_factory()
        async with factory() as session:
            row = ResearchReport(
                topic=config.topic[:200],
                status="COMPLETE",
                evidence_count=len(pack),
                llm_used=llm_used,
                report_json=_safe(doc),
            )
            session.add(row)
            await session.flush()
            doc["report_id"] = str(row.id)
            row.report_json = _safe(doc)
            await session.commit()
    return _safe(doc)


def _safe(value: Any) -> Any:
    """JSON round-trip so the document is storage-safe."""
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# Default evidence loader — stored records only, lazy imports
# ---------------------------------------------------------------------------


async def _default_evidence_loader(config: ReportConfig) -> Dict[str, EvidenceItem]:
    from forex_ai_agent.agent.experiments import list_experiments
    from forex_ai_agent.agent.tools import _list_probation, _list_research_history

    experiments: List[Dict[str, Any]] = []
    if config.experiment_ids:
        from forex_ai_agent.agent.experiments import get_experiment

        for eid in config.experiment_ids[:_MAX_EVIDENCE]:
            exp = await get_experiment(eid)
            if exp:
                experiments.append(exp)
    else:
        try:
            experiments = await list_experiments(strategy_id=None, limit=config.limit_experiments)
        except Exception:  # noqa: BLE001 — an empty runner must not kill the report
            experiments = []
    if config.strategy_ids:
        allowed = {str(s) for s in config.strategy_ids}
        experiments = [e for e in experiments if str(e.get("strategy_id")) in allowed]

    try:
        tasks = await _list_research_history(10)
    except Exception:  # noqa: BLE001
        tasks = []
    try:
        probation = await _list_probation()
    except Exception:  # noqa: BLE001
        probation = []

    health_by_strategy: Dict[str, Dict[str, Any]] = {}
    from forex_ai_agent.agent.health import get_strategy_health

    sids = set(config.strategy_ids or [])
    if not sids:
        sids = {str(e.get("strategy_id")) for e in experiments if e.get("strategy_id")}
    for sid in sorted(sids)[:8]:
        try:
            health_by_strategy[sid] = await get_strategy_health(sid)
        except Exception:  # noqa: BLE001 — a broken health card must not kill the report
            health_by_strategy[sid] = {}
    return build_evidence_pack(experiments, tasks, probation, health_by_strategy)


# ---------------------------------------------------------------------------
# Read APIs (cards from persisted rows, no re-runs)
# ---------------------------------------------------------------------------


def _row_to_doc(row: Any) -> Dict[str, Any]:
    doc = dict(row.report_json or {})
    doc.setdefault("report_id", str(row.id))
    doc.setdefault("topic", row.topic)
    doc.setdefault("generated_at", row.created_at.isoformat() if row.created_at else None)
    return doc


async def get_research_report(
    report_id: str, session_factory: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """One report card by id (prefix match like B5 plans)."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        rows = (
            (await session.execute(select(ResearchReport).order_by(ResearchReport.created_at.desc())))
            .scalars()
            .all()
        )
        key = str(report_id or "")
        matches = [r for r in rows if str(r.id).startswith(key)] if len(key) < 32 else []
        if len(key) >= 32:
            try:
                row = await session.get(ResearchReport, uuid.UUID(key))
                return _row_to_doc(row) if row else None
            except (TypeError, ValueError):
                return None
        if len(matches) != 1:
            return None
        return _row_to_doc(matches[0])


async def list_research_reports(
    limit: int = 20, session_factory: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """Report summaries (most recent first)."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(ResearchReport)
                    .order_by(ResearchReport.created_at.desc())
                    .limit(max(1, min(int(limit), 50)))
                )
            )
            .scalars()
            .all()
        )
        out = [
            {
                "report_id": str(r.id),
                "topic": r.topic,
                "status": r.status,
                "evidence_count": r.evidence_count,
                "llm_used": bool(r.llm_used),
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "title": (r.report_json or {}).get("title", ""),
            }
            for r in rows
        ]
    return out
