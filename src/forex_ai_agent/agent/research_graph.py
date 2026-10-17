"""Research graph edges (Stage B2) — SQLite, not a graph DB.

The roadmap asks for explicit, queryable research edges:

- ``Hypothesis.strategy_version_id`` — which saved strategy a hypothesis is about
- ``Experiment.hypothesis_id`` — enforced on the agent-experiment path
- edge rows for variant→outcome

Everything is derived from ALREADY-STORED records (nothing is re-run): agent
experiments carry ``parameter_changes``, per-variant metrics and a ``verdict``;
conclusions add a human ``decision``. The ``research_graph_edges`` table is the
link store — one row per variant tested, each carrying the hypothesis id
(when the experiment was run under one) and the strategy version id — so:

- "how many times have we tested this hypothesis?" → ``hypothesis_test_count``
- "what happens OOS when displacement > 0.25?" → ``aggregate_by_parameter``

The LLM may narrate these answers; only deterministic code computes them.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import String, select, text as sa_text
from sqlalchemy.orm import Mapped, mapped_column

from ..data.database import get_session_factory
from ..data.models.base import Base, JSONType, TimestampMixin, UUIDPkMixin

logger = logging.getLogger(__name__)

GRAPH_EDGE_SCHEMA_VERSION = 1

EDGE_KIND_EXPERIMENT = "variant_to_outcome"


class ResearchGraphEdge(UUIDPkMixin, TimestampMixin, Base):
    """One variant→outcome observation, linked to hypothesis/strategy.

    A "flat edge" rather than a relationship graph: rows are append-only
    observations that the aggregation queries below group and pivot. The
    agent experiment id is kept as a string so edges survive even if an
    experiment row is later purged. ``hypothesis_id`` is a free string (it
    accepts both DB hypotheses ids and research-pipeline hypothesis ids);
    ``strategy_version_id`` is the saved strategy the experiment mutated.
    """

    __tablename__ = "research_graph_edges"

    edge_kind: Mapped[str] = mapped_column(String(40), nullable=False, default=EDGE_KIND_EXPERIMENT)
    hypothesis_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    strategy_version_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    agent_experiment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    outcome: Mapped[str] = mapped_column(String(20), nullable=False)  # PASS/FAIL/INCONCLUSIVE or adopt/reject/defer
    parameter_changes: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    outcome_metrics: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    baseline_metrics: Mapped[dict] = mapped_column(JSONType, nullable=False, default=dict)
    reasons: Mapped[list] = mapped_column(JSONType, nullable=False, default=list)
    oos_sharpe: Mapped[Optional[float]] = mapped_column(nullable=True)
    schema_version: Mapped[int] = mapped_column(nullable=False, default=GRAPH_EDGE_SCHEMA_VERSION)


# ---------------------------------------------------------------------------
# Idempotent schema ensure (the live DB is create_all-managed; new columns on
# existing tables are added with ALTER TABLE, which SQLite supports additively)
# ---------------------------------------------------------------------------

_ENSURE_STATEMENTS: Tuple[Tuple[str, str], ...] = (
    # Future-proofing: give the existing tables the roadmap's link columns so
    # a later model release can map them without another migration step.
    ("hypotheses", "ALTER TABLE hypotheses ADD COLUMN strategy_version_id VARCHAR(64)"),
    ("agent_experiments", "ALTER TABLE agent_experiments ADD COLUMN hypothesis_id VARCHAR(64)"),
)


async def ensure_graph_schema(session_factory: Optional[Any] = None) -> Dict[str, Any]:
    """Create the edge table and add missing link columns, idempotently.

    Safe to call on every boot and from tests: existing data is untouched,
    repeated calls are no-ops. Returns what was done (for logging/tests).
    """
    from ..data.models.base import Base as ModelsBase

    factory = session_factory or get_session_factory()
    added_columns: List[str] = []
    edge_table_created = False

    async with factory() as session:
        conn = await session.connection()

        def _ensure(sync_conn: Any) -> None:
            nonlocal edge_table_created
            before = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            ModelsBase.metadata.create_all(sync_conn, tables=[ResearchGraphEdge.__table__])
            after = {
                row[0]
                for row in sync_conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            edge_table_created = "research_graph_edges" in after and "research_graph_edges" not in before
            for table, stmt in _ENSURE_STATEMENTS:
                if table not in after:
                    continue  # table itself missing → create_all owns it at boot
                cols = {
                    row[1]
                    for row in sync_conn.exec_driver_sql(f"PRAGMA table_info({table})").fetchall()
                }
                col_name = stmt.split("ADD COLUMN ", 1)[1].split(" ", 1)[0]
                if col_name not in cols:
                    sync_conn.exec_driver_sql(stmt)
                    added_columns.append(f"{table}.{col_name}")

        await conn.run_sync(_ensure)
        await session.commit()

    if edge_table_created or added_columns:
        logger.info(
            "research_graph_schema_ensured",
            edge_table_created=edge_table_created,
            columns=added_columns,
        )
    return {"edge_table_created": edge_table_created, "columns_added": added_columns}


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _safe(obj: Any) -> Any:
    """JSON-round-trip so numpy/decimal/uuid artifacts can't break JSON columns."""
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError):
        return {}


def _sharpe_from_metrics(metrics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not metrics:
        return None
    for key in ("sharpe_ratio", "sharpe", "oos_sharpe"):
        v = metrics.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


async def persist_agent_experiment(
    record: Any,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Persist one variant-runner experiment as graph edges (+ row when possible).

    ``record`` is an ``ExperimentRecord`` from ``VariantRunner.run_experiment``
    (its ``variant_changes`` map gives each variant's parameter mutations).
    The hypothesis link rides on every edge row when the experiment was run
    under one; unknown ids are kept verbatim — history is never dropped.

    Best-effort by design: returns a report dict, never raises — a logging
    failure must not break a research run.
    """
    factory = session_factory or get_session_factory()
    report: Dict[str, Any] = {"edges_written": 0, "experiment_row": None, "note": ""}
    try:
        hypothesis_id = (getattr(record, "hypothesis_id", None) or "").strip() or None
        variant_changes: Dict[str, Dict[str, Any]] = dict(
            getattr(record, "variant_changes", {}) or {}
        )
        base_metrics = dict(record.baseline_metrics or {})

        async with factory() as session:
            experiment_row_id: Optional[str] = None

            # The agent-experiments table has a NOT NULL user FK; the system
            # "alex" user (or any seeded user) satisfies it. No user → edges
            # only, recorded honestly in the report.
            from ..data.models.user import User
            from ..data.models.agent import AgentExperiment as AgentExperimentRow

            user_id = (
                await session.execute(select(User.id).where(User.username == "alex"))
            ).scalar_one_or_none()
            if user_id is None:
                user_id = (await session.execute(select(User.id).limit(1))).scalar_one_or_none()

            if user_id is not None:
                row = AgentExperimentRow(
                    user_id=user_id,
                    strategy_version_id=str(record.strategy_id)[:50],
                    base_version=str(record.base_version)[:20],
                    parameter_changes=_safe(dict(record.parameter_changes or {})),
                    dataset=_safe(dict(record.dataset or {})),
                    baseline_metrics=_safe(base_metrics),
                    variant_metrics=_safe(dict(record.variant_metrics or {})),
                    verdict=str(record.verdict)[:20],
                    reasons=_safe(list(record.reasons or [])),
                    recommendation=(str(record.recommendation) or None),
                )
                session.add(row)
                await session.flush()
                # SQLite's Uuid type stores CHAR(32) hex — str(uuid) would be
                # dashed and the UPDATE below would match nothing.
                experiment_row_id = row.id.hex if hasattr(row.id, "hex") else str(row.id)
                if hypothesis_id:
                    try:  # column exists after ensure_graph_schema (ALTER)
                        await session.execute(
                            sa_text(
                                "UPDATE agent_experiments SET hypothesis_id = :h WHERE id = :i"
                            ),
                            {"h": hypothesis_id, "i": experiment_row_id},
                        )
                    except Exception:  # noqa: BLE001 — advisory link only
                        pass
            else:
                report["note"] = "no user row available — edges only, no experiment row"

            for variant_id, metrics in dict(record.variant_metrics or {}).items():
                edge = ResearchGraphEdge(
                    edge_kind=EDGE_KIND_EXPERIMENT,
                    hypothesis_id=hypothesis_id,
                    strategy_version_id=str(record.strategy_id)[:64],
                    # The runner's record id (not the DB row id): conclude_experiment
                    # looks decision targets up by this exact value.
                    agent_experiment_id=str(record.id)[:64],
                    outcome=str(record.verdict)[:20],
                    parameter_changes=_safe(dict(variant_changes.get(variant_id, {}))),
                    outcome_metrics=_safe(dict(metrics or {})),
                    baseline_metrics=_safe(base_metrics),
                    reasons=_safe(list(record.reasons or [])),
                    oos_sharpe=_sharpe_from_metrics(metrics),
                    schema_version=GRAPH_EDGE_SCHEMA_VERSION,
                )
                session.add(edge)

            await session.commit()
            report["edges_written"] = len(record.variant_metrics or {})
            report["experiment_row"] = experiment_row_id
            if experiment_row_id is None and not report["note"]:
                report["note"] = "edges only"
            return report
    except Exception as exc:  # noqa: BLE001 — never break a research run
        logger.warning(
            "research_graph_persist_failed %s", f"{type(exc).__name__}: {exc}"
        )
        report["note"] = f"persist failed: {type(exc).__name__}: {exc}"
        return report


async def record_experiment_decision(
    agent_experiment_id: str,
    decision: str,
    notes: str = "",
    session_factory: Optional[Any] = None,
) -> int:
    """Propagate a human conclusion (adopt/reject/defer) onto the graph.

    Writes one decision edge per variant of the experiment so aggregation can
    answer "how often did we adopt changes like this?". Returns edges written.
    """
    decision = decision.strip().lower()
    if decision not in ("adopt", "reject", "defer"):
        raise ValueError(f"decision must be adopt/reject/defer, got {decision!r}")
    factory = session_factory or get_session_factory()
    async with factory() as session:
        sources = (
            await session.execute(
                select(ResearchGraphEdge).where(
                    ResearchGraphEdge.agent_experiment_id == agent_experiment_id,
                    ResearchGraphEdge.edge_kind == EDGE_KIND_EXPERIMENT,
                )
            )
        ).scalars().all()
        for src in sources:
            session.add(
                ResearchGraphEdge(
                    edge_kind="decision",
                    hypothesis_id=src.hypothesis_id,
                    strategy_version_id=src.strategy_version_id,
                    agent_experiment_id=src.agent_experiment_id,
                    outcome=decision,
                    parameter_changes=dict(src.parameter_changes or {}),
                    outcome_metrics={},
                    baseline_metrics={},
                    reasons=[notes] if notes else [],
                    oos_sharpe=src.oos_sharpe,
                    schema_version=GRAPH_EDGE_SCHEMA_VERSION,
                )
            )
        await session.commit()
        return len(sources)


# ---------------------------------------------------------------------------
# Graph queries — deterministic answers over stored records
# ---------------------------------------------------------------------------


def _outcome_counts(outcomes: List[str]) -> Dict[str, int]:
    return {
        "PASS": sum(1 for o in outcomes if o == "PASS"),
        "FAIL": sum(1 for o in outcomes if o == "FAIL"),
        "INCONCLUSIVE": sum(1 for o in outcomes if o == "INCONCLUSIVE"),
        "adopt": sum(1 for o in outcomes if o == "adopt"),
        "reject": sum(1 for o in outcomes if o == "reject"),
        "defer": sum(1 for o in outcomes if o == "defer"),
    }


async def hypothesis_test_count(
    hypothesis_id: str,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """How many times has this hypothesis been tested, and with what results?

    A join over the graph edges — exactly the roadmap's first example question.
    """
    factory = session_factory or get_session_factory()
    async with factory() as session:
        edges = (
            await session.execute(
                select(ResearchGraphEdge).where(
                    ResearchGraphEdge.hypothesis_id == hypothesis_id
                )
            )
        ).scalars().all()

    exp_edges = [e for e in edges if e.edge_kind == EDGE_KIND_EXPERIMENT]
    outcomes = [e.outcome for e in exp_edges]
    decisions = [e.outcome for e in edges if e.edge_kind == "decision"]
    # Parameter observations come from experiment edges only — decision edges
    # inherit their parameters and would double-count every variation.
    by_param: Dict[str, List[Dict[str, Any]]] = {}
    for e in exp_edges:
        for param, value in (e.parameter_changes or {}).items():
            by_param.setdefault(param, []).append({"value": value, "outcome": e.outcome})

    return {
        "hypothesis_id": hypothesis_id,
        "times_tested": len(outcomes),
        "experiments": len({e.agent_experiment_id for e in edges}),
        "outcomes": _outcome_counts(outcomes),
        "decisions": _outcome_counts(decisions),
        "strategy_versions": sorted({e.strategy_version_id for e in edges}),
        "parameters_varied": {k: len(v) for k, v in by_param.items()},
        "first_tested": _iso(min((e.created_at for e in edges if e.created_at), default=None)),
        "last_tested": _iso(max((e.created_at for e in edges if e.created_at), default=None)),
        "note": (
            "counted from stored experiment edges"
            if edges
            else "no stored tests reference this hypothesis id"
        ),
    }


async def aggregate_by_parameter(
    param_pattern: str,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Group experiment edges by one parameter's value; aggregate outcomes.

    ``param_pattern`` is a fuzzy name match (e.g. ``"displacement"`` matches
    ``displacement_atr_frac``). Answers "what happens OOS when
    displacement > 0.25?" from real records: per distinct value, how many
    experiments, how many failed, mean OOS Sharpe.
    """
    factory = session_factory or get_session_factory()
    pattern = param_pattern.strip().lower()
    async with factory() as session:
        edges = (
            await session.execute(
                select(ResearchGraphEdge).where(
                    ResearchGraphEdge.edge_kind == EDGE_KIND_EXPERIMENT
                )
            )
        ).scalars().all()

    groups: Dict[Tuple[str, Any], Dict[str, Any]] = {}
    matched = 0
    for e in edges:
        for name, value in (e.parameter_changes or {}).items():
            if pattern not in name.lower():
                continue
            matched += 1
            key = (name, _bucket_value(value))
            g = groups.setdefault(
                key,
                {"parameter": name, "value": value, "n": 0, "fails": 0, "adopt": 0, "oos_sharpes": []},
            )
            g["n"] += 1
            if e.outcome == "FAIL":
                g["fails"] += 1
            if e.outcome == "adopt":
                g["adopt"] += 1
            if e.oos_sharpe is not None:
                g["oos_sharpes"].append(float(e.oos_sharpe))

    def _finalize(g: Dict[str, Any]) -> Dict[str, Any]:
        sharpes = g.pop("oos_sharpes")
        g["mean_oos_sharpe"] = round(sum(sharpes) / len(sharpes), 4) if sharpes else None
        g["fail_rate"] = round(g["fails"] / g["n"], 3) if g["n"] else None
        return g

    buckets = sorted(
        (_finalize(dict(g)) for g in groups.values()),
        key=lambda g: (g["parameter"], str(g["value"])),
    )
    return {
        "parameter_pattern": pattern,
        "observations_matched": matched,
        "buckets": buckets,
        "note": (
            "outcomes and OOS Sharpe aggregated from stored experiment edges"
            if matched
            else "no stored observations match this parameter — nothing to conclude"
        ),
    }


def _bucket_value(value: Any) -> Any:
    """Bucket numeric values to 2 decimals so 0.26 and 0.26000001 group."""
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return str(value)


async def strategy_research_summary(
    strategy_version_id: str,
    session_factory: Optional[Any] = None,
) -> Dict[str, Any]:
    """Everything the graph knows about one strategy version."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        edges = (
            await session.execute(
                select(ResearchGraphEdge)
                .where(ResearchGraphEdge.strategy_version_id == strategy_version_id)
                .order_by(ResearchGraphEdge.created_at.desc())
            )
        ).scalars().all()

    exp_edges = [e for e in edges if e.edge_kind == EDGE_KIND_EXPERIMENT]
    return {
        "strategy_version_id": strategy_version_id,
        "experiment_count": len({e.agent_experiment_id for e in exp_edges}),
        "hypothesis_ids": sorted({e.hypothesis_id for e in edges if e.hypothesis_id}),
        "outcomes": _outcome_counts([e.outcome for e in exp_edges]),
        "decisions": _outcome_counts([e.outcome for e in edges if e.edge_kind == "decision"]),
        "parameters_explored": sorted({p for e in exp_edges for p in (e.parameter_changes or {})}),
        "last_activity": _iso(edges[0].created_at if edges else None),
    }


async def graph_summary(session_factory: Optional[Any] = None) -> Dict[str, Any]:
    """Cheap overview for the registry/status surfaces."""
    factory = session_factory or get_session_factory()
    async with factory() as session:
        n_edges = len((await session.execute(select(ResearchGraphEdge.id))).all())
    return {"edges": n_edges}
