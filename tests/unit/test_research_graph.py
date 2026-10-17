"""Stage-B2 unit tests: research graph edges (SQLite, not a graph DB).

Pins the roadmap acceptance criteria: every number the agent says about
"how many times has this hypothesis been tested?" / "what happens OOS when
displacement > 0.25?" comes from a deterministic join over STORED records —
edge rows written when experiments run, decision edges written when a human
concludes, all queryable without re-running anything.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import select, text as sa_text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# Import at module level so ResearchGraphEdge registers on Base.metadata
# before the fixture's create_all runs (mirrors app-boot import order).
from forex_ai_agent.agent import research_graph as rg
from forex_ai_agent.data.models.base import Base as ModelsBase
from forex_ai_agent.data.models.user import User

HYP = "a1b2c3d4"  # research-pipeline style short hypothesis id


@pytest.fixture()
async def session_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    # Production boot order: init_db (create_all) then ensure_graph_schema —
    # every test therefore sees the final schema incl. ALTERed link columns.
    await rg.ensure_graph_schema(factory)
    yield factory
    await engine.dispose()


def _user(username: str) -> User:
    return User(
        username=username,
        email=f"{username}@localhost",
        password_hash="!",  # not a login account; column is NOT NULL
        permission_level=0,
    )


class _Unserializable:
    def __str__(self):
        return "nan-artifact"


def _record(
    hyp_id: str = HYP,
    id: str = "exp42",
    verdict: str = "FAIL",
    changes: dict | None = None,
    variant_metrics: dict | None = None,
    variant_changes: dict | None = None,
) -> SimpleNamespace:
    """Stand-in for ExperimentRecord (runner field names, not ORM)."""
    changes = changes if changes is not None else {"displacement_atr_frac": 0.25}
    variant_metrics = variant_metrics if variant_metrics is not None else {
        "var001": {"sharpe_ratio": -0.8, "profit_factor": 0.9},
        "var002": {"sharpe_ratio": 0.4, "profit_factor": 1.4},
    }
    if variant_changes is None:
        variant_changes = {vid: dict(changes) for vid in variant_metrics}
    return SimpleNamespace(
        id=id,
        hypothesis_id=hyp_id,
        strategy_id="f39b8d40",
        base_version="v3",
        parameter_changes=dict(changes),
        dataset={"symbol": "GBP/JPY", "timeframe": "H1"},
        baseline_metrics={"sharpe_ratio": 0.6, "profit_factor": 1.3},
        variant_metrics=variant_metrics,
        verdict=verdict,
        reasons=["mean OOS Sharpe dropped below 0.5"],
        recommendation="Do not adopt changes.",
        variant_changes=variant_changes,
        created_at=datetime.utcnow(),
    )


async def _add_user(factory, username: str = "alex") -> None:
    async with factory() as session:
        session.add(_user(username))
        await session.commit()


# ---------------------------------------------------------------------------
# Schema ensure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_graph_schema_is_idempotent(session_factory):
    first = await rg.ensure_graph_schema(session_factory)
    second = await rg.ensure_graph_schema(session_factory)

    assert first["edge_table_created"] is False  # create_all pre-made it
    assert second == {"edge_table_created": False, "columns_added": []}


@pytest.mark.asyncio
async def test_ensure_adds_link_columns_to_existing_tables(session_factory):
    from sqlalchemy import text as sa_text

    # Simulate a pre-B2 database: drop and recreate without the link columns.
    async with session_factory() as session:
        await session.execute(sa_text("DROP TABLE hypotheses"))
        await session.execute(sa_text("CREATE TABLE hypotheses (id VARCHAR(32) PRIMARY KEY)"))
        await session.execute(sa_text("DROP TABLE agent_experiments"))
        await session.execute(
            sa_text("CREATE TABLE agent_experiments (id VARCHAR(32) PRIMARY KEY)")
        )
        await session.commit()

    report = await rg.ensure_graph_schema(session_factory)
    assert sorted(report["columns_added"]) == [
        "agent_experiments.hypothesis_id",
        "hypotheses.strategy_version_id",
    ]


# ---------------------------------------------------------------------------
# persist_agent_experiment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_without_user_writes_edges_only(session_factory):
    report = await rg.persist_agent_experiment(_record(), session_factory)

    assert report["edges_written"] == 2
    assert report["experiment_row"] is None
    assert "no user" in report["note"]
    assert (await rg.graph_summary(session_factory))["edges"] == 2


@pytest.mark.asyncio
async def test_persist_with_user_writes_row_and_edges(session_factory):
    from forex_ai_agent.data.models.agent import AgentExperiment

    await _add_user(session_factory)
    report = await rg.persist_agent_experiment(_record(), session_factory)

    assert report["experiment_row"] is not None
    async with session_factory() as session:
        row = (await session.execute(select(AgentExperiment))).scalars().one()
        assert row.strategy_version_id == "f39b8d40"
        assert row.verdict == "FAIL"
        assert row.parameter_changes["displacement_atr_frac"] == 0.25
    # hypothesis_id lives on an ALTER-added column (not on the mapped class);
    # read it with raw SQL like ensure_graph_schema wrote it.
    async with session_factory() as session:
        val = (
            await session.execute(
                sa_text("SELECT hypothesis_id FROM agent_experiments WHERE strategy_version_id='f39b8d40'")
            )
        ).scalar()
    assert val == HYP


@pytest.mark.asyncio
async def test_edges_carry_runner_experiment_id_and_per_variant_changes(session_factory):
    await _add_user(session_factory)
    await rg.persist_agent_experiment(
        _record(
            variant_changes={"var001": {"displacement_atr_frac": 0.25}, "var002": {"displacement_atr_frac": 0.45}}
        ),
        session_factory,
    )

    async with session_factory() as session:
        edges = (await session.execute(select(rg.ResearchGraphEdge))).scalars().all()
    assert {e.agent_experiment_id for e in edges} == {"exp42"}  # runner id, not row id
    by_variant = {e.outcome_metrics["sharpe_ratio"]: e for e in edges}
    assert by_variant[-0.8].parameter_changes == {"displacement_atr_frac": 0.25}
    assert by_variant[-0.8].oos_sharpe == -0.8
    assert by_variant[0.4].parameter_changes == {"displacement_atr_frac": 0.45}
    assert by_variant[0.4].hypothesis_id == HYP


@pytest.mark.asyncio
async def test_persist_survives_unserializable_metrics(session_factory):
    rec = _record(
        variant_metrics={"var001": {"sharpe_ratio": _Unserializable()}},
        variant_changes={"var001": {"displacement_atr_frac": 0.25}},
    )
    report = await rg.persist_agent_experiment(rec, session_factory)  # must not raise

    assert report["edges_written"] == 1
    assert (await rg.graph_summary(session_factory))["edges"] == 1


# ---------------------------------------------------------------------------
# record_experiment_decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decision_requires_valid_value(session_factory):
    with pytest.raises(ValueError):
        await rg.record_experiment_decision("exp42", "maybe", "", session_factory)


@pytest.mark.asyncio
async def test_decision_writes_one_edge_per_variant(session_factory):
    await rg.persist_agent_experiment(_record(), session_factory)
    n = await rg.record_experiment_decision("exp42", "adopt", "PF 1.4 OOS", session_factory)

    assert n == 2
    async with session_factory() as session:
        rows = (
            (
                await session.execute(
                    select(rg.ResearchGraphEdge).where(rg.ResearchGraphEdge.edge_kind == "decision")
                )
            )
            .scalars()
            .all()
        )
    assert len(rows) == 2
    assert {r.outcome for r in rows} == {"adopt"}
    assert {r.hypothesis_id for r in rows} == {HYP}
    # decisions inherit the source edge's parameters/metrics for aggregation
    assert all(r.parameter_changes for r in rows)
    assert all(r.oos_sharpe is not None for r in rows)
    assert all(r.reasons == ["PF 1.4 OOS"] for r in rows)


# ---------------------------------------------------------------------------
# hypothesis_test_count — the "tested 4 times?" join
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_test_count_counts_edges_and_decisions_separately(session_factory):
    await rg.persist_agent_experiment(_record(), session_factory)
    await rg.record_experiment_decision("exp42", "reject", "", session_factory)

    data = await rg.hypothesis_test_count(HYP, session_factory)
    assert data["times_tested"] == 2  # variant edges, NOT decisions
    assert data["experiments"] == 1
    assert data["outcomes"]["FAIL"] == 2
    assert data["decisions"]["reject"] == 2
    assert data["strategy_versions"] == ["f39b8d40"]
    assert data["parameters_varied"] == {"displacement_atr_frac": 2}
    assert data["first_tested"] is not None and data["last_tested"] is not None
    assert "counted from stored" in data["note"]


@pytest.mark.asyncio
async def test_test_count_for_unknown_hypothesis_is_zero(session_factory):
    data = await rg.hypothesis_test_count("deadbeef", session_factory)
    assert data["times_tested"] == 0
    assert data["experiments"] == 0
    assert "no stored tests" in data["note"]


@pytest.mark.asyncio
async def test_test_count_distinguishes_hypotheses(session_factory):
    await rg.persist_agent_experiment(_record(hyp_id=HYP, id="exp42"), session_factory)
    await rg.persist_agent_experiment(
        _record(hyp_id="eeee9999", id="exp43"),
        session_factory,
    )

    assert (await rg.hypothesis_test_count(HYP, session_factory))["times_tested"] == 2
    assert (await rg.hypothesis_test_count("eeee9999", session_factory))["times_tested"] == 2
    # distinct experiment ids are tracked per hypothesis, not globally
    assert (await rg.hypothesis_test_count(HYP, session_factory))["experiments"] == 1
    assert (await rg.hypothesis_test_count("eeee9999", session_factory))["experiments"] == 1


# ---------------------------------------------------------------------------
# aggregate_by_parameter — the "OOS when displacement > 0.25?" join
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_aggregate_buckets_by_value_with_fail_rate_and_mean_sharpe(session_factory):
    await rg.persist_agent_experiment(
        _record(
            changes={"displacement_atr_frac": 0.25},
            verdict="FAIL",
            variant_metrics={"var001": {"sharpe_ratio": -0.8}, "var002": {"sharpe_ratio": 0.4}},
        ),
        session_factory,
    )
    await rg.persist_agent_experiment(
        _record(
            changes={"displacement_atr_frac": 0.45},
            verdict="PASS",
            variant_metrics={"var001": {"sharpe_ratio": 0.9}, "var002": {"sharpe_ratio": 1.1}},
        ),
        session_factory,
    )

    data = await rg.aggregate_by_parameter("displacement", session_factory)
    assert data["observations_matched"] == 4
    buckets = {(b["parameter"], b["value"]): b for b in data["buckets"]}
    low = buckets[("displacement_atr_frac", 0.25)]
    high = buckets[("displacement_atr_frac", 0.45)]
    assert low["n"] == 2 and low["fails"] == 2 and low["fail_rate"] == 1.0
    assert low["mean_oos_sharpe"] == pytest.approx((-0.8 + 0.4) / 2)
    assert high["n"] == 2
    assert high["fail_rate"] == 0.0
    assert high["mean_oos_sharpe"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_aggregate_fuzzy_matches_parameter_name(session_factory):
    await rg.persist_agent_experiment(
        _record(changes={"displacement_atr_frac": 0.25}), session_factory
    )
    data = await rg.aggregate_by_parameter("DISPLACEMENT_ATR", session_factory)  # case/suffix
    assert data["observations_matched"] == 2


@pytest.mark.asyncio
async def test_aggregate_ignores_unrelated_parameters_and_decisions(session_factory):
    await rg.persist_agent_experiment(
        _record(changes={"stop_multiple": 1.5}, verdict="FAIL"), session_factory
    )
    await rg.record_experiment_decision("exp42", "reject", "", session_factory)

    data = await rg.aggregate_by_parameter("displacement", session_factory)
    assert data["observations_matched"] == 0
    assert data["buckets"] == []
    assert "nothing to conclude" in data["note"]


# ---------------------------------------------------------------------------
# strategy_research_summary / graph_summary
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strategy_summary_joins_edges_and_decisions(session_factory):
    await rg.persist_agent_experiment(_record(), session_factory)
    await rg.record_experiment_decision("exp42", "adopt", "", session_factory)

    s = await rg.strategy_research_summary("f39b8d40", session_factory)
    assert s["experiment_count"] == 1
    assert s["hypothesis_ids"] == [HYP]
    assert s["outcomes"]["FAIL"] == 2
    assert s["decisions"]["adopt"] == 2
    assert "displacement_atr_frac" in s["parameters_explored"]
    assert s["last_activity"] is not None


@pytest.mark.asyncio
async def test_graph_summary_counts_all_edges(session_factory):
    await rg.persist_agent_experiment(_record(), session_factory)
    await rg.record_experiment_decision("exp42", "defer", "", session_factory)
    assert (await rg.graph_summary(session_factory))["edges"] == 4


# ---------------------------------------------------------------------------
# answer_why routing — graph answers only when it has data
# ---------------------------------------------------------------------------


def _patch_db(monkeypatch, factory) -> None:
    """Point every DB resolver the graph/explain path uses at the test DB."""
    import forex_ai_agent.agent.explain as explain_mod

    monkeypatch.setattr(rg, "get_session_factory", lambda: factory)
    monkeypatch.setattr(explain_mod, "get_session_factory", lambda: factory)


@pytest.mark.asyncio
async def test_answer_why_hypothesis_question_uses_graph(session_factory, monkeypatch):
    import forex_ai_agent.agent.explain as explain_mod

    _patch_db(monkeypatch, session_factory)
    await rg.persist_agent_experiment(_record(), session_factory)

    card = await explain_mod.Explainer().answer_why(
        f"how many times have we tested hypothesis {HYP}?"
    )
    assert card.verdict == "ANSWERED"
    assert card.title.startswith("RESEARCH GRAPH")
    assert "2 experiment edges" in card.evidence[0].detail


@pytest.mark.asyncio
async def test_answer_why_parameter_question_uses_graph(session_factory, monkeypatch):
    import forex_ai_agent.agent.explain as explain_mod

    _patch_db(monkeypatch, session_factory)
    await rg.persist_agent_experiment(_record(), session_factory)

    card = await explain_mod.Explainer().answer_why(
        "what happens OOS when displacement > 0.25?"
    )
    assert card.verdict == "ANSWERED"
    assert "displacement" in card.title


@pytest.mark.asyncio
async def test_answer_why_falls_through_to_knowledge_when_graph_empty(
    session_factory, monkeypatch
):
    import forex_ai_agent.agent.explain as explain_mod

    _patch_db(monkeypatch, session_factory)  # schema ensured, but NO edges
    captured: dict = {}

    async def fake_search(self, query, limit=10, filters=None):
        captured["query"] = query
        return []

    monkeypatch.setattr(explain_mod.KnowledgeRetriever, "search", fake_search)

    card = await explain_mod.Explainer().answer_why(
        "why was my strategy rejected when volatility spiked?"
    )
    # NOT a graph card: "when" must not hijack unrelated questions
    assert not card.title.startswith("RESEARCH GRAPH")
    assert captured["query"] == "why was my strategy rejected when volatility spiked?"


@pytest.mark.asyncio
async def test_graph_card_without_data_reports_no_data(session_factory, monkeypatch):
    import forex_ai_agent.agent.explain as explain_mod

    _patch_db(monkeypatch, session_factory)
    card = await explain_mod.Explainer().research_graph_card(param_pattern="displacement")
    assert card.verdict == "NO_DATA"


# ---------------------------------------------------------------------------
# Runner integration: create_experiment/conclude_experiment write the graph
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_persists_and_conclude_propagates(
    session_factory, monkeypatch
):
    import forex_ai_agent.agent.experiments as exp_mod

    _patch_db(monkeypatch, session_factory)
    await _add_user(session_factory)

    async def fake_run_experiment(self, **kwargs):
        from forex_ai_agent.agent.experiments import ExperimentRecord

        record = ExperimentRecord(
            id="runexp1",
            hypothesis_id=kwargs.get("hypothesis_id"),
            strategy_id=kwargs["strategy_id"],
            base_version="v1",
            parameter_changes=kwargs["param_mutations"],
            dataset={},
            baseline_metrics={"sharpe_ratio": 0.5},
            variant_metrics={"varA": {"sharpe_ratio": 1.2}},
            verdict="PASS",
            reasons=["sharpe improved"],
            recommendation="Recommend adopting variant varA",
            variant_changes={"varA": dict(kwargs["param_mutations"])},
        )
        self._experiments[record.id] = record
        await rg.persist_agent_experiment(record)
        return record

    monkeypatch.setattr(exp_mod.VariantRunner, "run_experiment", fake_run_experiment)

    created = await exp_mod.create_experiment(
        strategy_id="f39b8d40",
        param_mutations={"displacement_atr_frac": 0.3},
        symbol="GBP/JPY",
        timeframe="H1",
        start_date="2024-01-01T00:00:00",
        end_date="2024-06-01T00:00:00",
        hypothesis_id=HYP,
    )
    assert created["experiment_id"] == "runexp1"

    # conclude → decision edges under the runner experiment id
    result = await exp_mod.conclude_experiment("runexp1", "adopt", "looks good")
    assert result["final_verdict"] == "ADOPT"

    count = await rg.hypothesis_test_count(HYP, session_factory)
    assert count["times_tested"] == 1
    assert count["decisions"]["adopt"] == 1
