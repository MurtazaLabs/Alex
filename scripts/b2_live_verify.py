"""B2 live verification against alex_local.db (no server needed).

Runs the exact production functions in-process: ensure_graph_schema (the call
lifespan makes at boot), registry dispatch of the three B2 tools, answer_why
routing, and a full write→query→cleanup cycle with clearly-tagged synthetic
verify rows (deleted afterwards so no fake research history pollutes the graph).
"""

import asyncio

from sqlalchemy import select


async def main() -> None:
    from forex_ai_agent.agent import research_graph as rg
    from forex_ai_agent.agent.explain import answer_why
    from forex_ai_agent.agent.tools import get_tool_registry

    print("== 1. ensure_graph_schema (live DB) ==")
    report = await rg.ensure_graph_schema()
    print("   ensure:", report)
    again = await rg.ensure_graph_schema()
    print("   idempotent re-run:", again)
    assert again["columns_added"] == [] and again["edge_table_created"] is False

    print("== 2. live schema check ==")
    import sqlite3

    conn = sqlite3.connect("alex_local.db")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "research_graph_edges" in tables, "edge table missing"
    edge_cols = {r[1] for r in conn.execute("PRAGMA table_info(research_graph_edges)")}

    def has_col(table, col):
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        return col in cols

    assert has_col("hypotheses", "strategy_version_id")
    assert has_col("agent_experiments", "hypothesis_id")
    print("   research_graph_edges cols:", sorted(edge_cols))
    print("   hypotheses.strategy_version_id + agent_experiments.hypothesis_id: OK")
    conn.close()

    print("== 3. registry dispatch of B2 tools ==")
    reg = get_tool_registry(0)
    assert len(reg._tools) == 39, f"registry size {len(reg._tools)}"
    summary = await reg.dispatch("research_graph_summary", {})
    print("   research_graph_summary:", summary)
    count = await reg.dispatch("hypothesis_test_count", {"hypothesis_id": "nonexist000"})
    print("   hypothesis_test_count (unknown id):", count)
    agg = await reg.dispatch("aggregate_by_parameter", {"param_pattern": "displacement"})
    print("   aggregate_by_parameter:", agg.note, "| matched:", agg.observations_matched)

    print("== 4. write→query→decision→cleanup cycle (synthetic, tagged) ==")

    from types import SimpleNamespace

    from forex_ai_agent.data.database import get_session_factory
    from forex_ai_agent.data.models.agent import AgentExperiment
    from forex_ai_agent.data.models.user import User

    factory = get_session_factory()
    async with factory() as session:
        user = (
            await session.execute(select(User).where(User.username == "alex"))
        ).scalar_one_or_none()
        if user is None:
            user = (await session.execute(select(User).limit(1))).scalar_one_or_none()
    print(f"   persisting under user: {user.username if user else None}")

    # Idempotent re-runs: clear any leftovers from a previous verify run first.
    from sqlalchemy import delete

    async with factory() as session:
        await session.execute(
            delete(rg.ResearchGraphEdge).where(
                rg.ResearchGraphEdge.strategy_version_id == "b2verify-strategy"
            )
        )
        await session.execute(
            delete(AgentExperiment).where(
                AgentExperiment.strategy_version_id == "b2verify-strategy"
            )
        )
        await session.commit()

    rec = SimpleNamespace(
        id="b2exp01",
        hypothesis_id="b2be4a5f",  # 8-hex like real research-pipeline ids
        strategy_id="b2verify-strategy",
        base_version="v1",
        parameter_changes={"displacement_atr_frac": 0.25},
        dataset={"symbol": "VERIFY", "timeframe": "H1"},
        baseline_metrics={"sharpe_ratio": 0.5},
        variant_metrics={"varV": {"sharpe_ratio": -0.9}},
        verdict="FAIL",
        reasons=["B2 live verification row"],
        recommendation="verify-only",
        variant_changes={"varV": {"displacement_atr_frac": 0.25}},
        created_at=None,
    )
    persist = await rg.persist_agent_experiment(rec, factory)
    print("   persist:", persist)
    assert persist["edges_written"] == 1 and persist["experiment_row"], "expected row+edge"

    data = await rg.hypothesis_test_count("b2be4a5f", factory)
    print("   count after write:", {k: data[k] for k in ("times_tested", "experiments", "strategy_versions")})
    assert data["times_tested"] == 1

    n = await rg.record_experiment_decision("b2exp01", "reject", "B2 live verify", factory)
    print("   decision edges written:", n)
    assert n == 1
    after = await rg.hypothesis_test_count("b2be4a5f", factory)
    assert after["decisions"]["reject"] == 1

    conn = sqlite3.connect("alex_local.db")
    row_h = conn.execute(
        "SELECT hypothesis_id FROM agent_experiments WHERE strategy_version_id='b2verify-strategy'"
    ).fetchone()
    print("   agent_experiments.hypothesis_id link:", row_h[0] if row_h else None)
    assert row_h and row_h[0] == "b2be4a5f"
    conn.close()

    # answer_why routing against the live DB
    card = await answer_why({"question": "how many times have we tested hypothesis b2be4a5f?"})
    print("   answer_why verdict:", card["verdict"], "|", card["title"])
    assert card["verdict"] == "ANSWERED"
    card2 = await answer_why({"question": "what happens OOS when displacement > 0.25?"})
    print("   answer_why param verdict:", card2["verdict"], "|", card2["title"])
    assert card2["verdict"] == "ANSWERED"

    print("== 5. cleanup synthetic verify rows ==")
    async with factory() as session:
        from sqlalchemy import delete

        ids = (
            await session.execute(
                select(AgentExperiment.id).where(AgentExperiment.strategy_version_id == "b2verify-strategy")
            )
        ).scalars().all()
        for i in ids:
            await session.execute(delete(AgentExperiment).where(AgentExperiment.id == i))
        await session.execute(
            delete(rg.ResearchGraphEdge).where(rg.ResearchGraphEdge.strategy_version_id == "b2verify-strategy")
        )
        await session.commit()
    conn = sqlite3.connect("alex_local.db")
    left = conn.execute("SELECT COUNT(*) FROM research_graph_edges").fetchone()[0]
    left_rows = conn.execute("SELECT COUNT(*) FROM agent_experiments").fetchone()[0]
    conn.close()
    print(f"   remaining edges: {left} | remaining agent_experiments rows: {left_rows}")
    print("\nB2 LIVE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
