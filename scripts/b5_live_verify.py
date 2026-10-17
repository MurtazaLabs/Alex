"""B5 live verification: structured research plans against the live system.

In-process (no server needed), following the B2/B3/B4 live-verify pattern:

1. ensure the ``research_plans`` table exists (boot parity) and record the
   pre-existing plan count
2. create a real plan via the production create path (one tagged pair,
   enqueued with the Director) and read the persisted card back
3. run ONE real budget-capped cycle (use_llm=False → deterministic fallback)
   with the plan-linked pair and verify the plan SELF-REPORTS progress from
   persisted records: pair state, checklist advance, conclusion card
4. tool surface: research_plan by prefix; start_research creates a plan too
5. clean up every synthetic row so the live DB stays pristine
"""

import asyncio

from sqlalchemy import delete, select


async def main() -> None:
    from forex_ai_agent.agent import director as dr
    from forex_ai_agent.agent import research_plan as rp

    print("== 1. schema ensure (boot parity) ==")
    result = await rp.ensure_plan_schema()
    print(f"   ensure_plan_schema -> {result}")
    plans_before = await rp.list_research_plans(limit=100)
    print(f"   existing plans in live DB: {len(plans_before)}")

    print("== 2. create a real plan (production create path) ==")
    created = await rp.create_research_plan(
        topic="B5 live verify — displacement continuation",
        pairs=["EUR/USD"],
        hypothesis_text="B5 live verify — plan-linked deterministic cycle",
        enqueued=True,
    )
    task_id = created["task_id"]
    print(f"   plan id: {task_id[:8]}…  status: {created['card']['status']}")
    assert created["card"]["pairs_progress"]["EUR/USD"]["status"] == "queued"
    assert created["card"]["experiment_total"] == 6
    card = await rp.get_research_plan(task_id)
    assert card is not None and card["status"] == rp.STATUS_IN_PROGRESS
    print("   persisted card read back OK (IN_PROGRESS, pair queued)")

    print("== 3. one REAL cycle (use_llm=False) — plan self-reports ==")
    config = dr.DirectorConfig(
        max_items_per_cycle=1,
        max_llm_calls_per_cycle=0,
        use_llm=False,
        history_days=180,
    )
    report = await dr.run_cycle(config)
    print(
        f"   cycle {report['cycle_id']}: items_run={report['items_run']} "
        f"validated={report['validated']} rejected={report['rejected']} errors={report['errors']}"
    )
    assert report["items_run"] == 1
    assert report["results"][0].get("plan_id") == task_id, report["results"]

    card = await rp.get_research_plan(task_id)
    pair_state = card["pairs_progress"]["EUR/USD"]
    print(
        f"   plan status: {card['status']} | progress {card['progress_percent']}% | "
        f"pair: {pair_state['status']} | conclusion: {card['conclusion']['verdict']}"
    )
    assert pair_state["status"] in ("validated", "rejected", "error")
    assert card["status"] == rp.STATUS_COMPLETED
    assert card["progress_percent"] == 100
    done = [s["step"] for s in card["experiment_checklist"] if s["state"] == "done"]
    assert done == ["baseline"], done
    assert card["conclusion"]["verdict"] == pair_state["status"]
    assert "RESEARCH #" in rp.render_plan_card(card)
    print(f"   checklist after finish: {[s['state'] for s in card['experiment_checklist']]}")

    # The Director row carries the stored conclusion card.
    snap = await dr.queue_snapshot()
    row = next(i for i in snap["queue"] if (i["result"] or {}).get("plan_id") == task_id)
    cc = row["result"].get("conclusion_card")
    assert cc and cc["plan_id"] == task_id and cc["verdict"] == pair_state["status"]
    print(f"   conclusion card on Director row: verdict={cc['verdict']}")

    print("== 4. tool surface ==")
    from forex_ai_agent.agent.tools import get_tool_registry

    reg = get_tool_registry(0)
    res = await reg.dispatch("research_plan", {"task_id": task_id[:8]})
    assert res.found and res.plans[0]["task_id"] == task_id
    print("   research_plan by prefix: found, rendered card present")
    listing = await reg.dispatch("research_plan", {})
    assert listing.found and any(p["task_id"] == task_id for p in listing.plans)
    print(f"   research_plan index lists {len(listing.plans)} plan(s)")
    from forex_ai_agent.agent import tools as t

    sr = await t.start_research(
        t.StartResearchArgs(topic="B5 live verify — start_research path", pair="GBP/USD")
    )
    assert sr.status == "started" and sr.proposal["task_id"] == sr.task_id
    sr_card = await rp.get_research_plan(sr.task_id)
    assert sr_card is not None
    print(f"   start_research created plan {sr.task_id[:8]}… (persisted)")

    print("== 5. cleanup synthetic verify rows ==")
    from forex_ai_agent.data.database import get_session_factory
    from forex_ai_agent.strategy import create_version_manager

    factory = get_session_factory()
    plan_ids = {task_id, sr.task_id}
    version_id = report["results"][0].get("version_id")
    async with factory() as session:
        rows = (
            (await session.execute(select(dr.DirectorQueueItem))).scalars().all()
        )
        linked = [i for i in rows if (i.result_json or {}).get("plan_id") in plan_ids]
        linked_ids = {str(i.id) for i in linked}
        for i in linked:
            await session.delete(i)
        from forex_ai_agent.agent.research_plan import ResearchPlan

        for pid in plan_ids:
            import uuid as _uuid

            plan = await session.get(ResearchPlan, _uuid.UUID(pid))
            if plan is not None:
                await session.delete(plan)
        # Audit rows tied to the verify items / cycle.
        events = (
            (await session.execute(select(dr.DirectorEvent))).scalars().all()
        )
        removed_events = 0
        for e in events:
            if e.item_id in linked_ids or (e.detail or {}).get("plan_id") in plan_ids:
                await session.delete(e)
                removed_events += 1
        if version_id:
            from forex_ai_agent.data.models.autonomy import StrategyProbation

            await session.execute(
                delete(StrategyProbation).where(StrategyProbation.version_id == version_id)
            )
        await session.commit()
        print(
            f"   deleted {len(linked)} director rows + {len(plan_ids)} plan rows "
            f"+ {removed_events} audit rows"
        )
    # The cycle's version (if the gate validated) — exact id from the report.
    try:
        vm = await create_version_manager()
        removed = 0
        if version_id and version_id in vm.versions:
            vm.versions.pop(version_id, None)
            removed += 1
        print(f"   cycle versions removed from store: {removed} ({version_id or 'none'})")
    except Exception as exc:  # noqa: BLE001
        print(f"   version cleanup skipped: {exc}")
    after = await rp.list_research_plans(limit=100)
    assert len(after) == len(plans_before), "cleanup must restore the plan count"
    snap_after = await dr.queue_snapshot()
    print(f"   plans after cleanup: {len(after)} | open director items: {snap_after['open_items']}")
    print("\nB5 LIVE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
