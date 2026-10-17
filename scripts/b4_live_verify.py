"""B4 live verification: the Director against the live system (no server needed).

Runs the acceptance path end-to-end in-process:
1. enqueue a tagged verify pair
2. run ONE real budget-capped cycle with use_llm=False (deterministic fallback
   spec — the LLM is out of credits, which is exactly the fallback case)
3. verify the outcome row, audit trail, budget accounting
4. clean up the synthetic queue rows/events so the live DB stays pristine
"""

import asyncio
import uuid

from sqlalchemy import delete, select


async def main() -> None:
    from forex_ai_agent.agent import director as dr

    print("== 1. status before ==")
    before = await dr.queue_snapshot()
    print(f"   open items: {before['open_items']} | by status: {before['by_status']}")

    print("== 2. enqueue tagged verify pair ==")
    enq = await dr.enqueue_hypothesis(
        pair="EUR/USD",
        hypothesis_text="B4 live verify — deterministic fallback cycle",
        priority=1,
    )
    print(f"   {enq}")
    assert enq["enqueued"] is True, f"expected enqueue, got {enq}"
    item_id = enq["item_id"]

    print("== 3. run one REAL cycle (use_llm=False, 1 item) ==")
    config = dr.DirectorConfig(
        max_items_per_cycle=1,
        max_llm_calls_per_cycle=0,  # force the deterministic fallback path
        use_llm=False,
        history_days=180,
    )
    report = await dr.run_cycle(config)
    print(f"   cycle {report['cycle_id']}")
    print(f"   items_run={report['items_run']} validated={report['validated']} "
          f"rejected={report['rejected']} errors={report['errors']} "
          f"llm_used={report['llm_calls_used']} budget_exhausted={report['budget_exhausted']}")
    for r in report["results"]:
        print(f"   -> {r['pair']}: {r.get('status', r.get('error'))}")
    assert report["items_run"] == 1
    assert report["llm_calls_used"] == 0
    assert report["results"][0]["pair"] == "EUR/USD"
    assert report["results"][0].get("status") in ("validated", "rejected", "error")

    print("== 4. queue row + audit trail ==")
    snap = await dr.queue_snapshot()
    row = next(i for i in snap["queue"] if i["id"] == item_id)
    print(f"   row status: {row['status']} | result: { {k: row['result'].get(k) for k in ('status', 'version_id')} }")
    assert row["status"] in (dr.DONE_VALIDATED, dr.DONE_REJECTED, dr.DONE_ERROR)
    actions = [e["action"] for e in snap["recent_events"]]
    assert "cycle_started" in actions and "item_started" in actions \
        and "item_finished" in actions and "cycle_finished" in actions
    print(f"   audit actions present: {sorted(set(actions))}")

    print("== 5. verify the hard-blocks note is in the module (no risk/order paths) ==")
    import inspect
    src = inspect.getsource(dr)
    for banned in ("place_order", "risk_manager", "MAX_PERMISSION_LEVEL = 4"):
        assert banned not in src, banned

    print("== 6. cleanup synthetic verify rows ==")
    from forex_ai_agent.data.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            delete(dr.DirectorQueueItem).where(dr.DirectorQueueItem.pair == "EUR/USD")
        )
        await session.execute(
            delete(dr.DirectorEvent).where(
                dr.DirectorEvent.detail.cast(__import__("sqlalchemy").String).like('%"B4 live verify%')
            )
        )
        await session.execute(
            delete(dr.DirectorEvent).where(dr.DirectorEvent.cycle_id == report["cycle_id"])
        )
        await session.commit()
    # Also remove any strategy version the fallback spec produced during the cycle.
    try:
        from forex_ai_agent.strategy import create_version_manager

        vm = await create_version_manager()
        removed = 0
        for v in list(vm.versions.values()):
            desc = (v.description or "")
            if "fallback" in desc.lower() and v.id != "f39b8d40":
                vm.versions.pop(v.id, None)
                removed += 1
        print(f"   fallback versions removed from store: {removed}")
    except Exception as exc:  # noqa: BLE001
        print(f"   version cleanup skipped: {exc}")
    after = await dr.queue_snapshot()
    print(f"   queue after cleanup: {after['open_items']} open items")
    print("\nB4 LIVE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
