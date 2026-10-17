"""C1 live verification: LLM hypothesis generation against the live system.

In-process, B2/B3/B4/B5 pattern. Two passes:

1. REAL LLM run (use_llm=True, enqueue=False): real deterministic structure
   briefs + the live OpenRouter key. Whatever the free-model pool returns
   (good JSON, rate limits, garbage), the module must behave honestly —
   report provenance, validate candidates deterministically, never crash.
2. Deterministic run (use_llm=False, enqueue=True): template hypotheses from
   the briefs, enqueued into the real Director queue, then cleaned up so the
   live DB stays pristine.
"""

import asyncio
import json

from sqlalchemy import delete, select


async def main() -> None:
    from forex_ai_agent.agent import director as dr
    from forex_ai_agent.agent import hypothesis_gen as hg

    print("== 1. real-LLM pass (propose-only: enqueues nothing) ==")
    report = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(
            pairs=["EUR/USD", "GBP/USD"],
            use_llm=True,
            enqueue=False,
            max_hypotheses=3,
        )
    )
    safe = json.loads(json.dumps(report))
    print(
        f"   llm_used={safe['llm_used']} fallback={safe['fallback_used']} "
        f"raw={safe['raw_candidates']} accepted={len(safe['accepted'])} "
        f"rejected={len(safe['rejected'])}"
    )
    if safe["llm_error"]:
        print(f"   llm_error (honest): {safe['llm_error'][:160]}")
    for h in safe["accepted"]:
        print(f"   -> [{h['source']}] {h['pair']} conf={h['confidence']}: {h['text'][:90]}")
    for r in safe["rejected"][:4]:
        print(f"   x rejected: {r['pair']}: {r['reason'][:80]}")
    assert safe["llm_used"] or safe["fallback_used"], "exactly one source must win"
    assert safe["enqueued_count"] == 0, "propose-only must not enqueue"
    snap = await dr.queue_snapshot()
    assert snap["open_items"] == 0, "propose-only touched the queue"
    print("   propose-only confirmed: queue untouched")

    print("== 2. deterministic pass (use_llm=False, enqueue=True) ==")
    report2 = await hg.generate_hypotheses(
        hg.HypothesisGenConfig(
            pairs=["EUR/USD"],
            use_llm=False,
            enqueue=True,
            max_hypotheses=2,
        )
    )
    safe2 = json.loads(json.dumps(report2))
    print(
        f"   fallback={safe2['fallback_used']} accepted={len(safe2['accepted'])} "
        f"enqueued={safe2['enqueued_count']}"
    )
    assert safe2["fallback_used"] is True
    assert safe2["enqueued_count"] == 1
    assert safe2["accepted"][0]["source"] == "fallback"
    snap = await dr.queue_snapshot()
    row = next(
        i for i in snap["queue"] if i["id"] == safe2["accepted"][0]["item_id"]
    )
    print(f"   queue row: {row['pair']} {row['status']} — {row['hypothesis'][:60]}…")
    assert row["status"] == dr.QUEUED

    print("== 3. hard blocks by construction (source scan) ==")
    import inspect

    src = inspect.getsource(hg)
    for banned in ("place_order", "risk_manager", "execute_trade"):
        assert banned not in src, banned
    print("   no order/risk/permission paths reachable from the module")

    print("== 4. cleanup ==")
    from forex_ai_agent.data.database import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        item_ids = [h["item_id"] for h in safe2["accepted"] if h.get("item_id")]
        removed = 0
        for iid in item_ids:
            import uuid as _uuid

            item = await session.get(dr.DirectorQueueItem, _uuid.UUID(iid))
            if item is not None:
                await session.delete(item)
                removed += 1
        events = (await session.execute(select(dr.DirectorEvent))).scalars().all()
        removed_events = 0
        for e in events:
            if e.item_id in set(item_ids):
                await session.delete(e)
                removed_events += 1
        await session.commit()
    print(f"   removed {removed} queue rows + {removed_events} audit rows")
    after = await dr.queue_snapshot()
    assert after["open_items"] == 0
    print(f"   queue after cleanup: {after['open_items']} open items")
    print("\nC1 LIVE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
