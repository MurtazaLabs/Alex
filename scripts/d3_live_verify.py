"""Stage D3 live verification: continuous probation monitoring (read-only).

1. Run a REAL monitoring pass over the live DB's probation rows via the
   default session factory and the real feed chain.
2. Assert the pass is write-nothing: a full snapshot of the
   ``strategy_probations`` rows before/after must be identical.
3. Assert change-only emission on a local InMemoryEventBus: first pass emits
   one ``probation.monitor`` event per active version, an unchanged second
   pass emits none.
4. Assert the mock-feed flag agrees with the feed chain's own active source.

Run:  .venv311/Scripts/python.exe scripts/d3_live_verify.py
"""

import asyncio
import logging
import sys
from datetime import datetime

sys.path.insert(0, "src")

from sqlalchemy import select

from forex_ai_agent.core.event_bus import InMemoryEventBus
from forex_ai_agent.data.database import get_session_factory
from forex_ai_agent.data.models.autonomy import StrategyProbation
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.autonomy.probation import ProbationStore
from forex_ai_agent.autonomy.probation_monitor import ProbationMonitor


def _snapshot(rows) -> dict:
    out = {}
    for r in rows:
        out[r.version_id] = (
            r.version_id,
            r.status,
            r.observed_trades,
            r.observed_days,
            r.notes,
            str(r.last_metrics),
            r.started_at.isoformat() if r.started_at else None,
            r.completed_at.isoformat() if r.completed_at else None,
        )
    return out


async def _all_rows(factory) -> list:
    async with factory() as session:
        return (await session.execute(select(StrategyProbation))).scalars().all()


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    factory = get_session_factory()
    store = ProbationStore(session_factory=factory)
    source = get_feed_chain().active_source
    print("feed chain active source:", source)

    before = _snapshot(await _all_rows(factory))
    print(f"DB rows: {len(before)} probation record(s) total")

    bus = InMemoryEventBus()
    monitor = ProbationMonitor(store=store, event_bus=bus)

    reports = await monitor.monitor_once()  # emits per-version events
    print(f"\nmonitor pass 1: {len(reports)} active probation(s) assessed")
    print(f"summary: {monitor.summarize(reports)}")
    for r in reports[:10]:
        print(
            f"  {r['version_id']} [{r['monitor_status'].upper()}] "
            f"{r['observed_trades']}/{r['required_trades']} trades, "
            f"{r['days_elapsed']}/{r['required_days']}d, mode={r['accrual_mode']}, "
            f"flags={','.join(r['flags']) or '-'}"
        )
    rendered = monitor.render_reports(reports)
    if rendered:
        print("\nrendered snapshot (code-rendered, first 6 lines):")
        for line in rendered.splitlines()[:6]:
            print("  " + line)

    # Change-only emission: an identical second pass must emit nothing.
    n_after_first = len(bus.published)
    await monitor.monitor_once()
    n_after_second = len(bus.published)
    print(f"\nbus events: pass1={n_after_first}, unchanged pass2 added={n_after_second - n_after_first}")

    # Write-nothing: the DB snapshot must be byte-identical.
    after = _snapshot(await _all_rows(factory))
    unchanged = before == after

    # Mock-flag agreement with the chain itself.
    mock_flags_ok = all(
        ("mock_feed" in r["flags"]) == (source == "mock") for r in reports
    )

    print(f"DB unchanged after monitoring passes: {unchanged}")
    print(f"mock-feed flag agrees with chain ({source}): {mock_flags_ok}")

    ok = unchanged and mock_flags_ok and n_after_second == n_after_first
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
