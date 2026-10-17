"""One unattended end-to-end research session (Stage C1→C2→C3 chained).

Roadmap Stage C accept: "end-to-end unattended research session produces a
written report whose every number traces to a stored, reproducible record."
This script runs the whole loop in-process, within an explicit budget:

  1. C1 hypothesis generation   (LLM proposes, deterministic validation,
                                 enqueue into the B4 Director queue)
  2. B4 Director cycles         (research brief → spec → backtest gate →
                                 version + probation for survivors)
  3. C2 experiment selection    (LLM picks what to test from the deterministic
                                 menu, code clamps + runs the real engines)
  4. C3 written report          (LLM narrates over the stored evidence;
                                 every number code-rendered + trace-checked)

The LLM is consulted at most BUDGET["max_total_llm"] times; every gate,
validation, cap and number is deterministic code. By default ALL side
effects are cleaned up afterwards (strategy files from the Director's
versioning, queue rows + events, this session's research-task rows,
probation rows, the report row) — set ALEX_KEEP=1 to keep them.

Run:  .venv311/Scripts/python.exe scripts/stage_c_e2e_session.py
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from sqlalchemy import delete, select  # noqa: E402

from forex_ai_agent.agent.director import (  # noqa: E402
    DirectorConfig,
    DirectorEvent,
    DirectorQueueItem,
    queue_snapshot,
    run_cycle,
)
from forex_ai_agent.agent.experiment_selector import (  # noqa: E402
    ExperimentSelConfig,
    select_experiments,
)
from forex_ai_agent.agent.hypothesis_gen import (  # noqa: E402
    HypothesisGenConfig,
    generate_hypotheses,
)
from forex_ai_agent.agent.report_gen import (  # noqa: E402
    ReportConfig,
    ResearchReport,
    generate_research_report,
)
from forex_ai_agent.data.database import get_session_factory  # noqa: E402
from forex_ai_agent.data.models.autonomy import StrategyProbation  # noqa: E402
from forex_ai_agent.data.models.research import ResearchTask  # noqa: E402

BUDGET = {
    "hypothesis_llm_calls": 1,
    "director_cycles": 2,
    "director_llm_calls": 2,       # 1 per cycle
    "experiment_llm_calls": 1,
    "report_llm_calls": 1,
    "max_total_llm": 5,
}
USED = {"llm_calls": 0}
PAIR = "EUR/USD"
SESSION_START = datetime.utcnow()
CREATED_VERSIONS: list = []
ENQUEUED_ITEM_IDS: list = []
CYCLE_IDS: list = []


def _hdr(title: str) -> None:
    print(f"\n──────── {title} ────────")


def _acc_budget(llm_used: bool, kind: str) -> None:
    if llm_used:
        USED["llm_calls"] += 1
    assert USED["llm_calls"] <= BUDGET["max_total_llm"], "LLM budget exceeded"


async def _cleanup() -> None:
    """Remove every side effect this session created. Best-effort, loud."""
    _hdr("CLEANUP (default: leave nothing behind)")
    factory = get_session_factory()

    # Strategy-version files created by the Director (root versions — no
    # children_ids links in pre-existing files to repair).
    removed_files = 0
    for vid in CREATED_VERSIONS:
        p = Path("strategies/versions") / f"{vid}.json"
        if p.exists():
            p.unlink()
            removed_files += 1
    print(f"strategy files removed: {removed_files} ({CREATED_VERSIONS})")

    async with factory() as session:
        # Director queue rows we enqueued (+ their events).
        if ENQUEUED_ITEM_IDS:
            items = (
                (await session.execute(select(DirectorQueueItem))).scalars().all()
            )
            ids = [i.id for i in items if str(i.id) in set(ENQUEUED_ITEM_IDS)]
            for i in ids:
                await session.execute(delete(DirectorQueueItem).where(DirectorQueueItem.id == i))
            events = (await session.execute(select(DirectorEvent))).scalars().all()
            ev_ids = [
                e.id for e in events
                if str(getattr(e, "item_id", "") or "") in set(ENQUEUED_ITEM_IDS)
                or (e.detail or {}).get("item_id") in set(ENQUEUED_ITEM_IDS)
                or e.cycle_id in set(CYCLE_IDS)
            ]
            for e in ev_ids:
                await session.execute(delete(DirectorEvent).where(DirectorEvent.id == e))
            print(f"director queue rows removed: {len(ids)}; events removed: {len(ev_ids)}")

        # Research tasks logged by this session's Director runs (the row's
        # concept column is concept_name; filter by this session's clock).
        tasks = (
            (await session.execute(select(ResearchTask))).scalars().all()
        )
        doomed = [
            t for t in tasks
            if t.created_at and t.created_at >= SESSION_START
        ]
        for t in doomed:
            await session.execute(delete(ResearchTask).where(ResearchTask.id == t.id))
        print(f"research-task rows removed: {len(doomed)}")

        # Probation rows for versions we created.
        if CREATED_VERSIONS:
            probs = (await session.execute(select(StrategyProbation))).scalars().all()
            pdoomed = [p for p in probs if str(p.version_id) in set(CREATED_VERSIONS)]
            for p in pdoomed:
                await session.execute(delete(StrategyProbation).where(StrategyProbation.id == p.id))
            print(f"probation rows removed: {len(pdoomed)}")

        await session.commit()

    # The report row (Uuid pk → uuid object, C3's cleanup lesson).
    if SESSION_STATE.get("report_id"):
        from forex_ai_agent.agent.report_gen import get_research_report

        rid = SESSION_STATE["report_id"]
        import uuid as _uuid

        async with factory() as session:
            try:
                row = await session.get(ResearchReport, _uuid.UUID(str(rid)))
                if row is not None:
                    await session.delete(row)
                    await session.commit()
                    print("report row removed:", str(rid)[:8])
            except (TypeError, ValueError):
                print("report row cleanup skipped (bad id)")


SESSION_STATE: dict = {}


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    print("UNATTENDED RESEARCH SESSION — Stage C accept criterion")
    print("constitution: the LLM proposes/narrates; deterministic code owns")
    print("every number, every gate, every cap. Budget:", BUDGET)

    # ── 1. C1: hypotheses ────────────────────────────────────────────────
    _hdr("STEP 1 · C1 hypothesis generation")
    hg = await generate_hypotheses(
        HypothesisGenConfig(pairs=[PAIR], use_llm=True, enqueue=True, max_hypotheses=2)
    )
    _acc_budget(hg["llm_used"], "hypotheses")
    print("llm_used:", hg["llm_used"], "| fallback:", hg["fallback_used"], "| error:", hg["llm_error"])
    for a in hg["accepted"]:
        print(f"  [enqueued={a.get('enqueued')}] {a['pair']}: {a['text'][:80]}…")
    for r in hg["rejected"]:
        print("  [rejected]", r["pair"], "—", r["reason"])
    ENQUEUED_ITEM_IDS.extend(a["item_id"] for a in hg["accepted"] if a.get("enqueued"))
    print("enqueued item ids:", [i[:8] for i in ENQUEUED_ITEM_IDS])

    # ── 2. B4: Director cycles ───────────────────────────────────────────
    _hdr("STEP 2 · Director cycles (spec → backtest gate → version + probation)")
    for n in range(BUDGET["director_cycles"]):
        rep = await run_cycle(
            DirectorConfig(
                max_items_per_cycle=2,
                max_llm_calls_per_cycle=1,
                use_llm=True,
                history_days=180,
            )
        )
        CYCLE_IDS.append(rep["cycle_id"])
        _acc_budget(rep["llm_calls_used"] > 0, "director")
        print(
            f"cycle {rep['cycle_id']}: items={rep['items_run']} validated={rep['validated']}"
            f" rejected={rep['rejected']} errors={rep['errors']}"
            f" llm={rep['llm_calls_used']} exhausted={rep['budget_exhausted']}"
        )
        for r in rep["results"]:
            vid = r.get("version_id")
            if vid:
                CREATED_VERSIONS.append(str(vid))
            print(f"  · {r.get('pair')}: {r.get('status')}" + (f" → version {str(vid)[:8]}" if vid else ""))
    snap = await queue_snapshot()
    print("open items after cycles:", snap["open_items"], "| by_status:", snap["by_status"])

    # ── 3. C2: experiment selection ──────────────────────────────────────
    _hdr("STEP 3 · C2 experiment selection (real engines)")
    sel = await select_experiments(
        ExperimentSelConfig(
            symbol=PAIR,
            max_experiments=1,
            max_oos_windows=2,
            window_months=3,
            step_months=2,
            use_llm=True,
        )
    )
    _acc_budget(sel["llm_used"], "experiments")
    print("llm_used:", sel["llm_used"], "| fallback:", sel["fallback_used"], "| error:", sel["llm_error"])
    for a in sel["accepted"]:
        outcome = a.get("experiment") or a.get("walk_forward") or {}
        print(
            f"  [{a.get('source')}] {a['strategy_id']} mutations={a.get('parameter_mutations')}"
            f" status={a.get('status')} verdict={outcome.get('verdict')}"
        )
    for r in sel["rejected"][:6]:
        print("  [rejected]", r.get("strategy_id"), "—", r.get("reason"))

    # ── 4. C3: written report ────────────────────────────────────────────
    _hdr("STEP 4 · C3 written report (LLM narrates, records own the numbers)")
    doc = await generate_research_report(
        ReportConfig(topic=f"Unattended session {SESSION_START:%Y-%m-%d %H:%M}", use_llm=True, persist=True)
    )
    _acc_budget(doc["llm_used"], "report")
    print("llm_used:", doc["llm_used"], "| error:", doc["llm_error"])
    print("evidence ledger:", len(doc.get("evidence") or []), "record(s)")
    for s in doc.get("sections") or []:
        first = (s.get("narrative") or "").splitlines()[0][:90]
        print(f"  [{s.get('source')}] {s.get('heading')}: {first}")
    tr = doc.get("traceability") or {}
    print("sections_replaced:", tr.get("sections_replaced"), "| discarded:", tr.get("discarded_sections"))
    SESSION_STATE["report_id"] = doc.get("report_id")
    print("report_id:", str(doc.get("report_id"))[:8], "(traceable to stored records; row cleaned up after)")

    # ── Summary ──────────────────────────────────────────────────────────
    _hdr("SESSION SUMMARY (every line cites the stage reports above)")
    summary = {
        "hypotheses_enqueued": len(ENQUEUED_ITEM_IDS),
        "director_cycles": len(CYCLE_IDS),
        "versions_created": CREATED_VERSIONS,
        "experiments_run": len([a for a in sel["accepted"] if a.get("status") == "RUN"]),
        "report_id": str(SESSION_STATE.get("report_id")),
        "llm_calls_used": USED["llm_calls"],
        "llm_budget": BUDGET["max_total_llm"],
    }
    print(json.dumps(summary, indent=2))
    print("\nconstitutional close-out: the LLM chose what to research, what to")
    print("test, and how to say it — the deterministic gates decided what")
    print("survived, what was rejected, and what every number means.")

    keep = os.environ.get("ALEX_KEEP") == "1"
    if keep:
        print("\nALEX_KEEP=1 → side effects KEPT for inspection (report row,")
        print("queue rows, strategy files, probation + research-task rows).")
    else:
        await _cleanup()
        snap2 = await queue_snapshot()
        print("\npost-cleanup open director items:", snap2["open_items"])

    out = {"summary": summary, "hypotheses": hg, "cycles": len(CYCLE_IDS),
           "experiments": sel, "report": {k: doc.get(k) for k in ("report_id", "title", "llm_used", "sections", "traceability")}}
    with open("scripts/stage_c_session_results.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
