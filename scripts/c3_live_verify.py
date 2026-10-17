"""C3 live verification: real evidence → real LLM narration → traceability →
persisted report row → cleanup.

Pass 1 (persist=False): create a REAL experiment in-process (VariantRunner
records are process-local), then generate a report over it with the real LLM.
Nothing is written to the DB.
Pass 2 (persist=True): generate + persist a real research_reports row, read
it back from storage, then DELETE it (cleanup) — the live DB stays untouched.

Run:  .venv311/Scripts/python.exe scripts/c3_live_verify.py
"""

import asyncio
import json
import logging
import sys

sys.path.insert(0, "src")

from forex_ai_agent.agent.experiments import create_experiment  # noqa: E402
from forex_ai_agent.agent.report_gen import (  # noqa: E402
    ReportConfig,
    generate_research_report,
    get_research_report,
    list_research_reports,
)
from forex_ai_agent.data.database import get_session_factory  # noqa: E402
from forex_ai_agent.data.models.base import Base  # noqa: E402


def _show(doc: dict) -> None:
    print("\ntitle:", doc.get("title"))
    print("llm_used:", doc.get("llm_used"), "| llm_error:", doc.get("llm_error"))
    print("evidence ledger:", len(doc.get("evidence") or []), "record(s)")
    for e in (doc.get("evidence") or [])[:3]:
        print("  -", e["id"], "·", e["summary"])
    for s in doc.get("sections") or []:
        first = (s.get("narrative") or "").splitlines()[0][:100]
        print(f"  [{s.get('source')}] {s.get('heading')}: {first}")
    print("sections_replaced:", (doc.get("traceability") or {}).get("sections_replaced"))


async def _cleanup(report_id: str) -> None:
    """Delete this run's row (and any rows left by earlier failed runs).

    The Uuid pk column needs uuid.UUID objects, not strings.
    """
    from sqlalchemy import delete, select

    from forex_ai_agent.agent.report_gen import ResearchReport

    factory = get_session_factory()
    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(ResearchReport).where(
                        ResearchReport.topic.like("C3 live verification%")
                    )
                )
            )
            .scalars()
            .all()
        )
        ids = [r.id for r in rows if str(r.id).startswith(report_id[:8]) or r.topic.startswith("C3 live verification")]
        for rid in ids:
            await session.execute(delete(ResearchReport).where(ResearchReport.id == rid))
        await session.commit()
        print(f"cleanup: removed {len(ids)} report row(s)")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    # Real evidence: one real experiment through the deterministic engine.
    exp = await create_experiment(
        strategy_id="f39b8d40",
        param_mutations={"atr_sl_mult": 2.0},
        symbol="EUR/USD",
        timeframe="H1",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    exp_id = str(exp.get("experiment_id"))
    print("seeded real experiment:", exp_id, "| verdict:", exp.get("verdict"))

    ok = True

    # Pass 1 — real LLM narration over real evidence, nothing persisted.
    doc = await generate_research_report(
        ReportConfig(topic="C3 live verification", experiment_ids=[exp_id], persist=False),
    )
    _show(doc)
    if not doc["llm_used"] and not doc.get("llm_error"):
        print("FAIL: neither LLM nor fallback produced narration")
        ok = False
    if not doc.get("evidence"):
        print("FAIL: empty evidence ledger")
        ok = False
    bad_sources = [s for s in doc["sections"] if s["source"] not in ("llm", "fallback")]
    if bad_sources:
        print("FAIL: unknown section sources", bad_sources)
        ok = False

    # Pass 2 — persist a real row, read it back, clean up.
    doc2 = await generate_research_report(
        ReportConfig(topic="C3 live verification (persist)", experiment_ids=[exp_id], persist=True),
    )
    rid = doc2.get("report_id")
    print("\npersisted report_id:", rid)
    if not rid:
        print("FAIL: report row was not persisted")
        return 1
    saved = await get_research_report(rid)
    if not saved or saved.get("title") != doc2.get("title"):
        print("FAIL: persisted row does not round-trip")
        ok = False
    else:
        print("round-trip OK — title:", saved.get("title"))
    recent = await list_research_reports(limit=5)
    print("recent reports in storage:", len(recent))
    await _cleanup(rid)
    after = await list_research_reports(limit=5)
    print("after cleanup:", len(after), "report row(s) in storage")
    if any(r["report_id"] == rid for r in after):
        print("FAIL: cleanup did not remove the row")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    with open("scripts/c3_live_results.json", "w", encoding="utf-8") as fh:
        json.dump({"pass1": doc, "pass2": doc2}, fh, indent=2, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
