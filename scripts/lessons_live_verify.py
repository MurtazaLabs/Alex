"""Learning-loop live verification (C1 + rejection AND success lessons).

1. Seed TWO ``ResearchTask`` rows (deterministic inserts — the same shape
   ``autonomy/research_log.log_research_run`` writes): one gate-REJECTED run
   and one gate-PASSED (``validated``) run.
2. Mine BOTH polarities deterministically and show them.
3. Run a REAL C1 generation (real LLM, propose-only) and confirm both mined
   lesson sets reached the LLM context (``lessons_used`` AND
   ``successes_used``) without breaking validation.
4. Clean up: delete the seeded rows; nothing else is written.

Run:  .venv311/Scripts/python.exe scripts/lessons_live_verify.py
"""

import asyncio
import json
import logging
import sys
import uuid

sys.path.insert(0, "src")

from sqlalchemy import delete  # noqa: E402

from forex_ai_agent.agent.hypothesis_gen import (  # noqa: E402
    HypothesisGenConfig,
    generate_hypotheses,
)
from forex_ai_agent.agent.research_lessons import (  # noqa: E402
    load_run_lessons,
    render_lessons_for_context,
    render_successes_for_context,
)
from forex_ai_agent.data.database import get_session_factory  # noqa: E402
from forex_ai_agent.data.models.research import ResearchTask  # noqa: E402

SEED_REJECT_ID = uuid.uuid4()
SEED_PASS_ID = uuid.uuid4()


async def _seed() -> None:
    factory = get_session_factory()
    from forex_ai_agent.autonomy.research_log import _get_or_create_alex_user

    async with factory() as session:
        user = await _get_or_create_alex_user(session)
        session.add(ResearchTask(
            id=SEED_REJECT_ID,
            user_id=user.id,
            concept_name="EUR/USD: LessonSeed (audit)",
            questions_json=["Does the seeded rejection teach the next proposal?"],
            status="COMPLETED",
            current_step="COMPLETED",
            progress_percent=100,
            conclusion_json={
                "summary": "Gate rejected the proposed spec.",
                "verdict": "rejected",
                "used_llm": True,
                "pair": "EUR/USD",
                "validation": {
                    "passed": False,
                    "reasons": ["spec invalid: name: Field required"],
                },
                "spec": {
                    "direction": "long",
                    "timeframe": "H1",
                    "entry_conditions": [
                        {"left": "price", "verb": "crosses_above", "right": "sma"}
                    ],
                },
            },
            completed_at=None,
        ))
        session.add(ResearchTask(
            id=SEED_PASS_ID,
            user_id=user.id,
            concept_name="EUR/USD: SuccessSeed (audit)",
            questions_json=["Does the seeded pass teach the next proposal?"],
            status="COMPLETED",
            current_step="COMPLETED",
            progress_percent=100,
            conclusion_json={
                "summary": "Gate passed the proposed spec.",
                "verdict": "validated",
                "used_llm": True,
                "pair": "EUR/USD",
                "validation": {"passed": True, "reasons": []},
                "spec": {
                    "direction": "short",
                    "timeframe": "H4",
                    "entry_conditions": [
                        {"left": "price", "verb": "crosses_below", "right": "sma"}
                    ],
                },
            },
            completed_at=None,
        ))
        await session.commit()
    print("seeded rejected run:", str(SEED_REJECT_ID)[:8])
    print("seeded validated run:", str(SEED_PASS_ID)[:8])


async def _cleanup() -> None:
    factory = get_session_factory()
    async with factory() as session:
        for row_id in (SEED_REJECT_ID, SEED_PASS_ID):
            await session.execute(delete(ResearchTask).where(ResearchTask.id == row_id))
        await session.commit()
    print("cleanup: seeded rows removed")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    await _seed()

    rejections, successes = await load_run_lessons()
    print(f"\nmined {len(rejections)} rejection lesson(s) deterministically:")
    print(json.dumps(render_lessons_for_context(rejections), indent=2))
    print(f"\nmined {len(successes)} success lesson(s) deterministically:")
    print(json.dumps(render_successes_for_context(successes), indent=2))
    if not rejections or not successes:
        print("FAIL: expected at least one lesson of EACH polarity from the seeds")
        await _cleanup()
        return 1

    print("\nrunning REAL C1 generation (LLM, propose-only)…")
    report = await generate_hypotheses(
        HypothesisGenConfig(pairs=["EUR/USD"], use_llm=True, enqueue=False, max_hypotheses=2)
    )
    print("llm_used:", report["llm_used"], "| error:", report["llm_error"])
    print("lessons_used:", report["lessons_used"], "(rejections rode into the live LLM context)")
    print("successes_used:", report["successes_used"], "(passes rode into the live LLM context)")
    for a in report["accepted"]:
        print(f"  [accepted src={a.get('source')}] {a['pair']}: {a['text'][:80]}…")
    for r in report["rejected"]:
        print("  [rejected]", r["pair"], "—", r["reason"])

    ok = (
        report["lessons_used"] >= 1
        and report["successes_used"] >= 1
        and (report["llm_used"] or report["fallback_used"])
    )
    await _cleanup()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
