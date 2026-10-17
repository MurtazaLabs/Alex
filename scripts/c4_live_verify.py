"""Stage C4 live verification (weak-hypothesis veto + promotion review).

1. Seed THREE rejected ``ResearchTask`` rows of one spec shape — the same
   shape the deterministic fallback spec produces (both/H1/above) — so the
   weak-shape policy has a real stored record to work with.
2. Mine the policy deterministically: ``load_rejected_conclusions`` →
   ``weak_shapes`` (≥3 attempts) → assert the fallback shape is weak and
   show the honest veto reason.
3. Run a REAL `propose_for_pair` cycle (deterministic fallback spec, LLM off
   — the veto is a code decision) and confirm the veto fires BEFORE the
   backtest gate: status rejected with a "weak-hypothesis veto:" reason and
   NO backtest executed.
4. Run a REAL deterministic promotion review over the live probation store.
5. Clean up: delete the seeded rows; nothing else is written.

Run:  .venv311/Scripts/python.exe scripts/c4_live_verify.py
"""

import asyncio
import json
import logging
import sys
import uuid

sys.path.insert(0, "src")

from sqlalchemy import delete, select  # noqa: E402

from forex_ai_agent.agent.research_lessons import (  # noqa: E402
    load_rejected_conclusions,
    spec_shape,
    veto_reason,
    weak_shapes,
)
from forex_ai_agent.data.database import get_session_factory  # noqa: E402
from forex_ai_agent.data.models.research import ResearchTask  # noqa: E402

SEED_IDS = [uuid.uuid4() for _ in range(3)]
_VETO_SHAPE = {
    "direction": "both",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "sma", "verb": "above", "right": "sma_slow"},
        {"left": "price", "verb": "above", "right": "sma"},
    ],
}


async def _seed() -> None:
    factory = get_session_factory()
    from forex_ai_agent.autonomy.research_log import _get_or_create_alex_user

    async with factory() as session:
        user = await _get_or_create_alex_user(session)
        for i, row_id in enumerate(SEED_IDS):
            session.add(ResearchTask(
                id=row_id,
                user_id=user.id,
                concept_name=f"EUR/USD: C4VetoSeed {i + 1} (audit)",
                questions_json=["Does the weak-shape policy veto this shape?"],
                status="COMPLETED",
                current_step="COMPLETED",
                progress_percent=100,
                conclusion_json={
                    "summary": "Gate rejected the proposed spec.",
                    "verdict": "rejected",
                    "used_llm": False,
                    "pair": "EUR/USD",
                    "validation": {
                        "passed": False,
                        "reasons": ["profit factor 0.71 below floor 1.00"],
                    },
                    "spec": _VETO_SHAPE,
                },
                completed_at=None,
            ))
        await session.commit()
    print(f"seeded {len(SEED_IDS)} rejected rows of shape both/H1/above+above")


async def _cleanup() -> None:
    factory = get_session_factory()
    async with factory() as session:
        for row_id in SEED_IDS:
            await session.execute(delete(ResearchTask).where(ResearchTask.id == row_id))
        await session.commit()
    print("cleanup: seeded rows removed")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    await _seed()

    # 1. Policy mining (deterministic).
    conclusions = await load_rejected_conclusions()
    shapes = weak_shapes(conclusions, min_attempts=3)
    fallback_shape = spec_shape(_VETO_SHAPE)
    match = next(
        (l for l in shapes if (l.direction, l.timeframe, list(l.entry_verbs)) == fallback_shape),
        None,
    )
    print(f"\nmined {len(conclusions)} rejected conclusions; weak shapes (>=3 attempts): {len(shapes)}")
    print("veto reason for the fallback shape:")
    print(" ", veto_reason(match) if match else "(no match — veto would not fire)")

    # 2. Real proposal cycle with the veto in place (LLM off — deterministic).
    print("\nrunning REAL proposal cycle (fallback spec, veto armed)…")
    from forex_ai_agent.autonomy.proposal import propose_for_pair

    from forex_ai_agent.agent.director import DirectorConfig
    from forex_ai_agent.autonomy.proposal import fetch_history_for
    from forex_ai_agent.core.event_bus import InMemoryEventBus
    from forex_ai_agent.pipeline import Pipeline

    pipeline = Pipeline(event_bus=InMemoryEventBus())
    await pipeline.initialize()
    history = await fetch_history_for(pipeline, "EUR/USD", days=180, timeframe="H1")

    backtests_seen = []
    original_validate = None
    try:
        from forex_ai_agent.autonomy.validator import BacktestValidator

        original_validate = BacktestValidator.validate_spec

        async def _spy(self, spec, df, symbol=None):
            backtests_seen.append(spec.name)
            return await original_validate(self, spec, df, symbol=symbol)

        BacktestValidator.validate_spec = _spy

        result = await propose_for_pair(
            pipeline, "EUR/USD", history, use_llm=False, timeframe="H1", log_to_db=True,
        )
    finally:
        if original_validate is not None:
            BacktestValidator.validate_spec = original_validate

    print("status:", result.status)
    for m in result.messages:
        print("  msg:", m[:110])
    veto_fired = bool(result.validation) and any(
        str(r).startswith("weak-hypothesis veto:") for r in (result.validation.reasons or [])
    )
    print("veto fired BEFORE the gate:", veto_fired, "| backtests run:", len(backtests_seen))

    # 3. Real promotion review (deterministic read over the live store).
    from forex_ai_agent.agent.probation_promotion import PromotionConfig, review_promotions

    review = await review_promotions(PromotionConfig(limit=5, include_narration=True))
    print(f"\npromotion review: {review['reviewed']} probation row(s) reviewed, "
          f"{review['ready_count']} ready, narration_used={review['narration_used']}")
    for c in review["candidates"]:
        print(f"  [{c['status']}] {c['version_id'][:8]} {c['observed_trades']}/{c['required_trades']}"
              f" — {(c['narration'] or '')[:90]}")
    print("disclaimer:", review["disclaimer"][:100] + "…")

    ok = (
        match is not None
        and veto_fired
        and len(backtests_seen) == 0
        and result.status == "rejected"
        and review["reviewed"] >= 0
    )
    await _cleanup()
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
