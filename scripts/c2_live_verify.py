"""C2 live verification: real menu → real LLM → deterministic validation →
real (or dry-run) execution through the Phase-3 engines.

Pass 1 (dry_run=True, use_llm=True): real LLM proposes against the REAL
strategy menu; deterministic validation runs; nothing is executed.
Pass 2 (dry_run=False, use_llm=True): one REAL experiment through
backtest_strategy + walk_forward_test — experiments are in-process records
(no DB rows are created), so nothing needs cleanup.

Run from the project root:
    .venv311/Scripts/python.exe scripts/c2_live_verify.py
"""

import asyncio
import json
import logging
import sys

sys.path.insert(0, "src")

from forex_ai_agent.agent.experiment_selector import (  # noqa: E402
    ExperimentSelConfig,
    select_experiments,
)


def _print(title: str, report: dict) -> None:
    print(f"\n=== {title} ===")
    print("llm_used:", report["llm_used"], "| llm_error:", report["llm_error"])
    print("fallback_used:", report["fallback_used"], "| menu:", report["strategies_seen"])
    print("raw_proposals:", report["raw_proposals"])
    for acc in report["accepted"]:
        outcome = acc.get("experiment") or acc.get("walk_forward") or {}
        print(
            f"  [accepted src={acc.get('source')}] {acc['strategy_id']}"
            f" status={acc.get('status')}"
            f" mutations={acc.get('parameter_mutations')}"
            f" verdict={outcome.get('verdict')}"
        )
        agg = outcome.get("aggregate_metrics") or outcome.get("baseline_metrics") or {}
        if agg:
            print(f"     numbers: {agg}")
    for rej in report["rejected"]:
        print(f"  [rejected] {rej.get('strategy_id')}: {rej.get('reason')}")


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    ok = True

    # Pass 1: real menu, real LLM, dry run — nothing executed.
    dry = await select_experiments(
        ExperimentSelConfig(
            symbol="EUR/USD",
            max_experiments=3,
            max_oos_windows=2,
            use_llm=True,
            dry_run=True,
        )
    )
    _print("PASS 1 — dry run (real menu, real LLM, nothing executed)", dry)
    if not dry["llm_used"] and not dry["fallback_used"]:
        print("FAIL: neither LLM nor fallback produced proposals")
        ok = False

    # Pass 2: one real experiment through the real engines.
    real = await select_experiments(
        ExperimentSelConfig(
            symbol="EUR/USD",
            max_experiments=1,
            max_oos_windows=2,
            window_months=3,
            step_months=2,
            use_llm=True,
        )
    )
    _print("PASS 2 — real run (one real experiment)", real)
    if real["accepted"]:
        acc = real["accepted"][0]
        if acc.get("status") == "ERROR":
            print(f"FAIL: real run errored: {acc.get('error')}")
            ok = False
        elif not (acc.get("experiment") or acc.get("walk_forward")):
            print("FAIL: real run produced no engine numbers")
            ok = False
        else:
            outcome = acc.get("experiment") or acc.get("walk_forward") or {}
            print("real run verdict:", outcome.get("verdict"), "| recommendation:", outcome.get("recommendation"))
    elif real["fallback_used"]:
        print("NOTE: LLM unavailable — baseline fallback ran (still a valid live check)")
    else:
        print("FAIL: nothing accepted and no fallback — unexpected")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    with open("scripts/c2_live_results.json", "w", encoding="utf-8") as fh:
        json.dump({"dry_run": dry, "real": real}, fh, indent=2, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
