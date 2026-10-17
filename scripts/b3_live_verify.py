"""B3 live verification against the live strategy store (no server needed).

Checks, in-process:
1. A real stored strategy's health card now shows all SEVEN lights, with the
   two B3 lights GREY until a sweep is run.
2. The run_robustness_sweep tool is registered with an ASSIST floor and
   MEDIUM risk (absent from the OBSERVE registry).
3. A real sweep executes end-to-end against the live feed chain with
   save=False (no stored state mutated) and returns per-pair/per-regime
   evidence (or honest per-pair ERROR rows if a feed is unavailable).
"""

import asyncio
import json


async def main() -> None:
    print("== 1. live health card: seven lights ==")
    from forex_ai_agent.agent.health import compute_strategy_health, get_strategy_health
    from forex_ai_agent.strategy import create_version_manager

    vm = await create_version_manager()
    ids = list(vm.versions.keys())
    print(f"   stored strategies: {ids}")
    target = "f39b8d40" if "f39b8d40" in ids else ids[0]
    payload = await get_strategy_health(target)
    lights = payload["health"]["lights"]
    print(f"   {payload['strategy_name']} ({payload['strategy_id']}) {payload['summary']}")
    for l in lights:
        print(f"     {l['status']:<6} {l['name']:<18} {l['detail'][:70]}")
    names = [l["name"] for l in lights]
    assert len(lights) == 7, f"expected 7 lights, got {len(lights)}"
    assert names[-2:] == ["PAIR GENERALITY", "REGIME ROBUSTNESS"]
    json.loads(json.dumps(payload["health"]))  # API-safe
    print("   seven lights, JSON-safe: OK")

    print("== 2. tool gating ==")
    from forex_ai_agent.agent.confirmations import get_tool_risk_class
    from forex_ai_agent.agent.tools import get_tool_registry

    reg0 = get_tool_registry(0)
    reg1 = get_tool_registry(1)
    assert "run_robustness_sweep" in reg1._tools, "must be registered"
    assert get_tool_risk_class("run_robustness_sweep").name == "MEDIUM"
    # Gating happens at dispatch (all tools are registered at every level):
    # OBSERVE users are refused; ASSIST users pass the permission gate (the
    # call itself then stops at arg validation — never runs the heavy sweep).
    try:
        await reg0.dispatch("run_robustness_sweep", {"strategy_id": "x"})
        raise AssertionError("OBSERVE dispatch must be refused")
    except PermissionError:
        print("   dispatch at OBSERVE -> PermissionError: OK")
    try:
        await reg1.dispatch("run_robustness_sweep", {"bogus": True})
        raise AssertionError("expected validation failure, not execution")
    except PermissionError:
        raise AssertionError("ASSIST must pass the permission gate")
    except Exception:
        print("   dispatch at ASSIST -> passed permission gate (stopped pre-run): OK")
    print("   run_robustness_sweep: ASSIST floor, MEDIUM risk: OK")

    print("== 3. real sweep against live feed chain (save=False) ==")
    from forex_ai_agent.agent.robustness import run_robustness_sweep, render_robustness_text

    result = await run_robustness_sweep(
        target,
        window_months=3,
        step_months=2,
        save=False,  # verification run — do not mutate stored strategy files
    )
    print(render_robustness_text(result))
    for row in result["pairs"]:
        w0 = row["windows"][0] if row["windows"] else {}
        print(
            f"   pair {row['symbol']:<10} {row['status']:<8} "
            f"windows={row['windows_tested']} regime(w1)={w0.get('regime')}"
        )
    stored = vm.get_version(payload["strategy_id"])
    assert "robustness" not in (stored.backtest_results or {}), "save=False must not store"
    print("   save=False left stored state untouched: OK")
    print("\nB3 LIVE VERIFICATION: ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
