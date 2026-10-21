"""E4 window runner daemon -- the script-owned live loop for the window.

Per the constitution pin, the application package must never construct a
non-shadow adapter; the single live vehicle is an explicit,
human-authorized script. This is it. It refuses to do anything unless:

* the window state is active (``e4_window_start.py --start``),
* an active SCOPE_WINDOW sign-off covers this pair/version,
* the version's approved spec hash still matches the rules on disk,
* the MT5 terminal connects at level 4 with shadow OFF,
* the market data feed is not the mock source.

Every cycle is fail-closed: signals become policy intents (audit log),
and intents reach the wire ONLY after a human approval record exists
(``e4_window_approve.py``). All outcomes are appended to audit logs.

The mandate risk gate runs FIRST each cycle: baselines (day/week start
equity, equity high-water mark) live in ``logs/live_window/
risk_baselines.json``. A breached mandate limit halts new intents:
daily/weekly resume on the next day/week; a drawdown halt resumes ONLY
after a human runs ``--clear-trip drawdown``.

ASCII output only (Windows console). Usage:

  python scripts/e4_window_run.py --once
  python scripts/e4_window_run.py --poll-seconds 60
  python scripts/e4_window_run.py --risk-status
  python scripts/e4_window_run.py --clear-trip drawdown --reason "postmortem done"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.brokers import MT5Adapter  # noqa: E402
from forex_ai_agent.config import get_settings  # noqa: E402
from forex_ai_agent.core.permission_manager import (  # noqa: E402
    PermissionLevel,
    PermissionManager,
)
from forex_ai_agent.data.providers.factory import get_feed_chain  # noqa: E402
from forex_ai_agent.execution.governor import (  # noqa: E402
    ExecutionGovernor,
    GovernorLimits,
)
from forex_ai_agent.execution.live import (  # noqa: E402
    DEFAULT_LIVE_CAPS,
    PositionTracker,
)
from forex_ai_agent.execution.live.risk_state import (  # noqa: E402
    RISK_TRIP_NAMES,
    RiskBaselineStore,
)
from forex_ai_agent.execution.live.window import (  # noqa: E402
    DEFAULT_WINDOW_STATE_DIR,
)
from forex_ai_agent.execution.live.window_runner import (  # noqa: E402
    LiveWindowRunner,
)
from forex_ai_agent.execution.policy import (  # noqa: E402
    ExecutionPolicy,
    PolicyCaps,
)
from forex_ai_agent.execution.tokens import ApprovalTokenStore  # noqa: E402

PAIR = "EUR/USD"
VERSION_ID = "3f425957"


def _risk_store() -> RiskBaselineStore:
    return RiskBaselineStore(Path(DEFAULT_WINDOW_STATE_DIR) / "risk_baselines.json")


def _risk_status() -> int:
    store = _risk_store()
    snap = store.snapshot()
    print("\n=== E4 RISK BASELINES (mandate v1.0) ===")
    print(f"state file:        {store.path}")
    print(f"day key:           {snap.day_key or '(never synced)'}")
    print(f"day start equity:  {snap.day_start_equity:.2f}")
    print(f"week key:          {snap.week_key or '(never synced)'}")
    print(f"week start equity: {snap.week_start_equity:.2f}")
    print(f"equity high mark:  {snap.equity_hwm:.2f}")
    trips = snap.active_trips()
    if trips:
        print(f"ACTIVE TRIPS:      {', '.join(trips)}")
        for name in trips:
            detail = snap.trips.get(name, {})
            print(f"  {name}: {detail}")
    else:
        print("ACTIVE TRIPS:      none")
    print("clear with: --clear-trip {daily|weekly|drawdown} --reason \"...\"")
    return 0


def _clear_trip(name: str, reason: str) -> int:
    if name not in RISK_TRIP_NAMES:
        print(f"REFUSED: unknown trip {name!r}; expected {RISK_TRIP_NAMES}")
        return 2
    if not reason.strip():
        print("REFUSED: --reason is required (this is a human decision record)")
        return 2
    store = _risk_store()
    snap = store.clear_trip(name, by="operator", reason=reason.strip())
    print(f"cleared trip: {name} (baseline rebased to current equity on next sync)")
    print(f"reason:       {reason.strip()}")
    print(f"remaining:    {snap.active_trips() or 'none'}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    if args.risk_status:
        return _risk_status()
    if args.clear_trip:
        return _clear_trip(args.clear_trip, args.reason or "")

    settings = get_settings()
    if int(settings.max_permission_level) < PermissionLevel.AUTONOMOUS:
        print("REFUSED: MAX_PERMISSION_LEVEL < 4 (no write permission granted)")
        return 2

    if get_feed_chain().active_source == "mock":
        print("REFUSED: market data source is 'mock' -- a live loop needs real bars")
        return 2

    ctor_kwargs = {}
    if settings.mt5_login and settings.mt5_password and settings.mt5_server:
        ctor_kwargs.update(
            login=int(settings.mt5_login),
            password=settings.mt5_password,
            server=settings.mt5_server,
        )
    if settings.mt5_path:
        ctor_kwargs["path"] = settings.mt5_path

    adapter = MT5Adapter(
        PermissionManager(PermissionLevel.AUTONOMOUS),
        shadow_mode=False,
        **ctor_kwargs,
    )
    try:
        adapter.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"REFUSED: cannot connect MT5 terminal: {exc}")
        return 2

    account = adapter.account()
    print("\n=== E4 WINDOW RUNNER (script-owned live loop) ===")
    print(f"account: #{account.login} {account.server} "
          f"equity={account.equity:.2f} {account.currency}")
    print(f"pair:    {PAIR}  version: {VERSION_ID}  "
          f"lots: {DEFAULT_LIVE_CAPS.min_lot}-{DEFAULT_LIVE_CAPS.max_lot}")

    policy = ExecutionPolicy(PolicyCaps(
        min_lot_size=DEFAULT_LIVE_CAPS.min_lot,
        max_lot_size=DEFAULT_LIVE_CAPS.max_lot,
    ))
    governor = ExecutionGovernor(GovernorLimits(
        daily_loss_limit=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_daily_loss_pct / 100.0),
        max_drawdown=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_drawdown_pct / 100.0),
        max_positions=DEFAULT_LIVE_CAPS.max_concurrent_positions,
        weekly_loss_limit=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_weekly_loss_pct / 100.0),
    ))
    governor.arm()
    tokens = ApprovalTokenStore()
    tracker = PositionTracker(symbol=PAIR)
    runner = LiveWindowRunner(
        adapter=adapter, feed=get_feed_chain(), tracker=tracker,
        policy=policy, governor=governor, tokens=tokens,
        version_id=VERSION_ID,
    )

    cycles = 0
    try:
        while args.max_cycles == 0 or cycles < args.max_cycles:
            summary = await runner.run_once()
            status = summary.get("status")
            if status == "halted":
                trips = ", ".join(summary.get("trips", []))
                print(f"[cycle {cycles + 1}] HALTED: mandate risk halt "
                      f"({trips}) equity={summary.get('equity'):.2f}")
                print("daily/weekly halts resume on the next day/week "
                      "(just restart); a drawdown halt needs: "
                      "--clear-trip drawdown --reason \"...\"")
                return 3
            if status == "blocked":
                print(f"[cycle {cycles + 1}] BLOCKED: {summary.get('reason')}")
                if cycles == 0:
                    print("fail-closed: runner will not loop while blocked")
                    return 2
            elif status == "no_data":
                print(f"[cycle {cycles + 1}] no_data bars={summary.get('bars')} "
                      f"equity={summary.get('equity')}")
            else:
                print(
                    f"[cycle {cycles + 1}] ok equity={summary.get('equity'):.2f} "
                    f"positions={summary.get('open_positions')} "
                    f"signals={summary.get('signals')} "
                    f"intents={summary.get('intents_valid')}"
                    f"/{summary.get('intents_refused')} "
                    f"dispatched={summary.get('dispatched')}"
                )
                if summary.get("note"):
                    print(f"    note: {summary['note']}")
            cycles += 1
            if summary.get("critical"):
                print("CRITICAL: transmitter halted (kill switch tripped) -- "
                      "human decision required; runner stopping")
                return 3
            if args.once:
                break
            await asyncio.sleep(int(args.poll_seconds))
    except KeyboardInterrupt:
        print("\nrunner stopped (Ctrl+C)")
        return 0
    print(f"\nrunner done ({cycles} cycles)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E4 live window runner")
    parser.add_argument("--once", action="store_true",
                        help="run exactly one cycle and exit")
    parser.add_argument("--poll-seconds", default=60)
    parser.add_argument("--max-cycles", type=int, default=0,
                        help="0 = run until stopped")
    parser.add_argument("--risk-status", action="store_true",
                        help="print mandate risk baselines + active trips "
                             "(no MT5 connection)")
    parser.add_argument("--clear-trip", choices=list(RISK_TRIP_NAMES),
                        help="HUMAN action: clear a mandate halt and rebase "
                             "its baseline (no MT5 connection)")
    parser.add_argument("--reason", default="",
                        help="required with --clear-trip; logged as the "
                             "human decision record")
    args = parser.parse_args()
    if args.once:
        args.max_cycles = 1
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
