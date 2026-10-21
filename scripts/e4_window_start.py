"""E4 window control: sign-off, version pin, eligibility, start/stop.

Operator tool for the 30-day live window. Every action is one explicit
flag; the default (no flags) prints status. ``--check`` and ``--start``
run the fresh four-check eligibility gate (version pin, probation
approval, order drill evidence, active SCOPE_WINDOW sign-off) and refuse
-- fail closed -- when anything is missing.

ASCII output only (Windows console). Usage:

  python scripts/e4_window_start.py
  python scripts/e4_window_start.py --record-signoff --approver murtaza \
      --declaration "I authorize the 30-day window for EUR/USD on 3f425957"
  python scripts/e4_window_start.py --approve-version --approver murtaza
  python scripts/e4_window_start.py --check
  python scripts/e4_window_start.py --start
  python scripts/e4_window_start.py --stop --reason "operator stop"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.execution.live.signoff import (  # noqa: E402
    SCOPE_WINDOW,
    LiveSignoff,
    SignoffStore,
)
from forex_ai_agent.execution.live.window import (  # noqa: E402
    DEFAULT_DRILL_DIR,
    DEFAULT_SIGNOFF_PATH,
    DEFAULT_VERSIONS_DIR,
    DEFAULT_WINDOW_STATE_DIR,
    E4_WINDOW_DAYS,
    check_eligibility,
    start_window,
    stop_window,
    window_status,
)
from forex_ai_agent.strategy.versioning import (  # noqa: E402
    StrategyVersionManager,
)

VERSION_ID = "3f425957"
PAIR = "EUR/USD"


def _print_status() -> int:
    status = window_status()
    state = status["state"]
    print("\n=== E4 WINDOW STATUS ===\n")
    if state is None:
        print("window: NOT STARTED (no state file)")
    else:
        print(f"window:     {state['window_id']}  status={state['status']}")
        print(f"pair:       {state['pair']}  version={state['version_id']}")
        print(f"started:    {state['started_at']}")
        print(f"ends:       {state['ends_at']}")
        print(f"remaining:  {status['remaining_days']} of {status['window_days']} days")
        if state["stop_reason"]:
            print(f"stopped:    {state['stopped_at']} reason={state['stop_reason']}")
    signoff = status["signoff"]
    if signoff is None:
        print("sign-off:   none recorded")
    else:
        active = "ACTIVE" if status["signoff_active"] else "EXPIRED/INACTIVE"
        print(f"sign-off:   {signoff['signoff_id']} [{active}] "
              f"by {signoff['approver']} until {signoff['expires_at']}")
    print(f"runner:     {status['runner']}")
    return 0


def _print_eligibility(eligibility) -> int:
    print()
    met = 0
    for check in eligibility.checks:
        tag = "[OK]  " if check.ok else "[FAIL]"
        if check.ok:
            met += 1
        print(f"  {tag} {check.name}: {check.detail}")
    total = len(eligibility.checks)
    print(f"\nELIGIBILITY: {met}/{total} checks met -> "
          f"{'ELIGIBLE' if eligibility.all_met else 'NOT ELIGIBLE'}")
    return 0 if eligibility.all_met else 2


async def _check() -> int:
    eligibility = await check_eligibility(version_id=VERSION_ID, pair=PAIR)
    return _print_eligibility(eligibility)


async def _start() -> int:
    ok, payload = await start_window(version_id=VERSION_ID, pair=PAIR)
    if ok:
        print(f"\nWINDOW STARTED: {payload.window_id}")
        print(f"  pair={payload.pair} version={payload.version_id}")
        print(f"  started={payload.started_at.isoformat()}")
        print(f"  ends={payload.ends_at.isoformat()} ({E4_WINDOW_DAYS} days)")
        print(f"  pinned_hash={payload.pinned_hash[:16]}...")
        print(f"  signoff={payload.signoff_id}")
        print("\nNext: scripts/e4_window_run.py (live loop, script-owned).")
        return 0
    print("\nWINDOW REFUSED -- fix the failing checks first:")
    return _print_eligibility(payload)


def _record_signoff(args: argparse.Namespace) -> int:
    if not args.declaration or not args.declaration.strip():
        print("ERROR: --record-signoff requires --declaration (the human's own words)")
        return 2
    now = datetime.utcnow()
    signoff = LiveSignoff(
        approver=args.approver,
        scope=SCOPE_WINDOW,
        pair=PAIR,
        strategy_id=VERSION_ID,
        declaration=args.declaration.strip(),
        granted_at=now,
        expires_at=now + timedelta(days=int(args.days)),
        config={
            "pair": PAIR,
            "version_id": VERSION_ID,
            "window_days": E4_WINDOW_DAYS,
            "strategy": "MeanRevBB Short EUR/USD",
            "recorded_from": "operator instruction 2026-10-17",
        },
    )
    SignoffStore(DEFAULT_SIGNOFF_PATH).record(signoff)
    print("SIGN-OFF RECORDED")
    print(f"  id:       {signoff.signoff_id}")
    print(f"  scope:    {signoff.scope}  pair={signoff.pair}  version={signoff.strategy_id}")
    print(f"  approver: {signoff.approver}")
    print(f"  granted:  {signoff.granted_at.isoformat()}")
    print(f"  expires:  {signoff.expires_at.isoformat()} ({args.days} days)")
    print(f"  log:      {DEFAULT_SIGNOFF_PATH}")
    return 0


def _approve_version(args: argparse.Namespace) -> int:
    return asyncio.run(_approve_version_async(args))


async def _approve_version_async(args: argparse.Namespace) -> int:
    manager = StrategyVersionManager(DEFAULT_VERSIONS_DIR)
    version = manager.approve(VERSION_ID, approver=args.approver)
    if version is None:
        print(f"ERROR: version {VERSION_ID} not found in {DEFAULT_VERSIONS_DIR}")
        return 2
    pinned = manager.verify_pin(VERSION_ID)
    print("VERSION APPROVED + HASH PINNED")
    print(f"  version:   {version.id} ({version.name})")
    print(f"  status:    {version.status}")
    print(f"  approved:  {version.approval.get('approved_by')} "
          f"at {version.approval.get('approved_at')}")
    print(f"  spec_hash: {version.approval.get('spec_hash')}")
    print(f"  pin check: {'OK' if pinned else 'FAILED'}")
    return 0 if pinned else 1


def _stop(args: argparse.Namespace) -> int:
    state = stop_window(reason=args.reason or "operator stop")
    if state is None:
        print("no window state to stop")
        return 0
    print(f"WINDOW {state.status.upper()}: {state.window_id} "
          f"({state.stop_reason})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="E4 live window control")
    parser.add_argument("--record-signoff", action="store_true")
    parser.add_argument("--approve-version", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--reason", default="")
    parser.add_argument("--approver", default="murtaza")
    parser.add_argument("--declaration", default="")
    parser.add_argument("--days", default=90,
                        help="sign-off validity in days (default 90)")
    args = parser.parse_args()

    if args.stop:
        return _stop(args)
    if args.record_signoff:
        return _record_signoff(args)
    if args.approve_version:
        return _approve_version(args)
    if args.check:
        return asyncio.run(_check())
    if args.start:
        return asyncio.run(_start())
    return _print_status()


if __name__ == "__main__":
    sys.exit(main())
