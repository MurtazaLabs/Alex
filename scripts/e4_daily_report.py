"""Stage E4 daily report: read-only P&L accounting over the MT5 statement.

What this script does (and does NOT do):

1. Builds the adapter ONLY through ``MT5Adapter.for_shadow()`` — permission
   level 0, shadow mode ON. Every write is refused before the transport by
   construction; the audit trail is asserted to contain zero WRITE calls.
2. Reads the account and the deal history for the requested day (default:
   today, UTC) plus the week-to-date window, filtered by
   ``ALEX_LIVE_MAGIC`` (314160 — distinct from the E2 shadow magic, so the
   daily E2 read-only check is unaffected).
3. Feeds the deals + equity into the deterministic ``build_daily_report``
   and prints the markdown, the JSON dict, and one ``RESULT:`` line for the
   scheduler to log.

This is the E4 verification-matrix script (read-only during the window):
it never transmits, never imports the transmitter, and never writes to the
DB. During the pre-window phase it shows an honest empty day.

Run:  .venv311/Scripts/python.exe scripts/e4_daily_report.py
      .venv311/Scripts/python.exe scripts/e4_daily_report.py 2026-10-16
"""

import json
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

from forex_ai_agent.brokers import MT5Adapter
from forex_ai_agent.execution.live import (
    ALEX_LIVE_MAGIC,
    TradeRecord,
    build_daily_report,
    format_daily_report_markdown,
)

SYMBOL = "EUR/USD"


def _parse_day(argv) -> datetime:
    if len(argv) > 1:
        return datetime.strptime(argv[1], "%Y-%m-%d")
    now = datetime.utcnow()
    return datetime(now.year, now.month, now.day)


def _deal_to_trade(deal) -> TradeRecord:
    """One deal row -> one accounting record (deal-based accounting)."""
    deal_type = int(deal.get("type", 0) or 0)
    return TradeRecord(
        ticket=int(deal.get("ticket", 0) or 0),
        symbol=str(deal.get("symbol", "")),
        side="BUY" if deal_type == 0 else "SELL",
        volume=float(deal.get("volume", 0.0) or 0.0),
        profit=float(deal.get("profit", 0.0) or 0.0),
        commission=float(deal.get("commission", 0.0) or 0.0),
        swap=float(deal.get("swap", 0.0) or 0.0),
        open_price=float(deal.get("price", 0.0) or 0.0),
    )


def main() -> int:
    day_start = _parse_day(sys.argv)
    day_end = day_start + timedelta(days=1)

    # Week-to-date window: Monday 00:00 UTC of the same week.
    week_start = day_start - timedelta(days=day_start.weekday())

    print("=== E4 DAILY REPORT (read-only) ===")
    print(f"day window:  {day_start.isoformat()} -> {day_end.isoformat()} (UTC)")
    print(f"week start:  {week_start.isoformat()} (UTC)")

    # ── 1. Read-only adapter ─────────────────────────────────────────────
    from forex_ai_agent.config import get_settings

    settings = get_settings()
    ctor_kwargs = {}
    if settings.mt5_login and settings.mt5_password and settings.mt5_server:
        ctor_kwargs.update(
            login=int(settings.mt5_login),
            password=settings.mt5_password,
            server=settings.mt5_server,
        )
    if settings.mt5_path:
        ctor_kwargs["path"] = settings.mt5_path

    adapter = MT5Adapter.for_shadow(**ctor_kwargs)
    try:
        adapter.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: cannot connect to the MT5 terminal: {exc}")
        print("RESULT: FAIL - no terminal")
        return 1

    try:
        account = adapter.account()
        print(f"\n[1] account #{account.login} {account.server} "
              f"{account.currency}: balance {account.balance:.2f} "
              f"equity {account.equity:.2f}")

        # ── 2. Deal history ──────────────────────────────────────────────
        day_deals = adapter.history_deals(day_start, day_end, magic=ALEX_LIVE_MAGIC)
        other_deals = [
            d for d in adapter.history_deals(day_start, day_end)
            if int(d.get("magic", 0) or 0) != ALEX_LIVE_MAGIC
        ]
        week_deals = adapter.history_deals(week_start, day_end, magic=ALEX_LIVE_MAGIC)
        print(f"[2] deals today: {len(day_deals)} alex-magic, "
              f"{len(other_deals)} other; week-to-date: {len(week_deals)} alex-magic")
        if other_deals:
            print("    note: non-Alex deals counted for statement hygiene only, "
                  "excluded from the report")

        trades = [_deal_to_trade(d) for d in day_deals]
        day_profit = sum(t.net_profit for t in trades)
        week_profit = sum(_deal_to_trade(d).net_profit for d in week_deals)

        # Starting equity = equity now minus everything earned since the
        # window opened (balance move + floating). Exact for a flat book.
        starting_equity = account.equity - day_profit
        weekly_start_equity = account.equity - week_profit

        # ── 3. Report ────────────────────────────────────────────────────
        report = build_daily_report(
            day=day_start.date(),
            trades=trades,
            starting_equity=starting_equity,
            ending_equity=account.equity,
            equity_points=[starting_equity, account.equity],
            weekly_pnl=week_profit,
            weekly_start_equity=weekly_start_equity,
        )
        print("\n[3] report:\n")
        print(format_daily_report_markdown(report))
        print("\n[3b] json:")
        print(json.dumps(report.to_dict(), indent=2))

        # ── 4. Read-only proof ───────────────────────────────────────────
        writes = adapter.write_calls()
        sent = adapter.sent_calls()
        levels = {c.mode for c in adapter.audit_trail()}
        readonly_ok = (
            not writes
            and not sent
            and levels <= {"READ", "CONTROL"}
        )
        print(f"\n[4] read-only proof: write_calls={len(writes)} "
              f"sent={len(sent)} modes={sorted(levels)}")
        if not readonly_ok:
            print("RESULT: FAIL - write call recorded on a read-only run")
            return 1

        print(f"\nRESULT: {report.verdict} - {report.trade_count} trades, "
              f"net {report.net_pnl:+.2f}, dd {report.max_drawdown:.2f}, "
              f"violations {len(report.violations)}")
        return 0
    finally:
        adapter.disconnect()


if __name__ == "__main__":
    sys.exit(main())
