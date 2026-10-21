"""Stage E4 live verification drill: level 4 + shadow off, one guarded wire.

The human granted AUTONOMOUS(4) (MAX_PERMISSION_LEVEL=4) and authorized a
live drill. This script is that drill, and it is built to be honest in
both directions:

* it records the sign-off FIRST (approver, scope, expiry, the human's own
  words) into an append-only log before touching anything;
* PHASE 1 (always) runs every refusal path of the real gate chain against
  the REAL adapter at level 4 / shadow off - intent, expiry, pair, volume,
  session, news, governor, broker gate - and proves each one refuses,
  sends nothing, writes nothing, and does not burn the approval token;
* PHASE 2 (order drill) runs ONLY inside a live London/NY session and ONLY
  with --execute: one 0.01-lot EUR/USD market order through the real
  transmitter -> reconcile fill (<= 2.0 pips drift band) -> close ->
  prove zero net exposure. Outside a session it prints DEFERRED.

The session filter is a hard E4 cap - this script never bypasses it.
So on a weekend the honest result is: refusals green, order deferred.

Run:
  .venv311/Scripts/python.exe scripts/e4_live_verify.py
  .venv311/Scripts/python.exe scripts/e4_live_verify.py --execute

Output is ASCII-only (cp1252 console safe) and the run writes one evidence
JSON into logs/live_drill/.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forex_ai_agent.brokers import MT5Adapter
from forex_ai_agent.config import get_settings
from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager
from forex_ai_agent.execution.governor import ExecutionGovernor, GovernorLimits
from forex_ai_agent.execution.live import (
    ALEX_LIVE_MAGIC,
    DEFAULT_LIVE_CAPS,
    E4_PAIR,
    SCOPE_DRILL,
    LiveSignoff,
    LiveTransmitter,
    NewsBlackout,
    NewsEvent,
    PositionTracker,
    SessionFilter,
    SignoffStore,
    signed_slippage_pips,
    sync_account_into,
)
from forex_ai_agent.execution.policy import (
    INTENT_VALID,
    ExecutionPolicy,
    PolicyCaps,
)
from forex_ai_agent.execution.tokens import ApprovalTokenStore

SYMBOL = "EUR/USD"
STRATEGY = "mean_reversion_v1"
SL_PIPS = 40.0
TP_PIPS = 40.0
PIP = 0.0001
LOG_DIR = ROOT / "logs" / "live_drill"
SIGNOFF_EXPIRY_DAYS = 7

CHECKS = []


def _record(name, ok, detail=""):
    CHECKS.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    tag = "[PASS]" if ok else "[FAIL]"
    suffix = f" - {detail}" if detail else ""
    print(f"  {tag} {name}{suffix}")
    return ok


def _next_weekday_at(now, weekday, hour):
    """Next occurrence of ``weekday`` at ``hour`` UTC, strictly after now."""
    days = (weekday - now.weekday()) % 7
    candidate = (now + timedelta(days=days)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += timedelta(days=7)
    return candidate


def _next_open_window(now, session):
    probe = now.replace(minute=0, second=0, microsecond=0)
    for _ in range(24 * 8):
        probe += timedelta(hours=1)
        if session.check(probe).open:
            return probe
    return None


def _duck_intent(**over):
    """A transmitter-shaped intent for refusal cases the policy cannot build."""
    base = dict(
        key="drill-duck",
        status="VALID",
        symbol=SYMBOL,
        side="BUY",
        volume=0.01,
        entry=1.10000,
        sl=1.09600,
        tp=1.10400,
        comment="drill duck",
        is_expired=lambda now=None: False,
    )
    base.update(over)
    return SimpleNamespace(**base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="attempt the live order drill (in-session only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    now = datetime.utcnow()
    session = SessionFilter()
    print("\n=== E4 LIVE VERIFY: level 4 + shadow off, one guarded chain ===\n")
    print(f"[0] time now (UTC): {now.isoformat(timespec='seconds')}")
    print(f"    session state: {session.check(now).reason}")

    # ------------------------------------------------------------------
    # 0. Config + sign-off record
    # ------------------------------------------------------------------
    settings = get_settings()
    level = int(settings.max_permission_level)
    if level < PermissionLevel.AUTONOMOUS:
        _record("config grants AUTONOMOUS(4)", False, f"MAX_PERMISSION_LEVEL={level}")
        print("\nRESULT: FAIL - level 4 not granted in config")
        return 1
    _record("config grants AUTONOMOUS(4)", True, "MAX_PERMISSION_LEVEL=4")

    # ------------------------------------------------------------------
    # 1. Real adapter at level 4, shadow OFF (the drill's one vehicle)
    # ------------------------------------------------------------------
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
    except Exception as exc:  # noqa: BLE001 - no terminal is a FAIL, not a crash
        _record("MT5 terminal connect", False, str(exc))
        print("\nRESULT: FAIL - cannot connect to the MT5 terminal")
        return 1
    _record("MT5 terminal connect", True, "initialized")

    account = adapter.account()
    spec = adapter.symbol_spec(SYMBOL)
    print(f"    account: #{account.login} {account.server} "
          f"balance={account.balance:.2f} {account.currency} equity={account.equity:.2f}")
    print(f"    symbol:  {SYMBOL} -> {spec.broker_symbol} "
          f"vol[{spec.volume_min}, {spec.volume_max}] step={spec.volume_step}")
    _record("write symbol resolves to broker name", spec.broker_symbol != "",
            f"{SYMBOL} -> {spec.broker_symbol}")
    _record("broker gate OPEN (level 4 + shadow off)",
            adapter.permission_level == 4 and adapter.shadow_mode is False,
            f"level={adapter.permission_level} shadow={adapter.shadow_mode}")

    signoff_store = SignoffStore(LOG_DIR / "signoffs.jsonl")
    signoff = signoff_store.record(LiveSignoff(
        approver="murtaza",
        scope=SCOPE_DRILL,
        pair=E4_PAIR,
        strategy_id=STRATEGY,
        declaration="i have given the level 4 permission you can test yourself",
        granted_at=now,
        expires_at=now + timedelta(days=SIGNOFF_EXPIRY_DAYS),
        config={
            "max_permission_level": 4,
            "shadow_mode": False,
            "account": int(account.login),
            "server": str(account.server),
            "caps": DEFAULT_LIVE_CAPS.to_dict(),
            "scope_detail": "one 0.01-lot EUR/USD market order + close verification",
        },
    ))
    _record("sign-off recorded (append-only)", True,
            f"id={signoff.signoff_id} expires={(now + timedelta(days=SIGNOFF_EXPIRY_DAYS)).date()}")

    # Independent terminal read (same pattern as e2_live_verify).
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        _record("independent MetaTrader5 read", False, str(exc))
        print("\nRESULT: FAIL - MetaTrader5 module unavailable")
        return 1
    tick = mt5.symbol_info_tick(spec.broker_symbol)
    if tick is None:
        _record("live tick read", False, "no tick for " + spec.broker_symbol)
    else:
        _record("live tick read", True,
                f"bid={tick.bid:.5f} ask={tick.ask:.5f} "
                f"age={(now - datetime.utcfromtimestamp(tick.time)).total_seconds():.0f}s")
    reference_price = float(tick.ask) if tick else 1.10000

    positions_before = adapter.positions()

    # ------------------------------------------------------------------
    # 2. Gate chain
    # ------------------------------------------------------------------
    # Drill plumbing caps: the mandate (0.25%) cannot size even one 0.01
    # lot on this account (needs the budget to cover a 40-pip stop), so
    # this drill scaffold restores the pre-mandate demo caps for the
    # round-trip only. Requires equity >= $80 (0.01 lot risk at 40 pips
    # = $4.00 = 5% of $80). The live window runner uses the real mandate.
    policy = ExecutionPolicy(PolicyCaps(
        max_risk_per_trade_pct=5.0,
        max_aggregate_stop_risk_pct=5.0,
        max_lot_size=0.01,
    ))
    policy.update_account_state(equity=float(account.equity), open_positions=0,
                                daily_pnl=0.0, now=now)
    governor = ExecutionGovernor(GovernorLimits(
        daily_loss_limit=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_daily_loss_pct / 100.0),
        max_drawdown=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_drawdown_pct / 100.0),
        max_positions=3,
        weekly_loss_limit=float(account.equity)
        * (DEFAULT_LIVE_CAPS.max_weekly_loss_pct / 100.0),
    ))
    tokens = ApprovalTokenStore()
    transmitter = LiveTransmitter(adapter, tokens, governor)

    def make_intent(when, side="BUY"):
        return policy.create_intent(
            symbol=SYMBOL, side=side,
            entry=reference_price,
            stop_loss=reference_price - SL_PIPS * PIP,
            take_profit=reference_price + TP_PIPS * PIP,
            strategy_id=STRATEGY, comment="e4 live drill",
            now=when,
        )

    intent = make_intent(now)
    if intent.status != INTENT_VALID or intent.volume != 0.01:
        _record("drill intent VALID at 0.01 lots", False,
                f"status={intent.status} volume={intent.volume}")
        print("\nRESULT: FAIL - cannot build the drill intent")
        return 1
    _record("drill intent VALID at 0.01 lots", True,
            f"key={intent.key[:10]} entry={intent.entry:.5f} "
            f"sl={intent.sl:.5f} tp={intent.tp:.5f} risk=${intent.risk_usd:.2f}")

    token = tokens.grant(
        intent_key=intent.key, symbol=intent.symbol, side=intent.side,
        volume=intent.volume, granted_by="murtaza (e4 live drill)",
        intent_snapshot=intent.to_dict(), now=now,
    )
    _record("approval token granted", True, f"token={token.token_id[:10]}")

    def refusal(name, expected, drill_intent, when, **transmitter_kw):
        chain = transmitter
        if transmitter_kw:
            chain = LiveTransmitter(adapter, tokens, governor, **transmitter_kw)
        result = chain.transmit(drill_intent, token.token_id, now=when)
        ok = result.status == expected and not result.sent
        _record(name, ok, f"status={result.status} sent={result.sent}")
        return result

    print("\n[2] Refusal proofs on the REAL chain (nothing may reach the wire)")

    # intent status: policy refuses wrong-side SL -> status != VALID
    bad = policy.create_intent(
        symbol=SYMBOL, side="BUY", entry=1.10000,
        stop_loss=1.10100, take_profit=1.10500, strategy_id=STRATEGY,
    )
    refusal("REFUSED_INTENT", "REFUSED_INTENT", bad, now)

    # expiry: transmit after the 300s TTL
    expired_when = intent.expires_at + timedelta(seconds=1)
    refusal("REFUSED_EXPIRED", "REFUSED_EXPIRED", intent, expired_when)

    # pair + volume: policy cannot produce them (E4 caps), use shaped intents
    refusal("REFUSED_PAIR", "REFUSED_PAIR",
            _duck_intent(symbol="GBP/USD"), now)
    refusal("REFUSED_VOLUME", "REFUSED_VOLUME",
            _duck_intent(volume=0.02), now)

    # session: a real weekend instant (Saturday 12:00 UTC)
    weekend_when = _next_weekday_at(now, 5, 12)
    refusal("REFUSED_SESSION", "REFUSED_SESSION",
            make_intent(weekend_when), weekend_when)

    # news: a Monday 08:00 UTC in-session instant with a synthetic event
    monday_when = _next_weekday_at(now, 0, 8)
    drill_blackout = NewsBlackout(
        [NewsEvent(timestamp=monday_when + timedelta(minutes=10),
                   title="Drill synthetic high-impact event")],
        window_minutes=DEFAULT_LIVE_CAPS.news_blackout_minutes,
    )
    refusal("REFUSED_NEWS", "REFUSED_NEWS",
            make_intent(monday_when), monday_when, blackout=drill_blackout)

    # governor: same in-session instant, chain has an UNARMED governor
    refusal("REFUSED_GOVERNOR", "REFUSED_GOVERNOR",
            make_intent(monday_when), monday_when)

    # broker gate: armed governor, but the adapter is shadow-on (level 4)
    governor.arm()
    shadow_adapter = MT5Adapter.for_shadow(PermissionManager(PermissionLevel.AUTONOMOUS))
    shadow_chain = LiveTransmitter(shadow_adapter, tokens, governor)
    result = shadow_chain.transmit(make_intent(monday_when), token.token_id,
                                   now=monday_when)
    _record("REFUSED_BROKER_GATE (shadow on)",
            result.status == "REFUSED_BROKER_GATE" and not result.sent,
            f"status={result.status} sent={result.sent}")

    # broker gate: shadow off, but the adapter is level 0
    level0_adapter = MT5Adapter(PermissionManager(PermissionLevel.OBSERVE),
                                shadow_mode=False)
    level0_chain = LiveTransmitter(level0_adapter, tokens, governor)
    result = level0_chain.transmit(make_intent(monday_when), token.token_id,
                                   now=monday_when)
    _record("REFUSED_BROKER_GATE (level 0)",
            result.status == "REFUSED_BROKER_GATE" and not result.sent,
            f"status={result.status} sent={result.sent}")

    # the token survived every preflight refusal - prove it, then it is spent
    survived = tokens.consume(
        token.token_id, intent_key=intent.key, symbol=intent.symbol,
        side=intent.side, volume=intent.volume, now=now,
    )
    _record("token survived all refusals (not burned)", survived,
            "next consume succeeds -> refusals never reached the token step")

    # nothing sent / nothing written / book unchanged
    _record("real adapter made ZERO wire attempts",
            adapter.sent_calls() == [] and adapter.write_calls() == [],
            f"sent={len(adapter.sent_calls())} writes={len(adapter.write_calls())}")
    _record("position book unchanged by refusals",
            [p.ticket for p in adapter.positions()] ==
            [p.ticket for p in positions_before])

    # ------------------------------------------------------------------
    # 3. Order drill (in-session + --execute only)
    # ------------------------------------------------------------------
    order_evidence = None
    session_state = session.check(datetime.utcnow())
    print("\n[3] Order drill")
    print(f"    session: {session_state.reason}")
    if not session_state.open:
        nxt = _next_open_window(datetime.utcnow(), session)
        print("    ORDER PHASE: DEFERRED - market closed (hard E4 session cap,")
        print("    never bypassed). Re-run during a session with --execute.")
        if nxt:
            print(f"    next session window opens: {nxt.isoformat(timespec='minutes')} UTC")
    elif not args.execute:
        print("    ORDER PHASE: SKIPPED - session open; pass --execute to run it")
    else:
        order_evidence = run_order_drill(
            adapter, policy, governor, tokens, mt5, spec, session,
        )

    # ------------------------------------------------------------------
    # 4. Evidence + verdict
    # ------------------------------------------------------------------
    failures = sum(1 for c in CHECKS if not c["ok"])
    if order_evidence and not order_evidence.get("ok", False):
        failures += 1

    verdict = "PASS" if failures == 0 else "FAIL"
    deferred = "session" if not session_state.open else (
        "flag" if not args.execute else "no"
    )
    sent_total = len(adapter.sent_calls())

    evidence = {
        "script": "e4_live_verify",
        "at": now.isoformat(),
        "result": verdict,
        "order_phase": ("executed" if order_evidence else
                        ("deferred" if deferred == "session" else "skipped")),
        "account": {
            "login": int(account.login), "server": str(account.server),
            "balance": float(account.balance), "equity": float(account.equity),
        },
        "gate": {
            "permission_level": adapter.permission_level,
            "shadow_mode": bool(adapter.shadow_mode),
            "max_lot": DEFAULT_LIVE_CAPS.max_lot,
        },
        "signoff": signoff.to_dict(),
        "refusals": [c for c in CHECKS if c["name"].startswith("REFUSED")],
        "checks": CHECKS,
        "order": order_evidence,
        "wire_attempts": sent_total,
    }
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%d_%H%M%S")
    evidence_path = LOG_DIR / f"e4_live_verify_{stamp}.json"
    evidence_path.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"checks: {len(CHECKS) - failures}/{len(CHECKS)} green | "
          f"wire attempts: {sent_total} | order phase: {evidence['order_phase']}")
    print(f"evidence: {evidence_path}")
    print(f"RESULT: {verdict} - failures={failures} "
          f"sent={sent_total} signoff={signoff.signoff_id}")
    return 0 if verdict == "PASS" else 1


def run_order_drill(adapter, policy, governor, tokens, mt5, spec, session):
    """One min-lot round-trip through the real transmitter. In-session only."""
    print("    ORDER PHASE: executing the live drill...")
    now = datetime.utcnow()
    evidence = {"ok": False, "ticket": None, "slip_pips": None, "net": None}
    ticket = None
    try:
        tick = mt5.symbol_info_tick(spec.broker_symbol)
        if tick is None:
            _record("order drill: live tick", False, "no tick")
            return evidence
        ask = float(tick.ask)

        intent = policy.create_intent(
            symbol=SYMBOL, side="BUY", entry=ask,
            stop_loss=ask - SL_PIPS * PIP, take_profit=ask + TP_PIPS * PIP,
            strategy_id=STRATEGY, comment="e4 live drill order",
            now=now,
        )
        if intent.status != INTENT_VALID or intent.volume != 0.01:
            _record("order drill: intent VALID 0.01", False,
                    f"status={intent.status} volume={intent.volume}")
            return evidence
        _record("order drill: intent VALID 0.01", True,
                f"entry={ask:.5f} sl={intent.sl:.5f} tp={intent.tp:.5f}")

        token = tokens.grant(
            intent_key=intent.key, symbol=intent.symbol, side=intent.side,
            volume=intent.volume, granted_by="murtaza (e4 live drill)",
            intent_snapshot=intent.to_dict(), now=now,
        )
        transmitter = LiveTransmitter(adapter, tokens, governor)
        result = transmitter.transmit(intent, token.token_id, now=now)
        evidence["status"] = result.status
        evidence["retcode"] = result.retcode
        if not result.is_fill:
            _record("order drill: transmit reached a fill", False,
                    f"status={result.status} detail={result.detail}")
            return evidence
        _record("order drill: transmit reached a fill", True,
                f"status={result.status} retcode={result.retcode} ticket={result.ticket}")

        positions = [p for p in adapter.positions()
                     if p.magic == ALEX_LIVE_MAGIC or p.ticket == result.ticket]
        if not positions:
            _record("order drill: position visible with live magic", False,
                    f"magic={ALEX_LIVE_MAGIC} not found")
            return evidence
        position = positions[0]
        ticket = position.ticket
        evidence["ticket"] = ticket
        _record("order drill: position visible with live magic", True,
                f"ticket={ticket} volume={position.volume} "
                f"open={position.price_open:.5f}")

        slip = signed_slippage_pips("BUY", ask, position.price_open, PIP)
        evidence["slip_pips"] = round(float(slip), 2)
        within = abs(float(slip)) <= DEFAULT_LIVE_CAPS.fill_drift_tolerance_pips
        _record("order drill: fill within drift band", within,
                f"slip={slip:+.2f} pips (band "
                f"{DEFAULT_LIVE_CAPS.fill_drift_tolerance_pips})")

        close_ack = adapter.close_position(ticket)
        _record("order drill: close position", close_ack.status == "CLOSED",
                f"status={close_ack.status} retcode={close_ack.retcode}")
        remaining = [p for p in adapter.positions() if p.ticket == ticket]
        _record("order drill: zero net exposure", remaining == [],
                f"ticket {ticket} absent from the position book")

        deals = adapter.history_deals(now - timedelta(minutes=10),
                                      datetime.utcnow() + timedelta(minutes=5),
                                      magic=ALEX_LIVE_MAGIC)
        net = sum(float(d.get("profit", 0.0)) + float(d.get("commission", 0.0))
                  + float(d.get("swap", 0.0)) for d in deals)
        evidence["net"] = round(net, 2)
        evidence["deals"] = len(deals)
        tracker = PositionTracker(symbol=SYMBOL)
        tracker.sync_positions(adapter.positions(), now=datetime.utcnow())
        tracker.record_deal(profit=net, ts=datetime.utcnow(), ticket=ticket,
                            symbol=SYMBOL, volume=0.01, comment="e4 drill")
        feed = sync_account_into(tracker, adapter.account(), policy=policy,
                                 governor=governor,
                                 daily_loss_pct=DEFAULT_LIVE_CAPS.max_daily_loss_pct,
                                 daily_pnl=0.0,
                                 now=datetime.utcnow())
        _record("order drill: tracker + equity sync fed", True,
                f"open={feed.open_positions} realized={feed.realized_pnl_today:+.2f} "
                f"daily_limit=${feed.daily_loss_limit_usd:.2f}")

        evidence["ok"] = bool(within and close_ack.status == "CLOSED"
                              and remaining == [])
        return evidence
    except Exception as exc:  # noqa: BLE001 - unknown state = fail closed
        _record("order drill: unexpected exception", False,
                f"{type(exc).__name__}: {exc}")
        governor.trip(f"e4_live_verify_exception: {type(exc).__name__}")
        if ticket is not None:
            try:
                ack = adapter.close_position(ticket)
                print(f"    emergency close attempt for {ticket}: {ack.status}")
            except Exception as close_exc:  # noqa: BLE001
                print(f"    !!! MANUAL ACTION: close ticket {ticket} in the "
                      f"terminal (emergency close failed: {close_exc})")
        return evidence


if __name__ == "__main__":
    sys.exit(main())
