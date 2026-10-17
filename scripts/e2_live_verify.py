"""Stage E2 live verification: read-only MT5 + would-have shadow ledger.

1. Fetch REAL H1 bars for EUR/USD from the resilient feed chain
   (mock-degraded feed ⇒ SKIP, same guard as e1).
2. Check 1 — read-only snapshot: build the adapter ONLY through
   ``MT5Adapter.for_shadow()`` (permission level 0, shadow mode on), read
   account / symbol spec / positions / pending orders, and compare against
   an INDEPENDENT direct read of the MetaTrader5 module (the declared
   snapshot). The audit trail of the whole session must contain only
   READ/CONTROL calls and zero writes.
3. Check 2 — reconciliation drill (read-only): a declared write-ahead
   intent book is reconciled against the broker's real positions under the
   declared policy string. In shadow mode nothing was ever sent, so every
   intent must land in HUMAN_REVIEW, the journal must be byte-identical
   before/after (never re-sent), and a re-read of positions must match the
   check-1 snapshot.
4. Check 3 — shadow run over ≥10 trading days: E0's own simulator produces
   the deterministic intent stream on the real bars (governed, record_only),
   then every intent is replayed through ShadowExecutor against the LIVE
   broker symbol spec (suffix-resolved: EUR/USD → EURUSDm) and the real
   bar stream. Fills land on the next real bar; actual fill uses the
   broker's live spread snapshot, predicted fill uses E0's model spread;
   fill drift must stay inside the declared band; rejections inside the
   declared rate; spec/volume rejections are hard failures.
5. Check 4 — zero real orders, two independent proofs: (a) the adapter's
   audit log — a deliberate submit attempt at level 0 is refused before
   the transport, ``sent_calls()`` stays empty; (b) the broker's own
   statement — no deal carrying ``ALEX_SHADOW_MAGIC`` exists in history,
   and the position book is unchanged from check 1.
6. Write nothing to the DB — pure reads and pure computation.

Declared bands (printed before the run):
  * shadow duration      ≥ 10 trading days of real H1 bars
  * |mean fill drift|    ≤ 3.0 pips, |max fill drift| ≤ 5.0 pips
  * rejection rate       ≤ 10% of submitted intents (spec/volume refusals
                          are NOT allowed — those mean live-spec mismatch)
  * real orders sent     exactly 0 (audit log AND broker statement)

Run:  .venv311/Scripts/python.exe scripts/e2_live_verify.py
"""

import asyncio
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict

sys.path.insert(0, "src")

import pandas as pd

from forex_ai_agent.autonomy.validator import SpecSignalGenerator
from forex_ai_agent.backtest.engine import BacktestConfig
from forex_ai_agent.backtest.execution import DEFAULT_SPREADS
from forex_ai_agent.brokers import (
    ALEX_SHADOW_MAGIC,
    MT5Adapter,
    OrderRequest,
    ShadowExecutor,
    broker_spread_pips,
    engine_atr,
    reconcile,
)
from forex_ai_agent.brokers.shadow import (
    REJECT_SPEC_UNAVAILABLE,
    REJECT_TRADE_DISABLED,
    REJECT_VOLUME_MAX,
    REJECT_VOLUME_MIN,
    REJECT_VOLUME_STEP,
)
from forex_ai_agent.core.permission_manager import PermissionLevel
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.execution.governor import ExecutionGovernor, GovernorLimits
from forex_ai_agent.execution.journal import IntentJournal
from forex_ai_agent.execution.simulator import ExecutionParams, run_simulation
from forex_ai_agent.strategy.spec import parse_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

SYMBOL = "EUR/USD"

MIN_TRADING_DAYS = 10
DRIFT_MEAN_BAND_PIPS = 3.0
DRIFT_MAX_BAND_PIPS = 5.0
REJECTION_RATE_BAND = 0.10
SHADOW_MAX_POSITIONS = 100
MODEL_SPREAD_PIPS = DEFAULT_SPREADS[SYMBOL]  # E0's model spread (predicted)

# Refusals that mean the live broker spec disagrees with our book (never
# acceptable): symbol lookup, trade mode, volume bounds/step.
SPEC_REJECTIONS = (
    REJECT_SPEC_UNAVAILABLE,
    REJECT_TRADE_DISABLED,
    REJECT_VOLUME_MIN,
    REJECT_VOLUME_MAX,
    REJECT_VOLUME_STEP,
)


def _norm(ts_str: str) -> datetime:
    """Second-precision timestamp from any isoformat string."""
    return datetime.fromisoformat(str(ts_str)[:19])


def _same_symbol(alex_form: str, raw_form: str) -> bool:
    """EUR/USD (adapter) vs EURUSDm (terminal): same instrument check."""
    if alex_form == raw_form:
        return True
    base = MT5Adapter.broker_symbol(alex_form)
    return raw_form == base or raw_form.startswith(base)


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    # ── Feed guard + real bars ────────────────────────────────────────────
    chain = get_feed_chain()
    print("feed chain active source:", chain.active_source)
    if chain.active_source == "mock":
        print("SKIP: feed degraded to mock — no real data to verify against")
        return 2

    end = datetime.utcnow()
    start = end - timedelta(days=30)
    bars = await chain.fetch_candles(SYMBOL, "H1", start, end=end, limit=800)
    if not bars or len(bars) < 240:
        print(f"FAIL: only {0 if not bars else len(bars)} live bars returned")
        return 1
    df = pd.DataFrame(
        {
            "open": [b.open for b in bars],
            "high": [b.high for b in bars],
            "low": [b.low for b in bars],
            "close": [b.close for b in bars],
            "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
        },
        index=[b.timestamp for b in bars],
    )
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    trading_days = len({d.date() for d in df.index if d.weekday() < 5})
    print(
        f"fetched {len(df)} real H1 bars "
        f"({df.index[0]} → {df.index[-1]}), {trading_days} trading days"
    )
    if trading_days < MIN_TRADING_DAYS:
        print(f"FAIL: only {trading_days} trading days, need ≥{MIN_TRADING_DAYS}")
        return 1

    print("\ndeclared bands:")
    print(f"  shadow duration   ≥ {MIN_TRADING_DAYS} trading days")
    print(f"  |mean drift| ≤ {DRIFT_MEAN_BAND_PIPS} pips, "
          f"|max drift| ≤ {DRIFT_MAX_BAND_PIPS} pips")
    print(f"  rejection rate ≤ {REJECTION_RATE_BAND:.0%} "
          f"(spec/volume refusals forbidden)")
    print("  real orders sent  exactly 0")

    # ── Check 1 — read-only snapshot vs an independent terminal read ──────
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

    adapter = MT5Adapter.for_shadow(**ctor_kwargs)  # shadow forced ON
    try:
        adapter.connect()
    except Exception as exc:  # noqa: BLE001 — no terminal is a FAIL, not a crash
        print(f"FAIL: cannot connect to the MT5 terminal: {exc}")
        return 1

    account = adapter.account()
    spec = adapter.symbol_spec(SYMBOL)  # suffix-resolves EUR/USD → EURUSDm
    positions = adapter.positions()
    pending = adapter.pending_orders()

    # Independent ground truth: bypass the adapter entirely.
    import MetaTrader5 as mt5

    raw_account = mt5.account_info()
    raw_positions = mt5.positions_get() or ()
    if raw_account is None:
        print("FAIL: independent MetaTrader5 read returned no account")
        return 1
    check1 = (
        account.login == int(raw_account.login)
        and account.server == str(raw_account.server)
        and account.currency == str(raw_account.currency)
        and account.balance == float(raw_account.balance)
        and account.equity == float(raw_account.equity)
        and account.margin == float(raw_account.margin)
        and len(positions) == len(raw_positions)
        and all(
            p.ticket == int(r.ticket)
            and p.side == ("BUY" if int(r.type) == 0 else "SELL")
            and p.volume == float(r.volume)
            and _same_symbol(p.symbol, str(r.symbol))
            for p, r in zip(
                sorted(positions, key=lambda x: x.ticket),
                sorted(raw_positions, key=lambda x: x.ticket),
            )
        )
        and adapter.permission_level == PermissionLevel.OBSERVE
        and adapter.shadow_mode is True
        and adapter.write_calls() == []
        and adapter.sent_calls() == []
        and {c.mode for c in adapter.audit_trail()} <= {"READ", "CONTROL"}
    )
    live_spread_pips = broker_spread_pips(spec, pip_size=pip_size_for_symbol(SYMBOL))
    print(f"\nCheck 1 — read-only snapshot vs independent terminal read: {check1}")
    print(f"  account #{account.login} {account.server}: balance {account.balance} "
          f"equity {account.equity} {account.currency}; "
          f"{len(positions)} position(s), {len(pending)} pending order(s)")
    print(f"  symbol {SYMBOL} → {spec.broker_symbol}: trade_allowed="
          f"{spec.trade_allowed}, digits={spec.digits}, "
          f"volume [{spec.volume_min}, {spec.volume_max}] step {spec.volume_step}, "
          f"live spread {live_spread_pips:.2f} pips")
    print(f"  permission={adapter.permission_level} "
          f"shadow={adapter.shadow_mode} writes=0 "
          f"audit={sorted({c.mode for c in adapter.audit_trail()})}")
    if not check1:
        print("RESULT: FAIL")
        return 1
    declared_tickets = [p.ticket for p in positions]

    # ── Check 2 — reconciliation drill (read-only, declared policy) ───────
    journal = IntentJournal()
    declared = [
        ("e2-declared-1", "BUY", 0.10),
        ("e2-declared-2", "SELL", 0.20),
        ("e2-declared-3", "BUY", 0.01),
    ]
    for key, side, volume in declared:
        journal.record(key, symbol=SYMBOL, side=side, volume=volume,
                       sl=None, tp=None, created_ts=datetime.utcnow())
    before_bytes = journal.save_snapshot()
    report = reconcile(journal.pending(), positions, ts=datetime.utcnow())
    after_bytes = journal.save_snapshot()
    check2 = (
        report.policy == "adopt_matching_review_unmatched_never_resend"
        and sorted(report.review) == sorted(k for k, _, _ in declared)
        and report.adopted == []
        and [p["ticket"] for p in report.broker_only] == sorted(declared_tickets)
        and before_bytes == after_bytes  # never re-sent, never mutated
        and adapter.write_calls() == []
        and [p.ticket for p in adapter.positions()] == declared_tickets
    )
    print(f"\nCheck 2 — reconciliation drill vs the real account (read-only): "
          f"{check2}")
    print(f"  policy={report.policy}: adopted={len(report.adopted)} "
          f"review={len(report.review)} broker_only={len(report.broker_only)} "
          f"divergence={report.divergence} "
          f"(shadow intents have no broker counterpart BY DESIGN)")
    print(f"  journal byte-identical after reconcile: "
          f"{before_bytes == after_bytes}; position re-read matches "
          f"check-1 snapshot: {[p.ticket for p in adapter.positions()] == declared_tickets}")
    if not check2:
        print("RESULT: FAIL")
        return 1

    # ── Check 3 — shadow run over the real bars ───────────────────────────
    rules = {
        "name": "E2 Shadow Probe",
        "direction": "both",
        "timeframe": "H1",
        "entry_conditions": [
            {"left": "sma", "verb": "above", "right": "sma_slow"},
            {"left": "price", "verb": "above", "right": "sma"},
        ],
    }
    spec_rules, errs = parse_spec(rules)
    if spec_rules is None:
        print("FAIL: probe spec invalid:", errs)
        return 1

    def fresh_gen():
        return SpecSignalGenerator(spec_rules, pip_size=pip_size_for_symbol(SYMBOL))

    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=10_000.0,
        symbols=[SYMBOL],
        timeframe="H1",
        warmup_bars=12,
    )
    limits = GovernorLimits(daily_loss_limit=1e9, max_drawdown=1e9,
                            max_positions=1000)

    # Phase 1: the deterministic intent stream, produced by E0's own
    # simulator on the real bars (write-ahead journal, nothing sent).
    signal_gov = ExecutionGovernor(limits)
    signal_gov.arm()
    run_simulation({SYMBOL: df}, fresh_gen(), config,
                   params=ExecutionParams(record_intents=True),
                   governor=signal_gov)
    intents_by_ts = defaultdict(list)
    for rec in signal_gov.journal.all():
        intents_by_ts[_norm(rec["created_ts"])].append(rec)
    total_intents = len(signal_gov.journal)
    if total_intents == 0:
        print("FAIL: signal run produced no journaled intents — vacuous shadow")
        return 1

    # Phase 2: replay every intent through the REAL chain — governor
    # write-ahead, live broker spec validation, shadow ledger fills.
    shadow_gov = ExecutionGovernor(
        GovernorLimits(daily_loss_limit=1e9, max_drawdown=1e9,
                       max_positions=SHADOW_MAX_POSITIONS)
    )
    shadow_gov.arm()
    executor = ShadowExecutor(adapter, shadow_gov)
    recorded = rejected = 0
    rejection_reasons: Dict[str, int] = {}
    for i, (ts, row) in enumerate(df.iterrows()):
        window = df.iloc[max(0, i - 200): i + 1]
        atr = engine_atr(window["high"].values, window["low"].values,
                         window["close"].values, 14)
        executor.on_bar(
            SYMBOL,
            ts.to_pydatetime(),
            open_=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            spread_pips=live_spread_pips,  # live broker spread = actual
            atr_pips=atr,                  # engine-parity ATR (same both sides)
            price_digits=spec.digits,
            bar_index=i + 1,
        )
        for rec in intents_by_ts.get(_norm(ts.isoformat()), ()):
            outcome = executor.submit(
                rec["key"],
                symbol=rec["symbol"],
                side=rec["side"],
                volume=rec["volume"],
                ts=ts.to_pydatetime(),
                sl=rec["sl"],
                tp=rec["tp"],
                bar_index=i + 1,
                comment=rec["comment"],
            )
            if outcome["status"] == "RECORDED":
                recorded += 1
            else:
                rejected += 1
                reason = str(outcome["reason"])
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
    last_ts = df.index[-1].to_pydatetime()
    executor.close_window(last_ts, reason="WINDOW_END")
    stats = executor.ledger.stats()
    drift = stats["fill_drift"]
    rejection_rate = rejected / max(1, recorded + rejected)
    spec_rejected = sum(
        n for reason, n in rejection_reasons.items()
        if reason in SPEC_REJECTIONS or reason.startswith(REJECT_SPEC_UNAVAILABLE)
    )
    check3 = (
        recorded + rejected == total_intents
        and stats["intents"] == total_intents
        and stats["filled"] >= 1
        and stats["open"] == 0  # window closed at the last real close
        and drift["count"] == stats["filled"]
        and abs(drift["mean_pips"]) <= DRIFT_MEAN_BAND_PIPS
        and max(abs(drift["max_pips"]), abs(drift["min_pips"]))
        <= DRIFT_MAX_BAND_PIPS
        and rejection_rate <= REJECTION_RATE_BAND
        and spec_rejected == 0
        and not shadow_gov.state.tripped
        and adapter.write_calls() == []
        and adapter.sent_calls() == []
    )
    print(f"\nCheck 3 — shadow run over {trading_days} trading days: {check3}")
    print(f"  intents submitted={recorded + rejected} "
          f"(stream {total_intents}): recorded={recorded} rejected={rejected} "
          f"rate={rejection_rate:.1%} (band ≤ {REJECTION_RATE_BAND:.0%}); "
          f"refusals={rejection_reasons or '{}'}")
    print(f"  fills={stats['filled']} pending_fill={stats['pending_fill']} "
          f"closed={stats['closed']} rejected_ledger={stats['rejected']}")
    print(f"  fill drift: mean {drift['mean_pips']:+.3f} pips "
          f"(band ±{DRIFT_MEAN_BAND_PIPS}), "
          f"range [{drift['min_pips']:+.3f}, {drift['max_pips']:+.3f}] "
          f"(band ±{DRIFT_MAX_BAND_PIPS}), count {drift['count']}")
    print(f"  would-have P&L: realized {stats['would_have_pnl']['realized']:+.2f} "
          f"total {stats['would_have_pnl']['total']:+.2f} "
          f"(actual spread {live_spread_pips:.2f} pips vs E0 model "
          f"{MODEL_SPREAD_PIPS} pip, predicted+actual on the same bars)")
    if not check3:
        print("RESULT: FAIL")
        return 1

    # ── Check 4 — zero real orders: audit log + broker statement ──────────
    gate_error = None
    try:
        adapter.submit_order(
            OrderRequest(key="e2-gate-probe", symbol=SYMBOL, side="BUY",
                         volume=0.10, comment="e2 live verify gate probe")
        )
    except Exception as exc:  # noqa: BLE001 — refusal IS the pass case
        gate_error = exc
    writes = adapter.write_calls()
    window_start = df.index[0].to_pydatetime()
    alex_deals = adapter.history_deals(window_start, datetime.utcnow(),
                                       magic=ALEX_SHADOW_MAGIC)
    all_deals = adapter.history_deals(window_start, datetime.utcnow())
    after_positions = adapter.positions()
    summary = adapter.audit_summary()
    check4 = (
        gate_error is not None
        and len(writes) == 1
        and writes[0].ok is False
        and writes[0].error == "permission_denied"
        and adapter.sent_calls() == []
        and alex_deals == []  # terminal-side: no Alex-magic deal EVER
        and [p.ticket for p in after_positions] == declared_tickets
        and summary["sent_writes"] == 0
        and summary["permission_level"] == PermissionLevel.OBSERVE
        and summary["shadow_mode"] is True
    )
    print(f"\nCheck 4 — zero real orders (audit log + broker statement): {check4}")
    print(f"  deliberate submit at level 0: {type(gate_error).__name__ if gate_error else 'NOT REFUSED'} "
          f"→ recorded {writes[0].error if writes else 'nothing'} before any transport")
    print(f"  sent_calls={len(adapter.sent_calls())} "
          f"sent_writes={summary['sent_writes']}")
    print(f"  broker history in window: {len(all_deals)} deal(s) total, "
          f"{len(alex_deals)} with magic {ALEX_SHADOW_MAGIC} (must be 0)")
    print(f"  position book unchanged: {declared_tickets} → "
          f"{[p.ticket for p in after_positions]}")
    print(f"  audit calls by mode: {summary['calls']}")
    adapter.disconnect()

    ok = check1 and check2 and check3 and check4
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
