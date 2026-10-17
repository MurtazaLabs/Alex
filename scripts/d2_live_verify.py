"""Stage D2 live verification: event-driven probation accrual on real bars.

1. Fetch REAL recent H1 bars for EUR/USD from the resilient feed chain.
2. Event parity: stream every bar into the EventDrivenProbationAccrual as a
   ``market.bar.closed`` event and assert the accrued trades are IDENTICAL
   (close times + profits) to a batch ``run_backtest`` replay over the same
   bars — the Stage D2 parity contract.
3. Backfill: start a second probation, publish ONLY the newest bar as an
   event, and let the accrual backfill the earlier window from the real feed
   chain; assert the backfilled stream's trades match a batch replay over its
   own window.
4. Write nothing to the real DB — all writes go to a throwaway SQLite file
   in a temp dir.

Run:  .venv311/Scripts/python.exe scripts/d2_live_verify.py
"""

import asyncio
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "src")

import pandas as pd
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.autonomy import probation_events as pe
from forex_ai_agent.autonomy.probation import ProbationStore
from forex_ai_agent.autonomy.probation_events import (
    MARKET_BAR_EVENT,
    EventDrivenProbationAccrual,
)
from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest
from forex_ai_agent.core.event_bus import Event, InMemoryEventBus
from forex_ai_agent.data.models.base import Base as ModelsBase
from forex_ai_agent.data.models.autonomy import StrategyProbation
from forex_ai_agent.data.providers.factory import get_feed_chain
from forex_ai_agent.strategy.spec import parse_spec
from forex_ai_agent.utils.math_helpers import pip_size_for_symbol

SYMBOL = "EUR/USD"
TIMEFRAME = "H1"


def _trade_sig(t: dict):
    ct = t.get("close_time")
    ct_iso = pd.Timestamp(ct).isoformat() if ct is not None else None
    return (ct_iso, round(float(t.get("profit", 0.0)), 10))


async def _temp_store(tmp: str) -> ProbationStore:
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{Path(tmp) / 'd2_verify.db'}",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    return ProbationStore(
        session_factory=async_sessionmaker(engine, expire_on_commit=False)
    )


def _bar_event(b) -> Event:
    return Event(
        event_type=MARKET_BAR_EVENT,
        aggregate_id=SYMBOL,
        aggregate_type="bar",
        payload={
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "timestamp": b.timestamp.isoformat(),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(getattr(b, "volume", 0) or 0),
        },
        source_agent="d2_live_verify",
    )


async def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("forex_ai_agent").setLevel(logging.ERROR)

    chain = get_feed_chain()
    print("feed chain active source:", chain.active_source)
    if chain.active_source == "mock":
        print("SKIP: feed degraded to mock — no real data to verify against")
        return 2

    end = datetime.utcnow()
    start = end - timedelta(days=30)
    bars = await chain.fetch_candles(SYMBOL, TIMEFRAME, start, end=end, limit=800)
    if not bars or len(bars) < 220:
        print(f"FAIL: only {0 if not bars else len(bars)} live bars returned")
        return 1
    print(
        f"fetched {len(bars)} real {TIMEFRAME} bars "
        f"({bars[0].timestamp} → {bars[-1].timestamp})"
    )

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

    rules = {
        "name": "D2 Event Parity Probe",
        "direction": "both",
        "timeframe": "H1",
        "entry_conditions": [
            {"left": "sma", "verb": "above", "right": "sma_slow"},
            {"left": "price", "verb": "above", "right": "sma"},
        ],
    }
    spec, errs = parse_spec(rules)
    if spec is None:
        print("FAIL: probe spec invalid:", errs)
        return 1

    def _reference(frame: pd.DataFrame):
        config = BacktestConfig(
            start_date=frame.index.min().to_pydatetime(),
            end_date=frame.index.max().to_pydatetime(),
            initial_balance=10_000.0,
            symbols=[SYMBOL],
            timeframe=TIMEFRAME,
            warmup_bars=12,
        )
        from forex_ai_agent.autonomy.validator import SpecSignalGenerator

        gen = SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(SYMBOL))
        return run_backtest({SYMBOL: frame}, gen, config)

    tmp = tempfile.mkdtemp(prefix="d2_verify_")
    try:
        # ── Phase A: bars streamed as events == batch replay ─────────────
        real_factory = get_feed_chain

        async def _empty_fetch(pair, tf, s, end=None, limit=None):
            return []

        pe.get_feed_chain = lambda: SimpleNamespace(
            active_source="mt5", fetch_candles=_empty_fetch
        )
        store = await _temp_store(tmp)
        await store.start("d2-event-parity", "D2Probe", SYMBOL, {"name": "D2Probe"})
        bus = InMemoryEventBus()
        accrual = EventDrivenProbationAccrual(
            store=store, event_bus=bus, initial_balance=10_000.0
        )

        async def _always_spec(version_id):
            return spec

        accrual._runner._load_spec = _always_spec
        await accrual.start()
        for b in bars:
            await bus.publish(_bar_event(b))

        stream = accrual._streams.get("d2-event-parity")
        if stream is None:
            print("FAIL: event accrual never built its stream (see warnings)")
            return 1
        accrued = stream.engine.position_manager.closed_trades
        reference = _reference(df)
        sig_a = [_trade_sig(t) for t in accrued]
        sig_ref = [_trade_sig(t) for t in reference.trades or []]
        phase_a = sig_a == sig_ref and len(sig_ref) > 0
        print(f"\nPhase A — event parity: accrued {len(sig_a)} trade(s) as events, "
              f"replay produced {len(sig_ref)}")
        for t in accrued[:5]:
            print(
                f"  event trade close={t.get('close_time')} "
                f"profit={t.get('profit', 0):.2f} reason={t.get('reason', '?')}"
            )
        prob = await store.get("d2-event-parity")
        print(f"  store observed_trades={prob.observed_trades}, "
              f"win_rate={prob.win_rate}, mode={prob.last_metrics.get('_accrual_mode')}")
        print(f"  close_time+profit sequences identical: {phase_a}")

        # ── Phase B: real-chain backfill from a single trailing event ────
        pe.get_feed_chain = real_factory
        # Gate-immune probe probation: a long window/quota so the store's
        # own gates (which correctly expire a 30-day-old probation) don't
        # retire the stream mid-verification.
        await store.start(
            "d2-backfill",
            "D2Probe",
            SYMBOL,
            {"name": "D2Probe"},
            required_trades=1000,
            required_days=365,
        )
        async with store._session_factory() as session:
            await session.execute(
                sa_update(StrategyProbation)
                .where(StrategyProbation.version_id == "d2-backfill")
                .values(started_at=df.index[0].to_pydatetime() - timedelta(hours=1))
            )
            await session.commit()

        await bus.publish(_bar_event(bars[-1]))  # ONLY the newest bar
        stream_b = accrual._streams.get("d2-backfill")
        if stream_b is None:
            print("FAIL: backfilled stream was never built")
            return 1
        own_df = stream_b.feed.data[SYMBOL]
        accrued_b = stream_b.engine.position_manager.closed_trades
        reference_b = _reference(own_df)
        sig_b = [_trade_sig(t) for t in accrued_b]
        sig_ref_b = [_trade_sig(t) for t in reference_b.trades or []]
        phase_b = sig_b == sig_ref_b
        print(f"\nPhase B — backfill: stream holds {len(own_df)} bars "
              f"(asked: {len(df)}), accrued {len(sig_b)} trade(s), "
              f"replay over same window produced {len(sig_ref_b)}")
        print(f"  close_time+profit sequences identical: {phase_b}")

        ok = phase_a and phase_b
        print("\nRESULT:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        pe.get_feed_chain = get_feed_chain
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
