"""Unit tests for the Stage D2 event-driven probation accrual.

Covers the D2 invariants:
- the extracted per-bar engine path produces byte-identical results to the
  batch ``run()`` loop (parity by construction),
- streaming the same bars as ``market.bar.closed`` events produces the exact
  same trades as the batch replay (close times + profits),
- the shared ``_replay_through`` watermark dedups exactly across BOTH accrual
  paths in either order,
- the replay runner yields to a fresh event stream and reclaims a stale one,
- duplicate / out-of-order events are ignored, gaps are repaired,
- the mock-data guard holds for both the pump and the accrual,
- the pump publishes only closed bars, in order, exactly once.

No network, no real files: synthetic bars + in-memory SQLite + in-memory bus.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import update as sa_update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.autonomy import probation_events
from forex_ai_agent.autonomy.probation import ProbationStore
from forex_ai_agent.autonomy.probation_events import (
    MARKET_BAR_EVENT,
    EventDrivenProbationAccrual,
    MarketBarEventPump,
)
from forex_ai_agent.autonomy.probation_runner import ProbationRunner
from forex_ai_agent.backtest.engine import (
    BacktestConfig,
    DataFeed,
    SignalGenerator,
    run_backtest,
)
from forex_ai_agent.core.event_bus import Event, InMemoryEventBus
from forex_ai_agent.data.models.base import Base as ModelsBase


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
async def store():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    s = ProbationStore(
        session_factory=async_sessionmaker(engine, expire_on_commit=False)
    )
    yield s
    await engine.dispose()


def _trend_bars(
    n: int = 160, seed: int = 11, start: datetime = datetime(2026, 8, 1)
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.full(n, 0.00018) + rng.normal(0, 0.0004, n)
    close = 1.10 + np.cumsum(steps)
    idx = pd.date_range(start, periods=n, freq="1h")
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close + np.abs(rng.normal(0, 0.0003, n)),
            "low": close - np.abs(rng.normal(0, 0.0003, n)),
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


def _bar_objects(df: pd.DataFrame) -> list:
    return [
        SimpleNamespace(
            timestamp=ts,
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["volume"]),
        )
        for ts, r in df.iterrows()
    ]


def _bars_from_df_window(df: pd.DataFrame, start_row: int, n: int) -> list:
    """Bar objects for df rows [start_row, start_row + n)."""
    window = df.iloc[start_row : start_row + n]
    return _bar_objects(window)


class _ScriptedLongGen(SignalGenerator):
    """Deterministic generator: a LONG every ``every`` bars, fixed ATR-less
    brackets. Lets tests control the trade count independent of spec verbs."""

    def __init__(
        self,
        spec: Any = None,
        pip_size: float = 0.0001,
        every: int = 10,
        cooldown_bars: int = 12,
        risk_per_trade_pct: float = 0.01,
        account_balance: float = 100_000.0,
    ) -> None:
        self.every = every
        self._count = 0

    def generate_signals(
        self,
        timestamp: datetime,
        data: Dict[str, Any],
        indicators: Dict[str, Dict[str, float]],
    ) -> List[Dict]:
        self._count += 1
        if self._count % self.every != 0:
            return []
        out = []
        for symbol, bar in data.items():
            out.append(
                {
                    "symbol": symbol,
                    "direction": "LONG",
                    "entry_price": float(bar.close),
                    "sl": float(bar.close) - 0.004,
                    "tp": float(bar.close) + 0.008,
                    "volume": 0.5,
                    "comment": "scripted",
                }
            )
        return out


def _bar_event(pair: str, tf: str, b: Any) -> Event:
    return Event(
        event_type=MARKET_BAR_EVENT,
        aggregate_id=pair,
        aggregate_type="bar",
        payload={
            "symbol": pair,
            "timeframe": tf,
            "timestamp": b.timestamp.isoformat(),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(getattr(b, "volume", 0) or 0),
        },
        source_agent="test",
    )


def _fake_chain(source: str = "mt5", bars: Optional[list] = None) -> SimpleNamespace:
    async def fetch(pair, tf, start, end=None, limit=None):
        return list(bars or [])

    return SimpleNamespace(active_source=source, fetch_candles=fetch)


def _patch_scripted_generator(monkeypatch, modules) -> None:
    for mod in modules:
        monkeypatch.setattr(mod, "SpecSignalGenerator", _ScriptedLongGen)


def _replay_reference(pair: str, df: pd.DataFrame):
    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=10_000.0,
        symbols=[pair],
        timeframe="H1",
        warmup_bars=12,
    )
    return run_backtest({pair: df}, _ScriptedLongGen(), config)


# ---------------------------------------------------------------------------
# Engine-level parity: extracted _process_bar vs batch run()
# ---------------------------------------------------------------------------


def test_incremental_process_bar_matches_batch_run():
    """The D2 extraction must not change the batch loop's numbers."""
    from forex_ai_agent.backtest.engine import BacktestEngine

    pair = "EUR/USD"
    df = _trend_bars(200, seed=3)

    batch = _replay_reference(pair, df)

    feed = DataFeed({})
    config = BacktestConfig(
        start_date=df.index.min().to_pydatetime(),
        end_date=df.index.max().to_pydatetime(),
        initial_balance=10_000.0,
        symbols=[pair],
        timeframe="H1",
        warmup_bars=12,
    )
    incremental_engine = BacktestEngine(config, feed, _ScriptedLongGen())
    for ts, r in df.iterrows():
        feed.append_bar(
            pair,
            ts,
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        )
        assert incremental_engine._process_bar(ts) is True

    assert len(batch.trades) == len(incremental_engine.position_manager.closed_trades)
    for bt, it in zip(
        batch.trades, incremental_engine.position_manager.closed_trades
    ):
        assert bt["close_time"] == it["close_time"]
        assert bt["profit"] == it["profit"]
    assert batch.equity_curve == incremental_engine.equity_curve
    assert len(batch.trades) > 0  # the comparison must be meaningful


# ---------------------------------------------------------------------------
# Event accrual parity + watermark dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_accrual_matches_batch_replay(store, monkeypatch):
    """THE D2 invariant: bars streamed as events accrue the exact same
    trades (close times + profits) as the batch replay engine."""
    pair = "EUR/USD"
    df = _trend_bars(160, seed=11)
    bars = _bar_objects(df)

    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mt5", [])
    )
    _patch_scripted_generator(monkeypatch, [probation_events])

    await store.start("v-par", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(store=store, event_bus=bus, initial_balance=10_000.0)
    _patch_load_spec(accrual)
    await accrual.start()

    for b in bars:
        await bus.publish(_bar_event(pair, "H1", b))

    reference = _replay_reference(pair, df)
    stream = accrual._streams["v-par"]
    accrued = stream.engine.position_manager.closed_trades

    assert len(accrued) == len(reference.trades)
    assert len(accrued) > 0
    for rt, at in zip(reference.trades, accrued):
        assert rt["close_time"] == at["close_time"]
        assert rt["profit"] == at["profit"]

    prob = await store.get("v-par")
    assert prob.observed_trades == len(reference.trades)
    metrics = prob.last_metrics
    assert metrics["_accrual_mode"] == "event"
    assert metrics["_event_through"] == bars[-1].timestamp.isoformat()
    assert metrics["_trades_n"] == len(reference.trades)


@pytest.mark.asyncio
async def test_shared_watermark_dedups_across_paths(store, monkeypatch):
    """Replay credits first, events continue: no double counting, and the
    union equals one full batch replay over all bars."""
    from forex_ai_agent.autonomy import probation_runner as pr_module
    from forex_ai_agent.strategy.spec import fallback_spec

    pair = "EUR/USD"
    df1 = _trend_bars(160, seed=11)
    df2 = _trend_bars(60, seed=12, start=df1.index[-1] + timedelta(hours=1))
    bars1 = _bar_objects(df1)
    bars2 = _bar_objects(df2)

    monkeypatch.setattr(pr_module, "get_feed_chain", lambda: _fake_chain("mt5"))
    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mt5", [])
    )
    _patch_scripted_generator(monkeypatch, [probation_events, pr_module])

    await store.start("v-x", "Strat", pair, {"name": "Strat"})

    runner = ProbationRunner(store=store, initial_balance=10_000.0)

    async def _spec(version_id):
        return fallback_spec(pair)

    async def _bars(pair_, tf, start, end):
        return bars1

    runner._load_spec = _spec
    runner._fetch_bars = _bars
    outcome = await runner.run_one("v-x")
    replay_trades = outcome["accrued"]
    assert replay_trades > 0

    # Now the SAME bars arrive as events: everything is at/under the
    # watermark → zero new credits.
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(
        store=store, event_bus=bus, initial_balance=10_000.0, runner=runner
    )
    await accrual.start()
    for b in bars1:
        await bus.publish(_bar_event(pair, "H1", b))
    prob = await store.get("v-x")
    assert prob.observed_trades == replay_trades

    # Continuation bars arrive as events → credited exactly once.
    for b in bars2:
        await bus.publish(_bar_event(pair, "H1", b))
    prob = await store.get("v-x")
    after_events = prob.observed_trades
    assert after_events > replay_trades

    # Invariant: credited union == one full batch replay over all bars.
    full = _replay_reference(pair, pd.concat([df1, df2]))
    assert after_events == len(full.trades)


@pytest.mark.asyncio
async def test_events_first_then_stale_replay_reclaims(store, monkeypatch):
    """Event stream accrues; a stale replay pass reclaims ownership but
    credits nothing new (identical close times, strict watermark)."""
    from forex_ai_agent.autonomy import probation_runner as pr_module
    from forex_ai_agent.strategy.spec import fallback_spec

    pair = "EUR/USD"
    df = _trend_bars(160, seed=11)
    bars = _bar_objects(df)

    monkeypatch.setattr(pr_module, "get_feed_chain", lambda: _fake_chain("mt5"))
    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mt5", [])
    )
    _patch_scripted_generator(monkeypatch, [probation_events, pr_module])

    await store.start("v-y", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    runner = ProbationRunner(store=store, initial_balance=10_000.0)

    async def _spec(version_id):
        return fallback_spec(pair)

    runner._load_spec = _spec  # before the event loop: the accrual shares it
    accrual = EventDrivenProbationAccrual(
        store=store, event_bus=bus, initial_balance=10_000.0, runner=runner
    )
    await accrual.start()
    for b in bars:
        await bus.publish(_bar_event(pair, "H1", b))

    prob = await store.get("v-y")
    event_trades = prob.observed_trades
    assert event_trades > 0
    assert prob.last_metrics["_accrual_mode"] == "event"

    async def _bars(pair_, tf, start, end):
        return bars

    runner._fetch_bars = _bars
    outcome = await runner.run_one("v-y")

    # Bar timestamps are hours old → stream counts as stale → replay ran.
    assert outcome["status"] != "event_owned"
    assert outcome["accrued"] == 0  # identical re-derived trades, all <= watermark
    prob = await store.get("v-y")
    assert prob.observed_trades == event_trades
    assert prob.last_metrics["_accrual_mode"] == "replay"


@pytest.mark.asyncio
async def test_runner_yields_to_fresh_event_stream(store, monkeypatch):
    """A fresh (_event_through < 45min old) event stream owns accrual."""
    await store.start("v-z", "Strat", "EUR/USD", {"name": "Strat"})
    await store.update(
        "v-z",
        metrics={
            "_accrual_mode": "event",
            "_event_through": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
        },
    )

    runner = ProbationRunner(store=store)

    async def _boom(*a, **k):  # must never be reached
        raise AssertionError("replay must not run while the event stream is fresh")

    runner._load_spec = _boom
    runner._fetch_bars = _boom
    outcome = await runner.run_one("v-z")
    assert outcome == {"version_id": "v-z", "status": "event_owned", "accrued": 0}


# ---------------------------------------------------------------------------
# Event hygiene: duplicates, ordering, gaps, guards, lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_and_out_of_order_events_ignored(store, monkeypatch):
    pair = "EUR/USD"
    df = _trend_bars(60, seed=7)
    bars = _bar_objects(df)

    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mt5", [])
    )
    _patch_scripted_generator(monkeypatch, [probation_events])

    await store.start("v-dup", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(store=store, event_bus=bus, initial_balance=10_000.0)
    _patch_load_spec(accrual)
    await accrual.start()

    for b in bars[:50]:
        await bus.publish(_bar_event(pair, "H1", b))
    stream = accrual._streams["v-dup"]
    through = stream.processed_through
    df_len = len(stream.feed.data[pair])

    # Replays of already-processed bars must be no-ops.
    for b in bars[:50]:
        await bus.publish(_bar_event(pair, "H1", b))
    assert stream.processed_through == through
    assert len(stream.feed.data[pair]) == df_len
    prob = await store.get("v-dup")
    assert prob.observed_trades == 0  # scripted trades start at bar 10 →
    # (positions from the first pass are still open; nothing new closed twice)


@pytest.mark.asyncio
async def test_gap_repair_backfills_missing_bars(store, monkeypatch):
    """An event that skips ahead triggers a deterministic backfill of the
    missed bars before it is processed."""
    pair = "EUR/USD"
    df = _trend_bars(10, seed=21)
    bars = _bar_objects(df)

    # The chain serves the full range when asked for the gap (T0, T3).
    chain = _fake_chain("mt5", bars)
    monkeypatch.setattr(probation_events, "get_feed_chain", lambda: chain)
    _patch_scripted_generator(monkeypatch, [probation_events])

    await store.start("v-gap", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(store=store, event_bus=bus, initial_balance=10_000.0)
    _patch_load_spec(accrual)
    await accrual.start()

    # First event seeds the stream (backfill up to bars[0] is empty).
    await bus.publish(_bar_event(pair, "H1", bars[0]))
    stream = accrual._streams["v-gap"]
    assert stream.processed_through == bars[0].timestamp

    # Jump 3 hours ahead: bars 1..2 must be repaired from the chain.
    await bus.publish(_bar_event(pair, "H1", bars[3]))
    assert stream.processed_through == bars[3].timestamp
    ingested = stream.feed.data[pair]
    assert len(ingested) == 4
    assert list(ingested.index) == [b.timestamp for b in bars[:4]]


@pytest.mark.asyncio
async def test_mock_guard_blocks_accrual_and_pump(store, monkeypatch):
    """No probation number may ever be derived from mock data."""
    pair = "EUR/USD"
    df = _trend_bars(30, seed=5)
    bars = _bar_objects(df)

    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mock", bars)
    )
    _patch_scripted_generator(monkeypatch, [probation_events])

    await store.start("v-mock", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(store=store, event_bus=bus, initial_balance=10_000.0)
    _patch_load_spec(accrual)
    await accrual.start()
    for b in bars:
        await bus.publish(_bar_event(pair, "H1", b))

    assert accrual._streams == {}
    prob = await store.get("v-mock")
    assert prob.observed_trades == 0

    pump = MarketBarEventPump(
        bus, subscriptions=_async_subscriptions([(pair, "H1")])
    )
    n_before = len(bus.published)  # the accrual test published its own events
    assert await pump.publish_new_bars() == 0
    assert len(bus.published) == n_before


def _async_subscriptions(subs):
    async def _subs():
        return list(subs)

    return _subs


async def _always_spec(version_id):
    """Spec loader stub for accruals that own their internal runner."""
    from forex_ai_agent.strategy.spec import fallback_spec

    return fallback_spec("EUR/USD")


def _patch_load_spec(accrual: EventDrivenProbationAccrual) -> None:
    """The accrual's internal runner would otherwise hit the real version
    store on disk; tests supply the deterministic fallback spec instead."""
    accrual._runner._load_spec = _always_spec


@pytest.mark.asyncio
async def test_pump_publishes_closed_bars_once(monkeypatch):
    pair = "EUR/USD"
    df = _trend_bars(4, seed=9)
    closed = _bar_objects(df)
    forming = SimpleNamespace(
        timestamp=datetime.utcnow() - timedelta(minutes=5),
        open=1.1, high=1.1, low=1.1, close=1.1, volume=1.0,
    )

    served = list(closed) + [forming]
    chain = _fake_chain("mt5", served)
    monkeypatch.setattr(probation_events, "get_feed_chain", lambda: chain)

    bus = InMemoryEventBus()
    pump = MarketBarEventPump(
        bus, subscriptions=_async_subscriptions([(pair, "H1")])
    )

    published = await pump.publish_new_bars()
    assert published == 4  # forming bar excluded, closed bars in order
    assert [e.payload["timestamp"] for e in bus.published] == [
        b.timestamp.isoformat() for b in closed
    ]

    # Same poll again → dedup by the newest published bar.
    assert await pump.publish_new_bars() == 0

    # One newer closed bar arrives → exactly one new event.
    newer = SimpleNamespace(
        timestamp=df.index[-1] + timedelta(hours=5),
        open=1.1, high=1.2, low=1.0, close=1.15, volume=1.0,
    )
    served.append(newer)
    assert await pump.publish_new_bars() == 1
    assert bus.published[-1].payload["close"] == 1.15


@pytest.mark.asyncio
async def test_started_at_change_rebuilds_stream(store, monkeypatch):
    from forex_ai_agent.data.models.autonomy import StrategyProbation

    pair = "EUR/USD"
    df = _trend_bars(30, seed=13)
    bars = _bar_objects(df)

    monkeypatch.setattr(
        probation_events, "get_feed_chain", lambda: _fake_chain("mt5", [])
    )
    _patch_scripted_generator(monkeypatch, [probation_events])

    await store.start("v-life", "Strat", pair, {"name": "Strat"})
    bus = InMemoryEventBus()
    accrual = EventDrivenProbationAccrual(store=store, event_bus=bus, initial_balance=10_000.0)
    _patch_load_spec(accrual)
    await accrual.start()

    for b in bars[:10]:
        await bus.publish(_bar_event(pair, "H1", b))
    stream = accrual._streams["v-life"]
    old_started = stream.started_at

    new_started = (datetime.utcnow() + timedelta(days=1)).isoformat()
    async with store._session_factory() as session:
        await session.execute(
            sa_update(StrategyProbation)
            .where(StrategyProbation.version_id == "v-life")
            .values(started_at=datetime.fromisoformat(new_started[:19]))
        )
        await session.commit()

    await bus.publish(_bar_event(pair, "H1", bars[10]))
    stream = accrual._streams["v-life"]
    assert stream.started_at != old_started
    assert stream.started_at[:19] == new_started[:19]
    assert stream.processed_through == bars[10].timestamp
