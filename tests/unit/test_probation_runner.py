"""Unit tests for the replay-based probation runner.

Uses synthetic bars + a synthetic version store — no network, no real files.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from forex_ai_agent.autonomy.probation import ProbationStore
from forex_ai_agent.data.models.base import Base as ModelsBase


@pytest.fixture()
async def store(session_factory=None):
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


def _trend_bars(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.full(n, 0.00018) + rng.normal(0, 0.0004, n)
    close = 1.10 + np.cumsum(steps)
    idx = pd.date_range("2026-08-01", periods=n, freq="1h")
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


def _started_row(version_id: str = "v-test-1") -> SimpleNamespace:
    """A ProbationStatus-shaped row that just started (no trades yet)."""
    return SimpleNamespace(
        version_id=version_id,
        strategy_name="TestStrat",
        pair="EUR/USD",
        status="active",
        observed_trades=0,
        required_trades=20,
        observed_days=0,
        required_days=14,
        days_elapsed=0,
        win_rate=None,
        profit_factor=None,
        worst_daily_loss_pct=None,
        last_metrics=None,
        eligible_for_approval=False,
        notes="",
        started_at=datetime.utcnow().isoformat(),
        completed_at=None,
    )


def _result_with_trades(times_profits):
    """BacktestResult-shaped object with trades closing at given times."""
    return SimpleNamespace(
        trades=[
            {
                "close_time": t,
                "profit": p,
                "open_time": t - timedelta(hours=2),
            }
            for t, p in times_profits
        ]
    )


def test_watermark_filter_dedups_trades():
    """Trades at/under the watermark are excluded; strictly-after are kept."""
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner

    r = ProbationRunner.__new__(ProbationRunner)  # no __init__ side effects
    r.initial_balance = 10_000.0
    wm = datetime(2026, 8, 10, 12, 0)
    prob = _started_row()
    prob.last_metrics = {"_replay_through": wm.isoformat()}
    trades = [
        {"close_time": datetime(2026, 8, 9), "profit": 1.0},    # before → out
        {"close_time": wm, "profit": 1.0},                      # equal → out
        {"close_time": datetime(2026, 8, 10, 13, 0), "profit": -1.0},  # after → in
        {"close_time": datetime(2026, 8, 12), "profit": 2.0},   # after → in
    ]
    kept = [
        t for t in trades if (r._close_time(t) or datetime.utcnow()) > wm
    ]
    assert len(kept) == 2
    metrics = r._cumulative_metrics(prob, kept)
    assert metrics["_trades_n"] == 2
    assert metrics["win_rate"] == 0.5
    assert metrics["_replay_through"] == datetime(2026, 8, 12).isoformat()


def test_cumulative_metrics_merge():
    """Second pass merges into first pass's aggregates (no overwrite)."""
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner

    r = ProbationRunner.__new__(ProbationRunner)
    r.initial_balance = 10_000.0
    prob = _started_row()

    first = _result_with_trades(
        [
            (datetime(2026, 8, 2), 100.0),
            (datetime(2026, 8, 3), -50.0),
        ]
    )
    m1 = r._cumulative_metrics(prob, first.trades)
    assert m1["_trades_n"] == 2
    assert m1["win_rate"] == 0.5
    assert m1["profit_factor"] == pytest.approx(2.0)

    prob.last_metrics = dict(m1)
    second = _result_with_trades([(datetime(2026, 8, 5), -80.0)])
    m2 = r._cumulative_metrics(prob, second.trades)
    assert m2["_trades_n"] == 3
    assert m2["_wins_n"] == 1
    assert m2["win_rate"] == pytest.approx(round(1 / 3, 4))
    assert m2["profit_factor"] == pytest.approx(round(100 / 130, 4))


def test_cumulative_metrics_no_new_trades_preserves_state():
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner

    r = ProbationRunner.__new__(ProbationRunner)
    r.initial_balance = 10_000.0
    prob = _started_row()
    prob.last_metrics = {
        "win_rate": 0.5,
        "profit_factor": 1.4,
        "_trades_n": 4,
        "_wins_n": 2,
        "_gross_win": 200.0,
        "_gross_loss": 100.0,
        "_replay_through": datetime(2026, 8, 10).isoformat(),
    }
    m = r._cumulative_metrics(prob, [])
    assert m["win_rate"] == 0.5
    assert m["profit_factor"] == 1.4
    assert m["_trades_n"] == 4
    assert m["_replay_through"] == datetime(2026, 8, 10).isoformat()


@pytest.mark.asyncio
async def test_run_one_waiting_for_bars(store):
    """Too-few bars → waiting status, no trades credited, state untouched."""
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner

    await store.start("v-1", "Strat", "EUR/USD", {"name": "Strat"})
    runner = ProbationRunner(store=store)

    async def _has_spec(*a, **k):
        from forex_ai_agent.strategy.spec import fallback_spec

        return fallback_spec("EUR/USD")

    async def _no_bars(*a, **k):
        return []

    runner._load_spec = _has_spec

    runner._fetch_bars = _no_bars
    outcome = await runner.run_one("v-1")
    assert outcome["status"] == "waiting_for_bars"
    prob = await store.get("v-1")
    assert prob.observed_trades == 0
    assert prob.status == "active"


@pytest.mark.asyncio
async def test_run_one_accrues_and_terminal(store):
    """Trades flow through: accrual updates counts; the store judges."""
    from forex_ai_agent.autonomy.probation_runner import ProbationRunner

    await store.start("v-2", "Strat", "EUR/USD", {"name": "Strat"})

    runner = ProbationRunner(store=store)

    async def _has_spec(*a, **k):
        from forex_ai_agent.strategy.spec import fallback_spec

        return fallback_spec("EUR/USD")

    runner._load_spec = _has_spec
    n_trades = 22
    profits = [120.0 if i % 2 == 0 else -60.0 for i in range(n_trades)]
    base = datetime.utcnow() - timedelta(hours=n_trades * 2)
    closes = [base + timedelta(hours=2 * i) for i in range(n_trades)]

    async def _fake_bars(*a, **k):
        return _trend_bars(60)

    async def _fake_replay(spec, pair, bars):
        return _result_with_trades(list(zip(closes, profits)))

    runner._fetch_bars = _fake_bars
    runner._replay = _fake_replay

    outcome = await runner.run_one("v-2")
    assert outcome["accrued"] == n_trades
    assert outcome["observed_trades"] == n_trades
    # 50% win rate, PF 2.0 → quota met, window not over → still active
    assert outcome["status"] == "active"

    # Idempotency: a second identical pass must credit ZERO new trades.
    outcome2 = await runner.run_one("v-2")
    assert outcome2["accrued"] == 0
    assert outcome2["observed_trades"] == n_trades

    # Force the window over → next pass flips to passed (win rate 50% ≥ 30%).
    from sqlalchemy import select, update as sa_update

    from forex_ai_agent.data.models.autonomy import StrategyProbation

    session_factory = store._session_factory
    async with session_factory() as session:
        await session.execute(
            sa_update(StrategyProbation)
            .where(StrategyProbation.version_id == "v-2")
            .values(started_at=datetime.utcnow() - timedelta(days=20))
        )
        await session.commit()

    outcome3 = await runner.run_one("v-2")
    assert outcome3["status"] == "passed"
    assert outcome3["observed_trades"] == n_trades  # still exactly once

    # And now the human-only button can fire.
    approved = await store.approve("v-2", approver="test")
    assert approved.status == "approved"
