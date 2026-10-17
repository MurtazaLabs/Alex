"""Tests for the resilient feed chain: failover order, health cooldown, failback,
all-feeds-down error, and the factory chain builder."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pytest

from forex_ai_agent.data.providers.base import (
    BaseMarketDataProvider,
    PriceBar,
    ProviderError,
)
from forex_ai_agent.data.providers.mock_provider import MockProvider
from forex_ai_agent.data.providers.resilient import (
    AllFeedsDownError,
    ResilientMarketDataProvider,
)


class FlakyProvider(BaseMarketDataProvider):
    """Fails N times before succeeding — for failover/failback tests."""

    def __init__(self, name: str, fail_times: int = 999, empty: bool = False):
        self.name = name
        self.fail_times = fail_times
        self.empty = empty
        self.calls = 0

    async def fetch_candles(self, symbol, timeframe, start, end=None, limit=1000):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(f"{self.name} down")
        return [] if self.empty else [
            PriceBar(timestamp=datetime.utcnow(), open=1.1, high=1.1, low=1.1, close=1.1)
        ]

    async def fetch_current_price(self, symbol):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(f"{self.name} down")
        return None if self.empty else PriceBar(
            timestamp=datetime.utcnow(), open=1.1, high=1.1, low=1.1, close=1.1
        )

    async def is_connected(self):
        return True


@pytest.fixture
def start() -> datetime:
    return datetime.utcnow() - timedelta(hours=5)


@pytest.mark.asyncio
async def test_failover_to_second_provider(start) -> None:
    primary = FlakyProvider("primary", fail_times=999)
    secondary = FlakyProvider("secondary", fail_times=0)
    chain = ResilientMarketDataProvider([primary, secondary])
    bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=10)
    assert len(bars) == 1
    assert secondary.calls == 1
    assert primary.calls == 1


@pytest.mark.asyncio
async def test_unhealthy_provider_is_skipped_then_reprobed(start) -> None:
    """After 3 failures the primary is skipped (cooldown); failback happens after."""
    primary = FlakyProvider("primary", fail_times=3)  # recovers after 3 failures
    secondary = FlakyProvider("secondary", fail_times=0)
    chain = ResilientMarketDataProvider([primary, secondary], cooldown_s=0.05)

    # Rounds 1-3: primary fails, secondary serves. After 3 failures primary is unhealthy.
    for _ in range(3):
        bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
        assert bars
    assert primary.calls == 3
    assert secondary.calls == 3

    # While unhealthy, the primary is skipped entirely (zero added latency).
    bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
    assert primary.calls == 3  # not re-probed
    assert secondary.calls == 4

    # After the short cooldown expires, the primary is re-probed and recovers (failback).
    import asyncio
    await asyncio.sleep(0.06)
    bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
    assert primary.calls == 4
    assert secondary.calls == 4


@pytest.mark.asyncio
async def test_all_feeds_down_raises(start) -> None:
    chain = ResilientMarketDataProvider(
        [FlakyProvider("a", fail_times=999), FlakyProvider("b", fail_times=999)]
    )
    with pytest.raises(AllFeedsDownError):
        await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
    with pytest.raises(AllFeedsDownError):
        await chain.fetch_current_price("EUR/USD")


@pytest.mark.asyncio
async def test_last_resort_never_marked_unhealthy(start) -> None:
    mock = MockProvider()
    chain = ResilientMarketDataProvider([FlakyProvider("dead", fail_times=999), mock])
    bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
    assert bars  # mock served
    report = {r["name"]: r["status"] for r in chain.health_report()}
    assert report["mock"].startswith("LAST_RESORT")


@pytest.mark.asyncio
async def test_health_report_shape(start) -> None:
    chain = ResilientMarketDataProvider([MockProvider()])
    report = chain.health_report()
    assert report[0]["name"] == "mock"
    assert "status" in report[0]
    assert chain.active_source == "mock"


@pytest.mark.asyncio
async def test_empty_result_counts_as_failure(start) -> None:
    p1 = FlakyProvider("empty", fail_times=0, empty=True)
    p2 = FlakyProvider("good", fail_times=0)
    chain = ResilientMarketDataProvider([p1, p2])
    bars = await chain.fetch_candles("EUR/USD", "H1", start, limit=5)
    assert bars
    assert p2.calls == 1
