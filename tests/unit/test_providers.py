"""Unit tests for market data providers."""

from __future__ import annotations

from datetime import datetime

import pytest

from forex_ai_agent.data.providers import (
    MockProvider,
    MT5Provider,
    get_market_data_provider,
    register_provider,
)


@pytest.mark.asyncio
async def test_mock_provider_reproducible():
    start = datetime(2026, 1, 1)
    p1 = MockProvider()
    a = await p1.fetch_candles("EUR/USD", "H1", start, limit=20)
    b = await p1.fetch_candles("EUR/USD", "H1", start, limit=20)
    assert [bar.close for bar in a] == [bar.close for bar in b]
    assert len(a) == 20


@pytest.mark.asyncio
async def test_mock_provider_bar_invariants():
    start = datetime(2026, 1, 1)
    provider = MockProvider(start_price=1.10)
    bars = await provider.fetch_candles("GBP/USD", "H1", start, limit=50)
    for bar in bars:
        assert bar.low <= bar.open <= bar.high
        assert bar.low <= bar.close <= bar.high
        assert bar.high >= bar.low
        assert bar.volume >= 0


@pytest.mark.asyncio
async def test_mock_provider_current_price_and_connection():
    provider = MockProvider()
    bar = await provider.fetch_current_price("EUR/USD")
    assert bar is not None
    assert bar.close > 0
    assert await provider.is_connected() is True


@pytest.mark.asyncio
async def test_mt5_provider_requires_terminal(monkeypatch):
    """Without the MetaTrader5 package the provider raises MT5Unavailable."""
    import builtins
    import sys
    from forex_ai_agent.data.providers.mt5_provider import MT5Unavailable

    monkeypatch.delitem(sys.modules, "MetaTrader5", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "MetaTrader5":
            raise ImportError("no mt5")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    provider = MT5Provider(login=123, password="x", server="demo")
    assert await provider.is_connected() is False
    with pytest.raises(MT5Unavailable):
        await provider.fetch_candles("EUR/USD", "H1", datetime(2026, 1, 1), limit=5)


def test_factory_defaults_to_mock():
    provider = get_market_data_provider("mock")
    assert isinstance(provider, MockProvider)


def test_factory_unknown_provider():
    with pytest.raises(ValueError):
        get_market_data_provider("does_not_exist")


def test_register_provider():
    class Custom(MockProvider):
        name = "custom"

    register_provider("custom", Custom)
    assert isinstance(get_market_data_provider("custom"), Custom)
