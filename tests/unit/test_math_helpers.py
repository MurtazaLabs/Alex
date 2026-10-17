"""Unit tests for deterministic risk / position sizing math."""

from __future__ import annotations

import pytest

from forex_ai_agent.utils.math_helpers import (
    pip_size_for_symbol,
    pip_value_per_lot,
    pips_between,
    position_size_lots,
    risk_amount_usd,
    risk_reward_ratio,
)


def test_pip_size_for_symbol():
    assert pip_size_for_symbol("EUR/USD") == 0.0001
    assert pip_size_for_symbol("USD/JPY") == 0.01
    assert pip_size_for_symbol("GBP/JPY") == 0.01
    with pytest.raises(ValueError):
        pip_size_for_symbol("EURUSD")


def test_pip_value_per_lot():
    assert pip_value_per_lot("EUR/USD") == 10.0
    assert pip_value_per_lot("USD/JPY") == 1000.0


def test_pips_between():
    assert pips_between(1.0850, 1.0820, 0.0001) == pytest.approx(30.0)
    assert pips_between(1.0820, 1.0850, 0.0001) == pytest.approx(30.0)
    with pytest.raises(ValueError):
        pips_between(1.0, 1.1, 0.0)


def test_risk_amount_usd():
    assert risk_amount_usd(10_000, 0.5) == 50.0
    assert risk_amount_usd(10_000, 1.0) == 100.0
    with pytest.raises(ValueError):
        risk_amount_usd(0, 0.5)
    with pytest.raises(ValueError):
        risk_amount_usd(10_000, 6.0)


def test_risk_reward_ratio():
    rr = risk_reward_ratio(entry=1.0850, stop_loss=1.0820, take_profit=1.0890)
    assert rr == pytest.approx(4.0 / 3.0)


def test_position_size_documented_example():
    """
    Doc example (RISK_MANAGEMENT.md):
    Equity $10,000 | risk 0.5% | EUR/USD long 1.0850, SL 1.0820 (30 pips)
    -> raw 0.1667 lots -> floor to 0.01 step -> 0.16 lots.
    """
    size = position_size_lots(
        account_equity=10_000,
        risk_percent=0.5,
        entry_price=1.0850,
        stop_loss=1.0820,
        symbol="EUR/USD",
    )
    assert size == pytest.approx(0.16)


def test_position_size_zero_risk_distance():
    size = position_size_lots(
        account_equity=10_000,
        risk_percent=0.5,
        entry_price=1.0850,
        stop_loss=1.0850,
        symbol="EUR/USD",
    )
    assert size == 0.0
