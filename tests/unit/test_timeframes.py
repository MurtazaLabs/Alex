"""Unit tests for timeframe utilities."""

from __future__ import annotations

import pytest

from forex_ai_agent.utils.timeframes import (
    TIMEFRAME_ORDER,
    Timeframe,
    higher_timeframes,
    is_valid_timeframe,
    lower_timeframes,
    timeframe_minutes,
)


def test_order_and_minutes():
    assert TIMEFRAME_ORDER == ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
    assert timeframe_minutes("M1") == 1
    assert timeframe_minutes("H1") == 60
    assert timeframe_minutes("D1") == 1440


def test_timeframe_enum():
    assert Timeframe.H4.minutes == 240
    assert Timeframe.M15.display_name == "15 Minutes"
    assert Timeframe.D1.position == 6
    assert Timeframe("H1") == "H1"


def test_is_valid_timeframe():
    assert is_valid_timeframe("H1")
    assert not is_valid_timeframe("P1")
    assert not is_valid_timeframe("")


def test_validity_errors():
    with pytest.raises(ValueError):
        timeframe_minutes("W1")


def test_higher_timeframes():
    assert higher_timeframes("M15") == ["M30", "H1", "H4", "D1"]


def test_lower_timeframes():
    assert lower_timeframes("H1") == ["M1", "M5", "M15", "M30"]


def test_relative_errors():
    with pytest.raises(ValueError):
        higher_timeframes("P1")
    with pytest.raises(ValueError):
        lower_timeframes("P1")
