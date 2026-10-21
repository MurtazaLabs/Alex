"""
Deterministic financial math helpers.

These are pure functions: the LLM must never be the source of risk math.
All values are computed here so they can be unit-tested against known values.
"""

from __future__ import annotations

import math


def pip_size_for_symbol(symbol: str) -> float:
    """
    Return the pip size for a currency pair.

    JPY-quoted pairs have 0.01 pips, everything else 0.0001.
    """
    if "/" not in symbol:
        raise ValueError(f"Invalid pair symbol: {symbol!r}")
    _, quote = symbol.split("/", maxsplit=1)
    return 0.01 if quote.upper() == "JPY" else 0.0001


def pips_between(price_a: float, price_b: float, pip_size: float) -> float:
    """Absolute distance between two prices expressed in pips."""
    if pip_size <= 0:
        raise ValueError(f"Invalid pip size: {pip_size!r}")
    return abs(price_a - price_b) / pip_size


def pip_value_per_lot(symbol: str, contract_size: float = 100_000.0) -> float:
    """
    Monetary value of one pip per standard lot for a pair.

    EUR/USD 1 lot: 0.0001 * 100 000 = $10 per pip.
    """
    return pip_size_for_symbol(symbol) * contract_size


def risk_amount_usd(account_equity: float, risk_percent: float) -> float:
    """
    Dollar risk per trade at ``risk_percent`` of account equity.

    ``risk_percent`` is expressed as a percentage (0.5 -> 0.5%).
    """
    if account_equity <= 0:
        raise ValueError("Account equity must be positive")
    if not 0 < risk_percent <= 5.0:
        raise ValueError(f"Risk percent out of bounds: {risk_percent!r}")
    return account_equity * (risk_percent / 100.0)


def risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Risk:Reward ratio (reward / risk) for a trade."""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    if risk <= 0:
        raise ValueError("Entry and stop loss cannot be equal")
    return reward / risk


def raw_position_size_lots(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    symbol: str,
    contract_size: float = 100_000.0,
) -> float:
    """
    Unclamped ideal position size in lots for the full risk budget.

    Position Size = Account Risk / (SL distance in pips x pip value per lot)
    Returns 0.0 when the setup is un-tradeable (e.g. zero risk distance).

    Callers use this to detect the mandate case where even the MINIMUM lot
    would risk more than the budget: raw < min_lot means the trade must be
    REFUSED, never clamped up to the minimum.
    """
    risk_amount = risk_amount_usd(account_equity, risk_percent)
    sl_pips = pips_between(entry_price, stop_loss, pip_size_for_symbol(symbol))
    pip_value = pip_value_per_lot(symbol, contract_size)

    if sl_pips <= 0 or pip_value <= 0:
        return 0.0

    return risk_amount / (sl_pips * pip_value)


def position_size_lots(
    account_equity: float,
    risk_percent: float,
    entry_price: float,
    stop_loss: float,
    symbol: str,
    contract_size: float = 100_000.0,
    min_lot_size: float = 0.01,
    max_lot_size: float = 100.0,
    lot_step: float = 0.01,
) -> float:
    """
    Calculate the position size in lots for a trade.

    Position Size = Account Risk / (SL distance in pips x pip value per lot)
    Rounds DOWN to the lot step, then clamps into [min_lot, max_lot].
    Returns 0.0 when the setup is un-tradeable (e.g. zero risk distance).

    Note: the clamp up to ``min_lot_size`` is only reachable when the raw
    size is at or above the minimum; risk-budget enforcement (refusing when
    raw < min) belongs to the caller (ExecutionPolicy._validate).
    """
    raw_size = raw_position_size_lots(
        account_equity, risk_percent, entry_price, stop_loss,
        symbol, contract_size,
    )
    if raw_size <= 0.0:
        return 0.0

    rounded = math.floor(raw_size / lot_step) * lot_step
    return max(min_lot_size, min(max_lot_size, rounded))
