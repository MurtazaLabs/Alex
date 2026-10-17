"""Initialise the database schema and seed reference data.

Usage:
    python -m scripts.init_db            # uses DATABASE_URL from settings/.env
    DATABASE_URL=sqlite+aiosqlite:///./forex_ai.db python -m scripts.init_db
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forex_ai_agent.data.database import dispose_engine, get_engine, get_session_factory, init_db  # noqa: E402
from forex_ai_agent.data.models import Pair  # noqa: E402
from forex_ai_agent.data.repositories import PairRepository  # noqa: E402

logging.basicConfig(level=logging.INFO)

DEFAULT_PAIRS = [
    # symbol, base, quote, pip_value
    ("EUR/USD", "EUR", "USD", 0.0001),
    ("GBP/USD", "GBP", "USD", 0.0001),
    ("USD/JPY", "USD", "JPY", 0.01),
    ("USD/CHF", "USD", "CHF", 0.0001),
    ("GBP/JPY", "GBP", "JPY", 0.01),
    ("AUD/USD", "AUD", "USD", 0.0001),
    ("USD/CAD", "USD", "CAD", 0.0001),
    ("XAU/USD", "XAU", "USD", 0.01),
]


async def _seed_pairs() -> None:
    async with get_session_factory()() as session:
        repo = PairRepository(session)
        for symbol, base, quote, pip in DEFAULT_PAIRS:
            if await repo.get_by_symbol(symbol) is None:
                await repo.add(
                    Pair(
                        symbol=symbol,
                        base_currency=base,
                        quote_currency=quote,
                        pip_value=pip,
                    )
                )
                logging.getLogger(__name__).info("seeded pair %s", symbol)


async def main() -> None:
    await init_db()
    await _seed_pairs()
    await dispose_engine()
    logging.getLogger(__name__).info("database initialised")


if __name__ == "__main__":
    asyncio.run(main())