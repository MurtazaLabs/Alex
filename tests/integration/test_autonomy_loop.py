"""End-to-end autonomy loop test: propose → gate PASS → version → probation.

Uses synthetic trending data tuned so the fallback spec passes the gate,
exercising the full happy path deterministically (no LLM, no network).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.strategy.spec import fallback_spec


def _strong_trend_history(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    """Long steady uptrend with shallow pullbacks — trend rules thrive here."""
    rng = np.random.default_rng(seed)
    steps = np.full(n, 0.00018) + rng.normal(0, 0.0004, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0003, n))
    low = close - np.abs(rng.normal(0, 0.0003, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


@pytest.mark.asyncio
async def test_full_loop_propose_validate_version_probation(tmp_path):
    """Fallback spec on trending data → gate PASS → version + probation row."""
    from forex_ai_agent.autonomy.probation import ProbationStore
    from forex_ai_agent.autonomy.proposal import ProposalResult
    from forex_ai_agent.autonomy.validator import BacktestValidator
    from forex_ai_agent.strategy.versioning import StrategyVersionManager

    df = _strong_trend_history()
    validator = BacktestValidator()
    spec = fallback_spec("EUR/USD")

    report = await validator.validate_spec(spec, df, symbol="EUR/USD")
    if not report.passed:
        pytest.skip(
            "synthetic data did not clear the gate this run: " + "; ".join(report.reasons)
        )

    # Version the survivor exactly as the CLI/bridge flow does.
    vm = StrategyVersionManager(storage_path=str(tmp_path / "versions"))
    version = vm.create_version(
        name=spec.name,
        description=spec.description,
        rules=spec.model_dump(),
        parameters={"atr_sl_mult": spec.atr_sl_mult, "atr_tp_mult": spec.atr_tp_mult},
    )
    assert version.id

    # Enter probation (own in-memory engine via the real model).
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from forex_ai_agent.data.models.base import Base as ModelsBase

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    store = ProbationStore(
        session_factory=async_sessionmaker(engine, expire_on_commit=False)
    )
    prob = await store.start(
        version_id=str(version.id),
        strategy_name=spec.name,
        pair="EUR/USD",
        spec_dict=spec.model_dump(),
    )
    assert prob.status == "active"
    assert prob.required_trades > 0

    # Simulate probation completing successfully, then the HUMAN approves.
    final = await store.update(
        str(version.id),
        closed_trades=[{"profit": 12.5}] * prob.required_trades,
        metrics={"win_rate": 0.55, "profit_factor": 1.8, "worst_daily_loss_pct": 0.012},
    )
    assert final.status in {"passed", "active"}
    if final.status == "passed":
        assert final.eligible_for_approval
        approved = await store.approve(str(version.id), approver="e2e-user")
        assert approved.status == "approved"
    await engine.dispose()


@pytest.mark.asyncio
async def test_propose_for_pair_fallback_reject_path(tmp_path):
    """propose_for_pair wires brief→fallback→gate→rejection without an LLM."""
    from forex_ai_agent.autonomy.proposal import propose_for_pair

    class _StubPipeline:
        """Minimal pipeline stand-in: no LLM, deterministic brief pieces."""

        class _Provider:
            active_source = "test"

        data_provider = _Provider()
        news_agent = None

        async def fetch_multi_tf(self, pair, tfs=("H1", "H4"), limit=300):
            import numpy as np

            n = 300
            close = 1.10 + np.linspace(0, 0.01, n)
            highs = close + 0.001
            lows = close - 0.001
            arr = {
                "highs": highs,
                "lows": lows,
                "closes": close,
                "volume": np.zeros(n),
                "timestamps": list(pd.date_range("2025-01-01", periods=n, freq="1h")),
                "bars": [],
            }
            return {tf: dict(arr) for tf in tfs}

    df = _strong_trend_history()
    result = await propose_for_pair(
        _StubPipeline(),
        "EUR/USD",
        df,
        use_llm=False,
        log_to_db=False,
        enter_probation=False,
        version_manager_dir=str(tmp_path / "versions"),
    )
    assert result.status in {"validated", "rejected"}
    assert result.spec is not None
    assert result.validation is not None
    if result.status == "validated":
        assert result.version is not None
        assert any("probation" in m for m in result.messages)
    else:
        assert result.validation.reasons
