"""Stage-A1 unit tests: reproducibility metadata + deterministic replay.

Pins the roadmap acceptance criteria: every repro dict is JSON-safe and
complete, hashes are stable, and re-running a stored repro reproduces the
same numbers (the engine has no RNG).
"""

from __future__ import annotations

import contextlib
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.agent import experiments as exp
from forex_ai_agent.agent.repro import (
    build_repro,
    cost_models_dict,
    feed_identity,
    gate_thresholds_dict,
    replay_spec_run,
    rules_hash,
)
from forex_ai_agent.autonomy.validator import BacktestValidator, GateThresholds
from forex_ai_agent.backtest.engine import BacktestConfig

_QUIET = io.StringIO()

RULES = {
    "name": "Trend Rider",
    "direction": "long",
    "timeframe": "H1",
    "entry_conditions": [
        {"left": "sma", "verb": "above", "right": "sma_slow"},
        {"left": "rsi", "verb": "above", "right": 55},
    ],
}


def _bars(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = np.full(n, 0.00010) + rng.normal(0, 0.0005, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0005, n))
    low = close - np.abs(rng.normal(0, 0.0005, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# rules_hash
# ---------------------------------------------------------------------------


def test_rules_hash_stable_across_key_order():
    a = rules_hash({"a": 1, "b": [1, 2]})
    b = rules_hash({"b": [1, 2], "a": 1})
    assert a == b
    assert len(a) == 64


def test_rules_hash_string_vs_equivalent_dict():
    assert rules_hash('{"a": 1}') == rules_hash({"a": 1})


def test_rules_hash_none_is_empty():
    assert rules_hash(None) == ""


# ---------------------------------------------------------------------------
# build_repro / gate_thresholds_dict / cost_models_dict
# ---------------------------------------------------------------------------


def test_build_repro_is_json_safe_and_complete():
    cfg = BacktestConfig(
        start_date=pd.Timestamp("2025-01-01").to_pydatetime(),
        end_date=pd.Timestamp("2025-03-01").to_pydatetime(),
        symbols=["EUR/USD"],
    )
    repro = build_repro(
        strategy_id="abc12345",
        strategy_name="Trend Rider",
        rules=RULES,
        symbol="EUR/USD",
        timeframe="H1",
        bar_count=1200,
        data_start=pd.Timestamp("2025-01-01"),
        data_end=pd.Timestamp("2025-03-01"),
        thresholds=GateThresholds(),
        backtest_config=cfg,
        walk_forward={"window_months": 1, "step_months": 1},
    )
    # Must survive a JSON round-trip unchanged.
    assert json.loads(json.dumps(repro)) == repro
    assert repro["random_seed"] is None
    assert repro["deterministic"] is True
    assert repro["repro_schema_version"] == 1
    assert repro["strategy"]["rules_sha256"] == rules_hash(RULES)
    assert repro["dataset"]["bar_count"] == 1200
    assert repro["gate_thresholds"]["min_trades_full"] == 20
    assert repro["cost_models"]["initial_balance"] == 100000.0
    assert repro["walk_forward"]["window_months"] == 1


def test_gate_thresholds_dict_covers_all_fields():
    d = gate_thresholds_dict(GateThresholds())
    for key in (
        "min_trades_full",
        "min_oos_trades",
        "min_oos_sharpe",
        "max_oos_drawdown",
        "max_overfitting_score",
        "min_win_rate",
        "min_profit_factor",
    ):
        assert key in d


def test_cost_models_dict_snapshots_models():
    from forex_ai_agent.backtest.execution import FixedSpreadModel

    cfg = BacktestConfig(
        start_date=pd.Timestamp("2025-01-01").to_pydatetime(),
        end_date=pd.Timestamp("2025-03-01").to_pydatetime(),
        symbols=["EUR/USD"],
        spread_model=FixedSpreadModel({"EUR/USD": 0.8}),
    )
    d = cost_models_dict(cfg)
    assert d["spread_model"]["type"] == "FixedSpreadModel"
    assert d["spread_model"]["state"]["spreads"]["EUR/USD"] == 0.8
    assert d["commission_model"] is None


def test_feed_identity_never_raises():
    result = feed_identity()
    assert "provider" in result and "chain" in result


# ---------------------------------------------------------------------------
# Gate reports carry repro + overfit_analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_report_carries_repro():
    df = _bars()
    with contextlib.redirect_stdout(_QUIET):
        validator = BacktestValidator()
        report = await validator.validate(RULES, df, symbol="EUR/USD", run_walk_forward=False)
    assert report.repro, "gate report missing repro dict"
    assert report.repro["dataset"]["symbol"] == "EUR/USD"
    assert report.repro["dataset"]["bar_count"] == len(df)
    assert report.repro["strategy"]["rules_sha256"] == rules_hash(RULES)
    assert report.repro["gate_thresholds"]["min_trades_full"] == 20
    assert report.to_dict()["repro"] == report.repro


@pytest.mark.asyncio
async def test_validation_report_wf_carries_overfit_analysis():
    df = _bars(1500)
    with contextlib.redirect_stdout(_QUIET):
        validator = BacktestValidator()
        report = await validator.validate(RULES, df, symbol="EUR/USD", run_walk_forward=True)
    wf = report.walk_forward or {}
    if wf:  # WF only runs when the full backtest has enough trades
        assert "overfit_analysis" in wf
        assert "windows" in wf


# ---------------------------------------------------------------------------
# Deterministic replay
# ---------------------------------------------------------------------------


def _df_to_bars(df: pd.DataFrame) -> list:
    return [
        SimpleNamespace(
            timestamp=t, open=r["open"], high=r["high"], low=r["low"],
            close=r["close"], volume=r["volume"],
        )
        for t, r in df.iterrows()
    ]


@pytest.mark.asyncio
async def test_replay_reproduces_gate_metrics_exactly():
    df = _bars()
    with contextlib.redirect_stdout(_QUIET):
        validator = BacktestValidator()
        report = await validator.validate(RULES, df, symbol="EUR/USD", run_walk_forward=False)
    stored = dict(report.full_backtest)

    replayed = await replay_spec_run(
        RULES,
        "EUR/USD",
        "H1",
        str(df.index.min()),
        str(df.index.max()),
        candles_fetcher=lambda symbol, tf, start, end: _df_to_bars(df),
    )
    m = replayed["metrics"]
    assert m["total_trades"] == stored["total_trades"]
    assert m["profit_factor"] == pytest.approx(stored["profit_factor"])
    assert m["sharpe_ratio"] == pytest.approx(stored["sharpe_ratio"])
    assert m["win_rate"] == pytest.approx(stored["win_rate"])
    assert m["max_drawdown"] == pytest.approx(stored["max_drawdown_pct"])


@pytest.mark.asyncio
async def test_replay_is_repeatable():
    df = _bars()
    fetcher = lambda symbol, tf, start, end: _df_to_bars(df)  # noqa: E731
    r1 = await replay_spec_run(
        RULES, "EUR/USD", "H1", str(df.index.min()), str(df.index.max()), candles_fetcher=fetcher
    )
    r2 = await replay_spec_run(
        RULES, "EUR/USD", "H1", str(df.index.min()), str(df.index.max()), candles_fetcher=fetcher
    )
    assert r1["metrics"] == r2["metrics"]


@pytest.mark.asyncio
async def test_replay_rejects_invalid_rules():
    with pytest.raises(ValueError, match="not a valid spec"):
        await replay_spec_run(
            {"name": "x", "entry_conditions": [{"left": "nope", "verb": "above", "right": 1}]},
            "EUR/USD",
            "H1",
            "2025-01-01T00:00:00",
            "2025-03-01T00:00:00",
            candles_fetcher=lambda *a: [],
        )


@pytest.mark.asyncio
async def test_replay_walk_forward_uses_extracted_core():
    """WF replay goes through run_walk_forward_on_df with synthetic frames."""
    df = _bars(1500)
    r = await replay_spec_run(
        RULES,
        "EUR/USD",
        "H1",
        str(df.index.min()),
        str(df.index.max()),
        candles_fetcher=lambda symbol, tf, start, end: _df_to_bars(df),
        want_walk_forward=True,
        window_months=1,
        step_months=1,
    )
    wf = r["walk_forward"]
    assert wf["windows_tested"] >= 1
    assert "overfit_analysis" in wf
    assert wf["verdict"] in {"STABLE", "MIXED", "UNSTABLE"}
