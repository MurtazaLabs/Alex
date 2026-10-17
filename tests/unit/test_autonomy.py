"""Phase-2 autonomy tests: spec whitelist/compiler, backtest gate, probation
state machine, and the human-only approval wall."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from forex_ai_agent.strategy.spec import (
    StrategySpec,
    SpecCondition,
    fallback_spec,
    parse_spec,
)


# ---------------------------------------------------------------------------
# Spec whitelist + compiler
# ---------------------------------------------------------------------------


class TestSpec:
    def test_valid_spec_parses(self):
        spec, errors = parse_spec(
            {
                "name": "Trend Rider",
                "direction": "long",
                "entry_conditions": [
                    {"left": "sma", "verb": "above", "right": "sma_slow"},
                    {"left": "rsi", "verb": "above", "right": 55},
                ],
            }
        )
        assert spec is not None
        assert errors == []
        assert spec.name == "Trend Rider"

    def test_unknown_indicator_rejected(self):
        spec, errors = parse_spec(
            {
                "name": "Bad",
                "entry_conditions": [{"left": "ichimoku", "verb": "above", "right": 1}],
            }
        )
        assert spec is None
        assert any("unknown indicator" in e for e in errors)

    def test_unknown_verb_rejected(self):
        spec, errors = parse_spec(
            {
                "name": "Bad",
                "entry_conditions": [{"left": "rsi", "verb": "moons", "right": 50}],
            }
        )
        assert spec is None
        assert any("unknown verb" in e for e in errors)

    def test_trade_numbers_rejected(self):
        # Extra fields are forbidden by the schema → whole proposal invalid.
        spec, errors = parse_spec(
            {
                "name": "Sneaky",
                "entry_conditions": [{"left": "sma", "verb": "above", "right": "ema"}],
                "stop_loss": 1.05,
                "lot_size": 5.0,
            }
        )
        assert spec is None
        assert errors  # pydantic flags unexpected fields... or ignores them

    def test_out_of_band_threshold_rejected(self):
        spec, errors = parse_spec(
            {
                "name": "BadRSI",
                "entry_conditions": [{"left": "rsi", "verb": "above", "right": 250}],
            }
        )
        assert spec is None
        assert any("out of range" in e for e in errors)

    def test_stateful_verb_rejected_in_live_compile(self):
        spec = StrategySpec(
            name="Cross",
            entry_conditions=[
                SpecCondition(left="ema", verb="crosses_above", right="sma")
            ],
        )
        with pytest.raises(ValueError, match="stateful"):
            spec.compile()

    def test_fallback_spec_is_valid_and_compiles(self):
        raw = fallback_spec("EUR/USD")
        spec, errors = parse_spec(raw.model_dump())
        assert spec is not None and errors == []
        rules = spec.compile()
        assert len(rules) == 1
        assert rules[0].required is True


# ---------------------------------------------------------------------------
# Backtest gate
# ---------------------------------------------------------------------------


def _synthetic_history(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Deterministic price path with trend legs (gives trend rules a chance)."""
    rng = np.random.default_rng(seed)
    drift = np.concatenate(
        [
            np.full(n // 3, 0.00008),
            np.full(n // 3, -0.00006),
            np.full(n - 2 * (n // 3), 0.00012),
        ]
    )
    steps = drift + rng.normal(0, 0.0009, n)
    close = 1.10 + np.cumsum(steps)
    high = close + np.abs(rng.normal(0, 0.0006, n))
    low = close - np.abs(rng.normal(0, 0.0006, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )


class TestValidator:
    @pytest.mark.asyncio
    async def test_too_short_history_rejected(self):
        from forex_ai_agent.autonomy.validator import BacktestValidator

        v = BacktestValidator()
        report = await v.validate_spec(
            fallback_spec("EUR/USD"), _synthetic_history(100)
        )
        assert not report.passed
        assert any("insufficient history" in r for r in report.reasons)

    @pytest.mark.asyncio
    async def test_invalid_spec_short_circuits(self):
        from forex_ai_agent.autonomy.validator import BacktestValidator

        v = BacktestValidator()
        report = await v.validate({"name": "x", "entry_conditions": []}, _synthetic_history())
        assert not report.passed
        assert any("spec invalid" in r for r in report.reasons)

    @pytest.mark.asyncio
    async def test_fallback_spec_full_gate_runs(self):
        """The fallback spec runs the whole gate end to end on synthetic data.

        Pass/fail depends on the data; the test asserts the machinery works
        and every rejection carries explicit reasons.
        """
        from forex_ai_agent.autonomy.validator import BacktestValidator

        v = BacktestValidator()
        report = await v.validate_spec(
            fallback_spec("EUR/USD"), _synthetic_history(), run_walk_forward=True
        )
        assert report.full_backtest, "expected full-backtest metrics"
        if report.walk_forward:
            assert "oos_sharpe" in report.walk_forward
            assert "overfitting_score" in report.walk_forward
        assert report.passed or report.reasons  # one of the two, always

    def test_generator_stateful_crosses(self):
        from forex_ai_agent.autonomy.validator import SpecSignalGenerator

        spec = StrategySpec(
            name="Cross",
            direction="long",
            entry_conditions=[
                SpecCondition(left="ema", verb="crosses_above", right="sma")
            ],
        )
        gen = SpecSignalGenerator(spec)
        ind = {"ema_20": 1.105, "sma_20": 1.104, "sma_50": 1.10, "rsi_14": 55, "atr_14": 0.001}
        bar = type("B", (), {"close": 1.1055})()

        class Bar:
            close = 1.1055

        # First bar: no previous state → no crosses.
        sig = gen.generate_signals(
            datetime.utcnow(), {"EUR/USD": Bar()}, {"EUR/USD": ind}
        )
        assert sig == []
        # Second bar with a real cross → one long signal with ATR brackets.
        gen._prev["EUR/USD"] = {"ema": 1.103, "sma": 1.104, "sma_slow": 1.10, "rsi": 50, "atr": 0.001, "price": 1.1035}
        sig = gen.generate_signals(
            datetime.utcnow(), {"EUR/USD": Bar()}, {"EUR/USD": ind}
        )
        assert len(sig) == 1
        s = sig[0]
        assert s["direction"] == "LONG"
        assert s["sl"] < s["entry_price"] < s["tp"]


# ---------------------------------------------------------------------------
# Probation state machine (in-memory SQLite)
# ---------------------------------------------------------------------------


@pytest.fixture
async def store():
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool

    from forex_ai_agent.data.models.base import Base as ModelsBase
    from forex_ai_agent.autonomy.probation import ProbationStore

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(ModelsBase.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield ProbationStore(session_factory=factory)
    await engine.dispose()


class TestProbation:
    async def test_start_and_progress(self, store):
        st = await store.start("v-1", "Trend", "EUR/USD", {})
        assert st.status == "active"
        # 20 winners → quota met but window not over → still active.
        metrics = {"win_rate": 0.62, "profit_factor": 1.9, "worst_daily_loss_pct": 0.01}
        st = await store.update(
            "v-1",
            closed_trades=[{"profit": 10.0}] * 20,
            metrics=metrics,
        )
        assert st.status == "active"
        assert st.observed_trades == 20
        assert not st.eligible_for_approval

    async def test_fail_on_bad_metrics_after_window(self, store):
        await store.start("v-2", "Bad", "EUR/USD", {}, required_days=0)
        st = await store.update(
            "v-2",
            closed_trades=[{"profit": -1.0}] * 20,
            metrics={"win_rate": 0.1, "profit_factor": 0.4},
        )
        assert st.status == "failed"
        assert "win_rate" in st.notes

    async def test_pass_then_human_approve_wall(self, store):
        await store.start("v-3", "Good", "GBP/JPY", {}, required_days=0)
        st = await store.update(
            "v-3",
            closed_trades=[{"profit": 5.0}] * 25,
            metrics={"win_rate": 0.6, "profit_factor": 2.0, "worst_daily_loss_pct": 0.01},
        )
        assert st.status == "passed", st.notes
        assert st.eligible_for_approval
        approved = await store.approve("v-3", approver="user")
        assert approved.status == "approved"

    async def test_approve_blocked_unless_passed(self, store):
        await store.start("v-4", "Early", "EUR/USD", {})
        with pytest.raises(ValueError, match="cannot approve"):
            await store.approve("v-4")

    async def test_reject_works_from_active(self, store):
        await store.start("v-6", "Nope", "EUR/USD", {})
        st = await store.reject("v-6", approver="user")
        assert st.status == "rejected"

    async def test_expired_when_window_over_without_quota(self, store):
        await store.start("v-5", "Sleepy", "EUR/USD", {}, required_days=0, required_trades=50)
        st = await store.update("v-5", closed_trades=[{"profit": 1.0}] * 3)
        assert st.status == "expired"


def test_adaptive_probation_quota_scales_with_frequency():
    """Quota must be reachable for slow strategies, strict for fast ones."""
    from datetime import datetime

    from forex_ai_agent.autonomy.proposal import _adaptive_probation_quota
    from forex_ai_agent.autonomy.validator import ValidationReport

    def report(total_trades):
        return ValidationReport(
            spec_name="X",
            passed=True,
            full_backtest={"total_trades": total_trades},
            data_range={
                "start": "2025-09-19 05:00:00",
                "end": "2026-09-18 07:00:00",
            },
        )

    slow = _adaptive_probation_quota(report(33))
    assert slow[0] >= 5                       # statistical floor
    # Reachability: expected trades within the window must cover the quota.
    rate = 33 / 365
    assert rate * slow[1] >= slow[0]

    fast = _adaptive_probation_quota(report(600))
    assert fast == (20, 14)                   # fast systems keep the classic bar

    broken = _adaptive_probation_quota(ValidationReport(spec_name="X", passed=False))
    assert broken == (20, 14)                 # fallback never crashes


# ---------------------------------------------------------------------------
# Bollinger Band vocabulary + reference templates (Phase 3)
# ---------------------------------------------------------------------------


class TestBBVocabulary:
    def test_bb_conditions_parse(self):
        spec, errors = parse_spec(
            {
                "name": "BB Touch",
                "direction": "long",
                "entry_conditions": [
                    {"left": "price", "verb": "below", "right": "bb_lower"},
                    {"left": "rsi", "verb": "below", "right": 30},
                ],
            }
        )
        assert spec is not None
        assert errors == []

    def test_bb_aliases_normalized(self):
        spec, errors = parse_spec(
            {
                "name": "BB Aliases",
                "entry_conditions": [
                    {"left": "lower_band", "verb": "above", "right": "bollinger_upper"},
                ],
            }
        )
        assert spec is not None
        assert errors == []
        assert spec.entry_conditions[0].left == "bb_lower"  # left normalizes

    def test_bb_numeric_threshold_range_checked(self):
        spec, errors = parse_spec(
            {
                "name": "BB Numeric",
                "entry_conditions": [
                    {"left": "bb_upper", "verb": "above", "right": -5},
                ],
            }
        )
        assert spec is None
        assert any("out of range" in e for e in errors)

    def test_templates_roundtrip_and_compile(self):
        from forex_ai_agent.analysis.signal_rules.base_rule import (
            IndicatorsSnapshot,
            MarketContext,
        )
        from forex_ai_agent.strategy.templates import mean_reversion_specs

        specs = mean_reversion_specs("EUR/USD")
        assert [s.direction for s in specs] == ["long", "short"]
        assert specs[0].name == "MeanRevBB Long EUR/USD"
        for s in specs:
            assert s.timeframe == "H1"
            assert (s.atr_sl_mult, s.atr_tp_mult) == (1.5, 2.0)  # confirmed
            parsed, errors = parse_spec(s.model_dump())  # map through whitelist
            assert parsed is not None and errors == []
            rules = parsed.compile()
            assert len(rules) == 1 and rules[0].required is True

        long_rule = parse_spec(specs[0].model_dump())[0].compile()[0]

        def ctx(price, rsi, bb_lower, bb_upper):
            return MarketContext(
                pair="EUR/USD",
                timeframe="H1",
                current_price=price,
                spread_pips=0.8,
                atr_pips=10.0,
                session="LONDON",
                news_state="CLEAR",
                indicators=IndicatorsSnapshot(
                    rsi=rsi, atr=0.0010,
                    bb_upper=bb_upper, bb_middle=(bb_upper + bb_lower) / 2,
                    bb_lower=bb_lower,
                ),
            )

        assert long_rule.condition(ctx(1.0900, 25.0, 1.0950, 1.1050)) is True
        # Price back inside the band → no setup.
        assert long_rule.condition(ctx(1.1000, 25.0, 1.0950, 1.1050)) is False

    def test_generator_bb_signal(self):
        from forex_ai_agent.autonomy.validator import SpecSignalGenerator
        from forex_ai_agent.strategy.templates import mean_reversion_specs

        long_spec = mean_reversion_specs("EUR/USD")[0]
        gen = SpecSignalGenerator(long_spec)
        ind = {
            "ema_20": 1.10, "sma_20": 1.10, "sma_50": 1.10,
            "rsi_14": 24.0, "atr_14": 0.0010,
            "bb_upper": 1.1050, "bb_middle": 1.1000, "bb_lower": 1.0950,
        }

        class Bar:
            close = 1.0900

        sig = gen.generate_signals(
            datetime.utcnow(), {"EUR/USD": Bar()}, {"EUR/USD": ind}
        )
        assert len(sig) == 1
        s = sig[0]
        assert s["direction"] == "LONG"
        assert s["sl"] < s["entry_price"] < s["tp"]

    def test_engine_snapshot_has_bb(self):
        from forex_ai_agent.backtest.engine import BacktestConfig, run_backtest

        df = _synthetic_history(400)
        captured = []

        class Spy:
            def generate_signals(self, timestamp, data, indicators):
                captured.append(
                    {s: dict(v) for s, v in indicators.items()}
                )
                return []

        config = BacktestConfig(
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            initial_balance=10_000.0,
            symbols=["EUR/USD"],
            timeframe="H1",
            warmup_bars=60,
        )
        run_backtest({"EUR/USD": df}, Spy(), config)
        rows = [v for c in captured for v in c.values() if "bb_upper" in v]
        assert rows, "engine must publish bb_upper/middle/lower"
        assert rows[0]["bb_lower"] < rows[0]["bb_middle"] < rows[0]["bb_upper"]


class TestProposeSpec:
    @pytest.mark.asyncio
    async def test_gate_reject_no_version(self, tmp_path):
        from forex_ai_agent.autonomy import proposal as prop
        from forex_ai_agent.autonomy.validator import ValidationReport
        from forex_ai_agent.strategy.templates import mean_reversion_specs

        class FakeValidator:
            async def validate_spec(self, spec, df, symbol="EUR/USD", **kw):
                return ValidationReport(
                    spec_name=spec.name, passed=False, reasons=["test reject"]
                )

        res = await prop.propose_spec(
            "EUR/USD",
            mean_reversion_specs("EUR/USD")[0],
            _synthetic_history(),
            validator=FakeValidator(),
            version_manager_dir=str(tmp_path / "versions"),
            enter_probation=False,
            log_to_db=False,
        )
        assert res.status == "rejected"
        assert res.version is None
        assert not (tmp_path / "versions").exists()

    @pytest.mark.asyncio
    async def test_pass_versions_and_starts_probation(self, tmp_path, monkeypatch):
        from forex_ai_agent.autonomy import proposal as prop
        from forex_ai_agent.autonomy.validator import ValidationReport
        from forex_ai_agent.strategy.templates import mean_reversion_specs

        class FakeValidator:
            async def validate_spec(self, spec, df, symbol="EUR/USD", **kw):
                return ValidationReport(
                    spec_name=spec.name,
                    passed=True,
                    full_backtest={"total_trades": 100},
                    data_range={
                        "start": "2025-09-19 05:00:00",
                        "end": "2026-09-18 07:00:00",
                    },
                )

        started = {}

        class FakeStore:
            async def start(self, **kw):
                started.update(kw)

                class P:
                    required_trades = kw["required_trades"]
                    required_days = kw["required_days"]

                return P()

        monkeypatch.setattr(
            "forex_ai_agent.autonomy.probation.ProbationStore", FakeStore
        )
        res = await prop.propose_spec(
            "EUR/USD",
            mean_reversion_specs("EUR/USD")[0],
            _synthetic_history(),
            validator=FakeValidator(),
            version_manager_dir=str(tmp_path / "versions"),
            enter_probation=True,
            log_to_db=False,
        )
        assert res.status == "validated"
        assert res.version is not None
        assert started["pair"] == "EUR/USD"
        assert started["version_id"] == str(res.version.id)
        assert started["strategy_name"] == "MeanRevBB Long EUR/USD"
        assert "probation active" in " ".join(res.messages)
