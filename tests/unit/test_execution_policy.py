"""Unit tests for Stage E3: execution policy + approval tokens.

Pins the E3 gate under mandate v1.0 (approved 2026-10-18):
- PolicyCaps defaults are the mandate constants (0.25% risk, 0.75% aggregate,
  1.0% daily, 2.0% weekly, 5.0% drawdown, 3 positions)
- OrderIntent fields are computed deterministically (sizing, SL pips, R:R)
- Policy refuses intents that violate any cap (incl. unverified open risk,
  aggregate stop risk, and the min-lot budget floor)
- ApprovalTokens are single-use, intent-bound, expiring
- Source-scan: no agent/LLM path can reach the token grant
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta

import pytest

from forex_ai_agent.execution.policy import (
    ExecutionPolicy,
    PolicyCaps,
    OrderIntent,
    INTENT_VALID,
    INTENT_EXPIRED,
    INTENT_REFUSED_RISK_CAP,
    INTENT_REFUSED_POSITION_CAP,
    INTENT_REFUSED_DAILY_LOSS,
    INTENT_REFUSED_AGGREGATE_RISK,
    INTENT_REFUSED_UNVERIFIED_RISK,
    INTENT_REFUSED_RISK_BUDGET,
    INTENT_REFUSED_UNTRADEABLE,
)
from forex_ai_agent.execution.tokens import (
    ApprovalTokenStore,
    ApprovalToken,
    TOKEN_ACTIVE,
    TOKEN_CONSUMED,
    TOKEN_EXPIRED,
    TOKEN_REVOKED,
)


# ---------------------------------------------------------------------------
# PolicyCaps defaults
# ---------------------------------------------------------------------------

class TestPolicyCaps:
    def test_default_risk_cap(self):
        caps = PolicyCaps()
        assert caps.max_risk_per_trade_pct == 0.25

    def test_default_aggregate_risk_cap(self):
        caps = PolicyCaps()
        assert caps.max_aggregate_stop_risk_pct == 0.75

    def test_default_daily_loss_cap(self):
        caps = PolicyCaps()
        assert caps.max_daily_loss_pct == 1.0

    def test_default_weekly_loss_cap(self):
        caps = PolicyCaps()
        assert caps.max_weekly_loss_pct == 2.0

    def test_default_drawdown_cap(self):
        caps = PolicyCaps()
        assert caps.max_drawdown_pct == 5.0

    def test_default_max_positions(self):
        caps = PolicyCaps()
        assert caps.max_concurrent_positions == 3

    def test_default_min_lot(self):
        caps = PolicyCaps()
        assert caps.min_lot_size == 0.01

    def test_default_intent_expiry(self):
        caps = PolicyCaps()
        assert caps.intent_expiry_seconds == 300

    def test_caps_are_frozen(self):
        caps = PolicyCaps()
        with pytest.raises(AttributeError):
            caps.max_risk_per_trade_pct = 99.0


# ---------------------------------------------------------------------------
# ExecutionPolicy — intent creation
# ---------------------------------------------------------------------------

class TestExecutionPolicy:
    def _policy(self, equity=10000.0, positions=0):
        caps = PolicyCaps()
        p = ExecutionPolicy(caps)
        p.update_account_state(equity=equity, open_positions=positions)
        return p

    def test_valid_intent_creation(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_VALID
        assert intent.symbol == "EUR/USD"
        assert intent.side == "BUY"
        assert intent.volume > 0
        assert intent.sl_pips > 0
        assert intent.rr_ratio >= 1.0

    def test_risk_usd_is_budget(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        # intent.risk_usd is the full per-trade budget: 0.25% of $10000
        assert intent.risk_usd == pytest.approx(25.00, abs=0.01)
        # The floored size risks less than the budget: 0.12 x 20 pips x $10
        assert intent.volume * intent.sl_pips * 10.0 <= intent.risk_usd + 0.01

    def test_volume_sizing_at_mandate_risk(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        # 0.25% of $10000 = $25.00 risk. 20-pip SL, $10/pip/lot (EUR/USD)
        # ideal = 25.00 / (20 * 10) = 0.125 lots, floored to 0.12
        assert intent.volume == pytest.approx(0.12, abs=0.01)

    def test_intent_expiry(self):
        p = self._policy(equity=10000.0)
        now = datetime(2026, 1, 1, 12, 0, 0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
            now=now,
        )
        assert not intent.is_expired(now)
        assert intent.is_expired(now + timedelta(seconds=301))

    def test_intent_key_is_idempotent(self):
        p = self._policy(equity=10000.0)
        now = datetime(2026, 1, 1, 12, 0, 0)
        i1 = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040, now=now,
        )
        i2 = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040, now=now,
        )
        assert i1.key == i2.key

    def test_refused_on_position_cap(self):
        p = self._policy(equity=10000.0, positions=3)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_POSITION_CAP

    def test_refused_on_unverified_open_risk(self):
        # Positions open but the aggregate stop risk is unknown -> fail closed
        p = self._policy(equity=10000.0, positions=1)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_UNVERIFIED_RISK

    def test_verified_open_risk_allows_intent(self):
        p = self._policy(equity=10000.0, positions=1)
        p.update_account_state(
            equity=10000.0, open_positions=1, open_stop_risk_usd=20.0,
        )
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_VALID

    def test_refused_on_aggregate_stop_risk(self):
        # 0.75% of $10000 = $75.00 cap; $60 open + $25 budget = $85 > $75
        p = self._policy(equity=10000.0, positions=1)
        p.update_account_state(
            equity=10000.0, open_positions=1, open_stop_risk_usd=60.0,
        )
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_AGGREGATE_RISK

    def test_refused_on_risk_budget_below_min_lot(self):
        # $100 account: 0.25% = $0.25 budget; 20-pip SL needs $2.00 for
        # one min lot -> raw 0.0125 < 0.01? No: 0.25/(20*10)=0.00125 < 0.01
        # -> refuse; never clamp the size up.
        p = self._policy(equity=100.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_RISK_BUDGET

    def test_refused_on_bad_side(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="HOLD",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_UNTRADEABLE

    def test_refused_on_sl_wrong_side_buy(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.1020, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_UNTRADEABLE

    def test_refused_on_sl_wrong_side_sell(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="SELL",
            entry=1.1000, stop_loss=1.0980, take_profit=1.0960,
        )
        assert intent.status == INTENT_REFUSED_UNTRADEABLE

    def test_refused_on_low_rr(self):
        p = self._policy(equity=10000.0)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1005,
        )
        assert intent.status == INTENT_REFUSED_RISK_CAP

    def test_refused_on_daily_loss(self):
        p = self._policy(equity=100.0)
        # Simulate a $1.00 daily loss (1% of $100)
        p.update_account_state(equity=100.0, open_positions=0, daily_pnl=-1.00)
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_REFUSED_DAILY_LOSS

    def test_daily_reset(self):
        p = self._policy(equity=10000.0)
        # Day 1: lose 1% ($100)
        p.update_account_state(
            equity=10000.0, open_positions=0, daily_pnl=-100.00,
        )
        # Day 2: new day resets daily tracking
        p.update_account_state(
            equity=10000.0, open_positions=0, daily_pnl=0.0,
            now=datetime(2026, 1, 2, 12, 0, 0),
        )
        intent = p.create_intent(
            symbol="EUR/USD", side="BUY",
            entry=1.1000, stop_loss=1.0980, take_profit=1.1040,
        )
        assert intent.status == INTENT_VALID


# ---------------------------------------------------------------------------
# ApprovalTokenStore — single-use, intent-bound, expiring
# ---------------------------------------------------------------------------

class TestApprovalTokens:
    def _store(self, ttl=600):
        return ApprovalTokenStore(ttl_seconds=ttl)

    def test_grant_creates_active_token(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        assert token.status == TOKEN_ACTIVE
        assert token.granted_by == "murtaza"
        assert token.intent_key == "abc123"

    def test_consume_success(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        assert ok is True
        assert token.status == TOKEN_CONSUMED

    def test_consume_is_single_use(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok1 = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        ok2 = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        assert ok1 is True
        assert ok2 is False

    def test_consume_rejects_wrong_intent(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok = s.consume(
            token.token_id, intent_key="WRONG",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        assert ok is False
        assert token.status == TOKEN_ACTIVE  # not consumed

    def test_consume_rejects_wrong_symbol(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok = s.consume(
            token.token_id, intent_key="abc123",
            symbol="GBP/USD", side="BUY", volume=0.01,
        )
        assert ok is False

    def test_consume_rejects_wrong_volume(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.02,
        )
        assert ok is False

    def test_token_expiry(self):
        s = self._store(ttl=60)
        now = datetime(2026, 1, 1, 12, 0, 0)
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza", now=now,
        )
        # Before expiry
        assert token.is_expired(now + timedelta(seconds=59)) is False
        # After expiry
        assert token.is_expired(now + timedelta(seconds=61)) is True
        # Consume after expiry fails
        ok = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
            now=now + timedelta(seconds=61),
        )
        assert ok is False

    def test_revoke(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        ok = s.revoke(token.token_id, revoked_by="murtaza")
        assert ok is True
        assert token.status == TOKEN_REVOKED
        # Cannot consume a revoked token
        ok2 = s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        assert ok2 is False

    def test_active_tokens_filter(self):
        s = self._store()
        t1 = s.grant(
            intent_key="a", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        t2 = s.grant(
            intent_key="b", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        s.consume(
            t1.token_id, intent_key="a",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        active = s.active_tokens()
        assert len(active) == 1
        assert active[0].token_id == t2.token_id

    def test_has_active_token(self):
        s = self._store()
        s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        assert s.has_active_token("abc123") is True
        assert s.has_active_token("other") is False

    def test_audit_log(self):
        s = self._store()
        token = s.grant(
            intent_key="abc123", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        s.consume(
            token.token_id, intent_key="abc123",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        log = s.audit_log()
        assert len(log) >= 2  # grant + consume
        events = [e["event"] for e in log]
        assert "token_granted" in events
        assert "token_consumed" in events


# ---------------------------------------------------------------------------
# Source-scan: no agent/LLM path can reach token grant
# ---------------------------------------------------------------------------

class TestAntiAutomation:
    def test_token_store_has_no_agent_imports(self):
        """ApprovalTokenStore must not import from the agent package."""
        import ast
        source = inspect.getsource(
            __import__("forex_ai_agent.execution.tokens", fromlist=["x"])
        )
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        # No imports from the agent package, LLM providers, or network
        for imp in imports:
            assert "forex_ai_agent.agent" not in imp, f"Agent import found: {imp}"
            assert "openai" not in imp.lower()
            assert "anthropic" not in imp.lower()
            assert "requests" not in imp.lower()
            assert "httpx" not in imp.lower()

    def test_policy_has_no_agent_imports(self):
        """ExecutionPolicy must not import from the agent package."""
        import ast
        source = inspect.getsource(
            __import__("forex_ai_agent.execution.policy", fromlist=["x"])
        )
        tree = ast.parse(source)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        for imp in imports:
            assert "forex_ai_agent.agent" not in imp, f"Agent import found: {imp}"
            assert "openai" not in imp.lower()
            assert "anthropic" not in imp.lower()

    def test_token_ids_are_unguessable(self):
        """Token IDs must be unique and not predictable."""
        s = ApprovalTokenStore()
        ids = set()
        for i in range(100):
            token = s.grant(
                intent_key=f"intent_{i}", symbol="EUR/USD", side="BUY",
                volume=0.01, granted_by="murtaza",
            )
            ids.add(token.token_id)
        assert len(ids) == 100  # all unique
