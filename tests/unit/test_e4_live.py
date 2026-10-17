"""Unit tests for Stage E4: live deployment components.

Pins the E4 gate:
- LiveCaps defaults are the hard-coded E4 constants (EUR/USD, 0.01 lot,
  3 positions, 5% risk, 15% daily, 25% weekly, +-30 min news, 300s expiry)
- BROKER_FAULT_TABLE: every MT5 retcode owns a declared verdict; unknown
  codes are CRITICAL (fail closed)
- SessionFilter: London/NY windows only, weekends closed, boundaries exact
- NewsBlackout: +-30 min around high-impact events, inclusive boundaries
- Fill reconciliation: signed slippage, EXACT/WITHIN/DRIFT/VOLUME_MISMATCH
- PositionTracker: broker-truth book, per-day realized P&L, equity sync
- LiveTransmitter: the full gate chain — every refusal path and the
  single-wire-attempt guarantee; CRITICAL retcodes trip the kill switch
- DailyReport: deterministic accounting + cap verdicts
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from forex_ai_agent.brokers.base import OrderAck, OrderRequest
from forex_ai_agent.execution.governor import ExecutionGovernor
from forex_ai_agent.execution.live import (
    ALEX_LIVE_MAGIC,
    BROKER_FAULT_TABLE,
    DEFAULT_LIVE_CAPS,
    E4_PAIR,
    FILL_DRIFT,
    FILL_EXACT,
    FILL_VOLUME_MISMATCH,
    FILL_WITHIN_TOLERANCE,
    SESSION_LONDON,
    SESSION_LONDON_NY,
    SESSION_NEW_YORK,
    TRANSMIT_CRITICAL,
    TRANSMIT_FILLED,
    TRANSMIT_PARTIAL,
    TRANSMIT_PLACED,
    TRANSMIT_REJECTED,
    TRANSMIT_REFUSED_BROKER_GATE,
    TRANSMIT_REFUSED_EXPIRED,
    TRANSMIT_REFUSED_GOVERNOR,
    TRANSMIT_REFUSED_INTENT,
    TRANSMIT_REFUSED_NEWS,
    TRANSMIT_REFUSED_PAIR,
    TRANSMIT_REFUSED_SESSION,
    TRANSMIT_REFUSED_TOKEN,
    TRANSMIT_REFUSED_VOLUME,
    TRANSMIT_RETRY,
    DailyReport,
    LiveCaps,
    LiveTransmitter,
    NewsBlackout,
    NewsEvent,
    PositionTracker,
    SessionFilter,
    TradeRecord,
    build_daily_report,
    classify_retcode,
    format_daily_report_markdown,
    reconcile_fill,
    signed_slippage_pips,
    summarize_reconciliations,
    sync_account_into,
)
from forex_ai_agent.execution.live.retcodes import (
    VERDICT_CRITICAL,
    VERDICT_FILLED,
    VERDICT_PARTIAL,
    VERDICT_PLACED,
    VERDICT_REJECTED,
    VERDICT_RETRY,
    TRADE_RETCODE_CONNECTION,
    TRADE_RETCODE_DONE,
    TRADE_RETCODE_DONE_PARTIAL,
    TRADE_RETCODE_INVALID_STOPS,
    TRADE_RETCODE_INVALID_VOLUME,
    TRADE_RETCODE_NO_MONEY,
    TRADE_RETCODE_PLACED,
    TRADE_RETCODE_REQUOTE,
)
from forex_ai_agent.execution.policy import ExecutionPolicy, INTENT_VALID
from forex_ai_agent.execution.tokens import ApprovalTokenStore

# A Wednesday — every session-window test is weekday-safe.
WEDNESDAY = datetime(2026, 10, 21)
MONDAY_10 = WEDNESDAY.replace(day=19, hour=10, minute=0)
MONDAY_13 = WEDNESDAY.replace(day=19, hour=13, minute=0)
MONDAY_19 = WEDNESDAY.replace(day=19, hour=19, minute=0)
SATURDAY = datetime(2026, 10, 17, 12, 0)
SUNDAY = datetime(2026, 10, 18, 12, 0)


# ---------------------------------------------------------------------------
# LiveCaps
# ---------------------------------------------------------------------------

class TestLiveCaps:
    def test_pair_is_eurusd(self):
        assert DEFAULT_LIVE_CAPS.pair == E4_PAIR == "EUR/USD"

    def test_min_lot_only(self):
        assert DEFAULT_LIVE_CAPS.min_lot == 0.01
        assert DEFAULT_LIVE_CAPS.max_lot == 0.01

    def test_position_cap(self):
        assert DEFAULT_LIVE_CAPS.max_concurrent_positions == 3

    def test_risk_caps_match_policy(self):
        from forex_ai_agent.execution.policy import PolicyCaps
        assert DEFAULT_LIVE_CAPS.max_risk_per_trade_pct == PolicyCaps().max_risk_per_trade_pct
        assert DEFAULT_LIVE_CAPS.max_daily_loss_pct == PolicyCaps().max_daily_loss_pct

    def test_weekly_cap_proposed_value(self):
        assert DEFAULT_LIVE_CAPS.max_weekly_loss_pct == 25.0

    def test_news_window(self):
        assert DEFAULT_LIVE_CAPS.news_blackout_minutes == 30

    def test_order_expiry(self):
        assert DEFAULT_LIVE_CAPS.order_expiry_seconds == 300

    def test_live_magic_differs_from_shadow(self):
        from forex_ai_agent.brokers.base import ALEX_SHADOW_MAGIC
        assert ALEX_LIVE_MAGIC != ALEX_SHADOW_MAGIC

    def test_to_dict_round_trip(self):
        data = DEFAULT_LIVE_CAPS.to_dict()
        assert data["pair"] == "EUR/USD"
        assert data["london_utc"] == [7, 16]
        assert data["new_york_utc"] == [12, 21]

    def test_caps_are_frozen(self):
        with pytest.raises(Exception):
            DEFAULT_LIVE_CAPS.pair = "GBP/USD"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fault table
# ---------------------------------------------------------------------------

class TestFaultTable:
    def test_done_is_filled(self):
        assert classify_retcode(TRADE_RETCODE_DONE).verdict == VERDICT_FILLED

    def test_placed_is_placed(self):
        assert classify_retcode(TRADE_RETCODE_PLACED).verdict == VERDICT_PLACED

    def test_partial_is_partial(self):
        assert classify_retcode(TRADE_RETCODE_DONE_PARTIAL).verdict == VERDICT_PARTIAL

    def test_requote_is_retry(self):
        action = classify_retcode(TRADE_RETCODE_REQUOTE)
        assert action.verdict == VERDICT_RETRY
        assert not action.trips_kill_switch

    def test_invalid_stops_is_rejected(self):
        assert classify_retcode(TRADE_RETCODE_INVALID_STOPS).verdict == VERDICT_REJECTED

    def test_invalid_volume_is_rejected(self):
        assert classify_retcode(TRADE_RETCODE_INVALID_VOLUME).verdict == VERDICT_REJECTED

    def test_no_money_is_critical(self):
        action = classify_retcode(TRADE_RETCODE_NO_MONEY)
        assert action.verdict == VERDICT_CRITICAL
        assert action.trips_kill_switch

    def test_connection_is_critical(self):
        assert classify_retcode(TRADE_RETCODE_CONNECTION).trips_kill_switch

    def test_unknown_code_is_critical(self):
        assert classify_retcode(99999).verdict == VERDICT_CRITICAL

    def test_none_code_is_critical(self):
        assert classify_retcode(None).verdict == VERDICT_CRITICAL

    def test_table_verdicts_are_declared(self):
        allowed = {
            VERDICT_FILLED, VERDICT_PLACED, VERDICT_PARTIAL,
            VERDICT_RETRY, VERDICT_REJECTED, VERDICT_CRITICAL,
        }
        for code, action in BROKER_FAULT_TABLE.items():
            assert action.verdict in allowed, f"{code} has undeclared verdict"
            assert action.retcode == code
            assert action.name and action.detail

    def test_only_critical_trips_kill_switch(self):
        for action in BROKER_FAULT_TABLE.values():
            assert action.trips_kill_switch == (action.verdict == VERDICT_CRITICAL)


# ---------------------------------------------------------------------------
# SessionFilter
# ---------------------------------------------------------------------------

class TestSessionFilter:
    def setup_method(self):
        self.filter = SessionFilter()

    def test_london_morning(self):
        check = self.filter.check(MONDAY_10)
        assert check.open
        assert check.session == SESSION_LONDON

    def test_london_ny_overlap(self):
        check = self.filter.check(MONDAY_13)
        assert check.open
        assert check.session == SESSION_LONDON_NY

    def test_ny_evening(self):
        check = self.filter.check(MONDAY_19)
        assert check.open
        assert check.session == SESSION_NEW_YORK

    def test_before_london_closed(self):
        check = self.filter.check(MONDAY_10.replace(hour=6, minute=59))
        assert not check.open
        assert check.reason == "outside_session_hours"

    def test_at_2100_closed(self):
        assert not self.filter.check(MONDAY_10.replace(hour=21, minute=0)).open

    def test_at_0700_open(self):
        assert self.filter.check(MONDAY_10.replace(hour=7, minute=0)).open

    def test_saturday_closed(self):
        check = self.filter.check(SATURDAY)
        assert not check.open
        assert check.reason == "weekend_closed"

    def test_sunday_closed(self):
        assert not self.filter.check(SUNDAY).open

    def test_session_name_matches_check(self):
        assert self.filter.session_name(MONDAY_13) == SESSION_LONDON_NY

    def test_custom_windows(self):
        narrow = SessionFilter(london=(9, 10), new_york=(20, 21))
        assert not narrow.check(MONDAY_10).open
        assert narrow.check(MONDAY_10.replace(hour=9, minute=30)).open


# ---------------------------------------------------------------------------
# NewsBlackout
# ---------------------------------------------------------------------------

class TestNewsBlackout:
    def test_inside_before_window_blocked(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now + timedelta(minutes=10), "CPI")])
        check = blk.check(now)
        assert check.blocked
        assert check.event.title == "CPI"
        assert check.reason == "news_blackout"

    def test_inside_after_window_blocked(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now - timedelta(minutes=20), "NFP")])
        assert blk.check(now).blocked

    def test_exactly_30_minutes_blocked(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now + timedelta(minutes=30), "FED")])
        assert blk.check(now).blocked

    def test_31_minutes_clear(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now + timedelta(minutes=31), "FED")])
        assert not blk.check(now).blocked

    def test_medium_impact_ignored_by_default(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now + timedelta(minutes=5), "Retail Sales", "medium")])
        assert not blk.check(now).blocked

    def test_watched_impacts_configurable(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout(
            [NewsEvent(now + timedelta(minutes=5), "Retail Sales", "medium")],
            impacts=("medium",),
        )
        assert blk.check(now).blocked

    def test_empty_calendar_clear(self):
        assert not NewsBlackout().check(datetime(2026, 10, 21)).blocked

    def test_refresh_replaces_calendar(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now, "A")])
        blk.refresh([NewsEvent(now + timedelta(days=1), "B")])
        assert not blk.check(now).blocked
        assert len(blk.events) == 1

    def test_minutes_to_event_reported(self):
        now = datetime(2026, 10, 21, 12, 0)
        blk = NewsBlackout([NewsEvent(now + timedelta(minutes=15), "CPI")])
        assert blk.check(now).minutes_to_event == 15.0


# ---------------------------------------------------------------------------
# Fill reconciliation
# ---------------------------------------------------------------------------

class TestFillReconciliation:
    def test_buy_slippage_sign_positive_when_worse(self):
        slip = signed_slippage_pips("BUY", 1.1000, 1.1002, 0.0001)
        assert slip == pytest.approx(2.0)

    def test_sell_slippage_sign_positive_when_worse(self):
        slip = signed_slippage_pips("SELL", 1.1000, 1.0998, 0.0001)
        assert slip == pytest.approx(2.0)

    def test_sell_fill_above_expected_is_negative(self):
        slip = signed_slippage_pips("SELL", 1.1000, 1.1002, 0.0001)
        assert slip == pytest.approx(-2.0)

    def test_bad_pip_size_raises(self):
        with pytest.raises(ValueError):
            signed_slippage_pips("BUY", 1.1, 1.1, 0.0)

    def test_bad_side_raises(self):
        with pytest.raises(ValueError):
            signed_slippage_pips("HOLD", 1.1, 1.1, 0.0001)

    def test_exact_fill(self):
        recon = reconcile_fill(
            intent_key="k1", symbol="EUR/USD", side="BUY",
            expected_price=1.1000, expected_volume=0.01,
            actual_price=1.1000, actual_volume=0.01,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        assert recon.verdict == FILL_EXACT
        assert recon.is_exact
        assert recon.within_tolerance

    def test_within_tolerance(self):
        recon = reconcile_fill(
            intent_key="k1", symbol="EUR/USD", side="BUY",
            expected_price=1.1000, expected_volume=0.01,
            actual_price=1.10015, actual_volume=0.01,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        assert recon.verdict == FILL_WITHIN_TOLERANCE
        assert recon.within_tolerance
        assert not recon.is_drift

    def test_drift_beyond_tolerance(self):
        recon = reconcile_fill(
            intent_key="k1", symbol="EUR/USD", side="BUY",
            expected_price=1.1000, expected_volume=0.01,
            actual_price=1.1005, actual_volume=0.01,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        assert recon.verdict == FILL_DRIFT
        assert recon.is_drift
        assert recon.slippage_pips == pytest.approx(5.0)

    def test_volume_mismatch(self):
        recon = reconcile_fill(
            intent_key="k1", symbol="EUR/USD", side="BUY",
            expected_price=1.1000, expected_volume=0.01,
            actual_price=1.1000, actual_volume=0.02,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        assert recon.verdict == FILL_VOLUME_MISMATCH

    def test_summary_empty(self):
        summary = summarize_reconciliations([])
        assert summary["count"] == 0

    def test_summary_counts(self):
        a = reconcile_fill(
            intent_key="a", symbol="EUR/USD", side="BUY",
            expected_price=1.1, expected_volume=0.01,
            actual_price=1.1, actual_volume=0.01,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        b = reconcile_fill(
            intent_key="b", symbol="EUR/USD", side="BUY",
            expected_price=1.1, expected_volume=0.01,
            actual_price=1.1005, actual_volume=0.01,
            pip_size=0.0001, tolerance_pips=2.0,
        )
        summary = summarize_reconciliations([a, b])
        assert summary["count"] == 2
        assert summary["drift_count"] == 1
        assert summary["max_slippage_pips"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# PositionTracker
# ---------------------------------------------------------------------------

class _Pos:
    def __init__(self, ticket, side, volume, symbol="EUR/USD", profit=0.0):
        self.ticket = ticket
        self.side = side
        self.volume = volume
        self.symbol = symbol
        self.profit = profit


class _Account:
    def __init__(self, equity, balance):
        self.equity = equity
        self.balance = balance


class TestPositionTracker:
    def test_empty_book(self):
        tracker = PositionTracker(symbol="EUR/USD")
        book = tracker.sync_positions([], now=MONDAY_10)
        assert book.count == 0
        assert tracker.open_count() == 0

    def test_book_counts(self):
        tracker = PositionTracker(symbol="EUR/USD")
        book = tracker.sync_positions(
            [_Pos(1, "BUY", 0.01, profit=1.5), _Pos(2, "SELL", 0.02, profit=-0.5)],
            now=MONDAY_10,
        )
        assert book.count == 2
        assert book.buy_count == 1
        assert book.sell_count == 1
        assert book.net_volume == pytest.approx(-0.01)
        assert book.floating_pnl == pytest.approx(1.0)
        assert tracker.open_count_on_pair("EUR/USD") == 2

    def test_dict_positions_supported(self):
        tracker = PositionTracker()
        book = tracker.sync_positions([{"ticket": 9, "side": "LONG", "volume": 0.01}])
        assert book.count == 1
        assert book.buy_count == 1

    def test_realized_pnl_per_day(self):
        tracker = PositionTracker()
        tracker.record_deal(profit=5.0, ts=MONDAY_10, ticket=1)
        tracker.record_deal(profit=-2.0, ts=MONDAY_10, ticket=2)
        tracker.record_deal(profit=10.0, ts=MONDAY_10 + timedelta(days=1), ticket=3)
        assert tracker.realized_pnl_today(MONDAY_10) == pytest.approx(3.0)
        assert tracker.realized_pnl("2026-10-20") == pytest.approx(10.0)

    def test_sync_account_computes_limit(self):
        tracker = PositionTracker(symbol="EUR/USD")
        tracker.sync_positions([_Pos(1, "BUY", 0.01)], now=MONDAY_10)
        feed = tracker.sync_account(
            _Account(equity=100.0, balance=100.0),
            daily_loss_pct=15.0,
            now=MONDAY_10,
        )
        assert feed.equity == 100.0
        assert feed.daily_loss_limit_usd == 15.0
        assert feed.open_positions == 1

    def test_sync_account_into_feeds_policy_and_governor(self):
        policy = ExecutionPolicy()
        governor = ExecutionGovernor()
        governor.arm()
        tracker = PositionTracker(symbol="EUR/USD")

        feed = sync_account_into(
            tracker, _Account(equity=200.0, balance=200.0),
            policy=policy, governor=governor, daily_loss_pct=15.0, now=MONDAY_10,
        )
        assert feed.daily_loss_limit_usd == 30.0
        assert governor.limits.daily_loss_limit == 30.0
        intent = policy.create_intent(
            symbol="EUR/USD", side="BUY", entry=1.10,
            stop_loss=1.0980, take_profit=1.1040, now=MONDAY_10,
        )
        assert intent.status == INTENT_VALID


# ---------------------------------------------------------------------------
# LiveTransmitter
# ---------------------------------------------------------------------------

class _StubBroker:
    """Duck-typed broker: records requests, returns a scripted ack."""

    def __init__(
        self,
        *,
        retcode=TRADE_RETCODE_DONE,
        status="FILLED",
        sent=True,
        ticket=777,
        bad=False,
        permission_level=4,
        shadow_mode=False,
    ):
        self.retcode = retcode
        self.status = status
        self.sent_flag = sent
        self.ticket = ticket
        self.bad = bad
        self.permission_level = permission_level
        self.shadow_mode = shadow_mode
        self.requests = []

    def submit_order(self, request):
        self.requests.append(request)
        if self.bad:
            raise ConnectionError("no terminal")
        return OrderAck(
            sent=self.sent_flag,
            status=self.status,
            retcode=self.retcode,
            ticket=self.ticket,
        )


def _make_intent(now, *, side="BUY", status=INTENT_VALID, volume=0.01,
                 symbol="EUR/USD", expires_in=300):
    from forex_ai_agent.execution.policy import OrderIntent
    return OrderIntent(
        key="intent-abc123", symbol=symbol, side=side, volume=volume,
        sl=1.0980, tp=1.1040, entry=1.1000,
        risk_usd=5.0, sl_pips=20.0, rr_ratio=2.0, equity=100.0,
        created_at=now, expires_at=now + timedelta(seconds=expires_in),
        status=status,
    )


def _grant(store, intent, *, volume=None, now=MONDAY_10):
    return store.grant(
        intent_key=intent.key, symbol=intent.symbol, side=intent.side,
        volume=intent.volume if volume is None else volume,
        granted_by="test-operator", now=now,
    )


def _make_transmitter(broker=None, *, governor=None, blackout=None, now=None):
    broker = broker or _StubBroker()
    governor = governor or ExecutionGovernor()
    store = ApprovalTokenStore()
    transmitter = LiveTransmitter(
        broker, store, governor, blackout=blackout or NewsBlackout(),
    )
    return transmitter, broker, governor, store


class TestTransmitterHappyPath:
    def _armed_governor(self):
        governor = ExecutionGovernor()
        governor.arm()
        return governor

    def test_filled_flow(self):
        governor = self._armed_governor()
        transmitter, broker, _, store = _make_transmitter(governor=governor)
        intent = _make_intent(MONDAY_10)
        token = _grant(store, intent)

        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)

        assert result.status == TRANSMIT_FILLED
        assert result.sent
        assert result.ticket == 777
        assert len(broker.requests) == 1
        wire = broker.requests[0]
        assert wire.magic == ALEX_LIVE_MAGIC
        assert wire.symbol == "EUR/USD"
        assert wire.volume == 0.01

    def test_token_is_single_use_across_calls(self):
        governor = self._armed_governor()
        transmitter, broker, _, store = _make_transmitter(governor=governor)
        intent = _make_intent(MONDAY_10)
        token = _grant(store, intent)

        first = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        second = transmitter.transmit(intent, token.token_id, now=MONDAY_10)

        assert first.status == TRANSMIT_FILLED
        assert second.status == TRANSMIT_REFUSED_TOKEN
        assert len(broker.requests) == 1

    def test_exactly_one_wire_attempt_per_call(self):
        governor = self._armed_governor()
        transmitter, broker, _, store = _make_transmitter(governor=governor)
        for _ in range(3):
            intent = _make_intent(MONDAY_10)
            token = _grant(store, intent)
            transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert len(broker.requests) == 3

    def test_placed_flow(self):
        governor = self._armed_governor()
        broker = _StubBroker(retcode=TRADE_RETCODE_PLACED, status="PLACED")
        transmitter, _, _, store = _make_transmitter(broker=broker, governor=governor)
        intent = _make_intent(MONDAY_10)
        token = _grant(store, intent)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_PLACED
        assert result.is_fill

    def test_partial_flow(self):
        governor = self._armed_governor()
        broker = _StubBroker(retcode=TRADE_RETCODE_DONE_PARTIAL, status="REJECTED")
        transmitter, _, _, store = _make_transmitter(broker=broker, governor=governor)
        intent = _make_intent(MONDAY_10)
        token = _grant(store, intent)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_PARTIAL


class TestTransmitterRefusals:
    def _armed(self):
        g = ExecutionGovernor()
        g.arm()
        return g

    def _intent_and_token(self, store, now=MONDAY_10, **kwargs):
        intent = _make_intent(now, **kwargs)
        token = _grant(store, intent, now=now)
        return intent, token

    def test_refused_intent_status(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        intent, token = self._intent_and_token(store, status="REFUSED_RISK_CAP")
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_INTENT
        assert not broker.requests

    def test_refused_expired(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        intent, token = self._intent_and_token(store, expires_in=-10)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_EXPIRED
        assert not broker.requests

    def test_refused_wrong_pair(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        intent, token = self._intent_and_token(store, symbol="GBP/USD")
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_PAIR
        assert not broker.requests

    def test_refused_volume_out_of_bounds(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        intent, token = self._intent_and_token(store, volume=0.05)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_VOLUME
        assert not broker.requests

    def test_refused_outside_session(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        saturday = SATURDAY.replace(minute=0)
        intent, token = self._intent_and_token(store, now=saturday)
        result = transmitter.transmit(intent, token.token_id, now=saturday)
        assert result.status == TRANSMIT_REFUSED_SESSION
        assert not broker.requests

    def test_refused_news_blackout(self):
        event_soon = MONDAY_10 + timedelta(minutes=10)
        blackout = NewsBlackout([NewsEvent(event_soon, "NFP")])
        transmitter, broker, _, store = _make_transmitter(
            governor=self._armed(), blackout=blackout,
        )
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_NEWS
        assert not broker.requests

    def test_refused_governor_disarmed(self):
        governor = ExecutionGovernor()  # DISARMED
        transmitter, broker, _, store = _make_transmitter(governor=governor)
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_GOVERNOR
        assert not broker.requests

    def test_refused_governor_halted(self):
        governor = self._armed()
        governor.trip("test")
        transmitter, broker, _, store = _make_transmitter(governor=governor)
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_GOVERNOR
        assert not broker.requests

    def test_refused_broker_low_permission(self):
        broker = _StubBroker(permission_level=0)
        transmitter, _, _, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_BROKER_GATE
        assert "level" in result.detail
        assert not broker.requests

    def test_refused_broker_shadow_mode(self):
        broker = _StubBroker(permission_level=4, shadow_mode=True)
        transmitter, _, _, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_BROKER_GATE
        assert "shadow" in result.detail
        assert not broker.requests

    def test_refused_token_unknown(self):
        transmitter, broker, _, _ = _make_transmitter(governor=self._armed())
        intent = _make_intent(MONDAY_10)
        result = transmitter.transmit(intent, "nope", now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_TOKEN
        assert not broker.requests

    def test_refused_token_wrong_volume(self):
        transmitter, broker, _, store = _make_transmitter(governor=self._armed())
        intent = _make_intent(MONDAY_10)
        token = _grant(store, intent, volume=0.02)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REFUSED_TOKEN
        assert not broker.requests

    def test_preflight_refusal_does_not_burn_token(self):
        """A session refusal must leave the token usable for the next window."""
        transmitter, _, _, store = _make_transmitter(governor=self._armed())
        saturday = SATURDAY.replace(minute=0)
        intent, token = self._intent_and_token(store, now=saturday)
        transmitter.transmit(intent, token.token_id, now=saturday)
        assert store.has_active_token(intent.key, now=saturday)

    def test_refusal_records_checks(self):
        transmitter, _, _, store = _make_transmitter(governor=self._armed())
        intent, token = self._intent_and_token(store, expires_in=-10)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        steps = [c["step"] for c in result.checks]
        assert "expiry" in steps
        assert result.checks[-1]["ok"] == "false"


class TestTransmitterFaults:
    def _armed(self):
        g = ExecutionGovernor()
        g.arm()
        return g

    def _intent_and_token(self, store, now=MONDAY_10):
        intent = _make_intent(now)
        token = _grant(store, intent, now=now)
        return intent, token

    def test_requote_is_retry_not_fill(self):
        broker = _StubBroker(retcode=TRADE_RETCODE_REQUOTE, status="REJECTED")
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_RETRY
        assert not result.is_fill
        assert not governor.state.tripped

    def test_invalid_stops_is_rejected_no_retry(self):
        broker = _StubBroker(retcode=TRADE_RETCODE_INVALID_STOPS, status="REJECTED")
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REJECTED
        assert not governor.state.tripped

    def test_no_money_trips_kill_switch(self):
        broker = _StubBroker(retcode=TRADE_RETCODE_NO_MONEY, status="REJECTED")
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_CRITICAL
        assert governor.state.tripped
        assert "NO_MONEY" in (governor.trip_reason or "")

    def test_connection_lost_trips_kill_switch(self):
        broker = _StubBroker(retcode=TRADE_RETCODE_CONNECTION, status="FAILED")
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_CRITICAL
        assert governor.state.tripped

    def test_unknown_retcode_trips_kill_switch(self):
        broker = _StubBroker(retcode=424242, status="???")
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_CRITICAL
        assert governor.state.tripped

    def test_broker_exception_trips_kill_switch(self):
        broker = _StubBroker(bad=True)
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_CRITICAL
        assert governor.state.tripped

    def test_adapter_pre_wire_refusal_is_rejected(self):
        broker = _StubBroker(sent=False, status="NO_TICK", retcode=None)
        transmitter, _, governor, store = _make_transmitter(broker=broker, governor=self._armed())
        intent, token = self._intent_and_token(store)
        result = transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        assert result.status == TRANSMIT_REJECTED
        assert not result.sent
        assert not governor.state.tripped

    def test_fill_recorded_in_journal(self):
        governor = self._armed()
        journal_record = governor.accept_intent(
            "intent-abc123", symbol="EUR/USD", side="BUY", volume=0.01,
            sl=1.0980, tp=1.1040, now=MONDAY_10,
        )
        assert journal_record is not None
        transmitter, _, _, store = _make_transmitter(governor=governor)
        intent, token = self._intent_and_token(store)
        transmitter.transmit(intent, token.token_id, now=MONDAY_10)
        executed = governor.journal.get("intent-abc123")
        assert executed["state"] == "EXECUTED"


# ---------------------------------------------------------------------------
# DailyReport
# ---------------------------------------------------------------------------

class TestDailyReport:
    def _trade(self, profit, ticket=1):
        return TradeRecord(
            ticket=ticket, symbol="EUR/USD", side="BUY", volume=0.01,
            profit=profit, commission=-0.07, swap=0.0,
        )

    def test_empty_day(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[],
            starting_equity=100.0, ending_equity=100.0,
        )
        assert report.trade_count == 0
        assert report.net_pnl == 0.0
        assert report.win_rate == 0.0
        assert report.verdict == "OK"

    def test_win_rate_and_profit_factor(self):
        trades = [self._trade(2.0, 1), self._trade(-1.0, 2), self._trade(3.0, 3)]
        report = build_daily_report(
            day=MONDAY_10.date(), trades=trades,
            starting_equity=100.0, ending_equity=104.0,
        )
        assert report.trade_count == 3
        assert report.wins == 2
        assert report.losses == 1
        assert report.win_rate == pytest.approx(66.7, abs=0.1)
        assert report.profit_factor is not None

    def test_net_pnl_includes_costs(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[self._trade(1.0)],
            starting_equity=100.0, ending_equity=100.93,
        )
        assert report.net_pnl == pytest.approx(0.93)

    def test_daily_breach_verdict(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[],
            starting_equity=100.0, ending_equity=84.0,
        )
        assert report.verdict == "DAILY_LOSS_BREACH"

    def test_weekly_breach_verdict(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[],
            starting_equity=100.0, ending_equity=99.0,
            weekly_pnl=-26.0, weekly_start_equity=100.0,
        )
        assert report.verdict == "WEEKLY_LOSS_BREACH"

    def test_max_drawdown_from_curve(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[],
            starting_equity=100.0, ending_equity=98.0,
            equity_points=[100.0, 104.0, 99.0, 98.0],
        )
        assert report.max_drawdown == pytest.approx(6.0)

    def test_to_dict_serializable(self):
        import json
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[self._trade(1.0)],
            starting_equity=100.0, ending_equity=100.93,
        )
        data = report.to_dict()
        json.dumps(data)  # must not raise
        assert data["trade_count"] == 1

    def test_markdown_ascii_only(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[self._trade(1.0)],
            starting_equity=100.0, ending_equity=100.93,
        )
        text = format_daily_report_markdown(report)
        assert "E4 Daily Report" in text
        assert text.isascii()

    def test_verdict_boundary_exact_cap_is_breach(self):
        report = build_daily_report(
            day=MONDAY_10.date(), trades=[],
            starting_equity=100.0, ending_equity=85.0,
        )
        assert report.verdict == "DAILY_LOSS_BREACH"


# ---------------------------------------------------------------------------
# OrderExpiry
# ---------------------------------------------------------------------------

class TestOrderExpiry:
    def _order(self, ticket=1, placed_at=None):
        from forex_ai_agent.execution.live import TrackedOrder
        return TrackedOrder(
            ticket=ticket, symbol="EUR/USD",
            placed_at=placed_at or MONDAY_10,
            side="BUY", volume=0.01, key="k1",
        )

    def test_default_ttl_is_300(self):
        from forex_ai_agent.execution.live import OrderExpiry
        assert OrderExpiry().expiry_seconds == 300

    def test_rejects_bad_ttl(self):
        from forex_ai_agent.execution.live import OrderExpiry
        with pytest.raises(ValueError):
            OrderExpiry(0)

    def test_not_due_before_ttl(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(300)
        expiry.track(self._order())
        assert expiry.due(MONDAY_10 + timedelta(seconds=299)) == []

    def test_due_at_ttl(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(300)
        expiry.track(self._order())
        due = expiry.due(MONDAY_10 + timedelta(seconds=300))
        assert len(due) == 1
        assert due[0].order.ticket == 1
        assert due[0].age_seconds == pytest.approx(300.0)

    def test_due_sorted_oldest_first(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(60)
        expiry.track(self._order(1, MONDAY_10))
        expiry.track(self._order(2, MONDAY_10 + timedelta(seconds=10)))
        due = expiry.due(MONDAY_10 + timedelta(seconds=120))
        assert [e.order.ticket for e in due] == [1, 2]

    def test_resolve_stops_tracking(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(60)
        expiry.track(self._order())
        assert expiry.resolve(1).ticket == 1
        assert expiry.due(MONDAY_10 + timedelta(seconds=600)) == []
        assert len(expiry) == 0

    def test_resolve_unknown_is_none(self):
        from forex_ai_agent.execution.live import OrderExpiry
        assert OrderExpiry().resolve(99) is None

    def test_track_is_idempotent_by_ticket(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(60)
        expiry.track(self._order())
        expiry.track(self._order(placed_at=MONDAY_10 + timedelta(seconds=30)))
        assert len(expiry) == 1
        assert expiry.due(MONDAY_10 + timedelta(seconds=70)) == []

    def test_age_seconds(self):
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(300)
        expiry.track(self._order())
        assert expiry.age_seconds(1, MONDAY_10 + timedelta(seconds=45)) == pytest.approx(45.0)
        assert expiry.age_seconds(99) is None

    def test_expired_order_serializable(self):
        import json
        from forex_ai_agent.execution.live import OrderExpiry
        expiry = OrderExpiry(60)
        expiry.track(self._order())
        due = expiry.due(MONDAY_10 + timedelta(seconds=90))
        json.dumps(due[0].to_dict())


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation:
    def _window(self, **overrides):
        from forex_ai_agent.execution.live import evaluate_window
        kwargs = dict(
            window_start=date(2026, 9, 17),
            window_end=date(2026, 10, 17),
            declared_days=30,
            live_trades=42,
            live_pnl=18.5,
            shadow_predicted_pnl=20.0,
            tolerance_pct=20.0,
            unresolved_violations=0,
            fill_drift_count=0,
            volume_mismatch_count=0,
            orders_sent=42,
            orders_accounted=42,
        )
        kwargs.update(overrides)
        return evaluate_window(**kwargs)

    def test_all_pass_is_eligible(self):
        from forex_ai_agent.execution.live import EVAL_ELIGIBLE
        report = self._window()
        assert report.verdict == EVAL_ELIGIBLE
        assert report.all_passed

    def test_pnl_outside_band_extends(self):
        from forex_ai_agent.execution.live import EVAL_EXTEND_REVIEW
        report = self._window(live_pnl=-5.0)
        assert report.verdict == EVAL_EXTEND_REVIEW
        failed = [c.name for c in report.failed_criteria()]
        assert "pnl_within_tolerance" in failed

    def test_pnl_zero_prediction_requires_zero_live(self):
        from forex_ai_agent.execution.live import EVAL_ELIGIBLE
        report = self._window(shadow_predicted_pnl=0.0, live_pnl=0.0)
        assert report.verdict == EVAL_ELIGIBLE
        report2 = self._window(shadow_predicted_pnl=0.0, live_pnl=1.0)
        assert report2.verdict != EVAL_ELIGIBLE

    def test_unresolved_violation_retires(self):
        from forex_ai_agent.execution.live import EVAL_RETIRE_RECOMMENDED
        report = self._window(unresolved_violations=1)
        assert report.verdict == EVAL_RETIRE_RECOMMENDED

    def test_safety_hard_fail_retires(self):
        from forex_ai_agent.execution.live import EVAL_RETIRE_RECOMMENDED
        report = self._window(safety_hard_fail=True)
        assert report.verdict == EVAL_RETIRE_RECOMMENDED

    def test_fill_drift_fails_reconciliation(self):
        from forex_ai_agent.execution.live import EVAL_EXTEND_REVIEW
        report = self._window(fill_drift_count=2)
        assert report.verdict == EVAL_EXTEND_REVIEW
        failed = [c.name for c in report.failed_criteria()]
        assert "reconciliation_exact" in failed

    def test_short_window_fails_audit(self):
        report = self._window(
            window_start=date(2026, 10, 10), window_end=date(2026, 10, 17),
        )
        failed = [c.name for c in report.failed_criteria()]
        assert "audit_complete" in failed

    def test_unaccounted_orders_fail_audit(self):
        report = self._window(orders_sent=10, orders_accounted=9)
        failed = [c.name for c in report.failed_criteria()]
        assert "audit_complete" in failed

    def test_tolerance_boundary_inclusive(self):
        from forex_ai_agent.execution.live import EVAL_ELIGIBLE
        report = self._window(live_pnl=16.0)  # 20% of 20.0 exactly
        assert report.verdict == EVAL_ELIGIBLE

    def test_report_serializable(self):
        import json
        report = self._window()
        data = report.to_dict()
        json.dumps(data)
        assert data["elapsed_days"] == 30


# ---------------------------------------------------------------------------
# SignoffStore: recorded, time-boxed human sign-off (append-only)
# ---------------------------------------------------------------------------

class TestSignoffs:
    def _signoff(self, **overrides):
        from forex_ai_agent.execution.live import LiveSignoff, SCOPE_DRILL
        now = overrides.pop("granted_at", datetime(2026, 10, 17, 9, 0))
        base = dict(
            approver="murtaza",
            scope=SCOPE_DRILL,
            pair=E4_PAIR,
            strategy_id="mean_reversion_v1",
            declaration="i have given the level 4 permission you can test yourself",
            granted_at=now,
            expires_at=now + timedelta(days=7),
            config={"max_permission_level": 4},
        )
        base.update(overrides)
        return LiveSignoff(**base)

    def test_record_and_reload_round_trip(self, tmp_path):
        from forex_ai_agent.execution.live import SignoffStore
        store = SignoffStore(tmp_path / "signoffs.jsonl")
        signoff = self._signoff()
        store.record(signoff)
        loaded = store.records()
        assert len(loaded) == 1
        assert loaded[0].approver == "murtaza"
        assert loaded[0].granted_at == signoff.granted_at
        assert loaded[0].expires_at == signoff.expires_at
        assert loaded[0].signoff_id == signoff.signoff_id

    def test_append_only_two_records_two_lines(self, tmp_path):
        from forex_ai_agent.execution.live import SignoffStore
        store = SignoffStore(tmp_path / "signoffs.jsonl")
        store.record(self._signoff())
        store.record(self._signoff(granted_at=datetime(2026, 10, 18, 9, 0)))
        lines = [ln for ln in (tmp_path / "signoffs.jsonl").read_text(
            encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2

    def test_latest_returns_newest_matching_scope(self, tmp_path):
        from forex_ai_agent.execution.live import SCOPE_WINDOW, SignoffStore
        store = SignoffStore(tmp_path / "signoffs.jsonl")
        store.record(self._signoff(granted_at=datetime(2026, 10, 17, 9, 0)))
        store.record(self._signoff(
            scope=SCOPE_WINDOW, granted_at=datetime(2026, 10, 17, 10, 0),
        ))
        from forex_ai_agent.execution.live import SCOPE_DRILL
        latest_drill = store.latest(scope=SCOPE_DRILL)
        latest_window = store.latest(scope=SCOPE_WINDOW)
        assert latest_drill.granted_at.hour == 9
        assert latest_window.granted_at.hour == 10

    def test_is_active_fails_closed_at_boundaries(self):
        signoff = self._signoff(
            granted_at=datetime(2026, 10, 17, 9, 0),
            expires_at=datetime(2026, 10, 17, 12, 0),
        )
        assert not signoff.is_active(datetime(2026, 10, 17, 8, 59, 59))
        assert signoff.is_active(datetime(2026, 10, 17, 9, 0))
        assert signoff.is_active(datetime(2026, 10, 17, 11, 59, 59))
        assert not signoff.is_active(datetime(2026, 10, 17, 12, 0))

    def test_active_only_filters_expired(self, tmp_path):
        from forex_ai_agent.execution.live import SCOPE_DRILL, SignoffStore
        store = SignoffStore(tmp_path / "signoffs.jsonl")
        store.record(self._signoff(
            granted_at=datetime(2026, 10, 1, 9, 0),
            expires_at=datetime(2026, 10, 2, 9, 0),
        ))
        now = datetime(2026, 10, 17, 9, 0)
        assert store.latest(scope=SCOPE_DRILL) is not None
        assert store.latest(scope=SCOPE_DRILL, now=now, active_only=True) is None

    def test_invalid_signoffs_are_rejected(self):
        from forex_ai_agent.execution.live import LiveSignoff
        with pytest.raises(ValueError):
            self._signoff(approver="  ")
        with pytest.raises(ValueError):
            self._signoff(declaration="")
        with pytest.raises(ValueError):
            self._signoff(expires_at=datetime(2026, 10, 17, 8, 0))  # before grant

    def test_missing_log_reads_empty(self, tmp_path):
        from forex_ai_agent.execution.live import SignoffStore
        store = SignoffStore(tmp_path / "never_written.jsonl")
        assert store.records() == []
        assert store.latest() is None


# ---------------------------------------------------------------------------
# Source scans (anti-automation pins for the live package)
# ---------------------------------------------------------------------------

class TestLiveSourceScans:
    def test_transmitter_is_only_submit_order_caller(self):
        """submit_order may only be reached from execution/live/transmitter.py."""
        import forex_ai_agent
        from pathlib import Path
        root = Path(forex_ai_agent.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            if "execution/live/transmitter.py" in str(path).replace("\\", "/"):
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ".submit_order(" in text:
                offenders.append(str(path))
        assert offenders == [], f"submit_order called outside transmitter: {offenders}"

    def test_live_package_has_no_llm_imports(self):
        import forex_ai_agent.execution.live as live_pkg
        from pathlib import Path
        banned = ("openai", "anthropic", "litellm", "langchain", "ollama")
        for path in Path(live_pkg.__file__).parent.glob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                assert token not in text, f"{path.name} imports LLM stack ({token})"

    def test_no_network_imports_in_live_modules(self):
        import forex_ai_agent.execution.live as live_pkg
        from pathlib import Path
        banned = ("requests", "httpx", "aiohttp", "urllib.request")
        for path in Path(live_pkg.__file__).parent.glob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for token in banned:
                assert token not in text, f"{path.name} contains network import ({token})"

    def test_non_shadow_adapter_never_constructed_in_the_app(self):
        """shadow off exists only in scripts/tests — the app always shadows.

        The single source of a non-shadow adapter must be an explicit,
        human-authorized drill/window entry point (a script), never the
        application package itself.
        """
        import forex_ai_agent
        from pathlib import Path
        root = Path(forex_ai_agent.__file__).parent
        offenders = []
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            if "shadow_mode=False" in text:
                offenders.append(str(path))
        assert offenders == [], (
            f"shadow_mode=False constructed inside the app package: {offenders}"
        )
