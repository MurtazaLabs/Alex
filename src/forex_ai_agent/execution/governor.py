"""Stage E1: deterministic execution governor — kill switch + journal.

The governor is the E1 fault-handling brain the simulator (and, from E2, the
shadow/broker path) runs under. It is DETERMINISTIC and OBSERVE-CONSERVATIVE:

- **Kill switch / breakers:** daily-loss, drawdown, position-count and
  manual trip sources all drive the same single switch. Tripped ⇒ every
  intent is refused (PENDING intents cancelled, nothing new accepted) until
  an EXPLICIT human ``arm()`` — never a timeout, never an LLM.
- **Feed guards:** stale-price (no orders on data older than
  ``max_staleness_bars``) and monotonic-clock guards (backwards timestamps
  are rejected, not rewound).
- **Write-ahead journaling:** every accepted intent is journaled before
  execution (the double-spend guard), every fill cancels-and-replaces the
  journal state (EXECUTED), every trip cancels PENDING intents.
- **Trip position policy (human choice, declared constant):** what happens
  to already-OPEN positions on a trip is NOT decided at trip time — the
  human declares it in ``GovernorLimits.trip_position_policy``:
  ``TRIP_POSITION_HOLD`` (default — positions keep their engine lifecycle,
  SL/TP still applies) or ``TRIP_POSITION_FLATTEN`` (the simulator closes
  every open position at market on the trip bar and reports each close
  back here for audit). The governor itself never closes anything.

Constitution: deterministic constants in code; no network, no broker, no
LLM; tripping is a pure state transition — it can never place, modify or
close anything by itself.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.state_manager import StateError, StateMachine
from .journal import (
    INTENT_CANCELLED,
    INTENT_PENDING,
    IntentJournal,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionGovernor",
    "GovernorLimits",
    "GovernorState",
    "TRIP_MANUAL",
    "TRIP_DAILY_LOSS",
    "TRIP_DRAWDOWN",
    "TRIP_POSITION_COUNT",
    "TRIP_POSITION_HOLD",
    "TRIP_POSITION_FLATTEN",
    "TRIP_POSITION_POLICIES",
    "BROKER_FAULT_TABLE",
]

GOVERNOR_STATES = ["DISARMED", "ARMED", "HALTED"]
TRIP_MANUAL = "MANUAL"
TRIP_DAILY_LOSS = "DAILY_LOSS"
TRIP_DRAWDOWN = "DRAWDOWN"
TRIP_POSITION_COUNT = "POSITION_COUNT"

#: What happens to already-OPEN positions when the kill switch trips.
#: Declared by the human in GovernorLimits — never decided at trip time.
TRIP_POSITION_HOLD = "POSITION_HOLD"          # SL/TP lifecycle continues
TRIP_POSITION_FLATTEN = "POSITION_FLATTEN"    # close everything at market
TRIP_POSITION_POLICIES = (TRIP_POSITION_HOLD, TRIP_POSITION_FLATTEN)


class GovernorLimits:
    """Deterministic breaker thresholds (mirrors E4 cap constants).

    ``trip_position_policy`` is the human's declared flatten-or-hold choice;
    it is validated HERE so an unknown policy is a construction error, not
    a silent fallback at trip time. Default is HOLD (least intervention).
    """

    def __init__(
        self,
        *,
        daily_loss_limit: float = 500.0,
        max_drawdown: float = 1000.0,
        max_positions: int = 10,
        max_staleness_bars: int = 3,
        trip_position_policy: str = TRIP_POSITION_HOLD,
    ) -> None:
        if trip_position_policy not in TRIP_POSITION_POLICIES:
            raise ValueError(
                "trip_position_policy must be one of "
                f"{TRIP_POSITION_POLICIES}, got {trip_position_policy!r}"
            )
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.max_positions = max_positions
        self.max_staleness_bars = max_staleness_bars
        self.trip_position_policy = trip_position_policy


class GovernorState:
    """Observable snapshot of the governor for tests/reports."""

    def __init__(self, machine: StateMachine, journal: IntentJournal) -> None:
        self._machine = machine
        self._journal = journal

    @property
    def state(self) -> str:
        return self._machine.current

    @property
    def tripped(self) -> bool:
        return self._machine.current == "HALTED"

    @property
    def pending_intents(self) -> int:
        return len(self._journal.pending())


class ExecutionGovernor:
    """Kill-switch + feed guards + write-ahead journaling for execution."""

    def __init__(
        self,
        limits: Optional[GovernorLimits] = None,
        journal: Optional[IntentJournal] = None,
    ) -> None:
        self.limits = limits or GovernorLimits()
        self.journal = journal or IntentJournal()
        # DISARMED → ARMED → HALTED; re-arm only via human arm() from HALTED.
        self._machine = StateMachine(
            "execution_governor",
            GOVERNOR_STATES,
            initial="DISARMED",
            transitions={
                "DISARMED": ["ARMED"],
                "ARMED": ["HALTED"],
                "HALTED": [],  # only reset via arm() (guarded, see below)
            },
        )
        self.trip_reason: Optional[str] = None
        self.tripped_at: Optional[datetime] = None
        self.events: List[Dict[str, Any]] = []
        # Audit trail of kill-switch flatten closes (FLATTEN policy only;
        # the simulator reports them here via record_flattened).
        self.flattened: List[Dict[str, Any]] = []
        self._last_ts: Optional[datetime] = None
        self._last_bar_seen: Dict[str, datetime] = {}

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @property
    def state(self) -> GovernorState:
        return GovernorState(self._machine, self.journal)

    def _transition(self, target: str) -> None:
        # HALTED → ARMED re-arm is the human-only path; implement outside
        # the machine's transition table via reset (guarded by is HALTED).
        if self._machine.current == "HALTED" and target == "ARMED":
            self._machine.reset(to="ARMED")
            return
        self._machine.transition(target)

    def _emit(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, **payload})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def arm(self) -> None:
        """HUMAN action: arm (or re-arm after a trip). Idempotent when armed."""
        if self._machine.current == "ARMED":
            return
        self._transition("ARMED")
        self._emit("governor_armed")

    def trip(self, reason: str, ts: Optional[datetime] = None) -> None:
        """Trip the kill switch. ANY source (manual, breaker, chaos) lands here."""
        if self._machine.current == "HALTED":
            return  # idempotent — first trip reason wins
        was = self._machine.current
        self._transition("HALTED")
        self.trip_reason = reason
        self.tripped_at = ts or datetime.utcnow()
        cancelled = 0
        for intent in self.journal.pending():
            self.journal.mark_cancelled(intent["key"], f"kill_switch:{reason}")
            cancelled += 1
        self._emit("kill_switch_tripped", reason=reason, from_state=was,
                   cancelled_intents=cancelled,
                   position_policy=self.limits.trip_position_policy)
        logger.warning("governor kill switch tripped (%s): %d pending intent(s) cancelled",
                       reason, cancelled)

    # ------------------------------------------------------------------
    # Acceptance (the gate every intent passes)
    # ------------------------------------------------------------------

    def accept_intent(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        sl: Optional[float],
        tp: Optional[float],
        now: datetime,
        current_bar: Optional[int] = None,
        comment: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Journal an intent write-ahead — or refuse it (returns None).

        Refusal reasons: governor not ARMED, kill switch tripped, stale
        market data, non-monotonic clock.
        """
        if self._machine.current != "ARMED":
            self._emit("intent_refused", key=key, reason=f"not_armed:{self._machine.current}")
            return None
        if not self._clock_is_monotonic(now):
            self._emit("intent_refused", key=key, reason="clock_regression")
            return None
        if self._data_is_stale(symbol, now):
            self._emit("intent_refused", key=key, reason="stale_data")
            return None

        # Idempotency: a duplicate key returns the ORIGINAL record (and is
        # reported as refused-new so callers never execute twice).
        existing = self.journal.get(key)
        record = self.journal.record(
            key, symbol=symbol, side=side, volume=volume,
            sl=sl, tp=tp, created_ts=now, created_bar=current_bar,
            comment=comment,
        )
        if existing is not None:
            self._emit("intent_duplicate_suppressed", key=key)
            return None
        return record

    def record_bar(self, symbol: str, ts: datetime) -> None:
        """Feed the freshness tracker (the simulator calls this per bar)."""
        self._last_bar_seen[symbol] = ts
        self._clock_is_monotonic(ts)  # updates the monotonic guard

    def _clock_is_monotonic(self, ts: datetime) -> bool:
        if self._last_ts is not None and ts < self._last_ts:
            return False
        self._last_ts = ts
        return True

    def _data_is_stale(self, symbol: str, now: datetime) -> bool:
        last = self._last_bar_seen.get(symbol)
        if last is None:
            return False  # nothing observed yet — acceptance decides on trip state only
        max_age = timedelta(
            minutes=60 * (self.limits.max_staleness_bars + 1)
        )
        return (now - last) > max_age

    # ------------------------------------------------------------------
    # Breakers (checked by the simulator each bar)
    # ------------------------------------------------------------------

    def check_breakers(
        self,
        *,
        equity: float,
        day_start_equity: float,
        peak_equity: float,
        open_positions: int,
        now: datetime,
    ) -> None:
        """Evaluate deterministic breakers; trip on the first breach."""
        if self.state.tripped:
            return
        if day_start_equity > 0 and (day_start_equity - equity) >= self.limits.daily_loss_limit:
            self.trip(TRIP_DAILY_LOSS, ts=now)
            return
        if (peak_equity - equity) >= self.limits.max_drawdown:
            self.trip(TRIP_DRAWDOWN, ts=now)
            return
        if open_positions > self.limits.max_positions:
            self.trip(TRIP_POSITION_COUNT, ts=now)

    # ------------------------------------------------------------------
    # Execution bookkeeping (journal transitions)
    # ------------------------------------------------------------------

    def record_fill(self, key: str, fill: Dict[str, Any]) -> None:
        intent = self.journal.mark_executed(key, fill)
        if intent is None:
            self._emit("fill_for_unknown_intent", key=key)

    def record_cancel(self, key: str, reason: str) -> None:
        self.journal.mark_cancelled(key, reason)

    def record_flattened(self, record: Dict[str, Any]) -> None:
        """Simulator reports ONE kill-switch flatten close (audit trail).

        The governor performs no close itself; under the declared
        TRIP_POSITION_FLATTEN policy the simulator closes positions through
        the engine's own trade-record path and reports each close here so
        the audit trail shows exactly what the trip flattened.
        """
        self.flattened.append(record)
        self._emit("position_flattened", **record)

    def pending_intents(self) -> List[Dict[str, Any]]:
        return self.journal.pending()


# ---------------------------------------------------------------------------
# Broker fault table (E1): deterministic handler per MT5 failure mode
# ---------------------------------------------------------------------------

#: MT5 retcode → deterministic simulator/broker-adapter behavior. The
#: adapter (E2) maps real retcodes onto these actions; chaos tests assert
#: each row's declared behavior.
BROKER_FAULT_TABLE = {
    "TRADE_RETCODE_REQUOTE": "REPRICE_THEN_FILL",    # re-price per policy, retry bounded
    "TRADE_RETCODE_PRICE_OFF": "REJECT",             # off-quotes → refuse intent
    "TRADE_RETCODE_INVALID_STOPS": "REJECT",         # invalid SL/TP → refuse + log
    "TRADE_RETCODE_MARKET_CLOSED": "CANCEL_INTENT",  # cancel, not retry
    "TRADE_RETCODE_TOO_MANY_REQUESTS": "BOUNDED_RETRY_THEN_PENDING",
    "TRADE_RETCODE_SERVER_BUSY": "BOUNDED_RETRY_THEN_PENDING",
    "TRADE_RETCODE_CONNECTION_LOST": "HALT_AND_RECONCILE",
    "TRADE_RETCODE_DONE": "FILL",
    "TRADE_RETCODE_PLACED": "FILL",
}
