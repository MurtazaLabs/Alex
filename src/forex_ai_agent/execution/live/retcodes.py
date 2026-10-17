"""Stage E4: broker fault table — every MT5 retcode maps to one action.

The transmitter never interprets retcodes inline. It looks them up here,
where each code has a declared verdict, because "what do we do on code X"
is a risk decision the human reviews, not a branch an engineer improvises
at 3am:

* FILLED    — the order executed. Record the fill.
* PLACED    — a pending order was accepted.
* PARTIAL   — partial fill. Record what filled; the rest is a human note.
* RETRY     — transient (requote / price off / rate limit). The caller may
              re-attempt ONCE with fresh data and a FRESH token; the
              transmitter itself never auto-retries.
* REJECTED  — the request was bad and retrying it unchanged is pointless
              (bad stops, bad volume, market closed, caps at venue).
* CRITICAL  — state unknown or safety-relevant (connection lost, account
              locked, trading disabled, out of money, unknown code). The
              kill switch trips: fail closed, human decides.

Unknown codes classify CRITICAL on purpose. An unrecognized outcome is
exactly the moment not to guess.

Pure data + a pure function: no network, no broker import, no LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

__all__ = [
    "VERDICT_FILLED",
    "VERDICT_PLACED",
    "VERDICT_PARTIAL",
    "VERDICT_RETRY",
    "VERDICT_REJECTED",
    "VERDICT_CRITICAL",
    "FaultAction",
    "BROKER_FAULT_TABLE",
    "classify_retcode",
]

VERDICT_FILLED = "FILLED"
VERDICT_PLACED = "PLACED"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_RETRY = "RETRY"
VERDICT_REJECTED = "REJECTED"
VERDICT_CRITICAL = "CRITICAL"

# MT5 trade server return codes (MetaTrader5.TRADE_RETCODE_*).
TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_CANCEL = 10007
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_DONE_PARTIAL = 10010
TRADE_RETCODE_ERROR = 10011
TRADE_RETCODE_TIMEOUT = 10012
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_PRICE = 10015
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_TRADE_DISABLED = 10017
TRADE_RETCODE_MARKET_CLOSED = 10018
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_PRICE_CHANGED = 10020
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_INVALID_EXPIRATION = 10022
TRADE_RETCODE_ORDER_CHANGED = 10023
TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
TRADE_RETCODE_NO_CHANGES = 10025
TRADE_RETCODE_SERVER_DISABLES_AT = 10026
TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
TRADE_RETCODE_LOCKED = 10028
TRADE_RETCODE_FROZEN = 10029
TRADE_RETCODE_INVALID_FILL = 10030
TRADE_RETCODE_CONNECTION = 10031
TRADE_RETCODE_ONLY_REAL = 10032
TRADE_RETCODE_LIMIT_ORDERS = 10033
TRADE_RETCODE_LIMIT_VOLUME = 10034
TRADE_RETCODE_INVALID_ORDER = 10035
TRADE_RETCODE_POSITION_CLOSED = 10036
TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
TRADE_RETCODE_LIMIT_POSITIONS = 10040
TRADE_RETCODE_REJECT_CANCEL = 10041
TRADE_RETCODE_LONG_ONLY = 10042
TRADE_RETCODE_SHORT_ONLY = 10043
TRADE_RETCODE_CLOSE_ONLY = 10044
TRADE_RETCODE_FIFO_CLOSE = 10045


@dataclass(frozen=True)
class FaultAction:
    """Declared handling for one retcode."""

    retcode: int
    verdict: str
    name: str
    detail: str

    @property
    def is_terminal(self) -> bool:
        """True when the outcome is final and the fill is (partially) real."""
        return self.verdict in (VERDICT_FILLED, VERDICT_PLACED, VERDICT_PARTIAL)

    @property
    def trips_kill_switch(self) -> bool:
        return self.verdict == VERDICT_CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retcode": self.retcode,
            "verdict": self.verdict,
            "name": self.name,
            "detail": self.detail,
        }


BROKER_FAULT_TABLE: Dict[int, FaultAction] = {
    TRADE_RETCODE_REQUOTE: FaultAction(
        TRADE_RETCODE_REQUOTE, VERDICT_RETRY, "REQUOTE",
        "quote moved; re-attempt once with fresh price and fresh token",
    ),
    TRADE_RETCODE_REJECT: FaultAction(
        TRADE_RETCODE_REJECT, VERDICT_REJECTED, "REJECT",
        "request rejected by dealer; do not retry unchanged",
    ),
    TRADE_RETCODE_CANCEL: FaultAction(
        TRADE_RETCODE_CANCEL, VERDICT_REJECTED, "CANCEL",
        "request cancelled by trader/dealer",
    ),
    TRADE_RETCODE_PLACED: FaultAction(
        TRADE_RETCODE_PLACED, VERDICT_PLACED, "PLACED",
        "pending order accepted",
    ),
    TRADE_RETCODE_DONE: FaultAction(
        TRADE_RETCODE_DONE, VERDICT_FILLED, "DONE",
        "order executed",
    ),
    TRADE_RETCODE_DONE_PARTIAL: FaultAction(
        TRADE_RETCODE_DONE_PARTIAL, VERDICT_PARTIAL, "DONE_PARTIAL",
        "partial fill; remainder is a human note",
    ),
    TRADE_RETCODE_ERROR: FaultAction(
        TRADE_RETCODE_ERROR, VERDICT_CRITICAL, "ERROR",
        "generic server error; outcome unknown, halt and reconcile",
    ),
    TRADE_RETCODE_TIMEOUT: FaultAction(
        TRADE_RETCODE_TIMEOUT, VERDICT_CRITICAL, "TIMEOUT",
        "no server response; outcome unknown, halt and reconcile",
    ),
    TRADE_RETCODE_INVALID: FaultAction(
        TRADE_RETCODE_INVALID, VERDICT_REJECTED, "INVALID",
        "malformed request; a code defect, not a market condition",
    ),
    TRADE_RETCODE_INVALID_VOLUME: FaultAction(
        TRADE_RETCODE_INVALID_VOLUME, VERDICT_REJECTED, "INVALID_VOLUME",
        "volume outside broker limits; sizing defect",
    ),
    TRADE_RETCODE_INVALID_PRICE: FaultAction(
        TRADE_RETCODE_INVALID_PRICE, VERDICT_RETRY, "INVALID_PRICE",
        "price not in market; one re-attempt with fresh tick",
    ),
    TRADE_RETCODE_INVALID_STOPS: FaultAction(
        TRADE_RETCODE_INVALID_STOPS, VERDICT_REJECTED, "INVALID_STOPS",
        "SL/TP invalid for this venue; stop-distance defect",
    ),
    TRADE_RETCODE_TRADE_DISABLED: FaultAction(
        TRADE_RETCODE_TRADE_DISABLED, VERDICT_CRITICAL, "TRADE_DISABLED",
        "trading disabled for the account/symbol; halt",
    ),
    TRADE_RETCODE_MARKET_CLOSED: FaultAction(
        TRADE_RETCODE_MARKET_CLOSED, VERDICT_REJECTED, "MARKET_CLOSED",
        "market closed; session filter should have prevented this",
    ),
    TRADE_RETCODE_NO_MONEY: FaultAction(
        TRADE_RETCODE_NO_MONEY, VERDICT_CRITICAL, "NO_MONEY",
        "insufficient funds; halt and notify human",
    ),
    TRADE_RETCODE_PRICE_CHANGED: FaultAction(
        TRADE_RETCODE_PRICE_CHANGED, VERDICT_RETRY, "PRICE_CHANGED",
        "price changed; one re-attempt with fresh tick",
    ),
    TRADE_RETCODE_PRICE_OFF: FaultAction(
        TRADE_RETCODE_PRICE_OFF, VERDICT_RETRY, "PRICE_OFF",
        "no quotes to process; one re-attempt with fresh tick",
    ),
    TRADE_RETCODE_INVALID_EXPIRATION: FaultAction(
        TRADE_RETCODE_INVALID_EXPIRATION, VERDICT_REJECTED, "INVALID_EXPIRATION",
        "expiration invalid; request defect",
    ),
    TRADE_RETCODE_ORDER_CHANGED: FaultAction(
        TRADE_RETCODE_ORDER_CHANGED, VERDICT_REJECTED, "ORDER_CHANGED",
        "order state changed; never retry blind",
    ),
    TRADE_RETCODE_TOO_MANY_REQUESTS: FaultAction(
        TRADE_RETCODE_TOO_MANY_REQUESTS, VERDICT_RETRY, "TOO_MANY_REQUESTS",
        "rate limited; one re-attempt after backoff",
    ),
    TRADE_RETCODE_NO_CHANGES: FaultAction(
        TRADE_RETCODE_NO_CHANGES, VERDICT_REJECTED, "NO_CHANGES",
        "no changes in request",
    ),
    TRADE_RETCODE_SERVER_DISABLES_AT: FaultAction(
        TRADE_RETCODE_SERVER_DISABLES_AT, VERDICT_CRITICAL, "SERVER_DISABLES_AT",
        "autotrading disabled server-side; halt",
    ),
    TRADE_RETCODE_CLIENT_DISABLES_AT: FaultAction(
        TRADE_RETCODE_CLIENT_DISABLES_AT, VERDICT_CRITICAL, "CLIENT_DISABLES_AT",
        "autotrading disabled terminal-side; halt",
    ),
    TRADE_RETCODE_LOCKED: FaultAction(
        TRADE_RETCODE_LOCKED, VERDICT_CRITICAL, "LOCKED",
        "request locked by dealer; halt and call the broker",
    ),
    TRADE_RETCODE_FROZEN: FaultAction(
        TRADE_RETCODE_FROZEN, VERDICT_REJECTED, "FROZEN",
        "order/position frozen",
    ),
    TRADE_RETCODE_INVALID_FILL: FaultAction(
        TRADE_RETCODE_INVALID_FILL, VERDICT_REJECTED, "INVALID_FILL",
        "unsupported filling mode; adapter defect",
    ),
    TRADE_RETCODE_CONNECTION: FaultAction(
        TRADE_RETCODE_CONNECTION, VERDICT_CRITICAL, "CONNECTION",
        "no connection to trade server; outcome unknown, halt and reconcile",
    ),
    TRADE_RETCODE_ONLY_REAL: FaultAction(
        TRADE_RETCODE_ONLY_REAL, VERDICT_REJECTED, "ONLY_REAL",
        "operation allowed on real accounts only",
    ),
    TRADE_RETCODE_LIMIT_ORDERS: FaultAction(
        TRADE_RETCODE_LIMIT_ORDERS, VERDICT_REJECTED, "LIMIT_ORDERS",
        "venue order-count limit reached",
    ),
    TRADE_RETCODE_LIMIT_VOLUME: FaultAction(
        TRADE_RETCODE_LIMIT_VOLUME, VERDICT_REJECTED, "LIMIT_VOLUME",
        "venue volume limit reached",
    ),
    TRADE_RETCODE_INVALID_ORDER: FaultAction(
        TRADE_RETCODE_INVALID_ORDER, VERDICT_REJECTED, "INVALID_ORDER",
        "wrong/missing order type",
    ),
    TRADE_RETCODE_POSITION_CLOSED: FaultAction(
        TRADE_RETCODE_POSITION_CLOSED, VERDICT_REJECTED, "POSITION_CLOSED",
        "position already closed",
    ),
    TRADE_RETCODE_INVALID_CLOSE_VOLUME: FaultAction(
        TRADE_RETCODE_INVALID_CLOSE_VOLUME, VERDICT_REJECTED, "INVALID_CLOSE_VOLUME",
        "close volume exceeds position",
    ),
    TRADE_RETCODE_CLOSE_ORDER_EXIST: FaultAction(
        TRADE_RETCODE_CLOSE_ORDER_EXIST, VERDICT_REJECTED, "CLOSE_ORDER_EXIST",
        "a close order already exists",
    ),
    TRADE_RETCODE_LIMIT_POSITIONS: FaultAction(
        TRADE_RETCODE_LIMIT_POSITIONS, VERDICT_REJECTED, "LIMIT_POSITIONS",
        "venue position limit reached; our caps should have prevented this",
    ),
    TRADE_RETCODE_REJECT_CANCEL: FaultAction(
        TRADE_RETCODE_REJECT_CANCEL, VERDICT_REJECTED, "REJECT_CANCEL",
        "order rejected then cancelled",
    ),
    TRADE_RETCODE_LONG_ONLY: FaultAction(
        TRADE_RETCODE_LONG_ONLY, VERDICT_REJECTED, "LONG_ONLY",
        "venue allows long positions only",
    ),
    TRADE_RETCODE_SHORT_ONLY: FaultAction(
        TRADE_RETCODE_SHORT_ONLY, VERDICT_REJECTED, "SHORT_ONLY",
        "venue allows short positions only",
    ),
    TRADE_RETCODE_CLOSE_ONLY: FaultAction(
        TRADE_RETCODE_CLOSE_ONLY, VERDICT_CRITICAL, "CLOSE_ONLY",
        "venue restricted to close-only; halt new opens",
    ),
    TRADE_RETCODE_FIFO_CLOSE: FaultAction(
        TRADE_RETCODE_FIFO_CLOSE, VERDICT_REJECTED, "FIFO_CLOSE",
        "venue requires FIFO close order first",
    ),
}


def classify_retcode(retcode: Optional[int]) -> FaultAction:
    """Map a retcode to its declared action; unknown codes are CRITICAL.

    A missing or unrecognized retcode is exactly the moment to stop, not
    to guess: fail closed.
    """
    if retcode is None:
        return FaultAction(
            -1, VERDICT_CRITICAL, "NO_RETCODE",
            "no retcode from terminal; outcome unknown",
        )
    action = BROKER_FAULT_TABLE.get(int(retcode))
    if action is None:
        return FaultAction(
            int(retcode), VERDICT_CRITICAL, "UNKNOWN",
            f"unrecognized retcode {retcode}; halt and reconcile",
        )
    return action
