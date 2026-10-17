"""Stage E4: live transmitter — the one guarded path to the wire.

This is the only component in the codebase that calls
``broker.submit_order``. Everything before it (signal, policy, governor,
tokens) prepared and authorized; everything after it (tracker, fills,
report) records what happened. The transmitter itself decides nothing
about the market — it evaluates a fixed gate chain and either refuses
with an auditable reason or performs exactly ONE wire attempt.

Gate chain (each step recorded in the result's ``checks`` list):

1. intent status VALID        -> else REFUSED_INTENT
2. intent not expired         -> else REFUSED_EXPIRED
3. symbol == caps.pair        -> else REFUSED_PAIR
4. volume within lot bounds   -> else REFUSED_VOLUME
5. session open               -> else REFUSED_SESSION
6. not in news blackout       -> else REFUSED_NEWS
7. governor ARMED             -> else REFUSED_GOVERNOR
8. broker unlocked (level 4,
   shadow mode off)           -> else REFUSED_BROKER_GATE
9. token consumed (single-
   use, intent-bound)         -> else REFUSED_TOKEN
10. exactly one order_send; the retcode is classified by the fault table.
    FILLED / PLACED / PARTIAL record the fill; RETRY and REJECTED return
    without further attempts (a retry needs a FRESH token by design);
    CRITICAL trips the kill switch — fail closed, human decides.

No auto-retry, no loops, no clock of its own: ``now`` is injected so the
whole chain is deterministic and testable. No LLM, no network beyond the
single ``broker.submit_order`` call the gate chain releases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .caps import ALEX_LIVE_MAGIC, LiveCaps, DEFAULT_LIVE_CAPS
from .news import NewsBlackout
from .retcodes import (
    VERDICT_CRITICAL,
    VERDICT_FILLED,
    VERDICT_PARTIAL,
    VERDICT_PLACED,
    VERDICT_REJECTED,
    VERDICT_RETRY,
    classify_retcode,
)
from .sessions import SessionFilter

__all__ = [
    "TRANSMIT_FILLED",
    "TRANSMIT_PLACED",
    "TRANSMIT_PARTIAL",
    "TRANSMIT_RETRY",
    "TRANSMIT_REJECTED",
    "TRANSMIT_CRITICAL",
    "TRANSMIT_REFUSED_INTENT",
    "TRANSMIT_REFUSED_EXPIRED",
    "TRANSMIT_REFUSED_PAIR",
    "TRANSMIT_REFUSED_VOLUME",
    "TRANSMIT_REFUSED_SESSION",
    "TRANSMIT_REFUSED_NEWS",
    "TRANSMIT_REFUSED_GOVERNOR",
    "TRANSMIT_REFUSED_BROKER_GATE",
    "TRANSMIT_REFUSED_TOKEN",
    "TransmissionResult",
    "LiveTransmitter",
]

TRANSMIT_FILLED = "FILLED"
TRANSMIT_PLACED = "PLACED"
TRANSMIT_PARTIAL = "PARTIAL"
TRANSMIT_RETRY = "RETRY"
TRANSMIT_REJECTED = "REJECTED"
TRANSMIT_CRITICAL = "CRITICAL_HALT"
TRANSMIT_REFUSED_INTENT = "REFUSED_INTENT"
TRANSMIT_REFUSED_EXPIRED = "REFUSED_EXPIRED"
TRANSMIT_REFUSED_PAIR = "REFUSED_PAIR"
TRANSMIT_REFUSED_VOLUME = "REFUSED_VOLUME"
TRANSMIT_REFUSED_SESSION = "REFUSED_SESSION"
TRANSMIT_REFUSED_NEWS = "REFUSED_NEWS"
TRANSMIT_REFUSED_GOVERNOR = "REFUSED_GOVERNOR"
TRANSMIT_REFUSED_BROKER_GATE = "REFUSED_BROKER_GATE"
TRANSMIT_REFUSED_TOKEN = "REFUSED_TOKEN"

#: Statuses in which the outcome is a real, recordable market event.
_TERMINAL_SENT = (TRANSMIT_FILLED, TRANSMIT_PLACED, TRANSMIT_PARTIAL)

_REQUIRED_WRITE_LEVEL = 4


@dataclass
class TransmissionResult:
    """Everything that happened for one transmit call — the audit record."""

    intent_key: str
    status: str
    sent: bool
    detail: str
    at: datetime
    ticket: Optional[int] = None
    retcode: Optional[int] = None
    fault: Optional[Dict[str, Any]] = None
    token_id: Optional[str] = None
    checks: List[Dict[str, str]] = field(default_factory=list)

    @property
    def is_fill(self) -> bool:
        return self.status in _TERMINAL_SENT

    @property
    def is_refusal(self) -> bool:
        return self.status.startswith("REFUSED_")

    @property
    def is_critical(self) -> bool:
        return self.status == TRANSMIT_CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_key": self.intent_key,
            "status": self.status,
            "sent": self.sent,
            "detail": self.detail,
            "at": self.at.isoformat(),
            "ticket": self.ticket,
            "retcode": self.retcode,
            "fault": self.fault,
            "token_id": self.token_id,
            "checks": list(self.checks),
        }


class LiveTransmitter:
    """Gate chain + exactly one wire attempt per call.

    Built once per process from the broker adapter, the token store, and
    the governor. ``transmit`` is the only method that can reach the wire
    and it reaches it once per call, or not at all.
    """

    def __init__(
        self,
        broker: Any,
        token_store: Any,
        governor: Any,
        *,
        caps: Optional[LiveCaps] = None,
        session: Optional[SessionFilter] = None,
        blackout: Optional[NewsBlackout] = None,
    ) -> None:
        self.broker = broker
        self.token_store = token_store
        self.governor = governor
        self.caps = caps or DEFAULT_LIVE_CAPS
        self.session = session or SessionFilter()
        self.blackout = blackout or NewsBlackout(window_minutes=self.caps.news_blackout_minutes)
        #: Every result ever produced, newest last — the transmitter's own
        #: audit trail alongside the broker's.
        self.history: List[TransmissionResult] = []

    # ------------------------------------------------------------------
    # The gate chain
    # ------------------------------------------------------------------

    def _check(
        self,
        result: TransmissionResult,
        step: str,
        ok: bool,
        detail: str = "",
    ) -> bool:
        result.checks.append({"step": step, "ok": "true" if ok else "false", "detail": detail})
        return ok

    def _refuse(
        self,
        result: TransmissionResult,
        status: str,
        detail: str,
    ) -> TransmissionResult:
        result.status = status
        result.detail = detail
        result.sent = False
        self._finalize(result)
        return result

    def _finalize(self, result: TransmissionResult) -> None:
        self.history.append(result)

    # ------------------------------------------------------------------
    # Transmit
    # ------------------------------------------------------------------

    def transmit(
        self,
        intent: Any,
        token_id: str,
        *,
        now: Optional[datetime] = None,
    ) -> TransmissionResult:
        """Evaluate the gate chain; perform at most one wire attempt."""
        check_time = now or datetime.utcnow()
        result = TransmissionResult(
            intent_key=str(getattr(intent, "key", "")),
            status="",
            sent=False,
            detail="",
            at=check_time,
            token_id=token_id or None,
        )

        # -- 1. intent state --------------------------------------------
        intent_status = str(getattr(intent, "status", ""))
        if intent_status != "VALID":
            self._check(result, "intent_status", False, f"status={intent_status}")
            return self._refuse(
                result, TRANSMIT_REFUSED_INTENT,
                f"intent status is {intent_status}, not VALID",
            )
        self._check(result, "intent_status", True, "VALID")

        # -- 2. expiry ---------------------------------------------------
        is_expired = getattr(intent, "is_expired", None)
        if callable(is_expired) and is_expired(check_time):
            self._check(result, "expiry", False, "intent expired")
            return self._refuse(
                result, TRANSMIT_REFUSED_EXPIRED,
                f"intent expired at {getattr(intent, 'expires_at', '?')}",
            )
        self._check(result, "expiry", True, "not expired")

        # -- 3. pair -----------------------------------------------------
        symbol = str(getattr(intent, "symbol", ""))
        if symbol != self.caps.pair:
            self._check(result, "pair", False, f"symbol={symbol}")
            return self._refuse(
                result, TRANSMIT_REFUSED_PAIR,
                f"symbol {symbol!r} is not the E4 pair {self.caps.pair!r}",
            )
        self._check(result, "pair", True, symbol)

        # -- 4. volume ---------------------------------------------------
        volume = float(getattr(intent, "volume", 0.0) or 0.0)
        if volume < self.caps.min_lot or volume > self.caps.max_lot:
            self._check(result, "volume", False, f"volume={volume}")
            return self._refuse(
                result, TRANSMIT_REFUSED_VOLUME,
                f"volume {volume} outside E4 bounds "
                f"[{self.caps.min_lot}, {self.caps.max_lot}]",
            )
        self._check(result, "volume", True, str(volume))

        # -- 5. session --------------------------------------------------
        session_check = self.session.check(check_time)
        if not session_check.open:
            self._check(result, "session", False, session_check.reason)
            return self._refuse(
                result, TRANSMIT_REFUSED_SESSION,
                f"session closed: {session_check.reason}",
            )
        self._check(result, "session", True, str(session_check.session or "open"))

        # -- 6. news blackout --------------------------------------------
        blackout_check = self.blackout.check(check_time)
        if blackout_check.blocked:
            event_title = blackout_check.event.title if blackout_check.event else "?"
            self._check(result, "news", False, event_title)
            return self._refuse(
                result, TRANSMIT_REFUSED_NEWS,
                f"news blackout around {event_title!r}",
            )
        self._check(result, "news", True, "clear")

        # -- 7. governor -------------------------------------------------
        governor_state = getattr(self.governor, "state", None)
        state_name = getattr(governor_state, "state", "") if governor_state else ""
        if state_name != "ARMED":
            self._check(result, "governor", False, f"state={state_name}")
            return self._refuse(
                result, TRANSMIT_REFUSED_GOVERNOR,
                f"governor state is {state_name or 'unknown'}, not ARMED",
            )
        self._check(result, "governor", True, "ARMED")

        # -- 8. broker gate peek (the adapter re-checks on send) ---------
        level = int(getattr(self.broker, "permission_level", 0) or 0)
        shadow = bool(getattr(self.broker, "shadow_mode", True))
        if level < _REQUIRED_WRITE_LEVEL:
            self._check(result, "broker_gate", False, f"level={level}")
            return self._refuse(
                result, TRANSMIT_REFUSED_BROKER_GATE,
                f"broker permission level {level} < {_REQUIRED_WRITE_LEVEL}",
            )
        if shadow:
            self._check(result, "broker_gate", False, "shadow_mode=on")
            return self._refuse(
                result, TRANSMIT_REFUSED_BROKER_GATE,
                "broker is in shadow mode",
            )
        self._check(result, "broker_gate", True, f"level={level}, shadow=off")

        # -- 9. token (single-use, intent-bound) -------------------------
        side = str(getattr(intent, "side", ""))
        consumed = self.token_store.consume(
            token_id,
            intent_key=result.intent_key,
            symbol=symbol,
            side=side,
            volume=volume,
            now=check_time,
        )
        if not consumed:
            self._check(result, "token", False, f"token={str(token_id)[:8]}")
            return self._refuse(
                result, TRANSMIT_REFUSED_TOKEN,
                "approval token invalid, expired, already used, or not bound "
                "to this intent",
            )
        self._check(result, "token", True, f"token={str(token_id)[:8]} consumed")

        # -- 10. exactly one wire attempt --------------------------------
        request = self._build_request(intent, symbol, side, volume)
        try:
            ack = self.broker.submit_order(request)
        except Exception as exc:  # noqa: BLE001 — unknown state = halt
            self._check(result, "send", False, f"{type(exc).__name__}: {exc}")
            self._critical(
                result, f"broker exception {type(exc).__name__}: {exc}"
            )
            return result
        self._check(
            result, "send", True,
            f"ack sent={getattr(ack, 'sent', '?')} status={getattr(ack, 'status', '?')}",
        )

        return self._classify(result, ack)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_request(
        self, intent: Any, symbol: str, side: str, volume: float
    ) -> Any:
        from ...brokers.base import OrderRequest

        return OrderRequest(
            key=str(getattr(intent, "key", "")),
            symbol=symbol,
            side=side,
            volume=volume,
            price=float(getattr(intent, "entry", 0.0) or 0.0),
            sl=getattr(intent, "sl", None),
            tp=getattr(intent, "tp", None),
            magic=ALEX_LIVE_MAGIC,
            comment=str(getattr(intent, "comment", "") or "alex e4"),
        )

    def _classify(self, result: TransmissionResult, ack: Any) -> TransmissionResult:
        ack_sent = bool(getattr(ack, "sent", False))
        ack_status = str(getattr(ack, "status", "") or "")
        retcode = getattr(ack, "retcode", None)
        ticket = getattr(ack, "ticket", None)
        reason = str(getattr(ack, "reason", "") or "")

        # The adapter can return sent=False BEFORE the wire (NO_TICK,
        # NOT_FOUND): a clean refusal, not a market event.
        if not ack_sent:
            self._check(result, "wire", False, f"adapter status={ack_status}")
            return self._refuse(
                result, TRANSMIT_REJECTED,
                f"adapter refused before wire: {ack_status} {reason}".strip(),
            )

        action = classify_retcode(retcode if retcode else None)
        result.retcode = retcode if isinstance(retcode, int) else None
        result.fault = action.to_dict()
        result.ticket = ticket if isinstance(ticket, int) and ticket else result.ticket

        if action.verdict == VERDICT_CRITICAL:
            result.detail = f"critical retcode: {action.name} ({action.detail})"
            self._critical(result, result.detail, ticket=result.ticket, retcode=result.retcode)
            return result

        if action.verdict == VERDICT_FILLED:
            result.status = TRANSMIT_FILLED
        elif action.verdict == VERDICT_PLACED:
            result.status = TRANSMIT_PLACED
        elif action.verdict == VERDICT_PARTIAL:
            result.status = TRANSMIT_PARTIAL
        elif action.verdict == VERDICT_RETRY:
            result.status = TRANSMIT_RETRY
        else:  # VERDICT_REJECTED
            result.status = TRANSMIT_REJECTED

        result.sent = True
        result.detail = reason or action.name
        result.checks.append(
            {"step": "retcode", "ok": "true", "detail": f"{action.name}={result.retcode}"}
        )

        if result.status in _TERMINAL_SENT:
            self._record_fill(result)

        self._finalize(result)
        return result

    def _critical(
        self,
        result: TransmissionResult,
        detail: str,
        *,
        ticket: Optional[int] = None,
        retcode: Optional[int] = None,
    ) -> None:
        """Trip the kill switch — the fail-closed path for unknown states."""
        result.status = TRANSMIT_CRITICAL
        result.detail = detail
        result.sent = True
        result.ticket = ticket
        if retcode is not None:
            result.retcode = retcode
        try:
            self.governor.trip(f"transmitter_critical:{detail}", ts=result.at)
            result.checks.append(
                {"step": "kill_switch", "ok": "true", "detail": "tripped"}
            )
        except Exception as exc:  # noqa: BLE001 — never mask the original
            result.checks.append(
                {"step": "kill_switch", "ok": "false", "detail": str(exc)}
            )
        self._finalize(result)

    def _record_fill(self, result: TransmissionResult) -> None:
        """Bank a real fill in the governor's journal (write-ahead closes)."""
        fill = {
            "ticket": result.ticket,
            "retcode": result.retcode,
            "status": result.status,
            "price": None,  # actual fill price is read back by the tracker
            "source": "transmitter",
            "at": result.at.isoformat(),
        }
        try:
            self.governor.record_fill(result.intent_key, fill)
            result.checks.append(
                {"step": "journal", "ok": "true", "detail": "fill recorded"}
            )
        except Exception as exc:  # noqa: BLE001 — the fill happened; note it
            result.checks.append(
                {"step": "journal", "ok": "false", "detail": str(exc)}
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def results(self) -> List[TransmissionResult]:
        return list(self.history)

    def sent_count(self) -> int:
        """How many calls reached the wire with a green outcome."""
        return sum(1 for r in self.history if r.sent and not r.is_refusal)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for r in self.history:
            counts[r.status] = counts.get(r.status, 0) + 1
        return {
            "calls": len(self.history),
            "sent": self.sent_count(),
            "statuses": counts,
        }
