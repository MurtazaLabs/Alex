"""Stage E4: session filter — London and New York windows only.

Forex trades around the clock, but the E4 window does not. Signals are
only transmittable while a major session is open, computed in UTC from
deterministic constants:

* London:    07:00–16:00 UTC
* New York:  12:00–21:00 UTC
* Union:     07:00–21:00 UTC, Monday–Friday

Weekends are outside the union by construction (Saturday and Sunday never
fall inside a 07:00–21:00 weekday rule we apply — the check is explicit).

Pure clock math over an injected ``now``: no network, no broker, no LLM,
fully testable. The filter answers with a structured verdict so every
refusal has a reason string an auditor can read.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .caps import london_window, new_york_window

__all__ = [
    "SESSION_LONDON",
    "SESSION_NEW_YORK",
    "SESSION_LONDON_NY",
    "SessionCheck",
    "SessionFilter",
]

SESSION_LONDON = "LONDON"
SESSION_NEW_YORK = "NEW_YORK"
SESSION_LONDON_NY = "LONDON+NEW_YORK"

# Monday=0 ... Sunday=6 (datetime.weekday()).
_WEEKEND_DAYS = (5, 6)


@dataclass(frozen=True)
class SessionCheck:
    """Structured result of one session check."""

    open: bool
    session: Optional[str]
    reason: str
    now: datetime

    def to_dict(self) -> Dict[str, Any]:
        return {
            "open": self.open,
            "session": self.session,
            "reason": self.reason,
            "now": self.now.isoformat(),
        }


class SessionFilter:
    """Deterministic London/New York session gate in UTC."""

    def __init__(
        self,
        *,
        london: tuple = None,
        new_york: tuple = None,
    ) -> None:
        self.london = tuple(london) if london else london_window()
        self.new_york = tuple(new_york) if new_york else new_york_window()

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _in_window(hour: float, window: tuple) -> bool:
        start, end = window
        return float(start) <= hour < float(end)

    @staticmethod
    def _hour_fraction(now: datetime) -> float:
        return now.hour + (now.minute / 60.0) + (now.second / 3600.0)

    # -- the check -------------------------------------------------------

    def check(self, now: Optional[datetime] = None) -> SessionCheck:
        """Can a signal be transmitted at ``now``?"""
        check_time = now or datetime.utcnow()

        if check_time.weekday() in _WEEKEND_DAYS:
            return SessionCheck(
                open=False,
                session=None,
                reason="weekend_closed",
                now=check_time,
            )

        hour = self._hour_fraction(check_time)
        in_london = self._in_window(hour, self.london)
        in_new_york = self._in_window(hour, self.new_york)

        if in_london and in_new_york:
            return SessionCheck(True, SESSION_LONDON_NY, "open", check_time)
        if in_london:
            return SessionCheck(True, SESSION_LONDON, "open", check_time)
        if in_new_york:
            return SessionCheck(True, SESSION_NEW_YORK, "open", check_time)
        return SessionCheck(False, None, "outside_session_hours", check_time)

    def is_open(self, now: Optional[datetime] = None) -> bool:
        return self.check(now).open

    def session_name(self, now: Optional[datetime] = None) -> Optional[str]:
        """The session label at ``now`` (None when closed)."""
        return self.check(now).session
