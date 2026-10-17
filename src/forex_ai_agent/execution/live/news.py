"""Stage E4: news blackout — no transmission around high-impact events.

Market-moving releases (NFP, CPI, central-bank decisions) turn order flow
into a coin flip for a few minutes. The E4 window refuses to transmit
within +/-30 minutes of any high-impact event on the calendar.

The event list comes from the existing news agent (the caller feeds it in);
this module is the deterministic filter over it:

* events carry a UTC timestamp, a title, and an impact level
* only ``impacts`` (default: high) participate
* a check at ``now`` is blocked when any event falls within the window
* the blocking event is returned in the verdict for the audit trail

Pure computation: no network, no scheduler, no LLM. If the calendar is
empty the filter passes everything — the fail direction is declared here
so it is a reviewable choice, not an accident (the caller is responsible
for refreshing the calendar; the daily report surfaces the event count).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

__all__ = [
    "IMPACT_HIGH",
    "NewsEvent",
    "BlackoutCheck",
    "NewsBlackout",
]

IMPACT_HIGH = "high"
IMPACT_MEDIUM = "medium"
IMPACT_LOW = "low"


@dataclass(frozen=True)
class NewsEvent:
    """One calendar entry with a UTC timestamp."""

    timestamp: datetime
    title: str
    impact: str = IMPACT_HIGH

    @property
    def normalized_impact(self) -> str:
        return str(self.impact or "").strip().lower()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "impact": self.normalized_impact,
        }


@dataclass(frozen=True)
class BlackoutCheck:
    """Structured result of one blackout check."""

    blocked: bool
    reason: str
    now: datetime
    event: Optional[NewsEvent] = None
    minutes_to_event: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "reason": self.reason,
            "now": self.now.isoformat(),
            "event": self.event.to_dict() if self.event else None,
            "minutes_to_event": self.minutes_to_event,
        }


class NewsBlackout:
    """Deterministic +/-window filter over high-impact calendar events."""

    def __init__(
        self,
        events: Optional[Iterable[NewsEvent]] = None,
        *,
        window_minutes: int = 30,
        impacts: Sequence[str] = (IMPACT_HIGH,),
    ) -> None:
        self.window_minutes = int(window_minutes)
        self.impacts = tuple(str(i).strip().lower() for i in impacts)
        self._events: List[NewsEvent] = []
        if events:
            self.refresh(events)

    # -- calendar management --------------------------------------------

    def refresh(self, events: Iterable[NewsEvent]) -> None:
        """Replace the calendar (sorted, filtered to the watched impacts)."""
        kept = [
            e for e in events
            if e.normalized_impact in self.impacts
        ]
        kept.sort(key=lambda e: e.timestamp)
        self._events = kept

    def add_event(self, event: NewsEvent) -> None:
        if event.normalized_impact not in self.impacts:
            return
        self._events.append(event)
        self._events.sort(key=lambda e: e.timestamp)

    @property
    def events(self) -> List[NewsEvent]:
        return list(self._events)

    @property
    def watched_impacts(self) -> tuple:
        return self.impacts

    # -- the check -------------------------------------------------------

    def check(self, now: Optional[datetime] = None) -> BlackoutCheck:
        """Is ``now`` inside the blackout window of any watched event?"""
        check_time = now or datetime.utcnow()
        half = timedelta(minutes=self.window_minutes)

        for event in self._events:
            delta = event.timestamp - check_time
            if -half <= delta <= half:
                return BlackoutCheck(
                    blocked=True,
                    reason="news_blackout",
                    now=check_time,
                    event=event,
                    minutes_to_event=round(delta.total_seconds() / 60.0, 1),
                )

        return BlackoutCheck(
            blocked=False,
            reason="clear",
            now=check_time,
        )

    def is_blocked(self, now: Optional[datetime] = None) -> bool:
        return self.check(now).blocked
