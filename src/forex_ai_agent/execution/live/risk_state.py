"""Stage E4: mandated risk baselines + trip persistence.

The mandate (ALEX_TRADING_MANDATE.md v1.0) halts NEW trades on:

- daily loss  >= 1% of the day's starting equity (resumes next UTC day),
- weekly loss >= 2% of the week's starting equity (resumes next ISO week),
- drawdown    >= 5% from the equity high-water mark (halt until a human
  explicitly clears it -- never a timeout).

This module persists exactly the baselines needed to evaluate those rules
across runner restarts: the day/week start equities, the equity
high-water mark, and the trips themselves. Pure file state: no network,
no broker, no LLM.

Fail-closed rules:
- a corrupt/unreadable state file raises (trading must not resume on
  silently reset baselines);
- ``clear_trip`` is the ONLY way a drawdown trip ends, and it REBASES the
  affected baseline to the current equity (the human resets the ruler).

Writes are atomic (temp file + os.replace). Trips and clears are also
appended to ``risk_events.jsonl`` next to the state file for audit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "BaselineSnapshot",
    "RiskBaselineStore",
    "RISK_TRIP_NAMES",
    "iso_week_key",
]

RISK_TRIP_NAMES = ("daily", "weekly", "drawdown")


def iso_week_key(now: datetime) -> str:
    """ISO week key, e.g. ``2026-W43`` (weeks start Monday)."""
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@dataclass(frozen=True)
class BaselineSnapshot:
    """Point-in-time view of the persisted risk baselines."""

    day_key: str
    week_key: str
    day_start_equity: float
    week_start_equity: float
    equity_hwm: float
    trips: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def tripped(self, name: str) -> bool:
        return name in self.trips

    def active_trips(self) -> List[str]:
        return [name for name in RISK_TRIP_NAMES if name in self.trips]

    def daily_loss_pct(self, equity: float) -> float:
        if self.day_start_equity <= 0:
            return 0.0
        return (
            (self.day_start_equity - equity) / self.day_start_equity
        ) * 100.0

    def weekly_loss_pct(self, equity: float) -> float:
        if self.week_start_equity <= 0:
            return 0.0
        return (
            (self.week_start_equity - equity) / self.week_start_equity
        ) * 100.0

    def drawdown_pct(self, equity: float) -> float:
        if self.equity_hwm <= 0:
            return 0.0
        return ((self.equity_hwm - equity) / self.equity_hwm) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day_key": self.day_key,
            "week_key": self.week_key,
            "day_start_equity": round(self.day_start_equity, 2),
            "week_start_equity": round(self.week_start_equity, 2),
            "equity_hwm": round(self.equity_hwm, 2),
            "trips": dict(self.trips),
        }


class RiskBaselineStore:
    """Persisted day/week/ high-water baselines + mandate trips."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.events_path = self.path.parent / "risk_events.jsonl"
        self._data: Dict[str, Any] = self._load()

    # ------------------------------------------------------------------
    # File IO
    # ------------------------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"risk baseline file unreadable ({self.path}): {exc} -- "
                "refusing to trade on unknown baselines"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"risk baseline file malformed ({self.path}) -- refusing "
                "to trade on unknown baselines"
            )
        return data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    def _append_event(self, record: Dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(record, ensure_ascii=True, default=str) + "\n"
            )

    # ------------------------------------------------------------------
    # Sync / snapshot
    # ------------------------------------------------------------------

    def sync(self, equity: float, now: datetime) -> BaselineSnapshot:
        """Roll day/week baselines, track the HWM, expire stale trips.

        Returns the fresh snapshot. Daily/weekly trips expire with their
        own key rollover; a drawdown trip persists until ``clear_trip``.
        """
        equity = float(equity)
        day_key = now.strftime("%Y-%m-%d")
        week_key = iso_week_key(now)
        data = self._data
        trips = data.setdefault("trips", {})
        changed = False

        if (
            data.get("day_key") != day_key
            or data.get("day_start_equity") is None
        ):
            data["day_key"] = day_key
            data["day_start_equity"] = equity
            if "daily" in trips:
                trips.pop("daily", None)
            changed = True

        if (
            data.get("week_key") != week_key
            or data.get("week_start_equity") is None
        ):
            data["week_key"] = week_key
            data["week_start_equity"] = equity
            if "weekly" in trips:
                trips.pop("weekly", None)
            changed = True

        if data.get("equity_hwm") is None:
            data["equity_hwm"] = equity
            changed = True
        elif equity > float(data["equity_hwm"]):
            data["equity_hwm"] = equity
            changed = True

        if changed:
            self._save()
        return self.snapshot()

    def snapshot(self) -> BaselineSnapshot:
        """Current baselines (zeros when never synced)."""
        data = self._data
        return BaselineSnapshot(
            day_key=str(data.get("day_key", "")),
            week_key=str(data.get("week_key", "")),
            day_start_equity=float(data.get("day_start_equity") or 0.0),
            week_start_equity=float(data.get("week_start_equity") or 0.0),
            equity_hwm=float(data.get("equity_hwm") or 0.0),
            trips=dict(data.get("trips") or {}),
        )

    # ------------------------------------------------------------------
    # Trips
    # ------------------------------------------------------------------

    def trip(
        self,
        name: str,
        *,
        now: Optional[datetime] = None,
        detail: Optional[Dict[str, Any]] = None,
    ) -> BaselineSnapshot:
        """Persist a mandate trip (idempotent per name)."""
        if name not in RISK_TRIP_NAMES:
            raise ValueError(
                f"unknown trip {name!r}; expected one of {RISK_TRIP_NAMES}"
            )
        at = (now or datetime.utcnow()).isoformat()
        trips = self._data.setdefault("trips", {})
        if name not in trips:
            trips[name] = {"tripped_at": at, "detail": dict(detail or {})}
            self._save()
            self._append_event({
                "event": "risk_tripped",
                "trip": name,
                "at": at,
                "detail": dict(detail or {}),
            })
        return self.snapshot()

    def clear_trip(
        self,
        name: str,
        *,
        now: Optional[datetime] = None,
        by: str = "operator",
        reason: str = "",
    ) -> BaselineSnapshot:
        """HUMAN action: clear a trip and rebase its ruler.

        The affected baseline is reset to None so the next sync re-roots
        it at the then-current equity (day/week start or high-water mark).
        """
        if name not in RISK_TRIP_NAMES:
            raise ValueError(
                f"unknown trip {name!r}; expected one of {RISK_TRIP_NAMES}"
            )
        at = (now or datetime.utcnow()).isoformat()
        trips = self._data.setdefault("trips", {})
        had = trips.pop(name, None)
        if name == "daily":
            self._data["day_start_equity"] = None
        elif name == "weekly":
            self._data["week_start_equity"] = None
        elif name == "drawdown":
            self._data["equity_hwm"] = None
        self._save()
        self._append_event({
            "event": "trip_cleared",
            "trip": name,
            "had_trip": had is not None,
            "by": by,
            "reason": reason,
            "at": at,
        })
        return self.snapshot()
