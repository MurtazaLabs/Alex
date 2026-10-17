"""Stage D3: continuous, deterministic monitoring wired to probation.

Closes the last open item of the Stage D accept line: the paper-trading
layer's numbers are now continuously watched — accrual progress vs quota and
window, current standing vs the row's own gate floors, the health of the
Stage D2 accrual infrastructure (event-stream freshness / replay fallback)
and feed integrity — and surfaced three ways: structured log lines, a
``probation.monitor`` event on the bus whenever a version's flag set CHANGES
(same event-driven pattern as the D2 bar pump), and a read-only API/tool
snapshot for humans and the agent.

Constitution compliance:

- **Deterministic** — every flag, projection and note is arithmetic over
  stored probation records; no LLM anywhere in this path.
- **Observe-only** — the monitor never mutates probation state: it never
  calls ``update``/``approve``/``reject`` (pinned by an exploding-store
  test). The verdict stays the store's gate evaluation; approve/reject stay
  the human-only buttons.
- **No synthetic data** — a degraded (mock) feed is itself a monitoring
  flag, never silently ignored.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ..core.event_bus import Event, EventBus, InMemoryEventBus
from ..data.providers.factory import get_feed_chain
from .probation import ProbationStore

logger = logging.getLogger(__name__)

#: Event type published when a probation's monitoring flag set changes.
PROBATION_MONITOR_EVENT = "probation.monitor"

#: Rate floors are only judged once there is a minimal sample (2 losing
#: trades out of 2 is noise, not a verdict).
_MIN_SAMPLE_FOR_RATE_FLOORS = 5
#: Projected trades < 85% of quota by window end → quota at risk.
_QUOTA_WARNING_FRACTION = 0.85
#: ≥80% of the window elapsed with quota unmet → window ending.
_WINDOW_WARNING_FRACTION = 0.8
#: Mirrors probation_runner._EVENT_OWNERSHIP_STALENESS.
_STREAM_STALENESS = timedelta(minutes=45)
#: A probation older than this with NO accrual stamp from either path.
_NO_ACCRUAL_AFTER = timedelta(hours=2)


def _parse_iso(raw: Any) -> Optional[datetime]:
    if isinstance(raw, str) and raw:
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


class ProbationMonitor:
    """Assesses every active probation and reports deterministic health."""

    def __init__(
        self,
        store: Optional[ProbationStore] = None,
        event_bus: Optional[EventBus] = None,
        *,
        publish: bool = True,
    ) -> None:
        self.store = store or ProbationStore()
        self._bus = event_bus or InMemoryEventBus()
        self._publish_default = publish
        self._last_flags: Dict[str, frozenset] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def monitor_once(self, *, publish: Optional[bool] = None) -> List[Dict[str, Any]]:
        """One monitoring pass over every ACTIVE probation.

        ``publish=False`` (read-only snapshots — API endpoint, tool) still
        computes the full report but never emits bus events.
        """
        do_publish = self._publish_default if publish is None else publish
        reports: List[Dict[str, Any]] = []
        for prob in await self.store.list(status="active"):
            report = self.assess(prob)
            reports.append(report)
            if do_publish:
                await self._emit_if_changed(prob.version_id, report)
        return reports

    def assess(self, prob: Any) -> Dict[str, Any]:
        """Deterministic health report for one probation row.

        Works off stored records only (``ProbationStatus``); projections are
        plain arithmetic — no LLM, no market calls, no state changes.
        """
        flags = self._flags(prob)

        observed = int(getattr(prob, "observed_trades", 0) or 0)
        required = int(getattr(prob, "required_trades", 0) or 0)
        days_elapsed = self._days_elapsed(prob)
        required_days = int(getattr(prob, "required_days", 0) or 0)
        win_rate = getattr(prob, "win_rate", None)
        profit_factor = getattr(prob, "profit_factor", None)
        worst_loss = getattr(prob, "worst_daily_loss_pct", None)
        metrics = getattr(prob, "last_metrics", None) or {}
        mode = metrics.get("_accrual_mode")
        event_age = self._event_age_minutes(metrics)

        rate = round(observed / days_elapsed, 3) if days_elapsed >= 1 else None
        projected = round(rate * required_days, 1) if rate is not None else None

        status = "critical" if flags & _CRITICAL_FLAGS else ("warn" if flags else "ok")
        return {
            "version_id": prob.version_id,
            "strategy_name": prob.strategy_name,
            "pair": prob.pair,
            "monitor_status": status,
            "flags": sorted(flags),
            "observed_trades": observed,
            "required_trades": required,
            "days_elapsed": days_elapsed,
            "required_days": required_days,
            "accrual_rate_per_day": rate,
            "projected_trades": projected,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "worst_daily_loss_pct": worst_loss,
            "accrual_mode": mode,
            "event_age_minutes": event_age,
            "notes": self._notes(
                observed, required, days_elapsed, required_days,
                win_rate, profit_factor, worst_loss, flags,
            ),
        }

    @staticmethod
    def summarize(reports: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {"active": len(reports), "ok": 0, "warn": 0, "critical": 0}
        for r in reports:
            counts[r["monitor_status"]] = counts.get(r["monitor_status"], 0) + 1
        return counts

    @staticmethod
    def render_reports(reports: List[Dict[str, Any]]) -> str:
        """Code-rendered one-liner per probation (never LLM prose)."""
        lines = []
        for r in reports:
            lines.append(
                f"{r['version_id']} ({r['strategy_name']}, {r['pair']}): "
                f"{r['monitor_status'].upper()} — {r['notes']}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Flags (all deterministic)
    # ------------------------------------------------------------------

    def _flags(self, prob: Any) -> set:
        flags: set = set()
        now = datetime.utcnow()

        observed = int(getattr(prob, "observed_trades", 0) or 0)
        required = int(getattr(prob, "required_trades", 0) or 0)
        required_days = int(getattr(prob, "required_days", 0) or 0)
        days_elapsed = self._days_elapsed(prob)
        started = self._started_at(prob)

        # --- accrual progress vs quota / window -------------------------
        if days_elapsed >= 1 and required > 0:
            projected = (observed / days_elapsed) * required_days
            if projected < required * _QUOTA_WARNING_FRACTION:
                flags.add("quota_at_risk")
        if (
            required_days > 0
            and days_elapsed >= _WINDOW_WARNING_FRACTION * required_days
            and observed < required
        ):
            flags.add("window_ending")

        # --- current standing vs the row's own gate floors --------------
        min_wr = float(getattr(prob, "min_win_rate", 0.30) or 0.0)
        min_pf = float(getattr(prob, "min_profit_factor", 0.95) or 0.0)
        max_loss = float(getattr(prob, "max_daily_loss_pct", 0.03) or 0.0)
        win_rate = getattr(prob, "win_rate", None)
        profit_factor = getattr(prob, "profit_factor", None)
        worst_loss = getattr(prob, "worst_daily_loss_pct", None)
        if observed >= _MIN_SAMPLE_FOR_RATE_FLOORS:
            if win_rate is not None and win_rate < min_wr:
                flags.add("win_rate_below_floor")
            if profit_factor is not None and profit_factor < min_pf:
                flags.add("profit_factor_below_floor")
        if worst_loss is not None and max_loss > 0 and worst_loss > max_loss:
            flags.add("worst_daily_loss_breach")

        # --- Stage D2 accrual infrastructure health ---------------------
        metrics = getattr(prob, "last_metrics", None) or {}
        mode = metrics.get("_accrual_mode")
        if mode == "event":
            through = _parse_iso(metrics.get("_event_through"))
            if through is None or now - through > _STREAM_STALENESS:
                flags.add("stream_stale")
        elif mode == "replay":
            replayed_at = _parse_iso(metrics.get("_replay_at"))
            if replayed_at is None or now - replayed_at > _STREAM_STALENESS:
                flags.add("accrual_stale")
        elif started is not None and now - started > _NO_ACCRUAL_AFTER:
            flags.add("no_accrual_yet")

        # --- feed integrity ---------------------------------------------
        try:
            if get_feed_chain().active_source == "mock":
                flags.add("mock_feed")
        except Exception:  # noqa: BLE001 — monitoring must never break on the feed seam
            pass
        return flags

    # ------------------------------------------------------------------
    # Emission (only on flag-set change)
    # ------------------------------------------------------------------

    async def _emit_if_changed(self, version_id: str, report: Dict[str, Any]) -> None:
        current = frozenset(report["flags"])
        if self._last_flags.get(version_id) == current:
            return
        first = version_id not in self._last_flags
        self._last_flags[version_id] = current
        try:
            await self._bus.publish(
                Event(
                    event_type=PROBATION_MONITOR_EVENT,
                    aggregate_id=version_id,
                    aggregate_type="probation",
                    payload=dict(report),
                    source_agent="probation_monitor",
                )
            )
        except Exception as exc:  # noqa: BLE001 — monitoring must never break the caller
            logger.warning("probation monitor emit failed for %s: %s", version_id, exc)
            return
        level = logger.warning if report["monitor_status"] == "critical" else logger.info
        level(
            "probation_monitor%s version=%s status=%s flags=%s",
            "_first" if first else "_change",
            version_id,
            report["monitor_status"],
            ",".join(report["flags"]) or "-",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _started_at(prob: Any) -> Optional[datetime]:
        raw = getattr(prob, "started_at", None)
        if isinstance(raw, datetime):
            return raw
        parsed = _parse_iso(raw)
        if parsed is not None:
            return parsed
        # ProbationStatus.started_at is an ISO string truncated to seconds.
        try:
            return datetime.fromisoformat(str(raw)[:19])
        except (TypeError, ValueError):
            return None

    def _days_elapsed(self, prob: Any) -> int:
        started = self._started_at(prob)
        if started is None:
            return int(getattr(prob, "days_elapsed", 0) or 0)
        return max((datetime.utcnow() - started).days, 0)

    @staticmethod
    def _event_age_minutes(metrics: Dict[str, Any]) -> Optional[float]:
        through = _parse_iso(metrics.get("_event_through"))
        if through is None:
            return None
        return round((datetime.utcnow() - through).total_seconds() / 60.0, 1)

    @staticmethod
    def _notes(
        observed: int,
        required: int,
        days_elapsed: int,
        required_days: int,
        win_rate: Optional[float],
        profit_factor: Optional[float],
        worst_loss: Optional[float],
        flags: set,
    ) -> str:
        """Deterministic one-line summary (code-rendered, never LLM)."""
        parts = [f"{observed}/{required} trades", f"{days_elapsed}/{required_days} days"]
        if win_rate is not None:
            parts.append(f"WR {win_rate:.0%}")
        if profit_factor is not None:
            parts.append(f"PF {profit_factor:.2f}")
        if worst_loss is not None:
            parts.append(f"worst day {worst_loss:.1%}")
        if flags:
            parts.append("flags: " + ",".join(sorted(flags)))
        return "; ".join(parts)


_CRITICAL_FLAGS = frozenset(
    {
        "mock_feed",
        "worst_daily_loss_breach",
        "win_rate_below_floor",
        "profit_factor_below_floor",
    }
)
