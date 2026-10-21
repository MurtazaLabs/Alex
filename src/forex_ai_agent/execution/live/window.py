"""Stage E4: the 30-day limited live window -- eligibility, state, lifecycle.

The window is the last gate before the executor-loop phase. It only
exists when ALL of these are true, checked fresh at start and re-checked
by the runner every cycle (fail closed):

1. ``version_pinned``  -- the version file is active and its rules+
   parameters still hash to the spec hash a human approved;
2. ``probation_approved`` -- the paper-probation row for that version
   reached the human ``approved`` state (``passed`` alone is not enough);
3. ``drill_verified``  -- an ``e4_live_verify`` evidence file shows a
   PASSED in-session order drill (order_phase=executed, >=1 wire attempt);
4. ``window_signoff``  -- an active, time-boxed SCOPE_WINDOW human
   sign-off exists for this pair and strategy.

Pure state + file I/O plus one read of the probation DB: no network, no
broker, no LLM. Nothing here can send anything.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from .signoff import SCOPE_WINDOW, LiveSignoff, SignoffStore
from ...strategy.versioning import spec_hash

__all__ = [
    "E4_WINDOW_DAYS",
    "DEFAULT_VERSIONS_DIR",
    "DEFAULT_SIGNOFF_PATH",
    "DEFAULT_DRILL_DIR",
    "DEFAULT_WINDOW_STATE_DIR",
    "CheckResult",
    "WindowEligibility",
    "WindowState",
    "WindowStore",
    "read_version_file",
    "check_eligibility",
    "start_window",
    "stop_window",
    "window_status",
]

#: The limited live deployment window length, fixed by the E4 plan.
E4_WINDOW_DAYS = 30

DEFAULT_VERSIONS_DIR = "strategies/versions"
DEFAULT_SIGNOFF_PATH = "logs/live_drill/signoffs.jsonl"
DEFAULT_DRILL_DIR = "logs/live_drill"
DEFAULT_WINDOW_STATE_DIR = "logs/live_window"


@dataclass(frozen=True)
class CheckResult:
    """One eligibility check and the exact reason it passed or failed."""

    name: str
    ok: bool
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class WindowEligibility:
    """The full set of fresh eligibility checks, in evaluation order."""

    checks: List[CheckResult] = field(default_factory=list)

    @property
    def all_met(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_met": self.all_met,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass
class WindowState:
    """The persisted state of the (single) live window."""

    window_id: str
    pair: str
    version_id: str
    pinned_hash: str
    signoff_id: str
    started_at: datetime
    ends_at: datetime
    status: str = "active"  # active | stopped | expired
    stopped_at: Optional[datetime] = None
    stop_reason: str = ""

    def is_active(self, now: datetime) -> bool:
        return self.status == "active" and self.started_at <= now < self.ends_at

    def remaining_days(self, now: datetime) -> float:
        return max(0.0, (self.ends_at - now).total_seconds() / 86400.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "pair": self.pair,
            "version_id": self.version_id,
            "pinned_hash": self.pinned_hash,
            "signoff_id": self.signoff_id,
            "started_at": self.started_at.isoformat(),
            "ends_at": self.ends_at.isoformat(),
            "status": self.status,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowState":
        return cls(
            window_id=str(data["window_id"]),
            pair=str(data.get("pair", "")),
            version_id=str(data.get("version_id", "")),
            pinned_hash=str(data.get("pinned_hash", "")),
            signoff_id=str(data.get("signoff_id", "")),
            started_at=datetime.fromisoformat(str(data["started_at"])),
            ends_at=datetime.fromisoformat(str(data["ends_at"])),
            status=str(data.get("status", "active")),
            stopped_at=(
                datetime.fromisoformat(str(data["stopped_at"]))
                if data.get("stopped_at") else None
            ),
            stop_reason=str(data.get("stop_reason", "")),
        )


class WindowStore:
    """Snapshot state file + append-only events log for the window."""

    def __init__(self, state_dir: Union[str, Path]) -> None:
        self.state_dir = Path(state_dir)
        self.state_path = self.state_dir / "window_state.json"
        self.events_path = self.state_dir / "window_events.jsonl"

    def load_state(self) -> Optional[WindowState]:
        if not self.state_path.exists():
            return None
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            return WindowState.from_dict(data)
        except Exception:
            return None

    def save_state(self, state: WindowState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(state.to_dict(), indent=2), encoding="utf-8",
        )

    def append_event(self, event: Dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        payload = dict(event)
        payload.setdefault("at", datetime.utcnow().isoformat())
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def events(self) -> List[Dict[str, Any]]:
        if not self.events_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
        return out


def read_version_file(
    versions_dir: Union[str, Path], version_id: str
) -> Optional[Dict[str, Any]]:
    """Read one version JSON straight from disk (no manager side effects)."""
    path = Path(versions_dir) / f"{version_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _check_version_pin(
    version_id: str, versions_dir: Union[str, Path]
) -> Tuple[CheckResult, Optional[str]]:
    """Verify the version is active and its spec still matches the pin."""
    data = read_version_file(versions_dir, version_id)
    if data is None:
        return CheckResult(
            "version_pinned", False, f"version file {version_id}.json not found",
        ), None
    approval = data.get("approval") or {}
    pinned = str(approval.get("spec_hash", ""))
    current = spec_hash(data.get("rules") or {}, data.get("parameters") or {})
    status = str(data.get("status", ""))
    if not pinned:
        return CheckResult(
            "version_pinned", False, "no human approval hash on the version",
        ), None
    if status != "active":
        return CheckResult(
            "version_pinned", False,
            f"version status is '{status}', not 'active'",
        ), None
    if pinned != current:
        return CheckResult(
            "version_pinned", False,
            f"spec changed since approval (pinned {pinned[:12]} != now {current[:12]})",
        ), None
    return CheckResult(
        "version_pinned", True,
        f"hash {pinned[:12]} pinned by {approval.get('approved_by', '?')} "
        f"at {approval.get('approved_at', '?')}",
    ), pinned


async def _check_probation(
    version_id: str, probation_store: Any = None
) -> CheckResult:
    try:
        if probation_store is None:
            from ...autonomy.probation import ProbationStore

            probation_store = ProbationStore()
        row = await probation_store.get(version_id)
        if row is None:
            return CheckResult(
                "probation_approved", False,
                f"no probation record for version {version_id}",
            )
        progress = (
            f"{row.observed_trades}/{row.required_trades} trades, "
            f"{row.observed_days}/{row.required_days} days"
        )
        if row.status == "approved":
            return CheckResult(
                "probation_approved", True, f"human-approved ({progress})",
            )
        if row.status == "passed":
            return CheckResult(
                "probation_approved", False,
                f"probation passed -- awaiting human approval ({progress})",
            )
        return CheckResult(
            "probation_approved", False,
            f"probation status '{row.status}' ({progress})",
        )
    except Exception as exc:
        return CheckResult(
            "probation_approved", False,
            f"check error: {type(exc).__name__}: {exc}",
        )


def _check_drill(drill_dir: Union[str, Path]) -> CheckResult:
    try:
        candidates = sorted(Path(drill_dir).glob("e4_live_verify_*.json"))
        if not candidates:
            return CheckResult(
                "drill_verified", False, "no e4_live_verify evidence found",
            )
        path = candidates[-1]
        data = json.loads(path.read_text(encoding="utf-8"))
        result = str(data.get("result", "?"))
        phase = str(data.get("order_phase", "?"))
        attempts = int(data.get("wire_attempts", 0) or 0)
        detail = (
            f"{path.name}: result={result} order_phase={phase} "
            f"wire_attempts={attempts}"
        )
        ok = result == "PASS" and phase == "executed" and attempts >= 1
        return CheckResult("drill_verified", ok, detail)
    except Exception as exc:
        return CheckResult(
            "drill_verified", False, f"check error: {type(exc).__name__}: {exc}",
        )


def _active_window_signoff(
    signoff_path: Union[str, Path], pair: str, version_id: str,
    now: datetime,
) -> Optional[LiveSignoff]:
    signoff = SignoffStore(signoff_path).latest(
        scope=SCOPE_WINDOW, now=now, active_only=True,
    )
    if signoff is None:
        return None
    if signoff.pair and signoff.pair != pair:
        return None
    if signoff.strategy_id and signoff.strategy_id != version_id:
        return None
    return signoff


def _check_signoff(
    signoff_path: Union[str, Path], pair: str, version_id: str,
    now: datetime,
) -> CheckResult:
    try:
        signoff = _active_window_signoff(signoff_path, pair, version_id, now)
        if signoff is None:
            return CheckResult(
                "window_signoff", False,
                "no active SCOPE_WINDOW sign-off for this pair/strategy",
            )
        return CheckResult(
            "window_signoff", True,
            f"{signoff.signoff_id} by {signoff.approver} "
            f"(expires {signoff.expires_at.isoformat()})",
        )
    except Exception as exc:
        return CheckResult(
            "window_signoff", False,
            f"check error: {type(exc).__name__}: {exc}",
        )


async def check_eligibility(
    *,
    version_id: str,
    pair: str = "EUR/USD",
    versions_dir: Union[str, Path] = DEFAULT_VERSIONS_DIR,
    signoff_path: Union[str, Path] = DEFAULT_SIGNOFF_PATH,
    drill_dir: Union[str, Path] = DEFAULT_DRILL_DIR,
    probation_store: Any = None,
    now: Optional[datetime] = None,
) -> WindowEligibility:
    """Run all four window eligibility checks, fresh, fail closed."""
    check_time = now or datetime.utcnow()
    pin, _hash = _check_version_pin(version_id, versions_dir)
    probation = await _check_probation(version_id, probation_store)
    drill = _check_drill(drill_dir)
    signoff = _check_signoff(signoff_path, pair, version_id, check_time)
    return WindowEligibility(checks=[pin, probation, drill, signoff])


async def start_window(
    *,
    version_id: str,
    pair: str = "EUR/USD",
    state_dir: Union[str, Path] = DEFAULT_WINDOW_STATE_DIR,
    versions_dir: Union[str, Path] = DEFAULT_VERSIONS_DIR,
    signoff_path: Union[str, Path] = DEFAULT_SIGNOFF_PATH,
    drill_dir: Union[str, Path] = DEFAULT_DRILL_DIR,
    probation_store: Any = None,
    now: Optional[datetime] = None,
) -> Tuple[bool, Any]:
    """Start the window iff every eligibility check passes, right now.

    Returns ``(True, WindowState)`` on success, or ``(False, eligibility)``
    (a ``WindowEligibility``) with the exact failing reasons. Nothing is
    written on refusal.
    """
    check_time = now or datetime.utcnow()
    store = WindowStore(state_dir)
    existing = store.load_state()
    if existing is not None and existing.is_active(check_time):
        return False, WindowEligibility(checks=[CheckResult(
            "no_active_window", False,
            f"window {existing.window_id} already active until "
            f"{existing.ends_at.isoformat()}",
        )])

    eligibility = await check_eligibility(
        version_id=version_id, pair=pair,
        versions_dir=versions_dir, signoff_path=signoff_path,
        drill_dir=drill_dir, probation_store=probation_store,
        now=check_time,
    )
    if not eligibility.all_met:
        return False, eligibility

    _pin, pinned_hash = _check_version_pin(version_id, versions_dir)
    signoff = _active_window_signoff(signoff_path, pair, version_id, check_time)
    if pinned_hash is None or signoff is None:  # pragma: no cover - raced
        return False, WindowEligibility(checks=[CheckResult(
            "race_check", False, "pin or sign-off vanished between checks",
        )])

    state = WindowState(
        window_id=uuid4().hex[:12],
        pair=pair,
        version_id=version_id,
        pinned_hash=pinned_hash,
        signoff_id=signoff.signoff_id,
        started_at=check_time,
        ends_at=check_time + timedelta(days=E4_WINDOW_DAYS),
    )
    store.save_state(state)
    store.append_event({"event": "window_started", **state.to_dict()})
    return True, state


def stop_window(
    *,
    state_dir: Union[str, Path] = DEFAULT_WINDOW_STATE_DIR,
    reason: str,
    now: Optional[datetime] = None,
) -> Optional[WindowState]:
    """Stop the window (human or safety action). Idempotent."""
    check_time = now or datetime.utcnow()
    store = WindowStore(state_dir)
    state = store.load_state()
    if state is None:
        return None
    if state.status != "active":
        return state
    state.status = "stopped"
    state.stopped_at = check_time
    state.stop_reason = str(reason or "stopped")
    store.save_state(state)
    store.append_event({
        "event": "window_stopped",
        "window_id": state.window_id,
        "reason": state.stop_reason,
    })
    return state


def window_status(
    *,
    state_dir: Union[str, Path] = DEFAULT_WINDOW_STATE_DIR,
    signoff_path: Union[str, Path] = DEFAULT_SIGNOFF_PATH,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Read-only status snapshot for scripts and the API (no async work)."""
    check_time = now or datetime.utcnow()
    store = WindowStore(state_dir)
    state = store.load_state()
    signoff = SignoffStore(signoff_path).latest(
        scope=SCOPE_WINDOW, now=check_time, active_only=False,
    )
    out: Dict[str, Any] = {
        "state": state.to_dict() if state else None,
        "active": bool(state and state.is_active(check_time)),
        "remaining_days": round(state.remaining_days(check_time), 2) if state else 0.0,
        "window_days": E4_WINDOW_DAYS,
        "signoff": signoff.to_dict() if signoff else None,
        "signoff_active": bool(signoff and signoff.is_active(check_time)),
        "runner": "scripts/e4_window_run.py",
    }
    return out
