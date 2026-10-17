"""Stage E4: recorded, time-boxed human sign-off for live execution.

The constitution allows live execution only behind an explicit human
sign-off — AUTONOMOUS(4) is granted PER scope, time-boxed, never
system-wide. This module is that record: an append-only JSONL log where
every line is one human decision (who, what, until when, in their own
words), alongside the config snapshot it was granted against.

Design rules:

* append-only — ``record`` adds lines, nothing ever rewrites or deletes;
  an expired or revoked grant is a NEW line, not an edit;
* time-boxed — every sign-off carries ``expires_at``; ``is_active`` is
  the only way to ask "is this still in force" and it fails closed;
* scoped — a sign-off for one drill does not open a window; callers
  filter by ``scope`` (drill vs window are different sign-offs).

Pure state + file I/O: no network, no broker, no LLM.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "SCOPE_DRILL",
    "SCOPE_WINDOW",
    "LiveSignoff",
    "SignoffStore",
]

#: A controlled live drill (one guarded round-trip, verification only).
SCOPE_DRILL = "e4-live-drill"
#: The 30-day limited live deployment window.
SCOPE_WINDOW = "e4-live-window"


def _parse_dt(value: Any) -> datetime:
    return datetime.fromisoformat(str(value))


@dataclass(frozen=True)
class LiveSignoff:
    """One recorded human decision authorizing (bounded) live execution."""

    approver: str
    scope: str
    pair: str
    strategy_id: str
    declaration: str
    granted_at: datetime
    expires_at: datetime
    config: Dict[str, Any] = field(default_factory=dict)
    signoff_id: str = ""

    def __post_init__(self) -> None:
        if not str(self.approver or "").strip():
            raise ValueError("signoff requires a named approver")
        if not str(self.scope or "").strip():
            raise ValueError("signoff requires a scope")
        if not str(self.declaration or "").strip():
            raise ValueError("signoff requires the human's own declaration")
        if self.expires_at <= self.granted_at:
            raise ValueError("signoff expiry must be after the grant time")
        if not self.signoff_id:
            object.__setattr__(self, "signoff_id", uuid.uuid4().hex[:12])

    def is_active(self, now: datetime) -> bool:
        """In force at ``now``: granted and not yet expired. Fails closed."""
        return self.granted_at <= now < self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signoff_id": self.signoff_id,
            "approver": self.approver,
            "scope": self.scope,
            "pair": self.pair,
            "strategy_id": self.strategy_id,
            "declaration": self.declaration,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiveSignoff":
        return cls(
            approver=str(data["approver"]),
            scope=str(data["scope"]),
            pair=str(data.get("pair", "")),
            strategy_id=str(data.get("strategy_id", "")),
            declaration=str(data["declaration"]),
            granted_at=_parse_dt(data["granted_at"]),
            expires_at=_parse_dt(data["expires_at"]),
            config=dict(data.get("config") or {}),
            signoff_id=str(data.get("signoff_id", "")),
        )


class SignoffStore:
    """Append-only JSONL log of human sign-offs, newest last."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)

    def record(self, signoff: LiveSignoff) -> LiveSignoff:
        """Append one sign-off. Returns it (with its id)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(signoff.to_dict(), ensure_ascii=True) + "\n")
        return signoff

    def records(self) -> List[LiveSignoff]:
        if not self.path.exists():
            return []
        out: List[LiveSignoff] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(LiveSignoff.from_dict(json.loads(line)))
        return out

    def latest(
        self,
        *,
        scope: Optional[str] = None,
        now: Optional[datetime] = None,
        active_only: bool = False,
    ) -> Optional[LiveSignoff]:
        """Newest matching sign-off (optionally in force at ``now``)."""
        candidates = self.records()
        if scope is not None:
            candidates = [s for s in candidates if s.scope == scope]
        if active_only:
            if now is None:
                raise ValueError("active_only requires now")
            candidates = [s for s in candidates if s.is_active(now)]
        return candidates[-1] if candidates else None
