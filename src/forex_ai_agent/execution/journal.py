"""Stage E1: write-ahead intent journal (crash-recovery / idempotency).

Every execution intent is RECORDED BEFORE it can be filled (write-ahead):
a crash between send and confirm leaves a PENDING record that restart-time
reconciliation resolves — adopt the broker-reported position, never re-send
blindly. The journal is the double-spend guard: intents are idempotent by
key, so a duplicate submission returns the original record instead of
creating a second one.

Storage is a plain serializable dict (``save_snapshot`` / ``load_snapshot``)
— the shape a persistent backend would store; E1 chaos tests use it to
simulate crashes byte-exactly. No network, no broker, no LLM: pure state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = ["IntentJournal", "INTENT_PENDING", "INTENT_EXECUTED", "INTENT_CANCELLED"]

INTENT_PENDING = "PENDING"
INTENT_EXECUTED = "EXECUTED"
INTENT_CANCELLED = "CANCELLED"


class IntentJournal:
    """Write-ahead record of execution intents (idempotent by key)."""

    def __init__(self) -> None:
        self._intents: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Write-ahead API
    # ------------------------------------------------------------------

    def record(
        self,
        key: str,
        *,
        symbol: str,
        side: str,
        volume: float,
        sl: Optional[float],
        tp: Optional[float],
        created_ts: datetime,
        created_bar: Optional[int] = None,
        comment: str = "",
    ) -> Dict[str, Any]:
        """Record an intent BEFORE execution (write-ahead).

        Idempotent: an existing record for ``key`` is returned unchanged —
        a duplicate submission never creates a second intent.
        """
        existing = self._intents.get(key)
        if existing is not None:
            return existing
        intent = {
            "key": key,
            "symbol": symbol,
            "side": side,
            "volume": float(volume),
            "sl": sl,
            "tp": tp,
            "state": INTENT_PENDING,
            "created_ts": created_ts.isoformat(),
            "created_bar": created_bar,
            "comment": comment,
            "fill": None,
            "cancel_reason": None,
        }
        self._intents[key] = intent
        return intent

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._intents.get(key)

    def mark_executed(self, key: str, fill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        intent = self._intents.get(key)
        if intent is None:
            return None
        if intent["state"] == INTENT_PENDING:
            intent["state"] = INTENT_EXECUTED
        intent["fill"] = fill
        return intent

    def mark_cancelled(self, key: str, reason: str) -> Optional[Dict[str, Any]]:
        intent = self._intents.get(key)
        if intent is None:
            return None
        if intent["state"] == INTENT_PENDING:
            intent["state"] = INTENT_CANCELLED
            intent["cancel_reason"] = reason
        return intent

    def pending(self) -> List[Dict[str, Any]]:
        return [i for i in self._intents.values() if i["state"] == INTENT_PENDING]

    def all(self) -> List[Dict[str, Any]]:
        return list(self._intents.values())

    def __len__(self) -> int:
        return len(self._intents)

    # ------------------------------------------------------------------
    # Crash-recovery snapshots
    # ------------------------------------------------------------------

    def save_snapshot(self) -> Dict[str, Any]:
        """Byte-exact serializable state (the crash image)."""
        return {"intents": {k: dict(v) for k, v in self._intents.items()}}

    def load_snapshot(self, snapshot: Dict[str, Any]) -> None:
        self._intents = {
            k: dict(v) for k, v in (snapshot.get("intents") or {}).items()
        }
