"""Stage E2: broker ↔ intent-book reconciliation.

Poll the broker's read-only positions and compare them with the gateway's
intent book (the E1 write-ahead journal). Any divergence is a *critical
event with a declared policy* — never silently adopted, never re-sent:

* a PENDING intent whose broker position matches (symbol + side + volume)
  is **adopted**: the journal records ``source=adopted_position`` and the
  intent stops being pending. Adopting what the broker already holds is
  what stops a restart from double-sending.
* a PENDING intent with no matching broker position stays pending and goes
  to **human review** — it is never re-sent automatically, because we
  cannot prove the terminal did not take it.
* a broker position with no matching intent is **unknown exposure** and
  also goes to human review — never auto-flattened, never silently merged
  into the book.

Pure state transitions over plain data: no network, no broker calls, no
LLM, no permission changes. The caller supplies positions it already read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from .base import PositionSnapshot

__all__ = [
    "RECONCILE_POLICY",
    "RECONCILE_ADOPT",
    "RECONCILE_REVIEW",
    "ReconciliationReport",
    "reconcile",
    "adopt_into",
]

# The declared policy string — recorded on every report so an auditor can
# see which rule was in force without reading code.
RECONCILE_POLICY = "adopt_matching_review_unmatched_never_resend"
RECONCILE_ADOPT = "ADOPT"
RECONCILE_REVIEW = "HUMAN_REVIEW"

# Broker positions are floats (lots); exact equality is wrong.
VOLUME_TOLERANCE = 1e-9

PositionLike = Union[PositionSnapshot, Dict[str, Any]]


def _normalize_side(side: Any) -> str:
    text = str(side or "").strip().upper()
    if text in ("LONG", "BUY", "POSITION_TYPE_BUY"):
        return "BUY"
    if text in ("SHORT", "SELL", "POSITION_TYPE_SELL"):
        return "SELL"
    return text


def _as_dict(position: PositionLike) -> Dict[str, Any]:
    if isinstance(position, PositionSnapshot):
        data = position.to_dict()
    elif hasattr(position, "to_dict"):
        data = position.to_dict()
    else:
        data = dict(position)
    return {
        "ticket": data.get("ticket"),
        "symbol": data.get("symbol"),
        "side": _normalize_side(data.get("side")),
        "volume": float(data.get("volume") or 0.0),
        "profit": float(data.get("profit") or 0.0),
        "magic": int(data.get("magic") or 0),
        "sl": data.get("sl"),
        "tp": data.get("tp"),
        "price_open": data.get("price_open"),
    }


@dataclass
class ReconciliationReport:
    """Result of one poll-and-compare cycle."""

    ts: datetime
    policy: str = RECONCILE_POLICY
    adopted: List[str] = field(default_factory=list)
    review: List[str] = field(default_factory=list)
    broker_only: List[Dict[str, Any]] = field(default_factory=list)
    matched_positions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def divergence(self) -> bool:
        """True when anything needs a human: unmatched intent OR unknown position."""
        return bool(self.review or self.broker_only)

    @property
    def ok(self) -> bool:
        """True when local book and broker book agree exactly."""
        return not self.divergence

    @property
    def events(self) -> List[Dict[str, Any]]:
        """Deterministic event stream (the audit trail for this cycle)."""
        out: List[Dict[str, Any]] = [
            {
                "kind": "reconciliation.adopted",
                "key": key,
                "policy": self.policy,
                "ts": self.ts.isoformat(),
            }
            for key in self.adopted
        ]
        out.extend(
            {
                "kind": "reconciliation.review",
                "key": key,
                "policy": self.policy,
                "ts": self.ts.isoformat(),
            }
            for key in self.review
        )
        out.extend(
            {
                "kind": "reconciliation.divergence",
                "policy": self.policy,
                "reason": "broker_position_without_intent",
                "position": position,
                "ts": self.ts.isoformat(),
            }
            for position in self.broker_only
        )
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts.isoformat(),
            "policy": self.policy,
            "adopted": list(self.adopted),
            "review": list(self.review),
            "broker_only": [dict(p) for p in self.broker_only],
            "divergence": self.divergence,
            "ok": self.ok,
        }


def reconcile(
    intents: Iterable[Dict[str, Any]],
    positions: Sequence[PositionLike],
    *,
    ts: Optional[datetime] = None,
    tolerance: float = VOLUME_TOLERANCE,
) -> ReconciliationReport:
    """Compare the intent book against the broker's open positions.

    ``intents`` are journal records (``state == PENDING`` is what a caller
    normally passes); ``positions`` are what the adapter just read. The
    function mutates nothing: the caller decides whether to apply the
    report via :func:`adopt_into`.
    """
    report = ReconciliationReport(ts=ts or datetime.utcnow(), policy=RECONCILE_POLICY)
    remaining: List[Dict[str, Any]] = [_as_dict(p) for p in positions]

    for intent in intents:
        key = str(intent.get("key") or intent.get("id") or "")
        side = _normalize_side(intent.get("side"))
        symbol = str(intent.get("symbol") or "")
        volume = float(intent.get("volume") or 0.0)

        match_index = None
        for index, position in enumerate(remaining):
            if position["symbol"] != symbol:
                continue
            if position["side"] != side:
                continue
            if abs(position["volume"] - volume) > tolerance:
                continue
            match_index = index
            break

        if match_index is None:
            # Local intent, no broker position: we cannot prove the order
            # was not taken. Hold as pending, flag for a human, re-send
            # nothing.
            report.review.append(key)
            continue

        matched = remaining.pop(match_index)
        report.adopted.append(key)
        report.matched_positions.append(dict(matched))

    # Whatever is left is exposure the broker reports that we cannot explain.
    report.broker_only.extend(dict(p) for p in remaining)
    return report


def adopt_into(report: ReconciliationReport, governor: Any) -> List[str]:
    """Apply a report's ADOPT decisions to an E1 governor's journal.

    Only adopted intents are touched; review items are left PENDING so a
    human sees them on the next poll. Returns the keys that changed.
    """
    applied: List[str] = []
    for key in report.adopted:
        record = governor.journal.get(key)
        if record is None:
            continue
        governor.record_fill(key, {"price": None, "source": "adopted_position"})
        applied.append(key)
    return applied
