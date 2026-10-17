"""Stage E4: fill reconciliation — expected vs actual, in pips.

Every transmission carries an expected price (the intent's entry) and an
expected volume. The broker's deal history is the truth. This module
compares the two deterministically:

* volume: exact match within float tolerance, else a volume mismatch
* price: slippage in pips (signed — positive means the fill was WORSE for
  a BUY, better for a SELL; the sign convention is declared per side so
  reports never lie about direction)
* verdict: EXACT / WITHIN_TOLERANCE / DRIFT / VOLUME_MISMATCH

DRIFT is a flag for the monitor (D3 pattern extended to live fills): it
does not by itself halt, but it is exactly the kind of divergence the
continuous-monitoring loop watches, and repeated drift escalates to human
review. The tolerance comes from ``LiveCaps.fill_drift_tolerance_pips``.

Pure arithmetic over supplied values: no broker call, no network, no LLM.
The caller reads the deal from the terminal and hands the prices in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

__all__ = [
    "FILL_EXACT",
    "FILL_WITHIN_TOLERANCE",
    "FILL_DRIFT",
    "FILL_VOLUME_MISMATCH",
    "FillReconciliation",
    "signed_slippage_pips",
    "reconcile_fill",
]

FILL_EXACT = "EXACT"
FILL_WITHIN_TOLERANCE = "WITHIN_TOLERANCE"
FILL_DRIFT = "DRIFT"
FILL_VOLUME_MISMATCH = "VOLUME_MISMATCH"

#: Lots are floats; exact equality is wrong.
VOLUME_TOLERANCE = 1e-9


def signed_slippage_pips(
    side: str,
    expected_price: float,
    actual_price: float,
    pip_size: float,
) -> float:
    """Slippage in pips, positive = worse for the trader.

    BUY fills above expected are worse; SELL fills below expected are
    worse. Getting the sign wrong turns an adverse-fill report into a
    favorable one, so the convention lives in one tested function.
    """
    if pip_size <= 0:
        raise ValueError(f"pip_size must be positive, got {pip_size!r}")
    raw = (actual_price - expected_price) / pip_size
    direction = str(side or "").strip().upper()
    if direction in ("BUY", "LONG"):
        return raw
    if direction in ("SELL", "SHORT"):
        return -raw
    raise ValueError(f"side must be BUY or SELL, got {side!r}")


@dataclass(frozen=True)
class FillReconciliation:
    """One fill compared against its intent."""

    intent_key: str
    symbol: str
    side: str
    expected_price: float
    actual_price: float
    expected_volume: float
    actual_volume: float
    slippage_pips: float
    tolerance_pips: float
    verdict: str
    checked_at: datetime
    ticket: Optional[int] = None

    @property
    def is_exact(self) -> bool:
        return self.verdict == FILL_EXACT

    @property
    def within_tolerance(self) -> bool:
        return self.verdict in (FILL_EXACT, FILL_WITHIN_TOLERANCE)

    @property
    def is_drift(self) -> bool:
        return self.verdict in (FILL_DRIFT, FILL_VOLUME_MISMATCH)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_key": self.intent_key,
            "symbol": self.symbol,
            "side": self.side,
            "expected_price": self.expected_price,
            "actual_price": self.actual_price,
            "expected_volume": self.expected_volume,
            "actual_volume": self.actual_volume,
            "slippage_pips": round(self.slippage_pips, 4),
            "tolerance_pips": self.tolerance_pips,
            "verdict": self.verdict,
            "checked_at": self.checked_at.isoformat(),
            "ticket": self.ticket,
        }


def reconcile_fill(
    *,
    intent_key: str,
    symbol: str,
    side: str,
    expected_price: float,
    expected_volume: float,
    actual_price: float,
    actual_volume: float,
    pip_size: float,
    tolerance_pips: float,
    now: Optional[datetime] = None,
    ticket: Optional[int] = None,
) -> FillReconciliation:
    """Compare one fill against its intent; verdict is deterministic."""
    check_time = now or datetime.utcnow()
    slip = signed_slippage_pips(side, expected_price, actual_price, pip_size)

    volume_match = abs(float(actual_volume) - float(expected_volume)) <= VOLUME_TOLERANCE
    if not volume_match:
        verdict = FILL_VOLUME_MISMATCH
    elif abs(slip) == 0.0:
        verdict = FILL_EXACT
    elif abs(slip) <= float(tolerance_pips):
        verdict = FILL_WITHIN_TOLERANCE
    else:
        verdict = FILL_DRIFT

    return FillReconciliation(
        intent_key=intent_key,
        symbol=symbol,
        side=str(side).strip().upper(),
        expected_price=float(expected_price),
        actual_price=float(actual_price),
        expected_volume=float(expected_volume),
        actual_volume=float(actual_volume),
        slippage_pips=slip,
        tolerance_pips=float(tolerance_pips),
        verdict=verdict,
        checked_at=check_time,
        ticket=ticket,
    )


def summarize_reconciliations(reports: List[FillReconciliation]) -> Dict[str, Any]:
    """Aggregate verdict counts and slippage stats for the daily report."""
    if not reports:
        return {
            "count": 0,
            "verdicts": {},
            "drift_count": 0,
            "avg_slippage_pips": 0.0,
            "max_slippage_pips": 0.0,
        }
    verdicts: Dict[str, int] = {}
    for report in reports:
        verdicts[report.verdict] = verdicts.get(report.verdict, 0) + 1
    slips = [abs(r.slippage_pips) for r in reports]
    return {
        "count": len(reports),
        "verdicts": verdicts,
        "drift_count": sum(
            1 for r in reports if r.verdict in (FILL_DRIFT, FILL_VOLUME_MISMATCH)
        ),
        "avg_slippage_pips": round(sum(slips) / len(slips), 4),
        "max_slippage_pips": round(max(slips), 4),
    }
