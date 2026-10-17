"""Paper-trading probation: the arena between backtest gate and human approval.

A validated strategy version enters ``active`` probation. Closed paper trades
produced while that version is active are fed to :meth:`ProbationStore.update`;
when enough trades/days accumulate the store applies the objective gates:

    passed   → eligible for the human approval queue
    failed   → rejected (bad win rate / PF / daily loss)
    expired  → not enough trades in the window (weak signal frequency)

Approval/rejection themselves are the human's buttons (CLI/dashboard) —
never this store's decision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import select

from ..data.database import get_session_factory
from ..data.models.autonomy import StrategyProbation

logger = logging.getLogger(__name__)


@dataclass
class ProbationStatus:
    """One probation record + computed progress for UI display."""

    version_id: str
    strategy_name: str
    pair: str
    status: str
    observed_trades: int
    required_trades: int
    observed_days: int
    required_days: int
    days_elapsed: int
    win_rate: Optional[float]
    profit_factor: Optional[float]
    worst_daily_loss_pct: Optional[float]
    last_metrics: Optional[Dict[str, Any]]
    eligible_for_approval: bool
    notes: str
    started_at: str
    completed_at: Optional[str]
    # Gate floors this row is judged against (Stage D3: the monitor needs
    # them to report verdict standing without re-reading the DB row).
    min_win_rate: float = 0.30
    min_profit_factor: float = 0.95
    max_daily_loss_pct: float = 0.03

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class ProbationStore:
    """CRUD + state machine over ``strategy_probations`` (async SQLAlchemy)."""

    def __init__(self, session_factory: Optional[Any] = None) -> None:
        self._session_factory = session_factory or get_session_factory()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        version_id: str,
        strategy_name: str,
        pair: str,
        spec_dict: Dict[str, Any],
        required_trades: int = 20,
        required_days: int = 14,
        min_win_rate: float = 0.30,
        min_profit_factor: float = 0.95,
        max_daily_loss_pct: float = 0.03,
    ) -> ProbationStatus:
        async with self._session_factory() as session:
            existing = (
                await session.execute(
                    select(StrategyProbation).where(
                        StrategyProbation.version_id == version_id
                    )
                )
            ).scalar_one_or_none()
            if existing:
                return self._to_status(existing)
            row = StrategyProbation(
                version_id=version_id,
                strategy_name=strategy_name,
                pair=pair,
                status="active",
                required_trades=required_trades,
                required_days=required_days,
                min_win_rate=min_win_rate,
                min_profit_factor=min_profit_factor,
                max_daily_loss_pct=max_daily_loss_pct,
                spec_json=spec_dict,
                started_at=datetime.utcnow(),
            )
            session.add(row)
            await session.commit()
            logger.info("Probation started for %s (%s)", version_id, strategy_name)
            return self._to_status(row)

    # ------------------------------------------------------------------
    # Trade ingestion + gate evaluation
    # ------------------------------------------------------------------

    async def update(
        self,
        version_id: str,
        closed_trades: Optional[List[Dict[str, Any]]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> ProbationStatus:
        """Record progress. ``closed_trades`` are new closed paper trades
        (dicts with at least ``profit``); ``metrics`` optionally carries
        engine aggregates (win rate, PF, worst daily loss, ...)."""
        closed_trades = closed_trades or []
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(StrategyProbation).where(
                        StrategyProbation.version_id == version_id
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise KeyError(f"no probation for {version_id}")
            if row.status != "active":
                return self._to_status(row)

            row.observed_trades += len(closed_trades)
            days_elapsed = (datetime.utcnow() - row.started_at).days
            row.observed_days = days_elapsed
            if metrics:
                row.last_metrics = {**(row.last_metrics or {}), **metrics}

            quota_met = row.observed_trades >= row.required_trades
            window_over = days_elapsed >= row.required_days

            if window_over and not quota_met:
                row.status = "expired"
                row.notes = (
                    f"only {row.observed_trades}/{row.required_trades} trades "
                    f"in {row.required_days} days"
                )
                row.completed_at = datetime.utcnow()
            elif quota_met:
                # Early verdict on failure; 'passed' only once the window is
                # also over (a strategy must survive the full probation).
                status, notes = self._judge(row)
                if status == "failed":
                    row.status, row.notes = status, notes
                    row.completed_at = datetime.utcnow()
                elif window_over:
                    row.status, row.notes = status, notes
                    row.completed_at = datetime.utcnow()
                else:
                    row.status, row.notes = "active", notes

            await session.commit()
            return self._to_status(row)

    # ------------------------------------------------------------------
    # HUMAN-ONLY transitions
    # ------------------------------------------------------------------

    async def approve(self, version_id: str, approver: str = "human") -> ProbationStatus:
        """HUMAN button: passed → approved (candidate goes live)."""
        return await self._decide(version_id, "approved", approver, from_status="passed")

    async def reject(self, version_id: str, approver: str = "human") -> ProbationStatus:
        """HUMAN button: any non-terminal status → rejected."""
        return await self._decide(version_id, "rejected", approver, from_status=None)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    async def get(self, version_id: str) -> Optional[ProbationStatus]:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(StrategyProbation).where(
                        StrategyProbation.version_id == version_id
                    )
                )
            ).scalar_one_or_none()
            return self._to_status(row) if row else None

    async def list(self, status: Optional[str] = None) -> List[ProbationStatus]:
        stmt = select(StrategyProbation).order_by(StrategyProbation.started_at.desc())
        if status:
            stmt = stmt.where(StrategyProbation.status == status)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [self._to_status(r) for r in rows]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _judge(row: StrategyProbation) -> tuple[str, str]:
        m = row.last_metrics or {}
        wr = m.get("win_rate")
        pf = m.get("profit_factor")
        worst_loss = m.get("worst_daily_loss_pct")
        reasons: List[str] = []
        if wr is None or wr < row.min_win_rate:
            reasons.append(f"win_rate {wr} < {row.min_win_rate}")
        if pf is None or pf < row.min_profit_factor:
            reasons.append(f"profit_factor {pf} < {row.min_profit_factor}")
        if worst_loss is not None and worst_loss > row.max_daily_loss_pct:
            reasons.append(f"worst_daily_loss {worst_loss} > {row.max_daily_loss_pct}")
        if reasons:
            return "failed", "; ".join(reasons)
        return "passed", "all probation gates met"

    async def _decide(
        self,
        version_id: str,
        decision: str,
        approver: str,
        from_status: Optional[str],
    ) -> ProbationStatus:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(StrategyProbation).where(
                        StrategyProbation.version_id == version_id
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise KeyError(f"no probation for {version_id}")
            if from_status is not None and row.status != from_status:
                raise ValueError(
                    f"cannot {decision}: status is {row.status!r}, needs {from_status!r}"
                )
            row.status = decision
            row.notes = f"{decision} by {approver}"
            row.completed_at = row.completed_at or datetime.utcnow()
            await session.commit()
            logger.info("Probation %s %s by %s", version_id, decision, approver)
            return self._to_status(row)

    @staticmethod
    def _to_status(row: StrategyProbation) -> ProbationStatus:
        m = row.last_metrics or {}
        days_elapsed = (datetime.utcnow() - row.started_at).days
        return ProbationStatus(
            version_id=row.version_id,
            strategy_name=row.strategy_name,
            pair=row.pair,
            status=row.status,
            observed_trades=row.observed_trades,
            required_trades=row.required_trades,
            observed_days=row.observed_days,
            required_days=row.required_days,
            days_elapsed=days_elapsed,
            win_rate=m.get("win_rate"),
            profit_factor=m.get("profit_factor"),
            worst_daily_loss_pct=m.get("worst_daily_loss_pct"),
            last_metrics=dict(m) if m else None,
            eligible_for_approval=(row.status == "passed"),
            notes=row.notes or "",
            started_at=row.started_at.isoformat() if row.started_at else "",
            completed_at=row.completed_at.isoformat() if row.completed_at else None,
            min_win_rate=float(row.min_win_rate) if row.min_win_rate is not None else 0.30,
            min_profit_factor=(
                float(row.min_profit_factor) if row.min_profit_factor is not None else 0.95
            ),
            max_daily_loss_pct=(
                float(row.max_daily_loss_pct) if row.max_daily_loss_pct is not None else 0.03
            ),
        )
