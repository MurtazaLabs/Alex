"""Stage E4: daily report — deterministic daily/weekly P&L accounting.

The report is a pure function of supplied data: closed trades, the day's
equity curve, and any violations recorded by the monitor. It produces the
same numbers the human reads on Discord and the audit log:

* trade count, wins, losses, win rate
* gross profit / gross loss, net P&L, profit factor
* return % vs the day's starting equity
* the day's peak-to-trough equity drawdown (when curve points are given)
* largest win / largest loss
* cap breaches (daily loss over LiveCaps, weekly loss over LiveCaps)
* a verdict: OK / DAILY_LOSS_BREACH / WEEKLY_LOSS_BREACH

No network, no broker, no LLM. The caller reads trades and equity from the
broker and hands them in; the report is the arithmetic over them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .caps import LiveCaps, DEFAULT_LIVE_CAPS
from .fills import FillReconciliation, summarize_reconciliations

__all__ = [
    "TradeRecord",
    "DailyReport",
    "build_daily_report",
    "format_daily_report_markdown",
]

VERDICT_OK = "OK"
VERDICT_DAILY_BREACH = "DAILY_LOSS_BREACH"
VERDICT_WEEKLY_BREACH = "WEEKLY_LOSS_BREACH"


@dataclass(frozen=True)
class TradeRecord:
    """One closed trade — the broker is the source of truth."""

    ticket: int
    symbol: str
    side: str
    volume: float
    profit: float
    commission: float = 0.0
    swap: float = 0.0
    opened_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    open_price: float = 0.0
    close_price: float = 0.0

    @property
    def net_profit(self) -> float:
        return float(self.profit) + float(self.commission) + float(self.swap)

    @property
    def is_win(self) -> bool:
        return self.net_profit > 0

    @property
    def is_loss(self) -> bool:
        return self.net_profit < 0

    @classmethod
    def from_dict(cls, row: Dict[str, Any]) -> "TradeRecord":
        return cls(
            ticket=int(row.get("ticket", 0) or 0),
            symbol=str(row.get("symbol", "")),
            side=str(row.get("side", "")),
            volume=float(row.get("volume", 0.0) or 0.0),
            profit=float(row.get("profit", 0.0) or 0.0),
            commission=float(row.get("commission", 0.0) or 0.0),
            swap=float(row.get("swap", 0.0) or 0.0),
            open_price=float(row.get("open_price", 0.0) or 0.0),
            close_price=float(row.get("close_price", 0.0) or 0.0),
        )


def _max_drawdown(points: Sequence[float]) -> float:
    """Peak-to-trough drawdown of an equity series (absolute, >= 0)."""
    peak = None
    worst = 0.0
    for value in points:
        if peak is None or value > peak:
            peak = value
        if peak is not None:
            worst = max(worst, peak - value)
    return worst


@dataclass(frozen=True)
class DailyReport:
    """One day (optionally embedded in a week) of live accounting."""

    day: str
    starting_equity: float
    ending_equity: float
    trades: Sequence[TradeRecord]
    max_drawdown: float
    realized_pnl: float
    returned_pct: float
    weekly_pnl: float
    weekly_start_equity: float
    verdict: str
    recons: Sequence[FillReconciliation] = field(default_factory=tuple)
    violations: Sequence[Dict[str, Any]] = field(default_factory=tuple)

    # -- derived ---------------------------------------------------------

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.trades if t.is_win)

    @property
    def losses(self) -> int:
        return sum(1 for t in self.trades if t.is_loss)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return round(100.0 * self.wins / self.trade_count, 1)

    @property
    def gross_profit(self) -> float:
        return sum(t.net_profit for t in self.trades if t.is_win)

    @property
    def gross_loss(self) -> float:
        return sum(t.net_profit for t in self.trades if t.is_loss)

    @property
    def net_pnl(self) -> float:
        return sum(t.net_profit for t in self.trades)

    @property
    def profit_factor(self) -> Optional[float]:
        if self.gross_loss == 0.0:
            return None
        return round(abs(self.gross_profit / self.gross_loss), 2)

    @property
    def largest_win(self) -> float:
        return max((t.net_profit for t in self.trades), default=0.0)

    @property
    def largest_loss(self) -> float:
        return min((t.net_profit for t in self.trades), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "day": self.day,
            "trade_count": self.trade_count,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "net_pnl": round(self.net_pnl, 2),
            "profit_factor": self.profit_factor,
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(self.ending_equity, 2),
            "returned_pct": self.returned_pct,
            "max_drawdown": round(self.max_drawdown, 2),
            "weekly_pnl": round(self.weekly_pnl, 2),
            "weekly_start_equity": round(self.weekly_start_equity, 2),
            "verdict": self.verdict,
            "fills": summarize_reconciliations(list(self.recons)),
            "violations": list(self.violations),
        }


def build_daily_report(
    *,
    day: date,
    trades: Iterable[TradeRecord],
    starting_equity: float,
    ending_equity: float,
    equity_points: Optional[Sequence[float]] = None,
    weekly_pnl: float = 0.0,
    weekly_start_equity: float = 0.0,
    recons: Optional[Iterable[FillReconciliation]] = None,
    violations: Optional[Iterable[Dict[str, Any]]] = None,
    caps: Optional[LiveCaps] = None,
) -> DailyReport:
    """Compute the day's accounting and its cap verdict.

    ``verdict`` is deterministic: a breach of the daily or weekly loss cap
    (in % of starting equity) is named explicitly; otherwise OK.
    """
    live = caps or DEFAULT_LIVE_CAPS
    trade_list = list(trades)
    recon_list = list(recons or ())
    violation_list = list(violations or ())

    points = list(equity_points) if equity_points else [
        float(starting_equity),
        float(ending_equity),
    ]
    dd = _max_drawdown(points)

    start_eq = float(starting_equity) or 0.0
    realized = float(ending_equity) - start_eq
    returned_pct = round(100.0 * realized / start_eq, 3) if start_eq > 0 else 0.0

    day_loss_limit = start_eq * (live.max_daily_loss_pct / 100.0)
    week_loss_limit = (
        float(weekly_start_equity) * (live.max_weekly_loss_pct / 100.0)
        if weekly_start_equity > 0
        else 0.0
    )

    verdict = VERDICT_OK
    if day_loss_limit > 0 and realized <= -day_loss_limit:
        verdict = VERDICT_DAILY_BREACH
    if week_loss_limit > 0 and float(weekly_pnl) <= -week_loss_limit:
        verdict = VERDICT_WEEKLY_BREACH

    return DailyReport(
        day=day.isoformat(),
        starting_equity=start_eq,
        ending_equity=float(ending_equity),
        trades=tuple(trade_list),
        max_drawdown=dd,
        realized_pnl=realized,
        returned_pct=returned_pct,
        weekly_pnl=float(weekly_pnl),
        weekly_start_equity=float(weekly_start_equity),
        verdict=verdict,
        recons=tuple(recon_list),
        violations=tuple(violation_list),
    )


def format_daily_report_markdown(report: DailyReport) -> str:
    """Human-readable report (Discord-friendly, ASCII only)."""
    pf = report.profit_factor
    pf_text = "n/a" if pf is None else f"{pf:.2f}"
    fills = summarize_reconciliations(list(report.recons))

    lines: List[str] = [
        f"# E4 Daily Report - {report.day}",
        "",
        f"- Verdict: **{report.verdict}**",
        f"- Starting equity: ${report.starting_equity:,.2f}",
        f"- Ending equity: ${report.ending_equity:,.2f}",
        f"- Realized P&L: ${report.realized_pnl:+,.2f} ({report.returned_pct:+.2f}%)",
        f"- Max drawdown: ${report.max_drawdown:,.2f}",
        "",
        f"- Trades: {report.trade_count} (W {report.wins} / L {report.losses}, "
        f"win rate {report.win_rate:.1f}%)",
        f"- Gross profit: ${report.gross_profit:,.2f}",
        f"- Gross loss: ${report.gross_loss:,.2f}",
        f"- Profit factor: {pf_text}",
        f"- Largest win: ${report.largest_win:+,.2f}",
        f"- Largest loss: ${report.largest_loss:+,.2f}",
        f"- Week-to-date P&L: ${report.weekly_pnl:+,.2f}",
        "",
        f"- Fill checks: {fills['count']} "
        f"(drift {fills['drift_count']}, avg slip {fills['avg_slippage_pips']:.2f}p, "
        f"max slip {fills['max_slippage_pips']:.2f}p)",
        f"- Violations recorded: {len(report.violations)}",
    ]
    for violation in report.violations:
        note = str(violation.get("detail", violation))
        lines.append(f"  - {note}")
    return "\n".join(lines)
