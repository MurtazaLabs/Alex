"""Probation runner: accrues paper-probation trades from real market replay.

The backtest gate proved a strategy on the past; probation must observe it on
the *present*. Instead of requiring a live tick-driven paper engine (an order
source that doesn't exist yet), the runner replays the strategy's own compiled
spec over market history using the exact same engine the gate used — same
whitelisted spec, same signal generator, same fills and costs. Trades accrue
in real time as bars close; the verdict (approve/reject) remains a
human-only button.

Correctness in three properties:

1. **Deterministic replay** — every pass replays the same window (probation
   start → now) with the same deterministic engine, so a position still open
   at the end of one pass is still open at the same bar in the next pass and
   is credited naturally when it later closes. No seam loses trades.
2. **Watermark dedup** — only trades closing STRICTLY after the newest
   already-credited ``close_time`` (persisted in ``last_metrics
   ["_replay_through"]``) are counted, so re-deriving old trades each pass
   never double-counts them.
3. **Cumulative metrics** — the judge reads whole-probation win rate / profit
   factor / worst daily loss, so aggregates are merged cumulatively from the
   previous ``last_metrics`` rather than per-window snapshots.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..backtest.engine import BacktestConfig, run_backtest
from ..data.providers.base import ProviderError
from ..data.providers.factory import get_feed_chain
from ..strategy.spec import StrategySpec, parse_spec
from ..strategy.versioning import StrategyVersionManager
from ..utils.math_helpers import pip_size_for_symbol
from .probation import ProbationStore, ProbationStatus
from .validator import SpecSignalGenerator

logger = logging.getLogger(__name__)

_MIN_REPLAY_BARS = 12      # don't start the engine for a handful of bars
_REPLAY_WARMUP_BARS = 12   # matches the engine convention the gate used

# Stage D2: while the event-driven accrual's stream wrote a bar this
# recently, it owns accrual and the replay pass yields.
_EVENT_OWNERSHIP_STALENESS = timedelta(minutes=45)


class ProbationRunner:
    """Replays active probations over market history and feeds the store."""

    def __init__(
        self,
        store: Optional[ProbationStore] = None,
        version_manager: Optional[StrategyVersionManager] = None,
        version_manager_dir: str = "strategies/versions",
        initial_balance: float = 10_000.0,
    ) -> None:
        self.store = store or ProbationStore()
        self._vm = version_manager
        self._vm_dir = version_manager_dir
        self.initial_balance = initial_balance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_once(self) -> List[Dict[str, Any]]:
        """One accrual pass over every active probation."""
        outcomes: List[Dict[str, Any]] = []
        for prob in await self.store.list(status="active"):
            try:
                outcomes.append(await self.run_one(prob.version_id))
            except Exception as exc:  # noqa: BLE001 — one bad row must not stop the pass
                logger.warning(
                    "probation replay failed for %s: %s: %s",
                    prob.version_id,
                    type(exc).__name__,
                    exc,
                )
                outcomes.append(
                    {"version_id": prob.version_id, "status": "error", "detail": str(exc)}
                )
        return outcomes

    async def run_one(self, version_id: str) -> Dict[str, Any]:
        """Replay one probation from its start to now, crediting new trades."""
        prob = await self.store.get(version_id)
        if prob is None:
            return {"version_id": version_id, "status": "not_found"}
        if prob.status != "active":
            return {"version_id": version_id, "status": prob.status, "accrued": 0}

        # Stage D2: while the event-driven accrual's stream is fresh it owns
        # accrual for this version (bars flow in incrementally). This replay
        # pass remains the deterministic fallback and reclaims ownership when
        # the stream goes stale (restart, outage). Both paths run the SAME
        # per-bar engine code, so re-derived trades carry identical close
        # times and the shared watermark below dedups exactly across paths.
        mode = (prob.last_metrics or {}).get("_accrual_mode")
        if mode == "event":
            raw_through = (prob.last_metrics or {}).get("_event_through")
            through = None
            if isinstance(raw_through, str):
                try:
                    through = datetime.fromisoformat(raw_through)
                except ValueError:
                    through = None
            if through is not None and datetime.utcnow() - through < _EVENT_OWNERSHIP_STALENESS:
                return {"version_id": version_id, "status": "event_owned", "accrued": 0}
            logger.info(
                "probation %s: event stream stale, replay reclaims accrual",
                version_id,
            )

        spec = await self._load_spec(version_id)
        if spec is None:
            return {"version_id": version_id, "status": "no_spec"}

        start = datetime.fromisoformat(prob.started_at[:19])
        end = datetime.utcnow()
        bars = await self._fetch_bars(prob.pair, spec.timeframe, start, end)
        if bars is None or len(bars) < _MIN_REPLAY_BARS + _REPLAY_WARMUP_BARS:
            return {
                "version_id": version_id,
                "status": "waiting_for_bars",
                "accrued": 0,
                "bars": 0 if bars is None else len(bars),
            }

        # Data-integrity guard: probation must NEVER accrue trades on
        # synthetic data. If every real feed failed and the chain's last
        # resort (mock) served the bars, skip this pass - the next pass
        # replays the same window from probation start, so nothing is lost.
        if get_feed_chain().active_source == "mock":
            logger.warning(
                "probation replay for %s skipped: feed degraded to mock "
                "(no real data source healthy)",
                version_id,
            )
            return {"version_id": version_id, "status": "skipped_mock_data", "accrued": 0}

        result = await self._replay(spec, prob.pair, bars)

        watermark = self._watermark_of(prob)
        if watermark is None:
            new_trades = list(result.trades or [])
        else:
            new_trades = [
                t
                for t in result.trades or []
                if (self._close_time(t) or start) > watermark
            ]
        metrics = self._cumulative_metrics(prob, new_trades)
        metrics["_accrual_mode"] = "replay"  # Stage D2 ownership stamp

        updated = await self.store.update(
            version_id, closed_trades=new_trades, metrics=metrics
        )
        if updated.status != "active":
            logger.info(
                "Probation %s reached terminal status %s (%s)",
                version_id,
                updated.status,
                updated.notes,
            )
        return {
            "version_id": version_id,
            "status": updated.status,
            "accrued": len(new_trades),
            "observed_trades": updated.observed_trades,
            "required_trades": updated.required_trades,
            "win_rate": updated.win_rate,
            "profit_factor": updated.profit_factor,
            "notes": updated.notes,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _vm_instance(self) -> StrategyVersionManager:
        if self._vm is None:
            self._vm = StrategyVersionManager(storage_path=self._vm_dir)
        return self._vm

    async def _load_spec(self, version_id: str) -> Optional[StrategySpec]:
        """Load the versioned rules from disk and parse through the whitelist."""
        try:
            version = await asyncio.to_thread(
                self._vm_instance().get_version, version_id
            )
        except Exception:  # noqa: BLE001 — missing/corrupt version row
            version = None
        if version is None:
            return None
        spec, _errors = parse_spec(getattr(version, "rules", None) or {})
        return spec

    async def _fetch_bars(
        self, pair: str, timeframe: str, start: datetime, end: datetime
    ) -> Optional[list]:
        """Bars covering the whole probation window from the resilient feed."""
        days = max((end - start).days, 2)
        bars_per_day = {"M15": 96, "M30": 48, "H1": 24, "H4": 6, "D1": 1}
        limit = min(days * bars_per_day.get(timeframe.upper(), 24) + 50, 100_000)
        try:
            return await get_feed_chain().fetch_candles(
                pair, timeframe, start, end=end, limit=limit
            )
        except ProviderError as exc:
            logger.warning("probation replay feed failure for %s: %s", pair, exc)
            return None

    async def _replay(self, spec: StrategySpec, pair: str, bars: list):
        """Run the SAME engine the gate used over the given bars."""
        df = pd.DataFrame(
            {
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
            },
            index=[b.timestamp for b in bars],
        )
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="last")].sort_index()

        config = BacktestConfig(
            start_date=df.index.min().to_pydatetime(),
            end_date=df.index.max().to_pydatetime(),
            initial_balance=self.initial_balance,
            symbols=[pair],
            timeframe=spec.timeframe,
            warmup_bars=_REPLAY_WARMUP_BARS,
        )
        gen = SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(pair))
        return await asyncio.to_thread(run_backtest, {pair: df}, gen, config)

    def _cumulative_metrics(
        self, prob: ProbationStatus, new_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge ONLY the newly credited trades into the cumulative aggregates
        the judge reads. The replay re-derives the whole window every pass, so
        aggregating the full window here would double-count previously
        credited trades — the watermark filter upstream guarantees
        ``new_trades`` is exactly the not-yet-counted subset."""
        prev = prob.last_metrics or {}
        # Carry everything forward EXCEPT the volatile timestamp; the internal
        # cumulative counters (_trades_n, _wins_n, ...) must survive passes
        # with zero new trades or the running totals would silently reset.
        out: Dict[str, Any] = {
            k: v for k, v in prev.items() if k != "_replay_at"
        }
        # Bookkeeping: when the last replay ran.
        out["_replay_at"] = datetime.utcnow().isoformat()
        newest_new = max(
            (self._close_time(t) for t in new_trades if self._close_time(t)),
            default=None,
        )
        prev_mark = self._watermark_of(prob)
        if newest_new is not None and (prev_mark is None or newest_new > prev_mark):
            out["_replay_through"] = newest_new.isoformat()
        elif prev_mark is not None:
            out["_replay_through"] = prev_mark.isoformat()

        if not new_trades:
            return out
        profits = [float(t.get("profit", 0.0)) for t in new_trades]
        wins_n = sum(1 for p in profits if p > 0)
        gross_win = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))

        prev_n = int(prev.get("_trades_n", 0))
        prev_wins = int(prev.get("_wins_n", 0))
        prev_gw = float(prev.get("_gross_win", 0.0))
        prev_gl = float(prev.get("_gross_loss", 0.0))

        cum_n = prev_n + len(profits)
        cum_wins = prev_wins + wins_n
        cum_gw = prev_gw + gross_win
        cum_gl = prev_gl + gross_loss

        out["_trades_n"] = cum_n
        out["_wins_n"] = cum_wins
        out["_gross_win"] = round(cum_gw, 2)
        out["_gross_loss"] = round(cum_gl, 2)
        out["win_rate"] = round(cum_wins / cum_n, 4) if cum_n else 0.0
        out["profit_factor"] = (
            round(cum_gw / cum_gl, 4) if cum_gl > 0 else (2.0 if cum_gw > 0 else 0.0)
        )

        # Worst calendar-day aggregate loss across BOTH prev-recorded and the
        # newly credited trades (prev worst is stored as a fraction).
        by_day: Dict[str, float] = {}
        for t in new_trades:
            ct = self._close_time(t)
            if ct is None:
                continue
            key = ct.strftime("%Y-%m-%d")
            by_day[key] = by_day.get(key, 0.0) + float(t.get("profit", 0.0))
        worst_prev = float(prev.get("worst_daily_loss_pct", 0.0) or 0.0)
        if by_day:
            worst_now = max(0.0, -min(by_day.values())) / self.initial_balance
            out["worst_daily_loss_pct"] = round(max(worst_prev, worst_now), 4)
        elif worst_prev:
            out["worst_daily_loss_pct"] = round(worst_prev, 4)
        return out

    def _watermark_of(self, prob: ProbationStatus) -> Optional[datetime]:
        raw = (prob.last_metrics or {}).get("_replay_through")
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None
        return None

    @staticmethod
    def _close_time(trade: Dict[str, Any]) -> Optional[datetime]:
        raw = trade.get("close_time")
        if isinstance(raw, datetime):
            return raw
        if isinstance(raw, str):
            try:
                return datetime.fromisoformat(raw)
            except ValueError:
                return None
        return None
