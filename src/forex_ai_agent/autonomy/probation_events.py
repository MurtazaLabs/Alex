"""Stage D2: event-driven probation accrual from streaming market bars.

The bar-close replay runner (``probation_runner.ProbationRunner``) re-derives
each active probation's whole window every 900s. This module is the primary
accrual path the roadmap called for: a pump publishes every newly **closed**
bar as a ``market.bar.closed`` event on the EventBus, and an accrual
subscriber streams those bars — in order, exactly once — into a persistent
per-version ``BacktestEngine`` via ``BacktestEngine._process_bar`` (the exact
per-bar code the gate's batch replay runs, extracted verbatim in Stage D2).
Trades therefore carry IDENTICAL fills, costs and close times to the replay,
so the shared ``_replay_through`` watermark dedups exactly across both paths
and the replay runner remains a deterministic fallback/reconciler.

Constitution compliance (no exceptions):

- **Deterministic numbers** — no LLM anywhere in this path. Signals come from
  the compiled spec's ``SpecSignalGenerator``; entries fill through the
  engine's OrderManager (spread + slippage + commission identical to the
  gate); swaps accrue once per completed day. Every number is engine output.
- **No LLM risk** — the accrual only observes and counts; the verdict
  (approve/reject) remains a human-only ProbationStore transition.
- **No synthetic data** — the pump and the accrual both refuse to run while
  the feed chain's active source is ``mock`` (same guard as the replay).
- **Watermark dedup** — trades are credited only when their ``close_time`` is
  strictly after the newest already-credited close (shared with the replay
  runner), and every bar is processed at most once per version (duplicate or
  out-of-order events are ignored).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ..backtest.engine import BacktestConfig, BacktestEngine, DataFeed
from ..core.event_bus import Event, EventBus, InMemoryEventBus
from ..data.providers.base import ProviderError
from ..data.providers.factory import get_feed_chain
from ..utils.math_helpers import pip_size_for_symbol
from .probation import ProbationStore
from .probation_runner import _REPLAY_WARMUP_BARS, ProbationRunner
from .validator import SpecSignalGenerator

logger = logging.getLogger(__name__)

#: Event type published by the pump and consumed by the accrual.
MARKET_BAR_EVENT = "market.bar.closed"

#: Approximate bar durations (minutes) — used for forming-bar filtering,
#: backfill sizing and gap detection.
_TF_MINUTES = {"M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440}


def _tf_delta(timeframe: str) -> timedelta:
    return timedelta(minutes=_TF_MINUTES.get(timeframe.upper(), 60))


class MarketBarEventPump:
    """Publishes newly closed market bars as events on the bus.

    Each poll fetches the newest closed bars for every subscribed
    (pair, timeframe) from the resilient feed chain and publishes one
    ``market.bar.closed`` event per bar, in bar-time order, deduplicated per
    subscription by the newest already-published bar. Bars that are still
    forming (younger than their timeframe) are never published, and the pump
    refuses to run on mock data — probation must never accrue on synthetics.
    """

    def __init__(
        self,
        event_bus: EventBus,
        subscriptions: Callable[[], Awaitable[List[Tuple[str, str]]]],
        poll_seconds: float = 30.0,
    ) -> None:
        self.event_bus = event_bus
        self.subscriptions = subscriptions
        self.poll_seconds = poll_seconds
        self._last_published: Dict[Tuple[str, str], datetime] = {}

    async def publish_new_bars(self) -> int:
        """One pump pass; returns the number of events published."""
        if get_feed_chain().active_source == "mock":
            logger.warning("market event pump skipped: feed degraded to mock")
            return 0
        published = 0
        for pair, tf in await self.subscriptions():
            key = (pair, tf.upper())
            try:
                bars = await self._fetch_since(pair, tf, self._last_published.get(key))
            except ProviderError as exc:
                logger.warning("market event pump fetch failed for %s %s: %s", pair, tf, exc)
                continue
            except Exception as exc:  # noqa: BLE001 — one bad subscription must not stop the pass
                logger.warning(
                    "market event pump fetch error for %s %s: %s: %s",
                    pair, tf, type(exc).__name__, exc,
                )
                continue
            for b in bars:
                await self.event_bus.publish(
                    Event(
                        event_type=MARKET_BAR_EVENT,
                        aggregate_id=pair,
                        aggregate_type="bar",
                        payload=self._payload(pair, tf, b),
                        source_agent="market_bar_event_pump",
                    )
                )
                self._last_published[key] = b.timestamp
                published += 1
        return published

    async def _fetch_since(self, pair: str, tf: str, since: Optional[datetime]) -> list:
        """Closed bars newer than ``since`` (or the last ~2 bars on boot)."""
        if since is None:
            start = datetime.utcnow() - 2 * _tf_delta(tf)
        else:
            start = since
        bars = await get_feed_chain().fetch_candles(pair, tf, start, limit=64) or []
        now = datetime.utcnow()
        out = []
        for b in bars:
            ts = getattr(b, "timestamp", None)
            if ts is None:
                continue
            if since is not None and ts <= since:
                continue  # already published
            if now < ts + _tf_delta(tf):
                continue  # bar still forming — only CLOSED bars are events
            out.append(b)
        out.sort(key=lambda b: b.timestamp)
        return out

    @staticmethod
    def _payload(pair: str, tf: str, b: Any) -> Dict[str, Any]:
        return {
            "symbol": pair,
            "timeframe": tf.upper(),
            "timestamp": b.timestamp.isoformat(),
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(getattr(b, "volume", 0) or 0),
        }


class _VersionStream:
    """Persistent per-version incremental engine state."""

    __slots__ = (
        "version_id",
        "pair",
        "timeframe",
        "started_at",
        "spec",
        "feed",
        "engine",
        "processed_through",
        "seen_trades",
    )

    def __init__(
        self,
        version_id: str,
        pair: str,
        timeframe: str,
        started_at: str,
        spec: Any,
        feed: DataFeed,
        engine: BacktestEngine,
    ) -> None:
        self.version_id = version_id
        self.pair = pair
        self.timeframe = timeframe
        self.started_at = started_at
        self.spec = spec
        self.feed = feed
        self.engine = engine
        self.processed_through: Optional[datetime] = None
        self.seen_trades = 0


class EventDrivenProbationAccrual:
    """Accrues probation trades from ``market.bar.closed`` events.

    One persistent ``BacktestEngine`` per active probation version. The first
    event for a version backfills every closed bar from the probation's start
    (the same window the replay runner covers) through the identical
    incremental path, then each subsequent event is processed exactly once.
    Closed trades are credited into :class:`ProbationStore` with the shared
    watermark and the runner's cumulative-metric merge, stamped with
    ``_accrual_mode="event"`` so the replay runner yields while the stream is
    fresh and reclaims it when stale.
    """

    def __init__(
        self,
        store: Optional[ProbationStore] = None,
        version_manager: Optional[Any] = None,
        version_manager_dir: str = "strategies/versions",
        initial_balance: float = 10_000.0,
        event_bus: Optional[EventBus] = None,
        runner: Optional[ProbationRunner] = None,
    ) -> None:
        self.store = store or ProbationStore()
        self.initial_balance = initial_balance
        self._runner = runner or ProbationRunner(
            store=self.store,
            version_manager=version_manager,
            version_manager_dir=version_manager_dir,
            initial_balance=initial_balance,
        )
        self._bus = event_bus or InMemoryEventBus()
        self._streams: Dict[str, _VersionStream] = {}
        self._subscribed = False

    # ------------------------------------------------------------------
    # Bus wiring
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if not self._subscribed:
            await self._bus.subscribe(MARKET_BAR_EVENT, self.on_bar_event)
            self._subscribed = True

    async def stop(self) -> None:
        if self._subscribed:
            await self._bus.unsubscribe(MARKET_BAR_EVENT, self.on_bar_event)
            self._subscribed = False

    async def subscriptions(self) -> List[Tuple[str, str]]:
        """(pair, timeframe) pairs the event pump must stream for."""
        out: List[Tuple[str, str]] = []
        for prob in await self.store.list(status="active"):
            spec = await self._runner._load_spec(prob.version_id)
            if spec is not None:
                out.append((prob.pair, spec.timeframe.upper()))
        return out

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    async def on_bar_event(self, event: Event) -> None:
        payload = getattr(event, "payload", None) or {}
        pair = payload.get("symbol")
        tf = str(payload.get("timeframe") or "").upper()
        raw_ts = payload.get("timestamp")
        if not pair or not tf or not raw_ts:
            return
        try:
            ts = datetime.fromisoformat(str(raw_ts))
        except ValueError:
            return
        for prob in await self.store.list(status="active"):
            if prob.pair != pair:
                continue
            try:
                await self._accrue_bar(prob, tf, ts, payload)
            except Exception as exc:  # noqa: BLE001 — one bad version must not stop the stream
                logger.warning(
                    "event accrual failed for %s: %s: %s",
                    prob.version_id,
                    type(exc).__name__,
                    exc,
                )

    async def _accrue_bar(
        self, prob: Any, tf: str, ts: datetime, payload: Dict[str, Any]
    ) -> None:
        # Data-integrity guard: probation must NEVER accrue on synthetic
        # data (same guard as the replay runner).
        if get_feed_chain().active_source == "mock":
            logger.warning(
                "event accrual for %s skipped: feed degraded to mock", prob.version_id
            )
            return

        stream = self._streams.get(prob.version_id)
        if stream is None or stream.started_at != prob.started_at:
            stream = await self._build_stream(prob, tf, before=ts)
            if stream is None:
                return
            self._streams[prob.version_id] = stream
        if stream.timeframe != tf:
            return
        if stream.processed_through is not None and ts <= stream.processed_through:
            return  # duplicate or out-of-order event — already reflected

        await self._repair_gap(stream, ts)
        self._ingest_bar(
            stream,
            ts,
            float(payload.get("open", 0.0)),
            float(payload.get("high", 0.0)),
            float(payload.get("low", 0.0)),
            float(payload.get("close", 0.0)),
            float(payload.get("volume", 0.0) or 0.0),
        )
        await self._credit(stream)

    # ------------------------------------------------------------------
    # Engine plumbing
    # ------------------------------------------------------------------

    async def _build_stream(
        self, prob: Any, tf: str, before: datetime
    ) -> Optional[_VersionStream]:
        """Create the persistent engine and backfill it from probation start.

        Backfill covers every closed bar from ``started_at`` up to (but
        excluding) the first event bar — the same window the replay runner
        would derive — fed through the identical incremental path, so state
        (positions, indicator windows, signal cooldowns) matches a batch
        replay of the same bars exactly.
        """
        spec = await self._runner._load_spec(prob.version_id)
        if spec is None:
            return None
        start = datetime.fromisoformat(prob.started_at[:19])

        feed = DataFeed({})
        config = BacktestConfig(
            start_date=start,
            end_date=before + timedelta(days=3650),
            initial_balance=self.initial_balance,
            symbols=[prob.pair],
            timeframe=tf,
            warmup_bars=_REPLAY_WARMUP_BARS,
        )
        generator = SpecSignalGenerator(spec, pip_size=pip_size_for_symbol(prob.pair))
        engine = BacktestEngine(config, feed, generator)
        stream = _VersionStream(
            prob.version_id, prob.pair, tf, prob.started_at, spec, feed, engine
        )

        bars = await self._fetch_bars(prob.pair, tf, start, before)
        for b in sorted(bars, key=lambda x: x.timestamp):
            # Client-side window enforcement: never ingest bars outside the
            # requested range, even if the chain over-serves.
            if b.timestamp < start or b.timestamp >= before:
                continue
            self._ingest_bar(
                stream,
                b.timestamp,
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                float(getattr(b, "volume", 0) or 0),
            )
        return stream

    async def _repair_gap(self, stream: _VersionStream, ts: datetime) -> None:
        """Fetch and ingest any bars the event stream missed (outage/poll gap)."""
        if stream.processed_through is None or ts <= stream.processed_through:
            return
        if ts - stream.processed_through < 1.5 * _tf_delta(stream.timeframe):
            return  # adjacent bars — nothing missing
        bars = await self._fetch_bars(
            stream.pair, stream.timeframe, stream.processed_through, ts
        )
        for b in sorted(bars, key=lambda x: x.timestamp):
            if (
                b.timestamp <= stream.processed_through or b.timestamp >= ts
            ):
                continue  # strictly between the processed bar and the event bar
            self._ingest_bar(
                stream,
                b.timestamp,
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                float(getattr(b, "volume", 0) or 0),
            )

    def _ingest_bar(
        self,
        stream: _VersionStream,
        ts: datetime,
        o: float,
        h: float,
        l: float,
        c: float,
        v: float,
    ) -> None:
        """Feed one bar through the engine — the same code the gate runs."""
        if stream.processed_through is not None and ts <= stream.processed_through:
            return
        stream.feed.append_bar(stream.pair, ts, o, h, l, c, v)
        stream.engine._process_bar(ts)
        stream.processed_through = ts

    async def _fetch_bars(
        self, pair: str, tf: str, start: datetime, end: datetime
    ) -> list:
        """Closed bars for backfill/gap repair from the resilient feed."""
        tf_min = _TF_MINUTES.get(tf.upper(), 60)
        seconds = max((end - start).total_seconds(), 0.0)
        limit = min(int(seconds // (tf_min * 60)) + 50, 100_000)
        try:
            return await get_feed_chain().fetch_candles(
                pair, tf, start, end=end, limit=limit
            ) or []
        except ProviderError as exc:
            logger.warning("event accrual backfill fetch failed for %s %s: %s", pair, tf, exc)
            return []

    # ------------------------------------------------------------------
    # Crediting
    # ------------------------------------------------------------------

    async def _credit(self, stream: _VersionStream) -> None:
        """Watermark-filter newly closed trades and push them to the store."""
        prob = await self.store.get(stream.version_id)
        if prob is None or prob.status != "active":
            self._streams.pop(stream.version_id, None)
            return

        trades = stream.engine.position_manager.closed_trades
        new_raw = trades[stream.seen_trades:]
        stream.seen_trades = len(trades)

        watermark = self._runner._watermark_of(prob)
        if watermark is None:
            new_trades = list(new_raw)
        else:
            fallback = datetime.fromisoformat(prob.started_at[:19])
            new_trades = [
                t for t in new_raw if (self._runner._close_time(t) or fallback) > watermark
            ]

        # The runner's cumulative merge is the single source of truth for the
        # judge's aggregates; the shared ``_replay_through`` watermark makes
        # replay-vs-event dedup exact because both paths run the same engine.
        metrics = self._runner._cumulative_metrics(prob, new_trades)
        metrics["_accrual_mode"] = "event"
        if stream.processed_through is not None:
            metrics["_event_through"] = stream.processed_through.isoformat()

        try:
            updated = await self.store.update(
                stream.version_id, closed_trades=new_trades, metrics=metrics
            )
        except KeyError:
            self._streams.pop(stream.version_id, None)
            return
        if updated.status != "active":
            logger.info(
                "Probation %s reached terminal status %s via event accrual (%s)",
                stream.version_id,
                updated.status,
                updated.notes,
            )
            self._streams.pop(stream.version_id, None)
