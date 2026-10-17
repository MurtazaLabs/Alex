"""Stage E0: deterministic execution simulator.

Models live-like execution microstructure — latency, partial fills,
requotes, rejections and spread widening — ON TOP of the exact same
OrderManager the backtest gate and the paper layer use, by SUBCLASSING
``BacktestEngine``. Everything except the order-execution stage
(indicators, spec signals, position lifecycle, SL/TP-vs-high/low semantics,
once-per-day swaps, accounting) is the engine's own inherited code, so
clean-mode results are IDENTICAL to ``run_backtest`` by construction and
degraded modes are explicit, seed-deterministic deltas on top.

Determinism contract: for a fixed (data, params, seed) tuple the simulator
reproduces the exact same fills, rejects and trades — a fresh ``random.
Random(seed)`` is created per run and every degradation decision draws from
it. No wall-clock, no network, no LLM, nothing sends anywhere: this module
is pure computation (source-scanned in tests).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..backtest.engine import BacktestConfig, BacktestEngine, DataFeed
from ..backtest.execution import OrderSide, OrderStatus, OrderType
from ..backtest.orders import ExecutionContext
from .governor import ExecutionGovernor, TRIP_POSITION_FLATTEN

logger = logging.getLogger(__name__)

__all__ = [
    "ExecutionMode",
    "ExecutionParams",
    "SimulatorResult",
    "ExecutionSimulator",
    "run_simulation",
]


class ExecutionMode:
    """Degradation toggles — flags are explicit, never ambient."""

    CLEAN = "clean"
    LATENCY = "latency"
    PARTIAL_FILL = "partial_fill"
    REQUOTE = "requote"
    SPREAD_WIDENING = "spread_widening"


@dataclass
class ExecutionParams:
    """Simulator microstructure parameters.

    Defaults are conservative mid-band values; scenarios override them.
    Every decision the simulator makes is a pure function of these, the
    bars, and ``seed``.
    """

    # --- latency (bar-time model; bars are the simulator's clock) -----
    # DEFAULTS ARE CLEAN (latency exactly 1 bar = the backtest engine's own
    # next-bar fill). Any degradation is an explicit scenario choice.
    latency_bars_min: int = 1  # signal bar T → earliest fill bar T+min
    latency_bars_max: int = 1
    # --- partial fills ------------------------------------------------
    partial_fill_probability: float = 0.0
    partial_fill_min_fraction: float = 0.5   # first tranche ≥ 50% of order
    partial_fill_max_fraction: float = 0.8   # subsequent tranches ≤ 80% of remainder
    max_partial_tranches: int = 3
    # --- requotes / rejections ----------------------------------------
    requote_probability: float = 0.0
    requote_max_pips: float = 2.0            # re-price by ≤ this many pips
    requote_max_attempts: int = 2            # then accept the re-priced fill
    rejection_probability: float = 0.0       # flat refusal (off-quotes etc.)
    # --- spread widening -----------------------------------------------
    spread_widen_multiplier: float = 1.0     # 1.0 = unchanged (clean parity)
    # --- bookkeeping ----------------------------------------------------
    seed: int = 7
    # Route clean-equivalent params through the intent book anyway, purely
    # to record fill diagnostics (behaviorally identical to the base engine:
    # latency 1 fills the next bar exactly like the base _process_orders).
    record_intents: bool = False

    def sanitized(self) -> "ExecutionParams":
        """Clamp to sane ranges so a bad scenario can't disable parity."""
        p = ExecutionParams(**{**self.__dict__})
        p.latency_bars_min = max(1, int(p.latency_bars_min))
        p.latency_bars_max = max(p.latency_bars_min, int(p.latency_bars_max))
        p.partial_fill_probability = min(max(p.partial_fill_probability, 0.0), 1.0)
        p.partial_fill_min_fraction = min(max(p.partial_fill_min_fraction, 0.01), 1.0)
        p.partial_fill_max_fraction = min(
            max(p.partial_fill_max_fraction, p.partial_fill_min_fraction), 1.0
        )
        p.max_partial_tranches = max(1, int(p.max_partial_tranches))
        p.requote_probability = min(max(p.requote_probability, 0.0), 1.0)
        p.requote_max_pips = max(p.requote_max_pips, 0.0)
        p.rejection_probability = min(max(p.rejection_probability, 0.0), 1.0)
        p.spread_widen_multiplier = max(p.spread_widen_multiplier, 0.0)
        return p


@dataclass
class SimulatorResult:
    """BacktestResult-compatible result plus execution diagnostics."""

    trades: List[Dict[str, Any]]
    equity_curve: List[Tuple[datetime, float]]
    metrics: Dict[str, Any]
    fills: List[Dict[str, Any]]
    rejected_intents: List[Dict[str, Any]]
    latency_histogram: Dict[str, int]
    partial_fills: int
    requotes: int
    rejections: int


class ExecutionSimulator(BacktestEngine):
    """BacktestEngine subclass that models live-like order execution.

    Inherited unchanged: indicators, signal generation, position lifecycle,
    SL/TP-vs-high/low semantics, once-per-day swap accrual, equity curve.

    Overridden: ``_process_orders`` only. CLEAN mode delegates to the base
    implementation (identical fills by construction). Degraded modes run
    the same OrderManager primitives per tranche and only add latency,
    volume splits, re-pricing and refusals on top.
    """

    def __init__(self, config: BacktestConfig, data_feed: DataFeed, signal_generator, *, params: Optional[ExecutionParams] = None, governor: Optional[ExecutionGovernor] = None):
        super().__init__(config, data_feed, signal_generator)
        self.params = (params or ExecutionParams()).sanitized()
        self.mode_flags = self._derive_flags(self.params)
        self._rng = random.Random(self.params.seed)
        # E1: when a governor is attached the simulator runs UNDER it —
        # intents must be accepted (write-ahead journaled) before they can
        # fill, breakers trip the kill switch, and a trip cancels every
        # deferred intent. A governor forces the intent path even for clean
        # params ("clean microstructure under governance").
        self.governor = governor

        # --- diagnostics -------------------------------------------------
        self.sim_fills: List[Dict[str, Any]] = []
        self.rejected_intents: List[Dict[str, Any]] = []
        self.latency_histogram: Dict[str, int] = {}
        self.partial_fill_count = 0
        self.requote_count = 0
        self.rejection_count = 0

        # Deferred intent book: {intent_key: {order, due_bar_index, created_bar, ...}}
        self._intents: Dict[str, Dict[str, Any]] = {}
        # Monotonic bar counter (per-symbol engines: one symbol per config)
        self._bar_index = 0
        self._intents_frozen = False
        # E1 breaker baselines (maintained by _process_bar under a governor)
        self._peak_equity = 0.0
        self._day_start_equity: Optional[float] = None

    # ------------------------------------------------------------------
    # Mode flags
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_flags(p: ExecutionParams) -> set:
        flags = {ExecutionMode.CLEAN}
        if p.record_intents:
            # Diagnostic routing through the intent path (latency 1 = the
            # base engine's own next-bar fill semantics).
            flags.add(ExecutionMode.LATENCY)
        if p.latency_bars_max > 1 or p.latency_bars_min > 1:
            flags.add(ExecutionMode.LATENCY)
        if p.partial_fill_probability > 0.0:
            flags.add(ExecutionMode.PARTIAL_FILL)
        if p.requote_probability > 0.0 or p.rejection_probability > 0.0:
            flags.add(ExecutionMode.REQUOTE)
        if p.spread_widen_multiplier != 1.0:
            flags.add(ExecutionMode.SPREAD_WIDENING)
        return flags

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _process_bar(self, timestamp: datetime) -> bool:
        """Count bars monotonically, run the engine's bar body, then enforce
        the governor's breakers on the post-bar state (E1)."""
        self._bar_index += 1
        if self.governor is not None:
            # New calendar day → reset the daily-loss baseline (deterministic,
            # derived from the engine's own day tracking).
            if self._last_bar_date is not None and timestamp.date() != self._last_bar_date:
                self._day_start_equity = self.account.equity
            elif self._day_start_equity is None:
                self._day_start_equity = self.account.initial_balance
            for symbol in self.config.symbols:
                self.governor.record_bar(symbol, timestamp)
        ok = super()._process_bar(timestamp)
        if self.governor is not None and ok:
            if self.account.equity > self._peak_equity:
                self._peak_equity = self.account.equity
            self.governor.check_breakers(
                equity=self.account.equity,
                day_start_equity=self._day_start_equity or self.account.initial_balance,
                peak_equity=self._peak_equity,
                open_positions=len(self.position_manager.positions),
                now=timestamp,
            )
            if self.governor.state.tripped:
                self._enforce_trip_policy()
        return ok

    def _enforce_trip_policy(self) -> None:
        """Kill switch tripped: enforce the declared trip-position policy.

        Deterministic post-state per ``GovernorLimits.trip_position_policy``:
        - HOLD: open positions keep the engine's own lifecycle — the next
          bar's ``update_positions`` still applies SL/TP; nothing is closed.
        - FLATTEN: every open position is closed AT MARKET through the
          engine's own ``PositionManager.close_position`` (the exact
          trade-record + balance-booking path SL/TP uses); the simulator
          reports each close to the governor's audit trail. No NEW close
          code, no LLM, no outside calls — pure bookkeeping.
        Both policies: every deferred intent is cancelled (unchanged).
        """
        reason = f"KILL_SWITCH:{self.governor.trip_reason}"
        for key, intent in list(self._intents.items()):
            self.order_manager.cancel_order(intent["order"].id)
            self.rejected_intents.append(
                {
                    "reason": reason,
                    "symbol": intent["order"].symbol,
                    "side": intent["order"].side.value,
                    "volume": intent["order"].volume,
                    "timestamp": self.current_time,
                    "created_bar": intent["created_bar"],
                    "executed_bar": None,
                    "comment": intent["order"].comment,
                }
            )
        self._intents.clear()
        if self.governor.limits.trip_position_policy == TRIP_POSITION_FLATTEN:
            self._flatten_positions_on_trip(reason)

    def _flatten_positions_on_trip(self, reason: str) -> None:
        """FLATTEN policy: close every open position at market, NOW.

        Uses the engine's own close path — ``PositionManager.
        close_position`` produces the identical trade record (profit,
        commission, swap, hold time) an SL/TP close produces, and the
        engine re-books account balance from closed trades at the start of
        the next bar, so the accounting never forks. The mid-bar equity
        snapshot is refreshed with the same ``account.update`` the bar body
        uses. Closes are reported to the governor's audit trail.
        """
        for pos_id, position in list(self.position_manager.positions.items()):
            # Mark price: the trip bar's close — the exact price the bar
            # body just used to mark this position; fall back to the last
            # mark if the symbol had no bar this bar.
            bar = self.current_bars.get(position.symbol)
            price = bar.close if bar is not None else position.current_price
            trade = self.position_manager.close_position(
                pos_id,
                price,
                commission=0.0,
                reason=reason,
                timestamp=self.current_time,
            )
            if trade is None:
                continue
            self.governor.record_flattened(
                {
                    "position_id": trade["id"],
                    "symbol": trade["symbol"],
                    "side": trade["side"],
                    "volume": trade["volume"],
                    "open_price": trade["open_price"],
                    "close_price": trade["close_price"],
                    "profit": trade["profit"],
                    "close_time": trade["close_time"],
                    "trip_reason": self.governor.trip_reason,
                }
            )
        # Same mid-bar equity refresh the bar body uses (realized P&L now
        # lives in closed_trades; unrealized is zero).
        self.account.update(self.position_manager)

    def _build_context(self, symbol: str, bar) -> ExecutionContext:
        """Same context construction as the engine's _process_orders."""
        return ExecutionContext(
            current_candle={
                "open": bar.open, "high": bar.high,
                "low": bar.low, "close": bar.close,
            },
            spread_pips=self.order_manager.spread_model.get_spread(
                symbol, self.current_time, self._get_session()
            ),
            atr_pips=self.indicators.get(symbol, {}).get("atr_14", 10.0),
            session=self._get_session(),
            timestamp=self.current_time,
        )

    def _widen_context(self, ctx: ExecutionContext) -> ExecutionContext:
        p = self.params
        if p.spread_widen_multiplier == 1.0:
            return ctx
        return ExecutionContext(
            current_candle=ctx.current_candle,
            spread_pips=ctx.spread_pips * p.spread_widen_multiplier,
            atr_pips=ctx.atr_pips,
            session=ctx.session,
            timestamp=ctx.timestamp,
        )

    # ------------------------------------------------------------------
    # Order execution stage (the only overridden engine stage)
    # ------------------------------------------------------------------

    def _process_orders(self) -> None:
        """Process pending orders with simulator microstructure.

        CLEAN mode: identical to BacktestEngine._process_orders (delegates).
        Otherwise: due intents execute through the SAME OrderManager; every
        degradation (latency, tranches, requotes, rejections) is a local,
        seed-deterministic transformation of that execution.
        """
        if self.mode_flags == {ExecutionMode.CLEAN}:
            super()._process_orders()
            return

        # 1) Execute intents whose latency window has elapsed.
        due = [k for k, it in self._intents.items() if it["due_bar"] <= self._bar_index]
        for key in sorted(due):
            intent = self._intents.pop(key)
            self._execute_intent(intent)

        # 2) Adopt any *foreign* pending orders (e.g. scenario-injected
        #    stop/limit orders) into the intent book so they execute through
        #    the same code path — never via the base _process_orders here,
        #    which would double-fill our own intent-owned orders.
        owned = {it["order"].id for it in self._intents.values()}
        for foreign in self.order_manager.get_pending_orders():
            if foreign.id in owned:
                continue
            self._intents[foreign.id] = {
                "key": foreign.id,
                "order": foreign,
                "created_bar": self._bar_index,
                "latency_bars": 0,
                "due_bar": self._bar_index,
                "foreign": True,
            }

    def _execute_intent(self, intent: Dict[str, Any]) -> None:
        """Run one deferred intent through the OrderManager now."""
        p = self.params
        order = intent["order"]
        symbol = order.symbol

        bar = self.current_bars.get(symbol)
        if bar is None:
            # No bar for this symbol at this instant: retry next bar.
            intent["due_bar"] = self._bar_index + 1
            self._intents[intent["key"]] = intent
            return

        ctx = self._widen_context(self._build_context(symbol, bar))
        lat = self._bar_index - intent["created_bar"]
        bucket = f"{intent['latency_bars']}bars" if lat == intent["latency_bars"] else f"{intent['latency_bars']}bars+delayed"
        self.latency_histogram[bucket] = self.latency_histogram.get(bucket, 0) + 1

        # Flat rejection (off-quotes / reject-mode scenarios).
        if p.rejection_probability > 0 and self._rng.random() < p.rejection_probability:
            self.order_manager.cancel_order(order.id)
            self.rejection_count += 1
            self.rejected_intents.append(
                {
                    "reason": "REJECTED",
                    "symbol": symbol,
                    "side": order.side.value,
                    "volume": order.volume,
                    "timestamp": self.current_time,
                    "created_bar": intent["created_bar"],
                    "executed_bar": self._bar_index,
                    "comment": order.comment,
                }
            )
            return

        # Requote: the counterparty re-prices ADVERSE to the taker by up to
        # requote_max_pips before filling (modeled as a shift of the fill
        # candle's close in the adverse direction). One requote event per
        # intent; all its tranches fill at the re-priced level.
        if p.requote_probability > 0 and self._rng.random() < p.requote_probability:
            shift = self._rng.uniform(0.0, p.requote_max_pips) * 0.0001
            candle = dict(ctx.current_candle)
            if order.side == OrderSide.BUY:
                candle["close"] = candle["close"] + shift
            else:
                candle["close"] = candle["close"] - shift
            ctx = ExecutionContext(
                current_candle=candle,
                spread_pips=ctx.spread_pips,
                atr_pips=ctx.atr_pips,
                session=ctx.session,
                timestamp=ctx.timestamp,
            )
            self.requote_count += 1

        # Volume laddering (partial fills): deterministic tranches. Each
        # tranche runs through the SAME OrderManager as its own child order
        # (execute_order only fills PENDING orders, so one order cannot fill
        # twice); the parent intent order is cancelled so nothing fills it.
        tranches: List[float] = [order.volume]
        if p.partial_fill_probability > 0 and self._rng.random() < p.partial_fill_probability:
            remaining = order.volume
            tranches = []
            first = max(
                p.partial_fill_min_fraction * order.volume, 0.01
            )
            first = min(first, remaining)
            tranches.append(round(first, 2))
            remaining = round(remaining - first, 2)
            attempts = 1
            while remaining >= 0.01 and attempts < p.max_partial_tranches:
                if self._rng.random() < p.partial_fill_probability:
                    take = min(
                        remaining,
                        max(remaining * p.partial_fill_max_fraction, 0.01),
                    )
                    tranches.append(round(take, 2))
                    remaining = round(remaining - take, 2)
                    attempts += 1
                else:
                    break
            if len(tranches) > 1:
                self.partial_fill_count += 1
            tranches = [t for t in tranches if t >= 0.01]

        if len(tranches) > 1:
            self.order_manager.cancel_order(order.id)

        filled_any = False
        for i, vol in enumerate(tranches):
            tranche_order = order if len(tranches) == 1 else self._new_tranche_order(order, vol, i)
            else_volume = tranche_order.volume
            tranche_order.volume = vol
            fills = self.order_manager.execute_order(tranche_order, ctx)
            tranche_order.volume = else_volume
            for fill in fills:
                filled_any = True
                fill_record = {
                    "order_id": fill.order_id,
                    "symbol": fill.symbol,
                    "side": fill.side.value,
                    "volume": fill.volume,
                    "price": fill.price,
                    "commission": fill.commission,
                    "timestamp": fill.timestamp,
                    "fill_type": fill.fill_type,
                    "created_bar": intent["created_bar"],
                    "executed_bar": self._bar_index,
                    "latency_bars": self._bar_index - intent["created_bar"],
                    "tranche": i,
                    "tranches": len(tranches),
                }
                self.sim_fills.append(fill_record)
                if self.governor is not None and intent.get("journal_key"):
                    self.governor.record_fill(intent["journal_key"], dict(fill_record))
                self._open_sim_position(tranche_order, fill)

        # Not filled this attempt and still pending (e.g. a foreign stop
        # order that has not triggered yet): retry on a later bar.
        if not filled_any and order.status == OrderStatus.PENDING:
            intent["due_bar"] = self._bar_index + 1
            self._intents[intent["key"]] = intent
        elif self.governor is not None and not filled_any and intent.get("journal_key"):
            # Fully refused/rejected execution: the journaled intent must not
            # stay PENDING forever.
            self.governor.record_cancel(intent["journal_key"], "execution_refused")

    def _rekey_order(self, order, new_id: str) -> None:
        """Rename an order deterministically (create_order mints uuid4)."""
        old_id = order.id
        order.id = new_id
        self.order_manager.orders.pop(old_id, None)
        self.order_manager.orders[new_id] = order

    def _new_tranche_order(self, parent, vol: float, index: int):
        """Create a child order for one tranche with a deterministic id."""
        child = self.order_manager.create_order(
            symbol=parent.symbol,
            side=parent.side,
            order_type=parent.type,
            volume=vol,
            sl=parent.sl,
            tp=parent.tp,
            magic=parent.magic,
            comment=parent.comment,
        )
        self._rekey_order(child, f"{parent.id}-t{index}")
        return child

    def _open_sim_position(self, order, fill) -> None:
        """Open the position on the fill — mirrors the engine's own code."""
        if order.type in (OrderType.MARKET, OrderType.STOP):
            self.position_manager.open_position(
                symbol=fill.symbol,
                side=fill.side,
                volume=fill.volume,
                price=fill.price,
                sl=order.sl,
                tp=order.tp,
                commission=fill.commission,
                magic=order.magic,
                comment=order.comment,
                open_time=self.current_time,
            )

    # ------------------------------------------------------------------
    # Intent creation (replaces the engine's immediate order execution)
    # ------------------------------------------------------------------

    def _execute_signal(self, signal: Dict) -> None:
        """Signals become latency-carrying intents, not immediate orders.

        The base engine's ``_execute_signal`` calls ``create_order`` and the
        NEXT bar's ``_process_orders`` fills it. Here the order is still
        created identically (same OrderManager, same pending semantics) but
        its execution is deferred by the latency window via the intent book.
        """
        p = self.params
        if self.on_signal:
            self.on_signal(signal)

        symbol = signal.get("symbol")
        direction = signal.get("direction")
        bar = self.current_bars.get(symbol)
        entry_price = signal.get("entry_price", bar.close if bar else 0)
        sl = signal.get("sl")
        tp = signal.get("tp")
        volume = signal.get("volume", 1.0)
        if not symbol or direction is None:
            return

        from ..backtest.execution import OrderSide

        side = OrderSide.BUY if direction == "LONG" else OrderSide.SELL

        governed = self.governor is not None
        journal_key: Optional[str] = None
        if governed:
            # Write-ahead: the intent is journaled (or refused) BEFORE any
            # order exists. Duplicates are suppressed by key.
            key = f"sim-{self._bar_index:06d}-{len(self.order_manager.orders) + 1:04d}"
            record = self.governor.accept_intent(
                key,
                symbol=symbol,
                side=side.value,
                volume=volume,
                sl=sl,
                tp=tp,
                now=self.current_time,
                current_bar=self._bar_index,
                comment=str(signal.get("comment", "")),
            )
            if record is None:
                refusal = "governor_refused"
                if self.governor.events and "reason" in self.governor.events[-1]:
                    refusal = self.governor.events[-1]["reason"]
                self.rejection_count += 1
                self.rejected_intents.append(
                    {
                        "reason": refusal,
                        "symbol": symbol,
                        "side": side.value,
                        "volume": volume,
                        "timestamp": self.current_time,
                        "created_bar": self._bar_index,
                        "executed_bar": None,
                        "comment": str(signal.get("comment", "")),
                    }
                )
                return
            journal_key = key

        order = self.order_manager.create_order(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            volume=volume,
            sl=sl,
            tp=tp,
            magic=signal.get("magic", 0),
            comment=signal.get("comment", ""),
        )

        if not governed and self.mode_flags == {ExecutionMode.CLEAN}:
            # Clean mode: EXACT base semantics — the order stays pending and
            # the inherited base _process_orders fills it on the next bar.
            return
        if self.mode_flags == {ExecutionMode.CLEAN}:
            # Clean mode: EXACT base semantics — the order stays pending and
            # the inherited base _process_orders fills it on the next bar.
            return
        # Deterministic order id: fill records carry order_id, and the
        # determinism contract requires two runs to produce identical ones
        # (create_order mints a uuid4).
        self._rekey_order(order, f"sim-{self._bar_index:06d}-{len(self.order_manager.orders):04d}")
        latency = self._rng.randint(p.latency_bars_min, p.latency_bars_max)
        created = self._bar_index
        intent_key = f"{order.id}"
        self._intents[intent_key] = {
            "key": intent_key,
            "order": order,
            "created_bar": created,
            "latency_bars": latency,
            "due_bar": created + latency,
            "journal_key": journal_key,
        }

    def _on_run_end(self) -> None:
        """Flush diagnostics into the SimulatorResult-compatible fields."""
        # Any intents still pending at end-of-run never filled — record them.
        for key, intent in list(self._intents.items()):
            self.rejected_intents.append(
                {
                    "reason": "UNFILLED_AT_END",
                    "symbol": intent["order"].symbol,
                    "side": intent["order"].side.value,
                    "volume": intent["order"].volume,
                    "timestamp": self.current_time,
                    "created_bar": intent["created_bar"],
                    "executed_bar": None,
                    "comment": intent["order"].comment,
                }
            )
            if self.governor is not None and intent.get("journal_key"):
                self.governor.record_cancel(intent["journal_key"], "unfilled_at_end")
        self._intents.clear()
        self._intents_frozen = True

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------

    def result(self) -> SimulatorResult:
        base = self._generate_result()
        return SimulatorResult(
            trades=base.trades,
            equity_curve=base.equity_curve,
            metrics=base.metrics,
            fills=self.sim_fills,
            rejected_intents=self.rejected_intents,
            latency_histogram=self.latency_histogram,
            partial_fills=self.partial_fill_count,
            requotes=self.requote_count,
            rejections=self.rejection_count,
        )


def run_simulation(
    data: Dict[str, Any],
    signal_generator,
    config: BacktestConfig,
    params: Optional[ExecutionParams] = None,
    governor: Optional[ExecutionGovernor] = None,
) -> SimulatorResult:
    """Batch entry point mirroring ``run_backtest`` (including its logging)."""
    feed = DataFeed(data)
    engine = ExecutionSimulator(config, feed, signal_generator, params=params,
                                governor=governor)
    engine.run()
    engine._on_run_end()
    return engine.result()
