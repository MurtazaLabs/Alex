"""Stage E4: the live window runner -- the ONLY loop that may go to the wire.

This module is imported by ``scripts/e4_window_run.py`` and never
constructed inside the application package (the non-shadow adapter pin
enforces that a human-authorized script owns the vehicle). It runs one
guarded cycle:

0. risk gate: mandate baselines (day/week start equity, equity HWM) are
   synced from persisted state; an active daily/weekly/drawdown trip
   HALTS the cycle before anything else happens (no intents, no sends);
1. window gate: an active window state, an active SCOPE_WINDOW sign-off,
   the version's approved spec hash unchanged, and an unlocked broker --
   any failure blocks the cycle;
2. account sync: broker positions + account feed the tracker, which
   feeds the policy caps and the governor's dollar breaker;
3. signal: the pinned spec is evaluated on the latest CLOSED H1 bar with
   exactly the same indicator math as the backtest engine;
4. intent: the policy turns a signal into a deterministic OrderIntent
   (or a recorded refusal) -- intents are appended to an audit log;
5. dispatch: an intent is transmitted ONLY after a human approval record
   exists for its key. The runner then grants a single-use token in its
   own process and calls the transmitter, which re-runs the full gate
   chain; every transmission outcome is appended to the audit log.

Plan evidence before the window opens, per trade after it does. No LLM,
no retries, no second vehicle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from .caps import DEFAULT_LIVE_CAPS, LiveCaps
from .risk_state import BaselineSnapshot, RiskBaselineStore
from .signoff import SCOPE_WINDOW, SignoffStore
from .tracker import PositionTracker, stop_risk_usd, sync_account_into
from .transmitter import TRANSMIT_CRITICAL, LiveTransmitter
from .window import (
    DEFAULT_DRILL_DIR,
    DEFAULT_SIGNOFF_PATH,
    DEFAULT_VERSIONS_DIR,
    DEFAULT_WINDOW_STATE_DIR,
    WindowStore,
    read_version_file,
)
from ..policy import INTENT_VALID, OrderIntent
from ...backtest.engine import BarData
from ...strategy.spec import parse_spec
from ...strategy.versioning import spec_hash
from ...autonomy.validator import SpecSignalGenerator
from ...utils.math_helpers import pip_size_for_symbol, pip_value_per_lot

__all__ = [
    "indicator_snapshot",
    "LiveWindowRunner",
]

_WARMUP_BARS = 51
_LOOKBACK_BARS = 201
_BARS_TO_FETCH = 800


def _ema(data: np.ndarray, period: int) -> float:
    if len(data) < period:
        return float(data[-1])
    alpha = 2 / (period + 1)
    ema = data[-period]
    for v in data[-period + 1:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)


def _atr(
    highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int
) -> float:
    if len(highs) < period + 1:
        return float(highs[-1] - lows[-1])
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])),
    )
    return float(np.mean(tr[-period:]))


def _rsi(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def indicator_snapshot(df: pd.DataFrame) -> Optional[Dict[str, float]]:
    """The backtest engine's exact indicator math over the trailing window.

    Mirrors ``BacktestEngine._calculate_indicators`` (same 201-bar window,
    same formulas) so a live signal sees precisely what the validated
    spec saw in the backtest. Returns None while warming up.
    """
    if df is None or len(df) < _WARMUP_BARS:
        return None
    lookback = df.iloc[max(0, len(df) - _LOOKBACK_BARS):]
    closes = lookback["close"].values.astype(float)
    highs = lookback["high"].values.astype(float)
    lows = lookback["low"].values.astype(float)

    sma_20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(closes[-1])
    bb_window = closes[-20:] if len(closes) >= 20 else closes
    bb_middle = float(np.mean(bb_window))
    bb_std = float(np.std(bb_window))
    return {
        "sma_20": sma_20,
        "sma_50": (
            float(np.mean(closes[-50:])) if len(closes) >= 50 else float(closes[-1])
        ),
        "ema_20": _ema(closes, 20),
        "atr_14": _atr(highs, lows, closes, 14),
        "rsi_14": _rsi(closes, 14),
        "bb_middle": bb_middle,
        "bb_upper": bb_middle + 2.0 * bb_std,
        "bb_lower": bb_middle - 2.0 * bb_std,
    }


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def intent_from_record(record: Dict[str, Any]) -> OrderIntent:
    """Rebuild the exact policy intent that was recorded for audit."""
    return OrderIntent(
        key=str(record["key"]),
        symbol=str(record["symbol"]),
        side=str(record["side"]),
        volume=float(record["volume"]),
        sl=float(record["sl"]),
        tp=float(record["tp"]),
        entry=float(record["entry"]),
        risk_usd=float(record.get("risk_usd", 0.0)),
        sl_pips=float(record.get("sl_pips", 0.0)),
        rr_ratio=float(record.get("rr_ratio", 0.0)),
        equity=float(record.get("equity", 0.0)),
        created_at=datetime.fromisoformat(str(record["created_at"])),
        expires_at=datetime.fromisoformat(str(record["expires_at"])),
        strategy_id=str(record.get("strategy_id", "")),
        source_id=str(record.get("source_id", "")),
        comment=str(record.get("comment", "")),
        status=str(record.get("status", INTENT_VALID)),
        refusal_reason=str(record.get("refusal_reason", "")),
    )


@dataclass
class _GateResult:
    ok: bool
    reason: str = ""


class LiveWindowRunner:
    """One guarded cycle of the live window. Fail closed, no retries."""

    def __init__(
        self,
        *,
        adapter: Any,
        feed: Any,
        tracker: PositionTracker,
        policy: Any,
        governor: Any,
        tokens: Any,
        version_id: str,
        caps: LiveCaps = DEFAULT_LIVE_CAPS,
        transmitter: Optional[Any] = None,
        state_dir: Union[str, Path] = DEFAULT_WINDOW_STATE_DIR,
        versions_dir: Union[str, Path] = DEFAULT_VERSIONS_DIR,
        signoff_path: Union[str, Path] = DEFAULT_SIGNOFF_PATH,
        drill_dir: Union[str, Path] = DEFAULT_DRILL_DIR,
        daily_loss_pct: Optional[float] = None,
        risk_store: Optional[RiskBaselineStore] = None,
    ) -> None:
        self.adapter = adapter
        self.feed = feed
        self.tracker = tracker
        self.policy = policy
        self.governor = governor
        self.tokens = tokens
        self.version_id = version_id
        self.caps = caps
        self.pair = caps.pair
        self.transmitter = transmitter or LiveTransmitter(adapter, tokens, governor)
        self.state_dir = Path(state_dir)
        self.versions_dir = versions_dir
        self.signoff_path = signoff_path
        self.drill_dir = drill_dir
        self.daily_loss_pct = (
            float(daily_loss_pct)
            if daily_loss_pct is not None
            else float(caps.max_daily_loss_pct)
        )
        self.store = WindowStore(state_dir)
        self.signoffs = SignoffStore(signoff_path)
        self.risk_store = risk_store or RiskBaselineStore(
            self.state_dir / "risk_baselines.json"
        )
        self.intents_path = self.state_dir / "intents.jsonl"
        self.approvals_path = self.state_dir / "approvals.jsonl"
        self.transmissions_path = self.state_dir / "transmissions.jsonl"
        self._generator: Optional[SpecSignalGenerator] = None

    # ------------------------------------------------------------------
    # Gates
    # ------------------------------------------------------------------

    def _gate(self, now: datetime) -> _GateResult:
        state = self.store.load_state()
        if state is None:
            return _GateResult(False, "no window state -- run e4_window_start.py --start")
        if not state.is_active(now):
            return _GateResult(
                False, f"window {state.window_id} is {state.status} (not active)",
            )
        if state.version_id != self.version_id:
            return _GateResult(
                False,
                f"window runs version {state.version_id}, runner configured "
                f"for {self.version_id}",
            )
        signoff = self.signoffs.latest(
            scope=SCOPE_WINDOW, now=now, active_only=True,
        )
        if signoff is None or signoff.pair != self.pair:
            return _GateResult(False, "no active SCOPE_WINDOW sign-off for this pair")
        if signoff.strategy_id and signoff.strategy_id != self.version_id:
            return _GateResult(
                False, "active sign-off names a different strategy version",
            )
        version = read_version_file(self.versions_dir, self.version_id)
        if version is None:
            return _GateResult(False, f"version file {self.version_id}.json missing")
        approval = version.get("approval") or {}
        pinned = str(approval.get("spec_hash", ""))
        current = spec_hash(version.get("rules") or {}, version.get("parameters") or {})
        if str(version.get("status", "")) != "active":
            return _GateResult(False, "version is no longer active")
        if not pinned or pinned != current:
            return _GateResult(False, "spec changed since human approval -- pin broken")
        level = int(getattr(self.adapter, "permission_level", 0) or 0)
        shadow = bool(getattr(self.adapter, "shadow_mode", True))
        if level < 4 or shadow:
            return _GateResult(
                False, f"broker gate closed (level={level} shadow={shadow})",
            )
        return _GateResult(True)

    # ------------------------------------------------------------------
    # Mandate risk gate (baselines + trips)
    # ------------------------------------------------------------------

    def _risk_gate(
        self, snapshot: BaselineSnapshot, equity: float, now: datetime,
    ) -> List[str]:
        """Persist any breached mandate halt and return the active trips.

        Daily/weekly trips expire with their day/week key rollover (handled
        by ``RiskBaselineStore.sync``); a drawdown trip persists until a
        human runs ``--clear-trip drawdown``.
        """
        caps = self.caps
        if not snapshot.tripped("daily") and snapshot.day_start_equity > 0:
            loss_pct = snapshot.daily_loss_pct(equity)
            if loss_pct >= caps.max_daily_loss_pct:
                self.risk_store.trip("daily", now=now, detail={
                    "equity": round(equity, 2),
                    "day_start_equity": round(snapshot.day_start_equity, 2),
                    "loss_pct": round(loss_pct, 4),
                    "limit_pct": caps.max_daily_loss_pct,
                })
        if not snapshot.tripped("weekly") and snapshot.week_start_equity > 0:
            loss_pct = snapshot.weekly_loss_pct(equity)
            if loss_pct >= caps.max_weekly_loss_pct:
                self.risk_store.trip("weekly", now=now, detail={
                    "equity": round(equity, 2),
                    "week_start_equity": round(snapshot.week_start_equity, 2),
                    "loss_pct": round(loss_pct, 4),
                    "limit_pct": caps.max_weekly_loss_pct,
                })
        if not snapshot.tripped("drawdown") and snapshot.equity_hwm > 0:
            dd_pct = snapshot.drawdown_pct(equity)
            if dd_pct >= caps.max_drawdown_pct:
                self.risk_store.trip("drawdown", now=now, detail={
                    "equity": round(equity, 2),
                    "equity_hwm": round(snapshot.equity_hwm, 2),
                    "drawdown_pct": round(dd_pct, 4),
                    "limit_pct": caps.max_drawdown_pct,
                })
        return self.risk_store.snapshot().active_trips()

    def _pending_risk_usd(self, now: datetime) -> Optional[float]:
        """Stop risk of recorded-but-unsent VALID intents, in USD.

        Returns None (fail closed) when any pending intent cannot be
        verified -- unknown symbol or malformed prices.
        """
        done = {t.get("intent_key") for t in _read_jsonl(self.transmissions_path)}
        total = 0.0
        for record in _read_jsonl(self.intents_path):
            if record.get("status") != INTENT_VALID:
                continue
            key = str(record.get("key", ""))
            if not key or key in done or not order_key_ok(record, now):
                continue
            symbol = str(record.get("symbol", ""))
            if "/" not in symbol:
                return None
            try:
                entry = float(record["entry"])
                sl = float(record["sl"])
                volume = float(record["volume"])
            except (KeyError, TypeError, ValueError):
                return None
            pip_size = pip_size_for_symbol(symbol)
            pips = abs(entry - sl) / pip_size
            total += pips * pip_value_per_lot(symbol) * abs(volume)
        return round(total, 4)

    def _aggregate_stop_risk(
        self, positions: List[Any], now: datetime,
    ) -> Optional[float]:
        """Open positions + pending intents stop risk, or None if unknown."""
        open_risk = stop_risk_usd(positions, symbol=self.pair)
        pending_risk = self._pending_risk_usd(now)
        if open_risk is None or pending_risk is None:
            return None
        return round(open_risk + pending_risk, 4)

    def _spec(self) -> Tuple[Optional[Any], str]:
        version = read_version_file(self.versions_dir, self.version_id)
        if version is None:
            return None, "version file missing"
        spec, errors = parse_spec(version.get("rules") or {})
        if spec is None:
            return None, f"spec invalid: {'; '.join(errors) if errors else 'unknown'}"
        return spec, ""

    # ------------------------------------------------------------------
    # Signals -> intents
    # ------------------------------------------------------------------

    async def _latest_closed_bar(self, now: datetime) -> Optional[pd.DataFrame]:
        start = now - timedelta(days=30)
        bars = await self.feed.fetch_candles(
            self.pair, "H1", start, end=now, limit=_BARS_TO_FETCH,
        )
        if not bars:
            return None
        return pd.DataFrame(
            {
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [float(getattr(b, "volume", 0) or 0) for b in bars],
            },
            index=[b.timestamp for b in bars],
        )

    def _produce_intents(
        self, now: datetime, bars: pd.DataFrame, equity: float,
    ) -> Dict[str, Any]:
        spec, error = self._spec()
        if spec is None:
            return {"signals": 0, "valid": 0, "refused": 0, "error": error}
        snapshot = indicator_snapshot(bars)
        if snapshot is None:
            return {"signals": 0, "valid": 0, "refused": 0,
                    "error": f"warming up ({len(bars)} bars)"}
        if self._generator is None:
            self._generator = SpecSignalGenerator(
                spec,
                risk_per_trade_pct=float(self.caps.max_risk_per_trade_pct),
                account_balance=equity,
            )
        last = bars.iloc[-1]
        bar = BarData(
            timestamp=bars.index[-1].to_pydatetime(),
            open=float(last["open"]), high=float(last["high"]),
            low=float(last["low"]), close=float(last["close"]),
            volume=float(last.get("volume", 0.0)),
        )
        signals = self._generator.generate_signals(
            now, {self.pair: bar}, {self.pair: snapshot},
        )

        done = {t.get("intent_key") for t in _read_jsonl(self.transmissions_path)}
        recorded = _read_jsonl(self.intents_path)
        pending_sides = {
            str(r.get("side")) for r in recorded
            if r.get("status") == INTENT_VALID
            and r.get("key") not in done
            and order_key_ok(r, now)
        }

        valid = refused = 0
        for signal in signals:
            side = "BUY" if str(signal.get("direction")) == "LONG" else "SELL"
            if side in pending_sides:
                continue
            intent = self.policy.create_intent(
                symbol=self.pair,
                side=side,
                entry=float(signal["entry_price"]),
                stop_loss=float(signal["sl"]),
                take_profit=float(signal["tp"]),
                strategy_id=self.version_id,
                source_id="e4-window",
                comment=str(signal.get("comment", "e4 window")),
                now=now,
            )
            record = intent.to_dict()
            record["recorded_at"] = now.isoformat()
            if intent.status == INTENT_VALID and not (
                float(self.caps.min_lot) <= float(intent.volume) <= float(self.caps.max_lot)
            ):
                record["status"] = "REFUSED_WINDOW_LOT_CAP"
                record["refusal_reason"] = (
                    f"policy volume {intent.volume} outside window lot band "
                    f"[{self.caps.min_lot}, {self.caps.max_lot}]"
                )
            _append_jsonl(self.intents_path, record)
            if record["status"] == INTENT_VALID:
                valid += 1
                pending_sides.add(side)
            else:
                refused += 1
        return {"signals": len(signals), "valid": valid, "refused": refused,
                "error": ""}

    # ------------------------------------------------------------------
    # Approval -> token -> transmitter
    # ------------------------------------------------------------------

    def _dispatch(self, now: datetime) -> Dict[str, Any]:
        approvals = _read_jsonl(self.approvals_path)
        transmissions = _read_jsonl(self.transmissions_path)
        done = {t.get("intent_key") for t in transmissions}
        intents: Dict[str, Dict[str, Any]] = {}
        for record in _read_jsonl(self.intents_path):
            intents[str(record.get("key"))] = record

        dispatched = 0
        critical = False
        skipped = 0
        for approval in approvals:
            key = str(approval.get("intent_key", ""))
            if not key or key in done:
                continue
            record = intents.get(key)
            if record is None:
                _append_jsonl(self.transmissions_path, {
                    "intent_key": key, "status": "APPROVAL_WITHOUT_INTENT",
                    "approval_id": approval.get("approval_id", ""),
                    "at": now.isoformat(),
                })
                skipped += 1
                continue
            intent = intent_from_record(record)
            if intent.status != INTENT_VALID:
                _append_jsonl(self.transmissions_path, {
                    "intent_key": key, "status": "REFUSED_BY_POLICY",
                    "reason": intent.refusal_reason, "at": now.isoformat(),
                })
                skipped += 1
                continue
            if intent.is_expired(now):
                _append_jsonl(self.transmissions_path, {
                    "intent_key": key, "status": "EXPIRED_BEFORE_SEND",
                    "at": now.isoformat(),
                })
                skipped += 1
                continue
            token = self.tokens.grant(
                intent_key=intent.key,
                symbol=intent.symbol,
                side=intent.side,
                volume=intent.volume,
                granted_by=str(approval.get("approver", "operator")),
                intent_snapshot=intent.to_dict(),
                now=now,
            )
            result = self.transmitter.transmit(intent, token.token_id, now=now)
            _append_jsonl(self.transmissions_path, {
                "intent_key": key,
                "status": result.status,
                "sent": bool(result.sent),
                "ticket": getattr(result, "ticket", None),
                "approval_id": approval.get("approval_id", ""),
                "approver": approval.get("approver", ""),
                "at": now.isoformat(),
            })
            dispatched += 1
            if result.status == TRANSMIT_CRITICAL:
                critical = True
        return {"dispatched": dispatched, "skipped": skipped, "critical": critical}

    # ------------------------------------------------------------------
    # One cycle
    # ------------------------------------------------------------------

    async def run_once(self, *, now: Optional[datetime] = None) -> Dict[str, Any]:
        check_time = now or datetime.utcnow()
        gate = self._gate(check_time)
        if not gate.ok:
            return {"status": "blocked", "reason": gate.reason}

        try:
            account = self.adapter.account()
        except Exception as exc:
            self.store.append_event({
                "event": "cycle_blocked",
                "reason": f"account unavailable: {type(exc).__name__}: {exc}",
            })
            return {"status": "blocked",
                    "reason": f"account unavailable: {type(exc).__name__}: {exc}"}

        try:
            positions = self.adapter.positions()
        except Exception:
            positions = []
        self.tracker.sync_positions(positions, now=check_time)

        # Mandate risk gate: sync baselines on the live equity, persist any
        # breach, and halt the cycle (no intents, no sends) if tripped.
        equity_now = float(getattr(account, "equity", 0.0) or 0.0)
        if equity_now <= 0.0:
            equity_now = float(getattr(account, "balance", 0.0) or 0.0)
        snapshot = self.risk_store.sync(equity_now, check_time)
        active_trips = self._risk_gate(snapshot, equity_now, check_time)
        if active_trips:
            self.store.append_event({
                "event": "cycle_halted",
                "trips": active_trips,
                "equity": equity_now,
                "reason": "mandate risk halt active: " + ", ".join(active_trips),
            })
            return {
                "status": "halted",
                "at": check_time.isoformat(),
                "equity": equity_now,
                "trips": active_trips,
                "risk": self.risk_store.snapshot().to_dict(),
                "reason": "mandate risk halt active: " + ", ".join(active_trips),
            }

        aggregate_risk = self._aggregate_stop_risk(positions, check_time)
        feed_state = sync_account_into(
            self.tracker, account,
            policy=self.policy, governor=self.governor,
            daily_loss_pct=self.daily_loss_pct, daily_pnl=0.0,
            day_start_equity=snapshot.day_start_equity,
            week_start_equity=snapshot.week_start_equity,
            equity_hwm=snapshot.equity_hwm,
            open_stop_risk_usd=aggregate_risk,
            now=check_time,
        )

        # Keep the governor's dollar breakers on the mandate baselines.
        limits = getattr(self.governor, "limits", None)
        if limits is not None:
            limits.daily_loss_limit = round(
                snapshot.day_start_equity * (self.caps.max_daily_loss_pct / 100.0), 2,
            )
            limits.max_drawdown = round(
                snapshot.equity_hwm * (self.caps.max_drawdown_pct / 100.0), 2,
            )
            if hasattr(limits, "weekly_loss_limit"):
                limits.weekly_loss_limit = round(
                    snapshot.week_start_equity
                    * (self.caps.max_weekly_loss_pct / 100.0),
                    2,
                )

        bars = await self._latest_closed_bar(check_time)
        if bars is None or len(bars) < _WARMUP_BARS:
            return {"status": "no_data",
                    "bars": 0 if bars is None else len(bars),
                    "equity": feed_state.equity}

        produced = self._produce_intents(check_time, bars, feed_state.equity)
        dispatched = self._dispatch(check_time)
        summary = {
            "status": "ok",
            "at": check_time.isoformat(),
            "equity": feed_state.equity,
            "open_positions": feed_state.open_positions,
            "realized_pnl_today": feed_state.realized_pnl_today,
            "signals": produced["signals"],
            "intents_valid": produced["valid"],
            "intents_refused": produced["refused"],
            "risk": {
                "trips": snapshot.active_trips(),
                "day_start_equity": snapshot.day_start_equity,
                "week_start_equity": snapshot.week_start_equity,
                "equity_hwm": snapshot.equity_hwm,
                "open_stop_risk_usd": aggregate_risk,
            },
            **dispatched,
        }
        if produced.get("error"):
            summary["note"] = produced["error"]
        return summary


def order_key_ok(record: Dict[str, Any], now: datetime) -> bool:
    """True while a recorded VALID intent has not yet expired."""
    try:
        return datetime.fromisoformat(str(record["expires_at"])) > now
    except Exception:
        return False
