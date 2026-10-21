"""Stage E3: execution policy + approval workflow API endpoints.

Endpoints:
- POST /execution/intent          — create an order intent from a signal
- GET  /execution/intent/{key}    — inspect an intent
- GET  /execution/intents         — list recent intents
- POST /execution/intent/{key}/approve — grant approval token (human)
- POST /execution/token/consume   — consume token (transmission path)
- POST /execution/token/{id}/revoke  — revoke a token (human)
- GET  /execution/tokens          — list active tokens
- GET  /execution/audit           — audit log (tokens + governor events)
- POST /execution/governor/arm    — arm the execution governor (human)
- POST /execution/governor/trip   — trip the kill switch (human)
- GET  /execution/governor/status — governor state
- GET  /execution/window/status — E4 window state + fresh eligibility
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ....execution.governor import ExecutionGovernor, GovernorLimits, TRIP_MANUAL
from ....execution.policy import ExecutionPolicy, PolicyCaps, OrderIntent
from ....execution.tokens import ApprovalTokenStore, ApprovalToken
from ....notify.discord import get_notifier

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/execution", tags=["execution"])


# ---------------------------------------------------------------------------
# Shared singletons (process-lifetime; in-memory for E3)
# ---------------------------------------------------------------------------

_caps = PolicyCaps()  # hard-coded defaults — the non-negotiable constants
_policy = ExecutionPolicy(_caps)
_token_store = ApprovalTokenStore()
_governor = ExecutionGovernor(GovernorLimits(
    daily_loss_limit=1.0,  # mandate daily cap on the $100 demo (1%); /account-state recomputes
    max_drawdown=5.0,      # mandate 5% halt from the equity high-water mark
    max_positions=_caps.max_concurrent_positions,
    trip_position_policy="POSITION_HOLD",
))
_intent_log: list = []  # recent intents (E4 persists to DB)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class IntentRequest(BaseModel):
    symbol: str = Field(..., examples=["EUR/USD"])
    side: str = Field(..., pattern="^(BUY|SELL)$")
    entry: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    strategy_id: str = ""
    source_id: str = ""
    comment: str = ""


class AccountStateRequest(BaseModel):
    equity: float = Field(..., gt=0, description="Current account equity in USD")
    open_positions: int = Field(..., ge=0)
    daily_pnl: float = 0.0


class ApproveRequest(BaseModel):
    granted_by: str = Field(..., min_length=1, description="Human operator identity")


class ConsumeRequest(BaseModel):
    token_id: str
    intent_key: str
    symbol: str
    side: str
    volume: float


class RevokeRequest(BaseModel):
    revoked_by: str = ""


class TripRequest(BaseModel):
    reason: str = Field(default="manual", max_length=200)


# ---------------------------------------------------------------------------
# Account state (caller feeds this each cycle)
# ---------------------------------------------------------------------------

@router.post("/account-state")
async def update_account_state(req: AccountStateRequest) -> dict:
    """Update the policy's view of the account (call each trading cycle)."""
    _policy.update_account_state(
        equity=req.equity,
        open_positions=req.open_positions,
        daily_pnl=req.daily_pnl,
    )
    # Update the governor's dollar halt from the fresh equity (mandate % constant)
    _governor.limits.daily_loss_limit = req.equity * (_caps.max_daily_loss_pct / 100.0)
    return {"status": "ok", "equity": req.equity, "open_positions": req.open_positions}


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------

@router.post("/intent")
async def create_intent(req: IntentRequest) -> dict:
    """Create a fully specified order intent, validated against all caps.

    Returns the intent (VALID or refused with a reason). Nothing is sent.
    """
    intent = _policy.create_intent(
        symbol=req.symbol, side=req.side, entry=req.entry,
        stop_loss=req.stop_loss, take_profit=req.take_profit,
        strategy_id=req.strategy_id, source_id=req.source_id,
        comment=req.comment,
    )
    _intent_log.append(intent)
    # Discord: notify on valid intents (needs approval)
    if intent.status == "VALID":
        notifier = get_notifier()
        if notifier.enabled:
            import asyncio
            asyncio.create_task(notifier.intent_created(
                symbol=intent.symbol, side=intent.side, volume=intent.volume,
                entry=intent.entry, sl=intent.sl, tp=intent.tp,
                risk_usd=intent.risk_usd, rr_ratio=intent.rr_ratio,
                intent_key=intent.key,
            ))
    return intent.to_dict()


@router.get("/intent/{key}")
async def get_intent(key: str) -> dict:
    """Inspect an intent by key."""
    for i in reversed(_intent_log):
        if i.key == key:
            return i.to_dict()
    raise HTTPException(status_code=404, detail=f"Intent {key} not found")


@router.get("/intents")
async def list_intents(limit: int = 20) -> dict:
    """List recent intents (most recent first)."""
    recent = [i.to_dict() for i in reversed(_intent_log[-limit:])]
    return {"intents": recent, "count": len(recent)}


# ---------------------------------------------------------------------------
# Approval tokens
# ---------------------------------------------------------------------------

@router.post("/intent/{key}/approve")
async def approve_intent(key: str, req: ApproveRequest) -> dict:
    """HUMAN action: grant a single-use approval token for an intent.

    The token is bound to (intent_key, symbol, side, volume) and expires
    in 10 minutes. It can be consumed exactly once.
    """
    # Find the intent
    intent = None
    for i in reversed(_intent_log):
        if i.key == key:
            intent = i
            break
    if intent is None:
        raise HTTPException(status_code=404, detail=f"Intent {key} not found")
    if intent.status != "VALID":
        raise HTTPException(
            status_code=400,
            detail=f"Intent is {intent.status}: {intent.refusal_reason}",
        )
    token = _token_store.grant(
        intent_key=intent.key, symbol=intent.symbol, side=intent.side,
        volume=intent.volume, granted_by=req.granted_by,
        intent_snapshot=intent.to_dict(),
    )
    # Discord: notify on token grant
    notifier = get_notifier()
    if notifier.enabled:
        import asyncio
        asyncio.create_task(notifier.token_granted(
            token_id=token.token_id, intent_key=intent.key,
            symbol=intent.symbol, side=intent.side,
            volume=intent.volume, granted_by=req.granted_by,
        ))
    return token.to_dict()


@router.post("/tokens/grant")
async def grant_token(
    intent_key: str, symbol: str, side: str, volume: float,
    granted_by: str, intent_snapshot: Optional[str] = None,
) -> dict:
    """HUMAN action: grant an approval token for a prepared intent."""
    import json
    snapshot = json.loads(intent_snapshot) if intent_snapshot else None
    token = _token_store.grant(
        intent_key=intent_key, symbol=symbol, side=side,
        volume=volume, granted_by=granted_by,
        intent_snapshot=snapshot,
    )
    return token.to_dict()


@router.post("/token/consume")
async def consume_token(req: ConsumeRequest) -> dict:
    """Consume a token to authorize order transmission (E4 path)."""
    ok = _token_store.consume(
        req.token_id,
        intent_key=req.intent_key,
        symbol=req.symbol,
        side=req.side,
        volume=req.volume,
    )
    if not ok:
        raise HTTPException(
            status_code=403,
            detail="Token invalid, expired, consumed, or order fields mismatch",
        )
    return {"status": "consumed", "token_id": req.token_id}


@router.post("/token/{token_id}/revoke")
async def revoke_token(token_id: str, req: RevokeRequest) -> dict:
    """HUMAN action: revoke a granted token before consumption."""
    ok = _token_store.revoke(token_id, revoked_by=req.revoked_by)
    if not ok:
        raise HTTPException(status_code=404, detail="Token not found or not active")
    return {"status": "revoked", "token_id": token_id}


@router.get("/tokens")
async def list_tokens() -> dict:
    """List all active (unconsumed, unexpired) tokens."""
    tokens = _token_store.active_tokens()
    return {"tokens": [t.to_dict() for t in tokens], "count": len(tokens)}


# ---------------------------------------------------------------------------
# Governor / kill switch
# ---------------------------------------------------------------------------

@router.get("/governor/status")
async def governor_status() -> dict:
    """Current governor state (ARMED / DISARMED / HALTED)."""
    state = _governor.state
    return {
        "state": state.state,
        "tripped": state.tripped,
        "trip_reason": _governor.trip_reason,
        "tripped_at": _governor.tripped_at.isoformat() if _governor.tripped_at else None,
        "pending_intents": state.pending_intents,
        "policy_caps": {
            "max_risk_per_trade_pct": _caps.max_risk_per_trade_pct,
            "max_daily_loss_pct": _caps.max_daily_loss_pct,
            "max_concurrent_positions": _caps.max_concurrent_positions,
            "min_lot_size": _caps.min_lot_size,
            "intent_expiry_seconds": _caps.intent_expiry_seconds,
        },
    }


@router.post("/governor/arm")
async def arm_governor() -> dict:
    """HUMAN action: arm the execution governor (required before any intent)."""
    _governor.arm()
    return {"status": "armed", "state": _governor.state.state}


@router.post("/governor/trip")
async def trip_governor(req: TripRequest) -> dict:
    """HUMAN action: trip the kill switch (halts all execution)."""
    _governor.trip(req.reason or TRIP_MANUAL)
    # Discord: alert on kill switch
    notifier = get_notifier()
    if notifier.enabled:
        import asyncio
        asyncio.create_task(notifier.kill_switch_tripped(reason=_governor.trip_reason or "manual"))
    return {
        "status": "tripped",
        "state": _governor.state.state,
        "reason": _governor.trip_reason,
    }


@router.get("/governor/audit")
async def governor_audit() -> dict:
    """Governor event log (trips, refusals, flatten records)."""
    return {"events": _governor.events[-100:], "flattened": _governor.flattened}


# ---------------------------------------------------------------------------
# Token audit
# ---------------------------------------------------------------------------

@router.get("/tokens/audit")
async def token_audit() -> dict:
    """Approval token audit log (grants, consumes, revokes, expiries)."""
    return {"audit": _token_store.audit_log()}


# ---------------------------------------------------------------------------
# E4 window (read-only; the live loop itself is a human-authorized script)
# ---------------------------------------------------------------------------

@router.get("/window/status")
async def window_status_endpoint() -> dict:
    """E4 window state, sign-off and FRESH eligibility (fail closed).

    Read-only: this endpoint can never start, stop or trade anything.
    """
    from ....execution.live.window import check_eligibility, window_status

    status = window_status()
    state = status.get("state")
    if state:
        try:
            eligibility = await check_eligibility(
                version_id=str(state["version_id"]),
                pair=str(state.get("pair", "EUR/USD")),
            )
            status["eligibility"] = eligibility.to_dict()
        except Exception as exc:  # noqa: BLE001 - status stays readable
            status["eligibility"] = {"error": f"{type(exc).__name__}: {exc}"}
    else:
        status["eligibility"] = {
            "all_met": False, "checks": [],
            "note": "no window started",
        }
    return status
