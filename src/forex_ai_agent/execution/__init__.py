"""Deterministic execution package — simulator, policy, tokens, governor.

Stages:
- E0: ExecutionSimulator (latency, fills, requotes — no network)
- E1: ExecutionGovernor + IntentJournal (kill switch, breakers, write-ahead)
- E3: ExecutionPolicy + ApprovalTokenStore (order intents, human approval)
- E4: MT5BrokerAdapter (live transmission — gated)
"""

from .simulator import (
    ExecutionSimulator,
    ExecutionMode,
    ExecutionParams,
    SimulatorResult,
)
from .governor import (
    ExecutionGovernor,
    GovernorLimits,
    GovernorState,
    TRIP_MANUAL,
    TRIP_DAILY_LOSS,
    TRIP_DRAWDOWN,
    TRIP_POSITION_COUNT,
    TRIP_POSITION_HOLD,
    TRIP_POSITION_FLATTEN,
    BROKER_FAULT_TABLE,
)
from .journal import IntentJournal, INTENT_PENDING, INTENT_EXECUTED, INTENT_CANCELLED
from .policy import (
    ExecutionPolicy,
    PolicyCaps,
    OrderIntent,
    INTENT_VALID,
    INTENT_EXPIRED,
    INTENT_REFUSED_RISK_CAP,
    INTENT_REFUSED_POSITION_CAP,
    INTENT_REFUSED_DAILY_LOSS,
    INTENT_REFUSED_UNTRADEABLE,
    INTENT_REFUSED_GOVERNOR,
)
from .tokens import (
    ApprovalTokenStore,
    ApprovalToken,
    TOKEN_ACTIVE,
    TOKEN_CONSUMED,
    TOKEN_EXPIRED,
    TOKEN_REVOKED,
)

__all__ = [
    # E0 simulator
    "ExecutionSimulator", "ExecutionMode", "ExecutionParams", "SimulatorResult",
    # E1 governor + journal
    "ExecutionGovernor", "GovernorLimits", "GovernorState",
    "TRIP_MANUAL", "TRIP_DAILY_LOSS", "TRIP_DRAWDOWN", "TRIP_POSITION_COUNT",
    "TRIP_POSITION_HOLD", "TRIP_POSITION_FLATTEN", "BROKER_FAULT_TABLE",
    "IntentJournal", "INTENT_PENDING", "INTENT_EXECUTED", "INTENT_CANCELLED",
    # E3 policy
    "ExecutionPolicy", "PolicyCaps", "OrderIntent",
    "INTENT_VALID", "INTENT_EXPIRED",
    "INTENT_REFUSED_RISK_CAP", "INTENT_REFUSED_POSITION_CAP",
    "INTENT_REFUSED_DAILY_LOSS", "INTENT_REFUSED_UNTRADEABLE",
    "INTENT_REFUSED_GOVERNOR",
    # E3 tokens
    "ApprovalTokenStore", "ApprovalToken",
    "TOKEN_ACTIVE", "TOKEN_CONSUMED", "TOKEN_EXPIRED", "TOKEN_REVOKED",
]
