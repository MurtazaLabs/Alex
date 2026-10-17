"""Confirmation policy engine for action tools (Phase 2).

Each action tool declares a risk class:
- Low: execute immediately (read tools, starting research)
- Medium: two-step with confirm token (10-min expiry)
- High: hard-gated by permission level; at level 0 returns "insufficient permission"
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RiskClass(Enum):
    """Risk classification for action tools."""
    LOW = "low"           # execute immediately
    MEDIUM = "medium"     # two-step with confirm token
    HIGH = "high"         # hard-gated by permission level


@dataclass
class ConfirmationToken:
    """Single-use confirmation token for medium-risk actions."""
    token: str
    tool_name: str
    args_summary: str
    expires_at: float
    created_at: float = field(default_factory=time.time)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    @property
    def ttl_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


@dataclass
class ActionProposal:
    """Proposal returned by medium-risk tools awaiting confirmation."""
    tool_name: str
    args: Dict[str, Any]
    impact_summary: str
    confirmation_token: ConfirmationToken
    risk_class: RiskClass = RiskClass.MEDIUM


@dataclass
class ActionResult:
    """Result of an action tool execution."""
    success: bool
    tool_name: str
    result: Any = None
    error: Optional[str] = None
    proposal: Optional[ActionProposal] = None
    requires_confirmation: bool = False


class ConfirmationPolicy:
    """Manages confirmation tokens and policy for action tools."""

    TOKEN_TTL_SECONDS = 600  # 10 minutes

    def __init__(self):
        self._pending: Dict[str, ConfirmationToken] = {}
        self._used: set[str] = set()

    def create_token(self, tool_name: str, args_summary: str) -> ConfirmationToken:
        """Create a new confirmation token."""
        token = ConfirmationToken(
            token=secrets.token_urlsafe(32),
            tool_name=tool_name,
            args_summary=args_summary,
            expires_at=time.time() + self.TOKEN_TTL_SECONDS,
        )
        self._pending[token.token] = token
        return token

    def consume_token(self, token: str) -> Optional[ConfirmationToken]:
        """Consume a confirmation token (single-use)."""
        if token in self._used:
            return None
        if token not in self._pending:
            return None
        t = self._pending.pop(token)
        if t.is_expired:
            return None
        self._used.add(token)
        return t

    def get_pending(self, tool_name: Optional[str] = None) -> list[ConfirmationToken]:
        """Get pending tokens, optionally filtered by tool."""
        tokens = list(self._pending.values())
        if tool_name:
            tokens = [t for t in tokens if t.tool_name == tool_name]
        return [t for t in tokens if not t.is_expired]

    def cleanup_expired(self) -> int:
        """Remove expired tokens. Returns count removed."""
        expired = [t for t in self._pending.values() if t.is_expired]
        for t in expired:
            self._pending.pop(t.token, None)
        return len(expired)


# Global confirmation policy instance
_policy: Optional[ConfirmationPolicy] = None


def get_confirmation_policy() -> ConfirmationPolicy:
    """Get or create the global confirmation policy."""
    global _policy
    if _policy is None:
        _policy = ConfirmationPolicy()
    return _policy


# Tool risk class registry
_TOOL_RISK_CLASS: Dict[str, RiskClass] = {}


def register_tool_risk(tool_name: str, risk_class: RiskClass) -> None:
    """Register a tool's risk class."""
    _TOOL_RISK_CLASS[tool_name] = risk_class


def get_tool_risk_class(tool_name: str) -> RiskClass:
    """Get a tool's risk class (defaults to LOW)."""
    return _TOOL_RISK_CLASS.get(tool_name, RiskClass.LOW)


def build_impact_summary(tool_name: str, args: Dict[str, Any]) -> str:
    """Build a human-readable impact summary for the proposal."""
    summaries = {
        "start_research": lambda a: f"Start research on '{a.get('topic', 'unknown')}' for {a.get('pair', 'all pairs')}",
        "remove_strategy": lambda a: f"Remove strategy {a.get('version_id', 'unknown')}",
        "archive_strategy": lambda a: f"Archive strategy {a.get('version_id', 'unknown')}",
        "reject_strategy": lambda a: f"Reject strategy {a.get('version_id', 'unknown')}: {a.get('reason', 'no reason')}",
        "create_strategy_draft": lambda a: f"Create strategy draft '{a.get('name', 'unnamed')}'",
    }
    fn = summaries.get(tool_name)
    if fn:
        return fn(args)
    return f"Execute {tool_name} with args: {args}"