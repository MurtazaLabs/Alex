"""Stage E3: approval tokens — the human's single-use authorization.

An approval token is the ONLY bridge between a prepared intent and a live
order. The flow is:

1. ExecutionPolicy produces an OrderIntent (deterministic, validated caps)
2. Human reviews the intent (symbol, volume, SL/TP, risk, R:R)
3. Human grants a single-use ApprovalToken bound to that intent
4. Token is consumed exactly once when the order is transmitted
5. Expired or unconsumed tokens are inert

Anti-automation guarantees (enforced here, verified by source-scan tests):
- Tokens can ONLY be created by ``grant()`` — no tool, handler, or LLM path
  reaches the token store constructor without going through grant
- Every token is single-use: ``consume()`` invalidates it immediately
- Tokens expire after a fixed TTL (default 10 minutes)
- A token is cryptographically bound to (intent_key, symbol, side, volume) —
  it cannot be used for a different order even if the intent is tampered with
- ``grant()`` requires an explicit ``granted_by`` string — the human's identity

Constitution: the LLM cannot grant tokens, the agent tools cannot grant
tokens, only the human operator can. This module has no imports from the
agent package — it is a leaf dependency with no path back to the LLM.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "ApprovalTokenStore",
    "ApprovalToken",
    "TOKEN_ACTIVE",
    "TOKEN_CONSUMED",
    "TOKEN_EXPIRED",
    "TOKEN_REVOKED",
]

TOKEN_ACTIVE = "ACTIVE"
TOKEN_CONSUMED = "CONSUMED"
TOKEN_EXPIRED = "EXPIRED"
TOKEN_REVOKED = "REVOKED"

#: Token TTL in seconds (10 minutes — long enough to review, short enough to expire)
DEFAULT_TOKEN_TTL_SECONDS = 600


@dataclass
class ApprovalToken:
    """A single-use, intent-bound approval token."""

    #: Opaque token ID (random, unguessable)
    token_id: str
    #: The intent this token authorizes
    intent_key: str
    #: Bound order fields — token is invalid for any other order
    symbol: str
    side: str
    volume: float
    #: Who granted it (human identity — never a tool or LLM)
    granted_by: str
    #: When granted
    granted_at: datetime
    #: When it expires
    expires_at: datetime
    #: Current state
    status: str = TOKEN_ACTIVE
    #: When consumed (if consumed)
    consumed_at: Optional[datetime] = None
    #: The intent dict snapshot at grant time (for audit)
    intent_snapshot: Optional[Dict[str, Any]] = None

    def is_active(self, now: Optional[datetime] = None) -> bool:
        """Check if the token is still usable."""
        check = now or datetime.utcnow()
        if self.status != TOKEN_ACTIVE:
            return False
        return check < self.expires_at

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        """Check if the token has passed its expiry time."""
        check = now or datetime.utcnow()
        return check >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "intent_key": self.intent_key,
            "symbol": self.symbol,
            "side": self.side,
            "volume": self.volume,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "status": self.status,
            "consumed_at": self.consumed_at.isoformat() if self.consumed_at else None,
        }


class ApprovalTokenStore:
    """In-memory store for approval tokens. Single-use, intent-bound, expiring.

    The store is the ONLY way an intent can be authorized for live transmission.
    It enforces:

    - Single-use: consume() invalidates the token immediately
    - Intent-bound: consume() validates the token matches the intent's fields
    - Expiry: tokens past their TTL are automatically expired
    - Audit: every grant, consume, expire, and revoke is logged
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
        secret_key: Optional[str] = None,
    ) -> None:
        self._ttl = ttl_seconds
        self._secret = secret_key or secrets.token_hex(32)
        self._tokens: Dict[str, ApprovalToken] = {}
        self._audit_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Grant (human-only path)
    # ------------------------------------------------------------------

    def grant(
        self,
        *,
        intent_key: str,
        symbol: str,
        side: str,
        volume: float,
        granted_by: str,
        intent_snapshot: Optional[Dict[str, Any]] = None,
        now: Optional[datetime] = None,
    ) -> ApprovalToken:
        """HUMAN action: grant a single-use approval token for an intent.

        This is the ONLY way to create a token. No tool, handler, or LLM
        path reaches this method — the source-scan test enforces this.

        Args:
            intent_key: The OrderIntent key this token authorizes
            symbol: Currency pair (bound — token invalid for other symbols)
            side: BUY or SELL (bound)
            volume: Position size in lots (bound)
            granted_by: Human identity (username or "operator")
            intent_snapshot: Full intent dict for audit trail
            now: Override timestamp (testing only)

        Returns:
            A new active ApprovalToken
        """
        check = now or datetime.utcnow()
        token_id = self._generate_token_id(intent_key, symbol, side, volume, check)
        expires_at = check + timedelta(seconds=self._ttl)

        token = ApprovalToken(
            token_id=token_id,
            intent_key=intent_key,
            symbol=symbol,
            side=side,
            volume=volume,
            granted_by=granted_by,
            granted_at=check,
            expires_at=expires_at,
            intent_snapshot=intent_snapshot,
        )
        self._tokens[token_id] = token
        self._audit(
            "token_granted", token_id=token_id, intent_key=intent_key,
            symbol=symbol, side=side, volume=volume, granted_by=granted_by,
            expires_at=expires_at.isoformat(),
        )
        logger.info(
            "approval token granted: %s for intent %s (%s %s %.2f lots by %s)",
            token_id[:8], intent_key[:8], side, symbol, volume, granted_by,
        )
        return token

    # ------------------------------------------------------------------
    # Consume (transmission path — validates everything)
    # ------------------------------------------------------------------

    def consume(
        self,
        token_id: str,
        *,
        intent_key: str,
        symbol: str,
        side: str,
        volume: float,
        now: Optional[datetime] = None,
    ) -> bool:
        """Consume a token to authorize transmission of an order.

        Validates: token exists, is ACTIVE, not expired, and the order
        fields match the token's bound fields. On success, the token is
        immediately invalidated (single-use).

        Returns:
            True if the token was valid and consumed, False otherwise
        """
        check = now or datetime.utcnow()
        token = self._tokens.get(token_id)
        if token is None:
            self._audit("token_consumed_unknown", token_id=token_id)
            return False

        # Check expiry
        if token.status == TOKEN_ACTIVE and check >= token.expires_at:
            token.status = TOKEN_EXPIRED
            self._audit("token_expired_during_consume", token_id=token_id)
            return False

        if token.status != TOKEN_ACTIVE:
            self._audit(
                "token_consumed_inactive", token_id=token_id,
                status=token.status,
            )
            return False

        # Validate intent binding
        if token.intent_key != intent_key:
            self._audit(
                "token_intent_mismatch", token_id=token_id,
                expected=token.intent_key, got=intent_key,
            )
            return False

        # Validate order field binding
        if (token.symbol != symbol or token.side != side
                or abs(token.volume - volume) > 1e-9):
            self._audit(
                "token_order_mismatch", token_id=token_id,
                expected=f"{token.side} {token.symbol} {token.volume}",
                got=f"{side} {symbol} {volume}",
            )
            return False

        # Consume (single-use)
        token.status = TOKEN_CONSUMED
        token.consumed_at = check
        self._audit(
            "token_consumed", token_id=token_id, intent_key=intent_key,
        )
        logger.info("approval token consumed: %s for intent %s", token_id[:8], intent_key[:8])
        return True

    # ------------------------------------------------------------------
    # Revoke (human-only — cancel a granted token)
    # ------------------------------------------------------------------

    def revoke(
        self, token_id: str, *, revoked_by: str = "",
        now: Optional[datetime] = None,
    ) -> bool:
        """HUMAN action: revoke a granted token before it is consumed."""
        token = self._tokens.get(token_id)
        if token is None or token.status != TOKEN_ACTIVE:
            return False
        token.status = TOKEN_REVOKED
        self._audit(
            "token_revoked", token_id=token_id, revoked_by=revoked_by,
        )
        logger.info("approval token revoked: %s by %s", token_id[:8], revoked_by)
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get(self, token_id: str) -> Optional[ApprovalToken]:
        return self._tokens.get(token_id)

    def active_tokens(self, now: Optional[datetime] = None) -> List[ApprovalToken]:
        """All currently active (non-expired, unconsumed) tokens."""
        check = now or datetime.utcnow()
        return [
            t for t in self._tokens.values()
            if t.is_active(check)
        ]

    def tokens_for_intent(self, intent_key: str) -> List[ApprovalToken]:
        """All tokens granted for a specific intent."""
        return [t for t in self._tokens.values() if t.intent_key == intent_key]

    def has_active_token(
        self, intent_key: str, now: Optional[datetime] = None,
    ) -> bool:
        """Check if an intent has at least one active token."""
        return any(
            t.is_active(now) for t in self.tokens_for_intent(intent_key)
        )

    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def __len__(self) -> int:
        return len(self._tokens)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_token_id(
        self, intent_key: str, symbol: str, side: str,
        volume: float, now: datetime,
    ) -> str:
        """Generate an unguessable token ID (HMAC of random secret + intent)."""
        raw = f"{intent_key}:{symbol}:{side}:{volume:.2f}:{now.isoformat()}:{secrets.token_hex(16)}"
        return hmac.new(
            self._secret.encode(), raw.encode(), hashlib.sha256,
        ).hexdigest()[:32]

    def _audit(self, event: str, **fields: Any) -> None:
        self._audit_log.append({"event": event, **fields})
