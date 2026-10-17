from __future__ import annotations

import warnings
from ..config import get_settings
from ..config.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class PermissionLevel:
    """Enumeration of permission levels for the trading system.

    Ladder (Phase 2+):
    OBSERVE(0)    — read/analyze/backtest/research only
    ASSIST(1)     — create strategies & experiments
    PAPER(2)      — probation management
    EXECUTION(3)  — prepare live orders
    AUTONOMOUS(4) — execute per fixed deterministic policy
    """

    OBSERVE = 0
    ASSIST = 1
    PAPER = 2
    EXECUTION = 3
    AUTONOMOUS = 4

    # Legacy aliases for backward compatibility
    RESEARCH_ONLY = OBSERVE
    MARKET_MONITORING = ASSIST
    SIGNAL_GENERATION = PAPER
    PAPER_TRADING = EXECUTION
    LIVE_TRADING = AUTONOMOUS

    @classmethod
    def label(cls, level: int) -> str:
        """Get human-readable label for permission level."""
        labels = {
            cls.OBSERVE: "OBSERVE",
            cls.ASSIST: "ASSIST",
            cls.PAPER: "PAPER",
            cls.EXECUTION: "EXECUTION",
            cls.AUTONOMOUS: "AUTONOMOUS",
        }
        return labels.get(level, f"Level {level}")

    @classmethod
    def can_execute(cls, from_level: int, to_level: int) -> bool:
        """Check if trading can execute from one level to another."""
        return from_level >= to_level


class PermissionManager:
    """Manages permission level enforcement for trading actions."""

    def __init__(self, current_level: int = PermissionLevel.OBSERVE):
        if not (0 <= current_level <= PermissionLevel.AUTONOMOUS):
            raise ValueError(
                f"Invalid permission level: {current_level}. "
                f"Must be between 0 and {PermissionLevel.AUTONOMOUS}"
            )
        self._current_level = current_level

    @property
    def current_level(self) -> int:
        """Get the current permission level."""
        return self._current_level

    # New property names (Phase 2+)
    @property
    def is_observe_only(self) -> bool:
        """Check if current level allows only observation/research."""
        return self._current_level == PermissionLevel.OBSERVE

    @property
    def can_assist(self) -> bool:
        """Check if strategy/experiment creation is allowed."""
        return self._current_level >= PermissionLevel.ASSIST

    @property
    def can_manage_paper(self) -> bool:
        """Check if probation/paper trading management is allowed."""
        return self._current_level >= PermissionLevel.PAPER

    @property
    def can_prepare_execution(self) -> bool:
        """Check if live order preparation is allowed."""
        return self._current_level >= PermissionLevel.EXECUTION

    @property
    def can_execute_autonomous(self) -> bool:
        """Check if autonomous execution is allowed."""
        return self._current_level >= PermissionLevel.AUTONOMOUS

    @property
    def requires_explicit_approval(self) -> bool:
        """Check if current action requires explicit user approval."""
        return self._current_level >= PermissionLevel.EXECUTION

    # Backward compatibility properties (deprecated)
    @property
    def is_research_only(self) -> bool:
        warnings.warn(
            "is_research_only is deprecated, use is_observe_only",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.is_observe_only

    @property
    def can_monitor_market(self) -> bool:
        warnings.warn(
            "can_monitor_market is deprecated, use can_assist",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.can_assist

    @property
    def can_generate_signals(self) -> bool:
        warnings.warn(
            "can_generate_signals is deprecated, use can_manage_paper",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.can_manage_paper

    @property
    def can_paper_trade(self) -> bool:
        warnings.warn(
            "can_paper_trade is deprecated, use can_prepare_execution",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.can_prepare_execution

    @property
    def can_live_trade(self) -> bool:
        warnings.warn(
            "can_live_trade is deprecated, use can_execute_autonomous",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.can_execute_autonomous

    def check_permission(self, required_level: int) -> bool:
        """Check if current level meets the required permission level."""
        if required_level > self._current_level:
            logger.warning(
                "permission_denied",
                required_level=required_level,
                current_level=self._current_level,
            )
            return False
        return True

    def require(self, required_level: int) -> None:
        """Raise if current level doesn't meet required level."""
        if not self.check_permission(required_level):
            raise PermissionError(
                f"Insufficient permission: required {PermissionLevel.label(required_level)} "
                f"({required_level}), current {PermissionLevel.label(self._current_level)} "
                f"({self._current_level})"
            )

    def upgrade_level(self, new_level: int) -> None:
        """Upgrade the permission level."""
        if not (0 <= new_level <= PermissionLevel.AUTONOMOUS):
            raise ValueError(
                f"Invalid permission level: {new_level}. "
                f"Must be between 0 and {PermissionLevel.AUTONOMOUS}"
            )
        if new_level <= self._current_level:
            logger.warning(
                "permission_not_downgraded",
                current_level=self._current_level,
                requested_level=new_level,
            )
            return
        previous = PermissionLevel.label(self._current_level)
        self._current_level = new_level
        logger.info(
            "permission_upgraded",
            previous_level=previous,
            new_level=PermissionLevel.label(self._current_level),
        )

    def downgrade_level(self, new_level: int) -> None:
        """Downgrade the permission level."""
        if not (0 <= new_level <= PermissionLevel.AUTONOMOUS):
            raise ValueError(
                f"Invalid permission level: {new_level}. "
                f"Must be between 0 and {PermissionLevel.AUTONOMOUS}"
            )
        if new_level >= self._current_level:
            logger.warning(
                "permission_not_upgraded",
                current_level=self._current_level,
                requested_level=new_level,
            )
            return
        previous = PermissionLevel.label(self._current_level)
        self._current_level = new_level
        logger.info(
            "permission_downgraded",
            previous_level=previous,
            new_level=PermissionLevel.label(self._current_level),
        )

    def __str__(self) -> str:
        return f"PermissionLevel.{PermissionLevel.label(self._current_level)}"
