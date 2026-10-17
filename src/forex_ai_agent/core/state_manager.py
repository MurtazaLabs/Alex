"""Tiny state machine used by agents and workflows."""

from __future__ import annotations

from typing import Dict, Iterable, Optional


class StateError(RuntimeError):
    """Raised on invalid state transitions."""


class StateMachine:
    """
    Validate and track state transitions for a named workflow.

    Example::
        sm = StateMachine("signal", ["WATCH", "TRIGGERED", "FILLED"],
                          initial="WATCH",
                          transitions={"WATCH": ["TRIGGERED", "INVALIDATED"]})
    """

    def __init__(
        self,
        name: str,
        states: Iterable[str],
        initial: str,
        transitions: Optional[Dict[str, Iterable[str]]] = None,
    ):
        self.name = name
        self._states = set(states)
        self._transitions: Dict[str, set] = {s: set() for s in self._states}
        if transitions:
            for src, targets in transitions.items():
                if src not in self._states:
                    raise StateError(f"Unknown state {src!r} in {name}")
                self._transitions[src] = set(targets)
                unknown = self._transitions[src] - self._states
                if unknown:
                    raise StateError(f"Unknown targets {sorted(unknown)} in {name}")
        if initial not in self._states:
            raise StateError(f"Initial state {initial!r} not in {sorted(self._states)}")
        self._initial = initial
        self._current = initial

    @property
    def current(self) -> str:
        """Current state name."""
        return self._current

    def is_in(self, *states: str) -> bool:
        """True when the machine is in any of the given states."""
        return self._current in states

    def allowed_targets(self) -> set:
        """States reachable from the current state."""
        return self._transitions[self._current]

    def can_transition(self, target: str) -> bool:
        """Whether the transition to ``target`` is allowed right now."""
        if target not in self._states:
            return False
        return target in self._transitions[self._current]

    def transition(self, target: str) -> str:
        """Move to ``target``, raising ``StateError`` when not allowed."""
        if target not in self._states:
            raise StateError(f"Cannot transition to unknown state {target!r}")
        if target not in self._transitions[self._current]:
            raise StateError(
                f"{self.name}: {self._current!r} -> {target!r} not allowed "
                f"(allowed: {sorted(self._transitions[self._current])})"
            )
        self._current = target
        return self._current

    def reset(self, to: Optional[str] = None) -> None:
        """Reset the machine to a state (defaults to the initial state)."""
        if to is not None and to not in self._states:
            raise StateError(f"Cannot reset to unknown state {to!r}")
        self._current = to if to is not None else self._initial
