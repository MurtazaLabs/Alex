"""Unit tests for the state machine."""

from __future__ import annotations

import pytest

from forex_ai_agent.core.state_manager import StateError, StateMachine


def make_machine() -> StateMachine:
    return StateMachine(
        name="signal",
        states=[
            "NO_TRADE",
            "WATCH",
            "SETUP_FORMING",
            "TRIGGERED",
            "FILLED",
            "INVALIDATED",
        ],
        initial="NO_TRADE",
        transitions={
            "NO_TRADE": ["WATCH"],
            "WATCH": ["SETUP_FORMING", "INVALIDATED"],
            "SETUP_FORMING": ["TRIGGERED", "INVALIDATED", "WATCH"],
            "TRIGGERED": ["FILLED", "INVALIDATED"],
            "INVALIDATED": ["NO_TRADE"],
        },
    )


def test_initial_state():
    sm = make_machine()
    assert sm.current == "NO_TRADE"
    assert sm.is_in("NO_TRADE")


def test_transition_and_properties():
    sm = make_machine()
    sm.transition("WATCH")
    assert sm.current == "WATCH"
    assert "SETUP_FORMING" in sm.allowed_targets()
    assert sm.can_transition("INVALIDATED")
    assert not sm.can_transition("TRIGGERED")


def test_invalid_transition_raises():
    sm = make_machine()
    with pytest.raises(StateError):
        sm.transition("TRIGGERED")


def test_unknown_state_raises():
    sm = make_machine()
    with pytest.raises(StateError):
        sm.transition("NOT_A_STATE")


def test_bad_machine_definition():
    with pytest.raises(StateError):
        StateMachine("x", ["A", "B"], initial="A", transitions={"A": ["C"]})


def test_reset():
    sm = make_machine()
    sm.transition("WATCH")
    sm.reset()
    assert sm.current == "NO_TRADE"
    sm.transition("WATCH")
    sm.reset(to="TRIGGERED")  # reset bypasses transition rules
    with pytest.raises(StateError):
        sm.reset(to="GHOST")
