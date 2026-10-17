"""Unit tests for the permission manager."""

from __future__ import annotations

import pytest

from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager


def test_default_level_is_research_only():
    pm = PermissionManager()
    assert pm.current_level == PermissionLevel.RESEARCH_ONLY
    assert pm.is_research_only


def test_level_labels():
    assert PermissionLevel.label(0) == "OBSERVE"
    assert PermissionLevel.label(1) == "ASSIST"
    assert PermissionLevel.label(2) == "PAPER"
    assert PermissionLevel.label(3) == "EXECUTION"
    assert PermissionLevel.label(4) == "AUTONOMOUS"
    assert PermissionLevel.label(77) == "Level 77"


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        PermissionManager(current_level=5)
    with pytest.raises(ValueError):
        PermissionManager(current_level=-1)


def test_check_permission_gates():
    pm = PermissionManager(current_level=PermissionLevel.SIGNAL_GENERATION)
    assert pm.check_permission(0)
    assert pm.check_permission(2)
    assert not pm.check_permission(3)
    assert not pm.check_permission(4)


def test_properties():
    pm = PermissionManager(current_level=PermissionLevel.LIVE_TRADING)
    assert pm.can_generate_signals
    assert pm.can_paper_trade
    assert pm.can_live_trade
    assert pm.requires_explicit_approval


def test_upgrade_level():
    pm = PermissionManager(current_level=PermissionLevel.RESEARCH_ONLY)
    pm.upgrade_level(PermissionLevel.PAPER_TRADING)
    assert pm.current_level == PermissionLevel.PAPER_TRADING


def test_cannot_downgrade_via_upgrade():
    pm = PermissionManager(current_level=PermissionLevel.LIVE_TRADING)
    pm.upgrade_level(PermissionLevel.SIGNAL_GENERATION)
    assert pm.current_level == PermissionLevel.LIVE_TRADING


def test_downgrade_level():
    pm = PermissionManager(current_level=PermissionLevel.LIVE_TRADING)
    pm.downgrade_level(PermissionLevel.RESEARCH_ONLY)
    assert pm.current_level == PermissionLevel.RESEARCH_ONLY


def test_cannot_upgrade_via_downgrade():
    pm = PermissionManager(current_level=PermissionLevel.RESEARCH_ONLY)
    pm.downgrade_level(PermissionLevel.PAPER_TRADING)
    assert pm.current_level == PermissionLevel.RESEARCH_ONLY


def test_can_execute_classmethod():
    assert PermissionLevel.can_execute(3, 3)
    assert PermissionLevel.can_execute(4, 2)
    assert not PermissionLevel.can_execute(1, 2)


def test_string_representation():
    pm = PermissionManager(current_level=PermissionLevel.ASSIST)
    assert "ASSIST" in str(pm)
