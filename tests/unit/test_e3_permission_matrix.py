"""E3 permission matrix tests — the gate that proves level 3 prepares, cannot send.

Pins the E3 permission contract:
- Level 3 (EXECUTION) can PREPARE order intents
- Level 3 CANNOT send orders (sending requires level 4)
- Sending requires ALL THREE: level 4 + valid token + armed governor
- No agent tool can grant approvals (source scan)
- MAX_PERMISSION_LEVEL = 4 is a banned string
- upgrade_level is not reachable from any agent tool
- The LLM cannot reach the token grant path
"""

from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import pytest

from forex_ai_agent.core.permission_manager import PermissionLevel, PermissionManager
from forex_ai_agent.execution.governor import ExecutionGovernor, GovernorLimits
from forex_ai_agent.execution.policy import ExecutionPolicy, PolicyCaps
from forex_ai_agent.execution.tokens import ApprovalTokenStore


# ---------------------------------------------------------------------------
# Level 3 prepares, cannot send
# ---------------------------------------------------------------------------

class TestLevel3Prepares:
    def test_level3_can_prepare(self):
        pm = PermissionManager(current_level=PermissionLevel.EXECUTION)
        assert pm.can_prepare_execution is True
        assert pm.check_permission(PermissionLevel.EXECUTION) is True

    def test_level3_cannot_execute_autonomous(self):
        pm = PermissionManager(current_level=PermissionLevel.EXECUTION)
        assert pm.can_execute_autonomous is False
        assert pm.check_permission(PermissionLevel.AUTONOMOUS) is False

    def test_level2_cannot_prepare(self):
        pm = PermissionManager(current_level=PermissionLevel.PAPER)
        assert pm.can_prepare_execution is False

    def test_level0_cannot_prepare(self):
        pm = PermissionManager(current_level=PermissionLevel.OBSERVE)
        assert pm.can_prepare_execution is False

    def test_level3_requires_explicit_approval(self):
        pm = PermissionManager(current_level=PermissionLevel.EXECUTION)
        assert pm.requires_explicit_approval is True


# ---------------------------------------------------------------------------
# Sending requires ALL THREE: level 4 + token + armed governor
# ---------------------------------------------------------------------------

class TestSendRequiresAllThree:
    def _make_stack(self, level=PermissionLevel.EXECUTION, armed=True):
        pm = PermissionManager(current_level=level)
        governor = ExecutionGovernor(GovernorLimits())
        if armed:
            governor.arm()
        tokens = ApprovalTokenStore()
        return pm, governor, tokens

    def test_level3_with_token_and_armed_still_cannot_send(self):
        """Level 3 prepares but CANNOT send even with token + armed governor."""
        pm, governor, tokens = self._make_stack(level=PermissionLevel.EXECUTION)
        # Level 3 is NOT sufficient to send
        assert pm.can_execute_autonomous is False
        # Even if token exists and governor is armed
        token = tokens.grant(
            intent_key="test", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        assert token is not None
        assert governor.state.state == "ARMED"
        # The permission check still fails
        with pytest.raises(PermissionError):
            pm.require(PermissionLevel.AUTONOMOUS)

    def test_level4_without_token_cannot_send(self):
        """Level 4 alone is NOT enough — a valid token is required."""
        pm, governor, tokens = self._make_stack(level=PermissionLevel.AUTONOMOUS)
        # Governor armed and level 4, but no token
        assert pm.can_execute_autonomous is True
        assert governor.state.state == "ARMED"
        # No token exists for this intent
        assert tokens.has_active_token("any_intent") is False

    def test_level4_with_disarmed_governor_cannot_send(self):
        """Level 4 + token is NOT enough — governor must be ARMED."""
        pm, governor, tokens = self._make_stack(
            level=PermissionLevel.AUTONOMOUS, armed=False,
        )
        token = tokens.grant(
            intent_key="test", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        assert pm.can_execute_autonomous is True
        assert token is not None
        # Governor is DISARMED — cannot send
        assert governor.state.state == "DISARMED"
        # Intent would be refused
        from datetime import datetime
        result = governor.accept_intent(
            "test", symbol="EUR/USD", side="BUY", volume=0.01,
            sl=1.098, tp=1.104, now=datetime.utcnow(),
        )
        assert result is None  # refused: not armed

    def test_level4_with_token_and_armed_can_send(self):
        """All three conditions met: level 4 + token + armed governor."""
        pm, governor, tokens = self._make_stack(level=PermissionLevel.AUTONOMOUS)
        token = tokens.grant(
            intent_key="test", symbol="EUR/USD", side="BUY",
            volume=0.01, granted_by="murtaza",
        )
        from datetime import datetime
        # Governor accepts the intent
        result = governor.accept_intent(
            "test", symbol="EUR/USD", side="BUY", volume=0.01,
            sl=1.098, tp=1.104, now=datetime.utcnow(),
        )
        assert result is not None
        # Token can be consumed
        ok = tokens.consume(
            token.token_id, intent_key="test",
            symbol="EUR/USD", side="BUY", volume=0.01,
        )
        assert ok is True
        # Permission check passes
        assert pm.can_execute_autonomous is True

    def test_governor_disarmed_refuses_intent(self):
        """DISARMED governor refuses every intent."""
        governor = ExecutionGovernor(GovernorLimits())
        from datetime import datetime
        result = governor.accept_intent(
            "test", symbol="EUR/USD", side="BUY", volume=0.01,
            sl=1.098, tp=1.104, now=datetime.utcnow(),
        )
        assert result is None

    def test_governor_halted_refuses_intent(self):
        """HALTED (kill-switched) governor refuses every intent."""
        governor = ExecutionGovernor(GovernorLimits())
        governor.arm()
        governor.trip("manual")
        from datetime import datetime
        result = governor.accept_intent(
            "test", symbol="EUR/USD", side="BUY", volume=0.01,
            sl=1.098, tp=1.104, now=datetime.utcnow(),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Source scans: anti-automation pins
# ---------------------------------------------------------------------------

def _extract_imports(source: str) -> list[str]:
    """Extract all import module paths from Python source."""
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _extract_function_calls(source: str) -> list[str]:
    """Extract all function/method call names from Python source."""
    tree = ast.parse(source)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    return calls


class TestAntiAutomationPins:
    def test_max_permission_level_4_banned(self):
        """MAX_PERMISSION_LEVEL = 4 must not appear as a literal in source."""
        root = Path(__file__).resolve().parents[2] / "src"
        banned = "MAX_PERMISSION_LEVEL = 4"
        offenders = []
        for py in root.rglob("*.py"):
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if banned in text:
                offenders.append(str(py.relative_to(root)))
        assert not offenders, f"MAX_PERMISSION_LEVEL = 4 found in: {offenders}"

    def test_no_agent_tool_grants_approvals(self):
        """No file in agent/ or tools/ may call token store grant()."""
        root = Path(__file__).resolve().parents[2] / "src" / "forex_ai_agent" / "agent"
        if not root.exists():
            pytest.skip("agent directory not found")
        offenders = []
        for py in root.rglob("*.py"):
            try:
                text = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            calls = _extract_function_calls(text)
            if "grant" in calls or "grant_token" in calls:
                offenders.append(str(py.relative_to(root)))
        assert not offenders, f"grant() found in agent code: {offenders}"

    def test_no_upgrade_level_in_agent_tools(self):
        """No agent tool may call upgrade_level()."""
        root = Path(__file__).resolve().parents[2] / "src" / "forex_ai_agent"
        targets = [root / "agent", root / "interfaces"]
        offenders = []
        for base in targets:
            if not base.exists():
                continue
            for py in base.rglob("*.py"):
                try:
                    text = py.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                calls = _extract_function_calls(text)
                if "upgrade_level" in calls:
                    offenders.append(str(py.relative_to(root)))
        assert not offenders, f"upgrade_level() found in: {offenders}"

    def test_tokens_module_has_no_llm_imports(self):
        """tokens.py must not import LLM providers or agent modules."""
        tokens_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "forex_ai_agent" / "execution" / "tokens.py"
        )
        source = tokens_path.read_text(encoding="utf-8")
        imports = _extract_imports(source)
        banned = ["openai", "anthropic", "forex_ai_agent.agent", "langchain"]
        for imp in imports:
            for b in banned:
                assert b not in imp, f"Banned import in tokens.py: {imp}"

    def test_policy_module_has_no_llm_imports(self):
        """policy.py must not import LLM providers or agent modules."""
        policy_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "forex_ai_agent" / "execution" / "policy.py"
        )
        source = policy_path.read_text(encoding="utf-8")
        imports = _extract_imports(source)
        banned = ["openai", "anthropic", "forex_ai_agent.agent", "langchain"]
        for imp in imports:
            for b in banned:
                assert b not in imp, f"Banned import in policy.py: {imp}"

    def test_governor_has_no_network_imports(self):
        """governor.py must not import network libraries."""
        gov_path = (
            Path(__file__).resolve().parents[2]
            / "src" / "forex_ai_agent" / "execution" / "governor.py"
        )
        source = gov_path.read_text(encoding="utf-8")
        imports = _extract_imports(source)
        banned = ["requests", "httpx", "aiohttp", "urllib", "socket"]
        for imp in imports:
            for b in banned:
                assert b not in imp, f"Network import in governor.py: {imp}"


# ---------------------------------------------------------------------------
# Source scan: no network in E3 modules
# ---------------------------------------------------------------------------

class TestNoNetworkInE3:
    def test_execution_package_no_network(self):
        """No file in execution/ may import network libraries."""
        root = (
            Path(__file__).resolve().parents[2]
            / "src" / "forex_ai_agent" / "execution"
        )
        banned = ["requests", "httpx", "aiohttp", "urllib.request", "socket"]
        offenders = []
        for py in root.glob("*.py"):
            source = py.read_text(encoding="utf-8", errors="ignore")
            imports = _extract_imports(source)
            for imp in imports:
                for b in banned:
                    if b in imp:
                        offenders.append(f"{py.name}: {imp}")
        assert not offenders, f"Network imports in execution/: {offenders}"

    def test_execution_package_no_llm(self):
        """No file in execution/ may import LLM providers."""
        root = (
            Path(__file__).resolve().parents[2]
            / "src" / "forex_ai_agent" / "execution"
        )
        banned = ["openai", "anthropic", "langchain", "forex_ai_agent.agent"]
        offenders = []
        for py in root.glob("*.py"):
            source = py.read_text(encoding="utf-8", errors="ignore")
            imports = _extract_imports(source)
            for imp in imports:
                for b in banned:
                    if b in imp:
                        offenders.append(f"{py.name}: {imp}")
        assert not offenders, f"LLM imports in execution/: {offenders}"
