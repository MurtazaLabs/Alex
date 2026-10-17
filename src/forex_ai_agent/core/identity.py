"""Identity & Constitution Loader for Alex.

Loads soul.md, identity.md, user.md on startup and makes them available
to all agents for behavior alignment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from forex_ai_agent.config.settings import get_settings


@dataclass
class SoulConstitution:
    """Parsed constitution from soul.md"""
    core_beliefs: Dict[str, List[str]] = field(default_factory=dict)
    values: Dict[str, str] = field(default_factory=dict)
    boundaries: Dict[str, List[str]] = field(default_factory=dict)
    ethical_guardrails: List[Dict[str, str]] = field(default_factory=list)
    decision_framework: List[str] = field(default_factory=list)
    raw_content: str = ""


@dataclass
class AgentIdentity:
    """Parsed identity from identity.md"""
    name: str = "Alex"
    codename: str = "Atlas"
    role: str = "Institutional-grade Research & Execution Assistant"
    mandate: str = ""
    permission_levels: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    interfaces: List[Dict[str, str]] = field(default_factory=list)
    architecture_principle: str = ""
    personality: Dict[str, str] = field(default_factory=dict)
    raw_content: str = ""


@dataclass
class UserProfile:
    """Parsed user profile from user.md"""
    experience: str = ""
    profession: str = ""
    timezone: str = ""
    trading_hours: str = ""
    motivations: Dict[str, int] = field(default_factory=dict)
    core_driver: str = ""
    style: str = ""
    timeframes: str = ""
    sessions: str = ""
    no_trade_sessions: str = ""
    pairs: List[str] = field(default_factory=list)
    strategy_lean: str = ""
    risk_params: Dict[str, Any] = field(default_factory=dict)
    psychological_triggers: List[Dict[str, str]] = field(default_factory=list)
    success_metrics: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, str] = field(default_factory=dict)
    alex_directives: List[str] = field(default_factory=list)
    emergency_protocol: Dict[str, str] = field(default_factory=dict)
    raw_content: str = ""


class IdentityLoader:
    """Loads and parses the three identity documents."""

    def __init__(self, base_path: Optional[Path] = None):
        if base_path:
            self.base_path = base_path
        else:
            # Find project root by looking for soul.md
            current = Path(__file__).parent.parent.parent
            while current != current.parent:
                if (current / "soul.md").exists():
                    self.base_path = current
                    break
                current = current.parent
            else:
                self.base_path = Path.cwd()
        self._soul: Optional[SoulConstitution] = None
        self._identity: Optional[AgentIdentity] = None
        self._user: Optional[UserProfile] = None

    def load_all(self) -> tuple[SoulConstitution, AgentIdentity, UserProfile]:
        """Load all three documents."""
        self._soul = self._load_soul()
        self._identity = self._load_identity()
        self._user = self._load_user()
        return self._soul, self._identity, self._user

    def _load_soul(self) -> SoulConstitution:
        path = self.base_path / "soul.md"
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        soul = SoulConstitution(raw_content=content)

        # Parse core beliefs
        soul.core_beliefs = self._parse_markdown_section(content, "CORE BELIEFS")
        # Parse values
        soul.values = self._parse_key_value_section(content, "VALUES")
        # Parse boundaries
        soul.boundaries = self._parse_markdown_section(content, "BOUNDARIES")
        # Parse ethical guardrails
        soul.ethical_guardrails = self._parse_guardrails(content)
        # Parse decision framework
        soul.decision_framework = self._parse_numbered_list(content, "DECISION FRAMEWORK")

        return soul

    def _load_identity(self) -> AgentIdentity:
        path = self.base_path / "identity.md"
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        identity = AgentIdentity(raw_content=content)

        # Extract name
        m = re.search(r'\|\s*\*\*Name\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            identity.name = m.group(1).strip()

        # Extract codename
        m = re.search(r'\|\s*\*\*Codename\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            identity.codename = m.group(1).strip()

        # Extract role
        m = re.search(r'\|\s*\*\*Role\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            identity.role = m.group(1).strip()

        # Extract mandate
        m = re.search(r'### Mandate Scope(.*?)(?:\n##|$)', content, re.DOTALL)
        if m:
            identity.mandate = m.group(1).strip()

        # Extract architecture principle
        m = re.search(r'\|\s*\*\*Architecture Principle\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            identity.architecture_principle = m.group(1).strip()

        return identity

    def _load_user(self) -> UserProfile:
        path = self.base_path / "user.md"
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        user = UserProfile(raw_content=content)

        # Parse experience
        m = re.search(r'\|\s*\*\*Experience\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.experience = m.group(1).strip()

        # Parse profession
        m = re.search(r'\|\s*\*\*Profession\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.profession = m.group(1).strip()

        # Parse timezone
        m = re.search(r'\|\s*\*\*Timezone\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.timezone = m.group(1).strip()

        # Parse trading hours
        m = re.search(r'\|\s*\*\*Typical Trading Hours\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.trading_hours = m.group(1).strip()

        # Parse core driver
        m = re.search(r'\*\*Core Driver\*\*:\s*"([^"]+)"', content)
        if m:
            user.core_driver = m.group(1).strip()

        # Parse style
        m = re.search(r'\|\s*\*\*Style\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.style = self._clean_markdown(m.group(1).strip())

        # Parse timeframes
        m = re.search(r'\|\s*\*\*Timeframes\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.timeframes = self._clean_markdown(m.group(1).strip())

        # Parse sessions
        m = re.search(r'\|\s*\*\*Sessions\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.sessions = self._clean_markdown(m.group(1).strip())

        # Parse no-trade sessions - look for Asian range block
        if "asian" in content.lower() and "block" in content.lower():
            m = re.search(r'(?:NO|NEVER)\s+(?:trade\s+)?(?:Asian|asian)\s+[^|]*', content, re.IGNORECASE)
            if m:
                user.no_trade_sessions = m.group(0).strip()
            else:
                user.no_trade_sessions = "Asian range (00:00-08:00 UTC)"

        # Parse pairs
        m = re.search(r'\|\s*\*\*Pairs\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            pairs_text = self._clean_markdown(m.group(1).strip())
            # Split by comma and clean
            user.pairs = [p.strip() for p in pairs_text.split(",") if p.strip()]

        # Parse strategy lean
        m = re.search(r'\|\s*\*\*Strategy Lean\*\*\s*\|\s*([^|]+)\s*\|', content)
        if m:
            user.strategy_lean = self._clean_markdown(m.group(1).strip())

        # Parse risk parameters
        user.risk_params = self._parse_risk_params(content)

        # Parse Alex directives
        user.alex_directives = self._parse_alex_directives(content)

        # Parse emergency protocol
        user.emergency_protocol = self._parse_emergency_protocol(content)

        return user

    def _clean_markdown(self, text: str) -> str:
        """Clean markdown formatting from text."""
        # Remove bold/italic
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _parse_markdown_section(self, content: str, section_name: str) -> Dict[str, List[str]]:
        """Parse a markdown section with sub-sections."""
        pattern = rf'##\s+{section_name}(.*?)(?:\n##|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return {}

        section_content = match.group(1)
        result = {}
        current_key = None
        for line in section_content.split('\n'):
            line = line.strip()
            if line.startswith('### '):
                current_key = line[4:].strip()
                result[current_key] = []
            elif line.startswith('- ') and current_key:
                result[current_key].append(self._clean_markdown(line[2:].strip()))
        return result

    def _parse_key_value_section(self, content: str, section_name: str) -> Dict[str, str]:
        """Parse a section with key-value pairs in table format."""
        pattern = rf'##\s+{section_name}(.*?)(?:\n##|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return {}

        section_content = match.group(1)
        result = {}
        for line in section_content.split('\n'):
            line = line.strip()
            if '|' in line and not line.startswith('|---') and not line.startswith('| Value'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    key = self._clean_markdown(parts[1].replace('**', ''))
                    value = self._clean_markdown(parts[2])
                    if key and value:
                        result[key] = value
        return result

    def _parse_guardrails(self, content: str) -> List[Dict[str, str]]:
        """Parse ethical guardrails table."""
        pattern = r'## ETHICAL GUARDRAILS.*?(?:\n##|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        section_content = match.group(1)
        guardrails = []
        for line in section_content.split('\n'):
            line = line.strip()
            if '|' in line and not line.startswith('|---') and not line.startswith('| Guardrail'):
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    guardrails.append({
                        "guardrail": self._clean_markdown(parts[1].replace('**', '')),
                        "trigger": self._clean_markdown(parts[2]),
                        "action": self._clean_markdown(parts[3])
                    })
        return guardrails

    def _parse_numbered_list(self, content: str, section_name: str) -> List[str]:
        """Parse numbered list in a section."""
        pattern = rf'##\s+{section_name}(.*?)(?:\n##|$)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if not match:
            return []

        section_content = match.group(1)
        items = []
        for line in section_content.split('\n'):
            line = line.strip()
            m = re.match(r'^\d+\.\s+(.+)', line)
            if m:
                items.append(self._clean_markdown(m.group(1).strip()))
        return items

    def _parse_risk_params(self, content: str) -> Dict[str, Any]:
        """Parse risk parameters from user.md."""
        params = {}
        patterns = {
            "max_risk_per_trade": r'\|\s*\*\*Max Risk / Trade\*\*\s*\|\s*([^|]+)\s*\|',
            "max_daily_loss": r'\|\s*\*\*Max Daily Loss\*\*\s*\|\s*([^|]+)\s*\|',
            "max_drawdown": r'\|\s*\*\*Max Drawdown\*\*\s*\|\s*([^|]+)\s*\|',
            "min_rr": r'\|\s*\*\*Risk/Reward Minimum\*\*\s*\|\s*([^|]+)\s*\|',
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, content)
            if m:
                params[key] = self._clean_markdown(m.group(1).strip())
        return params

    def _parse_alex_directives(self, content: str) -> List[str]:
        """Parse Alex-specific directives."""
        # Try multiple header variations
        for pattern in [
            r'## (?:VII\. )?ALEX-SPECIFIC DIRECTIVES',
            r'## VII\. ALEX-SPECIFIC DIRECTIVES',
            r'ALEX-SPECIFIC DIRECTIVES',
        ]:
            match = re.search(rf'{pattern}.*?(?:\n##|$)', content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1) if match.lastindex else match.group(0)
                directives = []
                for line in section_content.split('\n'):
                    line = line.strip()
                    if line.startswith('- ') or re.match(r'^\d+\.', line):
                        clean = self._clean_markdown(line.lstrip('- 1234567890. '))
                        if clean:
                            directives.append(clean)
                return directives
        return []

    def _parse_emergency_protocol(self, content: str) -> Dict[str, str]:
        """Parse emergency protocol table."""
        # Try multiple header variations
        for pattern in [
            r'## (?:X\. )?EMERGENCY PROTOCOL',
            r'## X\. EMERGENCY PROTOCOL',
            r'EMERGENCY PROTOCOL',
        ]:
            match = re.search(rf'{pattern}.*?(?:\n##|$)', content, re.DOTALL | re.IGNORECASE)
            if match:
                section_content = match.group(1) if match.lastindex else match.group(0)
                protocol = {}
                for line in section_content.split('\n'):
                    line = line.strip()
                    if '|' in line and not line.startswith('|---') and not line.startswith('| Scenario'):
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) >= 3:
                            scenario = self._clean_markdown(parts[1])
                            action = self._clean_markdown(parts[2])
                            if scenario and action:
                                protocol[scenario] = action
                return protocol
        return {}

    @property
    def soul(self) -> SoulConstitution:
        if self._soul is None:
            self.load_all()
        return self._soul

    @property
    def identity(self) -> AgentIdentity:
        if self._identity is None:
            self.load_all()
        return self._identity

    @property
    def user(self) -> UserProfile:
        if self._user is None:
            self.load_all()
        return self._user


# Global instance
_identity_loader: Optional[IdentityLoader] = None


def get_identity_loader() -> IdentityLoader:
    """Get global identity loader instance."""
    global _identity_loader
    if _identity_loader is None:
        _identity_loader = IdentityLoader()
        _identity_loader.load_all()
    return _identity_loader


def get_soul() -> SoulConstitution:
    return get_identity_loader().soul


def get_identity() -> AgentIdentity:
    return get_identity_loader().identity


def get_user_profile() -> UserProfile:
    return get_identity_loader().user


# Convenience functions for common checks
def is_asian_session_blocked() -> bool:
    """Check if Asian session trading is blocked (from soul.md)."""
    soul = get_soul()
    boundaries = soul.boundaries.get("Sessions", [])
    return any("asian" in b.lower() and "block" in b.lower() for b in boundaries)


def get_no_trade_sessions() -> List[str]:
    """Get list of blocked trading sessions."""
    user = get_user_profile()
    sessions = []
    if user.no_trade_sessions:
        sessions.append(user.no_trade_sessions)
    if is_asian_session_blocked():
        sessions.append("Asian (00:00-08:00 UTC)")
    return sessions


def get_risk_limits() -> Dict[str, Any]:
    """Get risk limits from user profile."""
    user = get_user_profile()
    return user.risk_params


def get_alex_directives() -> List[str]:
    """Get user's specific directives for Alex."""
    user = get_user_profile()
    return user.alex_directives