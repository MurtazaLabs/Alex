"""Strategy Version Management - Semantic versioning, lineage, diff, rollback."""

from __future__ import annotations

import asyncio
import json
import hashlib
import difflib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import uuid4
from pathlib import Path

from forex_ai_agent.core.event_bus import EventBus, InMemoryEventBus, Event
from forex_ai_agent.research.models import Hypothesis


class VersionType(Enum):
    """Type of version change."""
    MAJOR = "major"      # Breaking changes
    MINOR = "minor"      # New features, backward compatible
    PATCH = "patch"      # Bug fixes, backward compatible
    EXPERIMENTAL = "experimental"  # Experimental changes


@dataclass
class StrategyVersion:
    """A versioned strategy with full metadata."""
    id: str = field(default_factory=lambda: str(uuid4())[:8])
    version: str = "0.1.0"  # Semantic version
    version_type: VersionType = VersionType.PATCH
    parent_id: Optional[str] = None
    
    # Strategy definition
    name: str = ""
    description: str = ""
    rules: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Lineage
    children_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    status: str = "draft"  # draft, active, archived, deprecated
    
    # Performance tracking
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    live_results: Dict[str, Any] = field(default_factory=dict)
    
    def bump_version(self, change_type: VersionType) -> str:
        """Bump version according to semantic versioning."""
        major, minor, patch = map(int, self.version.split("."))
        
        if change_type == VersionType.MAJOR:
            major += 1
            minor = 0
            patch = 0
        elif change_type == VersionType.MINOR:
            minor += 1
            patch = 0
        elif change_type == VersionType.PATCH:
            patch += 1
        else:  # EXPERIMENTAL
            patch += 1
            self.version_type = VersionType.EXPERIMENTAL
            
        self.version = f"{major}.{minor}.{patch}"
        self.updated_at = datetime.utcnow()
        return self.version
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['version_type'] = self.version_type.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyVersion':
        data = data.copy()
        data['version_type'] = VersionType(data.get('version_type', 'patch'))
        data['created_at'] = datetime.fromisoformat(data['created_at']) if isinstance(data.get('created_at'), str) else data.get('created_at')
        data['updated_at'] = datetime.fromisoformat(data['updated_at']) if isinstance(data.get('updated_at'), str) else data.get('updated_at')
        return cls(**data)


@dataclass
class StrategyDiff:
    """Difference between two strategy versions."""
    from_version: str
    to_version: str
    added_rules: List[str] = field(default_factory=list)
    removed_rules: List[str] = field(default_factory=list)
    modified_rules: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)  # rule -> (old, new)
    added_params: List[str] = field(default_factory=list)
    removed_params: List[str] = field(default_factory=list)
    modified_params: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)
    summary: str = ""
    
    def has_changes(self) -> bool:
        return any([
            self.added_rules, self.removed_rules, self.modified_rules,
            self.added_params, self.removed_params, self.modified_params
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['modified_rules'] = {k: [v[0], v[1]] for k, v in self.modified_rules.items()}
        data['modified_params'] = {k: [v[0], v[1]] for k, v in self.modified_params.items()}
        return data


class StrategyVersionManager:
    """
    Manages strategy versions with semantic versioning, lineage tracking,
    diff capabilities, and rollback capability.
    """

    def __init__(
        self,
        storage_path: str = "strategies/versions",
        event_bus: Optional[EventBus] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.event_bus = event_bus or InMemoryEventBus()
        
        self.versions: Dict[str, StrategyVersion] = {}
        self.active_version_id: Optional[str] = None
        self._load_versions()
    
    def _load_versions(self):
        """Load all versions from storage."""
        for file in self.storage_path.glob("*.json"):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                version = StrategyVersion.from_dict(data)
                self.versions[version.id] = version
                if version.status == "active":
                    self.active_version_id = version.id
            except Exception as e:
                print(f"Failed to load version {file}: {e}")
    
    def _save_version(self, version: StrategyVersion):
        """Save a version to storage."""
        file_path = self.storage_path / f"{version.id}.json"
        with open(file_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def create_version(
        self,
        name: str,
        description: str = "",
        rules: Dict[str, Any] = None,
        parameters: Dict[str, Any] = None,
        parent_id: Optional[str] = None,
        change_type: VersionType = VersionType.PATCH,
    ) -> StrategyVersion:
        """Create a new strategy version."""
        parent = self.versions.get(parent_id) if parent_id else None
        
        if parent:
            # Create as child of parent
            version = StrategyVersion(
                name=name or f"{parent.name} v{parent.bump_version(change_type)}",
                description=description or parent.description,
                rules=rules or parent.rules.copy(),
                parameters=parameters or parent.parameters.copy(),
                parent_id=parent.id,
                version_type=change_type,
            )
            parent.children_ids.append(version.id)
            self._save_version(parent)
        else:
            # Create root version
            version = StrategyVersion(
                name=name,
                description=description,
                rules=rules or {},
                parameters=parameters or {},
                version_type=change_type,
            )
        
        self.versions[version.id] = version
        self._save_version(version)
        
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish(Event(
                event_type="strategy.version.created",
                aggregate_id=version.id,
                aggregate_type="strategy_version",
                payload={"version_id": version.id, "version": version.version},
                source_agent="strategy_version_manager",
            )))
        
        return version
    
    def update_version(self, version: StrategyVersion) -> None:
        """Persist an already-loaded version after in-place mutation
        (e.g. appending robustness/backtest evidence)."""
        self.versions[version.id] = version
        self._save_version(version)

    def get_version(self, version_id: str) -> Optional[StrategyVersion]:
        """Get a version by ID."""
        return self.versions.get(version_id)
    
    def get_version_by_string(self, version_string: str) -> Optional[StrategyVersion]:
        """Get version by version string (e.g., '1.2.0')."""
        for v in self.versions.values():
            if v.version == version_string:
                return v
        return None
    
    def set_active(self, version_id: str) -> bool:
        """Set a version as active."""
        version = self.versions.get(version_id)
        if not version:
            return False
        
        # Deactivate current active
        if self.active_version_id and self.active_version_id != version_id:
            old = self.versions.get(self.active_version_id)
            if old:
                old.status = "archived"
                self._save_version(old)
        
        version.status = "active"
        self.active_version_id = version.id
        self._save_version(version)
        
        if self.event_bus:
            asyncio.create_task(self.event_bus.publish(Event(
                event_type="strategy.version.activated",
                aggregate_id=version.id,
                aggregate_type="strategy_version",
                payload={"version_id": version.id, "version": version.version},
                source_agent="strategy_version_manager",
            )))
        return True
    
    def get_active(self) -> Optional[StrategyVersion]:
        """Get the currently active version."""
        if self.active_version_id:
            return self.versions.get(self.active_version_id)
        return None
    
    def diff(self, from_id: str, to_id: str) -> StrategyDiff:
        """Compute diff between two versions."""
        from_ver = self.versions.get(from_id)
        to_ver = self.versions.get(to_id)
        
        if not from_ver or not to_ver:
            raise ValueError("Version not found")
        
        diff = StrategyDiff(from_version=from_ver.version, to_version=to_ver.version)
        
        # Compare rules
        from_rules = set(from_ver.rules.keys())
        to_rules = set(to_ver.rules.keys())
        
        diff.added_rules = list(to_rules - from_rules)
        diff.removed_rules = list(from_rules - to_rules)
        
        common_rules = from_rules & to_rules
        for rule in common_rules:
            if from_ver.rules[rule] != to_ver.rules[rule]:
                diff.modified_rules[rule] = (from_ver.rules[rule], to_ver.rules[rule])
        
        # Compare parameters
        from_params = set(from_ver.parameters.keys())
        to_params = set(to_ver.parameters.keys())
        
        diff.added_params = list(to_params - from_params)
        diff.removed_params = list(from_params - to_params)
        
        common_params = from_params & to_params
        for param in common_params:
            if from_ver.parameters[param] != to_ver.parameters[param]:
                diff.modified_params[param] = (from_ver.parameters[param], to_ver.parameters[param])
        
        # Generate summary
        changes = []
        if diff.added_rules: changes.append(f"Added {len(diff.added_rules)} rules")
        if diff.removed_rules: changes.append(f"Removed {len(diff.removed_rules)} rules")
        if diff.modified_rules: changes.append(f"Modified {len(diff.modified_rules)} rules")
        if diff.added_params: changes.append(f"Added {len(diff.added_params)} params")
        if diff.removed_params: changes.append(f"Removed {len(diff.removed_params)} params")
        if diff.modified_params: changes.append(f"Modified {len(diff.modified_params)} params")
        
        diff.summary = "; ".join(changes) if changes else "No changes"
        return diff
    
    def rollback(self, version_id: str, new_change_type: VersionType = VersionType.PATCH) -> StrategyVersion:
        """Rollback to a previous version, creating a new version."""
        target = self.versions.get(version_id)
        if not target:
            raise ValueError(f"Version {version_id} not found")
        
        # Create new version based on target
        rolled_back = StrategyVersion(
            name=f"{target.name} (rollback)",
            description=f"Rollback to version {target.version}",
            rules=target.rules.copy(),
            parameters=target.parameters.copy(),
            parent_id=target.id,
            version_type=VersionType.PATCH,
        )
        rolled_back.bump_version(new_change_type)
        
        self.versions[rolled_back.id] = rolled_back
        self._save_version(rolled_back)
        
        # Set as active
        self.set_active(rolled_back.id)
        
        return rolled_back
    
    def get_lineage(self, version_id: str) -> List[StrategyVersion]:
        """Get full lineage of a version."""
        lineage = []
        current = self.versions.get(version_id)
        while current:
            lineage.append(current)
            if current.parent_id:
                current = self.versions.get(current.parent_id)
            else:
                break
        return list(reversed(lineage))
    
    def get_children(self, version_id: str) -> List[StrategyVersion]:
        """Get direct children of a version."""
        version = self.versions.get(version_id)
        if not version:
            return []
        return [self.versions[cid] for cid in version.children_ids if cid in self.versions]
    
    def list_versions(self, status: Optional[str] = None) -> List[StrategyVersion]:
        """List all versions, optionally filtered by status."""
        versions = list(self.versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def deprecate(self, version_id: str) -> bool:
        """Mark a version as deprecated."""
        version = self.versions.get(version_id)
        if not version:
            return False
        version.status = "deprecated"
        self._save_version(version)
        return True
    
    def get_changelog(self, version_id: str) -> List[Dict[str, Any]]:
        """Get changelog for a version (all changes from root)."""
        lineage = self.get_lineage(version_id)
        changelog = []
        for i, v in enumerate(lineage):
            if i == 0:
                changelog.append({
                    "version": v.version,
                    "type": "initial",
                    "description": v.description,
                    "date": v.created_at.isoformat(),
                })
            else:
                diff = self.diff(lineage[i-1].id, v.id)
                changelog.append({
                    "version": v.version,
                    "type": v.version_type.value,
                    "description": v.description,
                    "date": v.created_at.isoformat(),
                    "diff": diff.to_dict(),
                })
        return changelog


# Factory
async def create_version_manager(
    storage_path: str = "strategies/versions",
    event_bus: Optional[EventBus] = None,
) -> StrategyVersionManager:
    """Create and initialize version manager."""
    manager = StrategyVersionManager(storage_path=storage_path, event_bus=event_bus)
    return manager