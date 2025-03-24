"""State migration for dashboard configurations."""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import semver
from pathlib import Path
import copy
from collections import defaultdict

from .dashboard_state import DashboardState, StateManager

logger = logging.getLogger(__name__)

@dataclass
class MigrationMetadata:
    """Metadata for state migration."""
    source_version: str
    target_version: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    backwards_compatible: bool = True

@dataclass
class MigrationResult:
    """Result of state migration."""
    success: bool
    migrated_state: Optional[DashboardState]
    metadata: MigrationMetadata
    changes: List[str]
    errors: List[str]

class StateMigrator:
    """Manage state migrations between versions."""
    
    def __init__(
        self,
        state_manager: StateManager
    ):
        self.state_manager = state_manager
        self.migrations: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.current_version = "1.0.0"
        
        # Register built-in migrations
        self._register_migrations()
    
    def _register_migrations(self):
        """Register built-in migration handlers."""
        # 1.0.0 -> 1.1.0: Add chart animations
        self.register_migration(
            "1.0.0",
            "1.1.0",
            self._migrate_1_0_to_1_1
        )
        
        # 1.1.0 -> 1.2.0: Add advanced filters
        self.register_migration(
            "1.1.0",
            "1.2.0",
            self._migrate_1_1_to_1_2
        )
    
    def register_migration(
        self,
        source_version: str,
        target_version: str,
        handler: Callable[[DashboardState], DashboardState],
        description: str = "",
        backwards_compatible: bool = True
    ):
        """Register migration handler."""
        if not semver.VersionInfo.is_valid(source_version):
            raise ValueError(f"Invalid source version: {source_version}")
        if not semver.VersionInfo.is_valid(target_version):
            raise ValueError(f"Invalid target version: {target_version}")
        
        self.migrations[source_version][target_version] = {
            "handler": handler,
            "metadata": MigrationMetadata(
                source_version=source_version,
                target_version=target_version,
                description=description,
                backwards_compatible=backwards_compatible
            )
        }
        
        logger.info(
            f"Registered migration {source_version} -> {target_version}"
        )
    
    async def migrate_state(
        self,
        state: DashboardState,
        target_version: Optional[str] = None
    ) -> MigrationResult:
        """Migrate state to target version."""
        if not target_version:
            target_version = self.current_version
        
        current_version = state.metadata.get("version", "1.0.0")
        
        if current_version == target_version:
            return MigrationResult(
                success=True,
                migrated_state=state,
                metadata=MigrationMetadata(
                    source_version=current_version,
                    target_version=target_version,
                    description="No migration needed"
                ),
                changes=[],
                errors=[]
            )
        
        # Find migration path
        path = self._find_migration_path(
            current_version,
            target_version
        )
        
        if not path:
            return MigrationResult(
                success=False,
                migrated_state=None,
                metadata=MigrationMetadata(
                    source_version=current_version,
                    target_version=target_version,
                    description="No migration path found"
                ),
                changes=[],
                errors=[f"No migration path from {current_version} to {target_version}"]
            )
        
        # Execute migrations
        migrated_state = copy.deepcopy(state)
        changes = []
        errors = []
        
        try:
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                migration = self.migrations[source][target]
                handler = migration["handler"]
                metadata = migration["metadata"]
                
                try:
                    migrated_state = handler(migrated_state)
                    changes.append(
                        f"Migrated {source} -> {target}: {metadata.description}"
                    )
                except Exception as e:
                    errors.append(
                        f"Failed to migrate {source} -> {target}: {e}"
                    )
                    raise
            
            # Update version
            migrated_state.metadata["version"] = target_version
            
            return MigrationResult(
                success=True,
                migrated_state=migrated_state,
                metadata=MigrationMetadata(
                    source_version=current_version,
                    target_version=target_version,
                    description="Migration completed"
                ),
                changes=changes,
                errors=errors
            )
            
        except Exception as e:
            return MigrationResult(
                success=False,
                migrated_state=None,
                metadata=MigrationMetadata(
                    source_version=current_version,
                    target_version=target_version,
                    description=f"Migration failed: {e}"
                ),
                changes=changes,
                errors=errors + [str(e)]
            )
    
    def _find_migration_path(
        self,
        source: str,
        target: str
    ) -> Optional[List[str]]:
        """Find migration path between versions."""
        if source == target:
            return [source]
        
        queue = [(source, [source])]
        visited = {source}
        
        while queue:
            current, path = queue.pop(0)
            
            for next_version in self.migrations[current]:
                if next_version == target:
                    return path + [target]
                
                if next_version not in visited:
                    visited.add(next_version)
                    queue.append(
                        (next_version, path + [next_version])
                    )
        
        return None
    
    def _migrate_1_0_to_1_1(
        self,
        state: DashboardState
    ) -> DashboardState:
        """Migrate from v1.0.0 to v1.1.0."""
        # Add chart animations
        state.display["animations"] = {
            "enabled": True,
            "duration": 500,
            "easing": "cubic-in-out"
        }
        
        # Add chart transitions
        state.layout["transitions"] = {
            "type": "smooth",
            "duration": 300
        }
        
        return state
    
    def _migrate_1_1_to_1_2(
        self,
        state: DashboardState
    ) -> DashboardState:
        """Migrate from v1.1.0 to v1.2.0."""
        # Add advanced filters
        state.filters["advanced"] = {
            "enabled": False,
            "rules": [],
            "combine": "and"
        }
        
        # Add filter presets
        state.filters["presets"] = []
        
        return state
    
    async def validate_migration(
        self,
        state: DashboardState,
        target_version: str
    ) -> bool:
        """Validate migration compatibility."""
        current_version = state.metadata.get("version", "1.0.0")
        
        if current_version == target_version:
            return True
        
        path = self._find_migration_path(current_version, target_version)
        if not path:
            return False
        
        # Check backward compatibility
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            
            migration = self.migrations[source][target]
            if not migration["metadata"].backwards_compatible:
                return False
        
        return True
    
    async def list_migrations(
        self,
        include_details: bool = False
    ) -> Dict[str, Any]:
        """List available migrations."""
        migrations = {}
        
        for source in self.migrations:
            migrations[source] = {}
            for target, migration in self.migrations[source].items():
                if include_details:
                    migrations[source][target] = {
                        "metadata": migration["metadata"]
                    }
                else:
                    migrations[source][target] = migration["metadata"].description
        
        return migrations
    
    async def check_compatibility(
        self,
        states: List[DashboardState]
    ) -> Dict[str, bool]:
        """Check compatibility between states."""
        results = {}
        
        for state in states:
            version = state.metadata.get("version", "1.0.0")
            compatible = await self.validate_migration(
                state,
                self.current_version
            )
            results[version] = compatible
        
        return results

def create_state_migrator(
    state_manager: StateManager
) -> StateMigrator:
    """Create state migrator."""
    return StateMigrator(state_manager)

if __name__ == "__main__":
    # Example usage
    from .dashboard_state import create_state_manager
    from .dashboard_controls import add_dashboard_controls
    from .tuning_dashboard import create_tuning_dashboard
    from .tuning_monitor import create_tuning_monitor
    
    async def main():
        # Create components
        monitor = create_tuning_monitor(None)
        dashboard = create_tuning_dashboard(monitor)
        controls = add_dashboard_controls(dashboard)
        state_manager = create_state_manager(controls)
        migrator = create_state_migrator(state_manager)
        
        # Example migration
        old_state = DashboardState(
            filters={},
            layout={},
            theme="light",
            display={},
            metadata={"version": "1.0.0"}
        )
        
        result = await migrator.migrate_state(
            old_state,
            target_version="1.2.0"
        )
        
        if result.success:
            print("Migration successful!")
            print("Changes:", json.dumps(result.changes, indent=2))
            print("New state:", json.dumps(
                result.migrated_state.metadata,
                indent=2
            ))
        else:
            print("Migration failed:", result.errors)
    
    asyncio.run(main())
