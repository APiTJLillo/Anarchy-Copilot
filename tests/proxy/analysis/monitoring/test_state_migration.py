"""Tests for dashboard state migrations."""

import asyncio
import pytest
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any

from proxy.analysis.monitoring.state_migration import (
    StateMigrator,
    MigrationMetadata,
    MigrationResult,
    create_state_migrator
)
from proxy.analysis.monitoring.dashboard_state import (
    DashboardState,
    StateManager,
    create_state_manager
)
from proxy.analysis.monitoring.dashboard_controls import (
    DashboardControls,
    add_dashboard_controls
)
from proxy.analysis.monitoring.tuning_dashboard import create_tuning_dashboard

@pytest.fixture
def sample_states() -> Dict[str, DashboardState]:
    """Sample states for testing."""
    return {
        "1.0.0": DashboardState(
            filters={
                "history": 3600,
                "chart_types": ["resources", "performance"],
                "alert_types": ["warning", "error"]
            },
            layout={
                "grid_cols": 2,
                "chart_order": ["resources", "performance"]
            },
            theme="light",
            display={
                "chart_height": 400,
                "max_alerts": 50
            },
            metadata={"version": "1.0.0"}
        ),
        "1.1.0": DashboardState(
            filters={
                "history": 3600,
                "chart_types": ["resources", "performance"],
                "alert_types": ["warning", "error"]
            },
            layout={
                "grid_cols": 2,
                "chart_order": ["resources", "performance"],
                "transitions": {
                    "type": "smooth",
                    "duration": 300
                }
            },
            theme="light",
            display={
                "chart_height": 400,
                "max_alerts": 50,
                "animations": {
                    "enabled": True,
                    "duration": 500,
                    "easing": "cubic-in-out"
                }
            },
            metadata={"version": "1.1.0"}
        ),
        "1.2.0": DashboardState(
            filters={
                "history": 3600,
                "chart_types": ["resources", "performance"],
                "alert_types": ["warning", "error"],
                "advanced": {
                    "enabled": False,
                    "rules": [],
                    "combine": "and"
                },
                "presets": []
            },
            layout={
                "grid_cols": 2,
                "chart_order": ["resources", "performance"],
                "transitions": {
                    "type": "smooth",
                    "duration": 300
                }
            },
            theme="light",
            display={
                "chart_height": 400,
                "max_alerts": 50,
                "animations": {
                    "enabled": True,
                    "duration": 500,
                    "easing": "cubic-in-out"
                }
            },
            metadata={"version": "1.2.0"}
        )
    }

@pytest.fixture
async def migrator(mocker):
    """Create test migrator."""
    # Mock dashboard components
    monitor = mocker.Mock()
    dashboard = create_tuning_dashboard(monitor)
    controls = add_dashboard_controls(dashboard)
    state_manager = create_state_manager(controls)
    
    return create_state_migrator(state_manager)

@pytest.mark.asyncio
async def test_migration_path_finding(migrator, sample_states):
    """Test migration path finding."""
    # Test direct path
    path = migrator._find_migration_path("1.0.0", "1.1.0")
    assert path == ["1.0.0", "1.1.0"]
    
    # Test multi-step path
    path = migrator._find_migration_path("1.0.0", "1.2.0")
    assert path == ["1.0.0", "1.1.0", "1.2.0"]
    
    # Test same version
    path = migrator._find_migration_path("1.1.0", "1.1.0")
    assert path == ["1.1.0"]
    
    # Test invalid path
    path = migrator._find_migration_path("1.0.0", "2.0.0")
    assert path is None

@pytest.mark.asyncio
async def test_version_validation(migrator):
    """Test version validation."""
    # Valid versions
    migrator.register_migration(
        "1.0.0",
        "1.1.0",
        lambda x: x,
        "Test migration"
    )
    
    # Invalid versions
    with pytest.raises(ValueError):
        migrator.register_migration(
            "invalid",
            "1.0.0",
            lambda x: x
        )
    
    with pytest.raises(ValueError):
        migrator.register_migration(
            "1.0.0",
            "invalid",
            lambda x: x
        )

@pytest.mark.asyncio
async def test_migration_1_0_to_1_1(migrator, sample_states):
    """Test migration from 1.0.0 to 1.1.0."""
    state_1_0 = sample_states["1.0.0"]
    expected_1_1 = sample_states["1.1.0"]
    
    result = await migrator.migrate_state(state_1_0, "1.1.0")
    
    assert result.success
    assert result.migrated_state is not None
    
    # Check animations were added
    assert result.migrated_state.display["animations"] == expected_1_1.display["animations"]
    
    # Check transitions were added
    assert result.migrated_state.layout["transitions"] == expected_1_1.layout["transitions"]
    
    # Check version was updated
    assert result.migrated_state.metadata["version"] == "1.1.0"

@pytest.mark.asyncio
async def test_migration_1_1_to_1_2(migrator, sample_states):
    """Test migration from 1.1.0 to 1.2.0."""
    state_1_1 = sample_states["1.1.0"]
    expected_1_2 = sample_states["1.2.0"]
    
    result = await migrator.migrate_state(state_1_1, "1.2.0")
    
    assert result.success
    assert result.migrated_state is not None
    
    # Check advanced filters were added
    assert result.migrated_state.filters["advanced"] == expected_1_2.filters["advanced"]
    
    # Check presets were added
    assert result.migrated_state.filters["presets"] == expected_1_2.filters["presets"]
    
    # Check version was updated
    assert result.migrated_state.metadata["version"] == "1.2.0"

@pytest.mark.asyncio
async def test_multi_step_migration(migrator, sample_states):
    """Test multi-step migration from 1.0.0 to 1.2.0."""
    state_1_0 = sample_states["1.0.0"]
    expected_1_2 = sample_states["1.2.0"]
    
    result = await migrator.migrate_state(state_1_0, "1.2.0")
    
    assert result.success
    assert result.migrated_state is not None
    
    # Check all features were added
    assert result.migrated_state.display["animations"] == expected_1_2.display["animations"]
    assert result.migrated_state.layout["transitions"] == expected_1_2.layout["transitions"]
    assert result.migrated_state.filters["advanced"] == expected_1_2.filters["advanced"]
    assert result.migrated_state.filters["presets"] == expected_1_2.filters["presets"]
    
    # Check version was updated
    assert result.migrated_state.metadata["version"] == "1.2.0"

@pytest.mark.asyncio
async def test_migration_failure_handling(migrator, sample_states):
    """Test handling of migration failures."""
    # Register failing migration
    def failing_migration(state):
        raise ValueError("Migration failed")
    
    migrator.register_migration(
        "1.2.0",
        "1.3.0",
        failing_migration,
        "Failing migration"
    )
    
    state_1_2 = sample_states["1.2.0"]
    result = await migrator.migrate_state(state_1_2, "1.3.0")
    
    assert not result.success
    assert result.migrated_state is None
    assert len(result.errors) > 0
    assert "Migration failed" in result.errors[0]

@pytest.mark.asyncio
async def test_compatibility_checking(migrator, sample_states):
    """Test compatibility checking between states."""
    states = list(sample_states.values())
    compatibility = await migrator.check_compatibility(states)
    
    # All states should be compatible with current version
    assert all(compatibility.values())
    
    # Add incompatible state
    incompatible_state = DashboardState(
        filters={},
        layout={},
        theme="light",
        display={},
        metadata={"version": "2.0.0"}
    )
    states.append(incompatible_state)
    
    compatibility = await migrator.check_compatibility(states)
    assert not compatibility["2.0.0"]

@pytest.mark.asyncio
async def test_migration_metadata(migrator):
    """Test migration metadata tracking."""
    result = await migrator.list_migrations(include_details=True)
    
    # Check each migration has metadata
    for source_version, targets in result.items():
        for target_version, migration_info in targets.items():
            metadata = migration_info["metadata"]
            assert isinstance(metadata, MigrationMetadata)
            assert metadata.source_version == source_version
            assert metadata.target_version == target_version

@pytest.mark.asyncio
async def test_state_preservation(migrator, sample_states):
    """Test preservation of existing state during migration."""
    state_1_0 = sample_states["1.0.0"]
    original_filters = state_1_0.filters.copy()
    
    result = await migrator.migrate_state(state_1_0, "1.2.0")
    
    assert result.success
    # Check original filters were preserved
    for key, value in original_filters.items():
        assert result.migrated_state.filters[key] == value

if __name__ == "__main__":
    pytest.main([__file__])
