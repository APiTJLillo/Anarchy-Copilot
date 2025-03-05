"""Fuzz testing for state migrations."""

import asyncio
import pytest
import random
import string
import json
from typing import Dict, Any, List
from datetime import datetime, timedelta
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule
import hypothesis.strategies as st
from copy import deepcopy

from proxy.analysis.monitoring.state_migration import (
    StateMigrator,
    MigrationMetadata,
    MigrationResult,
    create_state_migrator
)
from proxy.analysis.monitoring.dashboard_state import DashboardState

# Custom Hypothesis strategies
@st.composite
def dashboard_filters(draw) -> Dict[str, Any]:
    """Generate random dashboard filters."""
    return {
        "history": draw(st.integers(min_value=60, max_value=86400)),
        "chart_types": draw(st.lists(
            st.sampled_from(["resources", "performance", "workers", "alerts"]),
            min_size=1,
            unique=True
        )),
        "alert_types": draw(st.lists(
            st.sampled_from(["warning", "error", "info", "debug"]),
            min_size=1,
            unique=True
        ))
    }

@st.composite
def dashboard_layout(draw) -> Dict[str, Any]:
    """Generate random dashboard layout."""
    return {
        "grid_cols": draw(st.integers(min_value=1, max_value=4)),
        "chart_order": draw(st.lists(
            st.sampled_from(["resources", "performance", "workers", "alerts"]),
            unique=True
        )),
        "transitions": {
            "type": draw(st.sampled_from(["none", "fade", "slide", "smooth"])),
            "duration": draw(st.integers(min_value=0, max_value=1000))
        }
    }

@st.composite
def dashboard_display(draw) -> Dict[str, Any]:
    """Generate random dashboard display settings."""
    return {
        "chart_height": draw(st.integers(min_value=200, max_value=1000)),
        "max_alerts": draw(st.integers(min_value=10, max_value=1000)),
        "animations": {
            "enabled": draw(st.booleans()),
            "duration": draw(st.integers(min_value=0, max_value=1000)),
            "easing": draw(st.sampled_from([
                "linear", "ease-in", "ease-out", "ease-in-out",
                "cubic-in", "cubic-out", "cubic-in-out"
            ]))
        }
    }

@st.composite
def dashboard_state(draw) -> DashboardState:
    """Generate random dashboard state."""
    return DashboardState(
        filters=draw(dashboard_filters()),
        layout=draw(dashboard_layout()),
        theme=draw(st.sampled_from(["light", "dark", "custom"])),
        display=draw(dashboard_display()),
        metadata={
            "version": draw(st.sampled_from(["1.0.0", "1.1.0", "1.2.0"])),
            "timestamp": datetime.now().isoformat(),
            "user": draw(st.text(min_size=1, max_size=50))
        }
    )

class MigrationStateMachine(RuleBasedStateMachine):
    """State machine for testing migrations."""
    
    def __init__(self):
        super().__init__()
        self.states: List[DashboardState] = []
        self.migrator = None
    
    async def setup(self):
        """Setup test environment."""
        # Create migrator
        monitor = None
        self.migrator = create_state_migrator(None)
    
    @rule(state=dashboard_state())
    async def add_state(self, state):
        """Add new state."""
        self.states.append(state)
        
        # Try migrating to latest version
        result = await self.migrator.migrate_state(
            state,
            self.migrator.current_version
        )
        
        if result.success:
            assert result.migrated_state.metadata["version"] == self.migrator.current_version
    
    @rule(state=st.sampled_from([
        "1.0.0", "1.1.0", "1.2.0", "invalid", "2.0.0"
    ]))
    async def migrate_to_version(self, version):
        """Try migrating to specific version."""
        if not self.states:
            return
        
        state = random.choice(self.states)
        result = await self.migrator.migrate_state(state, version)
        
        if semver.VersionInfo.is_valid(version):
            if result.success:
                assert result.migrated_state.metadata["version"] == version
        else:
            assert not result.success
    
    @rule()
    async def corrupt_state(self):
        """Randomly corrupt state to test robustness."""
        if not self.states:
            return
        
        state = random.choice(self.states)
        corrupted = deepcopy(state)
        
        # Apply random corruption
        corruption_type = random.choice([
            "remove_field",
            "change_type",
            "invalid_value",
            "nested_corruption"
        ])
        
        if corruption_type == "remove_field":
            field = random.choice(list(vars(corrupted).keys()))
            delattr(corrupted, field)
        
        elif corruption_type == "change_type":
            field = random.choice(list(vars(corrupted).keys()))
            setattr(corrupted, field, random.randint(0, 100))
        
        elif corruption_type == "invalid_value":
            field = random.choice(list(vars(corrupted).keys()))
            setattr(corrupted, field, None)
        
        elif corruption_type == "nested_corruption":
            if corrupted.filters:
                key = random.choice(list(corrupted.filters.keys()))
                corrupted.filters[key] = None
        
        # Try migrating corrupted state
        result = await self.migrator.migrate_state(
            corrupted,
            self.migrator.current_version
        )
        assert not result.success

@pytest.mark.asyncio
async def test_migration_state_machine():
    """Run state machine testing."""
    await MigrationStateMachine.TestCase.run()

@pytest.mark.asyncio
@given(state=dashboard_state())
async def test_migration_robustness(state, migrator):
    """Test migration with random valid states."""
    result = await migrator.migrate_state(state)
    
    if result.success:
        migrated = result.migrated_state
        
        # Check structure is maintained
        assert isinstance(migrated.filters, dict)
        assert isinstance(migrated.layout, dict)
        assert isinstance(migrated.display, dict)
        assert isinstance(migrated.metadata, dict)
        
        # Check version is valid
        assert semver.VersionInfo.is_valid(migrated.metadata["version"])

@pytest.mark.asyncio
async def test_migration_stress(migrator):
    """Test migrations under stress."""
    states = []
    tasks = []
    
    # Generate random states
    for _ in range(100):
        state = await st.builds(dashboard_state).example()
        states.append(state)
    
    # Migrate states concurrently
    async def migrate_state(state):
        return await migrator.migrate_state(state)
    
    for state in states:
        task = asyncio.create_task(migrate_state(state))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Check results
    success_count = sum(1 for r in results if r.success)
    assert success_count > 0

@pytest.mark.asyncio
@given(st.lists(dashboard_state(), min_size=1))
async def test_migration_chain(states, migrator):
    """Test chained migrations."""
    for i in range(len(states) - 1):
        source = states[i]
        target = states[i + 1]
        
        # Try migrating between consecutive states
        result = await migrator.migrate_state(
            source,
            target.metadata["version"]
        )
        
        if result.success:
            # Check migration maintained data integrity
            migrated = result.migrated_state
            assert migrated.theme in ["light", "dark", "custom"]
            assert all(isinstance(v, (int, float)) for v in migrated.display.values())

@pytest.mark.asyncio
async def test_invalid_states(migrator):
    """Test handling of invalid states."""
    invalid_states = [
        None,
        {},
        DashboardState(
            filters=None,
            layout=None,
            theme=None,
            display=None,
            metadata=None
        ),
        "not a state",
        42
    ]
    
    for state in invalid_states:
        with pytest.raises(Exception):
            await migrator.migrate_state(state)

@pytest.mark.asyncio
async def test_concurrent_migrations(migrator):
    """Test concurrent migration handling."""
    async def migrate_random_state():
        state = await st.builds(dashboard_state).example()
        return await migrator.migrate_state(state)
    
    # Run multiple migrations concurrently
    tasks = [
        migrate_random_state()
        for _ in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Check no migrations interfered with each other
    for result in results:
        if result.success:
            assert result.migrated_state is not None
            assert result.migrated_state.metadata["version"] == migrator.current_version

if __name__ == "__main__":
    pytest.main([__file__])
