"""State management for dashboard configurations."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import json
from pathlib import Path
import aiofiles
import hashlib
from collections import deque

from .dashboard_controls import DashboardControls, ControlConfig

logger = logging.getLogger(__name__)

@dataclass
class DashboardState:
    """Dashboard state configuration."""
    filters: Dict[str, Any]
    layout: Dict[str, Any]
    theme: str
    display: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StateConfig:
    """Configuration for state management."""
    state_dir: Path = Path("dashboard_states")
    max_history: int = 10
    auto_save: bool = True
    save_interval: int = 300  # seconds
    compression: bool = True

class StateManager:
    """Manage dashboard state persistence."""
    
    def __init__(
        self,
        controls: DashboardControls,
        config: StateConfig = None
    ):
        self.controls = controls
        self.config = config or StateConfig()
        
        # State storage
        self.current_state: Optional[DashboardState] = None
        self.state_history: deque[DashboardState] = deque(maxlen=self.config.max_history)
        
        # Setup storage
        self._setup_storage()
        
        # Add state routes
        self._add_state_routes()
        
        # Start auto-save if enabled
        if self.config.auto_save:
            self._start_auto_save()
    
    def _setup_storage(self):
        """Setup state storage."""
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
    
    def _add_state_routes(self):
        """Add state management routes."""
        self.controls.dashboard.app.router.add_get(
            "/api/state",
            self._handle_get_state
        )
        self.controls.dashboard.app.router.add_post(
            "/api/state",
            self._handle_save_state
        )
        self.controls.dashboard.app.router.add_get(
            "/api/state/history",
            self._handle_get_history
        )
        self.controls.dashboard.app.router.add_post(
            "/api/state/restore",
            self._handle_restore_state
        )
    
    def _get_current_state(self) -> DashboardState:
        """Get current dashboard state."""
        return DashboardState(
            filters={
                "history": self.controls.config.filter_history,
                "chart_types": self.controls.config.layout_config["chart_order"],
                "alert_types": ["warning", "error"]  # Default alert types
            },
            layout=self.controls.config.layout_config,
            theme=self.controls.config.theme,
            display={
                "chart_height": self.controls.config.chart_height,
                "max_alerts": self.controls.config.max_alerts
            },
            metadata={
                "version": "1.0",
                "user_agent": "dashboard"
            }
        )
    
    async def _save_state_to_file(
        self,
        state: DashboardState,
        filename: Optional[str] = None
    ) -> Path:
        """Save state to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dashboard_state_{timestamp}.json"
        
        file_path = self.config.state_dir / filename
        
        # Convert state to dict
        state_dict = asdict(state)
        state_dict["timestamp"] = state.timestamp.isoformat()
        
        # Save state
        async with aiofiles.open(file_path, "w") as f:
            await f.write(json.dumps(state_dict, indent=2))
        
        return file_path
    
    async def _load_state_from_file(
        self,
        file_path: Path
    ) -> DashboardState:
        """Load state from file."""
        async with aiofiles.open(file_path) as f:
            content = await f.read()
            data = json.loads(content)
        
        # Convert timestamp back to datetime
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        
        return DashboardState(**data)
    
    def _hash_state(
        self,
        state: DashboardState
    ) -> str:
        """Generate hash for state."""
        state_dict = asdict(state)
        state_dict["timestamp"] = state.timestamp.isoformat()
        return hashlib.sha256(
            json.dumps(state_dict, sort_keys=True).encode()
        ).hexdigest()
    
    async def _handle_get_state(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle state request."""
        state = self._get_current_state()
        return web.json_response(asdict(state))
    
    async def _handle_save_state(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle state save request."""
        state = self._get_current_state()
        
        # Save to history
        self.state_history.append(state)
        self.current_state = state
        
        # Save to file
        file_path = await self._save_state_to_file(state)
        
        return web.json_response({
            "status": "saved",
            "file": str(file_path),
            "hash": self._hash_state(state)
        })
    
    async def _handle_get_history(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle history request."""
        history = [
            {
                "timestamp": state.timestamp.isoformat(),
                "hash": self._hash_state(state),
                "filters": state.filters,
                "layout": state.layout,
                "theme": state.theme
            }
            for state in self.state_history
        ]
        
        return web.json_response(history)
    
    async def _handle_restore_state(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle state restore request."""
        data = await request.json()
        state_hash = data.get("hash")
        
        # Find state in history
        for state in self.state_history:
            if self._hash_state(state) == state_hash:
                await self._apply_state(state)
                return web.json_response({
                    "status": "restored",
                    "timestamp": state.timestamp.isoformat()
                })
        
        return web.Response(
            status=404,
            text="State not found"
        )
    
    async def _apply_state(
        self,
        state: DashboardState
    ):
        """Apply dashboard state."""
        # Update controls config
        self.controls.config.filter_history = state.filters["history"]
        self.controls.config.layout_config = state.layout
        self.controls.config.theme = state.theme
        self.controls.config.chart_height = state.display["chart_height"]
        self.controls.config.max_alerts = state.display["max_alerts"]
        
        # Update dashboard
        await self.controls._handle_filter(None)
        await self.controls._handle_layout(None)
        await self.controls._handle_theme(None)
    
    def _start_auto_save(self):
        """Start auto-save loop."""
        async def auto_save_loop():
            while True:
                try:
                    state = self._get_current_state()
                    await self._save_state_to_file(state)
                    logger.debug("Auto-saved dashboard state")
                except Exception as e:
                    logger.error(f"Auto-save failed: {e}")
                
                await asyncio.sleep(self.config.save_interval)
        
        asyncio.create_task(auto_save_loop())
    
    async def restore_latest(self) -> bool:
        """Restore latest state."""
        try:
            # Get most recent state file
            state_files = sorted(
                self.config.state_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if not state_files:
                return False
            
            state = await self._load_state_from_file(state_files[0])
            await self._apply_state(state)
            
            self.current_state = state
            self.state_history.append(state)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state: {e}")
            return False

def create_state_manager(
    controls: DashboardControls,
    config: Optional[StateConfig] = None
) -> StateManager:
    """Create state manager."""
    return StateManager(controls, config)

if __name__ == "__main__":
    # Example usage
    from .dashboard_controls import add_dashboard_controls
    from .tuning_dashboard import create_tuning_dashboard
    from .tuning_monitor import create_tuning_monitor
    from .distributed_tuning import create_distributed_tuner
    from .priority_tuning import create_priority_tuner
    from .priority_validation import create_priority_validator
    from .adaptive_priority import create_priority_learner
    from .notification_priority import create_priority_router
    from .notification_throttling import create_throttled_manager
    from .notification_channels import create_notification_manager
    
    async def main():
        # Create monitoring stack
        manager = create_notification_manager()
        throttler = create_throttled_manager(manager)
        router = create_priority_router(throttler)
        learner = create_priority_learner(router)
        validator = create_priority_validator(learner)
        tuner = create_priority_tuner(validator)
        dist_tuner = create_distributed_tuner(tuner)
        monitor = create_tuning_monitor(dist_tuner)
        
        # Create dashboard with controls and state
        dashboard = create_tuning_dashboard(monitor)
        controls = add_dashboard_controls(dashboard)
        state_manager = create_state_manager(controls)
        
        # Start components
        await monitor.start_monitoring()
        await dashboard.start()
        
        # Restore previous state
        await state_manager.restore_latest()
        
        try:
            # Run distributed tuning
            result = await dist_tuner.tune_distributed()
            print("Tuning completed:", json.dumps(result, indent=2))
            
        finally:
            await dashboard.stop()
            await monitor.stop_monitoring()
    
    asyncio.run(main())
