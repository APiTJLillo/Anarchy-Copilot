"""Interactive controls for tuning dashboard."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import plotly.graph_objects as go

from .tuning_dashboard import TuningDashboard, DashboardConfig

logger = logging.getLogger(__name__)

@dataclass
class ControlConfig:
    """Configuration for dashboard controls."""
    filter_history: int = 3600  # Seconds of history to show
    chart_height: int = 400
    max_alerts: int = 50
    theme: str = "light"
    layout_config: Dict[str, Any] = field(default_factory=lambda: {
        "grid_cols": 2,
        "chart_order": [
            "resources",
            "performance",
            "workers",
            "alerts"
        ]
    })

class DashboardControls:
    """Interactive controls for tuning dashboard."""
    
    def __init__(
        self,
        dashboard: TuningDashboard,
        config: ControlConfig = None
    ):
        self.dashboard = dashboard
        self.config = config or ControlConfig()
        
        # Add control routes
        self._add_control_routes()
        
        # Update template
        self._update_template()
    
    def _add_control_routes(self):
        """Add control API routes."""
        self.dashboard.app.router.add_post(
            "/api/controls/filter",
            self._handle_filter
        )
        self.dashboard.app.router.add_post(
            "/api/controls/layout",
            self._handle_layout
        )
        self.dashboard.app.router.add_post(
            "/api/controls/theme",
            self._handle_theme
        )
        self.dashboard.app.router.add_get(
            "/api/controls/config",
            self._handle_config
        )
    
    def _update_template(self):
        """Update dashboard template with controls."""
        controls_html = """
            <div class="controls">
                <div class="control-panel">
                    <h3>Filters</h3>
                    <div class="filter-group">
                        <label>History Window (seconds)</label>
                        <input type="number" id="history-filter" value="{{ filter_history }}"
                            min="60" max="86400" step="60">
                    </div>
                    <div class="filter-group">
                        <label>Chart Types</label>
                        <select id="chart-type-filter" multiple>
                            <option value="resources" selected>Resources</option>
                            <option value="performance" selected>Performance</option>
                            <option value="workers" selected>Workers</option>
                            <option value="alerts" selected>Alerts</option>
                        </select>
                    </div>
                    <div class="filter-group">
                        <label>Alert Severity</label>
                        <select id="alert-filter" multiple>
                            <option value="warning" selected>Warning</option>
                            <option value="error" selected>Error</option>
                        </select>
                    </div>
                </div>
                
                <div class="control-panel">
                    <h3>Layout</h3>
                    <div class="layout-group">
                        <label>Grid Columns</label>
                        <input type="number" id="grid-cols" value="{{ layout_config.grid_cols }}"
                            min="1" max="4">
                    </div>
                    <div class="layout-group">
                        <label>Chart Height</label>
                        <input type="number" id="chart-height" value="{{ chart_height }}"
                            min="200" max="1000" step="50">
                    </div>
                </div>
                
                <div class="control-panel">
                    <h3>Display</h3>
                    <div class="theme-group">
                        <label>Theme</label>
                        <select id="theme-select">
                            <option value="light" {% if theme == "light" %}selected{% endif %}>Light</option>
                            <option value="dark" {% if theme == "dark" %}selected{% endif %}>Dark</option>
                        </select>
                    </div>
                    <div class="display-group">
                        <label>Max Alerts</label>
                        <input type="number" id="max-alerts" value="{{ max_alerts }}"
                            min="10" max="1000">
                    </div>
                </div>
            </div>
            
            <style>
                .controls {
                    display: flex;
                    gap: 20px;
                    margin-bottom: 20px;
                    padding: 15px;
                    background: var(--control-bg, #f5f5f5);
                    border-radius: 5px;
                }
                
                .control-panel {
                    flex: 1;
                    padding: 10px;
                }
                
                .control-panel h3 {
                    margin: 0 0 10px 0;
                    font-size: 16px;
                }
                
                .filter-group,
                .layout-group,
                .theme-group,
                .display-group {
                    margin-bottom: 10px;
                }
                
                label {
                    display: block;
                    margin-bottom: 5px;
                    font-size: 14px;
                }
                
                input,
                select {
                    width: 100%;
                    padding: 5px;
                    border: 1px solid #ddd;
                    border-radius: 3px;
                }
                
                select[multiple] {
                    height: 100px;
                }
                
                /* Dark theme */
                [data-theme="dark"] {
                    --control-bg: #2d2d2d;
                    --text-color: #fff;
                    color: var(--text-color);
                }
                
                [data-theme="dark"] input,
                [data-theme="dark"] select {
                    background: #404040;
                    color: #fff;
                    border-color: #555;
                }
            </style>
            
            <script>
                // Add control handlers
                document.getElementById('history-filter').addEventListener('change', (e) => {
                    updateFilters({ history: parseInt(e.target.value) });
                });
                
                document.getElementById('chart-type-filter').addEventListener('change', (e) => {
                    const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
                    updateFilters({ charts: selected });
                });
                
                document.getElementById('alert-filter').addEventListener('change', (e) => {
                    const selected = Array.from(e.target.selectedOptions).map(opt => opt.value);
                    updateFilters({ alertTypes: selected });
                });
                
                document.getElementById('grid-cols').addEventListener('change', (e) => {
                    updateLayout({ gridCols: parseInt(e.target.value) });
                });
                
                document.getElementById('chart-height').addEventListener('change', (e) => {
                    updateLayout({ chartHeight: parseInt(e.target.value) });
                });
                
                document.getElementById('theme-select').addEventListener('change', (e) => {
                    updateTheme(e.target.value);
                });
                
                document.getElementById('max-alerts').addEventListener('change', (e) => {
                    updateDisplay({ maxAlerts: parseInt(e.target.value) });
                });
                
                // Control update functions
                async function updateFilters(filters) {
                    const response = await fetch('/api/controls/filter', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(filters)
                    });
                    if (response.ok) {
                        const result = await response.json();
                        updateDashboard(result.metrics);
                    }
                }
                
                async function updateLayout(layout) {
                    const response = await fetch('/api/controls/layout', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(layout)
                    });
                    if (response.ok) {
                        const result = await response.json();
                        applyLayout(result.layout);
                    }
                }
                
                async function updateTheme(theme) {
                    const response = await fetch('/api/controls/theme', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ theme })
                    });
                    if (response.ok) {
                        document.body.setAttribute('data-theme', theme);
                        updateChartTheme(theme);
                    }
                }
                
                function updateChartTheme(theme) {
                    const bgColor = theme === 'dark' ? '#1a1a1a' : '#ffffff';
                    const textColor = theme === 'dark' ? '#ffffff' : '#000000';
                    
                    Object.values(charts).forEach(chart => {
                        Plotly.relayout(chart, {
                            paper_bgcolor: bgColor,
                            plot_bgcolor: bgColor,
                            font: { color: textColor }
                        });
                    });
                }
                
                function applyLayout(layout) {
                    const dashboard = document.querySelector('.dashboard');
                    dashboard.style.gridTemplateColumns = `repeat(${layout.gridCols}, 1fr)`;
                    
                    const charts = document.querySelectorAll('.chart');
                    charts.forEach(chart => {
                        chart.style.height = `${layout.chartHeight}px`;
                    });
                    
                    // Reorder charts
                    const order = layout.chartOrder;
                    order.forEach((id, index) => {
                        const elem = document.getElementById(id);
                        if (elem) {
                            elem.style.order = index;
                        }
                    });
                    
                    // Resize charts
                    Object.values(window.charts).forEach(chart => {
                        Plotly.Plots.resize(chart);
                    });
                }
                
                // Load initial config
                async function loadConfig() {
                    const response = await fetch('/api/controls/config');
                    if (response.ok) {
                        const config = await response.json();
                        applyConfig(config);
                    }
                }
                
                function applyConfig(config) {
                    document.getElementById('history-filter').value = config.filter_history;
                    document.getElementById('grid-cols').value = config.layout_config.grid_cols;
                    document.getElementById('chart-height').value = config.chart_height;
                    document.getElementById('theme-select').value = config.theme;
                    document.getElementById('max-alerts').value = config.max_alerts;
                    
                    // Apply theme
                    updateTheme(config.theme);
                    
                    // Apply layout
                    applyLayout(config.layout_config);
                }
                
                // Initialize controls
                loadConfig();
            </script>
        """
        
        # Update template with controls
        template = self.dashboard.templates.get_template("index.html")
        template_source = template.source
        template_source = template_source.replace(
            '<div class="dashboard">',
            controls_html + '<div class="dashboard">'
        )
        
        # Update template
        self.dashboard.templates.globals.update({
            "filter_history": self.config.filter_history,
            "chart_height": self.config.chart_height,
            "max_alerts": self.config.max_alerts,
            "theme": self.config.theme,
            "layout_config": self.config.layout_config
        })
    
    async def _handle_filter(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle filter updates."""
        data = await request.json()
        
        if "history" in data:
            self.config.filter_history = data["history"]
        
        # Apply filters to metrics
        metrics = self.dashboard.monitor.get_performance_summary()
        
        # Filter metrics based on time window
        now = datetime.now()
        cutoff = now - timedelta(seconds=self.config.filter_history)
        
        filtered_metrics = {
            k: v for k, v in metrics.items()
            if isinstance(v, dict) and "timestamp" in v and v["timestamp"] >= cutoff
        }
        
        return web.json_response({
            "metrics": filtered_metrics
        })
    
    async def _handle_layout(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle layout updates."""
        data = await request.json()
        
        if "gridCols" in data:
            self.config.layout_config["grid_cols"] = data["gridCols"]
        
        if "chartHeight" in data:
            self.config.chart_height = data["chartHeight"]
        
        return web.json_response({
            "layout": self.config.layout_config
        })
    
    async def _handle_theme(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle theme updates."""
        data = await request.json()
        self.config.theme = data.get("theme", "light")
        
        return web.json_response({
            "theme": self.config.theme
        })
    
    async def _handle_config(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle config requests."""
        return web.json_response({
            "filter_history": self.config.filter_history,
            "chart_height": self.config.chart_height,
            "max_alerts": self.config.max_alerts,
            "theme": self.config.theme,
            "layout_config": self.config.layout_config
        })

def add_dashboard_controls(
    dashboard: TuningDashboard,
    config: Optional[ControlConfig] = None
) -> DashboardControls:
    """Add interactive controls to dashboard."""
    return DashboardControls(dashboard, config)

if __name__ == "__main__":
    # Example usage
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
        
        # Create dashboard with controls
        dashboard = create_tuning_dashboard(monitor)
        controls = add_dashboard_controls(dashboard)
        
        # Start monitoring and dashboard
        await monitor.start_monitoring()
        await dashboard.start()
        
        try:
            # Run distributed tuning
            result = await dist_tuner.tune_distributed()
            print("Tuning completed:", json.dumps(result, indent=2))
            
        finally:
            await dashboard.stop()
            await monitor.stop_monitoring()
    
    asyncio.run(main())
