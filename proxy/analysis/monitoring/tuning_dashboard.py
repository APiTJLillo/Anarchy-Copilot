"""Real-time dashboard for tuning performance monitoring."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import json
import aiohttp
from aiohttp import web
import jinja2
import webbrowser

from .tuning_monitor import TuningMonitor, MonitoringConfig

logger = logging.getLogger(__name__)

@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard."""
    refresh_interval: float = 1.0  # Seconds between updates
    port: int = 8050  # Dashboard web port
    max_points: int = 1000  # Max points to display
    template_dir: Optional[Path] = None  # Custom template directory
    enable_websocket: bool = True  # Enable WebSocket updates
    auto_open: bool = True  # Auto-open browser

class TuningDashboard:
    """Real-time dashboard for tuning monitoring."""
    
    def __init__(
        self,
        monitor: TuningMonitor,
        config: DashboardConfig = None
    ):
        self.monitor = monitor
        self.config = config or DashboardConfig()
        
        # Web components
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.ws_clients: Set[web.WebSocketResponse] = set()
        
        # Setup routes
        self._setup_routes()
        self._setup_templates()
    
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/metrics", self._handle_metrics)
        self.app.router.add_get("/ws", self._handle_websocket)
        self.app.router.add_static(
            "/static",
            Path(__file__).parent / "static"
        )
    
    def _setup_templates(self):
        """Setup Jinja2 templates."""
        template_dir = (
            self.config.template_dir or
            Path(__file__).parent / "templates"
        )
        
        if not template_dir.exists():
            template_dir.mkdir(parents=True)
            self._create_default_templates(template_dir)
        
        self.templates = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True
        )
    
    def _create_default_templates(
        self,
        template_dir: Path
    ):
        """Create default dashboard templates."""
        templates = {
            "index.html": """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Tuning Performance Dashboard</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                    <style>
                        body { margin: 0; padding: 20px; font-family: sans-serif; }
                        .dashboard { display: grid; gap: 20px; }
                        .chart { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
                        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
                        .metric { padding: 15px; background: #f5f5f5; border-radius: 5px; }
                        .metric h3 { margin: 0 0 10px 0; }
                        .metric p { margin: 0; font-size: 24px; font-weight: bold; }
                        .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                        .alert.warning { background: #fff3cd; }
                        .alert.error { background: #f8d7da; }
                    </style>
                </head>
                <body>
                    <div class="dashboard">
                        <div class="summary" id="summary"></div>
                        <div class="chart" id="resources"></div>
                        <div class="chart" id="performance"></div>
                        <div class="chart" id="workers"></div>
                        <div id="alerts"></div>
                    </div>
                    <script>
                        const ws = new WebSocket(`ws://${location.host}/ws`);
                        ws.onmessage = function(event) {
                            const data = JSON.parse(event.data);
                            if (data.type === 'metrics') {
                                updateDashboard(data.metrics);
                            }
                        };
                        
                        function updateDashboard(metrics) {
                            updateSummary(metrics);
                            updateCharts(metrics);
                            updateAlerts(metrics.alerts);
                        }
                        
                        function updateSummary(metrics) {
                            const summary = document.getElementById('summary');
                            summary.innerHTML = `
                                <div class="metric">
                                    <h3>Active Trials</h3>
                                    <p>${metrics.tuning.active_trials}</p>
                                </div>
                                <div class="metric">
                                    <h3>Completed Trials</h3>
                                    <p>${metrics.tuning.completed_trials}</p>
                                </div>
                                <div class="metric">
                                    <h3>Trial Throughput</h3>
                                    <p>${metrics.tuning.trial_throughput.toFixed(2)}/s</p>
                                </div>
                                <div class="metric">
                                    <h3>Best Score</h3>
                                    <p>${metrics.tuning.best_score?.toFixed(4) || 'N/A'}</p>
                                </div>
                            `;
                        }
                        
                        const charts = {
                            resources: null,
                            performance: null,
                            workers: null
                        };
                        
                        function updateCharts(metrics) {
                            if (!charts.resources) {
                                charts.resources = Plotly.newPlot('resources', [{
                                    y: [metrics.resources.cpu_percent],
                                    type: 'line',
                                    name: 'CPU'
                                }, {
                                    y: [metrics.resources.memory_percent],
                                    type: 'line',
                                    name: 'Memory'
                                }], {
                                    title: 'System Resources',
                                    yaxis: { range: [0, 100] }
                                });
                            } else {
                                Plotly.extendTraces('resources', {
                                    y: [[metrics.resources.cpu_percent], [metrics.resources.memory_percent]]
                                }, [0, 1]);
                            }
                            
                            if (!charts.performance) {
                                charts.performance = Plotly.newPlot('performance', [{
                                    y: [metrics.tuning.trial_throughput],
                                    type: 'line',
                                    name: 'Throughput'
                                }], {
                                    title: 'Trial Performance'
                                });
                            } else {
                                Plotly.extendTraces('performance', {
                                    y: [[metrics.tuning.trial_throughput]]
                                }, [0]);
                            }
                            
                            if (!charts.workers) {
                                const workerData = Object.entries(metrics.workers.utilization)
                                    .map(([name, util]) => ({
                                        type: 'bar',
                                        name: name,
                                        y: [util * 100]
                                    }));
                                
                                charts.workers = Plotly.newPlot('workers', workerData, {
                                    title: 'Worker Utilization',
                                    yaxis: { range: [0, 100] }
                                });
                            } else {
                                Plotly.update('workers', {
                                    y: [Object.values(metrics.workers.utilization).map(v => v * 100)]
                                });
                            }
                        }
                        
                        function updateAlerts(alerts) {
                            const alertsDiv = document.getElementById('alerts');
                            alertsDiv.innerHTML = alerts.map(alert => `
                                <div class="alert ${alert.severity}">
                                    ${alert.timestamp}: ${alert.message}
                                </div>
                            `).join('');
                        }
                    </script>
                </body>
                </html>
            """
        }
        
        for name, content in templates.items():
            with open(template_dir / name, "w") as f:
                f.write(content.strip())
    
    async def _handle_index(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle index page request."""
        template = self.templates.get_template("index.html")
        html = template.render()
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_metrics(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle metrics request."""
        summary = self.monitor.get_performance_summary()
        return web.json_response(summary)
    
    async def _handle_websocket(
        self,
        request: web.Request
    ) -> web.WebSocketResponse:
        """Handle WebSocket connection."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.ws_clients.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
        finally:
            self.ws_clients.remove(ws)
        
        return ws
    
    async def _broadcast_metrics(self):
        """Broadcast metrics to all WebSocket clients."""
        if not self.ws_clients:
            return
        
        metrics = self.monitor.get_performance_summary()
        message = {
            "type": "metrics",
            "metrics": metrics
        }
        
        # Broadcast to all clients
        for ws in self.ws_clients:
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send metrics: {e}")
    
    async def _update_loop(self):
        """Main update loop."""
        while True:
            try:
                await self._broadcast_metrics()
            except Exception as e:
                logger.error(f"Update error: {e}")
            
            await asyncio.sleep(self.config.refresh_interval)
    
    async def start(self):
        """Start dashboard server."""
        # Start web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(
            self.runner,
            "localhost",
            self.config.port
        )
        await self.site.start()
        
        # Start update loop
        asyncio.create_task(self._update_loop())
        
        # Open browser
        if self.config.auto_open:
            webbrowser.open(f"http://localhost:{self.config.port}")
        
        logger.info(
            f"Dashboard running at http://localhost:{self.config.port}"
        )
    
    async def stop(self):
        """Stop dashboard server."""
        if self.runner:
            await self.runner.cleanup()
        
        self.runner = None
        self.site = None
        
        # Close all WebSocket connections
        for ws in self.ws_clients:
            await ws.close()
        
        self.ws_clients.clear()

def create_tuning_dashboard(
    monitor: TuningMonitor,
    config: Optional[DashboardConfig] = None
) -> TuningDashboard:
    """Create tuning dashboard."""
    return TuningDashboard(monitor, config)

if __name__ == "__main__":
    # Example usage
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
        
        # Create dashboard
        dashboard = create_tuning_dashboard(monitor)
        
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
