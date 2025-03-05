"""Interactive controls for mutation visualization."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
import json
from pathlib import Path
from aiohttp import web
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

from .mutation_visualization import MutationVisualizer, VisualizationConfig
from tests.proxy.analysis.monitoring.test_mutation_coverage import MutationTestResult

@dataclass
class ControlConfig:
    """Configuration for interactive controls."""
    port: int = 8060
    auto_refresh: bool = True
    refresh_interval: float = 5.0
    max_history: int = 100
    enable_websocket: bool = True
    template_dir: Optional[Path] = None

class InteractiveControls:
    """Interactive controls for mutation visualization."""
    
    def __init__(
        self,
        visualizer: MutationVisualizer,
        config: ControlConfig = None
    ):
        self.visualizer = visualizer
        self.config = config or ControlConfig()
        
        # Web components
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.ws_clients: Set[web.WebSocketResponse] = set()
        
        # Data storage
        self.history: List[MutationTestResult] = []
        self.filters: Dict[str, Any] = {
            "operators": set(),
            "error_types": set(),
            "score_range": [0, 1],
            "time_range": None
        }
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup web routes."""
        self.app.router.add_get("/", self._handle_index)
        self.app.router.add_get("/api/data", self._handle_data)
        self.app.router.add_post("/api/filters", self._handle_filters)
        self.app.router.add_get("/ws", self._handle_websocket)
        self.app.router.add_static(
            "/static",
            Path(__file__).parent / "static"
        )
    
    async def _handle_index(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle index page request."""
        html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Mutation Testing Analysis</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        font-family: sans-serif;
                    }
                    .controls {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;
                        margin-bottom: 20px;
                    }
                    .control-panel {
                        padding: 15px;
                        background: #f5f5f5;
                        border-radius: 5px;
                    }
                    .filter-group {
                        margin-bottom: 10px;
                    }
                    .plot-container {
                        display: grid;
                        gap: 20px;
                        margin-top: 20px;
                    }
                    .plot {
                        background: white;
                        padding: 15px;
                        border-radius: 5px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }
                </style>
            </head>
            <body>
                <div class="controls">
                    <div class="control-panel">
                        <h3>Operator Filters</h3>
                        <div class="filter-group" id="operator-filters"></div>
                    </div>
                    <div class="control-panel">
                        <h3>Error Filters</h3>
                        <div class="filter-group" id="error-filters"></div>
                    </div>
                    <div class="control-panel">
                        <h3>Score Range</h3>
                        <div class="filter-group">
                            <input type="range" id="score-min" min="0" max="1" step="0.1" value="0">
                            <input type="range" id="score-max" min="0" max="1" step="0.1" value="1">
                            <span id="score-range"></span>
                        </div>
                    </div>
                    <div class="control-panel">
                        <h3>Time Range</h3>
                        <div class="filter-group">
                            <input type="datetime-local" id="time-start">
                            <input type="datetime-local" id="time-end">
                        </div>
                    </div>
                </div>
                
                <div class="plot-container">
                    <div class="plot" id="summary-plot"></div>
                    <div class="plot" id="detail-plot"></div>
                </div>
                
                <script>
                    // WebSocket connection
                    const ws = new WebSocket(`ws://${location.host}/ws`);
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        updatePlots(data);
                    };
                    
                    // Control handlers
                    function updateFilters() {
                        const filters = {
                            operators: Array.from(document.querySelectorAll('#operator-filters input:checked')).map(cb => cb.value),
                            error_types: Array.from(document.querySelectorAll('#error-filters input:checked')).map(cb => cb.value),
                            score_range: [
                                parseFloat(document.getElementById('score-min').value),
                                parseFloat(document.getElementById('score-max').value)
                            ],
                            time_range: [
                                document.getElementById('time-start').value,
                                document.getElementById('time-end').value
                            ]
                        };
                        
                        fetch('/api/filters', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify(filters)
                        }).then(response => response.json())
                          .then(data => updatePlots(data));
                    }
                    
                    // Update operator filters
                    fetch('/api/data')
                        .then(response => response.json())
                        .then(data => {
                            const operators = new Set();
                            const errorTypes = new Set();
                            
                            data.forEach(result => {
                                Object.keys(result.operator_stats).forEach(op => operators.add(op));
                                result.errors.forEach(error => {
                                    errorTypes.add(error.split(':')[0]);
                                });
                            });
                            
                            const operatorFilters = document.getElementById('operator-filters');
                            operators.forEach(op => {
                                const div = document.createElement('div');
                                div.innerHTML = `
                                    <label>
                                        <input type="checkbox" value="${op}" checked>
                                        ${op}
                                    </label>
                                `;
                                operatorFilters.appendChild(div);
                            });
                            
                            const errorFilters = document.getElementById('error-filters');
                            errorTypes.forEach(type => {
                                const div = document.createElement('div');
                                div.innerHTML = `
                                    <label>
                                        <input type="checkbox" value="${type}" checked>
                                        ${type}
                                    </label>
                                `;
                                errorFilters.appendChild(div);
                            });
                            
                            // Add event listeners
                            document.querySelectorAll('input[type=checkbox]').forEach(cb => {
                                cb.addEventListener('change', updateFilters);
                            });
                            
                            document.querySelectorAll('input[type=range]').forEach(range => {
                                range.addEventListener('input', () => {
                                    const min = document.getElementById('score-min').value;
                                    const max = document.getElementById('score-max').value;
                                    document.getElementById('score-range').textContent = 
                                        `${min} - ${max}`;
                                    updateFilters();
                                });
                            });
                            
                            document.querySelectorAll('input[type=datetime-local]').forEach(dt => {
                                dt.addEventListener('change', updateFilters);
                            });
                            
                            // Initial update
                            updatePlots(data);
                        });
                    
                    function updatePlots(data) {
                        // Update summary plot
                        Plotly.react(
                            'summary-plot',
                            data.summary_data,
                            data.summary_layout
                        );
                        
                        // Update detail plot
                        Plotly.react(
                            'detail-plot',
                            data.detail_data,
                            data.detail_layout
                        );
                    }
                </script>
            </body>
            </html>
        """
        
        return web.Response(text=html, content_type="text/html")
    
    async def _handle_data(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle data request."""
        data = []
        for result in self.history:
            if self._apply_filters(result):
                data.append({
                    "operator_stats": result.operator_stats,
                    "errors": result.errors,
                    "mutation_score": result.mutation_score,
                    "timestamp": datetime.now().isoformat()
                })
        
        return web.json_response(data)
    
    async def _handle_filters(
        self,
        request: web.Request
    ) -> web.Response:
        """Handle filter updates."""
        filters = await request.json()
        self.filters.update(filters)
        
        # Get filtered data
        filtered_results = [
            result for result in self.history
            if self._apply_filters(result)
        ]
        
        # Generate plot data
        summary_fig = self.visualizer.create_summary_plot(filtered_results[-1])
        detail_fig = self.visualizer.create_detail_plot(filtered_results[-1])
        
        return web.json_response({
            "summary_data": summary_fig.data,
            "summary_layout": summary_fig.layout,
            "detail_data": detail_fig.data,
            "detail_layout": detail_fig.layout
        })
    
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
                if msg.type == web.WSMsgType.ERROR:
                    print(f"WebSocket error: {ws.exception()}")
        finally:
            self.ws_clients.remove(ws)
        
        return ws
    
    def _apply_filters(
        self,
        result: MutationTestResult
    ) -> bool:
        """Apply filters to result."""
        # Operator filter
        if self.filters["operators"]:
            if not any(
                op in self.filters["operators"]
                for op in result.operator_stats
            ):
                return False
        
        # Error type filter
        if self.filters["error_types"]:
            if not any(
                error.split(":")[0] in self.filters["error_types"]
                for error in result.errors
            ):
                return False
        
        # Score range filter
        if self.filters["score_range"]:
            min_score, max_score = self.filters["score_range"]
            if not min_score <= result.mutation_score <= max_score:
                return False
        
        # Time range filter
        if self.filters["time_range"]:
            start, end = self.filters["time_range"]
            timestamp = getattr(result, "timestamp", None)
            if timestamp:
                if not start <= timestamp <= end:
                    return False
        
        return True
    
    async def start(self):
        """Start interactive controls."""
        # Setup web server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(
            self.runner,
            "localhost",
            self.config.port
        )
        await self.site.start()
        
        print(f"Interactive controls running at http://localhost:{self.config.port}")
    
    async def stop(self):
        """Stop interactive controls."""
        if self.runner:
            await self.runner.cleanup()
        
        self.runner = None
        self.site = None
        
        # Close WebSocket connections
        for ws in self.ws_clients:
            await ws.close()
        self.ws_clients.clear()
    
    async def add_result(
        self,
        result: MutationTestResult
    ):
        """Add new test result."""
        self.history.append(result)
        
        # Trim history if needed
        while len(self.history) > self.config.max_history:
            self.history.pop(0)
        
        # Notify WebSocket clients
        if self.ws_clients:
            data = {
                "summary_data": self.visualizer.create_summary_plot(result).data,
                "detail_data": self.visualizer.create_detail_plot(result).data
            }
            
            for ws in self.ws_clients:
                try:
                    await ws.send_json(data)
                except Exception as e:
                    print(f"Failed to send update: {e}")

def create_interactive_controls(
    visualizer: MutationVisualizer,
    config: Optional[ControlConfig] = None
) -> InteractiveControls:
    """Create interactive controls."""
    return InteractiveControls(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .mutation_visualization import create_mutation_visualizer
    from tests.proxy.analysis.monitoring.test_mutation_coverage import (
        test_mutation_detection,
        mutator,
        test_state
    )
    
    async def main():
        # Setup components
        visualizer = create_mutation_visualizer()
        controls = create_interactive_controls(visualizer)
        
        # Start controls
        await controls.start()
        
        try:
            # Run some test mutations
            while True:
                result = await test_mutation_detection(mutator, test_state)
                await controls.add_result(result)
                await asyncio.sleep(5)
                
        finally:
            await controls.stop()
    
    asyncio.run(main())
