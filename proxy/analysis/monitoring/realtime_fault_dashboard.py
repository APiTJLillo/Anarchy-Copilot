"""Real-time fault monitoring dashboard."""

import asyncio
import dash
from dash import html, dcc, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import logging
from collections import deque
import threading
from aiohttp import web
import socketio
from dataclasses import asdict

from .fault_visualization import FaultVisualizer
from .test_fault_correlation import FaultCorrelationAnalyzer, FaultEvent

logger = logging.getLogger(__name__)

class RealtimeFaultDashboard:
    """Real-time dashboard for fault monitoring."""
    
    def __init__(
        self,
        analyzer: FaultCorrelationAnalyzer,
        history_window: timedelta = timedelta(hours=1),
        update_interval: float = 5.0,
        port: int = 8050
    ):
        self.analyzer = analyzer
        self.visualizer = FaultVisualizer(analyzer)
        self.history_window = history_window
        self.update_interval = update_interval
        self.port = port
        
        # Initialize dashboard
        self.app = dash.Dash(__name__)
        self.setup_layout()
        
        # Real-time data
        self.recent_faults = deque(maxlen=1000)
        self.active_faults: Dict[str, FaultEvent] = {}
        
        # Socket.IO for real-time updates
        self.sio = socketio.AsyncServer(async_mode='aiohttp')
        self.sio_app = web.Application()
        self.sio.attach(self.sio_app)
        self.setup_socketio()
    
    def setup_layout(self):
        """Setup dashboard layout."""
        self.app.layout = html.Div([
            html.H1("Real-time Fault Monitor"),
            
            # Controls
            html.Div([
                html.Button(
                    "Pause Updates",
                    id="pause-button",
                    n_clicks=0
                ),
                dcc.Dropdown(
                    id="view-selector",
                    options=[
                        {"label": "Timeline", "value": "timeline"},
                        {"label": "Correlation", "value": "correlation"},
                        {"label": "Cascade", "value": "cascade"},
                        {"label": "Impact", "value": "impact"},
                        {"label": "Pattern", "value": "pattern"}
                    ],
                    value="timeline"
                ),
                dcc.Dropdown(
                    id="fault-filter",
                    multi=True,
                    placeholder="Filter fault types..."
                )
            ], style={"margin": "10px"}),
            
            # Main visualization
            dcc.Loading(
                id="loading",
                children=[
                    dcc.Graph(id="main-graph")
                ]
            ),
            
            # Active faults table
            html.Div([
                html.H3("Active Faults"),
                html.Table(
                    id="active-faults-table",
                    children=[
                        html.Thead(html.Tr([
                            html.Th("Type"),
                            html.Th("Start Time"),
                            html.Th("Duration"),
                            html.Th("Impact"),
                            html.Th("Related Faults")
                        ]))
                    ]
                )
            ]),
            
            # Metrics summary
            html.Div([
                html.H3("Metrics Summary"),
                dcc.Graph(id="metrics-summary")
            ]),
            
            # Hidden data store
            dcc.Store(id="fault-data"),
            
            # Update interval
            dcc.Interval(
                id="interval-component",
                interval=self.update_interval * 1000,  # Convert to milliseconds
                n_intervals=0
            )
        ])
        
        self.setup_callbacks()
    
    def setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            Output("main-graph", "figure"),
            [
                Input("view-selector", "value"),
                Input("fault-filter", "value"),
                Input("fault-data", "data")
            ]
        )
        def update_graph(view, fault_types, data):
            if not data:
                return go.Figure()
            
            # Filter faults if needed
            if fault_types:
                filtered_faults = [
                    f for f in self.analyzer.fault_history
                    if f.fault_type in fault_types
                ]
                self.analyzer.fault_history = filtered_faults
            
            # Get appropriate visualization
            if view == "timeline":
                return self.visualizer.create_fault_timeline()
            elif view == "correlation":
                return self.visualizer.create_correlation_heatmap()
            elif view == "cascade":
                return self.visualizer.create_cascade_graph()
            elif view == "impact":
                return self.visualizer.create_impact_analysis()
            elif view == "pattern":
                return self.visualizer.create_pattern_sankey()
        
        @self.app.callback(
            Output("active-faults-table", "children"),
            Input("fault-data", "data")
        )
        def update_active_faults(data):
            if not self.active_faults:
                return []
            
            rows = [
                html.Thead(html.Tr([
                    html.Th("Type"),
                    html.Th("Start Time"),
                    html.Th("Duration"),
                    html.Th("Impact"),
                    html.Th("Related Faults")
                ]))
            ]
            
            for fault in self.active_faults.values():
                rows.append(html.Tr([
                    html.Td(fault.fault_type),
                    html.Td(fault.timestamp.strftime("%H:%M:%S")),
                    html.Td(str(fault.duration)),
                    html.Td(str(fault.impact_metrics)),
                    html.Td(", ".join(fault.related_faults))
                ]))
            
            return rows
        
        @self.app.callback(
            Output("metrics-summary", "figure"),
            Input("fault-data", "data")
        )
        def update_metrics_summary(data):
            if not self.analyzer.fault_history:
                return go.Figure()
            
            # Calculate metrics
            metrics = {}
            for event in self.analyzer.fault_history:
                for metric, value in event.impact_metrics.items():
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(value)
            
            # Create summary figure
            fig = go.Figure()
            
            for metric, values in metrics.items():
                fig.add_trace(go.Box(
                    y=values,
                    name=metric
                ))
            
            fig.update_layout(
                title="Impact Metrics Distribution",
                yaxis_title="Value",
                showlegend=True
            )
            
            return fig
    
    def setup_socketio(self):
        """Setup Socket.IO event handlers."""
        @self.sio.on('connect')
        async def handle_connect(sid, environ):
            logger.info(f"Client connected: {sid}")
        
        @self.sio.on('fault_event')
        async def handle_fault_event(sid, data):
            event = FaultEvent(**data)
            self.add_fault_event(event)
            
            # Emit update to all clients
            await self.sio.emit(
                'fault_update',
                {
                    'event': asdict(event),
                    'active_faults': len(self.active_faults)
                }
            )
    
    def add_fault_event(self, event: FaultEvent):
        """Add new fault event."""
        self.recent_faults.append(event)
        self.analyzer.add_fault_event(event)
        
        # Update active faults
        self.active_faults[event.fault_type] = event
        
        # Remove expired faults
        current_time = datetime.now()
        expired = [
            fault_type
            for fault_type, fault in self.active_faults.items()
            if current_time - fault.timestamp > fault.duration
        ]
        for fault_type in expired:
            del self.active_faults[fault_type]
    
    async def start_dashboard(self):
        """Start the dashboard."""
        # Start Dash app in a separate thread
        threading.Thread(
            target=self.app.run_server,
            kwargs={"port": self.port, "debug": False},
            daemon=True
        ).start()
        
        # Start Socket.IO server
        runner = web.AppRunner(self.sio_app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port + 1)
        await site.start()
        
        logger.info(
            f"Dashboard running at http://localhost:{self.port}\n"
            f"Socket.IO server running at http://localhost:{self.port + 1}"
        )
        
        try:
            while True:
                # Prune old events
                cutoff = datetime.now() - self.history_window
                self.analyzer.fault_history = [
                    event for event in self.analyzer.fault_history
                    if event.timestamp > cutoff
                ]
                
                # Update visualizations
                await asyncio.sleep(self.update_interval)
        finally:
            await runner.cleanup()

async def run_dashboard(
    analyzer: Optional[FaultCorrelationAnalyzer] = None
):
    """Run the real-time dashboard."""
    if analyzer is None:
        analyzer = FaultCorrelationAnalyzer()
    
    dashboard = RealtimeFaultDashboard(analyzer)
    await dashboard.start_dashboard()

if __name__ == "__main__":
    asyncio.run(run_dashboard())
