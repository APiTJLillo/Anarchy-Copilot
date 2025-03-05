"""Visualization components for monitoring dashboard."""

import dash
from dash import html, dcc, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import json
import logging
from collections import defaultdict

from .alerts import Alert, AlertSeverity, AlertManager
from .metrics import MetricValue, TimeseriesMetric
from .storage import MetricStore, TimeseriesStore

logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """Interactive monitoring dashboard."""
    
    def __init__(
        self,
        alert_manager: AlertManager,
        metric_store: MetricStore,
        timeseries_store: TimeseriesStore,
        update_interval: int = 10
    ):
        self.alert_manager = alert_manager
        self.metric_store = metric_store
        self.timeseries_store = timeseries_store
        self.update_interval = update_interval
        
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Create dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Performance Monitoring Dashboard"),
                html.Div([
                    html.Button("Refresh", id="refresh-button"),
                    dcc.Interval(
                        id="auto-refresh",
                        interval=self.update_interval * 1000
                    )
                ])
            ], style={"margin": "20px"}),
            
            # Alert Summary
            html.Div([
                html.H2("Active Alerts"),
                html.Div([
                    dcc.Graph(id="alert-severity-chart"),
                    html.Div(id="alert-list")
                ], style={"display": "flex"})
            ], style={"margin": "20px"}),
            
            # Metric Overview
            html.Div([
                html.H2("Metric Overview"),
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id="time-range",
                        options=[
                            {"label": "Last Hour", "value": "1h"},
                            {"label": "Last 6 Hours", "value": "6h"},
                            {"label": "Last 24 Hours", "value": "24h"},
                            {"label": "Last 7 Days", "value": "7d"}
                        ],
                        value="1h"
                    )
                ]),
                dcc.Graph(id="metric-timeline")
            ], style={"margin": "20px"}),
            
            # Alert Details
            html.Div([
                html.H2("Alert Details"),
                dcc.Tabs([
                    dcc.Tab(label="Active Alerts", children=[
                        self._create_alert_table("active-alerts")
                    ]),
                    dcc.Tab(label="Recent Alerts", children=[
                        self._create_alert_table("recent-alerts")
                    ])
                ])
            ], style={"margin": "20px"}),
            
            # Alert Actions
            html.Div([
                html.H3("Alert Actions"),
                html.Div([
                    dcc.Input(
                        id="alert-id",
                        placeholder="Enter Alert ID",
                        style={"margin": "10px"}
                    ),
                    dcc.Input(
                        id="user",
                        placeholder="Enter Username",
                        style={"margin": "10px"}
                    ),
                    html.Button(
                        "Acknowledge",
                        id="acknowledge-button",
                        style={"margin": "10px"}
                    ),
                    html.Button(
                        "Resolve",
                        id="resolve-button",
                        style={"margin": "10px"}
                    )
                ])
            ], style={"margin": "20px"})
        ])
    
    def _create_alert_table(self, table_id: str) -> html.Table:
        """Create alert table component."""
        return html.Table([
            html.Thead(
                html.Tr([
                    html.Th("ID"),
                    html.Th("Rule"),
                    html.Th("Metric"),
                    html.Th("Value"),
                    html.Th("Threshold"),
                    html.Th("Severity"),
                    html.Th("Time"),
                    html.Th("Status")
                ])
            ),
            html.Tbody(id=table_id)
        ], style={"width": "100%"})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output("alert-severity-chart", "figure"),
             Output("alert-list", "children"),
             Output("metric-timeline", "figure"),
             Output("active-alerts", "children"),
             Output("recent-alerts", "children")],
            [Input("refresh-button", "n_clicks"),
             Input("auto-refresh", "n_intervals"),
             Input("time-range", "value")]
        )
        def update_dashboard(n_clicks, n_intervals, time_range):
            """Update dashboard components."""
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Create severity chart
            severity_counts = defaultdict(int)
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            severity_fig = go.Figure(data=[
                go.Bar(
                    x=list(severity_counts.keys()),
                    y=list(severity_counts.values()),
                    marker_color=[
                        "#36a64f" if s == "info" else
                        "#ffcc00" if s == "warning" else
                        "#ff0000" if s == "error" else
                        "#7b0000"
                        for s in severity_counts.keys()
                    ]
                )
            ])
            severity_fig.update_layout(
                title="Alert Severity Distribution",
                xaxis_title="Severity",
                yaxis_title="Count"
            )
            
            # Create alert list
            alert_list = html.Ul([
                html.Li(f"{alert.rule_name}: {alert.description}")
                for alert in sorted(
                    active_alerts,
                    key=lambda x: x.severity.value,
                    reverse=True
                )[:5]  # Show top 5 most severe
            ])
            
            # Create metric timeline
            time_ranges = {
                "1h": timedelta(hours=1),
                "6h": timedelta(hours=6),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7)
            }
            end_time = datetime.now()
            start_time = end_time - time_ranges[time_range]
            
            metrics_data = []
            alert_times = []
            
            # Get metrics that triggered alerts
            for alert in active_alerts:
                series = await self.timeseries_store.get_timeseries(
                    alert.metric_name,
                    start_time,
                    end_time
                )
                metrics_data.append({
                    "name": alert.metric_name,
                    "values": series.values,
                    "timestamps": series.timestamps
                })
                alert_times.append(alert.timestamp)
            
            timeline_fig = go.Figure()
            
            for metric in metrics_data:
                timeline_fig.add_trace(
                    go.Scatter(
                        x=metric["timestamps"],
                        y=metric["values"],
                        name=metric["name"],
                        mode="lines"
                    )
                )
            
            # Add alert markers
            timeline_fig.add_trace(
                go.Scatter(
                    x=alert_times,
                    y=[0] * len(alert_times),
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color="red"
                    ),
                    name="Alerts"
                )
            )
            
            timeline_fig.update_layout(
                title="Metric Timeline with Alerts",
                xaxis_title="Time",
                yaxis_title="Value"
            )
            
            # Create alert tables
            active_rows = [
                html.Tr([
                    html.Td(alert.id),
                    html.Td(alert.rule_name),
                    html.Td(alert.metric_name),
                    html.Td(f"{alert.value:.2f}"),
                    html.Td(f"{alert.threshold:.2f}"),
                    html.Td(alert.severity.value),
                    html.Td(alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
                    html.Td("Acknowledged" if alert.acknowledged else "Active")
                ])
                for alert in active_alerts
            ]
            
            recent_alerts = self.alert_manager.get_resolved_alerts(
                since=end_time - timedelta(hours=24)
            )
            recent_rows = [
                html.Tr([
                    html.Td(alert.id),
                    html.Td(alert.rule_name),
                    html.Td(alert.metric_name),
                    html.Td(f"{alert.value:.2f}"),
                    html.Td(f"{alert.threshold:.2f}"),
                    html.Td(alert.severity.value),
                    html.Td(alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")),
                    html.Td("Resolved")
                ])
                for alert in recent_alerts
            ]
            
            return (
                severity_fig,
                alert_list,
                timeline_fig,
                active_rows,
                recent_rows
            )
        
        @self.app.callback(
            Output("alert-id", "value"),
            [Input("acknowledge-button", "n_clicks"),
             Input("resolve-button", "n_clicks")],
            [State("alert-id", "value"),
             State("user", "value")]
        )
        async def handle_alert_action(
            ack_clicks,
            res_clicks,
            alert_id,
            user
        ):
            """Handle alert acknowledgment and resolution."""
            if not alert_id:
                return ""
            
            ctx = dash.callback_context
            if not ctx.triggered:
                return alert_id
            
            button_id = ctx.triggered[0]["prop_id"].split(".")[0]
            
            try:
                if button_id == "acknowledge-button" and user:
                    await self.alert_manager.acknowledge_alert(
                        alert_id,
                        user
                    )
                elif button_id == "resolve-button":
                    await self.alert_manager.resolve_alert(alert_id)
                
                return ""  # Clear input on success
                
            except Exception as e:
                logger.error(f"Error handling alert action: {e}")
                return alert_id  # Keep ID on error
    
    def run(self, host: str = "localhost", port: int = 8050, debug: bool = False):
        """Run the dashboard server."""
        self.app.run_server(host=host, port=port, debug=debug)
