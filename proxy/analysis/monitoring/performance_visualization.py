"""Performance visualization and analysis tools."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
from pathlib import Path
import logging
from dataclasses import asdict
from scipy import stats

from .test_performance_regression import (
    PerformanceMetrics,
    PerformanceHistory,
    PerformanceBudget
)

logger = logging.getLogger(__name__)

class PerformanceVisualizer:
    """Visualize performance metrics and trends."""
    
    def __init__(self, history: PerformanceHistory):
        self.history = history
        self.output_dir = Path("performance_visualizations")
        self.output_dir.mkdir(exist_ok=True)
    
    def create_trend_dashboard(self) -> go.Figure:
        """Create main performance trend dashboard."""
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Render Time Trends",
                "Memory Usage Trends",
                "CPU Usage Trends",
                "Frame Rate Trends",
                "DOM Metrics",
                "Layout Performance"
            )
        )
        
        # Add trend plots
        for test_name, metrics in self.history.history.items():
            timestamps = [
                datetime.fromisoformat(m["timestamp"])
                for m in metrics
            ]
            
            # Render time trends
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["render_time"] for m in metrics],
                    name=f"{test_name} (render)",
                    line={"dash": "solid"}
                ),
                row=1, col=1
            )
            
            # Memory usage trends
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["memory_usage"] / 1024 / 1024 for m in metrics],
                    name=f"{test_name} (memory)",
                    line={"dash": "solid"}
                ),
                row=1, col=2
            )
            
            # CPU usage trends
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["cpu_usage"] for m in metrics],
                    name=f"{test_name} (cpu)",
                    line={"dash": "solid"}
                ),
                row=2, col=1
            )
            
            # Frame rate trends
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["frame_rate"] for m in metrics],
                    name=f"{test_name} (fps)",
                    line={"dash": "solid"}
                ),
                row=2, col=2
            )
            
            # DOM metrics
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["dom_nodes"] for m in metrics],
                    name=f"{test_name} (nodes)",
                    line={"dash": "solid"}
                ),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["event_listeners"] for m in metrics],
                    name=f"{test_name} (listeners)",
                    line={"dash": "dotted"}
                ),
                row=3, col=1
            )
            
            # Layout performance
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["reflow_count"] for m in metrics],
                    name=f"{test_name} (reflows)",
                    line={"dash": "solid"}
                ),
                row=3, col=2
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[m["repaint_count"] for m in metrics],
                    name=f"{test_name} (repaints)",
                    line={"dash": "dotted"}
                ),
                row=3, col=2
            )
        
        # Add performance budgets
        fig.add_hline(
            y=PerformanceBudget.MAX_RENDER_TIME,
            line=dict(color="red", dash="dash"),
            row=1, col=1
        )
        fig.add_hline(
            y=PerformanceBudget.MAX_MEMORY_USAGE / 1024 / 1024,
            line=dict(color="red", dash="dash"),
            row=1, col=2
        )
        fig.add_hline(
            y=PerformanceBudget.MAX_CPU_USAGE,
            line=dict(color="red", dash="dash"),
            row=2, col=1
        )
        fig.add_hline(
            y=PerformanceBudget.MIN_FRAME_RATE,
            line=dict(color="red", dash="dash"),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title="Performance Trends Dashboard"
        )
        
        return fig
    
    def create_regression_heatmap(self) -> go.Figure:
        """Create heatmap of performance regressions."""
        # Collect regression data
        tests = list(self.history.history.keys())
        metrics = [
            "render_time",
            "memory_usage",
            "cpu_usage",
            "frame_rate",
            "dom_nodes",
            "event_listeners",
            "reflow_count",
            "repaint_count"
        ]
        
        regression_matrix = np.zeros((len(tests), len(metrics)))
        
        for i, test in enumerate(tests):
            if len(self.history.history[test]) < 2:
                continue
                
            recent = self.history.history[test][-5:]  # Last 5 runs
            baseline = self.history.history[test][:-5]  # Earlier runs
            
            for j, metric in enumerate(metrics):
                recent_mean = np.mean([r[metric] for r in recent])
                baseline_mean = np.mean([b[metric] for b in baseline])
                
                if baseline_mean > 0:
                    change = (recent_mean - baseline_mean) / baseline_mean
                    regression_matrix[i, j] = change
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=regression_matrix,
            x=metrics,
            y=tests,
            colorscale="RdYlGn_r",  # Red for regressions, green for improvements
            zmid=0  # Center colorscale at 0
        ))
        
        fig.update_layout(
            title="Performance Regression Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Tests"
        )
        
        return fig
    
    def create_component_comparison(self) -> go.Figure:
        """Create component performance comparison."""
        component_tests = [
            t for t in self.history.history.keys()
            if t.startswith("component_")
        ]
        
        if not component_tests:
            return None
        
        # Collect latest metrics
        components = []
        render_times = []
        memory_usages = []
        frame_rates = []
        
        for test in component_tests:
            if not self.history.history[test]:
                continue
                
            latest = self.history.history[test][-1]
            components.append(test.replace("component_", ""))
            render_times.append(latest["render_time"])
            memory_usages.append(latest["memory_usage"] / 1024 / 1024)
            frame_rates.append(latest["frame_rate"])
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=(
                "Render Time (ms)",
                "Memory Usage (MB)",
                "Frame Rate (fps)"
            )
        )
        
        fig.add_trace(
            go.Bar(x=components, y=render_times),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=components, y=memory_usages),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=components, y=frame_rates),
            row=1, col=3
        )
        
        # Add budget lines
        fig.add_hline(
            y=PerformanceBudget.MAX_RENDER_TIME,
            line=dict(color="red", dash="dash"),
            row=1, col=1
        )
        fig.add_hline(
            y=PerformanceBudget.MAX_MEMORY_USAGE / 1024 / 1024,
            line=dict(color="red", dash="dash"),
            row=1, col=2
        )
        fig.add_hline(
            y=PerformanceBudget.MIN_FRAME_RATE,
            line=dict(color="red", dash="dash"),
            row=1, col=3
        )
        
        fig.update_layout(
            height=400,
            title="Component Performance Comparison"
        )
        
        return fig
    
    def create_interaction_analysis(self) -> go.Figure:
        """Create interaction performance analysis."""
        interaction_tests = [
            t for t in self.history.history.keys()
            if t.startswith("interaction_")
        ]
        
        if not interaction_tests:
            return None
        
        # Collect metrics over time
        fig = go.Figure()
        
        for test in interaction_tests:
            metrics = self.history.history[test]
            timestamps = [
                datetime.fromisoformat(m["timestamp"])
                for m in metrics
            ]
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=[m["render_time"] for m in metrics],
                name=test.replace("interaction_", ""),
                mode="lines+markers"
            ))
        
        fig.add_hline(
            y=PerformanceBudget.MAX_INTERACTION_TIME,
            line=dict(color="red", dash="dash")
        )
        
        fig.update_layout(
            title="Interaction Performance Over Time",
            xaxis_title="Timestamp",
            yaxis_title="Response Time (ms)"
        )
        
        return fig
    
    def create_statistical_summary(self) -> Dict[str, Any]:
        """Create statistical summary of performance metrics."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for test_name, metrics in self.history.history.items():
            if len(metrics) < 2:
                continue
            
            test_summary = {}
            for metric in metrics[0].keys():
                if metric == "timestamp":
                    continue
                
                values = [m[metric] for m in metrics]
                
                test_summary[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "trend": stats.linregress(
                        range(len(values)),
                        values
                    ).slope
                }
            
            summary["metrics"][test_name] = test_summary
        
        return summary
    
    def save_visualizations(self):
        """Save all visualizations."""
        # Create and save trend dashboard
        trend_fig = self.create_trend_dashboard()
        trend_fig.write_html(
            str(self.output_dir / "performance_trends.html")
        )
        
        # Create and save regression heatmap
        heatmap_fig = self.create_regression_heatmap()
        heatmap_fig.write_html(
            str(self.output_dir / "regression_heatmap.html")
        )
        
        # Create and save component comparison
        comp_fig = self.create_component_comparison()
        if comp_fig:
            comp_fig.write_html(
                str(self.output_dir / "component_comparison.html")
            )
        
        # Create and save interaction analysis
        inter_fig = self.create_interaction_analysis()
        if inter_fig:
            inter_fig.write_html(
                str(self.output_dir / "interaction_analysis.html")
            )
        
        # Save statistical summary
        summary = self.create_statistical_summary()
        with open(self.output_dir / "statistical_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create index page
        self._create_index_page()
    
    def _create_index_page(self):
        """Create HTML index page linking all visualizations."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 2em; }
                .viz-link { 
                    display: block;
                    margin: 1em 0;
                    padding: 1em;
                    background: #f0f0f0;
                    border-radius: 4px;
                    text-decoration: none;
                    color: #333;
                }
                .viz-link:hover {
                    background: #e0e0e0;
                }
            </style>
        </head>
        <body>
            <h1>Performance Analysis Dashboard</h1>
            <div id="visualizations">
                <a class="viz-link" href="performance_trends.html">
                    Performance Trends Dashboard
                </a>
                <a class="viz-link" href="regression_heatmap.html">
                    Regression Analysis Heatmap
                </a>
                <a class="viz-link" href="component_comparison.html">
                    Component Performance Comparison
                </a>
                <a class="viz-link" href="interaction_analysis.html">
                    Interaction Performance Analysis
                </a>
                <a class="viz-link" href="statistical_summary.json">
                    Statistical Summary (JSON)
                </a>
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / "index.html", "w") as f:
            f.write(html)

def visualize_performance():
    """Generate performance visualizations."""
    history = PerformanceHistory(Path("performance_history.json"))
    visualizer = PerformanceVisualizer(history)
    visualizer.save_visualizations()

if __name__ == "__main__":
    visualize_performance()
