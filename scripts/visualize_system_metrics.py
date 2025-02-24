#!/usr/bin/env python3
"""System metrics visualization for performance tests."""
import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsVisualizer:
    """Generate visualizations for system metrics."""
    
    def __init__(self, output_path: Path):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_cpu_metrics(self, path: Path) -> pd.DataFrame:
        """Load CPU metrics from SAR CSV."""
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def load_memory_metrics(self, path: Path) -> pd.DataFrame:
        """Load memory metrics from SAR CSV."""
        df = pd.read_csv(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def create_visualizations(self, cpu_df: pd.DataFrame, memory_df: pd.DataFrame) -> str:
        """Create performance visualization report."""
        # Create subplot figure
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'CPU Usage Over Time',
                'Memory Usage Over Time',
                'CPU Usage Distribution',
                'Memory Usage Distribution',
                'CPU Usage Heatmap',
                'System Load Timeline'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "histogram"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ]
        )
        
        # CPU Usage Timeline
        fig.add_trace(
            go.Scatter(
                x=cpu_df['timestamp'],
                y=cpu_df['%user'],
                name='User CPU',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=cpu_df['timestamp'],
                y=cpu_df['%system'],
                name='System CPU',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Memory Usage Timeline
        fig.add_trace(
            go.Scatter(
                x=memory_df['timestamp'],
                y=memory_df['kbmemused'] / 1024,  # Convert to MB
                name='Memory Used (MB)',
                line=dict(color='green')
            ),
            row=1, col=2
        )
        
        # CPU Usage Distribution
        fig.add_trace(
            go.Histogram(
                x=cpu_df['%user'] + cpu_df['%system'],
                name='Total CPU Usage',
                nbinsx=30
            ),
            row=2, col=1
        )
        
        # Memory Usage Distribution
        fig.add_trace(
            go.Histogram(
                x=memory_df['kbmemused'] / 1024,
                name='Memory Usage (MB)',
                nbinsx=30
            ),
            row=2, col=2
        )
        
        # CPU Usage Heatmap
        cpu_matrix = self._create_cpu_heatmap(cpu_df)
        fig.add_trace(
            go.Heatmap(
                z=cpu_matrix,
                colorscale='Viridis',
                name='CPU Usage Pattern'
            ),
            row=3, col=1
        )
        
        # System Load Timeline
        fig.add_trace(
            go.Scatter(
                x=cpu_df['timestamp'],
                y=cpu_df['%iowait'],
                name='IO Wait',
                line=dict(color='purple')
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="System Performance Metrics",
            showlegend=True
        )
        
        # Add annotations for performance insights
        self._add_performance_insights(fig, cpu_df, memory_df)
        
        # Save HTML report
        fig.write_html(self.output_path)
        
        # Generate summary statistics
        summary = self._generate_summary(cpu_df, memory_df)
        summary_path = self.output_path.parent / "metrics_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        
        return str(self.output_path)
    
    def _create_cpu_heatmap(self, cpu_df: pd.DataFrame) -> np.ndarray:
        """Create CPU usage pattern heatmap data."""
        # Create time-of-day vs. CPU usage matrix
        hours = cpu_df['timestamp'].dt.hour
        cpu_usage = cpu_df['%user'] + cpu_df['%system']
        
        matrix = np.zeros((24, 10))  # 24 hours x 10 CPU usage ranges
        for h, c in zip(hours, cpu_usage):
            usage_bin = min(int(c / 10), 9)
            matrix[h, usage_bin] += 1
        
        return matrix
    
    def _add_performance_insights(self, 
                                fig: go.Figure,
                                cpu_df: pd.DataFrame,
                                memory_df: pd.DataFrame):
        """Add performance insights annotations."""
        insights = []
        
        # CPU insights
        cpu_total = cpu_df['%user'] + cpu_df['%system']
        if cpu_total.max() > 90:
            insights.append(
                f"High CPU usage detected: {cpu_total.max():.1f}% peak"
            )
        
        # Memory insights
        mem_used_gb = memory_df['kbmemused'].max() / (1024 * 1024)
        if mem_used_gb > 1:
            insights.append(
                f"High memory usage: {mem_used_gb:.2f}GB peak"
            )
        
        # IO insights
        if cpu_df['%iowait'].max() > 10:
            insights.append(
                f"High IO wait detected: {cpu_df['%iowait'].max():.1f}% peak"
            )
        
        # Add insights to plot
        for i, insight in enumerate(insights):
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=1.0,
                y=1.0 - (i * 0.05),
                text=insight,
                showarrow=False,
                font=dict(size=12, color="red")
            )
    
    def _generate_summary(self, cpu_df: pd.DataFrame, 
                         memory_df: pd.DataFrame) -> str:
        """Generate performance metrics summary."""
        cpu_total = cpu_df['%user'] + cpu_df['%system']
        mem_used_gb = memory_df['kbmemused'] / (1024 * 1024)
        
        summary = [
            "System Performance Summary",
            "========================",
            "",
            "CPU Usage:",
            f"  Average: {cpu_total.mean():.1f}%",
            f"  Peak: {cpu_total.max():.1f}%",
            f"  95th percentile: {cpu_total.quantile(0.95):.1f}%",
            "",
            "Memory Usage:",
            f"  Average: {mem_used_gb.mean():.2f}GB",
            f"  Peak: {mem_used_gb.max():.2f}GB",
            f"  95th percentile: {mem_used_gb.quantile(0.95):.2f}GB",
            "",
            "IO Wait:",
            f"  Average: {cpu_df['%iowait'].mean():.1f}%",
            f"  Peak: {cpu_df['%iowait'].max():.1f}%",
            "",
            "Performance Insights:",
        ]
        
        # Add insights
        if cpu_total.max() > 90:
            summary.append("  ⚠️ CPU usage exceeded 90%")
        if mem_used_gb.max() > 1:
            summary.append(f"  ⚠️ Memory usage peaked at {mem_used_gb.max():.2f}GB")
        if cpu_df['%iowait'].max() > 10:
            summary.append("  ⚠️ High IO wait times detected")
        
        return "\n".join(summary)

def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Visualize system metrics")
    parser.add_argument("--cpu", type=Path, required=True, help="CPU metrics CSV")
    parser.add_argument("--memory", type=Path, required=True, help="Memory metrics CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path")
    
    args = parser.parse_args()
    
    try:
        visualizer = MetricsVisualizer(args.output)
        
        # Load data
        cpu_df = visualizer.load_cpu_metrics(args.cpu)
        memory_df = visualizer.load_memory_metrics(args.memory)
        
        # Create visualizations
        output_path = visualizer.create_visualizations(cpu_df, memory_df)
        logger.info("Generated visualization report: %s", output_path)
        
    except Exception as e:
        logger.error("Failed to generate visualizations: %s", e)
        raise

if __name__ == "__main__":
    main()
