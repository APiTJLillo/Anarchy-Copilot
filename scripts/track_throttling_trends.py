#!/usr/bin/env python3
"""Track and analyze performance trends for alert throttling system."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dataclasses import dataclass

@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics."""
    timestamp: datetime
    commit: str
    throughput: float
    memory_usage: float
    storage_size: float
    cleanup_time: float
    alerts_per_second: float

class TrendAnalyzer:
    """Analyze performance trends over time."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: List[PerformanceSnapshot] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load historical performance data."""
        if self.history_file.exists():
            try:
                data = json.loads(self.history_file.read_text())
                for entry in data:
                    self.history.append(PerformanceSnapshot(
                        timestamp=datetime.fromisoformat(entry["timestamp"]),
                        commit=entry["commit"],
                        throughput=entry["throughput"],
                        memory_usage=entry["memory_usage"],
                        storage_size=entry["storage_size"],
                        cleanup_time=entry["cleanup_time"],
                        alerts_per_second=entry["alerts_per_second"]
                    ))
            except Exception as e:
                print(f"Error loading history: {e}", file=sys.stderr)

    def _save_history(self) -> None:
        """Save historical performance data."""
        try:
            data = [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "commit": snapshot.commit,
                    "throughput": snapshot.throughput,
                    "memory_usage": snapshot.memory_usage,
                    "storage_size": snapshot.storage_size,
                    "cleanup_time": snapshot.cleanup_time,
                    "alerts_per_second": snapshot.alerts_per_second
                }
                for snapshot in self.history
            ]
            self.history_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error saving history: {e}", file=sys.stderr)

    def add_snapshot(self, results: Dict[str, Any], commit: str) -> None:
        """Add new performance snapshot."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            commit=commit,
            throughput=results['concurrent_performance']['alerts_per_second'],
            memory_usage=results['concurrent_performance']['total_memory_mb'],
            storage_size=results['storage_performance']['file_size_kb'],
            cleanup_time=results['cleanup_performance']['avg_cleanup_ms'],
            alerts_per_second=results['throttling_performance']['1000']['alerts_per_second']
        )
        
        self.history.append(snapshot)
        self._save_history()

    def calculate_trends(self, days: int = 30) -> Dict[str, Any]:
        """Calculate performance trends over specified period."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [s for s in self.history if s.timestamp >= cutoff]
        
        if len(recent) < 2:
            return {}
        
        def calc_change(values: List[float]) -> Tuple[float, float]:
            if not values:
                return 0.0, 0.0
            baseline = sum(values[:3]) / min(3, len(values))  # Average of first 3 points
            current = sum(values[-3:]) / min(3, len(values))  # Average of last 3 points
            return ((current - baseline) / baseline * 100), current

        trends = {}
        for metric in ['throughput', 'memory_usage', 'storage_size', 'cleanup_time', 'alerts_per_second']:
            values = [getattr(s, metric) for s in recent]
            change, current = calc_change(values)
            trends[metric] = {
                'change': change,
                'current': current,
                'trend': 'improving' if change > 0 else 'degrading'
            }
        
        return trends

    def create_trend_visualization(self) -> go.Figure:
        """Create visualization of performance trends."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Throughput Trend",
                "Memory Usage Trend",
                "Storage Size Trend",
                "Cleanup Time Trend",
                "Alerts/Second Trend"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Helper function to add trend line
        def add_trendline(values: List[float]) -> List[float]:
            if len(values) < 2:
                return values
            z = np.polyfit(range(len(values)), values, 1)
            return np.poly1d(z)(range(len(values)))
        
        # Plot metrics
        metrics = {
            'throughput': (1, 1),
            'memory_usage': (1, 2),
            'storage_size': (2, 1),
            'cleanup_time': (2, 2),
            'alerts_per_second': (3, 1)
        }
        
        dates = [s.timestamp for s in self.history]
        
        for metric, (row, col) in metrics.items():
            values = [getattr(s, metric) for s in self.history]
            
            # Main line
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color='#3498db')
                ),
                row=row, col=col
            )
            
            # Trend line
            if len(values) >= 2:
                trend = add_trendline(values)
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=trend,
                        mode='lines',
                        name=f'{metric} trend',
                        line=dict(
                            color='#e74c3c',
                            dash='dash'
                        ),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Performance Trends Over Time"
        )
        
        return fig

    def create_html_report(self, output_path: Path) -> None:
        """Generate HTML trend report."""
        trends = self.calculate_trends()
        fig = self.create_trend_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Alert Throttling Trend Analysis</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .summary {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .trend {{
                    padding: 4px 8px;
                    border-radius: 4px;
                    color: white;
                    font-weight: bold;
                    display: inline-block;
                }}
                .improving {{ background-color: #2ecc71; }}
                .degrading {{ background-color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Performance Trend Analysis</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>30-Day Trends</h2>
                {"" if trends else "<p>Insufficient data for trend analysis</p>"}
        """
        
        if trends:
            for metric, data in trends.items():
                html += f"""
                <p>
                    {metric.replace('_', ' ').title()}: 
                    <span class="trend {data['trend']}">
                        {data['change']:+.1f}%
                    </span>
                    (Current: {data['current']:.2f})
                </p>
                """
        
        html += f"""
            </div>

            <div class="chart-container">
                {fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html)

def main() -> int:
    """Main entry point."""
    try:
        results_dir = Path("benchmark_results")
        if not results_dir.exists():
            print("No benchmark results directory found.")
            return 1
        
        results_file = results_dir / "type_suggestion_benchmarks.json"
        if not results_file.exists():
            print("No benchmark results found.")
            return 1
        
        # Load current results
        results = json.loads(results_file.read_text())
        
        # Get current commit
        commit = os.getenv("GITHUB_SHA", "local")
        
        # Initialize trend analyzer
        analyzer = TrendAnalyzer(results_dir / "performance_history.json")
        
        # Add new snapshot
        analyzer.add_snapshot(results, commit)
        
        # Generate trend report
        output_path = results_dir / "performance_trends.html"
        analyzer.create_html_report(output_path)
        
        print(f"\nTrend analysis generated at: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error analyzing trends: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
