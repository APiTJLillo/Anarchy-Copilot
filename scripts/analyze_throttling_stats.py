#!/usr/bin/env python3
"""Statistical analysis for alert throttling performance."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

@dataclass
class MetricStats:
    """Statistical analysis results for a metric."""
    mean: float
    median: float
    std_dev: float
    trend: float  # Slope of linear regression
    p_value: float
    confidence_interval: Tuple[float, float]
    z_score_latest: float
    is_anomaly: bool

class StatisticalAnalyzer:
    """Analyze performance data using statistical methods."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.data = self._load_data()
        self.metrics = [
            'throughput', 'memory_usage', 'storage_size',
            'cleanup_time', 'alerts_per_second'
        ]

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare historical data."""
        if not self.history_file.exists():
            return pd.DataFrame()
        
        try:
            data = json.loads(self.history_file.read_text())
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
            return pd.DataFrame()

    def analyze_metric(self, metric: str, window_days: int = 30) -> MetricStats:
        """Perform statistical analysis on a metric."""
        if self.data.empty:
            return None
        
        # Get recent data
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = self.data[self.data['timestamp'] >= cutoff][metric].values
        
        if len(recent) < 2:
            return None
        
        # Basic statistics
        mean = np.mean(recent)
        median = np.median(recent)
        std_dev = np.std(recent)
        
        # Linear regression for trend
        X = np.arange(len(recent)).reshape(-1, 1)
        y = recent.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, y)
        trend = model.coef_[0][0]
        
        # Statistical significance
        _, p_value = stats.ttest_1samp(recent, mean)
        
        # Confidence interval
        conf_int = stats.t.interval(
            alpha=0.95,
            df=len(recent)-1,
            loc=mean,
            scale=stats.sem(recent)
        )
        
        # Z-score of latest value
        latest = recent[-1]
        z_score = (latest - mean) / std_dev if std_dev > 0 else 0
        
        # Anomaly detection
        is_anomaly = abs(z_score) > 2
        
        return MetricStats(
            mean=mean,
            median=median,
            std_dev=std_dev,
            trend=trend,
            p_value=p_value,
            confidence_interval=conf_int,
            z_score_latest=z_score,
            is_anomaly=is_anomaly
        )

    def detect_changepoints(self, metric: str) -> List[datetime]:
        """Detect significant changes in metric values."""
        if self.data.empty:
            return []
        
        values = self.data[metric].values
        if len(values) < 10:
            return []
        
        # Use CUSUM algorithm for change detection
        def cusum(data: np.ndarray, threshold: float = 1.0) -> List[int]:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return []
            
            z_scores = (data - mean) / std
            cusums = np.cumsum(z_scores)
            
            changes = []
            max_cusum = 0
            max_index = 0
            
            for i, cusum in enumerate(cusums):
                if abs(cusum) > max_cusum:
                    max_cusum = abs(cusum)
                    max_index = i
                
                if max_cusum > threshold:
                    changes.append(max_index)
                    max_cusum = 0
            
            return changes
        
        change_indices = cusum(values)
        return [self.data.iloc[i]['timestamp'] for i in change_indices]

    def create_statistical_visualization(self) -> go.Figure:
        """Create visualization of statistical analysis."""
        fig = make_subplots(
            rows=len(self.metrics), cols=2,
            subplot_titles=[
                f"{m.replace('_', ' ').title()} Distribution" for m in self.metrics
            ] * 2,
            vertical_spacing=0.05,
            horizontal_spacing=0.1
        )
        
        for i, metric in enumerate(self.metrics, 1):
            if metric not in self.data.columns:
                continue
            
            values = self.data[metric].values
            if len(values) < 2:
                continue
            
            # Histogram
            fig.add_trace(
                go.Histogram(
                    x=values,
                    name=f"{metric} dist",
                    nbinsx=20,
                    showlegend=False
                ),
                row=i, col=1
            )
            
            # QQ plot
            theoretical_q = stats.norm.ppf(
                np.linspace(0.01, 0.99, len(values))
            )
            observed_q = np.sort(values)
            
            fig.add_trace(
                go.Scatter(
                    x=theoretical_q,
                    y=observed_q,
                    mode='markers',
                    name=f"{metric} QQ",
                    showlegend=False
                ),
                row=i, col=2
            )
            
            # Add reference line
            min_val = min(theoretical_q)
            max_val = max(theoretical_q)
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=i, col=2
            )
        
        fig.update_layout(
            height=300 * len(self.metrics),
            title_text="Statistical Analysis of Performance Metrics"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate statistical analysis report."""
        stats_by_metric = {
            metric: self.analyze_metric(metric)
            for metric in self.metrics
        }
        
        changepoints = {
            metric: self.detect_changepoints(metric)
            for metric in self.metrics
        }
        
        fig = self.create_statistical_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Statistical Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .stats-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-stats {{
                    margin: 20px 0;
                    padding: 10px;
                    border-left: 4px solid #3498db;
                }}
                .anomaly {{
                    background: #fff3cd;
                    border-color: #ffeeba;
                }}
                .trend {{
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                .trend-up {{ color: #28a745; }}
                .trend-down {{ color: #dc3545; }}
            </style>
        </head>
        <body>
            <h1>Statistical Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="stats-container">
        """
        
        for metric, stats in stats_by_metric.items():
            if not stats:
                continue
            
            trend_class = "trend-up" if stats.trend > 0 else "trend-down"
            anomaly_class = " anomaly" if stats.is_anomaly else ""
            
            html += f"""
                <div class="metric-stats{anomaly_class}">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>
                        Trend: <span class="trend {trend_class}">{stats.trend:+.2f}</span>
                        (p = {stats.p_value:.3f})
                    </p>
                    <ul>
                        <li>Mean: {stats.mean:.2f} ± {stats.std_dev:.2f}</li>
                        <li>Median: {stats.median:.2f}</li>
                        <li>95% CI: [{stats.confidence_interval[0]:.2f}, {stats.confidence_interval[1]:.2f}]</li>
                        <li>Latest Z-score: {stats.z_score_latest:.2f}</li>
                    </ul>
            """
            
            if changepoints[metric]:
                html += "<p>Significant changes detected at:</p><ul>"
                for cp in changepoints[metric]:
                    html += f"<li>{cp.strftime('%Y-%m-%d %H:%M:%S')}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += f"""
            </div>

            <div class="stats-container">
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
        
        history_file = results_dir / "performance_history.json"
        if not history_file.exists():
            print("No performance history found.")
            return 1
        
        analyzer = StatisticalAnalyzer(history_file)
        output_path = results_dir / "statistical_analysis.html"
        analyzer.generate_report(output_path)
        
        print(f"\nStatistical analysis report generated at: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error performing statistical analysis: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
