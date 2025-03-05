#!/usr/bin/env python3
"""Analyze correlations between performance alerts and metrics."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import networkx as nx
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CorrelationResult:
    """Result of correlation analysis."""
    metric_pair: Tuple[str, str]
    correlation: float
    p_value: float
    lag: int
    direction: str  # "leads" or "follows"
    strength: str  # "strong", "moderate", "weak"
    significance: bool

class AlertCorrelationAnalyzer:
    """Analyze correlations between alerts and performance metrics."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        min_correlation: float = 0.5
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.min_correlation = min_correlation
        self.correlations: List[CorrelationResult] = []

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load correlation analysis configuration."""
        default_config = {
            "analysis": {
                "max_lag": 12,  # Hours
                "min_samples": 10,
                "correlation_window": "24h",
                "significance_level": 0.05
            },
            "metrics": {
                "include_derived": True,
                "exclude_patterns": ["temp_", "debug_"]
            },
            "correlation": {
                "strength_thresholds": {
                    "strong": 0.7,
                    "moderate": 0.5,
                    "weak": 0.3
                }
            },
            "visualization": {
                "max_correlations": 20,
                "show_lag_plots": True,
                "highlight_threshold": 0.8
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _prepare_time_series(self) -> pd.DataFrame:
        """Prepare time series data for correlation analysis."""
        # Load alert history
        alert_file = self.history_dir / "alert_history.json"
        if not alert_file.exists():
            raise FileNotFoundError("Alert history not found")

        alerts = pd.DataFrame(json.loads(alert_file.read_text()))
        alerts['timestamp'] = pd.to_datetime(alerts['timestamp'])

        # Load performance metrics
        metrics = []
        for metric_file in self.history_dir.glob("**/performance_*.json"):
            data = json.loads(metric_file.read_text())
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            metrics.append(df)

        if not metrics:
            raise ValueError("No performance metrics found")

        # Merge metrics
        metrics_df = pd.concat(metrics)
        
        # Create time index
        start_time = min(alerts['timestamp'].min(), metrics_df['timestamp'].min())
        end_time = max(alerts['timestamp'].max(), metrics_df['timestamp'].max())
        timerange = pd.date_range(start_time, end_time, freq='1H')
        
        # Resample and merge data
        result = pd.DataFrame(index=timerange)
        
        # Add alert counts
        alert_counts = alerts.set_index('timestamp').resample('1H').size()
        result['alert_count'] = alert_counts

        # Add metrics
        for col in metrics_df.columns:
            if col != 'timestamp':
                if not any(p in col for p in self.config['metrics']['exclude_patterns']):
                    resampled = metrics_df.set_index('timestamp')[col].resample('1H').mean()
                    result[col] = resampled

        # Add derived metrics if configured
        if self.config['metrics']['include_derived']:
            result['alert_rate_change'] = result['alert_count'].pct_change()
            result['alert_acceleration'] = result['alert_rate_change'].pct_change()

        return result.fillna(method='ffill')

    def calculate_correlations(self) -> None:
        """Calculate correlations between metrics and alerts."""
        data = self._prepare_time_series()
        max_lag = self.config['analysis']['max_lag']
        
        # Calculate correlations for each metric pair
        metrics = [col for col in data.columns if col != 'timestamp']
        
        for i, metric1 in enumerate(metrics):
            for metric2 in metrics[i+1:]:
                best_correlation = 0
                best_lag = 0
                best_pvalue = 1.0
                
                # Test different lags
                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        shifted1 = data[metric1].shift(-lag)
                        shifted2 = data[metric2]
                    else:
                        shifted1 = data[metric1]
                        shifted2 = data[metric2].shift(lag)
                    
                    # Calculate correlation
                    mask = ~(shifted1.isna() | shifted2.isna())
                    if mask.sum() >= self.config['analysis']['min_samples']:
                        correlation, pvalue = stats.pearsonr(
                            shifted1[mask],
                            shifted2[mask]
                        )
                        
                        if abs(correlation) > abs(best_correlation):
                            best_correlation = correlation
                            best_lag = lag
                            best_pvalue = pvalue
                
                if abs(best_correlation) >= self.min_correlation:
                    # Determine correlation strength
                    thresholds = self.config['correlation']['strength_thresholds']
                    if abs(best_correlation) >= thresholds['strong']:
                        strength = "strong"
                    elif abs(best_correlation) >= thresholds['moderate']:
                        strength = "moderate"
                    else:
                        strength = "weak"
                    
                    # Determine lead/lag relationship
                    direction = "leads" if best_lag < 0 else "follows"
                    if best_lag == 0:
                        direction = "concurrent"
                    
                    self.correlations.append(CorrelationResult(
                        metric_pair=(metric1, metric2),
                        correlation=best_correlation,
                        p_value=best_pvalue,
                        lag=abs(best_lag),
                        direction=direction,
                        strength=strength,
                        significance=best_pvalue < self.config['analysis']['significance_level']
                    ))

    def build_correlation_graph(self) -> nx.Graph:
        """Build correlation graph for visualization."""
        G = nx.Graph()
        
        # Add nodes for all unique metrics
        metrics = set()
        for corr in self.correlations:
            metrics.add(corr.metric_pair[0])
            metrics.add(corr.metric_pair[1])
        
        for metric in metrics:
            G.add_node(metric)
        
        # Add edges for correlations
        for corr in self.correlations:
            if corr.significance:
                G.add_edge(
                    corr.metric_pair[0],
                    corr.metric_pair[1],
                    weight=abs(corr.correlation),
                    correlation=corr.correlation,
                    lag=corr.lag,
                    direction=corr.direction
                )
        
        return G

    def create_visualization(self) -> go.Figure:
        """Create visualization of correlation analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Correlation Matrix",
                "Correlation Network",
                "Lag Analysis",
                "Strength Distribution"
            )
        )
        
        # Correlation matrix
        matrix_data = {}
        for corr in self.correlations:
            matrix_data[corr.metric_pair] = corr.correlation
        
        metrics = list(set(m for pair in matrix_data.keys() for m in pair))
        matrix = np.zeros((len(metrics), len(metrics)))
        
        for i, m1 in enumerate(metrics):
            for j, m2 in enumerate(metrics):
                if (m1, m2) in matrix_data:
                    matrix[i,j] = matrix_data[(m1, m2)]
                elif (m2, m1) in matrix_data:
                    matrix[i,j] = matrix_data[(m2, m1)]
        
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=metrics,
                y=metrics,
                colorscale='RdBu',
                zmid=0
            ),
            row=1, col=1
        )
        
        # Network visualization
        G = self.build_correlation_graph()
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.extend([edge[2]['correlation'], edge[2]['correlation'], None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    color=edge_colors,
                    colorscale='RdBu',
                    width=2
                ),
                hoverinfo='none'
            ),
            row=1, col=2
        )
        
        # Node visualization
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=10,
                    color='lightblue'
                )
            ),
            row=1, col=2
        )
        
        # Lag analysis
        lags = [c.lag for c in self.correlations if c.significance]
        directions = [c.direction for c in self.correlations if c.significance]
        
        fig.add_trace(
            go.Histogram(
                x=lags,
                nbinsx=self.config['analysis']['max_lag'],
                name="Lag Distribution"
            ),
            row=2, col=1
        )
        
        # Strength distribution
        strengths = [abs(c.correlation) for c in self.correlations]
        
        fig.add_trace(
            go.Histogram(
                x=strengths,
                nbinsx=20,
                name="Correlation Strength"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Alert Correlation Analysis"
        )
        
        return fig

    def generate_report(self) -> str:
        """Generate correlation analysis report."""
        lines = ["# Alert Correlation Analysis", ""]
        
        # Summary statistics
        lines.extend([
            "## Summary",
            f"- Total Correlations Analyzed: {len(self.correlations)}",
            f"- Significant Correlations: {sum(1 for c in self.correlations if c.significance)}",
            f"- Strong Correlations: {sum(1 for c in self.correlations if c.strength == 'strong')}",
            ""
        ])
        
        # Key findings
        strong_correlations = [c for c in self.correlations if c.strength == 'strong']
        if strong_correlations:
            lines.extend(["## Key Correlations", ""])
            for corr in sorted(strong_correlations, key=lambda x: abs(x.correlation), reverse=True):
                lines.extend([
                    f"### {corr.metric_pair[0]} vs {corr.metric_pair[1]}",
                    f"- Correlation: {corr.correlation:.3f}",
                    f"- Relationship: {corr.metric_pair[0]} {corr.direction} {corr.metric_pair[1]} by {corr.lag} hours",
                    f"- Strength: {corr.strength}",
                    f"- P-value: {corr.p_value:.4f}",
                    ""
                ])
        
        # Add recommendations
        lines.extend([
            "## Recommendations",
            "Based on the correlation analysis:"
        ])
        
        for corr in strong_correlations[:5]:  # Top 5 strongest correlations
            if corr.direction == "leads":
                lines.append(
                    f"- Monitor {corr.metric_pair[0]} as an early warning for {corr.metric_pair[1]} issues "
                    f"({corr.lag} hour(s) lead time)"
                )
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Initialize analyzer
        analyzer = AlertCorrelationAnalyzer(history_dir)
        
        # Calculate correlations
        analyzer.calculate_correlations()
        
        # Generate visualizations
        fig = analyzer.create_visualization()
        fig.write_html(history_dir / "correlation_analysis.html")
        
        # Generate report
        report = analyzer.generate_report()
        report_file = history_dir / "correlation_report.md"
        report_file.write_text(report)
        
        logger.info(f"Correlation analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during correlation analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
