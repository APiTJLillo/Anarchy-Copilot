#!/usr/bin/env python3
"""Analyze causal relationships in performance metrics and alerts."""

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
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CausalEffect:
    """Representation of a causal relationship."""
    cause: str
    effect: str
    strength: float  # Effect size
    lag: int
    confidence: float
    f_stat: float
    p_value: float
    direction: str  # "forward", "backward", "bidirectional"

class CausalAnalyzer:
    """Analyze causal relationships in performance data."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        max_lag: int = 24  # Hours
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.max_lag = max_lag
        self.causal_effects: List[CausalEffect] = []
        self.causal_graph: Optional[nx.DiGraph] = None

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load causal analysis configuration."""
        default_config = {
            "analysis": {
                "min_samples": 48,  # Minimum samples for analysis
                "significance_level": 0.05,
                "min_effect_size": 0.1,
                "max_lags": 24,  # Hours
                "granger_test_method": "ssr_chi2test"
            },
            "filters": {
                "min_variance": 0.01,
                "max_missing": 0.2,
                "detrend": True,
                "remove_seasonality": True
            },
            "visualization": {
                "max_causes": 5,  # Top causes to show per effect
                "min_confidence": 0.8
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _prepare_data(self) -> pd.DataFrame:
        """Prepare time series data for causal analysis."""
        # Load and combine data
        alert_file = self.history_dir / "alert_history.json"
        alerts = pd.DataFrame(json.loads(alert_file.read_text()))
        alerts['timestamp'] = pd.to_datetime(alerts['timestamp'])
        
        metrics = []
        for metric_file in self.history_dir.glob("**/performance_*.json"):
            data = pd.DataFrame(json.loads(metric_file.read_text()))
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            metrics.append(data)
        
        df = pd.concat([alerts] + metrics)
        df = df.set_index('timestamp').sort_index()
        
        # Resample to hourly frequency
        df = df.resample('1H').mean()
        
        # Handle missing values
        if df.isna().mean().max() > self.config['filters']['max_missing']:
            logger.warning("High proportion of missing values")
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove low variance series
        variances = df.var()
        df = df.loc[:, variances > self.config['filters']['min_variance']]
        
        # Detrending if configured
        if self.config['filters']['detrend']:
            df = df.diff().fillna(0)
        
        # Remove seasonality if configured
        if self.config['filters']['remove_seasonality']:
            df = self._remove_seasonality(df)
        
        return df

    def _remove_seasonality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove seasonal patterns from time series."""
        result = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            # Decompose series
            decomposition = sm.tsa.seasonal_decompose(
                df[col],
                period=24,  # 24 hours
                extrapolate_trend='freq'
            )
            # Remove seasonal component
            result[col] = decomposition.trend + decomposition.resid
            
        return result

    def _test_granger_causality(
        self,
        data: pd.DataFrame,
        var1: str,
        var2: str
    ) -> Tuple[float, float, int, str]:
        """Test for Granger causality between two variables."""
        maxlag = self.config['analysis']['max_lags']
        test_method = self.config['analysis']['granger_test_method']
        
        # Test forward direction
        forward_tests = grangercausalitytests(
            data[[var1, var2]],
            maxlag=maxlag,
            verbose=False
        )
        
        # Test reverse direction
        reverse_tests = grangercausalitytests(
            data[[var2, var1]],
            maxlag=maxlag,
            verbose=False
        )
        
        # Find best lag and direction
        best_f = 0
        best_p = 1
        best_lag = 0
        direction = "none"
        
        for lag in range(1, maxlag + 1):
            # Forward tests
            f_forward = forward_tests[lag][0][test_method][0]
            p_forward = forward_tests[lag][0][test_method][1]
            
            # Reverse tests
            f_reverse = reverse_tests[lag][0][test_method][0]
            p_reverse = reverse_tests[lag][0][test_method][1]
            
            if p_forward < self.config['analysis']['significance_level']:
                if f_forward > best_f:
                    best_f = f_forward
                    best_p = p_forward
                    best_lag = lag
                    direction = "forward"
            
            if p_reverse < self.config['analysis']['significance_level']:
                if f_reverse > best_f:
                    best_f = f_reverse
                    best_p = p_reverse
                    best_lag = -lag
                    direction = "backward"
        
        return best_f, best_p, best_lag, direction

    def analyze_causality(self) -> None:
        """Perform causal analysis on metrics and alerts."""
        data = self._prepare_data()
        
        if len(data) < self.config['analysis']['min_samples']:
            logger.warning("Insufficient samples for causal analysis")
            return
        
        # Test each pair of variables
        variables = list(data.columns)
        for i, var1 in enumerate(variables[:-1]):
            for var2 in variables[i+1:]:
                # Test Granger causality
                f_stat, p_value, lag, direction = self._test_granger_causality(
                    data, var1, var2
                )
                
                if direction != "none":
                    # Calculate effect size using correlation at optimal lag
                    if lag > 0:
                        corr = data[var1].shift(lag).corr(data[var2])
                    else:
                        corr = data[var2].shift(-lag).corr(data[var1])
                    
                    effect_size = abs(corr)
                    if effect_size > self.config['analysis']['min_effect_size']:
                        # Create causal effect
                        if direction == "forward":
                            cause, effect = var1, var2
                        else:
                            cause, effect = var2, var1
                            lag = abs(lag)
                        
                        self.causal_effects.append(CausalEffect(
                            cause=cause,
                            effect=effect,
                            strength=effect_size,
                            lag=lag,
                            confidence=1 - p_value,
                            f_stat=f_stat,
                            p_value=p_value,
                            direction=direction
                        ))

    def build_causal_graph(self) -> nx.DiGraph:
        """Build directed graph of causal relationships."""
        G = nx.DiGraph()
        
        # Add nodes and edges
        for effect in self.causal_effects:
            if not G.has_node(effect.cause):
                G.add_node(effect.cause)
            if not G.has_node(effect.effect):
                G.add_node(effect.effect)
            
            G.add_edge(
                effect.cause,
                effect.effect,
                weight=effect.strength,
                lag=effect.lag,
                confidence=effect.confidence
            )
        
        self.causal_graph = G
        return G

    def identify_root_causes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Identify root causes for each metric."""
        if not self.causal_graph:
            self.build_causal_graph()
        
        root_causes = {}
        
        # For each node, find all paths leading to it
        for node in self.causal_graph.nodes():
            paths = []
            for source in self.causal_graph.nodes():
                if source != node:
                    try:
                        # Find all paths from source to node
                        for path in nx.all_simple_paths(self.causal_graph, source, node):
                            # Calculate path strength
                            strength = 1.0
                            total_lag = 0
                            for i in range(len(path) - 1):
                                edge = self.causal_graph.edges[path[i], path[i+1]]
                                strength *= edge['weight']
                                total_lag += edge['lag']
                            
                            paths.append({
                                'cause': source,
                                'path': path,
                                'strength': strength,
                                'total_lag': total_lag
                            })
                    except nx.NetworkXNoPath:
                        continue
            
            if paths:
                # Sort paths by strength
                paths.sort(key=lambda x: x['strength'], reverse=True)
                root_causes[node] = paths
        
        return root_causes

    def create_visualization(self) -> go.Figure:
        """Create visualization of causal relationships."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Causal Graph",
                "Effect Sizes",
                "Lag Distribution",
                "Confidence Distribution"
            )
        )
        
        # Causal graph visualization
        G = self.causal_graph or self.build_causal_graph()
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        edge_colors = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_colors.extend([edge[2]['weight'], edge[2]['weight'], None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(
                    color=edge_colors,
                    colorscale='Viridis',
                    width=2
                ),
                hoverinfo='none'
            ),
            row=1, col=1
        )
        
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
            row=1, col=1
        )
        
        # Effect sizes
        strengths = [effect.strength for effect in self.causal_effects]
        fig.add_trace(
            go.Histogram(
                x=strengths,
                name="Effect Sizes",
                nbinsx=20
            ),
            row=1, col=2
        )
        
        # Lag distribution
        lags = [effect.lag for effect in self.causal_effects]
        fig.add_trace(
            go.Histogram(
                x=lags,
                name="Lags",
                nbinsx=self.max_lag
            ),
            row=2, col=1
        )
        
        # Confidence distribution
        confidences = [effect.confidence for effect in self.causal_effects]
        fig.add_trace(
            go.Histogram(
                x=confidences,
                name="Confidence",
                nbinsx=20
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Causal Analysis Results"
        )
        
        return fig

    def generate_report(self) -> str:
        """Generate causal analysis report."""
        lines = ["# Causal Analysis Report", ""]
        
        # Summary statistics
        lines.extend([
            "## Summary",
            f"- Total Causal Relationships: {len(self.causal_effects)}",
            f"- Average Effect Size: {np.mean([e.strength for e in self.causal_effects]):.3f}",
            f"- Average Confidence: {np.mean([e.confidence for e in self.causal_effects]):.3f}",
            f"- Median Lag: {np.median([e.lag for e in self.causal_effects]):.1f} hours",
            ""
        ])
        
        # Root causes
        root_causes = self.identify_root_causes()
        lines.extend(["## Root Causes Analysis", ""])
        
        for effect, causes in root_causes.items():
            lines.extend([
                f"### Effects on {effect}",
                "Top root causes:"
            ])
            
            for cause in causes[:self.config['visualization']['max_causes']]:
                lines.extend([
                    f"- {cause['cause']}:",
                    f"  - Path: {' -> '.join(cause['path'])}",
                    f"  - Strength: {cause['strength']:.3f}",
                    f"  - Total Lag: {cause['total_lag']} hours",
                    ""
                ])
        
        # Strong causal relationships
        strong_effects = [
            e for e in self.causal_effects
            if e.confidence >= self.config['visualization']['min_confidence']
        ]
        if strong_effects:
            lines.extend(["## Strong Causal Relationships", ""])
            for effect in sorted(strong_effects, key=lambda x: x.strength, reverse=True):
                lines.extend([
                    f"### {effect.cause} → {effect.effect}",
                    f"- Effect Size: {effect.strength:.3f}",
                    f"- Lag: {effect.lag} hours",
                    f"- Confidence: {effect.confidence:.3f}",
                    f"- F-statistic: {effect.f_stat:.3f}",
                    ""
                ])
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Initialize analyzer
        analyzer = CausalAnalyzer(history_dir)
        
        # Perform analysis
        analyzer.analyze_causality()
        
        # Generate visualization
        fig = analyzer.create_visualization()
        fig.write_html(history_dir / "causal_analysis.html")
        
        # Generate report
        report = analyzer.generate_report()
        report_file = history_dir / "causal_analysis.md"
        report_file.write_text(report)
        
        logger.info(f"Causal analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during causal analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
