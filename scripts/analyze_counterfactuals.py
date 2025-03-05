#!/usr/bin/env python3
"""Analyze counterfactual scenarios for performance metrics."""

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
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CounterfactualScenario:
    """Definition of a counterfactual scenario."""
    metric: str
    intervention: str  # "increase", "decrease", "stabilize"
    magnitude: float
    timestamp: datetime
    duration: timedelta

@dataclass
class CounterfactualResult:
    """Results of counterfactual analysis."""
    scenario: CounterfactualScenario
    impact: Dict[str, float]
    confidence: Dict[str, float]
    propagation_path: List[str]
    time_to_effect: Dict[str, timedelta]
    side_effects: Dict[str, float]

class CounterfactualAnalyzer:
    """Analyze counterfactual scenarios in performance data."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        causal_graph: Optional[nx.DiGraph] = None
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.causal_graph = causal_graph
        self.data: Optional[pd.DataFrame] = None
        self.var_models: Dict[str, Any] = {}

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load counterfactual analysis configuration."""
        default_config = {
            "analysis": {
                "confidence_level": 0.95,
                "max_propagation_steps": 5,
                "min_effect_size": 0.1,
                "simulation_iterations": 1000
            },
            "interventions": {
                "max_magnitude": 0.5,  # Maximum 50% change
                "min_duration": "1h",
                "max_duration": "24h"
            },
            "propagation": {
                "decay_factor": 0.8,
                "threshold": 0.1,
                "max_side_effects": 5
            },
            "modeling": {
                "var_lags": 24,
                "train_test_split": 0.8,
                "cv_folds": 5
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _load_data(self) -> None:
        """Load and prepare time series data."""
        if self.data is not None:
            return

        metrics = []
        for metric_file in self.history_dir.glob("**/performance_*.json"):
            data = pd.DataFrame(json.loads(metric_file.read_text()))
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            metrics.append(data)

        if not metrics:
            raise ValueError("No performance metrics found")

        self.data = pd.concat(metrics)
        self.data = self.data.set_index('timestamp').sort_index()
        self.data = self.data.resample('1H').mean()
        self.data = self.data.fillna(method='ffill').fillna(method='bfill')

    def _fit_var_models(self) -> None:
        """Fit VAR models for counterfactual simulation."""
        if not self.data is None:
            self._load_data()

        # Prepare data for VAR modeling
        data = self.data.copy()
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(data),
            index=data.index,
            columns=data.columns
        )

        # Split into training and test sets
        train_size = int(len(data) * self.config['modeling']['train_test_split'])
        train_data = scaled_data[:train_size]

        # Fit VAR model
        model = sm.tsa.VAR(train_data)
        lags = self.config['modeling']['var_lags']
        fitted = model.fit(maxlags=lags, ic='aic')

        self.var_models = {
            'model': fitted,
            'scaler': scaler
        }

    def simulate_intervention(
        self,
        scenario: CounterfactualScenario
    ) -> pd.DataFrame:
        """Simulate the effect of an intervention."""
        if not self.var_models:
            self._fit_var_models()

        # Prepare base data
        data = self.data.copy()
        scaler = self.var_models['scaler']
        model = self.var_models['model']

        # Create intervention
        intervention_idx = scenario.timestamp
        duration_steps = int(scenario.duration.total_seconds() / 3600)  # Convert to hours
        
        scaled_data = pd.DataFrame(
            scaler.transform(data),
            index=data.index,
            columns=data.columns
        )

        # Apply intervention
        metric_idx = data.columns.get_loc(scenario.metric)
        original_values = scaled_data.iloc[
            data.index.get_loc(intervention_idx):
            data.index.get_loc(intervention_idx) + duration_steps,
            metric_idx
        ].copy()

        if scenario.intervention == "increase":
            change = scenario.magnitude
        elif scenario.intervention == "decrease":
            change = -scenario.magnitude
        else:  # stabilize
            change = 0

        scaled_data.iloc[
            data.index.get_loc(intervention_idx):
            data.index.get_loc(intervention_idx) + duration_steps,
            metric_idx
        ] += change

        # Simulate forward propagation
        simulation_steps = self.config['analysis']['max_propagation_steps']
        simulated = scaled_data.copy()
        
        for step in range(simulation_steps):
            # Use VAR model to predict next step
            lag_data = simulated.iloc[-model.k_ar:]
            next_step = model.forecast(lag_data.values, steps=1)
            
            if len(simulated) < len(data):
                simulated.loc[simulated.index[-1] + pd.Timedelta(hours=1)] = next_step[0]

        # Convert back to original scale
        return pd.DataFrame(
            scaler.inverse_transform(simulated),
            index=simulated.index,
            columns=data.columns
        )

    def analyze_counterfactual(
        self,
        scenario: CounterfactualScenario
    ) -> CounterfactualResult:
        """Analyze the impact of a counterfactual scenario."""
        # Simulate intervention
        simulated = self.simulate_intervention(scenario)
        original = self.data.copy()

        # Calculate impacts
        impact = {}
        confidence = {}
        time_to_effect = {}
        side_effects = {}

        # Calculate immediate impact on target metric
        target_impact = (
            simulated[scenario.metric] - original[scenario.metric]
        ).mean()
        impact[scenario.metric] = target_impact

        # Calculate propagation effects
        if self.causal_graph:
            propagation_path = []
            visited = {scenario.metric}
            queue = [(scenario.metric, 0)]

            while queue:
                current, depth = queue.pop(0)
                propagation_path.append(current)

                if depth >= self.config['analysis']['max_propagation_steps']:
                    continue

                # Find downstream metrics
                for _, downstream in self.causal_graph.edges(current):
                    if downstream not in visited:
                        visited.add(downstream)
                        queue.append((downstream, depth + 1))

                        # Calculate impact
                        diff = simulated[downstream] - original[downstream]
                        effect = diff.mean()
                        conf = 1 - diff.std() / abs(effect) if abs(effect) > 0 else 0

                        if abs(effect) > self.config['propagation']['threshold']:
                            impact[downstream] = effect
                            confidence[downstream] = conf
                            time_to_effect[downstream] = self._calculate_time_to_effect(
                                original[downstream],
                                simulated[downstream],
                                scenario.timestamp
                            )

        # Identify side effects
        for col in simulated.columns:
            if col not in impact and col != scenario.metric:
                diff = simulated[col] - original[col]
                effect = diff.mean()
                if abs(effect) > self.config['propagation']['threshold']:
                    side_effects[col] = effect

        return CounterfactualResult(
            scenario=scenario,
            impact=impact,
            confidence=confidence,
            propagation_path=propagation_path,
            time_to_effect=time_to_effect,
            side_effects=side_effects
        )

    def _calculate_time_to_effect(
        self,
        original: pd.Series,
        simulated: pd.Series,
        start_time: datetime
    ) -> timedelta:
        """Calculate time until significant effect is observed."""
        diff = simulated - original
        threshold = self.config['propagation']['threshold'] * original.std()
        
        effect_mask = abs(diff) > threshold
        if not effect_mask.any():
            return timedelta(hours=0)
        
        first_effect = effect_mask[effect_mask].index[0]
        return first_effect - start_time

    def create_visualization(
        self,
        result: CounterfactualResult
    ) -> go.Figure:
        """Create visualization of counterfactual analysis."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Impact Over Time",
                "Propagation Graph",
                "Confidence Levels",
                "Side Effects"
            )
        )

        # Impact over time
        for metric, impact in result.impact.items():
            fig.add_trace(
                go.Scatter(
                    x=self.data.index,
                    y=self.data[metric],
                    name=f"{metric} (Original)",
                    line=dict(dash='dot')
                ),
                row=1, col=1
            )
            
            simulated = self.simulate_intervention(result.scenario)
            fig.add_trace(
                go.Scatter(
                    x=simulated.index,
                    y=simulated[metric],
                    name=f"{metric} (Simulated)",
                    line=dict(dash='solid')
                ),
                row=1, col=1
            )

        # Propagation graph
        if self.causal_graph and result.propagation_path:
            pos = nx.spring_layout(self.causal_graph)
            
            # Plot edges
            edge_trace = go.Scatter(
                x=[],
                y=[],
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            for edge in self.causal_graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_trace['x'] += (x0, x1, None)
                edge_trace['y'] += (y0, y1, None)

            fig.add_trace(edge_trace, row=1, col=2)

            # Plot nodes
            node_x = []
            node_y = []
            node_colors = []
            for node in self.causal_graph.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                if node in result.impact:
                    node_colors.append('#2ecc71')
                elif node in result.side_effects:
                    node_colors.append('#e74c3c')
                else:
                    node_colors.append('#95a5a6')

            fig.add_trace(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode='markers+text',
                    text=list(self.causal_graph.nodes()),
                    marker=dict(
                        color=node_colors,
                        size=10
                    )
                ),
                row=1, col=2
            )

        # Confidence levels
        confidence_metrics = list(result.confidence.keys())
        confidence_values = list(result.confidence.values())

        fig.add_trace(
            go.Bar(
                x=confidence_metrics,
                y=confidence_values,
                name="Confidence"
            ),
            row=2, col=1
        )

        # Side effects
        if result.side_effects:
            metrics = list(result.side_effects.keys())
            effects = list(result.side_effects.values())

            fig.add_trace(
                go.Bar(
                    x=metrics,
                    y=effects,
                    name="Side Effects",
                    marker_color='#e74c3c'
                ),
                row=2, col=2
            )

        fig.update_layout(
            height=800,
            title_text=f"Counterfactual Analysis: {result.scenario.intervention} "
                      f"{result.scenario.metric} by {result.scenario.magnitude:.1%}"
        )

        return fig

    def generate_report(
        self,
        result: CounterfactualResult
    ) -> str:
        """Generate counterfactual analysis report."""
        lines = ["# Counterfactual Analysis Report", ""]
        
        # Scenario details
        lines.extend([
            "## Scenario",
            f"- Metric: {result.scenario.metric}",
            f"- Intervention: {result.scenario.intervention}",
            f"- Magnitude: {result.scenario.magnitude:.1%}",
            f"- Duration: {result.scenario.duration}",
            f"- Start Time: {result.scenario.timestamp}",
            ""
        ])
        
        # Direct impact
        lines.extend([
            "## Direct Impact",
            f"- Target Metric: {result.scenario.metric}",
            f"- Impact Magnitude: {result.impact.get(result.scenario.metric, 0):.3f}",
            ""
        ])
        
        # Propagation effects
        if result.propagation_path:
            lines.extend([
                "## Propagation Effects",
                "Affected metrics in order of propagation:"
            ])
            
            for metric in result.propagation_path:
                if metric in result.impact:
                    lines.extend([
                        f"### {metric}",
                        f"- Impact: {result.impact[metric]:.3f}",
                        f"- Confidence: {result.confidence.get(metric, 0):.1%}",
                        f"- Time to Effect: {result.time_to_effect.get(metric, timedelta(0))}",
                        ""
                    ])
        
        # Side effects
        if result.side_effects:
            lines.extend([
                "## Side Effects",
                "Unintended effects on other metrics:"
            ])
            
            for metric, effect in result.side_effects.items():
                lines.extend([
                    f"- {metric}: {effect:.3f}"
                ])
            lines.append("")
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            "Based on the counterfactual analysis:"
        ])
        
        # Add recommendations based on results
        if abs(result.impact.get(result.scenario.metric, 0)) < self.config['analysis']['min_effect_size']:
            lines.append(
                "- The proposed intervention may not have sufficient impact. "
                "Consider increasing the magnitude or duration."
            )
        
        if result.side_effects:
            lines.append(
                "- Be cautious of side effects on other metrics. "
                "Consider implementing additional monitoring."
            )
        
        if any(conf < 0.8 for conf in result.confidence.values()):
            lines.append(
                "- Some effects have low confidence levels. "
                "Validate with smaller scale testing first."
            )
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Load causal graph if available
        causal_graph_file = history_dir / "causal_graph.json"
        causal_graph = None
        if causal_graph_file.exists():
            graph_data = json.loads(causal_graph_file.read_text())
            causal_graph = nx.node_link_graph(graph_data)
        
        # Initialize analyzer
        analyzer = CounterfactualAnalyzer(history_dir, causal_graph=causal_graph)
        
        # Example scenario
        scenario = CounterfactualScenario(
            metric="throughput",
            intervention="increase",
            magnitude=0.2,  # 20% increase
            timestamp=datetime.now(),
            duration=timedelta(hours=4)
        )
        
        # Analyze counterfactual
        result = analyzer.analyze_counterfactual(scenario)
        
        # Generate visualization
        fig = analyzer.create_visualization(result)
        fig.write_html(history_dir / "counterfactual_analysis.html")
        
        # Generate report
        report = analyzer.generate_report(result)
        report_file = history_dir / "counterfactual_analysis.md"
        report_file.write_text(report)
        
        logger.info(f"Counterfactual analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during counterfactual analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
