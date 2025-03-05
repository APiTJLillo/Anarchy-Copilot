#!/usr/bin/env python3
"""Sensitivity analysis for counterfactual scenarios."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.analyze_counterfactuals import CounterfactualAnalyzer, CounterfactualScenario

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SensitivityResult:
    """Results of sensitivity analysis."""
    parameter: str
    values: List[float]
    impacts: Dict[str, List[float]]
    elasticities: Dict[str, float]
    thresholds: Dict[str, float]
    stability_scores: Dict[str, float]

class SensitivityAnalyzer:
    """Analyze sensitivity of counterfactual scenarios."""
    
    def __init__(
        self,
        history_dir: Path,
        config_file: Optional[Path] = None,
        workers: int = 4
    ):
        self.history_dir = history_dir
        self.config = self._load_config(config_file)
        self.workers = workers
        self.counterfactual_analyzer = CounterfactualAnalyzer(history_dir)

    def _load_config(self, config_file: Optional[Path]) -> Dict[str, Any]:
        """Load sensitivity analysis configuration."""
        default_config = {
            "parameters": {
                "magnitude": {
                    "range": [0.1, 0.5],  # 10% to 50% change
                    "steps": 10
                },
                "duration": {
                    "range": [1, 24],  # 1 to 24 hours
                    "steps": 8
                }
            },
            "analysis": {
                "stability_threshold": 0.1,  # 10% variation threshold
                "significance_level": 0.05,
                "min_effect_size": 0.05
            },
            "visualization": {
                "contour_resolution": 100,
                "heatmap_colorscale": "RdBu",
                "surface_colorscale": "Viridis"
            }
        }

        if config_file and config_file.exists():
            custom_config = json.loads(config_file.read_text())
            default_config.update(custom_config)

        return default_config

    def _generate_parameter_values(self, parameter: str) -> np.ndarray:
        """Generate parameter values for sensitivity analysis."""
        config = self.config["parameters"][parameter]
        return np.linspace(
            config["range"][0],
            config["range"][1],
            config["steps"]
        )

    def _calculate_elasticity(
        self,
        parameter_values: np.ndarray,
        impact_values: np.ndarray
    ) -> float:
        """Calculate elasticity (% change in output / % change in input)."""
        # Use central differences
        param_pct_change = np.diff(parameter_values) / parameter_values[:-1]
        impact_pct_change = np.diff(impact_values) / impact_values[:-1]
        
        # Average elasticity across range
        valid_mask = ~(np.isnan(param_pct_change) | np.isnan(impact_pct_change))
        if not valid_mask.any():
            return 0.0
        
        return np.mean(impact_pct_change[valid_mask] / param_pct_change[valid_mask])

    def _find_threshold(
        self,
        parameter_values: np.ndarray,
        impact_values: np.ndarray
    ) -> float:
        """Find threshold where impact becomes significant."""
        min_effect = self.config["analysis"]["min_effect_size"]
        
        # Find first value exceeding threshold
        significant = np.abs(impact_values) > min_effect
        if not significant.any():
            return float('inf')
        
        return parameter_values[significant.argmax()]

    def _calculate_stability_score(
        self,
        impact_values: np.ndarray
    ) -> float:
        """Calculate stability score based on impact variation."""
        if len(impact_values) < 2:
            return 1.0
            
        # Calculate relative variations
        variations = np.abs(np.diff(impact_values)) / np.abs(impact_values[:-1])
        variations = variations[~np.isnan(variations)]
        
        if len(variations) == 0:
            return 1.0
        
        # Score based on how many variations exceed threshold
        threshold = self.config["analysis"]["stability_threshold"]
        stability = 1.0 - np.mean(variations > threshold)
        
        return stability

    def analyze_sensitivity(
        self,
        base_scenario: CounterfactualScenario,
        parameter: str
    ) -> SensitivityResult:
        """Analyze sensitivity to a specific parameter."""
        parameter_values = self._generate_parameter_values(parameter)
        impacts: Dict[str, List[float]] = {}
        
        def analyze_value(value: float) -> Tuple[float, Dict[str, float]]:
            # Create modified scenario
            scenario_dict = base_scenario.__dict__.copy()
            if parameter == "duration":
                scenario_dict["duration"] = timedelta(hours=value)
            else:
                scenario_dict[parameter] = value
            
            modified_scenario = CounterfactualScenario(**scenario_dict)
            
            # Analyze counterfactual
            result = self.counterfactual_analyzer.analyze_counterfactual(
                modified_scenario
            )
            
            return value, result.impact

        # Run analyses in parallel
        scenario_impacts: Dict[float, Dict[str, float]] = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(analyze_value, value)
                for value in parameter_values
            ]
            
            for future in as_completed(futures):
                value, impact = future.result()
                scenario_impacts[value] = impact

        # Organize results by metric
        metrics = set()
        for impact in scenario_impacts.values():
            metrics.update(impact.keys())

        for metric in metrics:
            impacts[metric] = [
                scenario_impacts[value].get(metric, 0.0)
                for value in parameter_values
            ]

        # Calculate elasticities and thresholds
        elasticities = {}
        thresholds = {}
        stability_scores = {}
        
        for metric, values in impacts.items():
            elasticities[metric] = self._calculate_elasticity(
                parameter_values,
                np.array(values)
            )
            thresholds[metric] = self._find_threshold(
                parameter_values,
                np.array(values)
            )
            stability_scores[metric] = self._calculate_stability_score(
                np.array(values)
            )

        return SensitivityResult(
            parameter=parameter,
            values=list(parameter_values),
            impacts=impacts,
            elasticities=elasticities,
            thresholds=thresholds,
            stability_scores=stability_scores
        )

    def analyze_joint_sensitivity(
        self,
        base_scenario: CounterfactualScenario,
        parameter1: str,
        parameter2: str
    ) -> Dict[str, np.ndarray]:
        """Analyze sensitivity to two parameters jointly."""
        values1 = self._generate_parameter_values(parameter1)
        values2 = self._generate_parameter_values(parameter2)
        
        # Create parameter grid
        param_grid = list(product(values1, values2))
        impacts: Dict[str, List[float]] = defaultdict(list)
        
        def analyze_point(point: Tuple[float, float]) -> Tuple[Tuple[float, float], Dict[str, float]]:
            val1, val2 = point
            scenario_dict = base_scenario.__dict__.copy()
            
            if parameter1 == "duration":
                scenario_dict["duration"] = timedelta(hours=val1)
            else:
                scenario_dict[parameter1] = val1
                
            if parameter2 == "duration":
                scenario_dict["duration"] = timedelta(hours=val2)
            else:
                scenario_dict[parameter2] = val2
            
            modified_scenario = CounterfactualScenario(**scenario_dict)
            result = self.counterfactual_analyzer.analyze_counterfactual(
                modified_scenario
            )
            
            return point, result.impact

        # Run analyses in parallel
        point_impacts: Dict[Tuple[float, float], Dict[str, float]] = {}
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = [
                executor.submit(analyze_point, point)
                for point in param_grid
            ]
            
            for future in as_completed(futures):
                point, impact = future.result()
                point_impacts[point] = impact

        # Organize results into grids for each metric
        metrics = set()
        for impact in point_impacts.values():
            metrics.update(impact.keys())

        impact_grids = {}
        for metric in metrics:
            grid = np.zeros((len(values1), len(values2)))
            for i, v1 in enumerate(values1):
                for j, v2 in enumerate(values2):
                    grid[i, j] = point_impacts[(v1, v2)].get(metric, 0.0)
            impact_grids[metric] = grid

        return impact_grids

    def create_visualization(
        self,
        result: SensitivityResult
    ) -> go.Figure:
        """Create visualization of sensitivity analysis results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Impact vs Parameter",
                "Elasticity Analysis",
                "Threshold Analysis",
                "Stability Scores"
            )
        )

        # Impact curves
        for metric, values in result.impacts.items():
            fig.add_trace(
                go.Scatter(
                    x=result.values,
                    y=values,
                    name=f"{metric} Impact",
                    mode='lines+markers'
                ),
                row=1, col=1
            )

        # Elasticity bars
        metrics = list(result.elasticities.keys())
        elasticities = [result.elasticities[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=elasticities,
                name="Elasticity"
            ),
            row=1, col=2
        )

        # Threshold analysis
        thresholds = [result.thresholds[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=thresholds,
                name="Threshold"
            ),
            row=2, col=1
        )

        # Stability scores
        stability = [result.stability_scores[m] for m in metrics]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=stability,
                name="Stability",
                marker_color=[
                    '#2ecc71' if s > 0.8 else '#e74c3c'
                    for s in stability
                ]
            ),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text=f"Sensitivity Analysis for {result.parameter}"
        )

        return fig

    def create_joint_visualization(
        self,
        impact_grids: Dict[str, np.ndarray],
        parameter1: str,
        parameter2: str
    ) -> go.Figure:
        """Create visualization of joint sensitivity analysis."""
        metrics = list(impact_grids.keys())
        cols = min(2, len(metrics))
        rows = (len(metrics) + 1) // 2
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=metrics,
            specs=[[{"type": "surface"}] * cols] * rows
        )

        for i, metric in enumerate(metrics):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Surface(
                    z=impact_grids[metric],
                    name=metric,
                    colorscale=self.config["visualization"]["surface_colorscale"],
                    showscale=True if i == 0 else False
                ),
                row=row,
                col=col
            )

        fig.update_layout(
            height=400 * rows,
            title_text=f"Joint Sensitivity Analysis: {parameter1} vs {parameter2}",
            scene=dict(
                xaxis_title=parameter1,
                yaxis_title=parameter2,
                zaxis_title="Impact"
            )
        )

        return fig

    def generate_report(
        self,
        results: Dict[str, SensitivityResult],
        joint_results: Optional[Dict[str, Dict[str, np.ndarray]]] = None
    ) -> str:
        """Generate sensitivity analysis report."""
        lines = ["# Sensitivity Analysis Report", ""]
        
        # Individual parameter results
        for parameter, result in results.items():
            lines.extend([
                f"## {parameter.title()} Sensitivity",
                "",
                "### Elasticity Analysis",
                "Impact elasticity by metric:",
                ""
            ])
            
            # Sort metrics by elasticity magnitude
            sorted_metrics = sorted(
                result.elasticities.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            for metric, elasticity in sorted_metrics:
                lines.append(f"- {metric}: {elasticity:.3f}")
            lines.append("")
            
            lines.extend([
                "### Threshold Analysis",
                "Parameter thresholds for significant impact:",
                ""
            ])
            
            for metric, threshold in result.thresholds.items():
                if threshold == float('inf'):
                    lines.append(f"- {metric}: No significant threshold found")
                else:
                    lines.append(f"- {metric}: {threshold:.3f}")
            lines.append("")
            
            lines.extend([
                "### Stability Analysis",
                "Stability scores by metric:",
                ""
            ])
            
            for metric, score in result.stability_scores.items():
                stability = "High" if score > 0.8 else "Medium" if score > 0.5 else "Low"
                lines.append(f"- {metric}: {score:.2f} ({stability} stability)")
            lines.append("")

        # Joint sensitivity results
        if joint_results:
            lines.extend(["## Joint Sensitivity Analysis", ""])
            
            for params, impacts in joint_results.items():
                param1, param2 = params.split("_vs_")
                lines.extend([
                    f"### {param1.title()} vs {param2.title()}",
                    "Key interactions:",
                    ""
                ])
                
                for metric, grid in impacts.items():
                    # Find maximum impact point
                    max_idx = np.unravel_index(np.argmax(np.abs(grid)), grid.shape)
                    max_impact = grid[max_idx]
                    
                    lines.extend([
                        f"#### {metric}",
                        f"- Maximum impact: {max_impact:.3f}",
                        f"- Occurs at {param1}={max_idx[0]:.2f}, {param2}={max_idx[1]:.2f}",
                        ""
                    ])
        
        # Recommendations
        lines.extend([
            "## Recommendations",
            "Based on sensitivity analysis:"
        ])
        
        for parameter, result in results.items():
            high_elasticity_metrics = [
                metric for metric, elasticity in result.elasticities.items()
                if abs(elasticity) > 1.0
            ]
            
            if high_elasticity_metrics:
                lines.append(
                    f"- {parameter.title()} changes strongly affect: "
                    f"{', '.join(high_elasticity_metrics)}"
                )
            
            unstable_metrics = [
                metric for metric, score in result.stability_scores.items()
                if score < 0.5
            ]
            
            if unstable_metrics:
                lines.append(
                    f"- Use caution when adjusting {parameter} due to unstable "
                    f"response in: {', '.join(unstable_metrics)}"
                )

        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        history_dir = Path("benchmark_results/performance_history")
        if not history_dir.exists():
            logger.error("No performance history directory found")
            return 1
        
        # Create base scenario
        base_scenario = CounterfactualScenario(
            metric="throughput",
            intervention="increase",
            magnitude=0.2,
            timestamp=datetime.now(),
            duration=timedelta(hours=4)
        )
        
        # Initialize analyzer
        analyzer = SensitivityAnalyzer(history_dir)
        
        # Analyze individual parameter sensitivity
        parameters = ["magnitude", "duration"]
        results = {}
        for parameter in parameters:
            results[parameter] = analyzer.analyze_sensitivity(
                base_scenario,
                parameter
            )
            
            # Generate visualization
            fig = analyzer.create_visualization(results[parameter])
            fig.write_html(
                history_dir / f"sensitivity_{parameter}.html"
            )
        
        # Analyze joint sensitivity
        joint_results = {}
        for p1, p2 in combinations(parameters, 2):
            impact_grids = analyzer.analyze_joint_sensitivity(
                base_scenario,
                p1,
                p2
            )
            
            key = f"{p1}_vs_{p2}"
            joint_results[key] = impact_grids
            
            # Generate visualization
            fig = analyzer.create_joint_visualization(
                impact_grids,
                p1,
                p2
            )
            fig.write_html(
                history_dir / f"sensitivity_{key}.html"
            )
        
        # Generate report
        report = analyzer.generate_report(results, joint_results)
        report_file = history_dir / "sensitivity_analysis.md"
        report_file.write_text(report)
        
        logger.info(f"Sensitivity analysis complete. Reports written to {history_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during sensitivity analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
