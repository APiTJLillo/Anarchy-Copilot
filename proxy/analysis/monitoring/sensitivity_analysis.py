"""Sensitivity analysis for power analysis parameters."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import logging
from pathlib import Path
import json
from itertools import product
import concurrent.futures

from .power_analysis import PowerAnalyzer, PowerConfig

logger = logging.getLogger(__name__)

@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis."""
    parameter_ranges: Dict[str, List[float]] = None
    num_samples: int = 1000
    num_iterations: int = 100
    parallel: bool = True
    max_workers: int = -1
    confidence_level: float = 0.95
    output_path: Optional[Path] = None
    save_results: bool = True
    
    def __post_init__(self):
        if self.parameter_ranges is None:
            self.parameter_ranges = {
                "effect_size": [0.1, 0.2, 0.3, 0.5, 0.8],
                "alpha": [0.01, 0.05, 0.1],
                "sample_size": [30, 50, 100, 200, 500],
                "variance": [0.5, 1.0, 2.0]
            }

class SensitivityAnalyzer:
    """Analyze sensitivity of power analysis parameters."""
    
    def __init__(
        self,
        power_analyzer: PowerAnalyzer,
        config: SensitivityConfig
    ):
        self.power_analyzer = power_analyzer
        self.config = config
        self.sensitivity_results: Dict[str, Dict[str, Any]] = {}
        self.parameter_importance: Dict[str, float] = {}
    
    def analyze_sensitivity(
        self,
        test_type: str = "t_test"
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis."""
        try:
            # Generate parameter combinations
            param_combinations = self._generate_combinations()
            
            # Analyze each combination
            results = self._analyze_combinations(
                param_combinations,
                test_type
            )
            
            # Calculate parameter importance
            importance = self._calculate_importance(results)
            
            # Generate sensitivity metrics
            metrics = self._calculate_sensitivity_metrics(results)
            
            # Store results
            analysis = {
                "parameter_importance": importance,
                "sensitivity_metrics": metrics,
                "recommendations": self._generate_recommendations(
                    importance,
                    metrics
                )
            }
            
            self.sensitivity_results[test_type] = analysis
            
            if self.config.save_results:
                self._save_results(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Sensitivity analysis failed: {e}")
            return {"error": str(e)}
    
    def _generate_combinations(self) -> List[Dict[str, float]]:
        """Generate parameter combinations for analysis."""
        param_names = list(self.config.parameter_ranges.keys())
        param_values = list(self.config.parameter_ranges.values())
        
        combinations = []
        for values in product(*param_values):
            combination = dict(zip(param_names, values))
            combinations.append(combination)
        
        return combinations
    
    def _analyze_combinations(
        self,
        combinations: List[Dict[str, float]],
        test_type: str
    ) -> List[Dict[str, Any]]:
        """Analyze power for parameter combinations."""
        results = []
        
        if self.config.parallel:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.config.max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._analyze_single_combination,
                        combination,
                        test_type
                    )
                    for combination in combinations
                ]
                
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
        else:
            for combination in combinations:
                result = self._analyze_single_combination(
                    combination,
                    test_type
                )
                if result:
                    results.append(result)
        
        return results
    
    def _analyze_single_combination(
        self,
        params: Dict[str, float],
        test_type: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze single parameter combination."""
        try:
            power_result = self.power_analyzer.analyze_power(
                test_type=test_type,
                observed_effect=params["effect_size"],
                sample_size=int(params["sample_size"]),
                variance=params["variance"]
            )
            
            return {
                "parameters": params,
                "power": power_result["achieved_power"],
                "required_n": power_result["required_sample_size"]
            }
            
        except Exception as e:
            logger.error(f"Analysis failed for parameters {params}: {e}")
            return None
    
    def _calculate_importance(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate parameter importance scores."""
        # Convert results to DataFrame
        df = pd.DataFrame([
            {**r["parameters"], "power": r["power"]}
            for r in results
        ])
        
        importance = {}
        baseline_var = np.var(df["power"])
        
        # Calculate importance for each parameter
        for param in self.config.parameter_ranges:
            # Calculate variance reduction
            param_groups = df.groupby(param)["power"]
            within_group_var = np.mean([np.var(g) for _, g in param_groups])
            
            # Importance score based on variance reduction
            importance[param] = (baseline_var - within_group_var) / baseline_var
        
        # Normalize scores
        total = sum(importance.values())
        if total > 0:
            importance = {
                k: v/total for k, v in importance.items()
            }
        
        self.parameter_importance = importance
        return importance
    
    def _calculate_sensitivity_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """Calculate sensitivity metrics for parameters."""
        df = pd.DataFrame([
            {**r["parameters"], "power": r["power"]}
            for r in results
        ])
        
        metrics = {}
        for param in self.config.parameter_ranges:
            # Calculate average change in power per unit change in parameter
            changes = []
            values = sorted(self.config.parameter_ranges[param])
            
            for i in range(len(values) - 1):
                power1 = df[df[param] == values[i]]["power"].mean()
                power2 = df[df[param] == values[i+1]]["power"].mean()
                param_change = values[i+1] - values[i]
                
                if param_change != 0:
                    changes.append((power2 - power1) / param_change)
            
            # Calculate metrics
            metrics[param] = {
                "mean_sensitivity": float(np.mean(changes)),
                "std_sensitivity": float(np.std(changes)),
                "max_sensitivity": float(np.max(np.abs(changes))),
                "variance_explained": float(
                    df.groupby(param)["power"].mean().var() / df["power"].var()
                )
            }
        
        return metrics
    
    def _generate_recommendations(
        self,
        importance: Dict[str, float],
        metrics: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations based on sensitivity analysis."""
        recommendations = []
        
        # Sort parameters by importance
        sorted_params = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # High importance parameters
        for param, imp in sorted_params[:2]:
            recommendations.append({
                "type": "focus",
                "parameter": param,
                "importance": imp,
                "message": (
                    f"Focus on {param} as it explains {imp*100:.1f}% "
                    "of power variation"
                ),
                "priority": "high"
            })
        
        # High sensitivity parameters
        for param, metric in metrics.items():
            if metric["max_sensitivity"] > 0.5:
                recommendations.append({
                    "type": "sensitivity",
                    "parameter": param,
                    "sensitivity": metric["max_sensitivity"],
                    "message": (
                        f"Exercise caution with {param} due to high "
                        "sensitivity to changes"
                    ),
                    "priority": "medium"
                })
        
        # Low importance parameters
        for param, imp in sorted_params[-2:]:
            if imp < 0.1:
                recommendations.append({
                    "type": "efficiency",
                    "parameter": param,
                    "importance": imp,
                    "message": f"Consider fixing {param} due to low importance",
                    "priority": "low"
                })
        
        return recommendations
    
    def plot_sensitivity(self) -> Dict[str, go.Figure]:
        """Create sensitivity analysis visualizations."""
        plots = {}
        
        # Parameter importance plot
        fig = go.Figure()
        params = list(self.parameter_importance.keys())
        importance = list(self.parameter_importance.values())
        
        fig.add_trace(go.Bar(
            x=params,
            y=importance,
            name="Parameter Importance"
        ))
        
        fig.update_layout(
            title="Parameter Importance",
            xaxis_title="Parameter",
            yaxis_title="Relative Importance",
            showlegend=False
        )
        
        plots["importance"] = fig
        
        # Sensitivity curves
        if self.sensitivity_results:
            test_type = list(self.sensitivity_results.keys())[0]
            metrics = self.sensitivity_results[test_type]["sensitivity_metrics"]
            
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    "Mean Sensitivity",
                    "Variance Explained"
                )
            )
            
            params = list(metrics.keys())
            mean_sens = [m["mean_sensitivity"] for m in metrics.values()]
            var_exp = [m["variance_explained"] for m in metrics.values()]
            
            fig.add_trace(
                go.Bar(x=params, y=mean_sens),
                row=1,
                col=1
            )
            
            fig.add_trace(
                go.Bar(x=params, y=var_exp),
                row=2,
                col=1
            )
            
            plots["sensitivity"] = fig
        
        return plots
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save sensitivity analysis results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / "sensitivity_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
            
            # Save visualizations
            plots = self.plot_sensitivity()
            for name, fig in plots.items():
                fig.write_html(str(output_path / f"{name}.html"))
            
            logger.info(f"Saved sensitivity analysis to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save sensitivity analysis: {e}")

def create_sensitivity_analyzer(
    power_analyzer: PowerAnalyzer,
    output_path: Optional[Path] = None
) -> SensitivityAnalyzer:
    """Create sensitivity analyzer."""
    config = SensitivityConfig(output_path=output_path)
    return SensitivityAnalyzer(power_analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .power_analysis import create_analyzer
    
    power_analyzer = create_analyzer()
    sensitivity_analyzer = create_sensitivity_analyzer(
        power_analyzer,
        Path("sensitivity_analysis")
    )
    
    # Run sensitivity analysis
    results = sensitivity_analyzer.analyze_sensitivity()
    print(json.dumps(results, indent=2))
