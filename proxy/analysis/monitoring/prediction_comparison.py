"""Comparative analysis of prediction models and interpretations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import logging
from pathlib import Path
import json
from collections import defaultdict

from .performance_prediction import PerformancePredictor
from .prediction_interpretation import PredictionInterpreter
from .performance_metrics import PerformanceMetric

logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for prediction comparison."""
    time_periods: List[timedelta] = None
    metrics_to_compare: List[str] = None
    baseline_window: timedelta = timedelta(hours=24)
    error_threshold: float = 0.1
    min_samples: int = 100
    comparison_methods: List[str] = None
    output_path: Optional[Path] = None
    save_results: bool = True
    
    def __post_init__(self):
        if self.time_periods is None:
            self.time_periods = [
                timedelta(hours=1),
                timedelta(hours=6),
                timedelta(hours=24)
            ]
        if self.metrics_to_compare is None:
            self.metrics_to_compare = ["fps", "memory_mb", "cpu_percent"]
        if self.comparison_methods is None:
            self.comparison_methods = ["rf", "prophet", "combined"]

class PredictionComparator:
    """Compare and evaluate prediction models."""
    
    def __init__(
        self,
        predictor: PerformancePredictor,
        interpreter: PredictionInterpreter,
        config: ComparisonConfig
    ):
        self.predictor = predictor
        self.interpreter = interpreter
        self.config = config
        
        self.comparison_results: Dict[str, Dict[str, Any]] = {}
        self.model_rankings: Dict[str, List[Dict[str, Any]]] = {}
        self.temporal_performance: Dict[str, List[Dict[str, Any]]] = []
    
    def compare_predictions(
        self,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Compare predictions across different models and time periods."""
        results = {}
        
        try:
            for metric in self.config.metrics_to_compare:
                metric_results = self._compare_metric_predictions(metric, metrics)
                if metric_results:
                    results[metric] = metric_results
            
            # Update comparison history
            self._update_comparison_history(results)
            
            # Generate rankings
            self._update_model_rankings(results)
            
            # Save results if configured
            if self.config.save_results:
                self._save_comparison_results(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction comparison failed: {e}")
            return {}
    
    def _compare_metric_predictions(
        self,
        metric: str,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Compare predictions for specific metric."""
        if len(metrics) < self.config.min_samples:
            return {}
        
        comparison = {}
        
        # Get actual values
        actual_values = self._get_actual_values(metric, metrics)
        
        # Compare across time periods
        for period in self.config.time_periods:
            period_comparison = {}
            
            # Get predictions for each method
            for method in self.config.comparison_methods:
                predictions = self._get_predictions(
                    metric,
                    metrics,
                    method,
                    period
                )
                
                if predictions:
                    # Calculate performance metrics
                    performance = self._calculate_performance(
                        actual_values,
                        predictions
                    )
                    period_comparison[method] = performance
            
            if period_comparison:
                comparison[str(period)] = period_comparison
        
        # Add interpretation comparison
        comparison["interpretability"] = self._compare_interpretations(
            metric,
            metrics
        )
        
        return comparison
    
    def _get_actual_values(
        self,
        metric: str,
        metrics: List[PerformanceMetric]
    ) -> List[float]:
        """Get actual metric values."""
        values = []
        for m in metrics:
            if metric == "fps":
                values.append(m.fps)
            elif metric == "memory_mb":
                values.append(m.memory_mb)
            elif metric == "cpu_percent":
                values.append(m.cpu_percent)
        return values
    
    def _get_predictions(
        self,
        metric: str,
        metrics: List[PerformanceMetric],
        method: str,
        period: timedelta
    ) -> List[float]:
        """Get predictions for specific method and period."""
        try:
            if method == "combined":
                return self.predictor.predict_performance(metrics)["predictions"][metric]["forecast"]
            elif method in ["rf", "prophet"]:
                model = self.predictor.models[metric][method]
                df = self.predictor._prepare_training_data(metrics)
                X = df[self.predictor.models[metric]["features"]].values
                if method == "rf":
                    X_scaled = self.predictor.scaler.transform(X)
                    return model.predict(X_scaled).tolist()
                else:  # prophet
                    future = model.make_future_dataframe(
                        periods=int(period.total_seconds() / 60),
                        freq="T"
                    )
                    forecast = model.predict(future)
                    return forecast["yhat"].values.tolist()
        except Exception as e:
            logger.error(f"Failed to get predictions for {method}: {e}")
            return []
    
    def _calculate_performance(
        self,
        actual: List[float],
        predicted: List[float]
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        actual = np.array(actual)
        predicted = np.array(predicted[:len(actual)])
        
        return {
            "mse": float(mean_squared_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)),
            "explained_variance": float(explained_variance_score(actual, predicted)),
            "mean_error": float(np.mean(np.abs(actual - predicted))),
            "max_error": float(np.max(np.abs(actual - predicted))),
            "error_std": float(np.std(actual - predicted))
        }
    
    def _compare_interpretations(
        self,
        metric: str,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Compare model interpretations."""
        explanations = self.interpreter.explain_predictions(metrics)
        if not explanations or metric not in explanations:
            return {}
        
        metric_exp = explanations[metric]
        
        # Compare feature importance stability
        stability = self._analyze_importance_stability(
            metric,
            metric_exp["global"]["importance"]
        )
        
        # Compare local vs global explanations
        local_global_agreement = self._analyze_explanation_agreement(
            metric_exp["global"],
            metric_exp["local"]
        )
        
        return {
            "feature_stability": stability,
            "local_global_agreement": local_global_agreement,
            "temporal_consistency": self._analyze_temporal_consistency(
                metric,
                metric_exp["temporal"]
            )
        }
    
    def _analyze_importance_stability(
        self,
        metric: str,
        importance: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze feature importance stability."""
        if metric not in self.comparison_results:
            return {"stability_score": 1.0}
        
        previous = self.comparison_results[metric].get(
            "interpretability", {}
        ).get("feature_stability", {})
        
        if not previous:
            return {"stability_score": 1.0}
        
        # Calculate rank correlation of feature importance
        current_ranks = pd.Series(importance).rank()
        previous_ranks = pd.Series(previous).rank()
        
        stability = current_ranks.corr(previous_ranks)
        
        return {
            "stability_score": float(stability),
            "rank_changes": [
                {
                    "feature": feat,
                    "rank_change": int(current_ranks[feat] - previous_ranks[feat])
                }
                for feat in importance
                if feat in previous_ranks
            ]
        }
    
    def _analyze_explanation_agreement(
        self,
        global_exp: Dict[str, Any],
        local_exp: Dict[str, Any]
    ) -> Dict[str, float]:
        """Analyze agreement between local and global explanations."""
        global_importance = pd.Series(global_exp["importance"])
        local_importance = pd.Series(local_exp["shap"])
        
        correlation = global_importance.corr(local_importance)
        rank_correlation = global_importance.rank().corr(local_importance.rank())
        
        return {
            "correlation": float(correlation),
            "rank_correlation": float(rank_correlation),
            "top_feature_overlap": float(
                len(set(global_exp["summary"]["top_features"]) &
                    set(sorted(local_exp["lime"].keys())[:3])) / 3
            )
        }
    
    def _analyze_temporal_consistency(
        self,
        metric: str,
        temporal_data: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Analyze temporal consistency of explanations."""
        if not temporal_data:
            return {}
        
        feature_stability = {}
        for feature, values in temporal_data.items():
            importance_values = [v["importance"] for v in values]
            feature_stability[feature] = float(np.std(importance_values))
        
        return {
            "mean_stability": float(np.mean(list(feature_stability.values()))),
            "feature_stability": feature_stability
        }
    
    def _update_comparison_history(
        self,
        results: Dict[str, Any]
    ):
        """Update comparison history."""
        self.comparison_results = results
        
        # Add temporal performance
        current_time = datetime.now()
        for metric, metric_results in results.items():
            performance = {
                "timestamp": current_time.isoformat(),
                "metric": metric,
                "performance": {}
            }
            
            for period, methods in metric_results.items():
                if isinstance(methods, dict):  # Skip interpretability
                    for method, metrics in methods.items():
                        performance["performance"][f"{period}_{method}"] = metrics
            
            self.temporal_performance.append(performance)
            
            # Trim history
            if len(self.temporal_performance) > self.config.min_samples:
                self.temporal_performance = \
                    self.temporal_performance[-self.config.min_samples:]
    
    def _update_model_rankings(
        self,
        results: Dict[str, Any]
    ):
        """Update model rankings."""
        rankings = defaultdict(list)
        
        for metric, metric_results in results.items():
            for period, methods in metric_results.items():
                if isinstance(methods, dict):  # Skip interpretability
                    period_rankings = []
                    for method, metrics in methods.items():
                        period_rankings.append({
                            "method": method,
                            "score": 1 - metrics["mse"],  # Higher is better
                            "metrics": metrics
                        })
                    
                    # Sort by score
                    period_rankings.sort(key=lambda x: x["score"], reverse=True)
                    rankings[f"{metric}_{period}"] = period_rankings
        
        self.model_rankings = dict(rankings)
    
    def plot_comparisons(self) -> Dict[str, go.Figure]:
        """Create comparison visualizations."""
        plots = {}
        
        # Plot performance comparison
        for metric in self.config.metrics_to_compare:
            fig = make_subplots(
                rows=2,
                cols=1,
                subplot_titles=(
                    f"{metric} - Prediction Error",
                    f"{metric} - Model Rankings"
                )
            )
            
            # Add error plot
            temporal = pd.DataFrame(self.temporal_performance)
            temporal = temporal[temporal["metric"] == metric]
            
            for period in self.config.time_periods:
                for method in self.config.comparison_methods:
                    values = [
                        p["performance"][f"{period}_{method}"]["rmse"]
                        for p in temporal.to_dict("records")
                    ]
                    timestamps = [
                        datetime.fromisoformat(p["timestamp"])
                        for p in temporal.to_dict("records")
                    ]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps,
                            y=values,
                            name=f"{period}_{method}",
                            mode="lines"
                        ),
                        row=1,
                        col=1
                    )
            
            # Add rankings plot
            rankings = self.model_rankings.get(f"{metric}_{str(self.config.time_periods[0])}", [])
            if rankings:
                fig.add_trace(
                    go.Bar(
                        x=[r["method"] for r in rankings],
                        y=[r["score"] for r in rankings],
                        name="Model Score"
                    ),
                    row=2,
                    col=1
                )
            
            fig.update_layout(height=800)
            plots[f"{metric}_comparison"] = fig
        
        return plots
    
    def _save_comparison_results(self, results: Dict[str, Any]):
        """Save comparison results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_path / "comparison_results.json", "w") as f:
                json.dump({
                    "results": results,
                    "rankings": self.model_rankings,
                    "temporal_performance": self.temporal_performance
                }, f, indent=2)
            
            # Save visualizations
            plots = self.plot_comparisons()
            for name, fig in plots.items():
                fig.write_html(str(output_path / f"{name}.html"))
            
            logger.info(f"Saved comparison results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save comparison results: {e}")

def create_comparator(
    predictor: PerformancePredictor,
    interpreter: PredictionInterpreter,
    output_path: Optional[Path] = None
) -> PredictionComparator:
    """Create prediction comparator."""
    config = ComparisonConfig(output_path=output_path)
    return PredictionComparator(predictor, interpreter, config)

if __name__ == "__main__":
    # Example usage
    from .performance_prediction import create_predictor
    from .prediction_interpretation import create_interpreter
    from .performance_metrics import monitor_performance
    
    # Create components
    monitor = monitor_performance()
    predictor = create_predictor(monitor.config)
    interpreter = create_interpreter(predictor)
    comparator = create_comparator(
        predictor,
        interpreter,
        output_path=Path("comparisons")
    )
    
    # Simulate metrics collection
    for _ in range(1000):
        time.sleep(0.1)
        monitor.record_frame(time.perf_counter())
    
    # Compare predictions
    comparator.compare_predictions(monitor.metrics)
