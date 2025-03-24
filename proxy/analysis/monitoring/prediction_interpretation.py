"""Model interpretation for performance predictions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
import shap
import lime
import lime.lime_tabular
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import logging
from pathlib import Path
import json
from collections import defaultdict
import eli5
from eli5.formatters import format_as_html, format_as_text

from .performance_prediction import PerformancePredictor, PredictionConfig
from .performance_metrics import PerformanceMetric

logger = logging.getLogger(__name__)

@dataclass
class InterpretationConfig:
    """Configuration for prediction interpretation."""
    num_features: int = 10
    num_samples: int = 1000
    kernel_width: float = 0.75
    feature_selection: str = "auto"
    output_path: Optional[Path] = None
    save_explanations: bool = True
    min_importance: float = 0.01
    max_display: int = 20
    local_radius: float = 0.2

class PredictionInterpreter:
    """Interpret performance predictions."""
    
    def __init__(
        self,
        predictor: PerformancePredictor,
        config: InterpretationConfig
    ):
        self.predictor = predictor
        self.config = config
        
        self.explainers: Dict[str, Any] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        self.local_explanations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize SHAP explainers
        self._initialize_explainers()
    
    def _initialize_explainers(self):
        """Initialize model explainers."""
        for metric, model_dict in self.predictor.models.items():
            rf_model = model_dict["rf"]
            X_background = self.predictor.scaler.transform(
                pd.DataFrame(columns=model_dict["features"]).head(100)
            )
            
            try:
                # Create SHAP explainer
                shap_explainer = shap.TreeExplainer(rf_model)
                
                # Create LIME explainer
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_background,
                    feature_names=model_dict["features"],
                    kernel_width=self.config.kernel_width,
                    mode="regression"
                )
                
                self.explainers[metric] = {
                    "shap": shap_explainer,
                    "lime": lime_explainer
                }
                
            except Exception as e:
                logger.error(f"Failed to initialize explainers for {metric}: {e}")
    
    def explain_predictions(
        self,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Generate explanations for predictions."""
        if not metrics:
            return {}
        
        explanations = {}
        
        try:
            # Prepare data
            df = self.predictor._prepare_training_data(metrics)
            
            for metric in ["fps", "memory_mb", "cpu_percent"]:
                if metric in self.predictor.models:
                    # Get model and features
                    model_dict = self.predictor.models[metric]
                    features = model_dict["features"]
                    X = df[features].values
                    X_scaled = self.predictor.scaler.transform(X)
                    
                    # Generate explanations
                    global_explanation = self._explain_global(
                        metric,
                        X_scaled,
                        features
                    )
                    local_explanation = self._explain_local(
                        metric,
                        X_scaled[-1:],
                        features
                    )
                    temporal_importance = self._analyze_temporal_importance(
                        metric,
                        df,
                        features
                    )
                    
                    explanations[metric] = {
                        "global": global_explanation,
                        "local": local_explanation,
                        "temporal": temporal_importance
                    }
            
            # Store explanations
            if self.config.save_explanations:
                self._save_explanations(explanations)
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate explanations: {e}")
            return {}
    
    def _explain_global(
        self,
        metric: str,
        X: np.ndarray,
        features: List[str]
    ) -> Dict[str, Any]:
        """Generate global feature importance."""
        if metric not in self.explainers:
            return {}
        
        try:
            # Get SHAP values
            shap_values = self.explainers[metric]["shap"].shap_values(X)
            
            # Calculate feature importance
            importance = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(features, importance))
            
            # Sort features by importance
            sorted_features = sorted(
                importance_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            
            # Keep top features
            top_features = sorted_features[:self.config.num_features]
            
            # Store feature importance
            self.feature_importance[metric] = dict(top_features)
            
            return {
                "importance": dict(top_features),
                "interactions": self._get_feature_interactions(
                    metric,
                    X,
                    features
                ),
                "summary": self._get_importance_summary(
                    metric,
                    top_features
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to generate global explanation for {metric}: {e}")
            return {}
    
    def _explain_local(
        self,
        metric: str,
        X: np.ndarray,
        features: List[str]
    ) -> Dict[str, Any]:
        """Generate local explanation for prediction."""
        if metric not in self.explainers:
            return {}
        
        try:
            # Get LIME explanation
            lime_exp = self.explainers[metric]["lime"].explain_instance(
                X[0],
                self.predictor.models[metric]["rf"].predict,
                num_features=self.config.num_features
            )
            
            # Get SHAP values for instance
            shap_values = self.explainers[metric]["shap"].shap_values(X)[0]
            
            # Combine explanations
            explanation = {
                "lime": {
                    feat: weight
                    for feat, weight in lime_exp.as_list()
                },
                "shap": dict(zip(features, shap_values)),
                "feature_values": dict(zip(features, X[0])),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store local explanation
            self.local_explanations[metric].append(explanation)
            if len(self.local_explanations[metric]) > self.config.num_samples:
                self.local_explanations[metric].pop(0)
            
            return explanation
            
        except Exception as e:
            logger.error(f"Failed to generate local explanation for {metric}: {e}")
            return {}
    
    def _get_feature_interactions(
        self,
        metric: str,
        X: np.ndarray,
        features: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze feature interactions."""
        try:
            # Calculate SHAP interaction values
            shap_interaction = self.explainers[metric]["shap"].shap_interaction_values(X)
            
            # Get mean absolute interaction values
            interaction_values = np.abs(shap_interaction).mean(axis=0)
            
            # Find top interactions
            interactions = []
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    if interaction_values[i, j] > self.config.min_importance:
                        interactions.append({
                            "features": [features[i], features[j]],
                            "importance": float(interaction_values[i, j])
                        })
            
            # Sort by importance
            interactions.sort(key=lambda x: x["importance"], reverse=True)
            
            return {
                "interactions": interactions[:self.config.max_display],
                "total_interactions": len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze interactions for {metric}: {e}")
            return {}
    
    def _analyze_temporal_importance(
        self,
        metric: str,
        df: pd.DataFrame,
        features: List[str]
    ) -> Dict[str, Any]:
        """Analyze feature importance over time."""
        try:
            # Create time windows
            windows = pd.TimeGrouper(freq="1H")
            temporal_importance = defaultdict(list)
            
            for name, group in df.groupby(windows):
                if len(group) < self.config.num_samples:
                    continue
                
                X = group[features].values
                X_scaled = self.predictor.scaler.transform(X)
                
                # Calculate SHAP values for window
                shap_values = self.explainers[metric]["shap"].shap_values(X_scaled)
                importance = np.abs(shap_values).mean(axis=0)
                
                for feat, imp in zip(features, importance):
                    temporal_importance[feat].append({
                        "timestamp": name.isoformat(),
                        "importance": float(imp)
                    })
            
            return dict(temporal_importance)
            
        except Exception as e:
            logger.error(f"Failed to analyze temporal importance for {metric}: {e}")
            return {}
    
    def _get_importance_summary(
        self,
        metric: str,
        features: List[Tuple[str, float]]
    ) -> Dict[str, Any]:
        """Create feature importance summary."""
        total_importance = sum(abs(imp) for _, imp in features)
        
        return {
            "top_features": [feat for feat, _ in features[:3]],
            "importance_distribution": {
                feat: float(imp / total_importance)
                for feat, imp in features
            },
            "cumulative_importance": [
                {
                    "features": feat,
                    "cumulative": float(
                        sum(abs(imp) for _, imp in features[:i+1]) / total_importance
                    )
                }
                for i, (feat, _) in enumerate(features)
            ]
        }
    
    def plot_interpretations(self) -> Dict[str, go.Figure]:
        """Create interpretation visualizations."""
        plots = {}
        
        # Plot global importance
        for metric, importance in self.feature_importance.items():
            fig = go.Figure()
            
            # Sort features by importance
            sorted_features = sorted(
                importance.items(),
                key=lambda x: abs(x[1])
            )
            features = [f[0] for f in sorted_features]
            values = [f[1] for f in sorted_features]
            
            fig.add_trace(go.Bar(
                y=features,
                x=values,
                orientation="h",
                name="Feature Importance"
            ))
            
            fig.update_layout(
                title=f"Feature Importance - {metric}",
                xaxis_title="SHAP Value",
                yaxis_title="Feature",
                height=max(400, len(features) * 20)
            )
            
            plots[f"{metric}_importance"] = fig
        
        # Plot temporal importance
        for metric in self.feature_importance:
            local_exp = self.local_explanations[metric]
            if not local_exp:
                continue
            
            fig = go.Figure()
            
            timestamps = [
                datetime.fromisoformat(exp["timestamp"])
                for exp in local_exp
            ]
            
            for feature in self.feature_importance[metric]:
                values = [
                    exp["shap"].get(feature, 0)
                    for exp in local_exp
                ]
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=values,
                    name=feature,
                    mode="lines"
                ))
            
            fig.update_layout(
                title=f"Temporal Feature Importance - {metric}",
                xaxis_title="Time",
                yaxis_title="SHAP Value",
                height=500
            )
            
            plots[f"{metric}_temporal"] = fig
        
        return plots
    
    def _save_explanations(self, explanations: Dict[str, Any]):
        """Save explanations to file."""
        if not self.config.output_path:
            return
        
        try:
            output_path = Path(self.config.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save explanations
            with open(output_path / "explanations.json", "w") as f:
                json.dump(explanations, f, indent=2)
            
            # Save visualizations
            plots = self.plot_interpretations()
            for name, fig in plots.items():
                fig.write_html(str(output_path / f"{name}.html"))
            
            logger.info(f"Saved explanations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save explanations: {e}")

def create_interpreter(
    predictor: PerformancePredictor,
    output_path: Optional[Path] = None
) -> PredictionInterpreter:
    """Create prediction interpreter."""
    config = InterpretationConfig(output_path=output_path)
    return PredictionInterpreter(predictor, config)

if __name__ == "__main__":
    # Example usage
    from .performance_prediction import create_predictor
    from .performance_metrics import monitor_performance
    
    # Create components
    monitor = monitor_performance()
    predictor = create_predictor(
        monitor.config,
        model_path=Path("models")
    )
    interpreter = create_interpreter(
        predictor,
        output_path=Path("explanations")
    )
    
    # Simulate metrics collection
    for _ in range(1000):
        time.sleep(0.1)
        monitor.record_frame(time.perf_counter())
    
    # Generate and save explanations
    predictor.update_models(monitor.metrics)
    explanations = interpreter.explain_predictions(monitor.metrics)
    
    print(json.dumps(explanations, indent=2))
