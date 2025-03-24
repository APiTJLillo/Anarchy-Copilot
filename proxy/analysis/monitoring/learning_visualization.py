"""Visualization tools for optimization learning."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

from .optimization_learning import OptimizationLearner, LearningConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for learning visualization."""
    width: int = 1200
    height: int = 800
    dark_mode: bool = False
    show_grid: bool = True
    show_legend: bool = True
    interactive: bool = True
    output_path: Optional[Path] = None

class LearningVisualizer:
    """Visualize optimization learning process."""
    
    def __init__(
        self,
        learner: OptimizationLearner,
        config: VisualizationConfig
    ):
        self.learner = learner
        self.config = config
    
    def create_learning_dashboard(self) -> go.Figure:
        """Create comprehensive learning dashboard."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Impact Prediction Performance",
                "Success Rate Over Time",
                "Feature Importances",
                "Model Confidence",
                "Priority Distribution",
                "Learning Progress"
            ]
        )
        
        # Add impact prediction performance
        impact_data = self._get_impact_prediction_data()
        fig.add_trace(
            go.Scatter(
                x=impact_data["actual"],
                y=impact_data["predicted"],
                mode="markers",
                name="Impact Predictions"
            ),
            row=1,
            col=1
        )
        
        # Add success rate over time
        success_data = self._get_success_rate_data()
        fig.add_trace(
            go.Scatter(
                x=success_data["time"],
                y=success_data["rate"],
                mode="lines",
                name="Success Rate"
            ),
            row=1,
            col=2
        )
        
        # Add feature importances
        importance_data = self._get_feature_importance_data()
        fig.add_trace(
            go.Bar(
                x=importance_data["features"],
                y=importance_data["importance"],
                name="Feature Importance"
            ),
            row=2,
            col=1
        )
        
        # Add model confidence
        confidence_data = self._get_model_confidence_data()
        fig.add_trace(
            go.Histogram(
                x=confidence_data["confidence"],
                name="Model Confidence"
            ),
            row=2,
            col=2
        )
        
        # Add priority distribution
        priority_data = self._get_priority_distribution_data()
        fig.add_trace(
            go.Pie(
                labels=priority_data["priority"],
                values=priority_data["count"],
                name="Priority Distribution"
            ),
            row=3,
            col=1
        )
        
        # Add learning progress
        progress_data = self._get_learning_progress_data()
        fig.add_trace(
            go.Scatter(
                x=progress_data["samples"],
                y=progress_data["accuracy"],
                mode="lines",
                name="Learning Progress"
            ),
            row=3,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=self.config.show_legend,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_impact_analysis(self) -> go.Figure:
        """Create impact prediction analysis visualization."""
        # Get prediction data
        data = self._get_impact_prediction_data()
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Prediction vs Actual",
                "Error Distribution",
                "Impact Over Time",
                "Residual Plot"
            ]
        )
        
        # Prediction vs Actual
        fig.add_trace(
            go.Scatter(
                x=data["actual"],
                y=data["predicted"],
                mode="markers",
                name="Predictions"
            ),
            row=1,
            col=1
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[min(data["actual"]), max(data["actual"])],
                y=[min(data["actual"]), max(data["actual"])],
                mode="lines",
                name="Perfect Prediction",
                line=dict(dash="dash")
            ),
            row=1,
            col=1
        )
        
        # Error Distribution
        errors = data["predicted"] - data["actual"]
        fig.add_trace(
            go.Histogram(
                x=errors,
                name="Prediction Error"
            ),
            row=1,
            col=2
        )
        
        # Impact Over Time
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["actual"],
                mode="lines+markers",
                name="Actual Impact"
            ),
            row=2,
            col=1
        )
        
        # Residual Plot
        fig.add_trace(
            go.Scatter(
                x=data["predicted"],
                y=errors,
                mode="markers",
                name="Residuals"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=self.config.show_legend,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_feature_analysis(self) -> go.Figure:
        """Create feature importance analysis visualization."""
        importance_data = self._get_feature_importance_data()
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Feature Importances",
                "Feature Correlations",
                "Feature Impact on Success",
                "Feature Stability"
            ]
        )
        
        # Feature Importances
        fig.add_trace(
            go.Bar(
                x=importance_data["features"],
                y=importance_data["importance"],
                name="Importance"
            ),
            row=1,
            col=1
        )
        
        # Feature Correlations
        correlations = self._get_feature_correlations()
        fig.add_trace(
            go.Heatmap(
                z=correlations["matrix"],
                x=correlations["features"],
                y=correlations["features"],
                colorscale="RdBu",
                name="Correlations"
            ),
            row=1,
            col=2
        )
        
        # Feature Impact on Success
        success_impact = self._get_feature_success_impact()
        fig.add_trace(
            go.Bar(
                x=success_impact["features"],
                y=success_impact["impact"],
                name="Success Impact"
            ),
            row=2,
            col=1
        )
        
        # Feature Stability
        stability = self._get_feature_stability()
        fig.add_trace(
            go.Scatter(
                x=stability["time"],
                y=stability["stability"],
                mode="lines",
                name="Stability"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=self.config.show_legend,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def create_success_analysis(self) -> go.Figure:
        """Create success prediction analysis visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Success Rate Over Time",
                "ROC Curve",
                "Confusion Matrix",
                "Confidence Distribution"
            ]
        )
        
        # Success Rate
        success_data = self._get_success_rate_data()
        fig.add_trace(
            go.Scatter(
                x=success_data["time"],
                y=success_data["rate"],
                mode="lines",
                name="Success Rate"
            ),
            row=1,
            col=1
        )
        
        # ROC Curve
        roc_data = self._get_roc_curve_data()
        fig.add_trace(
            go.Scatter(
                x=roc_data["fpr"],
                y=roc_data["tpr"],
                mode="lines",
                name=f"ROC (AUC = {roc_data['auc']:.2f})"
            ),
            row=1,
            col=2
        )
        
        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash"),
                name="Random"
            ),
            row=1,
            col=2
        )
        
        # Confusion Matrix
        confusion = self._get_confusion_matrix_data()
        fig.add_trace(
            go.Heatmap(
                z=confusion["matrix"],
                x=["Predicted 0", "Predicted 1"],
                y=["Actual 0", "Actual 1"],
                colorscale="Viridis",
                name="Confusion Matrix"
            ),
            row=2,
            col=1
        )
        
        # Confidence Distribution
        confidence_data = self._get_model_confidence_data()
        fig.add_trace(
            go.Histogram(
                x=confidence_data["confidence"],
                name="Model Confidence"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            showlegend=self.config.show_legend,
            template="plotly_dark" if self.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def _get_impact_prediction_data(self) -> Dict[str, np.ndarray]:
        """Get impact prediction performance data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "actual": np.array([]),
                "predicted": np.array([]),
                "time": np.array([])
            }
        
        actual = []
        predicted = []
        times = []
        
        for entry in feedback_data:
            actual.append(entry["actual_impact"])
            predicted.append(entry["original_impact"])
            times.append(datetime.fromisoformat(entry["timestamp"]))
        
        return {
            "actual": np.array(actual),
            "predicted": np.array(predicted),
            "time": np.array(times)
        }
    
    def _get_success_rate_data(self) -> Dict[str, np.ndarray]:
        """Get success rate over time data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "time": np.array([]),
                "rate": np.array([])
            }
        
        times = []
        success_rate = []
        window = 10
        
        for i, entry in enumerate(feedback_data):
            times.append(datetime.fromisoformat(entry["timestamp"]))
            successes = sum(
                1 for j in range(max(0, i - window + 1), i + 1)
                if feedback_data[j]["success"]
            )
            rate = successes / min(window, i + 1)
            success_rate.append(rate)
        
        return {
            "time": np.array(times),
            "rate": np.array(success_rate)
        }
    
    def _get_feature_importance_data(self) -> Dict[str, List]:
        """Get feature importance data."""
        importances = self.learner.feature_importances
        if not importances:
            return {
                "features": [],
                "importance": []
            }
        
        features = []
        importance = []
        
        for feature, value in sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            features.append(feature)
            importance.append(value)
        
        return {
            "features": features,
            "importance": importance
        }
    
    def _get_model_confidence_data(self) -> Dict[str, np.ndarray]:
        """Get model confidence distribution data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {"confidence": np.array([])}
        
        # Use recent predictions only
        recent_data = feedback_data[-100:]
        X = np.array([
            self.learner._extract_features(
                OptimizationSuggestion(
                    type=entry["suggestion_type"],
                    description="",
                    impact=entry["original_impact"],
                    complexity=1,
                    priority=entry["priority"]
                ),
                [],
                entry["features"]
            )
            for entry in recent_data
        ])
        
        try:
            proba = self.learner.success_model.predict_proba(X)
            confidence = np.max(proba, axis=1)
            return {"confidence": confidence}
        except:
            return {"confidence": np.array([])}
    
    def _get_priority_distribution_data(self) -> Dict[str, List]:
        """Get priority distribution data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "priority": [],
                "count": []
            }
        
        counts = defaultdict(int)
        for entry in feedback_data:
            counts[entry["priority"]] += 1
        
        return {
            "priority": list(counts.keys()),
            "count": list(counts.values())
        }
    
    def _get_learning_progress_data(self) -> Dict[str, np.ndarray]:
        """Get learning progress data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "samples": np.array([]),
                "accuracy": np.array([])
            }
        
        samples = []
        accuracy = []
        window = 20
        
        for i in range(len(feedback_data)):
            samples.append(i + 1)
            window_data = feedback_data[max(0, i - window + 1):i + 1]
            correct = sum(
                1 for entry in window_data
                if entry["success"] == (entry["actual_impact"] > 0.5)
            )
            accuracy.append(correct / len(window_data))
        
        return {
            "samples": np.array(samples),
            "accuracy": np.array(accuracy)
        }
    
    def _get_feature_correlations(self) -> Dict[str, Any]:
        """Get feature correlation data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "matrix": np.array([[]]),
                "features": []
            }
        
        features = []
        data = []
        
        for entry in feedback_data:
            features = list(entry["features"].keys())
            data.append(list(entry["features"].values()))
        
        if not data:
            return {
                "matrix": np.array([[]]),
                "features": []
            }
        
        correlation_matrix = np.corrcoef(np.array(data).T)
        
        return {
            "matrix": correlation_matrix,
            "features": features
        }
    
    def _get_feature_success_impact(self) -> Dict[str, List]:
        """Get feature impact on success data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "features": [],
                "impact": []
            }
        
        feature_success = defaultdict(list)
        
        for entry in feedback_data:
            for feature, value in entry["features"].items():
                feature_success[feature].append(
                    (float(value), entry["success"])
                )
        
        features = []
        impact = []
        
        for feature, values in feature_success.items():
            features.append(feature)
            success_rate = sum(
                1 for _, success in values if success
            ) / len(values)
            impact.append(success_rate)
        
        return {
            "features": features,
            "impact": impact
        }
    
    def _get_feature_stability(self) -> Dict[str, np.ndarray]:
        """Get feature stability over time data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "time": np.array([]),
                "stability": np.array([])
            }
        
        times = []
        stability = []
        window = 20
        
        for i in range(len(feedback_data)):
            times.append(
                datetime.fromisoformat(feedback_data[i]["timestamp"])
            )
            
            if i < window:
                stability.append(1.0)
                continue
            
            # Calculate feature value stability
            window_data = feedback_data[i - window:i]
            feature_values = defaultdict(list)
            
            for entry in window_data:
                for feature, value in entry["features"].items():
                    feature_values[feature].append(float(value))
            
            # Calculate average coefficient of variation
            cvs = []
            for values in feature_values.values():
                if np.mean(values) != 0:
                    cv = np.std(values) / np.mean(values)
                    cvs.append(cv)
            
            stability.append(1 / (1 + np.mean(cvs)))
        
        return {
            "time": np.array(times),
            "stability": np.array(stability)
        }
    
    def _get_roc_curve_data(self) -> Dict[str, np.ndarray]:
        """Get ROC curve data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "fpr": np.array([]),
                "tpr": np.array([]),
                "auc": 0.0
            }
        
        try:
            X = np.array([
                self.learner._extract_features(
                    OptimizationSuggestion(
                        type=entry["suggestion_type"],
                        description="",
                        impact=entry["original_impact"],
                        complexity=1,
                        priority=entry["priority"]
                    ),
                    [],
                    entry["features"]
                )
                for entry in feedback_data
            ])
            
            y = np.array([entry["success"] for entry in feedback_data])
            
            y_score = self.learner.success_model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_score)
            roc_auc = auc(fpr, tpr)
            
            return {
                "fpr": fpr,
                "tpr": tpr,
                "auc": roc_auc
            }
        except:
            return {
                "fpr": np.array([]),
                "tpr": np.array([]),
                "auc": 0.0
            }
    
    def _get_confusion_matrix_data(self) -> Dict[str, np.ndarray]:
        """Get confusion matrix data."""
        feedback_data = self.learner._load_feedback_data()
        if not feedback_data:
            return {
                "matrix": np.array([[0, 0], [0, 0]])
            }
        
        try:
            X = np.array([
                self.learner._extract_features(
                    OptimizationSuggestion(
                        type=entry["suggestion_type"],
                        description="",
                        impact=entry["original_impact"],
                        complexity=1,
                        priority=entry["priority"]
                    ),
                    [],
                    entry["features"]
                )
                for entry in feedback_data
            ])
            
            y = np.array([entry["success"] for entry in feedback_data])
            
            y_pred = self.learner.success_model.predict(X)
            cm = confusion_matrix(y, y_pred)
            
            return {"matrix": cm}
        except:
            return {
                "matrix": np.array([[0, 0], [0, 0]])
            }
    
    def save_visualizations(
        self,
        output_path: Optional[Path] = None
    ):
        """Save all visualizations."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save dashboard
            dashboard = self.create_learning_dashboard()
            dashboard.write_html(str(path / "learning_dashboard.html"))
            
            # Save impact analysis
            impact = self.create_impact_analysis()
            impact.write_html(str(path / "impact_analysis.html"))
            
            # Save feature analysis
            feature = self.create_feature_analysis()
            feature.write_html(str(path / "feature_analysis.html"))
            
            # Save success analysis
            success = self.create_success_analysis()
            success.write_html(str(path / "success_analysis.html"))
            
            logger.info(f"Saved visualizations to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_learning_visualizer(
    learner: OptimizationLearner,
    output_path: Optional[Path] = None
) -> LearningVisualizer:
    """Create learning visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return LearningVisualizer(learner, config)

if __name__ == "__main__":
    # Example usage
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(
        learner,
        output_path=Path("learning_visualization")
    )
    
    # Generate visualizations
    viz.save_visualizations()
