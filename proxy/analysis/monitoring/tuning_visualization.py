"""Visualization tools for hyperparameter tuning results."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate
from optuna.visualization import plot_slice, plot_contour

from .hyperparameter_tuning import HyperparameterTuner, TuningConfig

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for tuning visualization."""
    width: int = 1200
    height: int = 800
    dark_mode: bool = False
    interactive: bool = True
    show_history: bool = True
    show_importance: bool = True
    plot_3d: bool = True
    output_path: Optional[Path] = None

class TuningVisualizer:
    """Visualize hyperparameter tuning results."""
    
    def __init__(
        self,
        tuner: HyperparameterTuner,
        config: VisualizationConfig
    ):
        self.tuner = tuner
        self.config = config
    
    def create_dashboard(self) -> go.Figure:
        """Create comprehensive tuning visualization dashboard."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Optimization History",
                "Parameter Importance",
                "Parallel Coordinates",
                "Parameter Relationships",
                "Learning Curves",
                "Model Comparison"
            ],
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "bar"}]
            ]
        )
        
        # Add optimization history
        self._add_optimization_history(fig, row=1, col=1)
        
        # Add parameter importance
        self._add_parameter_importance(fig, row=1, col=2)
        
        # Add parallel coordinates
        self._add_parallel_coordinates(fig, row=2, col=1)
        
        # Add parameter relationships
        self._add_parameter_relationships(fig, row=2, col=2)
        
        # Add learning curves
        self._add_learning_curves(fig, row=3, col=1)
        
        # Add model comparison
        self._add_model_comparison(fig, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def create_study_visualization(
        self,
        model_name: str
    ) -> Dict[str, go.Figure]:
        """Create detailed study visualizations."""
        if model_name not in self.tuner.study_results:
            return {}
        
        study = self.tuner.study_results[model_name]
        
        return {
            "history": self._create_optuna_history(study),
            "parallel": self._create_optuna_parallel(study),
            "slices": self._create_optuna_slices(study),
            "contour": self._create_optuna_contour(study)
        }
    
    def create_parameter_analysis(
        self,
        model_name: str
    ) -> go.Figure:
        """Create detailed parameter analysis visualization."""
        if model_name not in self.tuner.best_params:
            return go.Figure()
        
        params = self.tuner.best_params[model_name]
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Parameter Values",
                "Parameter Correlations",
                "Parameter Distributions",
                "Parameter Sensitivity"
            ]
        )
        
        # Add parameter values
        self._add_parameter_values(fig, params, row=1, col=1)
        
        # Add parameter correlations
        self._add_parameter_correlations(fig, model_name, row=1, col=2)
        
        # Add parameter distributions
        self._add_parameter_distributions(fig, model_name, row=2, col=1)
        
        # Add parameter sensitivity
        self._add_parameter_sensitivity(fig, model_name, row=2, col=2)
        
        return fig
    
    def _add_optimization_history(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add optimization history subplot."""
        for model_name, study in self.tuner.study_results.items():
            values = [t.value for t in study.trials if t.value is not None]
            best_values = np.minimum.accumulate(values)
            
            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    mode="markers",
                    name=f"{model_name} Trials",
                    marker=dict(size=6),
                    showlegend=True
                ),
                row=row,
                col=col
            )
            
            # Add best values line
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(best_values))),
                    y=best_values,
                    mode="lines",
                    name=f"{model_name} Best",
                    line=dict(width=2)
                ),
                row=row,
                col=col
            )
        
        fig.update_xaxes(title_text="Trial", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_parameter_importance(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add parameter importance subplot."""
        if not self.tuner.study_results:
            return
        
        for model_name, study in self.tuner.study_results.items():
            importance = optuna.importance.get_param_importances(study)
            
            fig.add_trace(
                go.Bar(
                    x=list(importance.keys()),
                    y=list(importance.values()),
                    name=model_name
                ),
                row=row,
                col=col
            )
        
        fig.update_xaxes(title_text="Parameter", row=row, col=col)
        fig.update_yaxes(title_text="Importance", row=row, col=col)
    
    def _add_parallel_coordinates(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add parallel coordinates subplot."""
        if not self.tuner.study_results:
            return
        
        for model_name, study in self.tuner.study_results.items():
            trials_df = study.trials_dataframe()
            
            params = [c for c in trials_df.columns if c.startswith("params_")]
            if not params:
                continue
                
            fig.add_trace(
                go.Parcoords(
                    line=dict(
                        color=trials_df["value"],
                        colorscale="Viridis"
                    ),
                    dimensions=[
                        dict(
                            range=[trials_df[p].min(), trials_df[p].max()],
                            label=p.replace("params_", ""),
                            values=trials_df[p]
                        )
                        for p in params
                    ]
                ),
                row=row,
                col=col
            )
    
    def _add_parameter_relationships(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add parameter relationships subplot."""
        if not self.tuner.study_results:
            return
        
        for model_name, study in self.tuner.study_results.items():
            trials_df = study.trials_dataframe()
            params = [c for c in trials_df.columns if c.startswith("params_")]
            
            if len(params) < 2:
                continue
            
            # Create correlation matrix
            corr_matrix = trials_df[params].corr()
            
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.index,
                    y=corr_matrix.columns,
                    colorscale="RdBu",
                    name=model_name
                ),
                row=row,
                col=col
            )
    
    def _add_learning_curves(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add learning curves subplot."""
        if not self.tuner.study_results:
            return
        
        for model_name, study in self.tuner.study_results.items():
            trials_df = study.trials_dataframe()
            
            if "intermediate_values" not in trials_df.columns:
                continue
            
            # Extract learning curves
            curves = []
            for values in trials_df["intermediate_values"]:
                if isinstance(values, dict):
                    curves.append(list(values.values()))
            
            if not curves:
                continue
            
            # Calculate mean and std
            curves = np.array(curves)
            mean_curve = np.mean(curves, axis=0)
            std_curve = np.std(curves, axis=0)
            steps = list(range(len(mean_curve)))
            
            # Add mean curve
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=mean_curve,
                    mode="lines",
                    name=f"{model_name} Mean",
                    line=dict(width=2)
                ),
                row=row,
                col=col
            )
            
            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=steps + steps[::-1],
                    y=np.concatenate([
                        mean_curve + std_curve,
                        (mean_curve - std_curve)[::-1]
                    ]),
                    fill="toself",
                    fillcolor=f"rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{model_name} Std",
                    showlegend=False
                ),
                row=row,
                col=col
            )
        
        fig.update_xaxes(title_text="Step", row=row, col=col)
        fig.update_yaxes(title_text="Value", row=row, col=col)
    
    def _add_model_comparison(
        self,
        fig: go.Figure,
        row: int,
        col: int
    ):
        """Add model comparison subplot."""
        if not self.tuner.best_params:
            return
        
        models = []
        scores = []
        times = []
        
        for model_name, study in self.tuner.study_results.items():
            models.append(model_name)
            scores.append(study.best_value)
            times.append(sum(
                t.duration.total_seconds()
                for t in study.trials
                if t.duration
            ))
        
        # Add scores
        fig.add_trace(
            go.Bar(
                x=models,
                y=scores,
                name="Best Score",
                marker_color="blue"
            ),
            row=row,
            col=col
        )
        
        # Add times on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=models,
                y=times,
                name="Total Time",
                mode="lines+markers",
                yaxis="y2",
                marker_color="red"
            ),
            row=row,
            col=col
        )
        
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Score", row=row, col=col)
        fig.update_yaxes(
            title_text="Time (s)",
            overlaying="y",
            side="right",
            row=row,
            col=col
        )
    
    def _create_optuna_history(
        self,
        study: Any
    ) -> go.Figure:
        """Create Optuna optimization history plot."""
        fig = plot_optimization_history(study)
        fig.update_layout(
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            width=800,
            height=500
        )
        return fig
    
    def _create_optuna_parallel(
        self,
        study: Any
    ) -> go.Figure:
        """Create Optuna parallel coordinate plot."""
        fig = plot_parallel_coordinate(study)
        fig.update_layout(
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            width=1000,
            height=600
        )
        return fig
    
    def _create_optuna_slices(
        self,
        study: Any
    ) -> go.Figure:
        """Create Optuna parameter slice plots."""
        fig = plot_slice(study)
        fig.update_layout(
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            width=1000,
            height=800
        )
        return fig
    
    def _create_optuna_contour(
        self,
        study: Any
    ) -> go.Figure:
        """Create Optuna contour plots."""
        fig = plot_contour(study)
        fig.update_layout(
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            width=800,
            height=800
        )
        return fig
    
    def save_visualizations(
        self,
        model_name: Optional[str] = None
    ):
        """Save visualization results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save dashboard
            dashboard = self.create_dashboard()
            dashboard.write_html(str(output_path / "tuning_dashboard.html"))
            
            # Save model-specific visualizations
            if model_name and model_name in self.tuner.study_results:
                study_viz = self.create_study_visualization(model_name)
                for name, fig in study_viz.items():
                    fig.write_html(
                        str(output_path / f"{model_name}_{name}.html")
                    )
                
                param_analysis = self.create_parameter_analysis(model_name)
                param_analysis.write_html(
                    str(output_path / f"{model_name}_params.html")
                )
            
            logger.info(f"Saved visualizations to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_tuning_visualizer(
    tuner: HyperparameterTuner,
    output_path: Optional[Path] = None
) -> TuningVisualizer:
    """Create tuning visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return TuningVisualizer(tuner, config)

if __name__ == "__main__":
    # Example usage
    from .hyperparameter_tuning import create_hyperparameter_tuner
    from .profile_learning import create_profile_learning
    from .profile_analysis import create_profile_analyzer
    from .prediction_profiling import create_prediction_profiler
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(metrics)
    predictor = create_easing_predictor(stats)
    visualizer = create_prediction_visualizer(predictor)
    controls = create_interactive_controls(visualizer)
    realtime = create_realtime_prediction(controls)
    performance = create_prediction_performance(realtime)
    profiler = create_prediction_profiler(performance)
    analyzer = create_profile_analyzer(profiler)
    learning = create_profile_learning(analyzer)
    tuner = create_hyperparameter_tuner(learning)
    viz = create_tuning_visualizer(
        tuner,
        output_path=Path("tuning_viz")
    )
    
    # Get training data
    data = learning._prepare_training_data("execution_time")
    if data is not None:
        X, y = data
        
        # Perform tuning
        tuner.tune_random_forest(X, y)
        tuner.tune_xgboost(X, y)
        
        # Generate visualizations
        viz.save_visualizations("rf")  # Random Forest visualizations
        viz.save_visualizations("xgb")  # XGBoost visualizations
