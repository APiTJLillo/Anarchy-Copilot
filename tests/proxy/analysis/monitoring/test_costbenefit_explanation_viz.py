"""Visualization methods for model explanations."""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pytest
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from .test_costbenefit_explainability import ModelExplainer, ExplainabilityConfig
from .test_costbenefit_ml import MLPredictor, MLConfig

@dataclass
class ExplanationVisualizationConfig:
    """Configuration for explanation visualizations."""
    plot_width: int = 1000
    plot_height: int = 600
    interactive: bool = True
    theme: str = "plotly_white"
    color_scheme: Dict[str, str] = field(default_factory=lambda: {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#3498db",
        "background": "#ecf0f1"
    })
    export_formats: List[str] = field(default_factory=lambda: ["html", "png", "svg"])
    show_feature_importance: bool = True
    show_correlation_matrix: bool = True
    show_shap_summary: bool = True
    show_lime_explanations: bool = True
    max_features_display: int = 10
    confidence_interval: float = 0.95

class ExplanationVisualizer:
    """Visualize model explanations."""

    def __init__(
        self,
        explainer: ModelExplainer,
        config: ExplanationVisualizationConfig
    ):
        self.explainer = explainer
        self.config = config

    def create_feature_importance_plot(
        self,
        importance_data: Dict[str, Any]
    ) -> go.Figure:
        """Create feature importance visualization."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                "Random Forest Feature Importance",
                "Feature Correlations"
            ),
            vertical_spacing=0.2
        )
        
        # Random Forest importance
        if "random_forest" in importance_data:
            items = sorted(
                importance_data["random_forest"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_features_display]
            
            features, values = zip(*items)
            
            fig.add_trace(
                go.Bar(
                    x=features,
                    y=values,
                    name="Feature Importance",
                    marker_color=self.config.color_scheme["positive"]
                ),
                row=1, col=1
            )
        
        # Feature correlations
        if "correlations" in importance_data:
            corr_features = []
            corr_values = []
            p_values = []
            
            for feature, stats in importance_data["correlations"].items():
                corr_features.append(feature)
                corr_values.append(stats["correlation"])
                p_values.append(stats["p_value"])
            
            fig.add_trace(
                go.Bar(
                    x=corr_features,
                    y=corr_values,
                    name="Correlation",
                    marker_color=[
                        self.config.color_scheme["positive"] if v > 0
                        else self.config.color_scheme["negative"]
                        for v in corr_values
                    ]
                ),
                row=2, col=1
            )
            
            # Add significance markers
            significant = [
                f"p={p:.3f}" if p < 0.05 else ""
                for p in p_values
            ]
            
            fig.add_trace(
                go.Scatter(
                    x=corr_features,
                    y=[max(corr_values) * 1.1] * len(corr_features),
                    text=significant,
                    mode="text",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Feature Analysis",
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.theme,
            showlegend=True
        )
        
        return fig

    def create_shap_summary_plot(
        self,
        explanation: Dict[str, Any]
    ) -> go.Figure:
        """Create SHAP summary visualization."""
        if "shap" not in explanation:
            return None
        
        shap_values = np.array(explanation["shap"]["values"])
        feature_names = list(explanation["feature_values"].keys())
        feature_values = list(explanation["feature_values"].values())
        
        fig = go.Figure()
        
        # Calculate SHAP summary statistics
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_shap)
        
        # Create SHAP summary plot
        for idx in sorted_idx[-self.config.max_features_display:]:
            fig.add_trace(
                go.Box(
                    y=shap_values[:, idx],
                    name=feature_names[idx],
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker=dict(
                        color=self.config.color_scheme["neutral"],
                        size=4
                    )
                )
            )
        
        fig.update_layout(
            title="SHAP Value Distribution",
            xaxis_title="Features",
            yaxis_title="SHAP Value",
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.theme
        )
        
        return fig

    def create_lime_explanation_plot(
        self,
        explanation: Dict[str, Any]
    ) -> go.Figure:
        """Create LIME explanation visualization."""
        if "lime" not in explanation:
            return None
        
        lime_exp = explanation["lime"]
        local_exp = lime_exp["local_exp"]
        
        fig = go.Figure()
        
        # Extract feature importance from LIME
        features = []
        importance = []
        
        for class_idx, exp_list in local_exp.items():
            for feat_id, feat_importance in exp_list:
                features.append(
                    list(explanation["feature_values"].keys())[feat_id]
                )
                importance.append(feat_importance)
        
        # Sort by absolute importance
        sorted_idx = np.argsort(np.abs(importance))[-self.config.max_features_display:]
        
        fig.add_trace(
            go.Bar(
                x=[features[i] for i in sorted_idx],
                y=[importance[i] for i in sorted_idx],
                marker_color=[
                    self.config.color_scheme["positive"]
                    if imp > 0 else self.config.color_scheme["negative"]
                    for imp in [importance[i] for i in sorted_idx]
                ]
            )
        )
        
        fig.update_layout(
            title="LIME Feature Contributions",
            xaxis_title="Features",
            yaxis_title="Contribution",
            height=self.config.plot_height,
            width=self.config.plot_width,
            template=self.config.theme
        )
        
        return fig

    def save_visualization(
        self,
        fig: go.Figure,
        name: str,
        output_dir: str
    ) -> None:
        """Save visualization in multiple formats."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fmt in self.config.export_formats:
            file_path = output_path / f"{name}.{fmt}"
            if fmt == "html":
                fig.write_html(str(file_path))
            else:
                fig.write_image(str(file_path))

@pytest.fixture
def explanation_visualizer(model_explainer):
    """Create explanation visualizer for testing."""
    config = ExplanationVisualizationConfig()
    return ExplanationVisualizer(model_explainer, config)

@pytest.mark.asyncio
async def test_feature_importance_visualization(
    explanation_visualizer,
    model_explainer,
    tmp_path
):
    """Test feature importance visualization."""
    # Get feature importance data
    importance = model_explainer.analyze_feature_importance()
    
    # Create visualization
    fig = explanation_visualizer.create_feature_importance_plot(importance)
    
    # Verify plot
    assert isinstance(fig, go.Figure)
    
    # Save visualization
    explanation_visualizer.save_visualization(
        fig,
        "feature_importance",
        str(tmp_path)
    )
    
    # Verify files
    for fmt in explanation_visualizer.config.export_formats:
        assert (tmp_path / f"feature_importance.{fmt}").exists()

@pytest.mark.asyncio
async def test_shap_visualization(
    explanation_visualizer,
    model_explainer,
    ml_predictor,
    tmp_path
):
    """Test SHAP visualization."""
    # Train model and get explanation
    alert = create_test_alert()
    await ml_predictor.process_alert(alert)
    prediction = await ml_predictor.process_alert(alert)
    explanation = await model_explainer.explain_prediction(alert, prediction)
    
    # Create visualization
    fig = explanation_visualizer.create_shap_summary_plot(explanation)
    
    # Verify plot
    assert isinstance(fig, go.Figure)
    
    # Save visualization
    explanation_visualizer.save_visualization(
        fig,
        "shap_summary",
        str(tmp_path)
    )

def create_test_alert():
    """Create test alert for visualization."""
    from .test_costbenefit_alerts import Alert, AlertThreshold
    
    return Alert(
        alert_id="test_visualization",
        threshold=AlertThreshold(
            metric_name="cpu_percent",
            warning_threshold=70,
            critical_threshold=90
        ),
        current_value=85,
        threshold_value=70,
        severity="warning",
        start_time=datetime.now(),
        duration=0.0,
        message="Test alert for visualization"
    )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
