"""Ensemble predictions for mutation filter presets."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Type
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .preset_predictions import PresetPredictor, PredictionConfig, ModelMetadata

@dataclass
class EnsembleConfig:
    """Configuration for ensemble predictions."""
    models: Dict[str, Type[BaseEstimator]] = field(default_factory=lambda: {
        "rf": RandomForestRegressor,
        "gb": GradientBoostingRegressor,
        "et": ExtraTreesRegressor,
        "lasso": LassoCV,
        "mlp": MLPRegressor
    })
    weights_update_interval: int = 100  # samples
    min_weight: float = 0.1
    diversity_threshold: float = 0.3
    blend_method: str = "weighted"  # weighted, stacking, or boosting
    enable_uncertainty: bool = True

@dataclass
class EnsembleState:
    """State of ensemble model."""
    weights: Dict[str, float]
    errors: Dict[str, List[float]]
    diversity: Dict[Tuple[str, str], float]
    last_update: datetime
    training_samples: int

class PresetEnsemble:
    """Ensemble predictor for mutation presets."""
    
    def __init__(
        self,
        predictor: PresetPredictor,
        config: EnsembleConfig = None
    ):
        self.predictor = predictor
        self.config = config or EnsembleConfig()
        
        # Ensemble components
        self.models: Dict[str, Dict[str, BaseEstimator]] = {}
        self.states: Dict[str, EnsembleState] = {}
        
        # Setup ensemble models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ensemble models."""
        for preset_name in self.predictor.models:
            self.models[preset_name] = {}
            
            # Create model instances
            for name, model_class in self.config.models.items():
                if name == "mlp":
                    model = model_class(
                        hidden_layer_sizes=(100, 50),
                        max_iter=1000
                    )
                else:
                    model = model_class()
                
                self.models[preset_name][name] = model
            
            # Initialize state
            self.states[preset_name] = EnsembleState(
                weights={name: 1.0 / len(self.config.models)
                        for name in self.config.models},
                errors={name: [] for name in self.config.models},
                diversity={},
                last_update=datetime.now(),
                training_samples=0
            )
    
    async def train_ensemble(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Train ensemble models."""
        if preset_name not in self.predictor.models:
            return {"status": "no_base_model"}
        
        # Get training data
        preset = await self.predictor.analytics.preset_manager.get_preset(preset_name)
        results = [
            res for _, name, res in self.predictor.analytics.history
            if name == preset_name
        ]
        
        if len(results) < self.predictor.config.min_training_samples:
            return {
                "status": "insufficient_data",
                "samples_needed": (
                    self.predictor.config.min_training_samples - len(results)
                )
            }
        
        # Prepare data
        X, y = self.predictor._prepare_features(preset, results)
        X_scaled = self.predictor.scalers[preset_name].transform(X)
        
        # Train each model
        predictions = {}
        for name, model in self.models[preset_name].items():
            model.fit(X_scaled, y)
            predictions[name] = model.predict(X_scaled)
        
        # Update ensemble state
        state = self.states[preset_name]
        state.training_samples = len(X)
        
        # Calculate errors
        for name, preds in predictions.items():
            error = mean_squared_error(y, preds)
            state.errors[name].append(error)
        
        # Calculate diversity
        for name1 in predictions:
            for name2 in predictions:
                if name1 < name2:
                    diversity = np.corrcoef(
                        predictions[name1],
                        predictions[name2]
                    )[0, 1]
                    state.diversity[(name1, name2)] = diversity
        
        # Update weights based on performance and diversity
        self._update_weights(preset_name)
        
        return {
            "status": "success",
            "models": {
                name: {
                    "error": np.mean(state.errors[name]),
                    "weight": state.weights[name]
                }
                for name in self.config.models
            }
        }
    
    def _update_weights(
        self,
        preset_name: str
    ):
        """Update ensemble weights."""
        state = self.states[preset_name]
        
        # Calculate base weights from errors
        mean_errors = {
            name: np.mean(errors[-self.config.weights_update_interval:])
            for name, errors in state.errors.items()
        }
        
        total_error = sum(mean_errors.values())
        if total_error > 0:
            weights = {
                name: 1 - (error / total_error)
                for name, error in mean_errors.items()
            }
        else:
            weights = {
                name: 1.0 / len(mean_errors)
                for name in mean_errors
            }
        
        # Adjust weights based on diversity
        if self.config.blend_method == "weighted":
            for (name1, name2), div in state.diversity.items():
                if div > self.config.diversity_threshold:
                    # Reduce weights of correlated models
                    factor = 1 - (div - self.config.diversity_threshold)
                    weights[name1] *= factor
                    weights[name2] *= factor
        
        # Normalize and apply minimum weight
        total = sum(weights.values())
        state.weights = {
            name: max(self.config.min_weight,
                     weight / total)
            for name, weight in weights.items()
        }
        
        # Re-normalize
        total = sum(state.weights.values())
        state.weights = {
            name: weight / total
            for name, weight in state.weights.items()
        }
    
    async def predict_ensemble(
        self,
        preset_name: str,
        horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate ensemble predictions."""
        if preset_name not in self.models:
            return {"status": "no_ensemble"}
        
        horizon = horizon or self.predictor.config.forecast_horizon
        
        # Get current state
        preset = await self.predictor.analytics.preset_manager.get_preset(preset_name)
        latest_results = [
            res for _, name, res in self.predictor.analytics.history[-10:]
            if name == preset_name
        ]
        
        if not latest_results:
            return {"status": "no_recent_data"}
        
        # Generate predictions from each model
        predictions = {name: [] for name in self.models[preset_name]}
        uncertainties = {name: [] for name in self.models[preset_name]}
        current_state = latest_results[-1]
        
        for _ in range(horizon):
            # Prepare features
            X = self.predictor._prepare_features(preset, [current_state])[0]
            X_scaled = self.predictor.scalers[preset_name].transform(X.reshape(1, -1))
            
            # Get predictions from each model
            for name, model in self.models[preset_name].items():
                pred = model.predict(X_scaled)[0]
                predictions[name].append(pred)
                
                # Calculate uncertainty
                if self.config.enable_uncertainty:
                    if hasattr(model, "estimators_"):
                        # For ensemble models
                        preds = [
                            est.predict(X_scaled)[0]
                            for est in model.estimators_
                        ]
                        uncertainty = np.std(preds)
                    else:
                        # For other models, use historical error
                        uncertainty = np.std(self.states[preset_name].errors[name][-10:])
                    
                    uncertainties[name].append(uncertainty)
            
            # Update state with weighted average
            current_state.mutation_score = np.average(
                [predictions[name][-1] for name in predictions],
                weights=[self.states[preset_name].weights[name]
                        for name in predictions]
            )
        
        # Combine predictions
        if self.config.blend_method == "weighted":
            ensemble_predictions = [
                np.average(
                    [predictions[name][i] for name in predictions],
                    weights=[self.states[preset_name].weights[name]
                            for name in predictions]
                )
                for i in range(horizon)
            ]
            
            ensemble_uncertainty = [
                np.sqrt(np.average(
                    [uncertainties[name][i] ** 2 for name in uncertainties],
                    weights=[self.states[preset_name].weights[name]
                            for name in uncertainties]
                ))
                for i in range(horizon)
            ]
            
        elif self.config.blend_method == "stacking":
            # Use a meta-model (simple average for now)
            ensemble_predictions = [
                np.mean([predictions[name][i] for name in predictions])
                for i in range(horizon)
            ]
            ensemble_uncertainty = [
                np.std([predictions[name][i] for name in predictions])
                for i in range(horizon)
            ]
        
        else:  # boosting
            # Use sequential predictions
            ensemble_predictions = predictions[
                max(predictions.keys(),
                    key=lambda k: self.states[preset_name].weights[k])
            ]
            ensemble_uncertainty = uncertainties[
                max(uncertainties.keys(),
                    key=lambda k: self.states[preset_name].weights[k])
            ]
        
        return {
            "status": "success",
            "predictions": ensemble_predictions,
            "uncertainty": ensemble_uncertainty,
            "model_predictions": predictions,
            "model_uncertainties": uncertainties,
            "weights": self.states[preset_name].weights
        }
    
    async def create_ensemble_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create ensemble visualization plots."""
        predictions = await self.predict_ensemble(preset_name)
        if predictions["status"] != "success":
            return {}
        
        plots = {}
        
        # Model comparison plot
        comparison_fig = go.Figure()
        
        # Add individual model predictions
        for name, preds in predictions["model_predictions"].items():
            comparison_fig.add_trace(
                go.Scatter(
                    y=preds,
                    mode="lines",
                    name=f"{name} (w={predictions['weights'][name]:.2f})",
                    line=dict(dash="dash")
                )
            )
        
        # Add ensemble prediction with uncertainty
        comparison_fig.add_trace(
            go.Scatter(
                y=predictions["predictions"],
                mode="lines",
                name="Ensemble",
                line=dict(color="black", width=2)
            )
        )
        
        # Add uncertainty bounds
        upper = np.array(predictions["predictions"]) + np.array(predictions["uncertainty"])
        lower = np.array(predictions["predictions"]) - np.array(predictions["uncertainty"])
        
        comparison_fig.add_trace(
            go.Scatter(
                y=upper,
                mode="lines",
                name="Uncertainty",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        comparison_fig.add_trace(
            go.Scatter(
                y=lower,
                mode="lines",
                fill="tonexty",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        comparison_fig.update_layout(
            title="Model Comparison",
            yaxis_title="Prediction"
        )
        plots["comparison"] = comparison_fig
        
        # Weight evolution plot
        if preset_name in self.states:
            state = self.states[preset_name]
            weights_fig = go.Figure()
            
            for name in self.config.models:
                weights_fig.add_trace(
                    go.Scatter(
                        y=[state.weights[name]],
                        name=name,
                        mode="markers+text",
                        text=[f"{state.weights[name]:.2f}"],
                        textposition="top center"
                    )
                )
            
            weights_fig.update_layout(
                title="Model Weights",
                yaxis_title="Weight",
                showlegend=True
            )
            plots["weights"] = weights_fig
        
        return plots

def create_preset_ensemble(
    predictor: PresetPredictor,
    config: Optional[EnsembleConfig] = None
) -> PresetEnsemble:
    """Create preset ensemble predictor."""
    return PresetEnsemble(predictor, config)

if __name__ == "__main__":
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        
        # Create test preset
        await manager.save_preset(
            "test_preset",
            "Test preset",
            {
                "operators": ["type_mutation"],
                "error_types": ["TypeError"],
                "score_range": [0.5, 1.0],
                "time_range": None
            }
        )
        
        # Add synthetic data
        for _ in range(100):
            result = MutationTestResult()
            result.total_mutations = 100
            result.killed_mutations = np.random.randint(50, 90)
            result.survived_mutations = 100 - result.killed_mutations
            result.operator_stats = {
                "type_mutation": {
                    "killed": result.killed_mutations,
                    "survived": result.survived_mutations
                }
            }
            
            await analytics.record_result("test_preset", result)
        
        # Train models
        await predictor.train_model("test_preset")
        await ensemble.train_ensemble("test_preset")
        
        # Generate predictions
        predictions = await ensemble.predict_ensemble("test_preset")
        print("Ensemble predictions:", predictions)
        
        # Create plots
        plots = await ensemble.create_ensemble_plots("test_preset")
        for name, fig in plots.items():
            fig.write_html(f"test_preset_ensemble_{name}.html")
    
    asyncio.run(main())
