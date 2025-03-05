"""Predictive analytics for mutation filter presets."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path

from .preset_analytics import PresetAnalytics, AnalyticsConfig
from .mutation_presets import FilterPreset

@dataclass
class PredictionConfig:
    """Configuration for predictive analytics."""
    model_dir: Path = Path("preset_models")
    forecast_horizon: int = 10
    training_window: int = 100
    retrain_interval: int = 24  # hours
    min_training_samples: int = 30
    confidence_level: float = 0.95
    feature_importance_threshold: float = 0.05

@dataclass
class ModelMetadata:
    """Metadata for trained models."""
    preset_name: str
    created: datetime
    metrics: Dict[str, float]
    features: List[str]
    parameters: Dict[str, Any]
    training_size: int

class PresetPredictor:
    """Predict preset performance and optimize settings."""
    
    def __init__(
        self,
        analytics: PresetAnalytics,
        config: PredictionConfig = None
    ):
        self.analytics = analytics
        self.config = config or PredictionConfig()
        
        # Model storage
        self.models: Dict[str, RandomForestRegressor] = {}
        self.metadata: Dict[str, ModelMetadata] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        # Setup model directory
        self.config.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing models
        self._load_models()
    
    def _load_models(self):
        """Load saved models."""
        for model_file in self.config.model_dir.glob("*.joblib"):
            try:
                preset_name = model_file.stem
                model_data = joblib.load(model_file)
                
                self.models[preset_name] = model_data["model"]
                self.metadata[preset_name] = model_data["metadata"]
                self.scalers[preset_name] = model_data["scaler"]
                
            except Exception as e:
                print(f"Failed to load model {model_file}: {e}")
    
    async def save_model(
        self,
        preset_name: str
    ):
        """Save model to disk."""
        if preset_name not in self.models:
            return
        
        model_file = self.config.model_dir / f"{preset_name}.joblib"
        model_data = {
            "model": self.models[preset_name],
            "metadata": self.metadata[preset_name],
            "scaler": self.scalers[preset_name]
        }
        
        joblib.dump(model_data, model_file)
    
    def _prepare_features(
        self,
        preset: FilterPreset,
        results: List[Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector."""
        # Extract features
        features = []
        targets = []
        
        for result in results:
            # Feature vector
            feature_dict = {
                # Operator features
                **{f"op_{op}": 1 if op in preset.filters["operators"] else 0
                   for op in self.analytics.stats[preset.name].operator_coverage.keys()},
                
                # Error type features
                **{f"err_{err}": 1 if err in preset.filters["error_types"] else 0
                   for err in self.analytics.stats[preset.name].error_rates.keys()},
                
                # Score range features
                "min_score": preset.filters["score_range"][0],
                "max_score": preset.filters["score_range"][1],
                
                # Historical features
                "prev_score": result.mutation_score,
                "killed_ratio": result.killed_mutations / result.total_mutations
            }
            
            features.append(list(feature_dict.values()))
            targets.append(result.mutation_score)
        
        return np.array(features), np.array(targets)
    
    async def train_model(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Train prediction model for preset."""
        # Get preset and results
        preset = await self.analytics.preset_manager.get_preset(preset_name)
        results = [
            res for _, name, res in self.analytics.history
            if name == preset_name
        ]
        
        if len(results) < self.config.min_training_samples:
            return {
                "status": "insufficient_data",
                "required": self.config.min_training_samples,
                "available": len(results)
            }
        
        # Prepare data
        X, y = self._prepare_features(preset, results)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=5,
            random_state=42
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test_scaled)
        metrics = {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "feature_importance": dict(zip(
                model.feature_names_in_,
                model.feature_importances_
            ))
        }
        
        # Save model
        self.models[preset_name] = model
        self.scalers[preset_name] = scaler
        self.metadata[preset_name] = ModelMetadata(
            preset_name=preset_name,
            created=datetime.now(),
            metrics=metrics,
            features=list(model.feature_names_in_),
            parameters=model.get_params(),
            training_size=len(X_train)
        )
        
        await self.save_model(preset_name)
        
        return {
            "status": "success",
            "metrics": metrics
        }
    
    async def predict_performance(
        self,
        preset_name: str,
        horizon: Optional[int] = None
    ) -> Dict[str, Any]:
        """Predict future performance."""
        if preset_name not in self.models:
            return {"status": "no_model"}
        
        horizon = horizon or self.config.forecast_horizon
        model = self.models[preset_name]
        scaler = self.scalers[preset_name]
        
        # Get current state
        preset = await self.analytics.preset_manager.get_preset(preset_name)
        latest_results = [
            res for _, name, res in self.analytics.history[-10:]
            if name == preset_name
        ]
        
        if not latest_results:
            return {"status": "no_recent_data"}
        
        # Generate predictions
        predictions = []
        confidence_intervals = []
        current_state = latest_results[-1]
        
        for _ in range(horizon):
            # Prepare features
            X = self._prepare_features(preset, [current_state])[0]
            X_scaled = scaler.transform(X.reshape(1, -1))
            
            # Make prediction
            pred = model.predict(X_scaled)[0]
            
            # Calculate confidence interval
            pred_std = np.std([
                tree.predict(X_scaled)[0]
                for tree in model.estimators_
            ])
            ci = np.array([
                pred - 1.96 * pred_std,
                pred + 1.96 * pred_std
            ])
            
            predictions.append(pred)
            confidence_intervals.append(ci)
            
            # Update state for next prediction
            current_state.mutation_score = pred
        
        return {
            "status": "success",
            "predictions": predictions,
            "confidence_intervals": confidence_intervals,
            "horizon": horizon
        }
    
    async def optimize_settings(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Optimize preset settings."""
        if preset_name not in self.models:
            return {"status": "no_model"}
        
        model = self.models[preset_name]
        scaler = self.scalers[preset_name]
        metadata = self.metadata[preset_name]
        
        # Get current preset
        preset = await self.analytics.preset_manager.get_preset(preset_name)
        
        # Find important features
        important_features = {
            feature: importance
            for feature, importance in metadata.metrics["feature_importance"].items()
            if importance > self.config.feature_importance_threshold
        }
        
        # Generate optimization suggestions
        suggestions = []
        
        # Operator suggestions
        for feature, importance in important_features.items():
            if feature.startswith("op_"):
                operator = feature[3:]
                current = operator in preset.filters["operators"]
                
                # Test alternative
                test_features = self._prepare_features(preset, [preset])[0]
                test_features_scaled = scaler.transform(test_features.reshape(1, -1))
                
                baseline = model.predict(test_features_scaled)[0]
                
                # Flip operator status
                idx = list(model.feature_names_in_).index(feature)
                test_features[0, idx] = not current
                test_features_scaled = scaler.transform(test_features.reshape(1, -1))
                
                alternative = model.predict(test_features_scaled)[0]
                
                if alternative > baseline:
                    suggestions.append({
                        "type": "operator",
                        "operator": operator,
                        "action": "add" if not current else "remove",
                        "expected_improvement": alternative - baseline,
                        "confidence": importance
                    })
        
        # Score range suggestions
        current_min = preset.filters["score_range"][0]
        current_max = preset.filters["score_range"][1]
        
        test_ranges = [
            (max(0.0, current_min - 0.1), current_max),
            (current_min, min(1.0, current_max + 0.1))
        ]
        
        for min_score, max_score in test_ranges:
            test_preset = FilterPreset(
                name=preset.name,
                description=preset.description,
                filters={
                    **preset.filters,
                    "score_range": [min_score, max_score]
                }
            )
            
            test_features = self._prepare_features(test_preset, [preset])[0]
            test_features_scaled = scaler.transform(test_features.reshape(1, -1))
            
            prediction = model.predict(test_features_scaled)[0]
            
            if prediction > model.predict(
                scaler.transform(
                    self._prepare_features(preset, [preset])[0].reshape(1, -1)
                )
            )[0]:
                suggestions.append({
                    "type": "score_range",
                    "range": [min_score, max_score],
                    "expected_improvement": prediction - baseline,
                    "confidence": np.mean([
                        importance
                        for feature, importance in important_features.items()
                        if "score" in feature
                    ])
                })
        
        return {
            "status": "success",
            "suggestions": sorted(
                suggestions,
                key=lambda s: s["expected_improvement"],
                reverse=True
            )
        }
    
    async def create_prediction_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create prediction visualizations."""
        predictions = await self.predict_performance(preset_name)
        if predictions["status"] != "success":
            return {}
        
        plots = {}
        
        # Performance forecast plot
        forecast_fig = go.Figure()
        
        # Historical data
        historical = [
            res.mutation_score
            for _, name, res in self.analytics.history[-30:]
            if name == preset_name
        ]
        
        forecast_fig.add_trace(
            go.Scatter(
                y=historical,
                mode="lines",
                name="Historical",
                line=dict(color="blue")
            )
        )
        
        # Predictions
        forecast_fig.add_trace(
            go.Scatter(
                y=predictions["predictions"],
                mode="lines",
                name="Forecast",
                line=dict(color="red", dash="dash")
            )
        )
        
        # Confidence intervals
        ci_lower = [ci[0] for ci in predictions["confidence_intervals"]]
        ci_upper = [ci[1] for ci in predictions["confidence_intervals"]]
        
        forecast_fig.add_trace(
            go.Scatter(
                y=ci_upper,
                mode="lines",
                name="Upper CI",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        forecast_fig.add_trace(
            go.Scatter(
                y=ci_lower,
                mode="lines",
                name="Lower CI",
                fill="tonexty",
                line=dict(width=0),
                showlegend=False
            )
        )
        
        forecast_fig.update_layout(
            title="Performance Forecast",
            yaxis_title="Mutation Score"
        )
        plots["forecast"] = forecast_fig
        
        # Feature importance plot
        if preset_name in self.metadata:
            importance_fig = go.Figure(
                go.Bar(
                    x=list(self.metadata[preset_name].metrics["feature_importance"].keys()),
                    y=list(self.metadata[preset_name].metrics["feature_importance"].values()),
                    name="Feature Importance"
                )
            )
            
            importance_fig.update_layout(
                title="Feature Importance",
                xaxis_title="Feature",
                yaxis_title="Importance Score"
            )
            plots["importance"] = importance_fig
        
        return plots

def create_preset_predictor(
    analytics: PresetAnalytics,
    config: Optional[PredictionConfig] = None
) -> PresetPredictor:
    """Create preset predictor."""
    return PresetPredictor(analytics, config)

if __name__ == "__main__":
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        
        # Train model for test preset
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
        
        # Add some synthetic data
        for _ in range(50):
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
        
        # Train and get predictions
        await predictor.train_model("test_preset")
        predictions = await predictor.predict_performance("test_preset")
        print("Predictions:", predictions)
        
        # Get optimization suggestions
        suggestions = await predictor.optimize_settings("test_preset")
        print("Optimization suggestions:", json.dumps(suggestions, indent=2))
        
        # Generate plots
        plots = await predictor.create_prediction_plots("test_preset")
        for name, fig in plots.items():
            fig.write_html(f"test_preset_prediction_{name}.html")
    
    asyncio.run(main())
