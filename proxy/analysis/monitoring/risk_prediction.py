"""Risk prediction for strategy changes."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .risk_assessment import (
    RiskAnalyzer, RiskAssessment, RiskConfig,
    ChangeImpact, StrategyRecommendation
)

@dataclass
class PredictionConfig:
    """Configuration for risk prediction."""
    enabled: bool = True
    forecast_horizon: int = 12  # periods
    update_interval: float = 300.0  # 5 minutes
    min_history: int = 100
    confidence_level: float = 0.95
    feature_window: int = 10
    seasonality_window: int = 24
    enable_trends: bool = True
    enable_seasonality: bool = True
    enable_monte_carlo: bool = True
    monte_carlo_samples: int = 1000
    max_simulations: int = 100
    visualization_dir: Optional[str] = "risk_predictions"

@dataclass
class RiskPrediction:
    """Predicted risk metrics."""
    timestamp: datetime
    strategy: str
    base_risk: float
    trend_risk: float
    seasonal_risk: float
    total_risk: float
    confidence_interval: Tuple[float, float]
    failure_probability: float
    predicted_impacts: Dict[str, float]
    simulation_results: Optional[List[float]] = None

@dataclass
class PredictionResult:
    """Results of risk prediction."""
    predictions: List[RiskPrediction]
    trends: Dict[str, float]
    seasonality: Dict[str, List[float]]
    anomalies: List[Dict[str, Any]]
    forecast_quality: Dict[str, float]

class RiskPredictor:
    """Predict future risks of strategy changes."""
    
    def __init__(
        self,
        analyzer: RiskAnalyzer,
        config: PredictionConfig = None
    ):
        self.analyzer = analyzer
        self.config = config or PredictionConfig()
        
        # Analysis state
        self.predictions: Dict[str, List[RiskPrediction]] = {}
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.last_update = datetime.min
        self.predictor_task: Optional[asyncio.Task] = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize prediction models."""
        self.models["risk"] = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.models["anomaly"] = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scalers["features"] = StandardScaler()
    
    async def start_predictor(self):
        """Start risk predictor."""
        if not self.config.enabled:
            return
        
        if self.predictor_task is None:
            self.predictor_task = asyncio.create_task(self._run_predictor())
    
    async def stop_predictor(self):
        """Stop risk predictor."""
        if self.predictor_task:
            self.predictor_task.cancel()
            try:
                await self.predictor_task
            except asyncio.CancelledError:
                pass
            self.predictor_task = None
    
    async def _run_predictor(self):
        """Run periodic predictions."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    for strategy in self.analyzer.assessments:
                        await self.predict_risks(strategy)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Predictor error: {e}")
                await asyncio.sleep(60)
    
    async def predict_risks(
        self,
        strategy: str,
        horizon: Optional[int] = None
    ) -> PredictionResult:
        """Predict future risks for strategy."""
        if horizon is None:
            horizon = self.config.forecast_horizon
        
        # Get historical data
        history = await self._get_historical_data(strategy)
        if len(history) < self.config.min_history:
            return PredictionResult([], {}, {}, [], {})
        
        # Prepare features
        features = await self._prepare_features(history)
        
        # Train models if needed
        if not self.models["risk"].is_fitted:
            self._train_models(features, history)
        
        # Generate predictions
        predictions = []
        current_time = datetime.now()
        
        for i in range(horizon):
            # Update features
            future_features = await self._forecast_features(
                features,
                i + 1
            )
            
            # Predict base risk
            base_risk = self.models["risk"].predict(future_features)[0]
            
            # Add trend component
            trend_risk = await self._calculate_trend_risk(
                history,
                i + 1
            ) if self.config.enable_trends else 0.0
            
            # Add seasonal component
            seasonal_risk = await self._calculate_seasonal_risk(
                history,
                current_time + timedelta(
                    seconds=(i + 1) * self.config.update_interval
                )
            ) if self.config.enable_seasonality else 0.0
            
            # Combine predictions
            total_risk = base_risk + trend_risk + seasonal_risk
            
            # Calculate confidence interval
            ci = await self._calculate_confidence_interval(
                total_risk,
                future_features
            )
            
            # Run Monte Carlo simulation
            simulation_results = await self._run_simulation(
                total_risk,
                future_features
            ) if self.config.enable_monte_carlo else None
            
            # Create prediction
            prediction = RiskPrediction(
                timestamp=current_time + timedelta(
                    seconds=(i + 1) * self.config.update_interval
                ),
                strategy=strategy,
                base_risk=base_risk,
                trend_risk=trend_risk,
                seasonal_risk=seasonal_risk,
                total_risk=total_risk,
                confidence_interval=ci,
                failure_probability=np.mean([r > 0.8 for r in simulation_results])
                if simulation_results else 0.0,
                predicted_impacts=await self._predict_impacts(future_features),
                simulation_results=simulation_results
            )
            
            predictions.append(prediction)
        
        # Store predictions
        self.predictions[strategy] = predictions
        
        # Calculate additional metrics
        trends = await self._analyze_trends(predictions)
        seasonality = await self._analyze_seasonality(history)
        anomalies = await self._detect_anomalies(predictions)
        quality = await self._evaluate_forecast_quality(
            history,
            self.predictions.get(strategy, [])
        )
        
        return PredictionResult(
            predictions=predictions,
            trends=trends,
            seasonality=seasonality,
            anomalies=anomalies,
            forecast_quality=quality
        )
    
    async def _get_historical_data(
        self,
        strategy: str
    ) -> List[Dict[str, Any]]:
        """Get historical risk data."""
        history = []
        current_time = datetime.now()
        
        for h in self.analyzer.balancer.history:
            if strategy in h.metrics:
                assessment = self.analyzer.assessments.get(strategy)
                if assessment:
                    history.append({
                        "timestamp": h.timestamp,
                        "risk_score": assessment.risk_score,
                        "impact_probability": assessment.impact_probability,
                        "metrics": h.metrics[strategy].__dict__,
                        "age": (current_time - h.timestamp).total_seconds()
                    })
        
        return history[-self.config.min_history:]
    
    async def _prepare_features(
        self,
        history: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Prepare feature matrix."""
        features = []
        
        for entry in history:
            feature_vector = [
                entry["risk_score"],
                entry["impact_probability"],
                entry["metrics"]["success_rate"],
                entry["metrics"]["memory_savings"],
                entry["metrics"]["execution_time"],
                entry["metrics"]["cpu_overhead"],
                entry["metrics"]["stability_score"],
                entry["age"] / 3600  # Convert to hours
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        if self.scalers["features"].n_samples_seen_ is None:
            features = self.scalers["features"].fit_transform(features)
        else:
            features = self.scalers["features"].transform(features)
        
        return features
    
    def _train_models(
        self,
        features: np.ndarray,
        history: List[Dict[str, Any]]
    ):
        """Train prediction models."""
        # Prepare target variables
        risks = np.array([h["risk_score"] for h in history])
        
        # Train risk predictor
        self.models["risk"].fit(features, risks)
        
        # Train anomaly detector
        self.models["anomaly"].fit(features)
    
    async def _forecast_features(
        self,
        features: np.ndarray,
        steps: int
    ) -> np.ndarray:
        """Forecast future feature values."""
        if len(features) < self.config.feature_window:
            return features[-1:]
        
        # Use last n observations for forecasting
        recent = features[-self.config.feature_window:]
        
        # Simple trend extrapolation
        trends = np.mean(
            np.diff(recent, axis=0),
            axis=0
        )
        
        return recent[-1:] + trends * steps
    
    async def _calculate_trend_risk(
        self,
        history: List[Dict[str, Any]],
        steps: int
    ) -> float:
        """Calculate trend component of risk."""
        if len(history) < 2:
            return 0.0
        
        risks = [h["risk_score"] for h in history]
        times = [i for i in range(len(risks))]
        
        slope, _, _, _, _ = stats.linregress(times, risks)
        
        return max(0.0, min(1.0, slope * steps))
    
    async def _calculate_seasonal_risk(
        self,
        history: List[Dict[str, Any]],
        timestamp: datetime
    ) -> float:
        """Calculate seasonal component of risk."""
        if len(history) < self.config.seasonality_window:
            return 0.0
        
        # Get hour of day
        hour = timestamp.hour
        
        # Calculate average risk for this hour
        matching_hours = [
            h["risk_score"]
            for h in history
            if h["timestamp"].hour == hour
        ]
        
        if not matching_hours:
            return 0.0
        
        return np.mean(matching_hours) - np.mean(
            [h["risk_score"] for h in history]
        )
    
    async def _calculate_confidence_interval(
        self,
        predicted_risk: float,
        features: np.ndarray
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        # Get prediction probabilities
        probas = self.models["risk"].predict_proba(features)
        
        # Calculate interval
        lower = np.percentile(
            probas,
            (1 - self.config.confidence_level) * 50
        )
        upper = np.percentile(
            probas,
            (1 + self.config.confidence_level) * 50
        )
        
        return (
            max(0.0, min(1.0, predicted_risk + lower)),
            max(0.0, min(1.0, predicted_risk + upper))
        )
    
    async def _run_simulation(
        self,
        predicted_risk: float,
        features: np.ndarray
    ) -> List[float]:
        """Run Monte Carlo simulation."""
        results = []
        
        for _ in range(self.config.monte_carlo_samples):
            # Add random noise to features
            noisy_features = features + np.random.normal(
                0,
                0.1,
                features.shape
            )
            
            # Predict with noise
            noisy_prediction = self.models["risk"].predict(noisy_features)[0]
            results.append(noisy_prediction)
        
        return results
    
    async def _predict_impacts(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Predict specific impact metrics."""
        impacts = {}
        
        # Memory impact
        impacts["memory"] = max(0.0, min(1.0, features[0, 3]))
        
        # CPU impact
        impacts["cpu"] = max(0.0, min(1.0, features[0, 5]))
        
        # Latency impact
        impacts["latency"] = max(0.0, min(1.0, features[0, 4]))
        
        # Stability impact
        impacts["stability"] = max(0.0, min(1.0, 1 - features[0, 6]))
        
        return impacts
    
    async def _analyze_trends(
        self,
        predictions: List[RiskPrediction]
    ) -> Dict[str, float]:
        """Analyze prediction trends."""
        if not predictions:
            return {}
        
        risks = [p.total_risk for p in predictions]
        times = list(range(len(risks)))
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            times,
            risks
        )
        
        return {
            "slope": slope,
            "intercept": intercept,
            "correlation": r_value ** 2,
            "significance": 1 - p_value,
            "error": std_err
        }
    
    async def _analyze_seasonality(
        self,
        history: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        """Analyze seasonal patterns."""
        if len(history) < self.config.seasonality_window:
            return {}
        
        # Group by hour
        hourly_risks = [[] for _ in range(24)]
        for entry in history:
            hour = entry["timestamp"].hour
            hourly_risks[hour].append(entry["risk_score"])
        
        # Calculate statistics
        return {
            "means": [np.mean(risks) if risks else 0.0 for risks in hourly_risks],
            "stds": [np.std(risks) if risks else 0.0 for risks in hourly_risks],
            "counts": [len(risks) for risks in hourly_risks]
        }
    
    async def _detect_anomalies(
        self,
        predictions: List[RiskPrediction]
    ) -> List[Dict[str, Any]]:
        """Detect anomalous predictions."""
        if not predictions:
            return []
        
        # Convert predictions to features
        features = np.array([
            [
                p.base_risk,
                p.trend_risk,
                p.seasonal_risk,
                p.total_risk,
                p.failure_probability
            ]
            for p in predictions
        ])
        
        # Detect anomalies
        scores = self.models["anomaly"].score_samples(features)
        
        anomalies = []
        for i, score in enumerate(scores):
            if score < -0.5:  # Anomaly threshold
                anomalies.append({
                    "timestamp": predictions[i].timestamp,
                    "risk_score": predictions[i].total_risk,
                    "anomaly_score": score,
                    "confidence": predictions[i].confidence_interval
                })
        
        return anomalies
    
    async def _evaluate_forecast_quality(
        self,
        history: List[Dict[str, Any]],
        predictions: List[RiskPrediction]
    ) -> Dict[str, float]:
        """Evaluate forecast quality metrics."""
        if not history or not predictions:
            return {}
        
        # Calculate error metrics
        actual = np.array([h["risk_score"] for h in history[-len(predictions):]])
        predicted = np.array([p.total_risk for p in predictions])
        
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "accuracy": 1 - mae
        }
    
    async def create_prediction_plots(self) -> Dict[str, go.Figure]:
        """Create prediction visualization plots."""
        plots = {}
        
        if not self.predictions:
            return plots
        
        # Risk forecast plot
        forecast_fig = go.Figure()
        
        for strategy, predictions in self.predictions.items():
            timestamps = [p.timestamp for p in predictions]
            risks = [p.total_risk for p in predictions]
            lower = [p.confidence_interval[0] for p in predictions]
            upper = [p.confidence_interval[1] for p in predictions]
            
            forecast_fig.add_trace(go.Scatter(
                x=timestamps,
                y=risks,
                name=f"{strategy} Risk",
                mode="lines",
                line=dict(color="blue")
            ))
            
            forecast_fig.add_trace(go.Scatter(
                x=timestamps + timestamps[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(0,0,255,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name=f"{strategy} Confidence"
            ))
        
        forecast_fig.update_layout(
            title="Risk Forecast",
            xaxis_title="Time",
            yaxis_title="Risk Score",
            showlegend=True
        )
        plots["forecast"] = forecast_fig
        
        # Impact prediction plot
        impact_fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Memory Impact",
                "CPU Impact",
                "Latency Impact",
                "Stability Impact"
            ]
        )
        
        for strategy, predictions in self.predictions.items():
            timestamps = [p.timestamp for p in predictions]
            memory = [p.predicted_impacts["memory"] for p in predictions]
            cpu = [p.predicted_impacts["cpu"] for p in predictions]
            latency = [p.predicted_impacts["latency"] for p in predictions]
            stability = [p.predicted_impacts["stability"] for p in predictions]
            
            impact_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=memory,
                    name=f"{strategy} Memory",
                    mode="lines"
                ),
                row=1,
                col=1
            )
            
            impact_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=cpu,
                    name=f"{strategy} CPU",
                    mode="lines"
                ),
                row=1,
                col=2
            )
            
            impact_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=latency,
                    name=f"{strategy} Latency",
                    mode="lines"
                ),
                row=2,
                col=1
            )
            
            impact_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=stability,
                    name=f"{strategy} Stability",
                    mode="lines"
                ),
                row=2,
                col=2
            )
        
        impact_fig.update_layout(
            height=800,
            showlegend=True,
            title="Impact Predictions"
        )
        plots["impacts"] = impact_fig
        
        # Simulation results plot
        sim_fig = go.Figure()
        
        for strategy, predictions in self.predictions.items():
            if not predictions[0].simulation_results:
                continue
            
            for i, prediction in enumerate(predictions):
                sim_fig.add_trace(go.Histogram(
                    x=prediction.simulation_results,
                    name=f"{strategy} T+{i}",
                    opacity=0.7,
                    nbinsx=20
                ))
        
        sim_fig.update_layout(
            title="Monte Carlo Simulation Results",
            xaxis_title="Risk Score",
            yaxis_title="Frequency",
            showlegend=True,
            barmode="overlay"
        )
        plots["simulation"] = sim_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"prediction_{name}.html"))
        
        return plots

def create_risk_predictor(
    analyzer: RiskAnalyzer,
    config: Optional[PredictionConfig] = None
) -> RiskPredictor:
    """Create risk predictor."""
    return RiskPredictor(analyzer, config)

if __name__ == "__main__":
    from .risk_assessment import create_risk_analyzer
    from .strategy_recommendations import create_strategy_advisor
    from .prevention_balancing import create_prevention_balancer
    from .leak_prevention import create_leak_prevention
    from .memory_leak_detection import create_leak_detector
    from .scheduler_profiling import create_profiling_hook
    
    async def main():
        # Setup components
        profiling = create_profiling_hook()
        detector = create_leak_detector(profiling)
        prevention = create_leak_prevention(detector)
        balancer = create_prevention_balancer(prevention)
        advisor = create_strategy_advisor(balancer)
        analyzer = create_risk_analyzer(advisor)
        predictor = create_risk_predictor(analyzer)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        
        try:
            while True:
                # Get predictions for all strategies
                for strategy in analyzer.assessments:
                    result = await predictor.predict_risks(strategy)
                    print(f"\nPredictions for {strategy}:")
                    
                    for i, pred in enumerate(result.predictions):
                        print(
                            f"\nT+{i}:"
                            f"\n  Total Risk: {pred.total_risk:.2%}"
                            f"\n  Confidence: [{pred.confidence_interval[0]:.2%}, "
                            f"{pred.confidence_interval[1]:.2%}]"
                            f"\n  Failure Probability: {pred.failure_probability:.2%}"
                        )
                    
                    if result.trends:
                        print("\nTrend Analysis:")
                        print(f"  Slope: {result.trends['slope']:.3f}")
                        print(f"  R²: {result.trends['correlation']:.3f}")
                
                # Create plots
                await predictor.create_prediction_plots()
                
                await asyncio.sleep(60)
        finally:
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
