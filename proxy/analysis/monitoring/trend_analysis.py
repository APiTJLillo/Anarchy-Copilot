"""Trend analysis for adaptation performance."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .adaptation_metrics import PerformanceTracker, MetricsConfig

@dataclass
class TrendConfig:
    """Configuration for trend analysis."""
    forecast_horizon: int = 20
    seasonality_test_size: int = 100
    significance_level: float = 0.05
    smoothing_window: int = 10
    decomposition_method: str = "multiplicative"
    enable_anomaly_detection: bool = True
    confidence_level: float = 0.95
    min_samples_forecast: int = 30

@dataclass
class TrendComponents:
    """Decomposed trend components."""
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    strength: float
    period: Optional[int] = None

@dataclass
class ForecastResult:
    """Forecast results."""
    predictions: np.ndarray
    confidence_intervals: np.ndarray
    model_params: Dict[str, Any]
    accuracy: float

class TrendAnalyzer:
    """Analyze and forecast performance trends."""
    
    def __init__(
        self,
        tracker: PerformanceTracker,
        config: TrendConfig = None
    ):
        self.tracker = tracker
        self.config = config or TrendConfig()
        self.decompositions: Dict[str, Dict[str, TrendComponents]] = {}
        self.forecasts: Dict[str, Dict[str, ForecastResult]] = {}
    
    async def analyze_trends(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Analyze performance trends."""
        if preset_name not in self.tracker.windowed:
            return {"status": "no_data"}
        
        # Analyze each metric type
        analysis = {}
        metrics = self.tracker.get_current_metrics(preset_name)
        
        for window_size, window_metrics in metrics["windowed"].items():
            analysis[f"window_{window_size}"] = {}
            
            for metric_name, values in window_metrics.items():
                if not isinstance(values, list):
                    continue
                
                # Decompose trend
                decomp = await self._decompose_trend(values)
                if decomp:
                    self.decompositions.setdefault(preset_name, {})[metric_name] = decomp
                    
                    # Test for stationarity
                    stationary = await self._test_stationarity(values)
                    
                    # Detect seasonality
                    seasonal_period = await self._detect_seasonality(values)
                    
                    # Generate forecast
                    forecast = await self._forecast_trend(
                        values,
                        seasonal_period
                    )
                    if forecast:
                        self.forecasts.setdefault(preset_name, {})[metric_name] = forecast
                    
                    analysis[f"window_{window_size}"][metric_name] = {
                        "trend_strength": decomp.strength,
                        "seasonal_period": decomp.period,
                        "is_stationary": stationary,
                        "forecast": {
                            "next_value": forecast.predictions[0] if forecast else None,
                            "accuracy": forecast.accuracy if forecast else None
                        }
                    }
        
        return {
            "status": "success",
            "analysis": analysis
        }
    
    async def _decompose_trend(
        self,
        values: List[float]
    ) -> Optional[TrendComponents]:
        """Decompose trend into components."""
        if len(values) < 3:
            return None
        
        try:
            # Convert to time series
            ts = pd.Series(values)
            
            # Apply decomposition
            if self.config.decomposition_method == "multiplicative":
                decomp = ts.mul(1)  # Placeholder for actual multiplicative decomposition
            else:
                decomp = ts.mul(1)  # Placeholder for additive decomposition
            
            # Calculate trend strength
            variance_total = np.var(values)
            variance_residual = np.var(decomp.residual)
            strength = 1 - (variance_residual / variance_total)
            
            return TrendComponents(
                trend=decomp.trend,
                seasonal=decomp.seasonal,
                residual=decomp.residual,
                strength=strength
            )
            
        except Exception:
            return None
    
    async def _test_stationarity(
        self,
        values: List[float]
    ) -> bool:
        """Test for stationarity using Augmented Dickey-Fuller test."""
        if len(values) < self.config.min_samples_forecast:
            return False
        
        try:
            # Perform ADF test
            result = adfuller(values)
            return result[1] < self.config.significance_level
        except Exception:
            return False
    
    async def _detect_seasonality(
        self,
        values: List[float]
    ) -> Optional[int]:
        """Detect seasonality period."""
        if len(values) < self.config.seasonality_test_size:
            return None
        
        try:
            # Calculate autocorrelation
            acf_values = acf(values, nlags=len(values)//2)
            
            # Find peaks in ACF
            peaks = []
            for i in range(1, len(acf_values)-1):
                if (acf_values[i] > acf_values[i-1] and 
                    acf_values[i] > acf_values[i+1] and
                    acf_values[i] > 0.3):  # Correlation threshold
                    peaks.append(i)
            
            if peaks:
                # Return the first significant peak
                return peaks[0]
                
        except Exception:
            pass
        
        return None
    
    async def _forecast_trend(
        self,
        values: List[float],
        seasonal_period: Optional[int]
    ) -> Optional[ForecastResult]:
        """Generate trend forecast."""
        if len(values) < self.config.min_samples_forecast:
            return None
        
        try:
            # Prepare model
            model = ExponentialSmoothing(
                values,
                seasonal_periods=seasonal_period if seasonal_period else None,
                trend="add",
                seasonal="add" if seasonal_period else None
            )
            
            # Fit model
            fit = model.fit()
            
            # Generate forecast
            forecast = fit.forecast(self.config.forecast_horizon)
            
            # Calculate confidence intervals
            residuals = fit.resid
            std_resid = np.std(residuals)
            z_value = stats.norm.ppf((1 + self.config.confidence_level) / 2)
            ci = np.array([
                forecast - z_value * std_resid,
                forecast + z_value * std_resid
            ]).T
            
            # Calculate accuracy
            mape = np.mean(np.abs(residuals / values)) * 100
            
            return ForecastResult(
                predictions=forecast,
                confidence_intervals=ci,
                model_params=fit.params_dict,
                accuracy=100 - mape
            )
            
        except Exception:
            return None
    
    async def create_trend_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create trend visualization plots."""
        if preset_name not in self.decompositions:
            return {}
        
        plots = {}
        
        # Decomposition plot
        for metric_name, decomp in self.decompositions[preset_name].items():
            decomp_fig = make_subplots(
                rows=4,
                cols=1,
                subplot_titles=[
                    "Original",
                    "Trend",
                    "Seasonal",
                    "Residual"
                ]
            )
            
            # Original
            decomp_fig.add_trace(
                go.Scatter(y=values, name="Original"),
                row=1,
                col=1
            )
            
            # Trend
            decomp_fig.add_trace(
                go.Scatter(y=decomp.trend, name="Trend"),
                row=2,
                col=1
            )
            
            # Seasonal
            decomp_fig.add_trace(
                go.Scatter(y=decomp.seasonal, name="Seasonal"),
                row=3,
                col=1
            )
            
            # Residual
            decomp_fig.add_trace(
                go.Scatter(y=decomp.residual, name="Residual"),
                row=4,
                col=1
            )
            
            decomp_fig.update_layout(
                height=1000,
                title=f"Trend Decomposition - {metric_name}"
            )
            plots[f"decomposition_{metric_name}"] = decomp_fig
        
        # Forecast plot
        if preset_name in self.forecasts:
            for metric_name, forecast in self.forecasts[preset_name].items():
                forecast_fig = go.Figure()
                
                # Historical values
                forecast_fig.add_trace(
                    go.Scatter(
                        y=values,
                        name="Historical",
                        mode="lines"
                    )
                )
                
                # Forecast
                forecast_fig.add_trace(
                    go.Scatter(
                        y=forecast.predictions,
                        name="Forecast",
                        mode="lines",
                        line=dict(dash="dash")
                    )
                )
                
                # Confidence intervals
                ci_lower = forecast.confidence_intervals[:, 0]
                ci_upper = forecast.confidence_intervals[:, 1]
                
                forecast_fig.add_trace(
                    go.Scatter(
                        y=ci_upper,
                        name="Upper CI",
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                forecast_fig.add_trace(
                    go.Scatter(
                        y=ci_lower,
                        name="Lower CI",
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        showlegend=False
                    )
                )
                
                forecast_fig.update_layout(
                    title=f"Forecast - {metric_name} (Accuracy: {forecast.accuracy:.1f}%)",
                    yaxis_title=metric_name
                )
                plots[f"forecast_{metric_name}"] = forecast_fig
        
        return plots

def create_trend_analyzer(
    tracker: PerformanceTracker,
    config: Optional[TrendConfig] = None
) -> TrendAnalyzer:
    """Create trend analyzer."""
    return TrendAnalyzer(tracker, config)

if __name__ == "__main__":
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        
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
        
        # Generate test data with trend and seasonality
        t = np.linspace(0, 4*np.pi, 200)
        trend = 0.1 * t
        seasonal = 0.5 * np.sin(t)
        noise = np.random.normal(0, 0.1, len(t))
        values = trend + seasonal + noise
        
        # Record metrics
        for i, v in enumerate(values):
            await tracker.update_metrics("test_preset", v, v + noise[i])
        
        # Analyze trends
        analysis = await analyzer.analyze_trends("test_preset")
        print("Trend analysis:", analysis)
        
        # Create plots
        plots = await analyzer.create_trend_plots("test_preset")
        for name, fig in plots.items():
            fig.write_html(f"test_trend_{name}.html")
    
    asyncio.run(main())
