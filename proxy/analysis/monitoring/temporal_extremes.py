"""Temporal modeling for extreme value analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from scipy import stats
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .extreme_value_analysis import (
    ExtremeValueAnalyzer, ExtremeValueConfig, ExtremeParams,
    ReturnLevel, ExtremeValueResult
)

@dataclass
class TemporalConfig:
    """Configuration for temporal modeling."""
    enabled: bool = True
    update_interval: float = 300.0  # 5 minutes
    seasonality_period: int = 24  # hours
    trend_window: int = 168  # hours
    min_history: int = 100
    confidence_level: float = 0.95
    enable_seasonality: bool = True
    enable_trend: bool = True
    enable_change_point: bool = True
    change_point_threshold: float = 0.05
    max_change_points: int = 5
    smoothing_window: int = 12  # hours
    prediction_horizon: int = 24  # hours
    visualization_dir: Optional[str] = "temporal_extremes"

@dataclass
class SeasonalPattern:
    """Seasonal pattern in extreme values."""
    period: int
    amplitude: float
    phase: float
    significance: float
    confidence_interval: Tuple[float, float]

@dataclass
class TrendComponent:
    """Trend component in extreme values."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    confidence_band: Tuple[np.ndarray, np.ndarray]

@dataclass
class ChangePoint:
    """Change point in extreme value behavior."""
    timestamp: datetime
    metric: str
    old_params: ExtremeParams
    new_params: ExtremeParams
    significance: float
    direction: str  # increasing, decreasing

@dataclass
class TemporalResult:
    """Results of temporal analysis."""
    seasonal_patterns: Dict[str, List[SeasonalPattern]]
    trends: Dict[str, TrendComponent]
    change_points: List[ChangePoint]
    predictions: Dict[str, pd.Series]
    prediction_intervals: Dict[str, Tuple[pd.Series, pd.Series]]
    model_diagnostics: Dict[str, Dict[str, float]]

class TemporalAnalyzer:
    """Analyze temporal patterns in extreme values."""
    
    def __init__(
        self,
        extreme_analyzer: ExtremeValueAnalyzer,
        config: TemporalConfig = None
    ):
        self.extreme_analyzer = extreme_analyzer
        self.config = config or TemporalConfig()
        
        # Analysis state
        self.results: Dict[str, TemporalResult] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.scaler = StandardScaler()
        
        # Monitoring state
        self.last_update = datetime.min
        self.analyzer_task: Optional[asyncio.Task] = None
    
    async def start_analyzer(self):
        """Start temporal analyzer."""
        if not self.config.enabled:
            return
        
        if self.analyzer_task is None:
            self.analyzer_task = asyncio.create_task(self._run_analyzer())
    
    async def stop_analyzer(self):
        """Stop temporal analyzer."""
        if self.analyzer_task:
            self.analyzer_task.cancel()
            try:
                await self.analyzer_task
            except asyncio.CancelledError:
                pass
            self.analyzer_task = None
    
    async def _run_analyzer(self):
        """Run periodic analysis."""
        while True:
            try:
                current_time = datetime.now()
                
                if (
                    current_time - self.last_update >=
                    timedelta(seconds=self.config.update_interval)
                ):
                    for scenario in self.extreme_analyzer.results:
                        await self.analyze_temporal_patterns(scenario)
                    
                    self.last_update = current_time
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                print(f"Temporal analyzer error: {e}")
                await asyncio.sleep(60)
    
    async def analyze_temporal_patterns(
        self,
        scenario_name: str
    ) -> Optional[TemporalResult]:
        """Analyze temporal patterns in extreme values."""
        if scenario_name not in self.extreme_analyzer.results:
            return None
        
        result = self.extreme_analyzer.results[scenario_name]
        if not result.distribution_params:
            return None
        
        # Initialize results
        seasonal_patterns = {}
        trends = {}
        change_points = []
        predictions = {}
        prediction_intervals = {}
        model_diagnostics = {}
        
        for metric, params in result.distribution_params.items():
            # Get time series data
            ts_data = await self._prepare_time_series(
                scenario_name,
                metric
            )
            if len(ts_data) < self.config.min_history:
                continue
            
            # Analyze seasonality
            if self.config.enable_seasonality:
                patterns = await self._analyze_seasonality(ts_data)
                if patterns:
                    seasonal_patterns[metric] = patterns
            
            # Analyze trend
            if self.config.enable_trend:
                trend = await self._analyze_trend(ts_data)
                if trend:
                    trends[metric] = trend
            
            # Detect change points
            if self.config.enable_change_point:
                points = await self._detect_change_points(
                    ts_data,
                    params
                )
                if points:
                    change_points.extend(points)
            
            # Fit temporal model and generate predictions
            model_result = await self._fit_temporal_model(
                ts_data,
                metric
            )
            if model_result:
                model, diagnostics = model_result
                predictions[metric] = model.forecast(
                    self.config.prediction_horizon
                )
                intervals = model.get_forecast(
                    self.config.prediction_horizon
                ).conf_int(
                    alpha=1 - self.config.confidence_level
                )
                prediction_intervals[metric] = (
                    intervals.iloc[:, 0],
                    intervals.iloc[:, 1]
                )
                model_diagnostics[metric] = diagnostics
        
        # Create result
        result = TemporalResult(
            seasonal_patterns=seasonal_patterns,
            trends=trends,
            change_points=change_points,
            predictions=predictions,
            prediction_intervals=prediction_intervals,
            model_diagnostics=model_diagnostics
        )
        
        self.results[scenario_name] = result
        
        return result
    
    async def _prepare_time_series(
        self,
        scenario: str,
        metric: str
    ) -> pd.Series:
        """Prepare time series data for analysis."""
        # Get threshold exceedances
        exceedances = self.extreme_analyzer.results[scenario].threshold_exceedances[metric]
        
        # Convert to time series
        timestamps = [
            self.extreme_analyzer.modeler.last_update -
            timedelta(hours=i)
            for i in range(len(exceedances))
        ]
        
        return pd.Series(
            exceedances,
            index=pd.DatetimeIndex(timestamps[::-1])
        ).sort_index()
    
    async def _analyze_seasonality(
        self,
        ts_data: pd.Series
    ) -> List[SeasonalPattern]:
        """Analyze seasonal patterns in time series."""
        patterns = []
        
        # Detrend data
        trend = np.polyfit(
            np.arange(len(ts_data)),
            ts_data.values,
            1
        )
        detrended = ts_data.values - np.polyval(
            trend,
            np.arange(len(ts_data))
        )
        
        # Perform spectral analysis
        freqs = np.fft.fftfreq(len(detrended))
        spectrum = np.abs(np.fft.fft(detrended))
        
        # Find significant periods
        significant_idx = np.where(
            spectrum > np.percentile(spectrum, 95)
        )[0]
        
        for idx in significant_idx:
            if freqs[idx] == 0:
                continue
                
            period = int(1 / abs(freqs[idx]))
            if period > len(ts_data) // 2:
                continue
            
            # Calculate amplitude and phase
            fft_vals = np.fft.fft(detrended)[idx]
            amplitude = np.abs(fft_vals) / len(detrended)
            phase = np.angle(fft_vals)
            
            # Calculate significance
            significance = spectrum[idx] / np.mean(spectrum)
            
            # Calculate confidence interval
            ci = stats.norm.interval(
                self.config.confidence_level,
                loc=amplitude,
                scale=amplitude / np.sqrt(len(ts_data))
            )
            
            patterns.append(SeasonalPattern(
                period=period,
                amplitude=amplitude,
                phase=phase,
                significance=significance,
                confidence_interval=ci
            ))
        
        return sorted(patterns, key=lambda x: x.significance, reverse=True)
    
    async def _analyze_trend(
        self,
        ts_data: pd.Series
    ) -> Optional[TrendComponent]:
        """Analyze trend component in time series."""
        times = np.arange(len(ts_data))
        values = ts_data.values
        
        # Fit trend line
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            times,
            values
        )
        
        if p_value > (1 - self.config.confidence_level):
            return None
        
        # Calculate confidence band
        mean_x = np.mean(times)
        n = len(times)
        
        def confidence_band(x):
            se = std_err * np.sqrt(
                1/n +
                (x - mean_x)**2 /
                np.sum((times - mean_x)**2)
            )
            return stats.t.interval(
                self.config.confidence_level,
                n-2,
                loc=slope*x + intercept,
                scale=se
            )
        
        conf_lower = []
        conf_upper = []
        
        for x in times:
            lower, upper = confidence_band(x)
            conf_lower.append(lower)
            conf_upper.append(upper)
        
        return TrendComponent(
            slope=slope,
            intercept=intercept,
            r_squared=r_value**2,
            p_value=p_value,
            confidence_band=(
                np.array(conf_lower),
                np.array(conf_upper)
            )
        )
    
    async def _detect_change_points(
        self,
        ts_data: pd.Series,
        params: ExtremeParams
    ) -> List[ChangePoint]:
        """Detect change points in extreme value behavior."""
        points = []
        
        # Use sliding window
        window_size = len(ts_data) // 4
        if window_size < self.config.min_history:
            return points
        
        for i in range(0, len(ts_data) - window_size):
            before = ts_data.iloc[i:i+window_size]
            after = ts_data.iloc[i+window_size:i+2*window_size]
            
            if len(before) < self.config.min_history or len(after) < self.config.min_history:
                continue
            
            # Compare distributions
            stat, p_value = stats.ks_2samp(before, after)
            
            if p_value < self.config.change_point_threshold:
                # Fit GPD to both windows
                try:
                    before_params = await self.extreme_analyzer._fit_gpd(
                        before.values,
                        params.threshold
                    )
                    after_params = await self.extreme_analyzer._fit_gpd(
                        after.values,
                        params.threshold
                    )
                    
                    # Determine direction
                    direction = (
                        "increasing"
                        if after_params.shape > before_params.shape
                        else "decreasing"
                    )
                    
                    points.append(ChangePoint(
                        timestamp=ts_data.index[i+window_size],
                        metric=params.name,
                        old_params=before_params,
                        new_params=after_params,
                        significance=1 - p_value,
                        direction=direction
                    ))
                    
                    if len(points) >= self.config.max_change_points:
                        break
                    
                except:
                    continue
        
        return sorted(points, key=lambda x: x.significance, reverse=True)
    
    async def _fit_temporal_model(
        self,
        ts_data: pd.Series,
        metric: str
    ) -> Optional[Tuple[Any, Dict[str, float]]]:
        """Fit temporal model to time series."""
        try:
            # Determine optimal order
            best_aic = np.inf
            best_order = None
            best_model = None
            
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            model = SARIMAX(
                                ts_data,
                                order=(p, d, q),
                                seasonal_order=(1, 1, 1, self.config.seasonality_period)
                                if self.config.enable_seasonality else (0, 0, 0, 0)
                            )
                            results = model.fit(disp=False)
                            
                            if results.aic < best_aic:
                                best_aic = results.aic
                                best_order = (p, d, q)
                                best_model = results
                        except:
                            continue
            
            if best_model is None:
                return None
            
            # Calculate diagnostics
            diagnostics = {
                "aic": best_model.aic,
                "bic": best_model.bic,
                "mae": np.mean(np.abs(best_model.resid)),
                "rmse": np.sqrt(np.mean(best_model.resid**2)),
                "mape": np.mean(np.abs(best_model.resid / ts_data)) * 100
            }
            
            return best_model, diagnostics
            
        except Exception as e:
            print(f"Model fitting error: {e}")
            return None
    
    async def create_temporal_plots(self) -> Dict[str, go.Figure]:
        """Create temporal analysis visualization plots."""
        plots = {}
        
        for scenario_name, result in self.results.items():
            if not result.seasonal_patterns:
                continue
            
            # Seasonal pattern plot
            season_fig = go.Figure()
            
            for metric, patterns in result.seasonal_patterns.items():
                times = np.linspace(0, 2*np.pi, 100)
                
                for pattern in patterns:
                    values = pattern.amplitude * np.cos(
                        times + pattern.phase
                    )
                    
                    season_fig.add_trace(go.Scatter(
                        x=times,
                        y=values,
                        name=f"{metric} ({pattern.period}h)",
                        text=[
                            f"Significance: {pattern.significance:.2f}"
                            for _ in times
                        ]
                    ))
            
            season_fig.update_layout(
                title=f"Seasonal Patterns - {scenario_name}",
                xaxis_title="Phase",
                yaxis_title="Amplitude",
                showlegend=True
            )
            plots[f"{scenario_name}_seasonality"] = season_fig
            
            # Trend plot
            trend_fig = go.Figure()
            
            for metric, trend in result.trends.items():
                times = np.arange(self.config.trend_window)
                values = trend.slope * times + trend.intercept
                
                trend_fig.add_trace(go.Scatter(
                    x=times,
                    y=values,
                    name=f"{metric} Trend",
                    mode="lines"
                ))
                
                trend_fig.add_trace(go.Scatter(
                    x=np.concatenate([times, times[::-1]]),
                    y=np.concatenate([
                        trend.confidence_band[1],
                        trend.confidence_band[0][::-1]
                    ]),
                    fill="toself",
                    fillcolor="rgba(0,0,255,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{metric} Confidence"
                ))
            
            trend_fig.update_layout(
                title=f"Trend Analysis - {scenario_name}",
                xaxis_title="Time (hours)",
                yaxis_title="Value",
                showlegend=True
            )
            plots[f"{scenario_name}_trends"] = trend_fig
            
            # Change point plot
            if result.change_points:
                change_fig = go.Figure()
                
                for point in result.change_points:
                    change_fig.add_vline(
                        x=point.timestamp,
                        line_dash="dash",
                        annotation_text=(
                            f"{point.metric}\n"
                            f"({point.direction})"
                        ),
                        line_color=(
                            "red" if point.direction == "increasing"
                            else "blue"
                        )
                    )
                
                change_fig.update_layout(
                    title=f"Change Points - {scenario_name}",
                    xaxis_title="Time",
                    yaxis_title="Value",
                    showlegend=True
                )
                plots[f"{scenario_name}_changes"] = change_fig
            
            # Prediction plot
            pred_fig = go.Figure()
            
            for metric, predictions in result.predictions.items():
                lower, upper = result.prediction_intervals[metric]
                
                pred_fig.add_trace(go.Scatter(
                    x=predictions.index,
                    y=predictions.values,
                    name=f"{metric} Forecast",
                    mode="lines"
                ))
                
                pred_fig.add_trace(go.Scatter(
                    x=pd.concat([
                        predictions.index,
                        predictions.index[::-1]
                    ]),
                    y=pd.concat([upper, lower[::-1]]),
                    fill="toself",
                    fillcolor="rgba(0,0,255,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{metric} Confidence"
                ))
            
            pred_fig.update_layout(
                title=f"Predictions - {scenario_name}",
                xaxis_title="Time",
                yaxis_title="Value",
                showlegend=True
            )
            plots[f"{scenario_name}_predictions"] = pred_fig
        
        # Save plots
        if self.config.visualization_dir:
            from pathlib import Path
            path = Path(self.config.visualization_dir)
            path.mkdir(parents=True, exist_ok=True)
            for name, fig in plots.items():
                fig.write_html(str(path / f"temporal_{name}.html"))
        
        return plots

def create_temporal_analyzer(
    extreme_analyzer: ExtremeValueAnalyzer,
    config: Optional[TemporalConfig] = None
) -> TemporalAnalyzer:
    """Create temporal analyzer."""
    return TemporalAnalyzer(extreme_analyzer, config)

if __name__ == "__main__":
    from .extreme_value_analysis import create_extreme_analyzer
    from .probabilistic_modeling import create_probabilistic_modeler
    from .whatif_analysis import create_whatif_analyzer
    from .scenario_planning import create_scenario_planner
    from .risk_prediction import create_risk_predictor
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
        planner = create_scenario_planner(predictor)
        whatif = create_whatif_analyzer(planner)
        modeler = create_probabilistic_modeler(whatif)
        extreme = create_extreme_analyzer(modeler)
        temporal = create_temporal_analyzer(extreme)
        
        # Start components
        await prevention.start_prevention()
        await balancer.start_balancing()
        await advisor.start_advisor()
        await analyzer.start_analyzer()
        await predictor.start_predictor()
        await planner.start_planner()
        await whatif.start_analyzer()
        await modeler.start_modeler()
        await extreme.start_analyzer()
        await temporal.start_analyzer()
        
        try:
            while True:
                # Analyze scenarios
                for scenario in extreme.results:
                    result = await temporal.analyze_temporal_patterns(scenario)
                    if result:
                        print(f"\nTemporal Analysis for {scenario}:")
                        
                        for metric, patterns in result.seasonal_patterns.items():
                            print(f"\n{metric} Seasonal Patterns:")
                            for pattern in patterns:
                                print(
                                    f"  Period: {pattern.period}h"
                                    f", Amplitude: {pattern.amplitude:.3f}"
                                    f", Significance: {pattern.significance:.2f}"
                                )
                        
                        for metric, trend in result.trends.items():
                            print(f"\n{metric} Trend:")
                            print(
                                f"  Slope: {trend.slope:.3e}"
                                f", R²: {trend.r_squared:.3f}"
                                f", p-value: {trend.p_value:.3e}"
                            )
                        
                        if result.change_points:
                            print("\nChange Points:")
                            for point in result.change_points:
                                print(
                                    f"  {point.timestamp}: {point.metric}"
                                    f" ({point.direction})"
                                    f", Significance: {point.significance:.2f}"
                                )
                
                # Create plots
                await temporal.create_temporal_plots()
                
                await asyncio.sleep(60)
        finally:
            await temporal.stop_analyzer()
            await extreme.stop_analyzer()
            await modeler.stop_modeler()
            await whatif.stop_analyzer()
            await planner.stop_planner()
            await predictor.stop_predictor()
            await analyzer.stop_analyzer()
            await advisor.stop_advisor()
            await balancer.stop_balancing()
            await prevention.stop_prevention()
    
    asyncio.run(main())
