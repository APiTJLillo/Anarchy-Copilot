"""Trend analysis for collaborative exploration patterns."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .collaborative_recommendations import CollaborativeRecommender, CollaborationConfig
from .solution_recommendations import SolutionRecommender

logger = logging.getLogger(__name__)

@dataclass
class TrendConfig:
    """Configuration for trend analysis."""
    forecast_horizon: int = 24  # hours
    window_size: int = 48  # hours
    seasonality_period: int = 24  # hours
    confidence_level: float = 0.95
    polynomial_degree: int = 2
    output_path: Optional[Path] = None

class ExplorationTrendAnalyzer:
    """Analyze and predict exploration trends."""
    
    def __init__(
        self,
        collab: CollaborativeRecommender,
        config: TrendConfig
    ):
        self.collab = collab
        self.config = config
        
        # Initialize models
        self.models = {
            "ridge": Ridge(alpha=1.0),
            "poly": PolynomialFeatures(degree=config.polynomial_degree)
        }
        
        # State management
        self.trend_cache = {}
        self.forecasts = {}
        self.patterns = {}
    
    def analyze_trends(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze exploration trends."""
        results = {
            "activity": self._analyze_activity_trends(start_time, end_time),
            "preferences": self._analyze_preference_trends(start_time, end_time),
            "solutions": self._analyze_solution_trends(start_time, end_time),
            "users": self._analyze_user_trends(start_time, end_time),
            "forecasts": self._generate_forecasts(start_time, end_time)
        }
        
        # Cache results
        self.trend_cache[datetime.now().isoformat()] = results
        
        return results
    
    def visualize_trends(
        self,
        trends: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of trend analysis."""
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Activity Patterns",
                "Preference Evolution",
                "Solution Popularity",
                "User Engagement",
                "Time Series Forecasts",
                "Pattern Detection"
            ]
        )
        
        # Activity patterns
        activity = trends["activity"]
        fig.add_trace(
            go.Scatter(
                x=activity["time_series"]["timestamps"],
                y=activity["time_series"]["values"],
                mode="lines",
                name="Activity"
            ),
            row=1,
            col=1
        )
        
        # Add trend line
        if "trend" in activity:
            fig.add_trace(
                go.Scatter(
                    x=activity["time_series"]["timestamps"],
                    y=activity["trend"]["values"],
                    mode="lines",
                    line=dict(dash="dash"),
                    name="Activity Trend"
                ),
                row=1,
                col=1
            )
        
        # Preference evolution
        prefs = trends["preferences"]
        for obj, values in prefs["evolution"].items():
            fig.add_trace(
                go.Scatter(
                    x=values["timestamps"],
                    y=values["weights"],
                    mode="lines+markers",
                    name=f"{obj} Weight"
                ),
                row=1,
                col=2
            )
        
        # Solution popularity
        solutions = trends["solutions"]
        fig.add_trace(
            go.Heatmap(
                z=solutions["popularity_matrix"],
                x=solutions["timestamps"],
                y=solutions["solution_ids"],
                colorscale="Viridis",
                name="Solution Popularity"
            ),
            row=2,
            col=1
        )
        
        # User engagement
        users = trends["users"]
        fig.add_trace(
            go.Scatter(
                x=users["time_series"]["timestamps"],
                y=users["time_series"]["active_users"],
                mode="lines",
                name="Active Users"
            ),
            row=2,
            col=2
        )
        
        # Forecasts
        forecasts = trends["forecasts"]
        for metric, forecast in forecasts.items():
            fig.add_trace(
                go.Scatter(
                    x=forecast["timestamps"],
                    y=forecast["values"],
                    mode="lines",
                    line=dict(dash="dot"),
                    name=f"{metric} Forecast"
                ),
                row=3,
                col=1
            )
            
            # Add confidence intervals
            fig.add_trace(
                go.Scatter(
                    x=forecast["timestamps"] + forecast["timestamps"][::-1],
                    y=list(forecast["upper_bound"]) + list(forecast["lower_bound"])[::-1],
                    fill="toself",
                    fillcolor="rgba(0,100,80,0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"{metric} Confidence"
                ),
                row=3,
                col=1
            )
        
        # Pattern detection
        if self.patterns:
            pattern_x = []
            pattern_y = []
            pattern_text = []
            
            for pattern in self.patterns.values():
                pattern_x.append(pattern["start_time"])
                pattern_y.append(pattern["strength"])
                pattern_text.append(pattern["description"])
            
            fig.add_trace(
                go.Scatter(
                    x=pattern_x,
                    y=pattern_y,
                    mode="markers+text",
                    text=pattern_text,
                    textposition="top center",
                    marker=dict(size=10),
                    name="Detected Patterns"
                ),
                row=3,
                col=2
            )
        
        return fig
    
    def _analyze_activity_trends(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Analyze exploration activity trends."""
        # Get activity data
        activity_data = []
        timestamps = []
        
        for user_id, interactions in self.collab.user_interactions.items():
            for interaction in interactions:
                ts = datetime.fromisoformat(interaction["timestamp"])
                if (
                    (not start_time or ts >= start_time) and
                    (not end_time or ts <= end_time)
                ):
                    activity_data.append(1)
                    timestamps.append(ts)
        
        if not activity_data:
            return {
                "time_series": {
                    "timestamps": [],
                    "values": []
                }
            }
        
        # Create time series
        df = pd.DataFrame({
            "timestamp": timestamps,
            "activity": activity_data
        })
        
        df = df.set_index("timestamp").resample("H").sum()
        
        # Analyze trend
        X = np.arange(len(df)).reshape(-1, 1)
        y = df["activity"].values
        
        self.models["ridge"].fit(X, y)
        trend = self.models["ridge"].predict(X)
        
        # Detect seasonality
        if len(df) >= 2 * self.config.seasonality_period:
            seasonal = bool(
                stats.pearsonr(
                    df["activity"].values[:-self.config.seasonality_period],
                    df["activity"].values[self.config.seasonality_period:]
                )[0] > 0.7
            )
        else:
            seasonal = False
        
        return {
            "time_series": {
                "timestamps": df.index.tolist(),
                "values": df["activity"].tolist()
            },
            "trend": {
                "values": trend.tolist(),
                "slope": float(self.models["ridge"].coef_[0]),
                "strength": float(np.corrcoef(y, trend)[0, 1])
            },
            "seasonality": {
                "detected": seasonal,
                "period": self.config.seasonality_period if seasonal else None
            },
            "statistics": {
                "mean": float(np.mean(y)),
                "std": float(np.std(y)),
                "autocorr": float(pd.Series(y).autocorr())
            }
        }
    
    def _analyze_preference_trends(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Analyze preference evolution trends."""
        preference_history = {}
        
        # Collect preference history
        for user_id, profiles in self.collab.user_profiles.items():
            for profile in profiles:
                ts = datetime.fromisoformat(profile["timestamp"])
                if (
                    (not start_time or ts >= start_time) and
                    (not end_time or ts <= end_time)
                ):
                    for obj, weight in profile["weights"].items():
                        if obj not in preference_history:
                            preference_history[obj] = {
                                "timestamps": [],
                                "weights": []
                            }
                        preference_history[obj]["timestamps"].append(ts)
                        preference_history[obj]["weights"].append(weight)
        
        if not preference_history:
            return {
                "evolution": {},
                "trends": {}
            }
        
        # Analyze trends
        trends = {}
        for obj, history in preference_history.items():
            if len(history["weights"]) > 1:
                slope, intercept = np.polyfit(
                    range(len(history["weights"])),
                    history["weights"],
                    1
                )
                
                trends[obj] = {
                    "slope": float(slope),
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "strength": float(
                        abs(
                            stats.pearsonr(
                                range(len(history["weights"])),
                                history["weights"]
                            )[0]
                        )
                    )
                }
        
        return {
            "evolution": preference_history,
            "trends": trends
        }
    
    def _analyze_solution_trends(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Analyze solution popularity trends."""
        # Create popularity matrix
        solutions = self.collab.recommender.explorer.optimizer.optimization_results["pareto_front"]["solutions"]
        solution_ids = [s["id"] for s in solutions]
        
        if not solution_ids:
            return {
                "popularity_matrix": [],
                "solution_ids": [],
                "timestamps": []
            }
        
        # Collect interaction timestamps
        timestamps = set()
        for interactions in self.collab.user_interactions.values():
            for interaction in interactions:
                ts = datetime.fromisoformat(interaction["timestamp"])
                if (
                    (not start_time or ts >= start_time) and
                    (not end_time or ts <= end_time)
                ):
                    timestamps.add(ts)
        
        timestamps = sorted(timestamps)
        
        if not timestamps:
            return {
                "popularity_matrix": [],
                "solution_ids": [],
                "timestamps": []
            }
        
        # Create popularity matrix
        popularity = np.zeros((len(solution_ids), len(timestamps)))
        
        for i, sol_id in enumerate(solution_ids):
            for j, ts in enumerate(timestamps):
                count = 0
                for interactions in self.collab.user_interactions.values():
                    for interaction in interactions:
                        if (
                            interaction["solution_id"] == sol_id and
                            datetime.fromisoformat(interaction["timestamp"]) <= ts
                        ):
                            count += 1
                popularity[i, j] = count
        
        return {
            "popularity_matrix": popularity.tolist(),
            "solution_ids": solution_ids,
            "timestamps": timestamps
        }
    
    def _analyze_user_trends(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Analyze user engagement trends."""
        if not self.collab.user_interactions:
            return {
                "time_series": {
                    "timestamps": [],
                    "active_users": []
                }
            }
        
        # Count active users over time
        active_users = []
        timestamps = []
        
        min_time = start_time or min(
            datetime.fromisoformat(interaction["timestamp"])
            for interactions in self.collab.user_interactions.values()
            for interaction in interactions
        )
        
        max_time = end_time or max(
            datetime.fromisoformat(interaction["timestamp"])
            for interactions in self.collab.user_interactions.values()
            for interaction in interactions
        )
        
        current = min_time
        while current <= max_time:
            active = set()
            
            for user_id, interactions in self.collab.user_interactions.items():
                for interaction in interactions:
                    ts = datetime.fromisoformat(interaction["timestamp"])
                    if ts <= current and ts > current - timedelta(hours=self.config.window_size):
                        active.add(user_id)
            
            timestamps.append(current)
            active_users.append(len(active))
            current += timedelta(hours=1)
        
        return {
            "time_series": {
                "timestamps": timestamps,
                "active_users": active_users
            },
            "statistics": {
                "total_users": len(self.collab.user_profiles),
                "avg_active": float(np.mean(active_users)),
                "peak_active": max(active_users),
                "engagement_rate": float(np.mean(active_users) / len(self.collab.user_profiles))
            }
        }
    
    def _generate_forecasts(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Generate time series forecasts."""
        forecasts = {}
        
        # Activity forecast
        activity = self._analyze_activity_trends(start_time, end_time)
        if activity["time_series"]["values"]:
            activity_forecast = self._forecast_timeseries(
                activity["time_series"]["values"],
                activity["seasonality"]["detected"]
            )
            
            forecasts["activity"] = {
                "timestamps": [
                    activity["time_series"]["timestamps"][-1] + timedelta(hours=i)
                    for i in range(1, self.config.forecast_horizon + 1)
                ],
                "values": activity_forecast["forecast"],
                "upper_bound": activity_forecast["upper_bound"],
                "lower_bound": activity_forecast["lower_bound"]
            }
        
        # User engagement forecast
        users = self._analyze_user_trends(start_time, end_time)
        if users["time_series"]["active_users"]:
            engagement_forecast = self._forecast_timeseries(
                users["time_series"]["active_users"],
                False
            )
            
            forecasts["engagement"] = {
                "timestamps": [
                    users["time_series"]["timestamps"][-1] + timedelta(hours=i)
                    for i in range(1, self.config.forecast_horizon + 1)
                ],
                "values": engagement_forecast["forecast"],
                "upper_bound": engagement_forecast["upper_bound"],
                "lower_bound": engagement_forecast["lower_bound"]
            }
        
        return forecasts
    
    def _forecast_timeseries(
        self,
        values: List[float],
        seasonal: bool
    ) -> Dict[str, Any]:
        """Generate time series forecast."""
        if len(values) < 2:
            return {
                "forecast": [],
                "upper_bound": [],
                "lower_bound": []
            }
        
        try:
            # Fit SARIMA model
            if seasonal:
                model = SARIMAX(
                    values,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, self.config.seasonality_period)
                )
            else:
                model = SARIMAX(
                    values,
                    order=(1, 1, 1)
                )
            
            results = model.fit(disp=False)
            
            # Generate forecast
            forecast = results.get_forecast(
                steps=self.config.forecast_horizon,
                alpha=1 - self.config.confidence_level
            )
            
            return {
                "forecast": forecast.predicted_mean.tolist(),
                "upper_bound": forecast.conf_int()[:, 1].tolist(),
                "lower_bound": forecast.conf_int()[:, 0].tolist()
            }
            
        except Exception as e:
            logger.warning(f"Forecasting failed: {e}")
            return {
                "forecast": [],
                "upper_bound": [],
                "lower_bound": []
            }
    
    def detect_patterns(self) -> Dict[str, Any]:
        """Detect recurring patterns in exploration."""
        if not self.collab.user_interactions:
            return {}
        
        patterns = {}
        
        # Activity patterns
        activity = self._analyze_activity_trends(None, None)
        if activity["time_series"]["values"]:
            # Daily patterns
            daily_pattern = self._detect_daily_pattern(
                activity["time_series"]["values"],
                activity["time_series"]["timestamps"]
            )
            if daily_pattern:
                patterns["daily_activity"] = daily_pattern
            
            # Weekly patterns
            weekly_pattern = self._detect_weekly_pattern(
                activity["time_series"]["values"],
                activity["time_series"]["timestamps"]
            )
            if weekly_pattern:
                patterns["weekly_activity"] = weekly_pattern
        
        # Preference patterns
        preferences = self._analyze_preference_trends(None, None)
        for obj, trend in preferences["trends"].items():
            if trend["strength"] > 0.7:
                patterns[f"preference_{obj}"] = {
                    "type": "preference_shift",
                    "object": obj,
                    "trend": trend["trend"],
                    "strength": trend["strength"],
                    "description": f"Strong {trend['trend']} trend in {obj} preference",
                    "start_time": preferences["evolution"][obj]["timestamps"][0]
                }
        
        # Update patterns
        self.patterns = patterns
        
        return patterns
    
    def _detect_daily_pattern(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Detect daily activity patterns."""
        if len(values) < 24:
            return None
        
        # Group by hour
        hourly = pd.DataFrame({
            "value": values,
            "hour": [ts.hour for ts in timestamps]
        }).groupby("hour").mean()
        
        # Check for significant variation
        if hourly["value"].std() / hourly["value"].mean() > 0.2:
            peak_hours = hourly.nlargest(3, "value").index.tolist()
            
            return {
                "type": "daily_cycle",
                "peak_hours": peak_hours,
                "strength": float(hourly["value"].std() / hourly["value"].mean()),
                "description": f"Daily activity peaks at hours {peak_hours}",
                "start_time": timestamps[0]
            }
        
        return None
    
    def _detect_weekly_pattern(
        self,
        values: List[float],
        timestamps: List[datetime]
    ) -> Optional[Dict[str, Any]]:
        """Detect weekly activity patterns."""
        if len(values) < 168:  # 7 days * 24 hours
            return None
        
        # Group by day of week
        daily = pd.DataFrame({
            "value": values,
            "day": [ts.strftime("%A") for ts in timestamps]
        }).groupby("day").mean()
        
        # Check for significant variation
        if daily["value"].std() / daily["value"].mean() > 0.2:
            peak_days = daily.nlargest(3, "value").index.tolist()
            
            return {
                "type": "weekly_cycle",
                "peak_days": peak_days,
                "strength": float(daily["value"].std() / daily["value"].mean()),
                "description": f"Weekly activity peaks on {', '.join(peak_days)}",
                "start_time": timestamps[0]
            }
        
        return None
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save trend analysis results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis
            analysis_path = path / "trend_analysis.json"
            with open(analysis_path, "w") as f:
                json.dump(
                    {
                        k: v for k, v in analysis.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_trends(analysis)
            viz.write_html(str(path / "trend_analysis.html"))
            
            # Save detected patterns
            if self.patterns:
                patterns_path = path / "detected_patterns.json"
                with open(patterns_path, "w") as f:
                    json.dump(self.patterns, f, indent=2)
            
            logger.info(f"Saved trend analysis to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_trend_analyzer(
    collab: CollaborativeRecommender,
    output_path: Optional[Path] = None
) -> ExplorationTrendAnalyzer:
    """Create trend analyzer."""
    config = TrendConfig(output_path=output_path)
    return ExplorationTrendAnalyzer(collab, config)

if __name__ == "__main__":
    # Example usage
    from .collaborative_recommendations import create_collaborative_recommender
    from .solution_recommendations import create_solution_recommender
    from .interactive_optimization import create_interactive_explorer
    from .multi_objective_optimization import create_multi_objective_optimizer
    from .simulation_optimization import create_simulation_optimizer
    from .monte_carlo_power import create_monte_carlo_analyzer
    from .power_analysis import create_chain_power_analyzer
    from .statistical_comparison import create_chain_statistician
    from .comparison_animation import create_chain_comparator
    from .chain_animation import create_chain_animator
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    
    # Create components
    filters = create_learning_filter()
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(animator)
    statistician = create_chain_statistician(comparator)
    power_analyzer = create_chain_power_analyzer(statistician)
    mc_analyzer = create_monte_carlo_analyzer(power_analyzer)
    sim_optimizer = create_simulation_optimizer(mc_analyzer)
    mo_optimizer = create_multi_objective_optimizer(sim_optimizer)
    explorer = create_interactive_explorer(mo_optimizer)
    recommender = create_solution_recommender(explorer)
    collab = create_collaborative_recommender(recommender)
    analyzer = create_trend_analyzer(
        collab,
        output_path=Path("trend_analysis")
    )
    
    # Add some example interactions
    for i in range(48):  # 2 days of data
        hour = i % 24
        # Simulate daily pattern
        if 9 <= hour <= 17:  # Active during work hours
            n_users = np.random.randint(3, 8)
        else:
            n_users = np.random.randint(0, 3)
        
        for _ in range(n_users):
            user_id = f"user_{np.random.randint(1, 10)}"
            collab.add_user_interaction(
                user_id,
                {
                    "power_error": np.random.uniform(0.3, 0.7),
                    "computation_cost": np.random.uniform(0.2, 0.5),
                    "stability": np.random.uniform(0.1, 0.4)
                },
                np.random.choice(5, 3).tolist()
            )
    
    # Analyze trends
    trends = analyzer.analyze_trends()
    
    # Detect patterns
    patterns = analyzer.detect_patterns()
    
    # Visualize results
    viz = analyzer.visualize_trends(trends)
    viz.show()
    
    # Save analysis
    analyzer.save_analysis(trends)
