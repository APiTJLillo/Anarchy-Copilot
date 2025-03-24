"""Anomaly detection for exploration trends."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .exploration_trends import ExplorationTrendAnalyzer, TrendConfig
from .collaborative_recommendations import CollaborativeRecommender

logger = logging.getLogger(__name__)

@dataclass
class AnomalyConfig:
    """Configuration for anomaly detection."""
    contamination: float = 0.1
    window_size: int = 24  # hours
    n_components: int = 3
    zscore_threshold: float = 3.0
    output_path: Optional[Path] = None

class ExplorationAnomalyDetector:
    """Detect anomalies in exploration patterns."""
    
    def __init__(
        self,
        analyzer: ExplorationTrendAnalyzer,
        config: AnomalyConfig
    ):
        self.analyzer = analyzer
        self.config = config
        
        # Initialize models
        self.models = {
            "isolation": IsolationForest(
                contamination=config.contamination,
                random_state=42
            ),
            "envelope": EllipticEnvelope(
                contamination=config.contamination,
                random_state=42
            )
        }
        
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=config.n_components)
        
        # State management
        self.anomaly_history = []
        self.feature_importance = {}
        self.current_context = None
    
    def detect_anomalies(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in exploration patterns."""
        results = {
            "activity": self._detect_activity_anomalies(start_time, end_time),
            "preferences": self._detect_preference_anomalies(start_time, end_time),
            "user": self._detect_user_anomalies(start_time, end_time),
            "collective": self._detect_collective_anomalies(start_time, end_time),
            "context": self._analyze_anomaly_context()
        }
        
        # Update history
        self.anomaly_history.append({
            "timestamp": datetime.now().isoformat(),
            "results": results
        })
        
        return results
    
    def visualize_anomalies(
        self,
        anomalies: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of detected anomalies."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Activity Anomalies",
                "Preference Anomalies",
                "User Anomalies",
                "Collective Anomalies"
            ]
        )
        
        # Activity anomalies
        activity = anomalies["activity"]
        if activity["time_series"]:
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
            
            # Add anomaly points
            anomaly_idx = np.where(activity["anomalies"])[0]
            fig.add_trace(
                go.Scatter(
                    x=[activity["time_series"]["timestamps"][i] for i in anomaly_idx],
                    y=[activity["time_series"]["values"][i] for i in anomaly_idx],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color="red",
                        symbol="x"
                    ),
                    name="Activity Anomalies"
                ),
                row=1,
                col=1
            )
        
        # Preference anomalies
        prefs = anomalies["preferences"]
        if prefs["patterns"]:
            for obj, pattern in prefs["patterns"].items():
                fig.add_trace(
                    go.Scatter(
                        x=pattern["timestamps"],
                        y=pattern["values"],
                        mode="lines",
                        name=f"{obj} Preference"
                    ),
                    row=1,
                    col=2
                )
                
                # Add anomaly points
                anomaly_idx = np.where(pattern["anomalies"])[0]
                fig.add_trace(
                    go.Scatter(
                        x=[pattern["timestamps"][i] for i in anomaly_idx],
                        y=[pattern["values"][i] for i in anomaly_idx],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            symbol="x"
                        ),
                        name=f"{obj} Anomalies"
                    ),
                    row=1,
                    col=2
                )
        
        # User anomalies
        users = anomalies["user"]
        if users["patterns"]:
            fig.add_trace(
                go.Heatmap(
                    z=users["patterns"]["behavior_matrix"],
                    x=users["patterns"]["timestamps"],
                    y=users["patterns"]["user_ids"],
                    colorscale="Viridis",
                    name="User Behavior"
                ),
                row=2,
                col=1
            )
            
            # Add anomaly markers
            anomaly_users = users["anomalies"]["users"]
            anomaly_times = users["anomalies"]["timestamps"]
            
            fig.add_trace(
                go.Scatter(
                    x=anomaly_times,
                    y=anomaly_users,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color="red",
                        symbol="x"
                    ),
                    name="User Anomalies"
                ),
                row=2,
                col=1
            )
        
        # Collective anomalies
        collective = anomalies["collective"]
        if collective["components"]:
            # Plot principal components
            for i, component in enumerate(collective["components"]):
                fig.add_trace(
                    go.Scatter(
                        x=component["timestamps"],
                        y=component["values"],
                        mode="lines",
                        name=f"PC{i+1}"
                    ),
                    row=2,
                    col=2
                )
                
                # Add anomaly points
                anomaly_idx = np.where(component["anomalies"])[0]
                fig.add_trace(
                    go.Scatter(
                        x=[component["timestamps"][i] for i in anomaly_idx],
                        y=[component["values"][i] for i in anomaly_idx],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            symbol="x"
                        ),
                        name=f"PC{i+1} Anomalies"
                    ),
                    row=2,
                    col=2
                )
        
        return fig
    
    def _detect_activity_anomalies(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Detect anomalies in activity patterns."""
        # Get activity data
        activity = self.analyzer._analyze_activity_trends(start_time, end_time)
        
        if not activity["time_series"]["values"]:
            return {
                "time_series": None,
                "anomalies": [],
                "scores": []
            }
        
        # Prepare features
        X = np.array(activity["time_series"]["values"]).reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Detect anomalies using multiple methods
        isolation_scores = self.models["isolation"].fit_predict(X_scaled)
        envelope_scores = self.models["envelope"].fit_predict(X_scaled)
        zscore_scores = np.abs(stats.zscore(X_scaled.flatten()))
        
        # Combine detection methods
        anomalies = (
            (isolation_scores == -1) |
            (envelope_scores == -1) |
            (zscore_scores > self.config.zscore_threshold)
        )
        
        return {
            "time_series": activity["time_series"],
            "anomalies": anomalies.tolist(),
            "scores": {
                "isolation": isolation_scores.tolist(),
                "envelope": envelope_scores.tolist(),
                "zscore": zscore_scores.tolist()
            }
        }
    
    def _detect_preference_anomalies(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Detect anomalies in preference patterns."""
        # Get preference data
        preferences = self.analyzer._analyze_preference_trends(start_time, end_time)
        
        if not preferences["evolution"]:
            return {
                "patterns": {},
                "global_anomalies": []
            }
        
        patterns = {}
        global_anomalies = []
        
        for obj, history in preferences["evolution"].items():
            if len(history["weights"]) > 1:
                # Prepare features
                X = np.array(history["weights"]).reshape(-1, 1)
                X_scaled = self.scaler.fit_transform(X)
                
                # Detect anomalies
                isolation_scores = self.models["isolation"].fit_predict(X_scaled)
                zscore_scores = np.abs(stats.zscore(X_scaled.flatten()))
                
                anomalies = (
                    (isolation_scores == -1) |
                    (zscore_scores > self.config.zscore_threshold)
                )
                
                patterns[obj] = {
                    "timestamps": history["timestamps"],
                    "values": history["weights"],
                    "anomalies": anomalies.tolist(),
                    "scores": {
                        "isolation": isolation_scores.tolist(),
                        "zscore": zscore_scores.tolist()
                    }
                }
                
                # Track global anomalies
                if np.any(anomalies):
                    global_anomalies.extend([
                        {
                            "objective": obj,
                            "timestamp": history["timestamps"][i],
                            "value": history["weights"][i],
                            "severity": float(zscore_scores[i])
                        }
                        for i in np.where(anomalies)[0]
                    ])
        
        return {
            "patterns": patterns,
            "global_anomalies": global_anomalies
        }
    
    def _detect_user_anomalies(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Detect anomalies in user behavior."""
        # Get user data
        users = self.analyzer._analyze_user_trends(start_time, end_time)
        
        if not users["time_series"]["active_users"]:
            return {
                "patterns": None,
                "anomalies": {
                    "users": [],
                    "timestamps": []
                }
            }
        
        # Create user behavior matrix
        user_ids = list(self.analyzer.collab.user_profiles.keys())
        timestamps = users["time_series"]["timestamps"]
        
        behavior_matrix = np.zeros((len(user_ids), len(timestamps)))
        
        for i, user_id in enumerate(user_ids):
            for j, ts in enumerate(timestamps):
                # Count user interactions in window
                count = sum(
                    1
                    for interaction in self.analyzer.collab.user_interactions.get(user_id, [])
                    if (
                        datetime.fromisoformat(interaction["timestamp"]) <= ts and
                        datetime.fromisoformat(interaction["timestamp"]) >
                        ts - timedelta(hours=self.config.window_size)
                    )
                )
                behavior_matrix[i, j] = count
        
        # Detect anomalies in behavior patterns
        X = behavior_matrix.reshape(len(user_ids), -1)
        X_scaled = self.scaler.fit_transform(X)
        
        # Use Isolation Forest for anomaly detection
        user_anomalies = self.models["isolation"].fit_predict(X_scaled)
        
        # Find anomalous timestamps for each user
        anomaly_users = []
        anomaly_times = []
        
        for i, is_anomaly in enumerate(user_anomalies):
            if is_anomaly == -1:
                # Find specific anomalous periods
                user_scores = stats.zscore(behavior_matrix[i])
                anomaly_periods = np.where(np.abs(user_scores) > self.config.zscore_threshold)[0]
                
                for j in anomaly_periods:
                    anomaly_users.append(user_ids[i])
                    anomaly_times.append(timestamps[j])
        
        return {
            "patterns": {
                "behavior_matrix": behavior_matrix.tolist(),
                "user_ids": user_ids,
                "timestamps": timestamps
            },
            "anomalies": {
                "users": anomaly_users,
                "timestamps": anomaly_times
            }
        }
    
    def _detect_collective_anomalies(
        self,
        start_time: Optional[datetime],
        end_time: Optional[datetime]
    ) -> Dict[str, Any]:
        """Detect collective anomalies across all metrics."""
        # Get all metrics
        activity = self.analyzer._analyze_activity_trends(start_time, end_time)
        users = self.analyzer._analyze_user_trends(start_time, end_time)
        preferences = self.analyzer._analyze_preference_trends(start_time, end_time)
        
        if not activity["time_series"]["values"]:
            return {
                "components": [],
                "global_patterns": []
            }
        
        # Create feature matrix
        features = []
        timestamps = activity["time_series"]["timestamps"]
        
        # Activity features
        features.append(activity["time_series"]["values"])
        
        # User features
        if users["time_series"]["active_users"]:
            features.append(users["time_series"]["active_users"])
        
        # Preference features
        for obj, history in preferences["evolution"].items():
            if len(history["weights"]) == len(timestamps):
                features.append(history["weights"])
        
        if not features:
            return {
                "components": [],
                "global_patterns": []
            }
        
        # Convert to array and scale
        X = np.array(features).T
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA
        components = self.pca.fit_transform(X_scaled)
        
        # Detect anomalies in principal components
        component_results = []
        global_patterns = []
        
        for i in range(min(self.config.n_components, components.shape[1])):
            # Calculate anomaly scores
            scores = stats.zscore(components[:, i])
            anomalies = np.abs(scores) > self.config.zscore_threshold
            
            component_results.append({
                "timestamps": timestamps,
                "values": components[:, i].tolist(),
                "anomalies": anomalies.tolist(),
                "explained_variance": float(
                    self.pca.explained_variance_ratio_[i]
                )
            })
            
            # Track global patterns
            if np.any(anomalies):
                global_patterns.extend([
                    {
                        "component": i,
                        "timestamp": timestamps[j],
                        "severity": float(scores[j]),
                        "contribution": self.pca.components_[i].tolist()
                    }
                    for j in np.where(anomalies)[0]
                ])
        
        return {
            "components": component_results,
            "global_patterns": global_patterns
        }
    
    def _analyze_anomaly_context(self) -> Dict[str, Any]:
        """Analyze context of detected anomalies."""
        if not self.anomaly_history:
            return {}
        
        context = {
            "temporal_patterns": self._analyze_temporal_context(),
            "feature_importance": self._analyze_feature_importance(),
            "correlation_patterns": self._analyze_correlation_patterns(),
            "severity_distribution": self._analyze_severity_distribution()
        }
        
        # Update current context
        self.current_context = context
        
        return context
    
    def _analyze_temporal_context(self) -> Dict[str, Any]:
        """Analyze temporal patterns in anomalies."""
        timestamps = []
        severities = []
        types = []
        
        for record in self.anomaly_history:
            results = record["results"]
            
            # Activity anomalies
            if results["activity"]["time_series"]:
                for i, is_anomaly in enumerate(results["activity"]["anomalies"]):
                    if is_anomaly:
                        timestamps.append(
                            results["activity"]["time_series"]["timestamps"][i]
                        )
                        severities.append(
                            abs(results["activity"]["scores"]["zscore"][i])
                        )
                        types.append("activity")
            
            # Preference anomalies
            for anomaly in results["preferences"]["global_anomalies"]:
                timestamps.append(anomaly["timestamp"])
                severities.append(anomaly["severity"])
                types.append("preference")
        
        if not timestamps:
            return {}
        
        # Analyze temporal distribution
        df = pd.DataFrame({
            "timestamp": timestamps,
            "severity": severities,
            "type": types
        })
        
        hourly = df.groupby([
            pd.Grouper(key="timestamp", freq="H"),
            "type"
        ]).size().unstack(fill_value=0)
        
        daily = df.groupby([
            pd.Grouper(key="timestamp", freq="D"),
            "type"
        ]).size().unstack(fill_value=0)
        
        return {
            "hourly_distribution": {
                "timestamps": hourly.index.tolist(),
                "values": hourly.to_dict("list")
            },
            "daily_distribution": {
                "timestamps": daily.index.tolist(),
                "values": daily.to_dict("list")
            },
            "type_distribution": df["type"].value_counts().to_dict()
        }
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """Analyze importance of different features."""
        if not self.anomaly_history:
            return {}
        
        # Collect feature scores
        feature_scores = defaultdict(list)
        
        for record in self.anomaly_history:
            results = record["results"]
            
            # Activity features
            if results["activity"]["time_series"]:
                feature_scores["activity"].extend(
                    results["activity"]["scores"]["zscore"]
                )
            
            # Preference features
            for obj, pattern in results["preferences"].get("patterns", {}).items():
                feature_scores[f"preference_{obj}"].extend(
                    pattern["scores"]["zscore"]
                )
        
        # Calculate importance metrics
        importance = {}
        for feature, scores in feature_scores.items():
            if scores:
                importance[feature] = {
                    "mean_severity": float(np.mean(np.abs(scores))),
                    "max_severity": float(np.max(np.abs(scores))),
                    "anomaly_rate": float(
                        np.mean(np.abs(scores) > self.config.zscore_threshold)
                    )
                }
        
        # Update feature importance
        self.feature_importance = importance
        
        return importance
    
    def _analyze_correlation_patterns(self) -> Dict[str, Any]:
        """Analyze correlations between different types of anomalies."""
        if not self.anomaly_history:
            return {}
        
        # Create correlation matrix
        features = []
        feature_names = []
        
        for record in self.anomaly_history:
            results = record["results"]
            
            # Activity anomalies
            if results["activity"]["time_series"]:
                features.append(results["activity"]["scores"]["zscore"])
                feature_names.append("activity")
            
            # Preference anomalies
            for obj, pattern in results["preferences"].get("patterns", {}).items():
                features.append(pattern["scores"]["zscore"])
                feature_names.append(f"preference_{obj}")
        
        if not features:
            return {}
        
        # Calculate correlations
        correlations = np.corrcoef(features)
        
        return {
            "correlation_matrix": correlations.tolist(),
            "feature_names": feature_names,
            "strong_correlations": [
                {
                    "feature1": feature_names[i],
                    "feature2": feature_names[j],
                    "correlation": float(correlations[i, j])
                }
                for i in range(len(feature_names))
                for j in range(i + 1, len(feature_names))
                if abs(correlations[i, j]) > 0.7
            ]
        }
    
    def _analyze_severity_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of anomaly severities."""
        if not self.anomaly_history:
            return {}
        
        severities = []
        types = []
        
        for record in self.anomaly_history:
            results = record["results"]
            
            # Activity anomalies
            if results["activity"]["time_series"]:
                severities.extend(results["activity"]["scores"]["zscore"])
                types.extend(["activity"] * len(results["activity"]["scores"]["zscore"]))
            
            # Preference anomalies
            for anomaly in results["preferences"]["global_anomalies"]:
                severities.append(anomaly["severity"])
                types.append("preference")
        
        if not severities:
            return {}
        
        # Calculate distribution metrics
        return {
            "overall": {
                "mean": float(np.mean(severities)),
                "std": float(np.std(severities)),
                "skew": float(stats.skew(severities)),
                "kurtosis": float(stats.kurtosis(severities))
            },
            "by_type": {
                t: {
                    "mean": float(np.mean([
                        s for s, typ in zip(severities, types)
                        if typ == t
                    ])),
                    "std": float(np.std([
                        s for s, typ in zip(severities, types)
                        if typ == t
                    ]))
                }
                for t in set(types)
            },
            "percentiles": {
                str(p): float(np.percentile(severities, p))
                for p in [25, 50, 75, 90, 95, 99]
            }
        }
    
    def save_anomalies(
        self,
        anomalies: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save anomaly detection results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save anomaly results
            anomaly_file = path / "anomaly_detection.json"
            with open(anomaly_file, "w") as f:
                json.dump(
                    {
                        k: v for k, v in anomalies.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_anomalies(anomalies)
            viz.write_html(str(path / "anomaly_detection.html"))
            
            # Save context analysis if available
            if self.current_context:
                context_file = path / "anomaly_context.json"
                with open(context_file, "w") as f:
                    json.dump(self.current_context, f, indent=2)
            
            logger.info(f"Saved anomaly detection results to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save anomalies: {e}")

def create_anomaly_detector(
    analyzer: ExplorationTrendAnalyzer,
    output_path: Optional[Path] = None
) -> ExplorationAnomalyDetector:
    """Create anomaly detector."""
    config = AnomalyConfig(output_path=output_path)
    return ExplorationAnomalyDetector(analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .exploration_trends import create_trend_analyzer
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
    analyzer = create_trend_analyzer(collab)
    detector = create_anomaly_detector(
        analyzer,
        output_path=Path("anomaly_detection")
    )
    
    # Analyze trends
    trends = analyzer.analyze_trends()
    
    # Detect anomalies
    anomalies = detector.detect_anomalies()
    
    # Visualize results
    viz = detector.visualize_anomalies(anomalies)
    viz.show()
    
    # Save results
    detector.save_anomalies(anomalies)
