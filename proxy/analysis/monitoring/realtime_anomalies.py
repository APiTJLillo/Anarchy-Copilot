"""Real-time anomaly detection for streaming data."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from collections import deque
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .anomaly_detection import ExplorationAnomalyDetector, AnomalyConfig
from .exploration_trends import ExplorationTrendAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class RealtimeConfig:
    """Configuration for real-time anomaly detection."""
    buffer_size: int = 1000
    update_interval: float = 1.0  # seconds
    alert_threshold: float = 0.95
    n_workers: int = 4
    visualization_interval: float = 5.0  # seconds
    output_path: Optional[Path] = None

class RealtimeAnomalyDetector:
    """Real-time anomaly detection for streaming data."""
    
    def __init__(
        self,
        detector: ExplorationAnomalyDetector,
        config: RealtimeConfig
    ):
        self.detector = detector
        self.config = config
        
        # Initialize buffers
        self.activity_buffer = deque(maxlen=config.buffer_size)
        self.preference_buffer = deque(maxlen=config.buffer_size)
        self.user_buffer = deque(maxlen=config.buffer_size)
        self.timestamp_buffer = deque(maxlen=config.buffer_size)
        
        # State management
        self.running = False
        self.alert_callbacks = []
        self.visualization_callbacks = []
        self.current_alerts = []
        self.worker_pool = ThreadPoolExecutor(max_workers=config.n_workers)
        
        # Statistics
        self.stats = {
            "processed_events": 0,
            "detected_anomalies": 0,
            "false_positives": 0,
            "processing_time": []
        }
    
    async def start_monitoring(self):
        """Start real-time monitoring."""
        self.running = True
        
        # Start monitoring tasks
        monitoring_task = asyncio.create_task(self._monitor_stream())
        visualization_task = asyncio.create_task(self._update_visualization())
        
        try:
            await asyncio.gather(monitoring_task, visualization_task)
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            self.running = False
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.running = False
    
    def add_alert_callback(
        self,
        callback: Callable[[Dict[str, Any]], None]
    ):
        """Add callback for anomaly alerts."""
        self.alert_callbacks.append(callback)
    
    def add_visualization_callback(
        self,
        callback: Callable[[go.Figure], None]
    ):
        """Add callback for visualization updates."""
        self.visualization_callbacks.append(callback)
    
    def process_event(
        self,
        event: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process incoming event for anomalies."""
        start_time = datetime.now()
        
        # Extract event data
        timestamp = event.get("timestamp", datetime.now())
        activity = event.get("activity", 0)
        preferences = event.get("preferences", {})
        user_id = event.get("user_id")
        
        # Update buffers
        self.timestamp_buffer.append(timestamp)
        self.activity_buffer.append(activity)
        self.preference_buffer.append(preferences)
        self.user_buffer.append(user_id)
        
        # Detect anomalies
        anomalies = self._detect_realtime_anomalies(
            list(self.timestamp_buffer),
            list(self.activity_buffer),
            list(self.preference_buffer),
            list(self.user_buffer)
        )
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats["processed_events"] += 1
        self.stats["processing_time"].append(processing_time)
        
        if anomalies["alerts"]:
            self.stats["detected_anomalies"] += 1
            
            # Trigger callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(anomalies)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")
        
        return anomalies
    
    async def _monitor_stream(self):
        """Monitor data stream for anomalies."""
        while self.running:
            try:
                # Process any new data in worker pool
                if len(self.timestamp_buffer) > 0:
                    last_timestamp = self.timestamp_buffer[-1]
                    
                    # Check for recent anomalies
                    anomalies = await asyncio.get_event_loop().run_in_executor(
                        self.worker_pool,
                        self._check_recent_anomalies,
                        last_timestamp
                    )
                    
                    if anomalies["alerts"]:
                        # Update current alerts
                        self.current_alerts.extend(anomalies["alerts"])
                        
                        # Trim old alerts
                        self.current_alerts = [
                            alert for alert in self.current_alerts
                            if (datetime.now() - alert["timestamp"]).total_seconds() < 3600
                        ]
                
                # Sleep for update interval
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_visualization(self):
        """Update real-time visualization."""
        while self.running:
            try:
                if len(self.timestamp_buffer) > 0:
                    # Create visualization
                    fig = self._create_realtime_visualization()
                    
                    # Trigger callbacks
                    for callback in self.visualization_callbacks:
                        try:
                            callback(fig)
                        except Exception as e:
                            logger.error(f"Visualization callback error: {e}")
                
                # Sleep for visualization interval
                await asyncio.sleep(self.config.visualization_interval)
                
            except Exception as e:
                logger.error(f"Visualization error: {e}")
                await asyncio.sleep(1.0)
    
    def _detect_realtime_anomalies(
        self,
        timestamps: List[datetime],
        activities: List[float],
        preferences: List[Dict[str, float]],
        user_ids: List[str]
    ) -> Dict[str, Any]:
        """Detect anomalies in real-time data."""
        results = {
            "alerts": [],
            "scores": {
                "activity": None,
                "preferences": {},
                "user": None
            }
        }
        
        if len(timestamps) < 2:
            return results
        
        # Activity anomalies
        activity_scores = self._detect_activity_anomalies(activities)
        results["scores"]["activity"] = activity_scores
        
        if activity_scores and np.max(activity_scores) > self.config.alert_threshold:
            results["alerts"].append({
                "type": "activity",
                "timestamp": timestamps[-1],
                "severity": float(np.max(activity_scores)),
                "description": "Unusual activity level detected"
            })
        
        # Preference anomalies
        for obj in self.detector.analyzer.config.objective_weights:
            pref_values = [p.get(obj, 0) for p in preferences]
            pref_scores = self._detect_preference_anomalies(pref_values)
            results["scores"]["preferences"][obj] = pref_scores
            
            if pref_scores and np.max(pref_scores) > self.config.alert_threshold:
                results["alerts"].append({
                    "type": "preference",
                    "objective": obj,
                    "timestamp": timestamps[-1],
                    "severity": float(np.max(pref_scores)),
                    "description": f"Unusual preference for {obj} detected"
                })
        
        # User anomalies
        if len(set(user_ids)) > 1:
            user_scores = self._detect_user_anomalies(user_ids, activities)
            results["scores"]["user"] = user_scores
            
            if user_scores:
                anomalous_users = [
                    user_ids[i] for i, score in enumerate(user_scores)
                    if score > self.config.alert_threshold
                ]
                
                for user_id in anomalous_users:
                    results["alerts"].append({
                        "type": "user",
                        "user_id": user_id,
                        "timestamp": timestamps[-1],
                        "severity": float(user_scores[user_ids.index(user_id)]),
                        "description": f"Unusual behavior detected for user {user_id}"
                    })
        
        return results
    
    def _detect_activity_anomalies(
        self,
        activities: List[float]
    ) -> Optional[np.ndarray]:
        """Detect anomalies in activity levels."""
        if len(activities) < 2:
            return None
        
        try:
            # Calculate rolling statistics
            window = min(24, len(activities))
            rolling_mean = pd.Series(activities).rolling(window).mean()
            rolling_std = pd.Series(activities).rolling(window).std()
            
            # Calculate z-scores
            z_scores = np.abs(
                (activities - rolling_mean) / rolling_std
            ).fillna(0)
            
            # Convert to anomaly scores
            scores = stats.norm.cdf(z_scores)
            
            return scores.values
            
        except Exception as e:
            logger.error(f"Activity anomaly detection error: {e}")
            return None
    
    def _detect_preference_anomalies(
        self,
        preferences: List[float]
    ) -> Optional[np.ndarray]:
        """Detect anomalies in preferences."""
        if len(preferences) < 2:
            return None
        
        try:
            # Use Isolation Forest for anomaly detection
            X = np.array(preferences).reshape(-1, 1)
            scores = self.detector.models["isolation"].fit_predict(X)
            
            # Convert to probability scores
            scores = np.where(scores == -1, 1, 0)
            
            # Add temporal weighting
            temporal_weights = np.linspace(0.5, 1.0, len(scores))
            scores = scores * temporal_weights
            
            return scores
            
        except Exception as e:
            logger.error(f"Preference anomaly detection error: {e}")
            return None
    
    def _detect_user_anomalies(
        self,
        user_ids: List[str],
        activities: List[float]
    ) -> Optional[np.ndarray]:
        """Detect anomalies in user behavior."""
        if len(user_ids) < 2:
            return None
        
        try:
            # Create user activity profiles
            user_profiles = {}
            for user_id, activity in zip(user_ids, activities):
                if user_id not in user_profiles:
                    user_profiles[user_id] = []
                user_profiles[user_id].append(activity)
            
            # Calculate user statistics
            user_stats = {}
            for user_id, profile in user_profiles.items():
                user_stats[user_id] = {
                    "mean": np.mean(profile),
                    "std": np.std(profile) or 1.0
                }
            
            # Calculate anomaly scores
            scores = np.zeros(len(user_ids))
            for i, user_id in enumerate(user_ids):
                stats = user_stats[user_id]
                z_score = abs(activities[i] - stats["mean"]) / stats["std"]
                scores[i] = stats.norm.cdf(z_score)
            
            return scores
            
        except Exception as e:
            logger.error(f"User anomaly detection error: {e}")
            return None
    
    def _check_recent_anomalies(
        self,
        last_timestamp: datetime
    ) -> Dict[str, Any]:
        """Check for anomalies in recent data."""
        # Get data from last hour
        start_time = last_timestamp - timedelta(hours=1)
        
        # Get indices for recent data
        recent_indices = [
            i for i, ts in enumerate(self.timestamp_buffer)
            if ts >= start_time
        ]
        
        if not recent_indices:
            return {"alerts": []}
        
        # Extract recent data
        recent_timestamps = [self.timestamp_buffer[i] for i in recent_indices]
        recent_activities = [self.activity_buffer[i] for i in recent_indices]
        recent_preferences = [self.preference_buffer[i] for i in recent_indices]
        recent_users = [self.user_buffer[i] for i in recent_indices]
        
        # Detect anomalies
        return self._detect_realtime_anomalies(
            recent_timestamps,
            recent_activities,
            recent_preferences,
            recent_users
        )
    
    def _create_realtime_visualization(self) -> go.Figure:
        """Create real-time visualization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Activity Anomalies",
                "Preference Anomalies",
                "User Anomalies",
                "Alert History"
            ]
        )
        
        # Activity anomalies
        if len(self.activity_buffer) > 0:
            fig.add_trace(
                go.Scatter(
                    x=list(self.timestamp_buffer),
                    y=list(self.activity_buffer),
                    mode="lines",
                    name="Activity"
                ),
                row=1,
                col=1
            )
            
            # Add alert markers
            activity_alerts = [
                alert for alert in self.current_alerts
                if alert["type"] == "activity"
            ]
            
            if activity_alerts:
                fig.add_trace(
                    go.Scatter(
                        x=[alert["timestamp"] for alert in activity_alerts],
                        y=[
                            self.activity_buffer[
                                list(self.timestamp_buffer).index(alert["timestamp"])
                            ]
                            for alert in activity_alerts
                        ],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            symbol="x"
                        ),
                        name="Activity Alerts"
                    ),
                    row=1,
                    col=1
                )
        
        # Preference anomalies
        if len(self.preference_buffer) > 0:
            for obj in self.detector.analyzer.config.objective_weights:
                values = [p.get(obj, 0) for p in self.preference_buffer]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(self.timestamp_buffer),
                        y=values,
                        mode="lines",
                        name=f"{obj} Preference"
                    ),
                    row=1,
                    col=2
                )
            
            # Add alert markers
            pref_alerts = [
                alert for alert in self.current_alerts
                if alert["type"] == "preference"
            ]
            
            if pref_alerts:
                for obj in self.detector.analyzer.config.objective_weights:
                    obj_alerts = [
                        alert for alert in pref_alerts
                        if alert["objective"] == obj
                    ]
                    
                    if obj_alerts:
                        fig.add_trace(
                            go.Scatter(
                                x=[alert["timestamp"] for alert in obj_alerts],
                                y=[
                                    self.preference_buffer[
                                        list(self.timestamp_buffer).index(alert["timestamp"])
                                    ].get(obj, 0)
                                    for alert in obj_alerts
                                ],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color="red",
                                    symbol="x"
                                ),
                                name=f"{obj} Alerts"
                            ),
                            row=1,
                            col=2
                        )
        
        # User anomalies
        if len(self.user_buffer) > 0:
            unique_users = list(set(self.user_buffer))
            user_indices = {user: i for i, user in enumerate(unique_users)}
            
            fig.add_trace(
                go.Scatter(
                    x=list(self.timestamp_buffer),
                    y=[user_indices[user] for user in self.user_buffer],
                    mode="markers",
                    marker=dict(size=5),
                    name="User Activity"
                ),
                row=2,
                col=1
            )
            
            # Add alert markers
            user_alerts = [
                alert for alert in self.current_alerts
                if alert["type"] == "user"
            ]
            
            if user_alerts:
                fig.add_trace(
                    go.Scatter(
                        x=[alert["timestamp"] for alert in user_alerts],
                        y=[user_indices[alert["user_id"]] for alert in user_alerts],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color="red",
                            symbol="x"
                        ),
                        name="User Alerts"
                    ),
                    row=2,
                    col=1
                )
        
        # Alert history
        if self.current_alerts:
            alert_times = [alert["timestamp"] for alert in self.current_alerts]
            alert_types = [alert["type"] for alert in self.current_alerts]
            alert_severities = [alert["severity"] for alert in self.current_alerts]
            
            fig.add_trace(
                go.Scatter(
                    x=alert_times,
                    y=alert_severities,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=[
                            {"activity": "blue", "preference": "green", "user": "red"}[t]
                            for t in alert_types
                        ]
                    ),
                    text=[alert["description"] for alert in self.current_alerts],
                    name="Alerts"
                ),
                row=2,
                col=2
            )
        
        return fig
    
    def save_state(
        self,
        output_path: Optional[Path] = None
    ):
        """Save detector state."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "statistics": self.stats,
                "current_alerts": self.current_alerts,
                "last_update": datetime.now().isoformat()
            }
            
            # Save state
            state_file = path / "realtime_state.json"
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)
            
            # Save current visualization
            if len(self.timestamp_buffer) > 0:
                viz = self._create_realtime_visualization()
                viz.write_html(str(path / "realtime_visualization.html"))
            
            logger.info(f"Saved detector state to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

def create_realtime_detector(
    detector: ExplorationAnomalyDetector,
    output_path: Optional[Path] = None
) -> RealtimeAnomalyDetector:
    """Create real-time detector."""
    config = RealtimeConfig(output_path=output_path)
    return RealtimeAnomalyDetector(detector, config)

if __name__ == "__main__":
    # Example usage
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    from .collaborative_recommendations import create_collaborative_recommender
    from .solution_recommendations import create_solution_recommender
    from .interactive_optimization import create_interactive_explorer
    from .multi_objective_optimization import create_multi_objective_optimizer
    from .simulation_optimization import create_simulation_optimizer
    from .monte_carlo_power import create_monte_carlo_analyzer
    from .power_analysis import create_chain_power_analyzer
    from .statistical_comparison import create_chain_statistician
    
    # Create components
    statistician = create_chain_statistician()
    power_analyzer = create_chain_power_analyzer(statistician)
    mc_analyzer = create_monte_carlo_analyzer(power_analyzer)
    sim_optimizer = create_simulation_optimizer(mc_analyzer)
    mo_optimizer = create_multi_objective_optimizer(sim_optimizer)
    explorer = create_interactive_explorer(mo_optimizer)
    recommender = create_solution_recommender(explorer)
    collab = create_collaborative_recommender(recommender)
    analyzer = create_trend_analyzer(collab)
    detector = create_anomaly_detector(analyzer)
    realtime = create_realtime_detector(
        detector,
        output_path=Path("realtime_monitoring")
    )
    
    async def main():
        # Example alert callback
        def on_alert(alert):
            print(f"Alert: {alert['description']}")
        
        # Example visualization callback
        def on_visualization(fig):
            fig.show()
        
        # Add callbacks
        realtime.add_alert_callback(on_alert)
        realtime.add_visualization_callback(on_visualization)
        
        # Start monitoring
        await realtime.start_monitoring()
        
        # Simulate some events
        for i in range(100):
            event = {
                "timestamp": datetime.now(),
                "activity": np.random.normal(10, 2),
                "preferences": {
                    "objective1": np.random.uniform(0, 1),
                    "objective2": np.random.uniform(0, 1)
                },
                "user_id": f"user_{np.random.randint(1, 5)}"
            }
            
            # Inject anomaly
            if i == 50:
                event["activity"] = 100
            
            realtime.process_event(event)
            await asyncio.sleep(0.1)
        
        # Stop monitoring
        realtime.stop_monitoring()
        
        # Save state
        realtime.save_state()
    
    # Run example
    asyncio.run(main())
