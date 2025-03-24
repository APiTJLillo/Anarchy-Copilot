"""Alert clustering for efficient alert management."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import asyncio

from .alert_management import AlertManager, Alert, AlertConfig, AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

@dataclass
class ClusterConfig:
    """Configuration for alert clustering."""
    similarity_threshold: float = 0.7
    time_window: int = 3600  # seconds
    min_cluster_size: int = 2
    feature_weights: Dict[str, float] = None
    update_interval: float = 300  # seconds
    output_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.feature_weights is None:
            self.feature_weights = {
                "type": 1.0,
                "severity": 0.8,
                "description": 0.6,
                "context": 0.4
            }

@dataclass
class AlertCluster:
    """Group of related alerts."""
    id: str
    alerts: Set[str]
    root_cause: Optional[Dict[str, Any]]
    pattern: Optional[Dict[str, Any]]
    first_occurrence: datetime
    last_update: datetime
    severity: AlertSeverity
    status: str
    metadata: Dict[str, Any]

class AlertClusterer:
    """Cluster related alerts to reduce alert fatigue."""
    
    def __init__(
        self,
        manager: AlertManager,
        config: ClusterConfig
    ):
        self.manager = manager
        self.config = config
        
        # Clustering state
        self.clusters = {}
        self.alert_to_cluster = {}
        self.similarity_matrix = None
        self.feature_vectors = None
        
        # Analysis state
        self.cluster_metrics = defaultdict(dict)
        self.pattern_cache = {}
        
        # Feature encoding state
        self._word_vocab = set()
        self._category_map = {}
        
        # Register with alert manager
        self.manager.add_callback("new", self.process_new_alert)
        self.manager.add_callback("update", self.update_clusters)
    
    def _extract_features(
        self,
        alert: Alert
    ) -> np.ndarray:
        """Extract feature vector from alert."""
        features = []
        
        # Type encoding
        type_features = self._encode_categorical(alert.type)
        features.extend(type_features * self.config.feature_weights["type"])
        
        # Severity encoding
        severity_features = np.zeros(len(AlertSeverity))
        severity_features[alert.severity.value - 1] = 1
        features.extend(severity_features * self.config.feature_weights["severity"])
        
        # Description encoding
        if alert.description:
            # Simple bag of words encoding
            words = set(alert.description.lower().split())
            self._word_vocab.update(words)
            
            word_features = np.zeros(len(self._word_vocab))
            for i, word in enumerate(self._word_vocab):
                if word in words:
                    word_features[i] = 1
            
            features.extend(word_features * self.config.feature_weights["description"])
        
        # Context encoding
        if alert.context:
            context_features = self._encode_context(alert.context)
            features.extend(context_features * self.config.feature_weights["context"])
        
        return np.array(features)
    
    def _encode_categorical(
        self,
        value: str,
        max_categories: int = 100
    ) -> np.ndarray:
        """Encode categorical value."""
        if value not in self._category_map and len(self._category_map) < max_categories:
            self._category_map[value] = len(self._category_map)
        
        if value in self._category_map:
            encoding = np.zeros(len(self._category_map))
            encoding[self._category_map[value]] = 1
            return encoding
        
        # Return zero vector for unknown categories
        return np.zeros(len(self._category_map))
    
    def _encode_context(
        self,
        context: Dict[str, Any]
    ) -> np.ndarray:
        """Encode context dictionary."""
        features = []
        
        # Extract numerical values
        numerical_values = []
        for value in context.values():
            if isinstance(value, (int, float)):
                numerical_values.append(value)
            elif isinstance(value, dict):
                numerical_values.extend(
                    v for v in value.values()
                    if isinstance(v, (int, float))
                )
        
        if numerical_values:
            # Basic statistics of numerical values
            features.extend([
                np.mean(numerical_values),
                np.std(numerical_values) if len(numerical_values) > 1 else 0,
                min(numerical_values),
                max(numerical_values)
            ])
        
        # Encode keys
        key_features = self._encode_categorical(
            ",".join(sorted(context.keys()))
        )
        features.extend(key_features)
        
        return np.array(features)
    
    def _find_similar_clusters(
        self,
        features: np.ndarray
    ) -> Dict[str, float]:
        """Find clusters with similar features."""
        similar_clusters = {}
        
        for cluster_id, cluster in self.clusters.items():
            if cluster.status == "resolved":
                continue
            
            # Get cluster features
            cluster_features = []
            for alert_id in cluster.alerts:
                if alert_id in self.manager.active_alerts:
                    alert = self.manager.active_alerts[alert_id]
                    cluster_features.append(self._extract_features(alert))
            
            if not cluster_features:
                continue
            
            # Calculate average cluster features
            avg_features = np.mean(cluster_features, axis=0)
            
            # Handle different feature vector lengths
            min_length = min(len(features), len(avg_features))
            similarity = cosine_similarity(
                features[:min_length].reshape(1, -1),
                avg_features[:min_length].reshape(1, -1)
            )[0, 0]
            
            if similarity >= self.config.similarity_threshold:
                similar_clusters[cluster_id] = float(similarity)
        
        return similar_clusters
    
    def _analyze_temporal_pattern(
        self,
        alerts: List[Alert]
    ) -> Dict[str, Any]:
        """Analyze temporal pattern of alerts."""
        timestamps = [alert.timestamp for alert in alerts]
        intervals = np.diff(sorted(timestamps))
        
        pattern = {
            "frequency": None,
            "regularity": 0.0,
            "trend": None,
            "confidence": 0.0
        }
        
        if len(intervals) >= 2:
            # Calculate frequency and regularity
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            pattern["frequency"] = float(mean_interval.total_seconds())
            pattern["regularity"] = float(
                1 - std_interval / mean_interval
                if mean_interval.total_seconds() > 0
                else 0
            )
            
            # Analyze trend
            times = [t.timestamp() for t in timestamps]
            counts = np.ones_like(times)
            
            slope, intercept = np.polyfit(times, counts, 1)
            pattern["trend"] = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing",
                "correlation": float(np.corrcoef(times, counts)[0, 1])
            }
            
            # Calculate confidence
            pattern["confidence"] = (
                0.7 * pattern["regularity"] +
                0.3 * abs(pattern["trend"]["correlation"])
            )
        
        return pattern
    
    def _analyze_severity_pattern(
        self,
        alerts: List[Alert]
    ) -> Dict[str, Any]:
        """Analyze severity pattern of alerts."""
        severities = [alert.severity.value for alert in alerts]
        
        pattern = {
            "trend": None,
            "distribution": None,
            "confidence": 0.0
        }
        
        if len(severities) >= 2:
            # Analyze trend
            times = [alert.timestamp.timestamp() for alert in alerts]
            slope, intercept = np.polyfit(times, severities, 1)
            
            pattern["trend"] = {
                "slope": float(slope),
                "direction": "increasing" if slope > 0 else "decreasing"
            }
            
            # Analyze distribution
            unique_severities = np.unique(severities)
            if len(unique_severities) > 1:
                severity_counts = {
                    int(s): severities.count(s)
                    for s in unique_severities
                }
                
                pattern["distribution"] = {
                    "counts": severity_counts,
                    "entropy": float(stats.entropy(list(severity_counts.values())))
                }
            
            # Calculate confidence
            if pattern["distribution"]:
                pattern["confidence"] = 1 - (
                    pattern["distribution"]["entropy"] /
                    np.log(len(AlertSeverity))
                )
        
        return pattern
    
    def _analyze_context_pattern(
        self,
        alerts: List[Alert]
    ) -> Dict[str, Any]:
        """Analyze context pattern of alerts."""
        contexts = [alert.context for alert in alerts if alert.context]
        
        pattern = {
            "common_factors": {},
            "correlations": [],
            "confidence": 0.0
        }
        
        if not contexts:
            return pattern
        
        # Find common factors
        pattern["common_factors"] = self._find_common_factors(contexts)
        
        # Find correlations
        if len(contexts) >= 3:
            corr_features = []
            for context in contexts:
                features = []
                
                # Extract numerical features
                for key, value in context.items():
                    if isinstance(value, (int, float)):
                        features.append(value)
                    elif isinstance(value, dict):
                        features.extend(
                            v for v in value.values()
                            if isinstance(v, (int, float))
                        )
                
                if features:
                    corr_features.append(features)
            
            if corr_features:
                # Calculate correlations
                corr_matrix = np.corrcoef(np.array(corr_features).T)
                significant_corr = np.where(np.abs(corr_matrix) > 0.7)
                
                pattern["correlations"] = [
                    {
                        "feature1": int(i),
                        "feature2": int(j),
                        "correlation": float(corr_matrix[i, j])
                    }
                    for i, j in zip(*significant_corr)
                    if i < j
                ]
        
        # Calculate confidence
        if pattern["common_factors"]:
            factor_confidence = np.mean([
                v["frequency"] if isinstance(v, dict) and "frequency" in v
                else np.mean([sv["frequency"] for sv in v.values()])
                for v in pattern["common_factors"].values()
            ])
            
            correlation_confidence = (
                len(pattern["correlations"]) /
                (len(contexts) * (len(contexts) - 1) / 2)
                if len(contexts) > 1
                else 0
            )
            
            pattern["confidence"] = (
                0.7 * factor_confidence +
                0.3 * correlation_confidence
            )
        
        return pattern
    
    def _find_common_factors(
        self,
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Find common factors in context dictionaries."""
        common_factors = {}
        
        # Get all keys
        all_keys = set()
        for context in contexts:
            all_keys.update(context.keys())
        
        # Analyze each key
        for key in all_keys:
            values = [
                context[key]
                for context in contexts
                if key in context
            ]
            
            if not values:
                continue
            
            if isinstance(values[0], (str, bool, int, float)):
                # For simple types, find most common value
                common_value = max(set(values), key=values.count)
                frequency = values.count(common_value) / len(values)
                
                if frequency >= 0.7:
                    common_factors[key] = {
                        "value": common_value,
                        "frequency": frequency
                    }
            
            elif isinstance(values[0], dict):
                # For dictionaries, find common key-value pairs
                common_pairs = {}
                all_sub_keys = set()
                for v in values:
                    all_sub_keys.update(v.keys())
                
                for sub_key in all_sub_keys:
                    sub_values = [
                        v[sub_key]
                        for v in values
                        if sub_key in v
                    ]
                    
                    if sub_values:
                        common_sub_value = max(
                            set(sub_values),
                            key=sub_values.count
                        )
                        frequency = sub_values.count(common_sub_value) / len(sub_values)
                        
                        if frequency >= 0.7:
                            common_pairs[sub_key] = {
                                "value": common_sub_value,
                                "frequency": frequency
                            }
                
                if common_pairs:
                    common_factors[key] = common_pairs
        
        return common_factors
    
    def _get_cluster_updates(
        self,
        cluster: AlertCluster
    ) -> List[Dict[str, Any]]:
        """Get updates for all alerts in cluster."""
        updates = []
        for alert_id in cluster.alerts:
            if alert_id in self.manager.active_alerts:
                alert = self.manager.active_alerts[alert_id]
                updates.extend(alert.updates)
        
        return sorted(updates, key=lambda x: x["timestamp"])
    
    async def start_clustering(self):
        """Start alert clustering."""
        while True:
            try:
                await self._update_clustering()
                self._update_patterns()
                self._update_metrics()
                await asyncio.sleep(self.config.update_interval)
            except Exception as e:
                logger.error(f"Clustering error: {e}")
                await asyncio.sleep(60)
    
    def process_new_alert(
        self,
        alert: Alert
    ):
        """Process new alert for clustering."""
        features = self._extract_features(alert)
        similar_clusters = self._find_similar_clusters(features)
        
        if similar_clusters:
            best_cluster = max(
                similar_clusters.items(),
                key=lambda x: x[1]
            )[0]
            self._add_to_cluster(alert.id, best_cluster)
        else:
            self._create_cluster([alert.id])
    
    def update_clusters(
        self,
        alert: Alert
    ):
        """Update clusters based on alert changes."""
        cluster_id = self.alert_to_cluster.get(alert.id)
        if not cluster_id:
            return
        
        cluster = self.clusters[cluster_id]
        cluster.severity = max(
            self.manager.active_alerts[alert_id].severity
            for alert_id in cluster.alerts
        )
        
        all_resolved = all(
            self.manager.active_alerts[alert_id].status == AlertStatus.RESOLVED
            for alert_id in cluster.alerts
        )
        
        if all_resolved:
            cluster.status = "resolved"
        else:
            cluster.status = "active"
        
        cluster.last_update = datetime.now()
        self._analyze_cluster_pattern(cluster)
    
    async def _update_clustering(self):
        """Update cluster assignments."""
        if not self.manager.active_alerts:
            return
        
        features = []
        alert_ids = []
        
        for alert_id, alert in self.manager.active_alerts.items():
            if alert.status != AlertStatus.RESOLVED:
                features.append(self._extract_features(alert))
                alert_ids.append(alert_id)
        
        if not features:
            return
        
        X = np.array(features)
        db = DBSCAN(
            eps=1 - self.config.similarity_threshold,
            min_samples=self.config.min_cluster_size,
            metric="cosine"
        ).fit(X)
        
        new_clusters = defaultdict(list)
        for i, label in enumerate(db.labels_):
            if label >= 0:
                new_clusters[f"cluster_{label}"].append(alert_ids[i])
        
        for cluster_alerts in new_clusters.values():
            existing_clusters = defaultdict(int)
            for alert_id in cluster_alerts:
                if alert_id in self.alert_to_cluster:
                    existing_clusters[self.alert_to_cluster[alert_id]] += 1
            
            if existing_clusters:
                best_cluster = max(
                    existing_clusters.items(),
                    key=lambda x: x[1]
                )[0]
                for alert_id in cluster_alerts:
                    self._add_to_cluster(alert_id, best_cluster)
            else:
                self._create_cluster(cluster_alerts)
    
    def _update_patterns(self):
        """Update cluster patterns."""
        for cluster in self.clusters.values():
            if cluster.status != "resolved":
                self._analyze_cluster_pattern(cluster)
    
    def _update_metrics(self):
        """Update cluster metrics."""
        for cluster_id, cluster in self.clusters.items():
            self.cluster_metrics[cluster_id].update({
                "size": len(cluster.alerts),
                "age": (datetime.now() - cluster.first_occurrence).total_seconds(),
                "update_rate": len([
                    u for u in self._get_cluster_updates(cluster)
                    if (datetime.now() - u["timestamp"]).total_seconds() < 3600
                ]) / 3600
            })
            
            if cluster.pattern:
                self.cluster_metrics[cluster_id].update({
                    "pattern_strength": cluster.pattern.get("strength", 0),
                    "pattern_regularity": cluster.pattern.get("regularity", 0)
                })
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of alert clusters."""
        return {
            "total_clusters": len(self.clusters),
            "active_clusters": len([
                c for c in self.clusters.values()
                if c.status == "active"
            ]),
            "cluster_sizes": {
                cluster_id: len(cluster.alerts)
                for cluster_id, cluster in self.clusters.items()
            },
            "severity_distribution": {
                severity.name: len([
                    c for c in self.clusters.values()
                    if c.severity == severity
                ])
                for severity in AlertSeverity
            },
            "metrics": dict(self.cluster_metrics)
        }
    
    def visualize_clusters(self) -> go.Figure:
        """Create visualization of alert clusters."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Cluster Network",
                "Cluster Evolution",
                "Pattern Analysis",
                "Cluster Metrics"
            ]
        )
        
        # Network visualization
        G = nx.Graph()
        for alert_id in self.manager.active_alerts:
            G.add_node(alert_id)
        
        for cluster in self.clusters.values():
            alert_ids = list(cluster.alerts)
            for i in range(len(alert_ids)):
                for j in range(i + 1, len(alert_ids)):
                    G.add_edge(alert_ids[i], alert_ids[j])
        
        pos = nx.spring_layout(G)
        
        # Plot edges
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="gray"),
                hoverinfo="none",
                showlegend=False
            ),
            row=1,
            col=1
        )
        
        # Plot nodes
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                marker=dict(
                    size=10,
                    color=[
                        self.manager.active_alerts[node].severity.value
                        for node in G.nodes()
                    ],
                    colorscale="Viridis"
                ),
                text=[node for node in G.nodes()],
                hoverinfo="text",
                showlegend=False
            ),
            row=1,
            col=1
        )
        
        # Evolution visualization
        cluster_growth = defaultdict(list)
        cluster_times = defaultdict(list)
        
        for cluster in self.clusters.values():
            times = sorted(
                self.manager.active_alerts[alert_id].timestamp
                for alert_id in cluster.alerts
            )
            
            for i, t in enumerate(times):
                cluster_growth[cluster.id].append(i + 1)
                cluster_times[cluster.id].append(t)
        
        for cluster_id in cluster_growth:
            fig.add_trace(
                go.Scatter(
                    x=cluster_times[cluster_id],
                    y=cluster_growth[cluster_id],
                    mode="lines+markers",
                    name=f"Cluster {cluster_id}"
                ),
                row=1,
                col=2
            )
        
        # Pattern analysis
        pattern_data = []
        pattern_labels = []
        
        for cluster in self.clusters.values():
            if cluster.pattern:
                pattern_data.append([
                    cluster.pattern.get("frequency", 0),
                    cluster.pattern.get("regularity", 0),
                    cluster.pattern.get("severity_trend", 0)
                ])
                pattern_labels.append(cluster.id)
        
        if pattern_data:
            fig.add_trace(
                go.Scatter3d(
                    x=[p[0] for p in pattern_data],
                    y=[p[1] for p in pattern_data],
                    z=[p[2] for p in pattern_data],
                    mode="markers",
                    marker=dict(size=5),
                    text=pattern_labels,
                    name="Patterns"
                ),
                row=2,
                col=1
            )
        
        # Metrics visualization
        if self.cluster_metrics:
            metrics = pd.DataFrame(self.cluster_metrics).T
            
            fig.add_trace(
                go.Heatmap(
                    z=metrics.values,
                    x=metrics.columns,
                    y=metrics.index,
                    colorscale="Viridis",
                    name="Metrics"
                ),
                row=2,
                col=2
            )
        
        return fig
    
    def _create_cluster(
        self,
        alert_ids: List[str]
    ) -> str:
        """Create new alert cluster."""
        cluster_id = f"cluster_{len(self.clusters)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        alerts = [
            self.manager.active_alerts[alert_id]
            for alert_id in alert_ids
        ]
        
        cluster = AlertCluster(
            id=cluster_id,
            alerts=set(alert_ids),
            root_cause=self._analyze_root_cause(alerts),
            pattern=None,
            first_occurrence=min(alert.timestamp for alert in alerts),
            last_update=datetime.now(),
            severity=max(alert.severity for alert in alerts),
            status="active",
            metadata={}
        )
        
        self.clusters[cluster_id] = cluster
        for alert_id in alert_ids:
            self.alert_to_cluster[alert_id] = cluster_id
        
        self._analyze_cluster_pattern(cluster)
        return cluster_id
    
    def _add_to_cluster(
        self,
        alert_id: str,
        cluster_id: str
    ) -> bool:
        """Add alert to existing cluster."""
        if cluster_id not in self.clusters:
            return False
        
        cluster = self.clusters[cluster_id]
        cluster.alerts.add(alert_id)
        self.alert_to_cluster[alert_id] = cluster_id
        
        alert = self.manager.active_alerts[alert_id]
        cluster.last_update = datetime.now()
        cluster.severity = max(cluster.severity, alert.severity)
        
        cluster.root_cause = self._analyze_root_cause([
            self.manager.active_alerts[aid]
            for aid in cluster.alerts
        ])
        self._analyze_cluster_pattern(cluster)
        
        return True
    
    def save_state(
        self,
        output_path: Optional[Path] = None
    ):
        """Save clusterer state."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            cluster_data = {
                "clusters": {
                    id: {
                        "alerts": list(cluster.alerts),
                        "root_cause": cluster.root_cause,
                        "pattern": cluster.pattern,
                        "first_occurrence": cluster.first_occurrence.isoformat(),
                        "last_update": cluster.last_update.isoformat(),
                        "severity": cluster.severity.name,
                        "status": cluster.status,
                        "metadata": cluster.metadata
                    }
                    for id, cluster in self.clusters.items()
                },
                "metrics": dict(self.cluster_metrics),
                "last_update": datetime.now().isoformat()
            }
            
            cluster_file = path / "cluster_state.json"
            with open(cluster_file, "w") as f:
                json.dump(cluster_data, f, indent=2)
            
            viz = self.visualize_clusters()
            viz.write_html(str(path / "cluster_visualization.html"))
            
            if self.pattern_cache:
                pattern_file = path / "pattern_analysis.json"
                with open(pattern_file, "w") as f:
                    json.dump(self.pattern_cache, f, indent=2)
            
            logger.info(f"Saved cluster state to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

def create_alert_clusterer(
    manager: AlertManager,
    output_path: Optional[Path] = None
) -> AlertClusterer:
    """Create alert clusterer."""
    config = ClusterConfig(output_path=output_path)
    return AlertClusterer(manager, config)

if __name__ == "__main__":
    # Example usage
    from .alert_management import create_alert_manager
    from .realtime_anomalies import create_realtime_detector
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    
    # Create components
    analyzer = create_trend_analyzer()
    detector = create_anomaly_detector(analyzer)
    realtime = create_realtime_detector(detector)
    manager = create_alert_manager(realtime)
    clusterer = create_alert_clusterer(
        manager,
        output_path=Path("alert_clusters")
    )
    
    async def main():
        # Start alert clustering
        await clusterer.start_clustering()
        
        # Example alerts
        for i in range(10):
            alert = Alert(
                id=f"alert_{i}",
                type="test",
                severity=AlertSeverity.MEDIUM,
                status=AlertStatus.NEW,
                description=f"Test alert {i}",
                timestamp=datetime.now(),
                source_data={},
                context={
                    "metric_value": np.random.normal(10, 2),
                    "related_metrics": [
                        {"name": "cpu", "value": np.random.normal(50, 10)}
                    ]
                }
            )
            
            # Process alert
            clusterer.process_new_alert(alert)
            await asyncio.sleep(1)
        
        # Get summary and visualize
        summary = clusterer.get_cluster_summary()
        print("Cluster Summary:", json.dumps(summary, indent=2))
        
        viz = clusterer.visualize_clusters()
        viz.show()
        
        # Save state
        clusterer.save_state()
    
    # Run example
    asyncio.run(main())
