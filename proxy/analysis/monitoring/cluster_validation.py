"""Validation metrics for alert clustering."""

import numpy as np
from typing import Dict, List, Any, Optional, Set
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import json
import asyncio

from .alert_clustering import AlertClusterer, AlertCluster
from .alert_management import Alert

logger = logging.getLogger(__name__)

class ClusterValidator:
    """Validate alert clustering quality."""
    
    def __init__(
        self,
        clusterer: AlertClusterer
    ):
        self.clusterer = clusterer
        self.validation_history = []
        
        # Register validation callback
        self.clusterer._update_clustering = self._wrap_clustering(
            self.clusterer._update_clustering
        )
    
    async def _wrap_clustering(
        self,
        original_fn
    ):
        """Wrap clustering update with validation."""
        async def wrapper(*args, **kwargs):
            await original_fn(*args, **kwargs)
            await self.validate_clusters()
        return wrapper
    
    async def validate_clusters(self):
        """Validate current cluster assignments."""
        try:
            # Get feature matrix
            features = []
            labels = []
            alerts = []
            
            for alert_id, cluster_id in self.clusterer.alert_to_cluster.items():
                if alert_id in self.clusterer.manager.active_alerts:
                    alert = self.clusterer.manager.active_alerts[alert_id]
                    features.append(self.clusterer._extract_features(alert))
                    labels.append(cluster_id)
                    alerts.append(alert)
            
            if not features:
                return
            
            X = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Calculate validation metrics
            metrics = {
                "timestamp": datetime.now(),
                "n_clusters": len(set(labels)),
                "n_alerts": len(alerts),
                "metrics": {
                    "silhouette": float(silhouette_score(X_scaled, labels)),
                    "calinski_harabasz": float(calinski_harabasz_score(X_scaled, labels)),
                    "davies_bouldin": float(davies_bouldin_score(X_scaled, labels))
                },
                "cluster_metrics": self._calculate_cluster_metrics(alerts, labels)
            }
            
            self.validation_history.append(metrics)
            
            # Log validation results
            logger.info(
                f"Cluster validation: "
                f"silhouette={metrics['metrics']['silhouette']:.3f}, "
                f"CH={metrics['metrics']['calinski_harabasz']:.3f}, "
                f"DB={metrics['metrics']['davies_bouldin']:.3f}"
            )
            
            # Check for potential issues
            self._check_validation_issues(metrics)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
    
    def _calculate_cluster_metrics(
        self,
        alerts: List[Alert],
        labels: List[str]
    ) -> Dict[str, Any]:
        """Calculate per-cluster validation metrics."""
        cluster_metrics = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            cluster_alerts = [
                alert for alert, label in zip(alerts, labels)
                if label == cluster_id
            ]
            
            if len(cluster_alerts) < 2:
                continue
            
            # Calculate cluster cohesion
            features = np.array([
                self.clusterer._extract_features(alert)
                for alert in cluster_alerts
            ])
            
            centroid = np.mean(features, axis=0)
            distances = np.linalg.norm(features - centroid, axis=1)
            
            cluster_metrics[cluster_id] = {
                "size": len(cluster_alerts),
                "cohesion": float(np.mean(distances)),
                "variance": float(np.var(distances)),
                "max_distance": float(np.max(distances)),
                "temporal_spread": self._calculate_temporal_spread(cluster_alerts),
                "severity_entropy": self._calculate_severity_entropy(cluster_alerts)
            }
        
        return cluster_metrics
    
    def _calculate_temporal_spread(
        self,
        alerts: List[Alert]
    ) -> float:
        """Calculate temporal spread of alerts."""
        timestamps = [alert.timestamp.timestamp() for alert in alerts]
        return float(np.std(timestamps))
    
    def _calculate_severity_entropy(
        self,
        alerts: List[Alert]
    ) -> float:
        """Calculate entropy of severity distribution."""
        severities = [alert.severity.value for alert in alerts]
        unique_severities = set(severities)
        counts = {s: severities.count(s) for s in unique_severities}
        probs = [count / len(severities) for count in counts.values()]
        return float(-np.sum(p * np.log(p) for p in probs))
    
    def _check_validation_issues(
        self,
        metrics: Dict[str, Any]
    ):
        """Check for potential clustering issues."""
        # Check overall metrics
        if metrics["metrics"]["silhouette"] < 0.2:
            logger.warning(
                "Low silhouette score indicates poor cluster separation"
            )
        
        if metrics["metrics"]["davies_bouldin"] > 2.0:
            logger.warning(
                "High Davies-Bouldin index indicates cluster overlap"
            )
        
        # Check individual clusters
        for cluster_id, cluster_metrics in metrics["cluster_metrics"].items():
            if cluster_metrics["cohesion"] > 0.8:
                logger.warning(
                    f"Cluster {cluster_id} shows low cohesion"
                )
            
            if cluster_metrics["variance"] > 1.5:
                logger.warning(
                    f"Cluster {cluster_id} shows high variance"
                )
    
    def get_validation_summary(
        self,
        window: int = 10
    ) -> Dict[str, Any]:
        """Get summary of recent validation metrics."""
        if not self.validation_history:
            return {}
        
        recent = self.validation_history[-window:]
        
        return {
            "current": recent[-1],
            "trends": {
                "silhouette": [m["metrics"]["silhouette"] for m in recent],
                "calinski_harabasz": [m["metrics"]["calinski_harabasz"] for m in recent],
                "davies_bouldin": [m["metrics"]["davies_bouldin"] for m in recent]
            },
            "cluster_evolution": {
                "sizes": [m["n_clusters"] for m in recent],
                "alerts": [m["n_alerts"] for m in recent]
            },
            "cluster_stability": self._calculate_stability(recent)
        }
    
    def _calculate_stability(
        self,
        metrics: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate cluster stability metrics."""
        if len(metrics) < 2:
            return {}
        
        # Calculate cluster churn
        cluster_sets = [set(m["cluster_metrics"].keys()) for m in metrics]
        churn_rates = []
        
        for i in range(len(cluster_sets) - 1):
            prev = cluster_sets[i]
            curr = cluster_sets[i + 1]
            
            if not prev:
                continue
            
            changes = len(prev.symmetric_difference(curr))
            churn_rates.append(changes / len(prev))
        
        # Calculate metric stability
        stability = {
            "cluster_churn": float(np.mean(churn_rates)) if churn_rates else 0.0,
            "metric_variance": {
                name: float(np.var([m["metrics"][name] for m in metrics]))
                for name in ["silhouette", "calinski_harabasz", "davies_bouldin"]
            }
        }
        
        return stability

def create_cluster_validator(
    clusterer: AlertClusterer
) -> ClusterValidator:
    """Create cluster validator."""
    return ClusterValidator(clusterer)

if __name__ == "__main__":
    # Example usage
    from .alert_clustering import create_alert_clusterer
    from .alert_management import create_alert_manager
    from .realtime_anomalies import create_realtime_detector
    from .anomaly_detection import create_anomaly_detector
    from .exploration_trends import create_trend_analyzer
    
    # Create components
    analyzer = create_trend_analyzer()
    detector = create_anomaly_detector(analyzer)
    realtime = create_realtime_detector(detector)
    manager = create_alert_manager(realtime)
    clusterer = create_alert_clusterer(manager)
    validator = create_cluster_validator(clusterer)
    
    async def main():
        # Start clustering
        await clusterer.start_clustering()
        
        # Check validation metrics
        summary = validator.get_validation_summary()
        print("Validation Summary:", json.dumps(summary, indent=2))
    
    # Run example
    asyncio.run(main())
