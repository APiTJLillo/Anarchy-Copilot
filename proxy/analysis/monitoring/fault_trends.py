"""Fault trend detection and prediction."""

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import joblib
from pathlib import Path
import json

from .test_fault_correlation import FaultEvent, FaultCorrelationAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class TrendInfo:
    """Information about detected trend."""
    metric: str
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    r_squared: float
    prediction_horizon: timedelta
    confidence: float
    related_faults: List[str]
    impact_estimate: float

@dataclass
class AnomalyInfo:
    """Information about detected anomaly."""
    timestamp: datetime
    fault_type: str
    metrics: Dict[str, float]
    severity: float
    deviation: float
    cluster_id: int
    is_novel: bool

class FaultTrendDetector:
    """Detect and analyze fault trends."""
    
    def __init__(
        self,
        analyzer: FaultCorrelationAnalyzer,
        model_dir: Optional[Path] = None,
        min_samples: int = 10,
        confidence_threshold: float = 0.7
    ):
        self.analyzer = analyzer
        self.model_dir = model_dir or Path("trend_models")
        self.model_dir.mkdir(exist_ok=True)
        self.min_samples = min_samples
        self.confidence_threshold = confidence_threshold
        
        self.scaler = StandardScaler()
        self.anomaly_detector = DBSCAN(eps=0.3, min_samples=3)
        self.trend_models: Dict[str, LinearRegression] = {}
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load existing models or train new ones."""
        model_file = self.model_dir / "trend_models.joblib"
        scaler_file = self.model_dir / "scaler.joblib"
        
        if model_file.exists() and scaler_file.exists():
            self.trend_models = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
        else:
            self._train_models()
    
    def _train_models(self):
        """Train trend detection models."""
        if not self.analyzer.fault_history:
            return
        
        # Collect metrics
        metrics = set()
        for event in self.analyzer.fault_history:
            metrics.update(event.impact_metrics.keys())
        
        # Prepare training data
        for metric in metrics:
            metric_data = []
            targets = []
            
            for i in range(len(self.analyzer.fault_history) - 1):
                event = self.analyzer.fault_history[i]
                next_event = self.analyzer.fault_history[i + 1]
                
                if metric in event.impact_metrics:
                    metric_data.append([
                        event.impact_metrics[metric],
                        (next_event.timestamp - event.timestamp).total_seconds(),
                        len(event.related_faults),
                        event.cascade_depth
                    ])
                    
                    targets.append(
                        next_event.impact_metrics.get(metric, 0)
                    )
            
            if len(metric_data) >= self.min_samples:
                X = self.scaler.fit_transform(metric_data)
                model = LinearRegression()
                model.fit(X, targets)
                self.trend_models[metric] = model
        
        # Save models
        joblib.dump(self.trend_models, self.model_dir / "trend_models.joblib")
        joblib.dump(self.scaler, self.model_dir / "scaler.joblib")
    
    def detect_trends(self) -> List[TrendInfo]:
        """Detect trends in fault metrics."""
        trends = []
        
        for metric in self.trend_models:
            metric_values = []
            timestamps = []
            
            for event in self.analyzer.fault_history:
                if metric in event.impact_metrics:
                    metric_values.append(event.impact_metrics[metric])
                    timestamps.append(event.timestamp)
            
            if len(metric_values) >= self.min_samples:
                # Calculate trend
                x = np.array([
                    (t - timestamps[0]).total_seconds()
                    for t in timestamps
                ]).reshape(-1, 1)
                y = np.array(metric_values)
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    x.flatten(), y
                )
                
                # Predict future value
                future_time = timestamps[-1] + timedelta(hours=1)
                future_x = np.array([
                    (future_time - timestamps[0]).total_seconds()
                ]).reshape(-1, 1)
                
                model = self.trend_models[metric]
                prediction = model.predict(
                    self.scaler.transform([[
                        metric_values[-1],
                        3600,  # 1 hour in seconds
                        np.mean([len(e.related_faults) for e in self.analyzer.fault_history]),
                        np.mean([e.cascade_depth for e in self.analyzer.fault_history])
                    ]])
                )[0]
                
                # Calculate confidence
                confidence = abs(r_value)
                if confidence >= self.confidence_threshold:
                    trends.append(TrendInfo(
                        metric=metric,
                        direction="increasing" if slope > 0 else "decreasing",
                        slope=slope,
                        r_squared=r_value ** 2,
                        prediction_horizon=timedelta(hours=1),
                        confidence=confidence,
                        related_faults=self._find_related_faults(metric),
                        impact_estimate=prediction
                    ))
        
        return sorted(
            trends,
            key=lambda t: t.confidence,
            reverse=True
        )
    
    def detect_anomalies(self) -> List[AnomalyInfo]:
        """Detect anomalous fault patterns."""
        if len(self.analyzer.fault_history) < self.min_samples:
            return []
        
        # Extract features
        features = []
        timestamps = []
        fault_types = []
        
        for event in self.analyzer.fault_history:
            feature_vector = [
                value for value in event.impact_metrics.values()
            ]
            feature_vector.extend([
                len(event.related_faults),
                event.cascade_depth,
                (event.timestamp - self.analyzer.fault_history[0].timestamp).total_seconds()
            ])
            features.append(feature_vector)
            timestamps.append(event.timestamp)
            fault_types.append(event.fault_type)
        
        # Scale features
        X = self.scaler.fit_transform(features)
        
        # Detect anomalies
        labels = self.anomaly_detector.fit_predict(X)
        anomalies = []
        
        for i, label in enumerate(labels):
            if label == -1:  # Anomaly
                event = self.analyzer.fault_history[i]
                
                # Calculate deviation
                cluster_distances = []
                for j, other_label in enumerate(labels):
                    if other_label != -1:
                        distance = np.linalg.norm(X[i] - X[j])
                        cluster_distances.append(distance)
                
                deviation = np.mean(cluster_distances) if cluster_distances else 0
                
                anomalies.append(AnomalyInfo(
                    timestamp=timestamps[i],
                    fault_type=fault_types[i],
                    metrics=event.impact_metrics,
                    severity=deviation,
                    deviation=deviation,
                    cluster_id=-1,
                    is_novel=self._is_novel_pattern(event)
                ))
        
        return sorted(
            anomalies,
            key=lambda a: a.severity,
            reverse=True
        )
    
    def _find_related_faults(self, metric: str) -> List[str]:
        """Find faults related to metric trend."""
        related = set()
        
        for event in self.analyzer.fault_history:
            if metric in event.impact_metrics:
                related.update(event.related_faults)
        
        return list(related)
    
    def _is_novel_pattern(self, event: FaultEvent) -> bool:
        """Check if fault pattern is novel."""
        # Look for similar patterns in history
        for historical in self.analyzer.fault_history[:-1]:  # Exclude current event
            if (
                historical.fault_type == event.fault_type and
                set(historical.related_faults) == set(event.related_faults) and
                abs(
                    sum(historical.impact_metrics.values()) -
                    sum(event.impact_metrics.values())
                ) < 0.1
            ):
                return False
        return True
    
    def predict_metrics(
        self,
        horizon: timedelta = timedelta(hours=1)
    ) -> Dict[str, List[Tuple[datetime, float]]]:
        """Predict future metric values."""
        predictions = {}
        
        for metric in self.trend_models:
            if len(self.analyzer.fault_history) < self.min_samples:
                continue
            
            model = self.trend_models[metric]
            last_event = self.analyzer.fault_history[-1]
            
            # Generate future timestamps
            future_times = []
            current = last_event.timestamp
            while current <= last_event.timestamp + horizon:
                future_times.append(current)
                current += timedelta(minutes=5)  # 5-minute intervals
            
            # Predict values
            predictions[metric] = []
            for future_time in future_times:
                X = self.scaler.transform([[
                    last_event.impact_metrics.get(metric, 0),
                    (future_time - last_event.timestamp).total_seconds(),
                    len(last_event.related_faults),
                    last_event.cascade_depth
                ]])
                
                predicted = model.predict(X)[0]
                predictions[metric].append((future_time, predicted))
        
        return predictions
    
    def save_analysis(self):
        """Save trend analysis results."""
        trends = self.detect_trends()
        anomalies = self.detect_anomalies()
        predictions = self.predict_metrics()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "trends": [
                {
                    "metric": t.metric,
                    "direction": t.direction,
                    "slope": t.slope,
                    "confidence": t.confidence,
                    "impact_estimate": t.impact_estimate,
                    "related_faults": t.related_faults
                }
                for t in trends
            ],
            "anomalies": [
                {
                    "timestamp": a.timestamp.isoformat(),
                    "fault_type": a.fault_type,
                    "severity": a.severity,
                    "is_novel": a.is_novel,
                    "metrics": a.metrics
                }
                for a in anomalies
            ],
            "predictions": {
                metric: [
                    {
                        "timestamp": t.isoformat(),
                        "value": v
                    }
                    for t, v in values
                ]
                for metric, values in predictions.items()
            }
        }
        
        with open(self.model_dir / "trend_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def analyze_fault_trends(analyzer: FaultCorrelationAnalyzer):
    """Run fault trend analysis."""
    detector = FaultTrendDetector(analyzer)
    return detector.save_analysis()

if __name__ == "__main__":
    # Example usage
    analyzer = FaultCorrelationAnalyzer()
    analysis = analyze_fault_trends(analyzer)
    print(json.dumps(analysis, indent=2))
