"""Anomaly analysis for performance trends."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .trend_analysis import TrendAnalyzer, TrendConfig
from .adaptation_metrics import PerformanceTracker

@dataclass
class AnomalyConfig:
    """Configuration for anomaly analysis."""
    contamination: float = 0.1
    zscore_threshold: float = 3.0
    window_size: int = 20
    min_deviation: float = 0.2
    enable_pattern_detection: bool = True
    enable_root_cause: bool = True
    confidence_level: float = 0.95
    persistence_threshold: int = 3

@dataclass
class AnomalyPattern:
    """Detected anomaly pattern."""
    start_time: datetime
    end_time: datetime
    pattern_type: str
    severity: float
    metrics: List[str]
    probability: float
    context: Dict[str, Any]

@dataclass
class RootCause:
    """Root cause analysis result."""
    metric: str
    correlation: float
    lag: int
    confidence: float
    evidence: List[str]

class AnomalyDetector:
    """Detect and analyze performance anomalies."""
    
    def __init__(
        self,
        tracker: PerformanceTracker,
        trend_analyzer: TrendAnalyzer,
        config: AnomalyConfig = None
    ):
        self.tracker = tracker
        self.trend_analyzer = trend_analyzer
        self.config = config or AnomalyConfig()
        
        # Detection state
        self.anomalies: Dict[str, List[AnomalyPattern]] = {}
        self.root_causes: Dict[str, List[RootCause]] = {}
        self.isolation_forests: Dict[str, Dict[str, IsolationForest]] = {}
        self.scalers: Dict[str, Dict[str, StandardScaler]] = {}
    
    async def detect_anomalies(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Detect anomalies in performance metrics."""
        if preset_name not in self.tracker.windowed:
            return {"status": "no_data"}
        
        # Initialize storage
        if preset_name not in self.anomalies:
            self.anomalies[preset_name] = []
            self.root_causes[preset_name] = []
        
        # Get current metrics
        metrics = self.tracker.get_current_metrics(preset_name)
        current_time = datetime.now()
        
        # Analyze each metric
        analysis = {}
        for window_size, window_metrics in metrics["windowed"].items():
            analysis[f"window_{window_size}"] = {}
            
            for metric_name, values in window_metrics.items():
                if not isinstance(values, list):
                    continue
                
                # Statistical anomalies
                stat_anomalies = await self._detect_statistical_anomalies(
                    values,
                    metric_name
                )
                
                # Isolation Forest anomalies
                iso_anomalies = await self._detect_isolation_anomalies(
                    preset_name,
                    metric_name,
                    values
                )
                
                # Pattern analysis
                if self.config.enable_pattern_detection:
                    patterns = await self._analyze_patterns(
                        values,
                        stat_anomalies + iso_anomalies
                    )
                else:
                    patterns = []
                
                # Root cause analysis
                if self.config.enable_root_cause:
                    causes = await self._analyze_root_causes(
                        preset_name,
                        metric_name,
                        stat_anomalies + iso_anomalies
                    )
                else:
                    causes = []
                
                # Store results
                result = {
                    "anomalies": {
                        "statistical": stat_anomalies,
                        "isolation": iso_anomalies
                    },
                    "patterns": patterns,
                    "root_causes": causes,
                    "summary": {
                        "total_anomalies": len(stat_anomalies) + len(iso_anomalies),
                        "pattern_types": list(set(p.pattern_type for p in patterns)),
                        "severity": max((a["severity"] for a in stat_anomalies + iso_anomalies), default=0)
                    }
                }
                
                analysis[f"window_{window_size}"][metric_name] = result
                
                # Store significant anomalies
                for anomaly in stat_anomalies + iso_anomalies:
                    if anomaly["severity"] > self.config.min_deviation:
                        pattern = AnomalyPattern(
                            start_time=current_time - timedelta(
                                seconds=len(values) - anomaly["index"]
                            ),
                            end_time=current_time,
                            pattern_type=anomaly["type"],
                            severity=anomaly["severity"],
                            metrics=[metric_name],
                            probability=anomaly["probability"],
                            context={
                                "window_size": window_size,
                                "value": anomaly["value"],
                                "expected": anomaly["expected"]
                            }
                        )
                        self.anomalies[preset_name].append(pattern)
        
        return {
            "status": "success",
            "analysis": analysis
        }
    
    async def _detect_statistical_anomalies(
        self,
        values: List[float],
        metric_name: str
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(values) < 3:
            return []
        
        anomalies = []
        mean = np.mean(values)
        std = np.std(values)
        
        for i, value in enumerate(values):
            # Z-score detection
            zscore = (value - mean) / (std + 1e-10)
            
            if abs(zscore) > self.config.zscore_threshold:
                anomalies.append({
                    "index": i,
                    "value": value,
                    "expected": mean,
                    "severity": abs(zscore) / self.config.zscore_threshold,
                    "probability": 1 - stats.norm.cdf(abs(zscore)),
                    "type": "statistical"
                })
        
        return anomalies
    
    async def _detect_isolation_anomalies(
        self,
        preset_name: str,
        metric_name: str,
        values: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using Isolation Forest."""
        if len(values) < 10:
            return []
        
        # Get or create model
        if preset_name not in self.isolation_forests:
            self.isolation_forests[preset_name] = {}
            self.scalers[preset_name] = {}
        
        if metric_name not in self.isolation_forests[preset_name]:
            self.isolation_forests[preset_name][metric_name] = IsolationForest(
                contamination=self.config.contamination,
                random_state=42
            )
            self.scalers[preset_name][metric_name] = StandardScaler()
        
        # Prepare data
        X = np.array(values).reshape(-1, 1)
        X_scaled = self.scalers[preset_name][metric_name].fit_transform(X)
        
        # Fit and predict
        model = self.isolation_forests[preset_name][metric_name]
        scores = model.fit_predict(X_scaled)
        decision_scores = model.score_samples(X_scaled)
        
        # Find anomalies
        anomalies = []
        for i, (score, decision) in enumerate(zip(scores, decision_scores)):
            if score == -1:  # Anomaly
                severity = abs(decision) / np.mean(np.abs(decision_scores))
                anomalies.append({
                    "index": i,
                    "value": values[i],
                    "expected": np.mean(values),
                    "severity": severity,
                    "probability": 1 - (0.5 + 0.5 * np.tanh(decision)),
                    "type": "isolation"
                })
        
        return anomalies
    
    async def _analyze_patterns(
        self,
        values: List[float],
        anomalies: List[Dict[str, Any]]
    ) -> List[AnomalyPattern]:
        """Analyze anomaly patterns."""
        if not anomalies:
            return []
        
        patterns = []
        anomaly_indices = set(a["index"] for a in anomalies)
        
        # Find consecutive anomalies
        current_pattern = []
        for i in range(len(values)):
            if i in anomaly_indices:
                current_pattern.append(i)
            elif current_pattern:
                if len(current_pattern) >= self.config.persistence_threshold:
                    # Analyze pattern type
                    pattern_values = [values[j] for j in current_pattern]
                    pattern_type = await self._classify_pattern(pattern_values)
                    
                    # Create pattern object
                    pattern = AnomalyPattern(
                        start_time=datetime.now(),  # Placeholder
                        end_time=datetime.now(),
                        pattern_type=pattern_type,
                        severity=np.mean([
                            a["severity"] for a in anomalies
                            if a["index"] in current_pattern
                        ]),
                        metrics=[],  # Filled later
                        probability=np.mean([
                            a["probability"] for a in anomalies
                            if a["index"] in current_pattern
                        ]),
                        context={
                            "length": len(current_pattern),
                            "indices": current_pattern.copy()
                        }
                    )
                    patterns.append(pattern)
                
                current_pattern = []
        
        return patterns
    
    async def _classify_pattern(
        self,
        values: List[float]
    ) -> str:
        """Classify pattern type."""
        if len(values) < 3:
            return "spike"
        
        # Calculate trend
        slope = np.polyfit(range(len(values)), values, 1)[0]
        
        # Calculate oscillation
        peaks = []
        for i in range(1, len(values)-1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
        
        if len(peaks) >= 2:
            return "oscillation"
        elif abs(slope) > 0.1:
            return "trend" if slope > 0 else "drop"
        else:
            return "level_shift"
    
    async def _analyze_root_causes(
        self,
        preset_name: str,
        metric_name: str,
        anomalies: List[Dict[str, Any]]
    ) -> List[RootCause]:
        """Analyze root causes of anomalies."""
        if not anomalies:
            return []
        
        causes = []
        metrics = self.tracker.get_current_metrics(preset_name)
        
        # Find correlated metrics
        for window_size, window_metrics in metrics["windowed"].items():
            for other_metric, other_values in window_metrics.items():
                if other_metric == metric_name or not isinstance(other_values, list):
                    continue
                
                # Calculate cross-correlation
                correlation, lag = await self._calculate_correlation(
                    other_values,
                    [a["value"] for a in anomalies]
                )
                
                if abs(correlation) > 0.7:  # Strong correlation threshold
                    cause = RootCause(
                        metric=other_metric,
                        correlation=correlation,
                        lag=lag,
                        confidence=abs(correlation),
                        evidence=[
                            f"Correlation: {correlation:.2f}",
                            f"Lag: {lag}",
                            f"Window size: {window_size}"
                        ]
                    )
                    causes.append(cause)
        
        return causes
    
    async def _calculate_correlation(
        self,
        x: List[float],
        y: List[float]
    ) -> Tuple[float, int]:
        """Calculate cross-correlation and lag."""
        if len(x) != len(y):
            return 0.0, 0
        
        correlations = []
        max_lag = min(len(x) // 4, 10)  # Limit maximum lag
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
            else:
                corr = np.corrcoef(x, y)[0, 1]
            
            correlations.append((corr, lag))
        
        # Return maximum correlation and its lag
        max_corr, best_lag = max(
            correlations,
            key=lambda x: abs(x[0])
        )
        return max_corr, best_lag
    
    async def create_anomaly_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create anomaly visualization plots."""
        if preset_name not in self.anomalies:
            return {}
        
        plots = {}
        
        # Anomaly timeline plot
        timeline_fig = go.Figure()
        
        for pattern in self.anomalies[preset_name]:
            timeline_fig.add_trace(
                go.Scatter(
                    x=[pattern.start_time, pattern.end_time],
                    y=[pattern.severity, pattern.severity],
                    mode="lines+markers",
                    name=pattern.pattern_type,
                    text=f"Probability: {pattern.probability:.2f}"
                )
            )
        
        timeline_fig.update_layout(
            title="Anomaly Timeline",
            xaxis_title="Time",
            yaxis_title="Severity"
        )
        plots["timeline"] = timeline_fig
        
        # Pattern distribution plot
        if self.anomalies[preset_name]:
            pattern_types = [p.pattern_type for p in self.anomalies[preset_name]]
            pattern_counts = pd.Series(pattern_types).value_counts()
            
            pattern_fig = go.Figure(
                go.Bar(
                    x=pattern_counts.index,
                    y=pattern_counts.values,
                    name="Pattern Distribution"
                )
            )
            
            pattern_fig.update_layout(
                title="Anomaly Pattern Distribution",
                xaxis_title="Pattern Type",
                yaxis_title="Count"
            )
            plots["patterns"] = pattern_fig
        
        # Root cause graph
        if self.root_causes[preset_name]:
            cause_fig = go.Figure()
            
            for cause in self.root_causes[preset_name]:
                cause_fig.add_trace(
                    go.Bar(
                        x=[cause.metric],
                        y=[abs(cause.correlation)],
                        name=cause.metric,
                        text=f"Lag: {cause.lag}"
                    )
                )
            
            cause_fig.update_layout(
                title="Root Cause Analysis",
                xaxis_title="Metric",
                yaxis_title="Correlation Strength"
            )
            plots["causes"] = cause_fig
        
        return plots

def create_anomaly_detector(
    tracker: PerformanceTracker,
    trend_analyzer: TrendAnalyzer,
    config: Optional[AnomalyConfig] = None
) -> AnomalyDetector:
    """Create anomaly detector."""
    return AnomalyDetector(tracker, trend_analyzer, config)

if __name__ == "__main__":
    from .trend_analysis import create_trend_analyzer
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
        detector = create_anomaly_detector(tracker, analyzer)
        
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
        
        # Generate test data with anomalies
        t = np.linspace(0, 4*np.pi, 200)
        base = 0.5 * np.sin(t)
        anomalies = np.zeros_like(t)
        anomalies[50:60] = 2.0  # Spike
        anomalies[100:120] = np.linspace(0, 1, 20)  # Trend
        values = base + anomalies + np.random.normal(0, 0.1, len(t))
        
        # Record metrics
        for i, v in enumerate(values):
            await tracker.update_metrics("test_preset", v, v)
        
        # Detect anomalies
        analysis = await detector.detect_anomalies("test_preset")
        print("Anomaly analysis:", analysis)
        
        # Create plots
        plots = await detector.create_anomaly_plots("test_preset")
        for name, fig in plots.items():
            fig.write_html(f"test_anomaly_{name}.html")
    
    asyncio.run(main())
