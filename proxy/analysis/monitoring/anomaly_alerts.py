"""Alert generation for performance anomalies."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .anomaly_analysis import AnomalyDetector, AnomalyPattern, RootCause

class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

class AlertState(Enum):
    """Alert states."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    IGNORED = "ignored"

@dataclass
class AlertConfig:
    """Configuration for alert generation."""
    severity_thresholds: Dict[AlertSeverity, float] = field(default_factory=lambda: {
        AlertSeverity.CRITICAL: 0.9,
        AlertSeverity.HIGH: 0.7,
        AlertSeverity.MEDIUM: 0.5,
        AlertSeverity.LOW: 0.3,
        AlertSeverity.INFO: 0.1
    })
    notification_cooldown: Dict[AlertSeverity, timedelta] = field(default_factory=lambda: {
        AlertSeverity.CRITICAL: timedelta(minutes=5),
        AlertSeverity.HIGH: timedelta(minutes=15),
        AlertSeverity.MEDIUM: timedelta(hours=1),
        AlertSeverity.LOW: timedelta(hours=4),
        AlertSeverity.INFO: timedelta(days=1)
    })
    max_alerts_per_pattern: int = 3
    deduplication_window: timedelta = timedelta(hours=1)
    enable_alert_correlation: bool = True
    correlation_threshold: float = 0.7
    max_stored_alerts: int = 1000

@dataclass
class AlertGroup:
    """Group of related alerts."""
    id: str
    pattern_type: str
    severity: AlertSeverity
    state: AlertState
    alerts: List[Dict[str, Any]]
    first_seen: datetime
    last_seen: datetime
    count: int = 0
    correlated_groups: Set[str] = field(default_factory=set)
    context: Dict[str, Any] = field(default_factory=dict)

class AlertManager:
    """Manage and generate anomaly alerts."""
    
    def __init__(
        self,
        detector: AnomalyDetector,
        config: AlertConfig = None
    ):
        self.detector = detector
        self.config = config or AlertConfig()
        
        # Alert storage
        self.active_alerts: Dict[str, AlertGroup] = {}
        self.alert_history: List[Dict[str, Any]] = []
        self.last_notification: Dict[AlertSeverity, datetime] = {
            severity: datetime.min
            for severity in AlertSeverity
        }
    
    def _get_severity(
        self,
        pattern: AnomalyPattern
    ) -> AlertSeverity:
        """Determine alert severity from pattern."""
        for severity, threshold in sorted(
            self.config.severity_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if pattern.severity >= threshold:
                return severity
        return AlertSeverity.INFO
    
    def _should_notify(
        self,
        severity: AlertSeverity,
        current_time: datetime
    ) -> bool:
        """Check if notification should be sent."""
        last_time = self.last_notification[severity]
        cooldown = self.config.notification_cooldown[severity]
        return current_time - last_time >= cooldown
    
    async def process_anomalies(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Process anomalies and generate alerts."""
        # Get latest anomalies
        anomalies = self.detector.anomalies.get(preset_name, [])
        if not anomalies:
            return {"status": "no_anomalies"}
        
        current_time = datetime.now()
        new_alerts = []
        
        for pattern in anomalies:
            # Generate alert ID
            alert_id = f"{preset_name}_{pattern.pattern_type}_{current_time.timestamp()}"
            
            # Determine severity
            severity = self._get_severity(pattern)
            
            # Check for existing alert group
            existing_group = None
            for group in self.active_alerts.values():
                if (
                    group.pattern_type == pattern.pattern_type and
                    group.severity == severity and
                    current_time - group.last_seen <= self.config.deduplication_window
                ):
                    existing_group = group
                    break
            
            # Create alert data
            alert_data = {
                "id": alert_id,
                "preset_name": preset_name,
                "pattern_type": pattern.pattern_type,
                "severity": severity.value,
                "timestamp": current_time.isoformat(),
                "metrics": pattern.metrics,
                "severity_score": pattern.severity,
                "probability": pattern.probability,
                "context": pattern.context
            }
            
            if existing_group:
                # Update existing group
                existing_group.alerts.append(alert_data)
                existing_group.last_seen = current_time
                existing_group.count += 1
                
                if len(existing_group.alerts) > self.config.max_alerts_per_pattern:
                    existing_group.alerts.pop(0)
            else:
                # Create new group
                new_group = AlertGroup(
                    id=alert_id,
                    pattern_type=pattern.pattern_type,
                    severity=severity,
                    state=AlertState.NEW,
                    alerts=[alert_data],
                    first_seen=current_time,
                    last_seen=current_time,
                    count=1
                )
                self.active_alerts[alert_id] = new_group
            
            # Add to history
            self.alert_history.append(alert_data)
            if len(self.alert_history) > self.config.max_stored_alerts:
                self.alert_history.pop(0)
            
            new_alerts.append(alert_data)
        
        # Correlate alert groups
        if self.config.enable_alert_correlation:
            await self._correlate_alerts()
        
        # Clean up old alerts
        await self._cleanup_alerts()
        
        return {
            "status": "success",
            "new_alerts": new_alerts,
            "active_groups": len(self.active_alerts),
            "total_alerts": len(self.alert_history)
        }
    
    async def _correlate_alerts(self):
        """Correlate alert groups."""
        for group1 in self.active_alerts.values():
            for group2 in self.active_alerts.values():
                if group1.id >= group2.id:
                    continue
                
                # Calculate temporal correlation
                time_diff = abs((group1.last_seen - group2.last_seen).total_seconds())
                max_time = max(
                    (group1.last_seen - group1.first_seen).total_seconds(),
                    (group2.last_seen - group2.first_seen).total_seconds()
                )
                temporal_corr = 1 - (time_diff / max_time) if max_time > 0 else 0
                
                # Calculate pattern similarity
                pattern_sim = 1.0 if group1.pattern_type == group2.pattern_type else 0.0
                
                # Calculate metric overlap
                metrics1 = set()
                metrics2 = set()
                for alert in group1.alerts:
                    metrics1.update(alert["metrics"])
                for alert in group2.alerts:
                    metrics2.update(alert["metrics"])
                
                metric_overlap = len(metrics1 & metrics2) / len(metrics1 | metrics2) if metrics1 or metrics2 else 0
                
                # Overall correlation score
                correlation = (temporal_corr + pattern_sim + metric_overlap) / 3
                
                if correlation >= self.config.correlation_threshold:
                    group1.correlated_groups.add(group2.id)
                    group2.correlated_groups.add(group1.id)
    
    async def _cleanup_alerts(self):
        """Clean up old alerts."""
        current_time = datetime.now()
        to_remove = []
        
        for alert_id, group in self.active_alerts.items():
            if (
                group.state in {AlertState.RESOLVED, AlertState.IGNORED} or
                current_time - group.last_seen > max(self.config.notification_cooldown.values())
            ):
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
    
    async def acknowledge_alert(
        self,
        alert_id: str
    ) -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].state = AlertState.ACKNOWLEDGED
            return True
        return False
    
    async def resolve_alert(
        self,
        alert_id: str,
        resolution_note: Optional[str] = None
    ) -> bool:
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            group = self.active_alerts[alert_id]
            group.state = AlertState.RESOLVED
            if resolution_note:
                group.context["resolution"] = resolution_note
            return True
        return False
    
    async def ignore_alert(
        self,
        alert_id: str,
        reason: Optional[str] = None
    ) -> bool:
        """Ignore an alert."""
        if alert_id in self.active_alerts:
            group = self.active_alerts[alert_id]
            group.state = AlertState.IGNORED
            if reason:
                group.context["ignore_reason"] = reason
            return True
        return False
    
    async def create_alert_plots(
        self,
        preset_name: Optional[str] = None
    ) -> Dict[str, go.Figure]:
        """Create alert visualization plots."""
        plots = {}
        
        # Alert timeline
        timeline_fig = go.Figure()
        
        alerts = self.alert_history
        if preset_name:
            alerts = [a for a in alerts if a["preset_name"] == preset_name]
        
        severities = []
        timestamps = []
        hover_texts = []
        
        for alert in alerts:
            severities.append(alert["severity"])
            timestamps.append(datetime.fromisoformat(alert["timestamp"]))
            hover_texts.append(
                f"Type: {alert['pattern_type']}<br>"
                f"Score: {alert['severity_score']:.2f}<br>"
                f"Probability: {alert['probability']:.2f}"
            )
        
        for severity in AlertSeverity:
            mask = [s == severity.value for s in severities]
            if any(mask):
                timeline_fig.add_trace(
                    go.Scatter(
                        x=[t for t, m in zip(timestamps, mask) if m],
                        y=[1 for m in mask if m],
                        mode="markers",
                        name=severity.value,
                        marker=dict(
                            size=12,
                            symbol="diamond" if severity in {AlertSeverity.CRITICAL, AlertSeverity.HIGH}
                                   else "circle",
                            color={
                                AlertSeverity.CRITICAL: "red",
                                AlertSeverity.HIGH: "orange",
                                AlertSeverity.MEDIUM: "yellow",
                                AlertSeverity.LOW: "blue",
                                AlertSeverity.INFO: "green"
                            }[severity]
                        ),
                        hovertext=[t for t, m in zip(hover_texts, mask) if m]
                    )
                )
        
        timeline_fig.update_layout(
            title="Alert Timeline",
            showlegend=True,
            yaxis_visible=False
        )
        plots["timeline"] = timeline_fig
        
        # Active alerts heatmap
        if self.active_alerts:
            active_fig = go.Figure()
            
            group_ids = list(self.active_alerts.keys())
            correlation_matrix = np.zeros((len(group_ids), len(group_ids)))
            
            for i, id1 in enumerate(group_ids):
                for j, id2 in enumerate(group_ids):
                    if id1 in self.active_alerts[id2].correlated_groups:
                        correlation_matrix[i, j] = 1
            
            active_fig.add_trace(
                go.Heatmap(
                    z=correlation_matrix,
                    x=group_ids,
                    y=group_ids,
                    colorscale="Viridis",
                    name="Alert Correlations"
                )
            )
            
            active_fig.update_layout(
                title="Active Alert Correlations",
                xaxis_title="Alert ID",
                yaxis_title="Alert ID"
            )
            plots["correlations"] = active_fig
        
        # Severity distribution
        severity_counts = pd.Series([
            alert["severity"] for alert in alerts
        ]).value_counts()
        
        dist_fig = go.Figure(
            go.Bar(
                x=severity_counts.index,
                y=severity_counts.values,
                name="Severity Distribution"
            )
        )
        
        dist_fig.update_layout(
            title="Alert Severity Distribution",
            xaxis_title="Severity",
            yaxis_title="Count"
        )
        plots["distribution"] = dist_fig
        
        return plots

def create_alert_manager(
    detector: AnomalyDetector,
    config: Optional[AlertConfig] = None
) -> AlertManager:
    """Create alert manager."""
    return AlertManager(detector, config)

if __name__ == "__main__":
    from .anomaly_analysis import create_anomaly_detector
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
        alert_manager = create_alert_manager(detector)
        
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
        
        # Generate test data
        for i in range(100):
            # Add some anomalies
            severity = np.random.choice([0.95, 0.75, 0.55, 0.35, 0.15])
            if np.random.random() < 0.1:  # 10% chance of anomaly
                await detector.detect_anomalies("test_preset")
                await alert_manager.process_anomalies("test_preset")
            
            if i % 20 == 0:
                plots = await alert_manager.create_alert_plots("test_preset")
                for name, fig in plots.items():
                    fig.write_html(f"test_alerts_{name}.html")
    
    asyncio.run(main())
