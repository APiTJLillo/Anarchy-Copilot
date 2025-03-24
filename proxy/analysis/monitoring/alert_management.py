"""Alert management system for anomaly detection."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from collections import defaultdict
from enum import Enum
import asyncio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .realtime_anomalies import RealtimeAnomalyDetector, RealtimeConfig
from .anomaly_detection import ExplorationAnomalyDetector

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class AlertStatus(Enum):
    """Alert status states."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

@dataclass
class AlertConfig:
    """Configuration for alert management."""
    severity_thresholds: Dict[AlertSeverity, float] = None
    auto_acknowledge: bool = True
    auto_resolve: bool = True
    resolution_window: int = 3600  # seconds
    alert_retention: int = 7  # days
    escalation_delay: int = 300  # seconds
    output_path: Optional[Path] = None
    
    def __post_init__(self):
        if self.severity_thresholds is None:
            self.severity_thresholds = {
                AlertSeverity.LOW: 0.7,
                AlertSeverity.MEDIUM: 0.8,
                AlertSeverity.HIGH: 0.9,
                AlertSeverity.CRITICAL: 0.95
            }

@dataclass
class Alert:
    """Alert record."""
    id: str
    type: str
    severity: AlertSeverity
    status: AlertStatus
    description: str
    timestamp: datetime
    source_data: Dict[str, Any]
    context: Dict[str, Any]
    resolution: Optional[Dict[str, Any]] = None
    assignee: Optional[str] = None
    updates: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.updates is None:
            self.updates = []
        self.updates.append({
            "timestamp": datetime.now(),
            "type": "creation",
            "details": "Alert created"
        })

class AlertManager:
    """Manage and track anomaly alerts."""
    
    def __init__(
        self,
        realtime: RealtimeAnomalyDetector,
        config: AlertConfig
    ):
        self.realtime = realtime
        self.config = config
        
        # Alert storage
        self.active_alerts = {}
        self.alert_history = []
        self.alert_metrics = defaultdict(int)
        
        # State tracking
        self.escalation_queue = []
        self.resolution_queue = []
        self.alert_callbacks = defaultdict(list)
        
        # Register with realtime detector
        self.realtime.add_alert_callback(self.process_alert)
    
    async def start_manager(self):
        """Start alert management."""
        # Start management tasks
        escalation_task = asyncio.create_task(self._process_escalations())
        resolution_task = asyncio.create_task(self._process_resolutions())
        cleanup_task = asyncio.create_task(self._cleanup_old_alerts())
        
        try:
            await asyncio.gather(
                escalation_task,
                resolution_task,
                cleanup_task
            )
        except Exception as e:
            logger.error(f"Alert management error: {e}")
    
    def process_alert(
        self,
        anomaly: Dict[str, Any]
    ):
        """Process incoming anomaly alert."""
        # Generate alert ID
        alert_id = f"alert_{len(self.alert_history)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine severity
        severity = self._determine_severity(anomaly["severity"])
        
        # Create alert record
        alert = Alert(
            id=alert_id,
            type=anomaly["type"],
            severity=severity,
            status=AlertStatus.NEW,
            description=anomaly["description"],
            timestamp=datetime.now(),
            source_data=anomaly,
            context=self._gather_context(anomaly)
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.alert_metrics["total"] += 1
        self.alert_metrics[f"severity_{severity.name}"] += 1
        
        # Auto-acknowledge if configured
        if self.config.auto_acknowledge:
            self.acknowledge_alert(alert_id)
        
        # Add to escalation queue if high severity
        if severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self.escalation_queue.append(alert_id)
        
        # Trigger callbacks
        self._trigger_callbacks("new", alert)
        
        return alert_id
    
    def acknowledge_alert(
        self,
        alert_id: str,
        assignee: Optional[str] = None
    ):
        """Acknowledge an alert."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.assignee = assignee
        alert.updates.append({
            "timestamp": datetime.now(),
            "type": "acknowledgment",
            "details": f"Alert acknowledged by {assignee or 'system'}"
        })
        
        self._trigger_callbacks("acknowledgment", alert)
        return True
    
    def update_alert(
        self,
        alert_id: str,
        status: AlertStatus,
        details: str,
        resolution: Optional[Dict[str, Any]] = None
    ):
        """Update alert status."""
        if alert_id not in self.active_alerts:
            return False
        
        alert = self.active_alerts[alert_id]
        alert.status = status
        alert.updates.append({
            "timestamp": datetime.now(),
            "type": "update",
            "details": details
        })
        
        if resolution:
            alert.resolution = resolution
            if status == AlertStatus.RESOLVED:
                self.alert_metrics["resolved"] += 1
        
        self._trigger_callbacks("update", alert)
        return True
    
    def add_callback(
        self,
        event_type: str,
        callback: Callable[[Alert], None]
    ):
        """Add callback for alert events."""
        self.alert_callbacks[event_type].append(callback)
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """Get active alerts with optional filtering."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert statistics."""
        return {
            "metrics": dict(self.alert_metrics),
            "active_count": len(self.active_alerts),
            "severity_distribution": {
                severity.name: len([
                    a for a in self.active_alerts.values()
                    if a.severity == severity
                ])
                for severity in AlertSeverity
            },
            "status_distribution": {
                status.value: len([
                    a for a in self.active_alerts.values()
                    if a.status == status
                ])
                for status in AlertStatus
            }
        }
    
    def visualize_alerts(self) -> go.Figure:
        """Create visualization of alert status."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Alert Timeline",
                "Severity Distribution",
                "Status Distribution",
                "Resolution Time"
            ]
        )
        
        # Alert timeline
        if self.alert_history:
            timestamps = [a.timestamp for a in self.alert_history]
            severities = [a.severity.value for a in self.alert_history]
            
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=severities,
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=[a.status.value for a in self.alert_history],
                        colorscale="Viridis"
                    ),
                    text=[a.description for a in self.alert_history],
                    name="Alerts"
                ),
                row=1,
                col=1
            )
        
        # Severity distribution
        severity_counts = [
            len([a for a in self.active_alerts.values() if a.severity == sev])
            for sev in AlertSeverity
        ]
        
        fig.add_trace(
            go.Bar(
                x=[sev.name for sev in AlertSeverity],
                y=severity_counts,
                name="Severity"
            ),
            row=1,
            col=2
        )
        
        # Status distribution
        status_counts = [
            len([a for a in self.active_alerts.values() if a.status == status])
            for status in AlertStatus
        ]
        
        fig.add_trace(
            go.Bar(
                x=[status.value for status in AlertStatus],
                y=status_counts,
                name="Status"
            ),
            row=2,
            col=1
        )
        
        # Resolution time
        resolved_alerts = [
            a for a in self.alert_history
            if a.resolution is not None
        ]
        
        if resolved_alerts:
            resolution_times = [
                (a.resolution["timestamp"] - a.timestamp).total_seconds() / 60
                for a in resolved_alerts
            ]
            
            fig.add_trace(
                go.Box(
                    y=resolution_times,
                    name="Resolution Time (minutes)"
                ),
                row=2,
                col=2
            )
        
        return fig
    
    async def _process_escalations(self):
        """Process alert escalations."""
        while True:
            try:
                # Check escalation queue
                for alert_id in list(self.escalation_queue):
                    if alert_id in self.active_alerts:
                        alert = self.active_alerts[alert_id]
                        
                        # Check if alert needs escalation
                        if (
                            alert.status in [AlertStatus.NEW, AlertStatus.ACKNOWLEDGED] and
                            (datetime.now() - alert.timestamp).total_seconds() >
                            self.config.escalation_delay
                        ):
                            # Escalate alert
                            alert.updates.append({
                                "timestamp": datetime.now(),
                                "type": "escalation",
                                "details": "Alert automatically escalated"
                            })
                            
                            self._trigger_callbacks("escalation", alert)
                            self.escalation_queue.remove(alert_id)
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Escalation processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_resolutions(self):
        """Process alert resolutions."""
        while True:
            try:
                # Check for auto-resolutions
                if self.config.auto_resolve:
                    for alert_id, alert in list(self.active_alerts.items()):
                        # Check if alert can be auto-resolved
                        if (
                            alert.status != AlertStatus.RESOLVED and
                            (datetime.now() - alert.timestamp).total_seconds() >
                            self.config.resolution_window
                        ):
                            latest_anomaly = self._check_continuing_anomaly(alert)
                            
                            if not latest_anomaly:
                                self.update_alert(
                                    alert_id,
                                    AlertStatus.RESOLVED,
                                    "Alert automatically resolved - no continuing anomaly",
                                    {
                                        "timestamp": datetime.now(),
                                        "resolver": "system",
                                        "reason": "auto_resolution"
                                    }
                                )
                
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Resolution processing error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts."""
        while True:
            try:
                current_time = datetime.now()
                retention_limit = current_time - timedelta(days=self.config.alert_retention)
                
                # Remove old resolved alerts
                old_alerts = [
                    alert_id
                    for alert_id, alert in self.active_alerts.items()
                    if (
                        alert.status == AlertStatus.RESOLVED and
                        alert.timestamp < retention_limit
                    )
                ]
                
                for alert_id in old_alerts:
                    del self.active_alerts[alert_id]
                
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Alert cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def _determine_severity(
        self,
        anomaly_score: float
    ) -> AlertSeverity:
        """Determine alert severity based on anomaly score."""
        for severity, threshold in sorted(
            self.config.severity_thresholds.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            if anomaly_score >= threshold:
                return severity
        return AlertSeverity.LOW
    
    def _gather_context(
        self,
        anomaly: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gather context information for alert."""
        context = {
            "detection_method": anomaly.get("detection_method", "unknown"),
            "related_metrics": [],
            "historical_context": {}
        }
        
        # Get related metrics
        if "scores" in anomaly:
            context["related_metrics"] = [
                {
                    "metric": metric,
                    "score": score
                }
                for metric, score in anomaly["scores"].items()
                if score > self.config.severity_thresholds[AlertSeverity.LOW]
            ]
        
        # Get historical context
        if hasattr(self.realtime.detector, "analyzer"):
            analyzer = self.realtime.detector.analyzer
            if hasattr(analyzer, "trend_cache"):
                recent_trends = analyzer.trend_cache.get(
                    max(analyzer.trend_cache.keys())
                )
                if recent_trends:
                    context["historical_context"] = {
                        "activity_trend": recent_trends["activity"].get("trend", {}),
                        "preference_trends": recent_trends["preferences"].get("trends", {})
                    }
        
        return context
    
    def _check_continuing_anomaly(
        self,
        alert: Alert
    ) -> bool:
        """Check if anomaly is still occurring."""
        # Get recent anomalies
        recent_alerts = [
            a for a in self.alert_history
            if (
                a.type == alert.type and
                a.timestamp > alert.timestamp and
                (datetime.now() - a.timestamp).total_seconds() <
                self.config.resolution_window / 2
            )
        ]
        
        return len(recent_alerts) > 0
    
    def _trigger_callbacks(
        self,
        event_type: str,
        alert: Alert
    ):
        """Trigger callbacks for event."""
        for callback in self.alert_callbacks[event_type]:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")
    
    def save_state(
        self,
        output_path: Optional[Path] = None
    ):
        """Save alert manager state."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save alert data
            alert_data = {
                "active_alerts": {
                    id: {
                        "type": alert.type,
                        "severity": alert.severity.name,
                        "status": alert.status.value,
                        "description": alert.description,
                        "timestamp": alert.timestamp.isoformat(),
                        "updates": alert.updates,
                        "resolution": alert.resolution
                    }
                    for id, alert in self.active_alerts.items()
                },
                "metrics": dict(self.alert_metrics),
                "last_update": datetime.now().isoformat()
            }
            
            alert_file = path / "alert_state.json"
            with open(alert_file, "w") as f:
                json.dump(alert_data, f, indent=2)
            
            # Save visualization
            viz = self.visualize_alerts()
            viz.write_html(str(path / "alert_visualization.html"))
            
            logger.info(f"Saved alert state to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save alert state: {e}")

def create_alert_manager(
    realtime: RealtimeAnomalyDetector,
    output_path: Optional[Path] = None
) -> AlertManager:
    """Create alert manager."""
    config = AlertConfig(output_path=output_path)
    return AlertManager(realtime, config)

if __name__ == "__main__":
    # Example usage
    from .realtime_anomalies import create_realtime_detector
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
    realtime = create_realtime_detector(detector)
    manager = create_alert_manager(
        realtime,
        output_path=Path("alert_management")
    )
    
    async def main():
        # Example alert callback
        def on_new_alert(alert):
            print(f"New alert: {alert.description}")
        
        def on_escalation(alert):
            print(f"Alert escalated: {alert.id}")
        
        # Add callbacks
        manager.add_callback("new", on_new_alert)
        manager.add_callback("escalation", on_escalation)
        
        # Start manager
        await manager.start_manager()
    
    # Run example
    asyncio.run(main())
