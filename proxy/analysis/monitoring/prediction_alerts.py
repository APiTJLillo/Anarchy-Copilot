"""Alert handlers for performance predictions."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import aiohttp
import aiosmtplib
from email.message import EmailMessage
import numpy as np

from .performance_prediction import PredictionResult, PerformancePredictor
from .alerts import AlertSeverity, AlertHandler
from .test_performance_regression import PerformanceBudget

logger = logging.getLogger(__name__)

@dataclass
class PredictionAlert:
    """Alert generated from predictions."""
    id: str
    timestamp: datetime
    metric: str
    test_name: str
    predicted_value: float
    current_value: float
    threshold: float
    probability: float
    time_to_breach: Optional[int]
    severity: AlertSeverity
    description: str
    contributing_factors: List[Tuple[str, float]]

class PredictionAlertManager:
    """Manage prediction-based alerts."""
    
    def __init__(
        self,
        alert_dir: Path,
        handlers: Optional[List[AlertHandler]] = None,
        threshold_buffer: float = 0.1,  # Buffer before threshold
        probability_threshold: float = 0.7,  # Minimum probability for alert
        urgency_days: int = 7  # Days threshold for urgent alerts
    ):
        self.alert_dir = alert_dir
        self.alert_dir.mkdir(parents=True, exist_ok=True)
        self.handlers = handlers or []
        self.threshold_buffer = threshold_buffer
        self.probability_threshold = probability_threshold
        self.urgency_days = urgency_days
        self._active_alerts: Dict[str, PredictionAlert] = {}
        
        # Load existing alerts
        self._load_alerts()
    
    def _load_alerts(self):
        """Load existing alerts from disk."""
        for alert_file in self.alert_dir.glob("*.json"):
            try:
                with alert_file.open() as f:
                    data = json.load(f)
                    alert = PredictionAlert(**data)
                    self._active_alerts[alert.id] = alert
            except Exception as e:
                logger.error(f"Error loading alert {alert_file}: {e}")
    
    def _save_alert(self, alert: PredictionAlert):
        """Save alert to disk."""
        alert_file = self.alert_dir / f"{alert.id}.json"
        with alert_file.open("w") as f:
            json.dump(asdict(alert), f, indent=2)
    
    def _determine_severity(
        self,
        probability: float,
        time_to_breach: Optional[int]
    ) -> AlertSeverity:
        """Determine alert severity based on probability and time."""
        if probability > 0.9:
            return AlertSeverity.CRITICAL
        elif probability > 0.8:
            return AlertSeverity.ERROR
        elif probability > 0.7:
            if time_to_breach and time_to_breach <= self.urgency_days:
                return AlertSeverity.ERROR
            return AlertSeverity.WARNING
        return AlertSeverity.INFO
    
    async def process_predictions(
        self,
        predictions: Dict[str, List[PredictionResult]],
        current_values: Dict[str, Dict[str, float]]
    ):
        """Process predictions and generate alerts."""
        new_alerts = []
        
        for test_name, test_predictions in predictions.items():
            for pred in test_predictions:
                # Get budget threshold
                threshold_attr = f"MAX_{pred.metric.upper()}"
                if not hasattr(PerformanceBudget, threshold_attr):
                    continue
                
                threshold = getattr(PerformanceBudget, threshold_attr)
                buffered_threshold = threshold * (1 - self.threshold_buffer)
                
                # Check if alert needed
                if (pred.probability_over_budget >= self.probability_threshold or
                    pred.predicted_value > buffered_threshold):
                    
                    # Create alert
                    alert = PredictionAlert(
                        id=f"pred_{test_name}_{pred.metric}_{datetime.now().timestamp()}",
                        timestamp=datetime.now(),
                        metric=pred.metric,
                        test_name=test_name,
                        predicted_value=pred.predicted_value,
                        current_value=current_values[test_name][pred.metric],
                        threshold=threshold,
                        probability=pred.probability_over_budget,
                        time_to_breach=pred.time_to_threshold,
                        severity=self._determine_severity(
                            pred.probability_over_budget,
                            pred.time_to_threshold
                        ),
                        description=self._generate_description(
                            pred,
                            test_name,
                            threshold,
                            current_values[test_name][pred.metric]
                        ),
                        contributing_factors=pred.contributing_factors
                    )
                    
                    # Add to active alerts
                    self._active_alerts[alert.id] = alert
                    self._save_alert(alert)
                    new_alerts.append(alert)
        
        # Send alerts
        if new_alerts:
            await self._send_alerts(new_alerts)
    
    def _generate_description(
        self,
        prediction: PredictionResult,
        test_name: str,
        threshold: float,
        current_value: float
    ) -> str:
        """Generate alert description."""
        description = (
            f"Performance degradation predicted for {test_name}\n"
            f"Metric: {prediction.metric}\n"
            f"Current value: {current_value:.2f}\n"
            f"Predicted value: {prediction.predicted_value:.2f}\n"
            f"Threshold: {threshold:.2f}\n"
            f"Probability of breach: {prediction.probability_over_budget:.1%}\n"
        )
        
        if prediction.time_to_threshold:
            description += (
                f"Estimated time to threshold breach: "
                f"{prediction.time_to_threshold} days\n"
            )
        
        if prediction.contributing_factors:
            description += "\nContributing factors:\n"
            for factor, importance in prediction.contributing_factors:
                description += f"- {factor}: {importance:.1%}\n"
        
        return description
    
    async def _send_alerts(self, alerts: List[PredictionAlert]):
        """Send alerts through handlers."""
        for handler in self.handlers:
            for alert in alerts:
                try:
                    await handler.handle_alert(alert)
                except Exception as e:
                    logger.error(f"Error sending alert through handler: {e}")
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert."""
        if alert_id in self._active_alerts:
            alert = self._active_alerts.pop(alert_id)
            alert_file = self.alert_dir / f"{alert.id}.json"
            if alert_file.exists():
                alert_file.unlink()
            
            # Archive alert
            archive_dir = self.alert_dir / "resolved"
            archive_dir.mkdir(exist_ok=True)
            with (archive_dir / f"{alert.id}.json").open("w") as f:
                data = asdict(alert)
                data["resolved_at"] = datetime.now().isoformat()
                json.dump(data, f, indent=2)
    
    def get_active_alerts(
        self,
        min_severity: Optional[AlertSeverity] = None
    ) -> List[PredictionAlert]:
        """Get active prediction alerts."""
        alerts = list(self._active_alerts.values())
        if min_severity:
            alerts = [
                a for a in alerts
                if a.severity.value >= min_severity.value
            ]
        return sorted(
            alerts,
            key=lambda a: (a.severity.value, a.probability),
            reverse=True
        )
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of active alerts."""
        alerts = self.get_active_alerts()
        return {
            "total_alerts": len(alerts),
            "by_severity": {
                severity.value: len([
                    a for a in alerts
                    if a.severity == severity
                ])
                for severity in AlertSeverity
            },
            "by_metric": {},
            "urgent_predictions": [
                {
                    "test": alert.test_name,
                    "metric": alert.metric,
                    "time_to_breach": alert.time_to_breach,
                    "probability": alert.probability
                }
                for alert in alerts
                if (
                    alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]
                    and alert.time_to_breach
                    and alert.time_to_breach <= self.urgency_days
                )
            ]
        }

class SlackPredictionHandler(AlertHandler):
    """Send prediction alerts to Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session = aiohttp.ClientSession()
    
    async def handle_alert(self, alert: PredictionAlert):
        """Handle prediction alert."""
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#7b0000"
        }[alert.severity]
        
        message = {
            "attachments": [
                {
                    "color": color,
                    "title": (
                        f"Performance Prediction Alert: {alert.test_name}"
                    ),
                    "fields": [
                        {
                            "title": "Metric",
                            "value": alert.metric,
                            "short": True
                        },
                        {
                            "title": "Severity",
                            "value": alert.severity.value,
                            "short": True
                        },
                        {
                            "title": "Current Value",
                            "value": f"{alert.current_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Predicted Value",
                            "value": f"{alert.predicted_value:.2f}",
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{alert.threshold:.2f}",
                            "short": True
                        },
                        {
                            "title": "Probability",
                            "value": f"{alert.probability:.1%}",
                            "short": True
                        }
                    ],
                    "text": alert.description
                }
            ]
        }
        
        try:
            async with self.session.post(
                self.webhook_url,
                json=message
            ) as response:
                return response.status < 400
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False

class EmailPredictionHandler(AlertHandler):
    """Send prediction alerts via email."""
    
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls
    
    async def handle_alert(self, alert: PredictionAlert):
        """Handle prediction alert."""
        message = EmailMessage()
        message.set_content(alert.description)
        
        message["Subject"] = (
            f"[{alert.severity.value.upper()}] Performance Prediction Alert: "
            f"{alert.test_name} - {alert.metric}"
        )
        message["From"] = self.from_addr
        message["To"] = ", ".join(self.to_addrs)
        
        try:
            async with aiosmtplib.SMTP(
                hostname=self.host,
                port=self.port,
                use_tls=self.use_tls
            ) as smtp:
                await smtp.login(self.username, self.password)
                await smtp.send_message(message)
            return True
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

async def monitor_predictions():
    """Run prediction monitoring."""
    # Setup predictor and alert manager
    predictor = PerformancePredictor(
        history_file=Path("performance_history.json")
    )
    
    alert_manager = PredictionAlertManager(
        alert_dir=Path("prediction_alerts"),
        handlers=[
            SlackPredictionHandler(
                webhook_url="https://hooks.slack.com/your-webhook"
            ),
            EmailPredictionHandler(
                host="smtp.example.com",
                port=587,
                username="alerts@example.com",
                password="your-password",
                from_addr="alerts@example.com",
                to_addrs=["team@example.com"]
            )
        ]
    )
    
    while True:
        try:
            # Generate predictions
            predictions = predictor.generate_predictions()
            
            # Get current values
            current_values = {}  # Implement getting current values
            
            # Process predictions
            await alert_manager.process_predictions(
                predictions,
                current_values
            )
            
            # Generate summary
            summary = alert_manager.get_alert_summary()
            logger.info(f"Alert summary: {json.dumps(summary, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error in prediction monitoring: {e}")
        
        await asyncio.sleep(300)  # Check every 5 minutes

if __name__ == "__main__":
    asyncio.run(monitor_predictions())
