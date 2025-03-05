#!/usr/bin/env python3
"""Handle and route performance monitoring alerts."""

import sys
import json
import asyncio
import aiohttp
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import smtplib
from email.message import EmailMessage
import threading
import queue
import redis
from prometheus_client import Counter, Gauge, start_http_server
import websockets
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Alert:
    """Performance alert representation."""
    alert_id: str
    type: str
    value: float
    threshold: float
    timestamp: float
    function_name: Optional[str] = None
    severity: str = "info"
    status: str = "new"
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    notes: Optional[str] = None

class AlertAggregator:
    """Aggregate and deduplicate alerts."""
    
    def __init__(
        self,
        window_size: int = 300,  # 5 minutes
        max_duplicates: int = 5
    ):
        self.window_size = window_size
        self.max_duplicates = max_duplicates
        self.alert_windows: Dict[str, List[Alert]] = {}
        self.last_cleanup = datetime.now()

    def should_emit(self, alert: Alert) -> bool:
        """Determine if alert should be emitted."""
        self._cleanup_old_alerts()
        
        alert_key = f"{alert.type}:{alert.function_name}"
        if alert_key not in self.alert_windows:
            self.alert_windows[alert_key] = []
        
        window = self.alert_windows[alert_key]
        
        # Check for duplicates in window
        recent_duplicates = [
            a for a in window
            if abs(a.timestamp - alert.timestamp) < self.window_size
        ]
        
        if len(recent_duplicates) >= self.max_duplicates:
            return False
        
        window.append(alert)
        return True

    def _cleanup_old_alerts(self):
        """Clean up old alerts."""
        if (datetime.now() - self.last_cleanup).seconds < 60:
            return
        
        current_time = datetime.now().timestamp()
        for key, window in self.alert_windows.items():
            self.alert_windows[key] = [
                alert for alert in window
                if current_time - alert.timestamp < self.window_size
            ]
        
        self.last_cleanup = datetime.now()

class AlertRouter:
    """Route alerts to appropriate handlers."""
    
    def __init__(self, config_file: Path):
        self.config = self._load_config(config_file)
        self.handlers: Dict[str, List[AlertHandler]] = {}
        self._setup_handlers()
        
        # Prometheus metrics
        self.alert_counter = Counter(
            "performance_alerts_total",
            "Total number of performance alerts",
            ["type", "severity"]
        )
        self.alert_value_gauge = Gauge(
            "performance_alert_value",
            "Current value of performance metric",
            ["type"]
        )

    def _load_config(self, config_file: Path) -> Dict[str, Any]:
        """Load alert routing configuration."""
        with config_file.open() as f:
            return yaml.safe_load(f)

    def _setup_handlers(self):
        """Setup alert handlers based on configuration."""
        for alert_type, config in self.config["alert_types"].items():
            self.handlers[alert_type] = []
            
            for handler_config in config["handlers"]:
                handler_type = handler_config["type"]
                if handler_type == "email":
                    handler = EmailAlertHandler(
                        smtp_host=handler_config["smtp_host"],
                        smtp_port=handler_config["smtp_port"],
                        from_addr=handler_config["from"],
                        to_addrs=handler_config["to"]
                    )
                elif handler_type == "slack":
                    handler = SlackAlertHandler(
                        webhook_url=handler_config["webhook_url"]
                    )
                elif handler_type == "redis":
                    handler = RedisAlertHandler(
                        host=handler_config["host"],
                        port=handler_config["port"],
                        channel=handler_config["channel"]
                    )
                else:
                    logger.warning(f"Unknown handler type: {handler_type}")
                    continue
                
                self.handlers[alert_type].append(handler)

    async def route_alert(self, alert: Alert):
        """Route alert to appropriate handlers."""
        # Update Prometheus metrics
        self.alert_counter.labels(
            type=alert.type,
            severity=alert.severity
        ).inc()
        self.alert_value_gauge.labels(
            type=alert.type
        ).set(alert.value)
        
        if alert.type not in self.handlers:
            logger.warning(f"No handlers for alert type: {alert.type}")
            return
        
        # Route to each handler
        for handler in self.handlers[alert.type]:
            try:
                await handler.handle_alert(alert)
            except Exception as e:
                logger.error(f"Error in handler {handler.__class__.__name__}: {e}")

class AlertHandler:
    """Base class for alert handlers."""
    
    async def handle_alert(self, alert: Alert):
        """Handle an alert."""
        raise NotImplementedError

class EmailAlertHandler(AlertHandler):
    """Handle alerts via email."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        from_addr: str,
        to_addrs: List[str]
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.from_addr = from_addr
        self.to_addrs = to_addrs

    async def handle_alert(self, alert: Alert):
        """Send alert via email."""
        msg = EmailMessage()
        msg.set_content(
            f"Performance Alert\n"
            f"Type: {alert.type}\n"
            f"Value: {alert.value}\n"
            f"Threshold: {alert.threshold}\n"
            f"Function: {alert.function_name}\n"
            f"Time: {datetime.fromtimestamp(alert.timestamp)}\n"
        )
        
        msg["Subject"] = f"Performance Alert: {alert.type}"
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        
        async def send_email():
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.send_message(msg)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, send_email)

class SlackAlertHandler(AlertHandler):
    """Handle alerts via Slack."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.session = aiohttp.ClientSession()

    async def handle_alert(self, alert: Alert):
        """Send alert to Slack."""
        message = {
            "text": f"*Performance Alert*",
            "attachments": [{
                "color": self._get_color(alert.severity),
                "fields": [
                    {
                        "title": "Type",
                        "value": alert.type,
                        "short": True
                    },
                    {
                        "title": "Value",
                        "value": f"{alert.value:.2f}",
                        "short": True
                    },
                    {
                        "title": "Threshold",
                        "value": f"{alert.threshold:.2f}",
                        "short": True
                    },
                    {
                        "title": "Function",
                        "value": alert.function_name or "system",
                        "short": True
                    }
                ],
                "footer": datetime.fromtimestamp(alert.timestamp).isoformat()
            }]
        }
        
        async with self.session.post(
            self.webhook_url,
            json=message
        ) as response:
            if response.status >= 400:
                raise RuntimeError(
                    f"Error sending to Slack: {response.status}"
                )

    def _get_color(self, severity: str) -> str:
        """Get Slack color for severity."""
        return {
            "info": "#36a64f",
            "warning": "#ffcc00",
            "error": "#ff0000",
            "critical": "#7b0000"
        }.get(severity, "#cccccc")

class RedisAlertHandler(AlertHandler):
    """Handle alerts via Redis pub/sub."""
    
    def __init__(
        self,
        host: str,
        port: int,
        channel: str
    ):
        self.redis_client = redis.Redis(host=host, port=port)
        self.channel = channel

    async def handle_alert(self, alert: Alert):
        """Publish alert to Redis channel."""
        message = json.dumps({
            "type": alert.type,
            "value": alert.value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp,
            "function_name": alert.function_name,
            "severity": alert.severity
        })
        
        async def publish():
            self.redis_client.publish(self.channel, message)
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, publish)

class AlertManager:
    """Manage and coordinate alert handling."""
    
    def __init__(
        self,
        config_file: Path,
        metrics_port: int = 9090
    ):
        self.aggregator = AlertAggregator()
        self.router = AlertRouter(config_file)
        self.alert_queue: queue.Queue = queue.Queue()
        self.websocket_clients: Set[websockets.WebSocketServerProtocol] = set()
        
        # Start Prometheus metrics server
        start_http_server(metrics_port)

    async def start(self, port: int = 8766):
        """Start alert manager."""
        # Start websocket server for real-time alerts
        websocket_server = await websockets.serve(
            self._handle_client,
            "localhost",
            port
        )
        
        # Start alert processing
        asyncio.create_task(self._process_alerts())
        
        logger.info(f"Alert manager started on port {port}")
        await websocket_server.wait_closed()

    def queue_alert(self, alert: Alert):
        """Queue an alert for processing."""
        if self.aggregator.should_emit(alert):
            self.alert_queue.put(alert)

    async def _process_alerts(self):
        """Process queued alerts."""
        while True:
            try:
                # Get alert from queue
                alert = self.alert_queue.get_nowait()
                
                # Route alert
                await self.router.route_alert(alert)
                
                # Notify websocket clients
                if self.websocket_clients:
                    message = json.dumps({
                        "type": "alert",
                        "data": vars(alert)
                    })
                    await asyncio.gather(*[
                        client.send(message)
                        for client in self.websocket_clients
                    ])
                
            except queue.Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
                await asyncio.sleep(1)

    async def _handle_client(
        self,
        websocket: websockets.WebSocketServerProtocol,
        path: str
    ):
        """Handle websocket client connection."""
        self.websocket_clients.add(websocket)
        try:
            while True:
                try:
                    message = await websocket.recv()
                    command = json.loads(message)
                    
                    if command["type"] == "acknowledge":
                        await self._handle_acknowledgment(command["alert_id"], command["user"])
                    
                except websockets.exceptions.ConnectionClosed:
                    break
                except json.JSONDecodeError:
                    logger.warning("Invalid message format")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
        
        finally:
            self.websocket_clients.remove(websocket)

    async def _handle_acknowledgment(
        self,
        alert_id: str,
        user: str
    ):
        """Handle alert acknowledgment."""
        # Update alert status
        # This would typically involve a database update
        pass

async def main() -> int:
    """Main entry point."""
    try:
        config_file = Path("config/alert_config.yml")
        if not config_file.exists():
            logger.error("No alert configuration file found")
            return 1
        
        manager = AlertManager(config_file)
        await manager.start()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in alert manager: {e}")
        return 1

if __name__ == "__main__":
    asyncio.run(main())
