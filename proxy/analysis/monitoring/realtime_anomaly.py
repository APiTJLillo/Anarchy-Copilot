"""Real-time anomaly detection and streaming analysis."""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import queue
import threading
import numpy as np
from collections import defaultdict
import time
import json
from pathlib import Path

from .anomaly_detection import AnomalyDetector, AnomalyConfig
from .alert_aggregation import AlertAggregator
from .resource_alerts import ResourceAlertManager

logger = logging.getLogger(__name__)

@dataclass
class StreamingConfig:
    """Configuration for real-time streaming analysis."""
    buffer_size: int = 1000
    update_interval: float = 1.0  # seconds
    batch_size: int = 10
    warmup_samples: int = 100
    alert_cooldown: timedelta = timedelta(minutes=5)
    sliding_window: timedelta = timedelta(minutes=30)

class StreamingDetector:
    """Real-time streaming anomaly detection."""
    
    def __init__(
        self,
        config: StreamingConfig,
        detector: AnomalyDetector,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        self.config = config
        self.detector = detector
        self.callbacks = callbacks or {}
        
        self.pattern_buffer: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self.streaming_stats: Dict[str, Any] = defaultdict(dict)
        self.last_alerts: Dict[str, datetime] = {}
        
        self._running = False
        self._processor_thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start real-time detection."""
        if self._running:
            return
        
        self._running = True
        self._processor_thread = threading.Thread(
            target=self._process_stream,
            daemon=True
        )
        self._processor_thread.start()
    
    def stop(self):
        """Stop real-time detection."""
        self._running = False
        if self._processor_thread:
            self._processor_thread.join()
            self._processor_thread = None
    
    def add_pattern(self, pattern: Dict[str, Any]):
        """Add pattern to processing queue."""
        key = pattern["group_key"]
        
        # Initialize streaming stats if needed
        if key not in self.streaming_stats:
            self._initialize_stats(key)
        
        # Add to buffer
        self.pattern_buffer[key].put(pattern)
        
        # Trim buffer if needed
        while self.pattern_buffer[key].qsize() > self.config.buffer_size:
            self.pattern_buffer[key].get()
    
    def _initialize_stats(self, key: str):
        """Initialize streaming statistics."""
        self.streaming_stats[key] = {
            "count": 0,
            "means": defaultdict(float),
            "m2s": defaultdict(float),  # For online variance
            "last_update": datetime.now(),
            "anomaly_counts": defaultdict(int),
            "recent_anomalies": []
        }
    
    def _process_stream(self):
        """Process streaming patterns."""
        while self._running:
            start_time = time.time()
            
            # Process each pattern group
            for key in list(self.pattern_buffer.keys()):
                self._process_group(key)
            
            # Sleep for remaining interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.config.update_interval - elapsed)
            time.sleep(sleep_time)
    
    def _process_group(self, key: str):
        """Process patterns for a group."""
        # Collect batch of patterns
        patterns = []
        while (
            len(patterns) < self.config.batch_size and
            not self.pattern_buffer[key].empty()
        ):
            patterns.append(self.pattern_buffer[key].get())
        
        if not patterns:
            return
        
        # Update streaming stats
        self._update_streaming_stats(key, patterns)
        
        # Check for anomalies if enough samples
        if self.streaming_stats[key]["count"] >= self.config.warmup_samples:
            self._check_anomalies(key, patterns)
    
    def _update_streaming_stats(
        self,
        key: str,
        patterns: List[Dict[str, Any]]
    ):
        """Update streaming statistics."""
        stats = self.streaming_stats[key]
        
        for pattern in patterns:
            stats["count"] += 1
            
            # Update running means and variances
            for metric, values in pattern["metrics_summary"].items():
                mean = values["mean"]
                delta = mean - stats["means"][metric]
                stats["means"][metric] += delta / stats["count"]
                delta2 = mean - stats["means"][metric]
                stats["m2s"][metric] += delta * delta2
        
        stats["last_update"] = datetime.now()
        
        # Calculate streaming standard deviations
        stats["stds"] = {
            metric: np.sqrt(m2 / (stats["count"] - 1))
            if stats["count"] > 1 else 0
            for metric, m2 in stats["m2s"].items()
        }
    
    def _check_anomalies(
        self,
        key: str,
        patterns: List[Dict[str, Any]]
    ):
        """Check for anomalies in new patterns."""
        stats = self.streaming_stats[key]
        current_time = datetime.now()
        
        # Skip if in cooldown
        if (
            key in self.last_alerts and
            current_time - self.last_alerts[key] < self.config.alert_cooldown
        ):
            return
        
        for pattern in patterns:
            # Detect anomalies
            anomalies = self.detector.detect_anomalies(pattern)
            if anomalies:
                # Update anomaly statistics
                for anomaly in anomalies:
                    stats["anomaly_counts"][anomaly["type"]] += 1
                    stats["recent_anomalies"].append({
                        "timestamp": current_time.isoformat(),
                        "pattern": pattern,
                        "anomaly": anomaly
                    })
                
                # Trim recent anomalies
                cutoff = current_time - self.config.sliding_window
                stats["recent_anomalies"] = [
                    a for a in stats["recent_anomalies"]
                    if datetime.fromisoformat(a["timestamp"]) > cutoff
                ]
                
                # Call callbacks
                self._handle_anomalies(key, anomalies, pattern)
                
                # Update last alert time
                self.last_alerts[key] = current_time
    
    def _handle_anomalies(
        self,
        key: str,
        anomalies: List[Dict[str, Any]],
        pattern: Dict[str, Any]
    ):
        """Handle detected anomalies."""
        # Call general anomaly callback
        if "on_anomaly" in self.callbacks:
            self.callbacks["on_anomaly"](key, anomalies, pattern)
        
        # Call type-specific callbacks
        for anomaly in anomalies:
            callback_name = f"on_{anomaly['type']}_anomaly"
            if callback_name in self.callbacks:
                self.callbacks[callback_name](key, anomaly, pattern)
    
    def get_streaming_summary(
        self,
        key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get summary of streaming statistics."""
        if key is not None:
            if key not in self.streaming_stats:
                return {}
            return self._get_group_summary(key)
        
        return {
            key: self._get_group_summary(key)
            for key in self.streaming_stats
        }
    
    def _get_group_summary(self, key: str) -> Dict[str, Any]:
        """Get summary for a pattern group."""
        stats = self.streaming_stats[key]
        current_time = datetime.now()
        
        return {
            "pattern_count": stats["count"],
            "last_update": stats["last_update"].isoformat(),
            "age_minutes": (current_time - stats["last_update"]).total_seconds() / 60,
            "metrics": {
                metric: {
                    "mean": stats["means"][metric],
                    "std": stats["stds"][metric]
                }
                for metric in stats["means"]
            },
            "anomaly_counts": dict(stats["anomaly_counts"]),
            "recent_anomalies": len(stats["recent_anomalies"]),
            "anomaly_rate": (
                len(stats["recent_anomalies"]) /
                self.config.sliding_window.total_seconds() * 3600  # per hour
            )
        }

class RealtimeMonitor:
    """High-level real-time monitoring interface."""
    
    def __init__(
        self,
        alert_manager: Optional[ResourceAlertManager] = None,
        config: Optional[StreamingConfig] = None,
        callbacks: Optional[Dict[str, Callable]] = None
    ):
        self.alert_manager = alert_manager or ResourceAlertManager(None)
        self.aggregator = AlertAggregator(None, self.alert_manager)
        self.detector = AnomalyDetector(AnomalyConfig(), self.aggregator)
        self.streaming = StreamingDetector(
            config or StreamingConfig(),
            self.detector,
            callbacks
        )
    
    def start(self):
        """Start real-time monitoring."""
        self.streaming.start()
    
    def stop(self):
        """Stop real-time monitoring."""
        self.streaming.stop()
    
    def process_metrics(self, metrics: Dict[str, Any]):
        """Process new metrics."""
        # Create alert pattern
        pattern = self.aggregator.process_metrics(metrics)
        if pattern:
            self.streaming.add_pattern(pattern)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            "streaming": self.streaming.get_streaming_summary(),
            "detector": {
                "patterns": len(self.detector.history),
                "models": len(self.detector.models)
            },
            "aggregator": {
                "groups": len(self.aggregator.alert_groups)
            }
        }

def create_monitor(
    callbacks: Optional[Dict[str, Callable]] = None
) -> RealtimeMonitor:
    """Create real-time monitor with default configuration."""
    return RealtimeMonitor(callbacks=callbacks)

if __name__ == "__main__":
    # Example usage with callbacks
    def on_anomaly(key: str, anomalies: List[Dict[str, Any]], pattern: Dict[str, Any]):
        print(f"Anomalies detected for {key}: {len(anomalies)}")
    
    monitor = create_monitor({
        "on_anomaly": on_anomaly
    })
    
    monitor.start()
    
    # Simulate metrics
    metrics = {
        "cpu_percent": 95.0,
        "memory_percent": 80.0,
        "timestamp": datetime.now().isoformat()
    }
    
    monitor.process_metrics(metrics)
    time.sleep(5)
    monitor.stop()
