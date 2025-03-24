"""Performance regression testing for monitoring dashboard."""

import pytest
import asyncio
from datetime import datetime, timedelta
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import statistics
import logging
from dataclasses import dataclass
import psutil
import gc

from proxy.analysis.monitoring.visualization import MonitoringDashboard
from .benchmark_visualization import generate_test_data, MockStore

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    render_time: float
    memory_usage: float
    cpu_usage: float
    frame_rate: float
    dom_nodes: int
    event_listeners: int
    reflow_count: int
    repaint_count: int

class PerformanceBudget:
    """Performance budget thresholds."""
    
    # Timing budgets (milliseconds)
    MAX_RENDER_TIME = 100
    MAX_INTERACTION_TIME = 50
    MAX_UPDATE_TIME = 200
    
    # Resource budgets
    MAX_MEMORY_USAGE = 100 * 1024 * 1024  # 100MB
    MAX_CPU_USAGE = 60  # percentage
    MIN_FRAME_RATE = 30  # fps
    
    # DOM budgets
    MAX_DOM_NODES = 1500
    MAX_EVENT_LISTENERS = 500
    
    # Layout budgets
    MAX_REFLOWS = 10  # per second
    MAX_REPAINTS = 20  # per second
    
    # Thresholds for regression detection
    REGRESSION_THRESHOLD = 1.2  # 20% degradation
    ALERT_THRESHOLD = 1.5  # 50% degradation

class PerformanceHistory:
    """Track and analyze performance history."""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.history: Dict[str, List[Dict]] = self._load_history()
    
    def _load_history(self) -> Dict[str, List[Dict]]:
        """Load performance history from file."""
        if not self.history_file.exists():
            return {}
        
        try:
            with self.history_file.open() as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading performance history: {e}")
            return {}
    
    def save_history(self):
        """Save performance history to file."""
        with self.history_file.open('w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_metrics(
        self,
        test_name: str,
        metrics: PerformanceMetrics
    ):
        """Add new metrics to history."""
        if test_name not in self.history:
            self.history[test_name] = []
        
        self.history[test_name].append({
            "timestamp": datetime.now().isoformat(),
            "render_time": metrics.render_time,
            "memory_usage": metrics.memory_usage,
            "cpu_usage": metrics.cpu_usage,
            "frame_rate": metrics.frame_rate,
            "dom_nodes": metrics.dom_nodes,
            "event_listeners": metrics.event_listeners,
            "reflow_count": metrics.reflow_count,
            "repaint_count": metrics.repaint_count
        })
        
        self.save_history()
    
    def detect_regressions(
        self,
        test_name: str,
        current_metrics: PerformanceMetrics,
        window_size: int = 10
    ) -> Dict[str, float]:
        """Detect performance regressions."""
        if test_name not in self.history:
            return {}
        
        # Get recent history
        history = self.history[test_name][-window_size:]
        if not history:
            return {}
        
        regressions = {}
        
        # Compare current metrics with historical averages
        historical_metrics = {
            "render_time": statistics.mean(h["render_time"] for h in history),
            "memory_usage": statistics.mean(h["memory_usage"] for h in history),
            "cpu_usage": statistics.mean(h["cpu_usage"] for h in history),
            "frame_rate": statistics.mean(h["frame_rate"] for h in history),
            "dom_nodes": statistics.mean(h["dom_nodes"] for h in history),
            "event_listeners": statistics.mean(h["event_listeners"] for h in history),
            "reflow_count": statistics.mean(h["reflow_count"] for h in history),
            "repaint_count": statistics.mean(h["repaint_count"] for h in history)
        }
        
        # Check each metric for regressions
        current_values = {
            "render_time": current_metrics.render_time,
            "memory_usage": current_metrics.memory_usage,
            "cpu_usage": current_metrics.cpu_usage,
            "frame_rate": current_metrics.frame_rate,
            "dom_nodes": current_metrics.dom_nodes,
            "event_listeners": current_metrics.event_listeners,
            "reflow_count": current_metrics.reflow_count,
            "repaint_count": current_metrics.repaint_count
        }
        
        for metric, historical in historical_metrics.items():
            current = current_values[metric]
            if historical > 0:  # Avoid division by zero
                ratio = current / historical
                if ratio > PerformanceBudget.REGRESSION_THRESHOLD:
                    regressions[metric] = ratio
        
        return regressions

class TestPerformanceRegression:
    """Performance regression tests."""
    
    @pytest.fixture
    def performance_history(self, tmp_path):
        """Create performance history tracker."""
        return PerformanceHistory(tmp_path / "performance_history.json")
    
    @pytest.fixture
    def dashboard(self):
        """Create dashboard for testing."""
        metrics, alerts = generate_test_data(10, 100, 5)
        store = MockStore(metrics)
        return MonitoringDashboard(
            alert_manager=None,
            metric_store=store,
            timeseries_store=store
        )

    def measure_rendering_performance(
        self,
        selenium_driver,
        component_id: str
    ) -> Tuple[float, List[float]]:
        """Measure rendering performance of component."""
        start_time = time.time()
        frame_timestamps = []
        
        # Start performance monitoring
        selenium_driver.execute_script("""
            window.performanceData = {
                frames: [],
                startTime: performance.now()
            };
            
            window.rafCallback = function() {
                window.performanceData.frames.push(performance.now());
                window.requestAnimationFrame(window.rafCallback);
            };
            
            window.requestAnimationFrame(window.rafCallback);
        """)
        
        # Wait for component to render
        selenium_driver.find_element_by_id(component_id)
        
        # Stop monitoring after 1 second
        time.sleep(1)
        
        # Get performance data
        frame_data = selenium_driver.execute_script("""
            window.cancelAnimationFrame(window.rafCallback);
            return window.performanceData;
        """)
        
        render_time = time.time() - start_time
        frame_times = [
            t - frame_data["startTime"]
            for t in frame_data["frames"]
        ]
        
        return render_time, frame_times

    def get_performance_metrics(
        self,
        selenium_driver,
        component_id: str
    ) -> PerformanceMetrics:
        """Collect comprehensive performance metrics."""
        # Measure rendering
        render_time, frame_times = self.measure_rendering_performance(
            selenium_driver,
            component_id
        )
        
        # Calculate frame rate
        if frame_times:
            frame_intervals = np.diff(frame_times)
            frame_rate = 1000 / np.mean(frame_intervals)  # Convert ms to fps
        else:
            frame_rate = 0
        
        # Get memory usage
        memory_usage = psutil.Process().memory_info().rss
        
        # Get CPU usage
        cpu_usage = psutil.Process().cpu_percent()
        
        # Get DOM metrics
        dom_metrics = selenium_driver.execute_script("""
            return {
                nodes: document.querySelectorAll('*').length,
                listeners: window.getEventListeners ?
                    Object.keys(window.getEventListeners(document)).length : 0
            };
        """)
        
        # Get layout metrics
        layout_metrics = selenium_driver.execute_script("""
            const entries = performance.getEntriesByType('layout-shift');
            return {
                reflows: entries.length,
                repaints: performance.getEntriesByType('paint').length
            };
        """)
        
        return PerformanceMetrics(
            render_time=render_time * 1000,  # Convert to milliseconds
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            frame_rate=frame_rate,
            dom_nodes=dom_metrics["nodes"],
            event_listeners=dom_metrics["listeners"],
            reflow_count=layout_metrics["reflows"],
            repaint_count=layout_metrics["repaints"]
        )
    
    def test_component_performance(
        self,
        selenium_driver,
        dashboard,
        performance_history
    ):
        """Test component rendering performance."""
        dashboard.run(port=8057)
        selenium_driver.get("http://localhost:8057")
        
        for component_id in [
            "alert-severity-chart",
            "metric-timeline",
            "active-alerts"
        ]:
            # Measure performance
            metrics = self.get_performance_metrics(
                selenium_driver,
                component_id
            )
            
            # Check against budget
            assert metrics.render_time <= PerformanceBudget.MAX_RENDER_TIME, \
                f"Render time exceeded budget for {component_id}"
            
            assert metrics.memory_usage <= PerformanceBudget.MAX_MEMORY_USAGE, \
                f"Memory usage exceeded budget for {component_id}"
            
            assert metrics.cpu_usage <= PerformanceBudget.MAX_CPU_USAGE, \
                f"CPU usage exceeded budget for {component_id}"
            
            assert metrics.frame_rate >= PerformanceBudget.MIN_FRAME_RATE, \
                f"Frame rate below budget for {component_id}"
            
            # Check for regressions
            regressions = performance_history.detect_regressions(
                f"component_{component_id}",
                metrics
            )
            
            if regressions:
                logger.warning(
                    f"Performance regressions detected for {component_id}:"
                    f"\n{json.dumps(regressions, indent=2)}"
                )
            
            # Store metrics
            performance_history.add_metrics(
                f"component_{component_id}",
                metrics
            )
    
    def test_interaction_performance(
        self,
        selenium_driver,
        dashboard,
        performance_history
    ):
        """Test interaction performance."""
        dashboard.run(port=8058)
        selenium_driver.get("http://localhost:8058")
        
        # Test interactions
        interactions = [
            ("refresh-button", "click"),
            ("time-range", "change"),
            ("alert-id", "input")
        ]
        
        for element_id, action in interactions:
            # Clear caches
            selenium_driver.execute_script("window.gc()")
            gc.collect()
            
            # Measure interaction
            start_time = time.time()
            
            if action == "click":
                selenium_driver.find_element_by_id(element_id).click()
            elif action == "change":
                selenium_driver.find_element_by_id(element_id).send_keys("1h")
            elif action == "input":
                selenium_driver.find_element_by_id(element_id).send_keys("test")
            
            # Wait for updates to complete
            time.sleep(0.1)
            
            interaction_time = (time.time() - start_time) * 1000
            
            # Check against budget
            assert interaction_time <= PerformanceBudget.MAX_INTERACTION_TIME, \
                f"Interaction time exceeded budget for {element_id}"
            
            # Get metrics after interaction
            metrics = self.get_performance_metrics(
                selenium_driver,
                "metric-timeline"  # Check main component
            )
            
            # Check for regressions
            regressions = performance_history.detect_regressions(
                f"interaction_{element_id}",
                metrics
            )
            
            if regressions:
                logger.warning(
                    f"Performance regressions detected for {element_id} interaction:"
                    f"\n{json.dumps(regressions, indent=2)}"
                )
            
            # Store metrics
            performance_history.add_metrics(
                f"interaction_{element_id}",
                metrics
            )
    
    def test_update_performance(
        self,
        selenium_driver,
        dashboard,
        performance_history
    ):
        """Test update performance."""
        dashboard.run(port=8059)
        selenium_driver.get("http://localhost:8059")
        
        # Trigger updates
        update_scenarios = [
            ("small", 5),
            ("medium", 20),
            ("large", 100)
        ]
        
        for size, num_updates in update_scenarios:
            start_time = time.time()
            
            for _ in range(num_updates):
                # Simulate data update
                selenium_driver.execute_script(
                    "window.dispatchEvent(new Event('data-update'))"
                )
                time.sleep(0.05)
            
            update_time = (time.time() - start_time) * 1000 / num_updates
            
            # Check against budget
            assert update_time <= PerformanceBudget.MAX_UPDATE_TIME, \
                f"Update time exceeded budget for {size} updates"
            
            # Get metrics after updates
            metrics = self.get_performance_metrics(
                selenium_driver,
                "metric-timeline"
            )
            
            # Check for regressions
            regressions = performance_history.detect_regressions(
                f"update_{size}",
                metrics
            )
            
            if regressions:
                logger.warning(
                    f"Performance regressions detected for {size} updates:"
                    f"\n{json.dumps(regressions, indent=2)}"
                )
            
            # Store metrics
            performance_history.add_metrics(
                f"update_{size}",
                metrics
            )

def analyze_performance_trends():
    """Analyze performance trends and generate report."""
    history = PerformanceHistory(Path("performance_history.json"))
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "trends": {},
        "regressions": [],
        "improvements": []
    }
    
    for test_name, metrics in history.history.items():
        if len(metrics) < 2:
            continue
        
        # Calculate trends
        report["trends"][test_name] = {
            metric: {
                "mean": statistics.mean(m[metric] for m in metrics),
                "std": statistics.stdev(m[metric] for m in metrics),
                "trend": (metrics[-1][metric] - metrics[0][metric]) /
                        metrics[0][metric] * 100  # Percentage change
            }
            for metric in [
                "render_time",
                "memory_usage",
                "cpu_usage",
                "frame_rate"
            ]
        }
        
        # Identify significant changes
        latest = metrics[-1]
        baseline = statistics.mean(
            m[metric] for m in metrics[:-1]
        )
        
        for metric in report["trends"][test_name]:
            change = (latest[metric] - baseline) / baseline
            if change > PerformanceBudget.REGRESSION_THRESHOLD:
                report["regressions"].append({
                    "test": test_name,
                    "metric": metric,
                    "change": change * 100
                })
            elif change < -PerformanceBudget.REGRESSION_THRESHOLD:
                report["improvements"].append({
                    "test": test_name,
                    "metric": metric,
                    "change": change * 100
                })
    
    # Save report
    with open("performance_trend_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report

if __name__ == "__main__":
    analyze_performance_trends()
