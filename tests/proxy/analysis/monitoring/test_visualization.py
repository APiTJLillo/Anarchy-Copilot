"""Tests for monitoring visualization components."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
from unittest.mock import AsyncMock, MagicMock, patch
import plotly.graph_objects as go
import dash.testing.composite as dash_testing
from dash.testing.composite import DashComposite

from proxy.analysis.monitoring.metrics import MetricValue, TimeseriesMetric
from proxy.analysis.monitoring.alerts import (
    Alert,
    AlertSeverity,
    AlertManager,
    AlertRule
)
from proxy.analysis.monitoring.storage import (
    MetricStore,
    TimeseriesStore
)
from proxy.analysis.monitoring.visualization import MonitoringDashboard

@pytest.fixture
def mock_alert_manager():
    """Create mock alert manager."""
    manager = MagicMock(spec=AlertManager)
    
    # Setup active alerts
    active_alerts = [
        Alert(
            id=f"test_{i}",
            rule_name=f"rule_{i}",
            metric_name=f"test.metric.{i}",
            value=100.0 + i * 10,
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now() - timedelta(minutes=i),
            description=f"Test alert {i}"
        )
        for i in range(3)
    ]
    manager.get_active_alerts.return_value = active_alerts
    
    # Setup resolved alerts
    resolved_alerts = [
        Alert(
            id=f"resolved_{i}",
            rule_name=f"rule_{i}",
            metric_name=f"test.metric.{i}",
            value=90.0 + i * 10,
            threshold=100.0,
            severity=AlertSeverity.INFO,
            timestamp=datetime.now() - timedelta(hours=i),
            description=f"Resolved alert {i}",
            resolved=True,
            resolved_at=datetime.now()
        )
        for i in range(2)
    ]
    manager.get_resolved_alerts.return_value = resolved_alerts
    
    return manager

@pytest.fixture
def mock_metric_store():
    """Create mock metric store."""
    store = AsyncMock(spec=MetricStore)
    
    async def mock_get_metric(*args, **kwargs):
        return TimeseriesMetric(
            name="test.metric",
            values=[float(i) for i in range(10)],
            timestamps=[
                datetime.now() - timedelta(minutes=i)
                for i in range(10)
            ]
        )
    
    store.get_metric = mock_get_metric
    return store

@pytest.fixture
def mock_timeseries_store():
    """Create mock timeseries store."""
    store = AsyncMock(spec=TimeseriesStore)
    
    async def mock_get_timeseries(*args, **kwargs):
        return TimeseriesMetric(
            name="test.metric",
            values=[float(i) for i in range(10)],
            timestamps=[
                datetime.now() - timedelta(minutes=i)
                for i in range(10)
            ]
        )
    
    store.get_timeseries = mock_get_timeseries
    return store

@pytest.fixture
def dashboard(
    mock_alert_manager,
    mock_metric_store,
    mock_timeseries_store
):
    """Create monitoring dashboard for testing."""
    return MonitoringDashboard(
        alert_manager=mock_alert_manager,
        metric_store=mock_metric_store,
        timeseries_store=mock_timeseries_store
    )

class TestMonitoringDashboard:
    """Test dashboard functionality."""

    def test_dashboard_initialization(self, dashboard):
        """Test dashboard setup."""
        assert dashboard.app is not None
        assert dashboard.update_interval == 10
    
    def test_layout_creation(self, dashboard):
        """Test dashboard layout creation."""
        layout = dashboard.app.layout
        
        # Check main components
        assert "Performance Monitoring Dashboard" in str(layout)
        assert "alert-severity-chart" in str(layout)
        assert "metric-timeline" in str(layout)
        assert "active-alerts" in str(layout)
        assert "recent-alerts" in str(layout)
    
    @pytest.mark.asyncio
    async def test_update_dashboard(self, dashboard):
        """Test dashboard update callback."""
        # Get callback
        update_callback = None
        for callback in dashboard.app.callback_map.values():
            if "update_dashboard" in str(callback.callback):
                update_callback = callback.callback
                break
        
        assert update_callback is not None
        
        # Test callback
        results = await update_callback(1, 1, "1h")
        
        # Check outputs
        severity_fig, alert_list, timeline_fig, active_rows, recent_rows = results
        
        # Verify severity chart
        assert isinstance(severity_fig, go.Figure)
        assert "Alert Severity Distribution" in severity_fig.layout.title.text
        
        # Verify alert list
        assert isinstance(alert_list, dash_testing.DashElement)
        assert "rule_0" in str(alert_list)
        
        # Verify timeline
        assert isinstance(timeline_fig, go.Figure)
        assert "Metric Timeline with Alerts" in timeline_fig.layout.title.text
        
        # Verify alert tables
        assert len(active_rows) == 3  # 3 active alerts
        assert len(recent_rows) == 2  # 2 resolved alerts
    
    @pytest.mark.asyncio
    async def test_alert_actions(self, dashboard):
        """Test alert action callbacks."""
        # Get callback
        action_callback = None
        for callback in dashboard.app.callback_map.values():
            if "handle_alert_action" in str(callback.callback):
                action_callback = callback.callback
                break
        
        assert action_callback is not None
        
        # Test acknowledge
        result = await action_callback(1, None, "test_1", "test_user")
        assert result == ""  # Input should clear
        dashboard.alert_manager.acknowledge_alert.assert_called_once_with(
            "test_1",
            "test_user"
        )
        
        # Test resolve
        result = await action_callback(None, 1, "test_1", None)
        assert result == ""  # Input should clear
        dashboard.alert_manager.resolve_alert.assert_called_once_with("test_1")
    
    @pytest.mark.asyncio
    async def test_error_handling(self, dashboard):
        """Test error handling in callbacks."""
        # Make alert manager throw errors
        dashboard.alert_manager.acknowledge_alert.side_effect = Exception("Test error")
        dashboard.alert_manager.resolve_alert.side_effect = Exception("Test error")
        
        # Get callback
        action_callback = None
        for callback in dashboard.app.callback_map.values():
            if "handle_alert_action" in str(callback.callback):
                action_callback = callback.callback
                break
        
        assert action_callback is not None
        
        # Test error cases
        result = await action_callback(1, None, "test_1", "test_user")
        assert result == "test_1"  # Input should stay on error
        
        result = await action_callback(None, 1, "test_1", None)
        assert result == "test_1"  # Input should stay on error

@pytest.mark.integration
class TestDashboardIntegration:
    """Integration tests using dash_duo."""
    
    def test_dashboard_load(self, dash_duo, dashboard):
        """Test dashboard loads correctly."""
        # Start dashboard
        dash_duo.start_server(dashboard.app)
        
        # Wait for components to load
        dash_duo.wait_for_element("#alert-severity-chart")
        dash_duo.wait_for_element("#metric-timeline")
        dash_duo.wait_for_element("#active-alerts")
        
        # Check initial state
        assert dash_duo.get_logs() == []  # No console errors
        
        # Test refresh button
        dash_duo.find_element("#refresh-button").click()
        dash_duo.wait_for_element_by_id("alert-severity-chart")
    
    def test_alert_interactions(self, dash_duo, dashboard):
        """Test alert interaction flows."""
        dash_duo.start_server(dashboard.app)
        
        # Enter alert ID and user
        alert_input = dash_duo.find_element("#alert-id")
        alert_input.send_keys("test_1")
        
        user_input = dash_duo.find_element("#user")
        user_input.send_keys("test_user")
        
        # Click acknowledge
        dash_duo.find_element("#acknowledge-button").click()
        
        # Verify inputs cleared
        assert alert_input.get_attribute("value") == ""
        assert user_input.get_attribute("value") == ""
    
    def test_time_range_selection(self, dash_duo, dashboard):
        """Test time range selector."""
        dash_duo.start_server(dashboard.app)
        
        # Select different time ranges
        dropdown = dash_duo.find_element("#time-range")
        
        for value in ["1h", "6h", "24h", "7d"]:
            dropdown.send_keys(value)
            dash_duo.wait_for_element_by_id("metric-timeline")
            
            # Verify timeline updated
            timeline = dash_duo.find_element("#metric-timeline")
            assert timeline is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
