"""Tests for metric storage components."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import json
import os
import numpy as np
from typing import AsyncGenerator

from proxy.analysis.monitoring.metrics import MetricValue, TimeseriesMetric
from proxy.analysis.monitoring.storage import (
    SQLiteMetricStore,
    TimeseriesStore,
    StatisticsStore
)

@pytest.fixture
async def db_path(tmp_path) -> Path:
    """Create temporary database path."""
    return tmp_path / "test_metrics.db"

@pytest.fixture
async def sqlite_store(db_path) -> AsyncGenerator[SQLiteMetricStore, None]:
    """Create SQLite store for testing."""
    store = SQLiteMetricStore(db_path)
    yield store
    if db_path.exists():
        os.unlink(db_path)

@pytest.fixture
async def timeseries_store(sqlite_store) -> TimeseriesStore:
    """Create timeseries store for testing."""
    return TimeseriesStore(
        metric_store=sqlite_store,
        default_window=timedelta(minutes=1)
    )

@pytest.fixture
async def stats_store(tmp_path) -> StatisticsStore:
    """Create statistics store for testing."""
    stats_dir = tmp_path / "stats"
    return StatisticsStore(stats_dir)

@pytest.fixture
def sample_metric() -> MetricValue:
    """Create sample metric."""
    return MetricValue(
        name="test.metric",
        value=42.0,
        timestamp=datetime.now(),
        tags={"test": "true"},
        source="test",
        unit="count"
    )

@pytest.fixture
def sample_timeseries() -> TimeseriesMetric:
    """Create sample time series."""
    now = datetime.now()
    return TimeseriesMetric(
        name="test.series",
        values=[float(i) for i in range(10)],
        timestamps=[now + timedelta(seconds=i) for i in range(10)],
        tags={"test": "true"},
        source="test",
        unit="count"
    )

class TestSQLiteMetricStore:
    """Test SQLite metric storage."""

    async def test_store_metric(self, sqlite_store, sample_metric):
        """Test storing single metric."""
        success = await sqlite_store.store_metric(sample_metric)
        assert success
        
        # Verify storage
        latest = await sqlite_store.get_latest(sample_metric.name)
        assert latest is not None
        assert latest.value == sample_metric.value
        assert latest.tags == sample_metric.tags
    
    async def test_store_metrics(self, sqlite_store):
        """Test storing multiple metrics."""
        now = datetime.now()
        metrics = [
            MetricValue(
                name="test.multi",
                value=float(i),
                timestamp=now + timedelta(seconds=i)
            )
            for i in range(5)
        ]
        
        success = await sqlite_store.store_metrics(metrics)
        assert success
        
        # Verify storage
        series = await sqlite_store.get_metric(
            name="test.multi",
            start_time=now - timedelta(seconds=1),
            end_time=now + timedelta(seconds=10)
        )
        assert len(series.values) == 5
    
    async def test_get_metric_with_tags(self, sqlite_store):
        """Test retrieving metrics with tag filtering."""
        now = datetime.now()
        metrics = [
            MetricValue(
                name="test.tags",
                value=1.0,
                timestamp=now,
                tags={"env": "prod"}
            ),
            MetricValue(
                name="test.tags",
                value=2.0,
                timestamp=now,
                tags={"env": "dev"}
            )
        ]
        
        await sqlite_store.store_metrics(metrics)
        
        # Query with tag filter
        series = await sqlite_store.get_metric(
            name="test.tags",
            start_time=now - timedelta(seconds=1),
            end_time=now + timedelta(seconds=1),
            tags={"env": "prod"}
        )
        
        assert len(series.values) == 1
        assert series.values[0] == 1.0
    
    async def test_get_latest(self, sqlite_store):
        """Test retrieving latest metric value."""
        now = datetime.now()
        metrics = [
            MetricValue(
                name="test.latest",
                value=float(i),
                timestamp=now + timedelta(seconds=i)
            )
            for i in range(3)
        ]
        
        await sqlite_store.store_metrics(metrics)
        
        latest = await sqlite_store.get_latest("test.latest")
        assert latest is not None
        assert latest.value == 2.0

class TestTimeseriesStore:
    """Test time series storage."""

    async def test_store_timeseries(self, timeseries_store, sample_timeseries):
        """Test storing time series data."""
        success = await timeseries_store.store_timeseries(sample_timeseries)
        assert success
        
        # Verify storage with aggregation
        series = await timeseries_store.get_timeseries(
            name=sample_timeseries.name,
            start_time=min(sample_timeseries.timestamps),
            end_time=max(sample_timeseries.timestamps)
        )
        
        assert len(series.values) == len(sample_timeseries.values)
    
    async def test_get_aggregated_timeseries(self, timeseries_store, sample_timeseries):
        """Test retrieving aggregated time series."""
        await timeseries_store.store_timeseries(sample_timeseries)
        
        # Get with aggregation window
        series = await timeseries_store.get_timeseries(
            name=sample_timeseries.name,
            start_time=min(sample_timeseries.timestamps),
            end_time=max(sample_timeseries.timestamps),
            window=timedelta(seconds=5)
        )
        
        # Should have fewer points due to aggregation
        assert len(series.values) < len(sample_timeseries.values)
    
    async def test_timeseries_caching(self, timeseries_store):
        """Test time series caching behavior."""
        now = datetime.now()
        series = TimeseriesMetric(
            name="test.cache",
            values=[float(i) for i in range(100)],
            timestamps=[now + timedelta(seconds=i) for i in range(100)]
        )
        
        await timeseries_store.store_timeseries(series)
        
        # Multiple reads should use cache
        for _ in range(3):
            result = await timeseries_store.get_timeseries(
                name="test.cache",
                start_time=now,
                end_time=now + timedelta(seconds=100)
            )
            assert len(result.values) == 100

class TestStatisticsStore:
    """Test statistics storage."""

    async def test_store_statistics(self, stats_store):
        """Test storing statistical results."""
        stats = {
            "mean": 42.0,
            "std": 5.0,
            "percentiles": {
                "p50": 41.0,
                "p90": 48.0
            }
        }
        
        success = await stats_store.store_statistics(
            name="test.stats",
            stats=stats
        )
        assert success
        
        # Verify storage
        results = await stats_store.get_statistics("test.stats")
        assert len(results) == 1
        assert "stats" in results[0]
        assert results[0]["stats"]["mean"] == 42.0
    
    async def test_get_statistics_time_range(self, stats_store):
        """Test retrieving statistics with time filtering."""
        now = datetime.now()
        
        # Store multiple statistics
        for i in range(3):
            await stats_store.store_statistics(
                name="test.time",
                stats={"value": i},
                timestamp=now + timedelta(hours=i)
            )
        
        # Query with time range
        results = await stats_store.get_statistics(
            name="test.time",
            start_time=now + timedelta(hours=1),
            end_time=now + timedelta(hours=2)
        )
        
        assert len(results) == 1
        assert results[0]["stats"]["value"] == 1
    
    async def test_statistics_file_organization(self, stats_store):
        """Test statistics file organization."""
        await stats_store.store_statistics(
            name="test.files",
            stats={"test": True}
        )
        
        # Verify directory structure
        stat_dir = stats_store.base_path / "test.files"
        assert stat_dir.exists()
        assert len(list(stat_dir.glob("*.json"))) == 1

async def test_full_storage_pipeline(
    sqlite_store,
    timeseries_store,
    stats_store,
    sample_timeseries
):
    """Test complete storage pipeline."""
    # Store time series data
    await timeseries_store.store_timeseries(sample_timeseries)
    
    # Calculate and store statistics
    stats = {
        "mean": np.mean(sample_timeseries.values),
        "std": np.std(sample_timeseries.values),
        "count": len(sample_timeseries.values)
    }
    await stats_store.store_statistics(
        name=sample_timeseries.name,
        stats=stats
    )
    
    # Verify retrieval
    series = await timeseries_store.get_timeseries(
        name=sample_timeseries.name,
        start_time=min(sample_timeseries.timestamps),
        end_time=max(sample_timeseries.timestamps)
    )
    assert len(series.values) == len(sample_timeseries.values)
    
    stored_stats = await stats_store.get_statistics(sample_timeseries.name)
    assert len(stored_stats) == 1
    assert "mean" in stored_stats[0]["stats"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
