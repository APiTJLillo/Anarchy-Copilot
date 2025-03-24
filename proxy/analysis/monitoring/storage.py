"""Storage backends for metrics data."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import json
import sqlite3
import aiosqlite
from pathlib import Path
import logging
from .metrics import MetricValue, TimeseriesMetric, MetricAggregation

logger = logging.getLogger(__name__)

class MetricStore(ABC):
    """Base class for metric storage."""
    
    @abstractmethod
    async def store_metric(self, metric: MetricValue) -> bool:
        """Store a single metric value."""
        pass
    
    @abstractmethod
    async def store_metrics(self, metrics: List[MetricValue]) -> bool:
        """Store multiple metric values."""
        pass
    
    @abstractmethod
    async def get_metric(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> TimeseriesMetric:
        """Retrieve metric values for a time range."""
        pass
    
    @abstractmethod
    async def get_latest(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[MetricValue]:
        """Get latest value for a metric."""
        pass

class SQLiteMetricStore(MetricStore):
    """SQLite-based metric storage."""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = str(db_path)
        self._setup_db()
    
    def _setup_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp DATETIME NOT NULL,
                    tags TEXT,
                    source TEXT,
                    unit TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_name_time 
                ON metrics(name, timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_metrics_tags
                ON metrics(tags)
            """)
    
    async def store_metric(self, metric: MetricValue) -> bool:
        """Store a single metric value."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    INSERT INTO metrics (name, value, timestamp, tags, source, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metric.name,
                        metric.value,
                        metric.timestamp.isoformat(),
                        json.dumps(metric.tags or {}),
                        metric.source,
                        metric.unit
                    )
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing metric: {e}")
            return False
    
    async def store_metrics(self, metrics: List[MetricValue]) -> bool:
        """Store multiple metric values."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.executemany(
                    """
                    INSERT INTO metrics (name, value, timestamp, tags, source, unit)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            m.name,
                            m.value,
                            m.timestamp.isoformat(),
                            json.dumps(m.tags or {}),
                            m.source,
                            m.unit
                        )
                        for m in metrics
                    ]
                )
                await db.commit()
            return True
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
            return False
    
    async def get_metric(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        tags: Optional[Dict[str, str]] = None
    ) -> TimeseriesMetric:
        """Retrieve metric values for a time range."""
        query = """
            SELECT value, timestamp, tags, source, unit
            FROM metrics
            WHERE name = ?
            AND timestamp BETWEEN ? AND ?
        """
        params = [name, start_time.isoformat(), end_time.isoformat()]
        
        if tags:
            # Match specific tags
            tags_json = json.dumps(tags)
            query += " AND tags = ?"
            params.append(tags_json)
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    if not rows:
                        return TimeseriesMetric(
                            name=name,
                            values=[],
                            timestamps=[],
                            tags=tags
                        )
                    
                    values = []
                    timestamps = []
                    source = None
                    unit = None
                    
                    for row in rows:
                        values.append(row[0])
                        timestamps.append(datetime.fromisoformat(row[1]))
                        if not source:
                            source = row[3]
                        if not unit:
                            unit = row[4]
                    
                    return TimeseriesMetric(
                        name=name,
                        values=values,
                        timestamps=timestamps,
                        tags=tags,
                        source=source,
                        unit=unit
                    )
        except Exception as e:
            logger.error(f"Error retrieving metric: {e}")
            return TimeseriesMetric(name=name, values=[], timestamps=[], tags=tags)
    
    async def get_latest(
        self,
        name: str,
        tags: Optional[Dict[str, str]] = None
    ) -> Optional[MetricValue]:
        """Get latest value for a metric."""
        query = """
            SELECT value, timestamp, tags, source, unit
            FROM metrics
            WHERE name = ?
        """
        params = [name]
        
        if tags:
            tags_json = json.dumps(tags)
            query += " AND tags = ?"
            params.append(tags_json)
        
        query += " ORDER BY timestamp DESC LIMIT 1"
        
        try:
            async with aiosqlite.connect(self.db_path) as db:
                async with db.execute(query, params) as cursor:
                    row = await cursor.fetchone()
                    
                    if not row:
                        return None
                    
                    return MetricValue(
                        name=name,
                        value=row[0],
                        timestamp=datetime.fromisoformat(row[1]),
                        tags=json.loads(row[2]) if row[2] else None,
                        source=row[3],
                        unit=row[4]
                    )
        except Exception as e:
            logger.error(f"Error retrieving latest metric: {e}")
            return None

class TimeseriesStore:
    """Storage for time series data with aggregation support."""
    
    def __init__(
        self,
        metric_store: MetricStore,
        default_window: timedelta = timedelta(minutes=5)
    ):
        self.store = metric_store
        self.default_window = default_window
        self._aggregation_cache: Dict[str, MetricAggregation] = {}
    
    async def store_timeseries(
        self,
        metric: TimeseriesMetric,
        aggregate: bool = True
    ) -> bool:
        """Store a time series of metrics."""
        # Store individual values
        metrics = [
            MetricValue(
                name=metric.name,
                value=value,
                timestamp=ts,
                tags=metric.tags,
                source=metric.source,
                unit=metric.unit
            )
            for value, ts in zip(metric.values, metric.timestamps)
        ]
        
        success = await self.store.store_metrics(metrics)
        
        # Optionally create aggregations
        if aggregate and success:
            await self._update_aggregations(metric)
        
        return success
    
    async def get_timeseries(
        self,
        name: str,
        start_time: datetime,
        end_time: datetime,
        window: Optional[timedelta] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> TimeseriesMetric:
        """Get time series data with optional aggregation."""
        # Get raw data
        series = await self.store.get_metric(
            name=name,
            start_time=start_time,
            end_time=end_time,
            tags=tags
        )
        
        # Optionally aggregate
        if window:
            return series.resample(window)
        return series
    
    async def _update_aggregations(self, metric: TimeseriesMetric):
        """Update aggregation cache."""
        aggregations = metric.aggregate(self.default_window)
        
        for agg in aggregations:
            cache_key = f"{metric.name}:{agg.start_time.isoformat()}"
            self._aggregation_cache[cache_key] = agg
            
            # Store aggregation summary
            await self.store.store_metric(
                MetricValue(
                    name=f"{metric.name}.agg",
                    value=agg.mean,
                    timestamp=agg.start_time,
                    tags={
                        **(metric.tags or {}),
                        "aggregation": "mean",
                        "window": str(self.default_window)
                    },
                    source=metric.source,
                    unit=metric.unit
                )
            )

class StatisticsStore:
    """Storage for statistical analysis results."""
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_statistics(
        self,
        name: str,
        stats: Dict[str, Any],
        timestamp: Optional[datetime] = None
    ):
        """Store statistical analysis results."""
        timestamp = timestamp or datetime.now()
        
        # Create stat directory if needed
        stat_dir = self.base_path / name
        stat_dir.mkdir(exist_ok=True)
        
        # Save stats with timestamp
        stat_file = stat_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with stat_file.open('w') as f:
                json.dump(
                    {
                        "timestamp": timestamp.isoformat(),
                        "stats": stats
                    },
                    f,
                    indent=2
                )
            return True
        except Exception as e:
            logger.error(f"Error storing statistics: {e}")
            return False
    
    async def get_statistics(
        self,
        name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve statistical analysis results."""
        stat_dir = self.base_path / name
        if not stat_dir.exists():
            return []
        
        try:
            results = []
            for stat_file in stat_dir.glob("*.json"):
                # Parse timestamp from filename
                ts_str = stat_file.stem
                try:
                    ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue
                
                # Check time range
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    continue
                
                # Load stats
                with stat_file.open() as f:
                    stats = json.load(f)
                    results.append(stats)
            
            return sorted(results, key=lambda x: x["timestamp"])
            
        except Exception as e:
            logger.error(f"Error retrieving statistics: {e}")
            return []
