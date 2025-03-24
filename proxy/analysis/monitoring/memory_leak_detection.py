"""Memory leak detection and analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import tracemalloc
from collections import defaultdict
import objsize
import gc
import sys
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .scheduler_profiling import ProfilingHook, ProfileSnapshot

@dataclass
class LeakConfig:
    """Configuration for leak detection."""
    enabled: bool = True
    sample_interval: float = 60.0  # seconds
    history_size: int = 100
    growth_threshold: float = 0.1  # 10% growth rate
    gc_threshold: int = 1000  # objects
    trace_depth: int = 3
    min_leak_size: int = 1024  # bytes
    confidence_level: float = 0.95
    enable_object_tracking: bool = True
    track_custom_types: Set[str] = field(default_factory=set)
    visualization_dir: Optional[str] = "leak_analysis"

@dataclass
class LeakSnapshot:
    """Snapshot of memory state."""
    timestamp: datetime
    total_size: int
    total_count: int
    type_stats: Dict[str, Dict[str, int]]
    traceback_stats: List[Dict[str, Any]]
    generation_counts: List[int]
    context: Dict[str, Any] = field(default_factory=dict)

class LeakPattern:
    """Pattern of memory growth."""
    def __init__(
        self,
        type_name: str,
        locations: List[str],
        growth_rate: float,
        correlation: float,
        size_impact: float
    ):
        self.type_name = type_name
        self.locations = locations
        self.growth_rate = growth_rate
        self.correlation = correlation
        self.size_impact = size_impact
        self.first_seen = datetime.now()
        self.last_seen = datetime.now()
        self.occurrence_count = 1
        self.context: Dict[str, Any] = {}

class LeakDetector:
    """Detect and analyze memory leaks."""
    
    def __init__(
        self,
        profiling_hook: ProfilingHook,
        config: LeakConfig = None
    ):
        self.profiling_hook = profiling_hook
        self.config = config or LeakConfig()
        
        # Snapshot storage
        self.snapshots: List[LeakSnapshot] = []
        self.patterns: Dict[str, LeakPattern] = {}
        
        # Object tracking
        self.type_histories: Dict[str, List[Tuple[datetime, int, int]]] = defaultdict(list)
        self.tracked_objects: Dict[int, datetime] = {}
        
        # Initialize tracemalloc if not started
        if not tracemalloc.is_tracing():
            tracemalloc.start(self.config.trace_depth)
    
    async def take_snapshot(self) -> LeakSnapshot:
        """Take memory snapshot."""
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics("traceback")
        
        # Collect type statistics
        type_stats = defaultdict(lambda: {"count": 0, "size": 0})
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            if (
                self.config.enable_object_tracking and
                (not self.config.track_custom_types or
                 type_name in self.config.track_custom_types)
            ):
                try:
                    size = objsize.get_deep_size(obj)
                    type_stats[type_name]["count"] += 1
                    type_stats[type_name]["size"] += size
                    
                    # Track object
                    self.tracked_objects[id(obj)] = datetime.now()
                except Exception:
                    continue
        
        # Create snapshot
        snapshot = LeakSnapshot(
            timestamp=datetime.now(),
            total_size=sum(s.size for s in stats),
            total_count=sum(s.count for s in stats),
            type_stats=dict(type_stats),
            traceback_stats=[{
                "traceback": [
                    f"{frame.filename}:{frame.lineno}"
                    for frame in stat.traceback
                ],
                "size": stat.size,
                "count": stat.count
            } for stat in stats],
            generation_counts=[
                len(gc.get_objects(i))
                for i in range(gc.get_count()[0])
            ]
        )
        
        # Store snapshot
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.config.history_size:
            self.snapshots.pop(0)
        
        # Update type histories
        current_time = datetime.now()
        for type_name, stats in type_stats.items():
            self.type_histories[type_name].append((
                current_time,
                stats["count"],
                stats["size"]
            ))
            
            # Trim old history
            while (
                len(self.type_histories[type_name]) > self.config.history_size or
                (
                    self.type_histories[type_name][0][0] <
                    current_time - timedelta(seconds=self.config.sample_interval * self.config.history_size)
                )
            ):
                self.type_histories[type_name].pop(0)
        
        return snapshot
    
    async def analyze_leaks(self) -> Dict[str, Any]:
        """Analyze memory for leaks."""
        if len(self.snapshots) < 2:
            return {"status": "insufficient_data"}
        
        # Calculate overall growth
        total_growth = (
            self.snapshots[-1].total_size -
            self.snapshots[0].total_size
        )
        growth_rate = total_growth / max(1, self.snapshots[-1].total_size)
        
        # Analyze type patterns
        type_analysis = await self._analyze_type_patterns()
        
        # Check generation counts
        gc_analysis = await self._analyze_gc_generations()
        
        # Find memory leaks
        leaks = await self._detect_leaks()
        
        # Track leaked objects
        if self.config.enable_object_tracking:
            leaked_objects = await self._track_leaked_objects()
        else:
            leaked_objects = {}
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "total_growth": total_growth,
            "growth_rate": growth_rate,
            "type_analysis": type_analysis,
            "gc_analysis": gc_analysis,
            "leaks": leaks,
            "leaked_objects": leaked_objects
        }
    
    async def _analyze_type_patterns(self) -> Dict[str, Any]:
        """Analyze object type patterns."""
        patterns = {}
        
        for type_name, history in self.type_histories.items():
            if len(history) < 2:
                continue
            
            # Calculate growth rate
            times = [t.timestamp() for t, _, _ in history]
            counts = [c for _, c, _ in history]
            sizes = [s for _, _, s in history]
            
            # Fit linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                times,
                sizes
            )
            
            # Calculate metrics
            growth_rate = slope / (sizes[0] + 1)  # Avoid division by zero
            correlation = r_value ** 2
            size_impact = sizes[-1] / max(1, self.snapshots[-1].total_size)
            
            if growth_rate > self.config.growth_threshold:
                # Find allocation sites
                locations = set()
                for snapshot in self.snapshots:
                    for stat in snapshot.traceback_stats:
                        for frame in stat.traceback:
                            if type_name in frame:
                                locations.add(frame)
                
                # Create or update pattern
                if type_name not in self.patterns:
                    self.patterns[type_name] = LeakPattern(
                        type_name=type_name,
                        locations=list(locations),
                        growth_rate=growth_rate,
                        correlation=correlation,
                        size_impact=size_impact
                    )
                else:
                    pattern = self.patterns[type_name]
                    pattern.growth_rate = growth_rate
                    pattern.correlation = correlation
                    pattern.size_impact = size_impact
                    pattern.last_seen = datetime.now()
                    pattern.occurrence_count += 1
                    pattern.locations = list(locations)
            
            patterns[type_name] = {
                "growth_rate": growth_rate,
                "correlation": correlation,
                "size_impact": size_impact,
                "total_size": sizes[-1],
                "total_count": counts[-1]
            }
        
        return patterns
    
    async def _analyze_gc_generations(self) -> Dict[str, Any]:
        """Analyze garbage collector generations."""
        if not self.snapshots:
            return {}
        
        current = self.snapshots[-1].generation_counts
        previous = (
            self.snapshots[-2].generation_counts
            if len(self.snapshots) > 1
            else [0] * len(current)
        )
        
        return {
            "counts": current,
            "deltas": [c - p for c, p in zip(current, previous)],
            "total_objects": sum(current),
            "oldest_generation_ratio": (
                current[-1] / sum(current)
                if sum(current) > 0 else 0
            )
        }
    
    async def _detect_leaks(self) -> List[Dict[str, Any]]:
        """Detect memory leaks."""
        leaks = []
        
        for type_name, pattern in self.patterns.items():
            if (
                pattern.growth_rate > self.config.growth_threshold and
                pattern.correlation > self.config.confidence_level and
                pattern.size_impact * self.snapshots[-1].total_size > self.config.min_leak_size
            ):
                leaks.append({
                    "type": type_name,
                    "growth_rate": pattern.growth_rate,
                    "correlation": pattern.correlation,
                    "size_impact": pattern.size_impact,
                    "total_size": self.snapshots[-1].type_stats.get(
                        type_name, {"size": 0}
                    )["size"],
                    "occurrence_count": pattern.occurrence_count,
                    "first_seen": pattern.first_seen.isoformat(),
                    "last_seen": pattern.last_seen.isoformat(),
                    "locations": pattern.locations
                })
        
        return sorted(
            leaks,
            key=lambda x: x["size_impact"],
            reverse=True
        )
    
    async def _track_leaked_objects(self) -> Dict[str, Any]:
        """Track potentially leaked objects."""
        current_time = datetime.now()
        leaked = defaultdict(lambda: {
            "count": 0,
            "total_size": 0,
            "age_stats": {
                "min": float("inf"),
                "max": 0,
                "avg": 0
            }
        })
        
        # Clean up tracked objects
        current_ids = {id(obj) for obj in gc.get_objects()}
        self.tracked_objects = {
            obj_id: timestamp
            for obj_id, timestamp in self.tracked_objects.items()
            if obj_id in current_ids
        }
        
        # Analyze remaining objects
        for obj_id, timestamp in self.tracked_objects.items():
            try:
                obj = next(
                    obj for obj in gc.get_objects()
                    if id(obj) == obj_id
                )
                type_name = type(obj).__name__
                age = (current_time - timestamp).total_seconds()
                size = objsize.get_deep_size(obj)
                
                stats = leaked[type_name]
                stats["count"] += 1
                stats["total_size"] += size
                stats["age_stats"]["min"] = min(stats["age_stats"]["min"], age)
                stats["age_stats"]["max"] = max(stats["age_stats"]["max"], age)
                stats["age_stats"]["avg"] = (
                    (stats["age_stats"]["avg"] * (stats["count"] - 1) + age) /
                    stats["count"]
                )
            except Exception:
                continue
        
        return {
            "total_tracked": len(self.tracked_objects),
            "type_stats": dict(leaked)
        }
    
    async def create_leak_visualizations(self) -> Dict[str, go.Figure]:
        """Create leak analysis visualizations."""
        plots = {}
        
        if not self.snapshots:
            return plots
        
        # Memory growth plot
        growth_fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=["Total Memory Usage", "Object Counts"]
        )
        
        timestamps = [s.timestamp for s in self.snapshots]
        sizes = [s.total_size for s in self.snapshots]
        counts = [s.total_count for s in self.snapshots]
        
        growth_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=sizes,
                name="Memory Size",
                mode="lines+markers"
            ),
            row=1,
            col=1
        )
        
        growth_fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=counts,
                name="Object Count",
                mode="lines+markers"
            ),
            row=2,
            col=1
        )
        
        growth_fig.update_layout(
            height=600,
            title="Memory Growth Over Time"
        )
        plots["growth"] = growth_fig
        
        # Type distribution plot
        if self.type_histories:
            type_fig = go.Figure()
            
            for type_name, history in self.type_histories.items():
                if len(history) < 2:
                    continue
                
                times = [t for t, _, _ in history]
                sizes = [s for _, _, s in history]
                
                type_fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=sizes,
                        name=type_name,
                        mode="lines"
                    )
                )
            
            type_fig.update_layout(
                title="Memory Usage by Type",
                xaxis_title="Time",
                yaxis_title="Size (bytes)"
            )
            plots["types"] = type_fig
        
        # Generation plot
        gen_fig = go.Figure()
        gen_counts = [s.generation_counts for s in self.snapshots]
        
        for gen in range(len(gen_counts[0])):
            gen_fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=[counts[gen] for counts in gen_counts],
                    name=f"Generation {gen}",
                    stackgroup="generations"
                )
            )
        
        gen_fig.update_layout(
            title="GC Generations",
            xaxis_title="Time",
            yaxis_title="Object Count"
        )
        plots["generations"] = gen_fig
        
        # Save plots
        if self.config.visualization_dir:
            for name, fig in plots.items():
                fig.write_html(f"{self.config.visualization_dir}/leak_{name}.html")
        
        return plots

def create_leak_detector(
    profiling_hook: ProfilingHook,
    config: Optional[LeakConfig] = None
) -> LeakDetector:
    """Create leak detector."""
    return LeakDetector(profiling_hook, config)

if __name__ == "__main__":
    from .scheduler_profiling import create_profiling_hook
    from .scheduler_monitoring import create_scheduler_monitor
    from .report_scheduler import create_report_scheduler
    from .validation_reports import create_validation_reporter
    from .rule_validation import create_rule_validator
    from .notification_rules import create_rule_engine
    from .alert_notifications import create_notification_manager
    from .anomaly_alerts import create_alert_manager
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
        notifier = create_notification_manager(alert_manager)
        engine = create_rule_engine(notifier)
        validator = create_rule_validator(engine)
        reporter = create_validation_reporter(validator)
        scheduler = create_report_scheduler(reporter)
        monitor = create_scheduler_monitor(scheduler)
        profiling = create_profiling_hook()
        leak_detector = create_leak_detector(profiling)
        
        monitor.add_hook(profiling)
        await monitor.start_monitoring()
        await scheduler.start()
        
        try:
            while True:
                # Take memory snapshot
                await leak_detector.take_snapshot()
                
                # Analyze leaks
                analysis = await leak_detector.analyze_leaks()
                if analysis["status"] == "success":
                    print("\nLeak Analysis:")
                    for leak in analysis["leaks"]:
                        print(
                            f"\n{leak['type']}:"
                            f"\n  Growth Rate: {leak['growth_rate']:.2%}"
                            f"\n  Size: {leak['total_size']:,} bytes"
                            f"\n  Impact: {leak['size_impact']:.2%}"
                        )
                
                # Create visualizations
                plots = await leak_detector.create_leak_visualizations()
                
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await scheduler.stop()
    
    asyncio.run(main())
