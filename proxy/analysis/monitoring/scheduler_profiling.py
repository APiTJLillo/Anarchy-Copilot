"""Performance profiling for report scheduler."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import logging
import cProfile
import pstats
import io
import tracemalloc
import linecache
from contextlib import contextmanager
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .scheduler_monitoring import (
    SchedulerMonitor, MonitoringConfig, MonitoringHook,
    SchedulerMetrics
)

@dataclass
class ProfilingConfig:
    """Configuration for scheduler profiling."""
    enabled: bool = True
    sample_interval: float = 0.1  # seconds
    max_snapshots: int = 100
    snapshot_on_alert: bool = True
    trace_malloc: bool = True
    profile_slow_tasks: bool = True
    slow_task_threshold: float = 5.0  # seconds
    profile_dir: Path = Path("profiling_results")
    flamegraph_enabled: bool = True
    line_profiling: bool = True
    max_frames: int = 25
    include_system_frames: bool = False

@dataclass
class ProfileSnapshot:
    """Snapshot of profiling data."""
    timestamp: datetime
    stats: pstats.Stats
    memory_snapshot: Optional[tracemalloc.Snapshot] = None
    context: Dict[str, Any] = field(default_factory=dict)

class ProfilingHook(MonitoringHook):
    """Hook for collecting profiling data."""
    
    def __init__(
        self,
        config: ProfilingConfig = None
    ):
        self.config = config or ProfilingConfig()
        self.profiler = cProfile.Profile()
        
        # Snapshot storage
        self.snapshots: List[ProfileSnapshot] = []
        
        # Memory tracking
        if self.config.trace_malloc:
            tracemalloc.start()
        
        # Create output directory
        if self.config.profile_dir:
            self.config.profile_dir.mkdir(parents=True, exist_ok=True)
    
    async def on_schedule_start(
        self,
        schedule: ReportSchedule
    ):
        """Start profiling for schedule."""
        self.profiler.enable()
    
    async def on_schedule_complete(
        self,
        schedule: ReportSchedule,
        duration: float,
        success: bool
    ):
        """Complete profiling for schedule."""
        self.profiler.disable()
        
        # Check if task was slow
        if (
            self.config.profile_slow_tasks and
            duration > self.config.slow_task_threshold
        ):
            await self._save_profile(schedule.name, "slow_task")
        
        # Take snapshot if enabled
        if len(self.snapshots) < self.config.max_snapshots:
            await self._take_snapshot({
                "schedule": schedule.name,
                "duration": duration,
                "success": success
            })
    
    async def on_error(
        self,
        schedule: Optional[ReportSchedule],
        error: Exception,
        context: Dict[str, Any]
    ):
        """Handle profiling on error."""
        if self.config.snapshot_on_alert:
            await self._take_snapshot({
                "schedule": schedule.name if schedule else None,
                "error": str(error),
                "traceback": traceback.format_exc(),
                **context
            })
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect profiling metrics."""
        metrics = {}
        
        if self.config.trace_malloc:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("lineno")
            
            metrics["memory"] = {
                "total": sum(stat.size for stat in stats),
                "count": sum(stat.count for stat in stats),
                "peak": max(stat.size for stat in stats)
            }
        
        return metrics
    
    async def _take_snapshot(
        self,
        context: Dict[str, Any]
    ):
        """Take profiling snapshot."""
        # Create stats
        stats = pstats.Stats(self.profiler)
        
        # Take memory snapshot if enabled
        memory_snapshot = None
        if self.config.trace_malloc:
            memory_snapshot = tracemalloc.take_snapshot()
        
        # Store snapshot
        self.snapshots.append(ProfileSnapshot(
            timestamp=datetime.now(),
            stats=stats,
            memory_snapshot=memory_snapshot,
            context=context
        ))
        
        # Trim old snapshots
        while len(self.snapshots) > self.config.max_snapshots:
            self.snapshots.pop(0)
    
    async def _save_profile(
        self,
        name: str,
        profile_type: str
    ):
        """Save profiling results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self.config.profile_dir / f"{name}_{profile_type}_{timestamp}"
        
        # Save stats
        stats = pstats.Stats(self.profiler)
        stats.dump_stats(f"{base_path}.prof")
        
        # Save readable stats
        with open(f"{base_path}.txt", "w") as f:
            stream = io.StringIO()
            stats.stream = stream
            stats.sort_stats("cumulative")
            stats.print_stats()
            f.write(stream.getvalue())
        
        # Save memory snapshot if enabled
        if self.config.trace_malloc:
            snapshot = tracemalloc.take_snapshot()
            with open(f"{base_path}_memory.txt", "w") as f:
                stats = snapshot.statistics("lineno")
                for stat in stats[:self.config.max_frames]:
                    frame = stat.traceback[0]
                    filename = frame.filename
                    line = linecache.getline(filename, frame.lineno).strip()
                    f.write(f"{filename}:{frame.lineno}: {line}\n")
                    f.write(f"    Size: {stat.size:,} bytes\n")
                    f.write(f"    Count: {stat.count:,} objects\n\n")
        
        # Generate flamegraph if enabled
        if self.config.flamegraph_enabled:
            await self._create_flamegraph(stats, base_path)
    
    async def _create_flamegraph(
        self,
        stats: pstats.Stats,
        base_path: Path
    ):
        """Create flame graph visualization."""
        # Convert stats to frame data
        frames = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if (
                not self.config.include_system_frames and
                any(p.startswith(("<", "[")) for p in func)
            ):
                continue
            
            frames.append({
                "function": f"{func[2]}:{func[1]}",
                "file": func[0],
                "calls": cc,
                "time": ct,
                "callers": len(callers)
            })
        
        # Create flame graph
        fig = go.Figure()
        
        df = pd.DataFrame(frames)
        df = df.sort_values("time", ascending=True)
        
        # Add bars
        fig.add_trace(go.Bar(
            x=df["time"],
            y=df["function"],
            orientation="h",
            text=df["calls"].map(lambda x: f"{x:,} calls"),
            hovertemplate=(
                "Function: %{y}<br>"
                "Time: %{x:.3f}s<br>"
                "%{text}<br>"
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title="CPU Time by Function",
            xaxis_title="Cumulative Time (seconds)",
            yaxis_title="Function",
            showlegend=False,
            height=max(600, len(frames) * 20)
        )
        
        fig.write_html(f"{base_path}_flame.html")
    
    async def analyze_snapshots(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Analyze profiling snapshots."""
        # Filter snapshots by time range
        snapshots = self.snapshots
        if start_time:
            snapshots = [s for s in snapshots if s.timestamp >= start_time]
        if end_time:
            snapshots = [s for s in snapshots if s.timestamp <= end_time]
        
        if not snapshots:
            return {"status": "no_data"}
        
        # Analyze CPU profiles
        cpu_stats = {
            "total_time": sum(
                sum(stat[2] for stat in s.stats.stats.values())
                for s in snapshots
            ),
            "function_calls": sum(
                sum(stat[0] for stat in s.stats.stats.values())
                for s in snapshots
            ),
            "top_functions": self._analyze_top_functions(snapshots),
            "call_patterns": self._analyze_call_patterns(snapshots)
        }
        
        # Analyze memory usage
        memory_stats = {}
        if self.config.trace_malloc:
            memory_stats = self._analyze_memory_usage(snapshots)
        
        return {
            "status": "success",
            "snapshot_count": len(snapshots),
            "time_range": {
                "start": snapshots[0].timestamp.isoformat(),
                "end": snapshots[-1].timestamp.isoformat()
            },
            "cpu_stats": cpu_stats,
            "memory_stats": memory_stats
        }
    
    def _analyze_top_functions(
        self,
        snapshots: List[ProfileSnapshot]
    ) -> List[Dict[str, Any]]:
        """Analyze most time-consuming functions."""
        # Aggregate function stats
        function_stats = {}
        for snapshot in snapshots:
            for func, (cc, nc, tt, ct, callers) in snapshot.stats.stats.items():
                if func not in function_stats:
                    function_stats[func] = {
                        "calls": 0,
                        "total_time": 0,
                        "cumulative_time": 0,
                        "callers": set()
                    }
                
                stats = function_stats[func]
                stats["calls"] += cc
                stats["total_time"] += tt
                stats["cumulative_time"] += ct
                stats["callers"].update(callers.keys())
        
        # Sort by cumulative time
        top_functions = sorted(
            (
                {
                    "function": f"{func[2]}:{func[1]}",
                    "file": func[0],
                    "calls": stats["calls"],
                    "total_time": stats["total_time"],
                    "cumulative_time": stats["cumulative_time"],
                    "caller_count": len(stats["callers"])
                }
                for func, stats in function_stats.items()
            ),
            key=lambda x: x["cumulative_time"],
            reverse=True
        )
        
        return top_functions[:self.config.max_frames]
    
    def _analyze_call_patterns(
        self,
        snapshots: List[ProfileSnapshot]
    ) -> Dict[str, Any]:
        """Analyze function call patterns."""
        # Build call graph
        call_graph = {}
        for snapshot in snapshots:
            for func, (cc, nc, tt, ct, callers) in snapshot.stats.stats.items():
                func_name = f"{func[2]}:{func[1]}"
                if func_name not in call_graph:
                    call_graph[func_name] = {
                        "callers": {},
                        "callees": {},
                        "total_calls": 0
                    }
                
                node = call_graph[func_name]
                node["total_calls"] += cc
                
                # Record callers
                for caller in callers:
                    caller_name = f"{caller[2]}:{caller[1]}"
                    node["callers"][caller_name] = (
                        node["callers"].get(caller_name, 0) +
                        callers[caller][0]
                    )
                    
                    # Add reverse edge
                    if caller_name not in call_graph:
                        call_graph[caller_name] = {
                            "callers": {},
                            "callees": {},
                            "total_calls": 0
                        }
                    call_graph[caller_name]["callees"][func_name] = (
                        call_graph[caller_name]["callees"].get(func_name, 0) +
                        callers[caller][0]
                    )
        
        # Analyze patterns
        patterns = {
            "hot_paths": self._find_hot_paths(call_graph),
            "leaf_functions": self._find_leaf_functions(call_graph),
            "root_functions": self._find_root_functions(call_graph),
            "cycles": self._find_cycles(call_graph)
        }
        
        return patterns
    
    def _find_hot_paths(
        self,
        call_graph: Dict[str, Dict[str, Any]]
    ) -> List[List[str]]:
        """Find high-traffic call paths."""
        paths = []
        visited = set()
        
        def dfs(node: str, path: List[str], total_calls: int):
            if len(path) >= self.config.max_frames:
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            # Check if path is "hot"
            if len(path) > 1 and total_calls >= 1000:  # Arbitrary threshold
                paths.append(path.copy())
            
            # Explore callees
            for callee, calls in call_graph[node]["callees"].items():
                dfs(callee, path.copy(), min(total_calls, calls))
        
        # Start from root functions
        roots = self._find_root_functions(call_graph)
        for root in roots:
            dfs(root, [], call_graph[root]["total_calls"])
        
        return sorted(
            paths,
            key=lambda p: min(
                call_graph[n]["total_calls"] for n in p
            ),
            reverse=True
        )[:10]  # Top 10 paths
    
    def _find_leaf_functions(
        self,
        call_graph: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Find functions that don't call others."""
        return [
            name for name, data in call_graph.items()
            if not data["callees"]
        ]
    
    def _find_root_functions(
        self,
        call_graph: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Find functions not called by others."""
        return [
            name for name, data in call_graph.items()
            if not data["callers"]
        ]
    
    def _find_cycles(
        self,
        call_graph: Dict[str, Dict[str, Any]]
    ) -> List[List[str]]:
        """Find recursive call cycles."""
        cycles = []
        visited = set()
        path = []
        
        def dfs(node: str):
            if node in path:
                cycle = path[path.index(node):]
                if tuple(cycle) not in visited:
                    cycles.append(cycle)
                    visited.add(tuple(cycle))
                return
            
            path.append(node)
            for callee in call_graph[node]["callees"]:
                dfs(callee)
            path.pop()
        
        for node in call_graph:
            dfs(node)
        
        return cycles
    
    def _analyze_memory_usage(
        self,
        snapshots: List[ProfileSnapshot]
    ) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not snapshots[0].memory_snapshot:
            return {}
        
        memory_stats = {
            "peak_size": 0,
            "peak_count": 0,
            "growth_rate": 0,
            "top_allocators": []
        }
        
        # Analyze each snapshot
        sizes = []
        counts = []
        for snapshot in snapshots:
            if not snapshot.memory_snapshot:
                continue
            
            stats = snapshot.memory_snapshot.statistics("lineno")
            total_size = sum(stat.size for stat in stats)
            total_count = sum(stat.count for stat in stats)
            
            sizes.append(total_size)
            counts.append(total_count)
            
            memory_stats["peak_size"] = max(
                memory_stats["peak_size"],
                total_size
            )
            memory_stats["peak_count"] = max(
                memory_stats["peak_count"],
                total_count
            )
        
        # Calculate growth rate
        if len(sizes) > 1:
            time_delta = (
                snapshots[-1].timestamp -
                snapshots[0].timestamp
            ).total_seconds()
            size_delta = sizes[-1] - sizes[0]
            
            memory_stats["growth_rate"] = size_delta / time_delta
        
        # Find top allocators
        if snapshots:
            latest = snapshots[-1].memory_snapshot
            stats = latest.statistics("lineno")
            memory_stats["top_allocators"] = [
                {
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size": stat.size,
                    "count": stat.count
                }
                for stat in stats[:self.config.max_frames]
            ]
        
        return memory_stats

def create_profiling_hook(
    config: Optional[ProfilingConfig] = None
) -> ProfilingHook:
    """Create profiling hook."""
    return ProfilingHook(config)

if __name__ == "__main__":
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
        
        # Add profiling hook
        profiling = create_profiling_hook()
        monitor.add_hook(profiling)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Run scheduler
        await scheduler.start()
        
        try:
            while True:
                # Analyze profiles
                analysis = await profiling.analyze_snapshots()
                if analysis["status"] == "success":
                    print("\nProfile Analysis:")
                    for func in analysis["cpu_stats"]["top_functions"][:5]:
                        print(
                            f"\n{func['function']}:"
                            f"\n  Calls: {func['calls']:,}"
                            f"\n  Time: {func['cumulative_time']:.3f}s"
                        )
                
                await asyncio.sleep(60)
        except KeyboardInterrupt:
            await scheduler.stop()
    
    asyncio.run(main())
