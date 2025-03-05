"""Profiling tools for prediction performance analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import time
import cProfile
import pstats
import io
import sys
import tracemalloc
from line_profiler import LineProfiler
import memory_profiler
import plotly.graph_objects as go
from functools import wraps
import inspect

from .prediction_performance import PredictionPerformance, PerformanceConfig

logger = logging.getLogger(__name__)

@dataclass
class ProfilingConfig:
    """Configuration for profiling tools."""
    line_profiling: bool = True
    memory_profiling: bool = True
    call_graph: bool = True
    trace_allocations: bool = True
    profile_interval: float = 60.0  # seconds
    max_depth: int = 10
    detailed_memory: bool = True
    output_path: Optional[Path] = None

class PredictionProfiler:
    """Profiling tools for prediction analysis."""
    
    def __init__(
        self,
        performance: PredictionPerformance,
        config: ProfilingConfig
    ):
        self.performance = performance
        self.config = config
        self.profiler = cProfile.Profile()
        self.line_profiler = LineProfiler()
        self.current_profile: Optional[pstats.Stats] = None
        self.memory_snapshots: List[tracemalloc.Snapshot] = []
        self.function_stats: Dict[str, Dict[str, Any]] = {}
        self.last_profile = 0.0
    
    def profile_function(
        self,
        func: Callable
    ) -> Callable:
        """Decorator to profile a function."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Line profiling
            if self.config.line_profiling:
                self.line_profiler.add_function(func)
                wrapper = self.line_profiler(func)
            
            # Memory profiling
            if self.config.memory_profiling:
                wrapper = memory_profiler.profile(func)
            
            # CPU profiling
            self.profiler.enable()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable()
                if time.time() - self.last_profile >= self.config.profile_interval:
                    self._process_profile()
        
        return wrapper
    
    def start_profiling(self):
        """Start profiling session."""
        self.profiler.enable()
        if self.config.trace_allocations:
            tracemalloc.start()
        logger.info("Started profiling")
    
    def stop_profiling(self):
        """Stop profiling session."""
        self.profiler.disable()
        if tracemalloc.is_tracing():
            self.memory_snapshots.append(tracemalloc.take_snapshot())
            tracemalloc.stop()
        self._process_profile()
        logger.info("Stopped profiling")
    
    def profile_section(
        self,
        section_name: str
    ) -> Callable:
        """Context manager for profiling code sections."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
                
                self.profiler.enable()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.profiler.disable()
                    end_time = time.time()
                    end_memory = tracemalloc.take_snapshot() if tracemalloc.is_tracing() else None
                    
                    self._add_section_stats(
                        section_name,
                        start_time,
                        end_time,
                        start_memory,
                        end_memory
                    )
            
            return wrapper
        return decorator
    
    def create_profile_report(self) -> Dict[str, Any]:
        """Create comprehensive profiling report."""
        report = {
            "execution": self._get_execution_stats(),
            "memory": self._get_memory_stats(),
            "functions": self._get_function_stats(),
            "bottlenecks": self._identify_bottlenecks(),
            "recommendations": self._generate_recommendations()
        }
        
        if self.config.output_path:
            self._save_report(report)
        
        return report
    
    def visualize_profile(self) -> go.Figure:
        """Create profiling visualization."""
        fig = go.Figure()
        
        # Add execution time distribution
        exec_times = [
            stats["execution_time"]
            for stats in self.function_stats.values()
        ]
        if exec_times:
            fig.add_trace(
                go.Box(
                    y=exec_times,
                    name="Execution Time",
                    boxpoints="all"
                )
            )
        
        # Add memory usage
        if self.config.memory_profiling:
            memory_usage = [
                stats.get("memory_delta", 0)
                for stats in self.function_stats.values()
            ]
            if memory_usage:
                fig.add_trace(
                    go.Box(
                        y=memory_usage,
                        name="Memory Usage",
                        boxpoints="all"
                    )
                )
        
        # Add call counts
        call_counts = [
            stats["call_count"]
            for stats in self.function_stats.values()
        ]
        if call_counts:
            fig.add_trace(
                go.Bar(
                    y=call_counts,
                    name="Call Count"
                )
            )
        
        fig.update_layout(
            title="Function Profile Distribution",
            showlegend=True,
            boxmode="group"
        )
        
        return fig
    
    def _process_profile(self):
        """Process current profile data."""
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        stats.sort_stats("cumulative")
        stats.print_stats()
        
        self.current_profile = stats
        self.last_profile = time.time()
        
        # Process function statistics
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            func_name = f"{func[2]}:{func[0]}({func[1]})"
            self.function_stats[func_name] = {
                "call_count": cc,
                "execution_time": tt,
                "cumulative_time": ct,
                "callers": len(callers)
            }
    
    def _add_section_stats(
        self,
        section_name: str,
        start_time: float,
        end_time: float,
        start_memory: Optional[tracemalloc.Snapshot],
        end_memory: Optional[tracemalloc.Snapshot]
    ):
        """Add profiling stats for code section."""
        stats = {
            "execution_time": end_time - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if start_memory and end_memory:
            memory_diff = end_memory.compare_to(start_memory, "lineno")
            stats["memory_delta"] = sum(
                stat.size_diff for stat in memory_diff
            )
            if self.config.detailed_memory:
                stats["memory_details"] = [
                    {
                        "file": stat.traceback[0].filename,
                        "line": stat.traceback[0].lineno,
                        "size_diff": stat.size_diff
                    }
                    for stat in memory_diff
                ]
        
        self.function_stats[section_name] = stats
    
    def _get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.current_profile:
            return {}
        
        total_time = self.current_profile.total_tt
        return {
            "total_time": total_time,
            "function_count": len(self.function_stats),
            "avg_time_per_call": total_time / max(
                sum(s["call_count"] for s in self.function_stats.values()),
                1
            )
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        if not self.memory_snapshots:
            return {}
        
        latest = self.memory_snapshots[-1]
        statistics = latest.statistics("traceback")
        
        return {
            "total_allocated": sum(stat.size for stat in statistics),
            "peak_allocated": max(stat.size for stat in statistics),
            "allocation_count": len(statistics),
            "top_allocations": [
                {
                    "size": stat.size,
                    "count": stat.count,
                    "traceback": [
                        f"{frame.filename}:{frame.lineno}"
                        for frame in stat.traceback
                    ]
                }
                for stat in sorted(
                    statistics,
                    key=lambda x: x.size,
                    reverse=True
                )[:10]
            ]
        }
    
    def _get_function_stats(self) -> Dict[str, Any]:
        """Get function-level statistics."""
        return {
            name: {
                "calls": stats["call_count"],
                "time": stats["execution_time"],
                "avg_time": stats["execution_time"] / max(stats["call_count"], 1),
                "memory": stats.get("memory_delta", 0)
            }
            for name, stats in self.function_stats.items()
        }
    
    def _identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Time bottlenecks
        time_threshold = np.mean([
            s["execution_time"] for s in self.function_stats.values()
        ]) * 2
        
        for name, stats in self.function_stats.items():
            if stats["execution_time"] > time_threshold:
                bottlenecks.append({
                    "type": "execution_time",
                    "function": name,
                    "value": stats["execution_time"],
                    "threshold": time_threshold
                })
        
        # Memory bottlenecks
        if self.memory_snapshots:
            memory_stats = self._get_memory_stats()
            total_memory = memory_stats["total_allocated"]
            for alloc in memory_stats["top_allocations"]:
                if alloc["size"] > total_memory * 0.1:  # 10% of total
                    bottlenecks.append({
                        "type": "memory_usage",
                        "location": alloc["traceback"][0],
                        "size": alloc["size"],
                        "percentage": alloc["size"] / total_memory
                    })
        
        return bottlenecks
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        bottlenecks = self._identify_bottlenecks()
        
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "execution_time":
                recommendations.append(
                    f"Optimize function {bottleneck['function']} "
                    f"which takes {bottleneck['value']:.2f}s to execute"
                )
            elif bottleneck["type"] == "memory_usage":
                recommendations.append(
                    f"Review memory allocation at {bottleneck['location']} "
                    f"using {bottleneck['size'] / 1024 / 1024:.1f}MB "
                    f"({bottleneck['percentage'] * 100:.1f}% of total)"
                )
        
        return recommendations
    
    def _save_report(self, report: Dict[str, Any]):
        """Save profiling report."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save main report
            report_file = output_path / "profile_report.json"
            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)
            
            # Save call graph if enabled
            if self.config.call_graph and self.current_profile:
                graph_file = output_path / "call_graph.dot"
                self.current_profile.dump_stats(str(graph_file))
            
            logger.info(f"Saved profiling report to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

def create_prediction_profiler(
    performance: PredictionPerformance,
    output_path: Optional[Path] = None
) -> PredictionProfiler:
    """Create prediction profiler."""
    config = ProfilingConfig(output_path=output_path)
    return PredictionProfiler(performance, config)

if __name__ == "__main__":
    # Example usage
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    async def main():
        # Create components
        easing = create_easing_functions()
        metrics = create_easing_metrics(easing)
        stats = create_easing_statistics(metrics)
        predictor = create_easing_predictor(stats)
        visualizer = create_prediction_visualizer(predictor)
        controls = create_interactive_controls(visualizer)
        realtime = create_realtime_prediction(controls)
        performance = create_prediction_performance(realtime)
        profiler = create_prediction_profiler(
            performance,
            output_path=Path("profiling_data")
        )
        
        # Start profiling
        profiler.start_profiling()
        
        # Run some predictions
        @profiler.profile_section("prediction_loop")
        async def run_predictions():
            await realtime.start_streaming()
            for _ in range(100):
                await realtime.send_update({
                    "value": np.random.random(),
                    "timestamp": datetime.now().isoformat()
                })
                await asyncio.sleep(0.1)
            await realtime.stop_streaming()
        
        await run_predictions()
        
        # Stop profiling
        profiler.stop_profiling()
        
        # Generate report
        report = profiler.create_profile_report()
        print(json.dumps(report, indent=2))
        
        # Create visualization
        fig = profiler.visualize_profile()
        fig.write_html("profile_visualization.html")
    
    asyncio.run(main())
