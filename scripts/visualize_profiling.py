#!/usr/bin/env python3
"""Visualize profiling results from performance analysis."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProfilingVisualizer:
    """Visualize and analyze profiling results."""
    
    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir
        self.profiles = self._load_profiles()
    
    def _load_profiles(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all profiling results."""
        profiles = {}
        
        for profile_file in self.profile_dir.glob("*.json"):
            name = profile_file.stem.split("_")[0]  # Get base function name
            
            try:
                data = json.loads(profile_file.read_text())
                if name not in profiles:
                    profiles[name] = []
                profiles[name].append(data)
            except json.JSONDecodeError:
                logger.warning(f"Failed to load profile: {profile_file}")
        
        return profiles

    def create_timeline_view(self) -> go.Figure:
        """Create timeline view of function execution times."""
        fig = go.Figure()
        
        for name, results in self.profiles.items():
            # Sort by timestamp
            sorted_results = sorted(
                results,
                key=lambda x: datetime.strptime(
                    x.get("timestamp", "0"), "%Y%m%d_%H%M%S"
                )
            )
            
            times = [r["duration_seconds"] for r in sorted_results]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(times))),
                    y=times,
                    name=name,
                    mode="lines+markers"
                )
            )
        
        fig.update_layout(
            title="Function Execution Timeline",
            xaxis_title="Run Index",
            yaxis_title="Duration (seconds)",
            showlegend=True
        )
        
        return fig

    def create_memory_analysis(self) -> go.Figure:
        """Create memory usage analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Peak Memory Usage",
                "Memory Growth Over Time",
                "Memory Usage Distribution",
                "Memory vs Duration"
            )
        )
        
        for name, results in self.profiles.items():
            peak_memory = [r["peak_memory_mb"] for r in results]
            durations = [r["duration_seconds"] for r in results]
            
            # Peak memory bar chart
            fig.add_trace(
                go.Bar(
                    x=[name],
                    y=[np.mean(peak_memory)],
                    error_y=dict(
                        type="data",
                        array=[np.std(peak_memory)]
                    ),
                    name=name
                ),
                row=1, col=1
            )
            
            # Memory growth over time
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(peak_memory))),
                    y=peak_memory,
                    name=name,
                    mode="lines+markers"
                ),
                row=1, col=2
            )
            
            # Memory distribution
            fig.add_trace(
                go.Histogram(
                    x=peak_memory,
                    name=name,
                    nbinsx=20
                ),
                row=2, col=1
            )
            
            # Memory vs Duration scatter
            fig.add_trace(
                go.Scatter(
                    x=durations,
                    y=peak_memory,
                    name=name,
                    mode="markers",
                    marker=dict(size=8)
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Memory Usage Analysis",
            showlegend=True
        )
        
        return fig

    def create_function_profile(self) -> go.Figure:
        """Create detailed function profiling visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Function Call Distribution",
                "Top Time-Consuming Functions",
                "IO Operations",
                "CPU vs Memory Usage"
            )
        )
        
        for name, results in self.profiles.items():
            # Function call distribution
            durations = [r["duration_seconds"] for r in results]
            fig.add_trace(
                go.Box(
                    y=durations,
                    name=name,
                    boxpoints="all"
                ),
                row=1, col=1
            )
            
            # Top functions
            if results and "top_functions" in results[0]:
                top_funcs = results[0]["top_functions"]
                times = [float(f.split()[0]) for f in top_funcs if f.strip()][:5]
                names = [f.split()[1] for f in top_funcs if f.strip()][:5]
                
                fig.add_trace(
                    go.Bar(
                        x=names,
                        y=times,
                        name=name
                    ),
                    row=1, col=2
                )
            
            # IO operations
            if results and "io_operations" in results[0]:
                reads = [r["io_operations"]["read_bytes"] / 1024 for r in results]
                writes = [r["io_operations"]["write_bytes"] / 1024 for r in results]
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(reads))),
                        y=reads,
                        name=f"{name} reads",
                        mode="lines"
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(writes))),
                        y=writes,
                        name=f"{name} writes",
                        mode="lines"
                    ),
                    row=2, col=1
                )
            
            # CPU vs Memory
            fig.add_trace(
                go.Scatter(
                    x=[r["peak_memory_mb"] for r in results],
                    y=[r.get("cpu_percent", 0) for r in results],
                    name=name,
                    mode="markers",
                    text=[f"Duration: {r['duration_seconds']:.2f}s" for r in results],
                    marker=dict(
                        size=10,
                        sizemode="area",
                        sizeref=2.*max(durations)/(40.**2),
                        sizemin=4
                    )
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=1000,
            title_text="Function Profile Analysis",
            showlegend=True
        )
        
        return fig

    def create_bottleneck_analysis(self) -> go.Figure:
        """Create bottleneck analysis visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Performance Bottlenecks",
                "Resource Usage Patterns",
                "Scaling Analysis",
                "Optimization Opportunities"
            )
        )
        
        for name, results in self.profiles.items():
            # Performance bottlenecks
            if results and "top_functions" in results[0]:
                top_funcs = results[0]["top_functions"]
                times = [float(f.split()[0]) for f in top_funcs if f.strip()][:5]
                names = [f.split()[1] for f in top_funcs if f.strip()][:5]
                
                total_time = sum(times)
                percentages = [t/total_time * 100 for t in times]
                
                fig.add_trace(
                    go.Bar(
                        x=names,
                        y=percentages,
                        name=name
                    ),
                    row=1, col=1
                )
            
            # Resource usage patterns
            memory_usage = [r["peak_memory_mb"] for r in results]
            cpu_usage = [r.get("cpu_percent", 0) for r in results]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(memory_usage))),
                    y=memory_usage,
                    name=f"{name} memory",
                    mode="lines"
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(cpu_usage))),
                    y=cpu_usage,
                    name=f"{name} CPU",
                    mode="lines"
                ),
                row=1, col=2
            )
            
            # Scaling analysis
            durations = [r["duration_seconds"] for r in results]
            data_sizes = [i * 1000 for i in range(len(durations))]  # Example sizes
            
            fig.add_trace(
                go.Scatter(
                    x=data_sizes,
                    y=durations,
                    name=name,
                    mode="markers+lines"
                ),
                row=2, col=1
            )
            
            # Optimization opportunities
            if results and "top_functions" in results[0]:
                top_funcs = results[0]["top_functions"]
                times = [float(f.split()[0]) for f in top_funcs if f.strip()][:5]
                names = [f.split()[1] for f in top_funcs if f.strip()][:5]
                
                optimization_score = [
                    t * (m/1024) for t, m in zip(times, memory_usage[:5])
                ]
                
                fig.add_trace(
                    go.Bar(
                        x=names,
                        y=optimization_score,
                        name=f"{name} optimization priority"
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=1000,
            title_text="Bottleneck Analysis",
            showlegend=True
        )
        
        return fig

    def generate_report(self) -> str:
        """Generate profiling analysis report."""
        lines = ["# Performance Profiling Report", ""]
        
        # Overall statistics
        total_runs = sum(len(results) for results in self.profiles.values())
        lines.extend([
            "## Overall Statistics",
            f"- Total Profile Runs: {total_runs}",
            f"- Functions Analyzed: {len(self.profiles)}",
            ""
        ])
        
        # Function summaries
        lines.append("## Function Performance Summary")
        
        for name, results in self.profiles.items():
            durations = [r["duration_seconds"] for r in results]
            memory_usage = [r["peak_memory_mb"] for r in results]
            
            lines.extend([
                f"### {name}",
                "#### Execution Time",
                f"- Mean: {np.mean(durations):.3f}s",
                f"- Std Dev: {np.std(durations):.3f}s",
                f"- Min: {min(durations):.3f}s",
                f"- Max: {max(durations):.3f}s",
                "",
                "#### Memory Usage",
                f"- Mean: {np.mean(memory_usage):.1f}MB",
                f"- Peak: {max(memory_usage):.1f}MB",
                ""
            ])
            
            # Top time-consuming functions
            if results and "top_functions" in results[0]:
                lines.extend([
                    "#### Top Time-Consuming Functions",
                    "```"
                ])
                for func in results[0]["top_functions"][:5]:
                    lines.append(func)
                lines.extend(["```", ""])
        
        # Bottleneck analysis
        lines.extend(["## Performance Bottlenecks", ""])
        
        for name, results in self.profiles.items():
            if results and "top_functions" in results[0]:
                total_time = sum(
                    float(f.split()[0])
                    for f in results[0]["top_functions"]
                    if f.strip()
                )
                
                lines.extend([
                    f"### {name} Bottlenecks",
                    f"- Total Analysis Time: {total_time:.2f}s",
                    "- Critical Paths:",
                ])
                
                for func in results[0]["top_functions"][:3]:
                    time = float(func.split()[0])
                    percent = (time / total_time) * 100
                    lines.append(f"  - {func.split()[1]}: {percent:.1f}%")
                lines.append("")
        
        # Optimization recommendations
        lines.extend(["## Optimization Recommendations", ""])
        
        for name, results in self.profiles.items():
            lines.extend([f"### {name}", ""])
            
            # Memory recommendations
            memory_usage = [r["peak_memory_mb"] for r in results]
            if max(memory_usage) > 1024:  # More than 1GB
                lines.append(
                    "- **High Memory Usage**: Consider implementing memory-efficient "
                    "algorithms or streaming processing"
                )
            
            # Time recommendations
            durations = [r["duration_seconds"] for r in results]
            if max(durations) > 10:  # More than 10 seconds
                lines.append(
                    "- **Long Execution Time**: Consider parallelization or "
                    "algorithmic optimizations"
                )
            
            # IO recommendations
            if results and "io_operations" in results[0]:
                total_io = (
                    results[0]["io_operations"]["read_bytes"] +
                    results[0]["io_operations"]["write_bytes"]
                ) / 1024 / 1024  # MB
                
                if total_io > 100:  # More than 100MB
                    lines.append(
                        "- **High I/O Usage**: Consider caching or reducing "
                        "data transfer"
                    )
            
            lines.append("")
        
        return "\n".join(lines)

def main() -> int:
    """Main entry point."""
    try:
        profile_dir = Path("profiling_results")
        if not profile_dir.exists():
            logger.error("No profiling results directory found")
            return 1
        
        visualizer = ProfilingVisualizer(profile_dir)
        
        # Generate visualizations
        timeline = visualizer.create_timeline_view()
        timeline.write_html(profile_dir / "timeline_view.html")
        
        memory = visualizer.create_memory_analysis()
        memory.write_html(profile_dir / "memory_analysis.html")
        
        function_profile = visualizer.create_function_profile()
        function_profile.write_html(profile_dir / "function_profile.html")
        
        bottleneck = visualizer.create_bottleneck_analysis()
        bottleneck.write_html(profile_dir / "bottleneck_analysis.html")
        
        # Generate report
        report = visualizer.generate_report()
        report_file = profile_dir / "profiling_report.md"
        report_file.write_text(report)
        
        logger.info(f"Profiling visualization complete. Reports written to {profile_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during profiling visualization: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
