"""Visualizer for performance profiling results."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import pstats
import logging
from typing import Dict, Any, List, Optional
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

class ProfileVisualizer:
    """Generate visualizations from profiling data."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        sns.set_context("paper")
    
    def visualize_cpu_profile(self, stats_file: Path) -> Path:
        """Create CPU profile visualization."""
        stats = pstats.Stats(str(stats_file))
        
        # Extract data
        function_stats = []
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            function_stats.append({
                "function": f"{func[2]}:{func[0]}",
                "calls": cc,
                "cumtime": ct,
                "tottime": tt,
                "percall": ct/cc if cc > 0 else 0
            })
        
        df = pd.DataFrame(function_stats)
        df = df.sort_values("cumtime", ascending=False).head(20)
        
        # Create visualization
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Cumulative Time by Function",
                "Calls per Function",
                "Time per Call",
                "Time Distribution"
            )
        )
        
        # Cumulative time bar chart
        fig.add_trace(
            go.Bar(
                x=df["function"],
                y=df["cumtime"],
                name="Cumulative Time"
            ),
            row=1, col=1
        )
        
        # Calls bar chart
        fig.add_trace(
            go.Bar(
                x=df["function"],
                y=df["calls"],
                name="Call Count"
            ),
            row=1, col=2
        )
        
        # Time per call scatter
        fig.add_trace(
            go.Scatter(
                x=df["function"],
                y=df["percall"],
                mode="markers",
                name="Time per Call"
            ),
            row=2, col=1
        )
        
        # Time distribution histogram
        fig.add_trace(
            go.Histogram(
                x=df["tottime"],
                nbinsx=20,
                name="Time Distribution"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            width=1500,
            showlegend=True,
            title_text="CPU Profile Analysis"
        )
        
        output_path = self.output_dir / "cpu_profile.html"
        fig.write_html(str(output_path))
        return output_path
    
    def visualize_memory_profile(self, snapshots: List[Dict[str, Any]]) -> Path:
        """Create memory profile visualization."""
        timestamps = []
        total_memory = []
        allocation_sizes = []
        object_counts = []
        
        for snapshot in snapshots:
            timestamps.append(snapshot["timestamp"])
            total_memory.append(snapshot["total"] / (1024 * 1024))  # Convert to MB
            allocation_sizes.extend(
                [trace["size"] / 1024 for trace in snapshot["traces"]]
            )
            object_counts.append(len(snapshot["traces"]))
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Memory Usage Over Time",
                "Object Count Over Time",
                "Allocation Size Distribution",
                "Memory vs Objects"
            )
        )
        
        # Memory usage timeline
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=total_memory,
                name="Total Memory (MB)"
            ),
            row=1, col=1
        )
        
        # Object count timeline
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=object_counts,
                name="Object Count"
            ),
            row=1, col=2
        )
        
        # Allocation size histogram
        fig.add_trace(
            go.Histogram(
                x=allocation_sizes,
                nbinsx=50,
                name="Allocation Sizes (KB)"
            ),
            row=2, col=1
        )
        
        # Memory vs Objects scatter
        fig.add_trace(
            go.Scatter(
                x=object_counts,
                y=total_memory,
                mode="markers",
                name="Memory vs Objects"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            width=1500,
            showlegend=True,
            title_text="Memory Profile Analysis"
        )
        
        output_path = self.output_dir / "memory_profile.html"
        fig.write_html(str(output_path))
        return output_path
    
    def visualize_concurrency(self, concurrency_data: Dict[str, Any]) -> Path:
        """Create concurrency visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Active Requests Over Time",
                "Request Duration Distribution",
                "Concurrency Level Distribution",
                "Duration vs Concurrency"
            )
        )
        
        # Active requests timeline
        fig.add_trace(
            go.Scatter(
                x=concurrency_data["timestamps"],
                y=concurrency_data["active_requests"],
                name="Active Requests"
            ),
            row=1, col=1
        )
        
        # Duration histogram
        fig.add_trace(
            go.Histogram(
                x=concurrency_data["durations"],
                nbinsx=30,
                name="Request Durations"
            ),
            row=1, col=2
        )
        
        # Concurrency level histogram
        fig.add_trace(
            go.Histogram(
                x=concurrency_data["concurrency_levels"],
                nbinsx=20,
                name="Concurrency Levels"
            ),
            row=2, col=1
        )
        
        # Duration vs Concurrency scatter
        fig.add_trace(
            go.Scatter(
                x=concurrency_data["concurrency_levels"],
                y=concurrency_data["durations"],
                mode="markers",
                name="Duration vs Concurrency"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=1000,
            width=1500,
            showlegend=True,
            title_text="Concurrency Analysis"
        )
        
        output_path = self.output_dir / "concurrency_profile.html"
        fig.write_html(str(output_path))
        return output_path
    
    def generate_profile_report(self, profile_data: Dict[str, Any]) -> Path:
        """Generate comprehensive profile report."""
        # Combine all profile data
        cpu_profile = self.visualize_cpu_profile(profile_data["cpu_stats"])
        memory_profile = self.visualize_memory_profile(profile_data["memory_snapshots"])
        concurrency_profile = self.visualize_concurrency(profile_data["concurrency"])
        
        # Create report
        report = [
            "# Performance Profile Report",
            "\n## CPU Profile",
            f"![CPU Profile]({cpu_profile})",
            "\n## Memory Profile",
            f"![Memory Profile]({memory_profile})",
            "\n## Concurrency Profile",
            f"![Concurrency Profile]({concurrency_profile})",
            "\n## Profile Summary",
            "\n### CPU Hotspots",
            "```",
            self._format_cpu_hotspots(profile_data["cpu_stats"]),
            "```",
            "\n### Memory Patterns",
            "```",
            self._format_memory_patterns(profile_data["memory_snapshots"]),
            "```",
            "\n### Concurrency Analysis",
            "```",
            self._format_concurrency_analysis(profile_data["concurrency"]),
            "```"
        ]
        
        # Save report
        report_path = self.output_dir / "profile_report.md"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        return report_path
    
    def _format_cpu_hotspots(self, stats_file: Path) -> str:
        """Format CPU hotspots for report."""
        stats = pstats.Stats(str(stats_file))
        output = io.StringIO()
        stats.sort_stats("cumtime").print_stats(10)
        return output.getvalue()
    
    def _format_memory_patterns(self, snapshots: List[Dict[str, Any]]) -> str:
        """Format memory patterns for report."""
        patterns = []
        for i, snapshot in enumerate(snapshots[:-1]):
            next_snapshot = snapshots[i + 1]
            diff = next_snapshot["total"] - snapshot["total"]
            if abs(diff) > 1024 * 1024:  # 1MB change
                patterns.append(
                    f"Memory {'increased' if diff > 0 else 'decreased'} by "
                    f"{diff / (1024*1024):.1f}MB at {next_snapshot['timestamp']}"
                )
        return "\n".join(patterns)
    
    def _format_concurrency_analysis(self, concurrency: Dict[str, Any]) -> str:
        """Format concurrency analysis for report."""
        return (
            f"Maximum concurrency: {max(concurrency['concurrency_levels'])}\n"
            f"Average duration: {np.mean(concurrency['durations']):.3f}s\n"
            f"Duration P95: {np.percentile(concurrency['durations'], 95):.3f}s\n"
            f"Active requests peak: {max(concurrency['active_requests'])}"
        )

def create_profile_visualization(
    profile_dir: Path,
    output_dir: Optional[Path] = None
) -> Path:
    """Create profile visualizations from test results."""
    if output_dir is None:
        output_dir = profile_dir / "visualizations"
    
    visualizer = ProfileVisualizer(output_dir)
    
    # Load profile data
    profile_data = {
        "cpu_stats": profile_dir / "cpu_profile.prof",
        "memory_snapshots": json.loads(
            (profile_dir / "memory_snapshots.json").read_text()
        ),
        "concurrency": json.loads(
            (profile_dir / "concurrency.json").read_text()
        )
    }
    
    # Generate report
    return visualizer.generate_profile_report(profile_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize performance profiles")
    parser.add_argument("profile_dir", type=Path, help="Profile data directory")
    parser.add_argument(
        "--output", type=Path,
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    report_path = create_profile_visualization(args.profile_dir, args.output)
    print(f"Profile report generated: {report_path}")
