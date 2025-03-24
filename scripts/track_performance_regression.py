#!/usr/bin/env python3
"""Track and analyze performance regressions over time."""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import numpy as np
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RegressionEvent:
    """Represents a detected regression event."""
    metric: str
    old_value: float
    new_value: float
    change_percent: float
    date: datetime
    commit: str
    severity: str

    def __str__(self) -> str:
        return (
            f"{self.severity.upper()}: {self.metric} regression detected\n"
            f"  Change: {self.old_value:.2f} → {self.new_value:.2f} "
            f"({self.change_percent:+.1f}%)\n"
            f"  Date: {self.date.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Commit: {self.commit[:8]}"
        )

class RegressionAnalyzer:
    """Analyzes performance data for regressions."""
    
    def __init__(self, results_dir: Path, budget_file: Path):
        self.results_dir = results_dir
        with open(budget_file) as f:
            self.budget = json.load(f)
        self.regressions: List[RegressionEvent] = []
        self.historical_data: Dict[str, List[Tuple[datetime, float]]] = {}

    def load_historical_data(self) -> None:
        """Load and process historical benchmark data."""
        metrics_of_interest = [
            "suggestions_per_second",
            "peak_memory_mb",
            "specificity_ratio",
            "scaling_factor"
        ]
        
        for file in sorted(self.results_dir.glob("type_suggestion_benchmarks_*.json")):
            try:
                data = json.loads(file.read_text())
                timestamp = datetime.fromtimestamp(data.get("timestamp", 0))
                commit = file.name.split("_")[-1].replace(".json", "")
                
                for metric in metrics_of_interest:
                    if metric not in self.historical_data:
                        self.historical_data[metric] = []
                    
                    value = self._extract_metric_value(data, metric)
                    if value is not None:
                        self.historical_data[metric].append((timestamp, value))
            
            except Exception as e:
                logger.warning(f"Error processing {file}: {e}")

    def _extract_metric_value(self, data: Dict[str, Any], metric: str) -> Optional[float]:
        """Extract metric value from benchmark data."""
        if metric == "suggestions_per_second":
            return (data.get("medium", {})
                   .get("data", {})
                   .get("stats", {})
                   .get("suggestions_per_second"))
        elif metric == "peak_memory_mb":
            return (data.get("memory_usage", {})
                   .get("data", {})
                   .get("max_memory_mb"))
        elif metric == "specificity_ratio":
            return (data.get("suggestion_quality", {})
                   .get("data", {})
                   .get("specificity_ratio"))
        elif metric == "scaling_factor":
            sizes = ["small", "medium", "large"]
            times = []
            for size in sizes:
                if size in data:
                    times.append(data[size]["data"]["duration"])
            if len(times) >= 2:
                return times[-1] / times[0]
        return None

    def detect_regressions(self, window_size: int = 7) -> None:
        """Detect performance regressions using sliding window analysis."""
        for metric, data in self.historical_data.items():
            if len(data) < window_size + 1:
                continue
            
            # Calculate baseline statistics
            baseline_values = [v for _, v in data[-window_size-1:-1]]
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            # Get latest value
            latest_date, latest_value = data[-1]
            
            # Calculate z-score
            z_score = (latest_value - baseline_mean) / (baseline_std if baseline_std > 0 else 1)
            
            # Calculate percent change
            percent_change = ((latest_value - baseline_mean) / baseline_mean) * 100
            
            # Check against thresholds
            thresholds = self.budget["thresholds"]["regression_failure"]
            warning_thresholds = self.budget["thresholds"]["regression_notification"]
            
            if self._is_regression(metric, percent_change, thresholds):
                self.regressions.append(RegressionEvent(
                    metric=metric,
                    old_value=baseline_mean,
                    new_value=latest_value,
                    change_percent=percent_change,
                    date=latest_date,
                    commit=self._get_latest_commit(),
                    severity="critical"
                ))
            elif self._is_regression(metric, percent_change, warning_thresholds):
                self.regressions.append(RegressionEvent(
                    metric=metric,
                    old_value=baseline_mean,
                    new_value=latest_value,
                    change_percent=percent_change,
                    date=latest_date,
                    commit=self._get_latest_commit(),
                    severity="warning"
                ))

    def _is_regression(self, metric: str, change: float, thresholds: Dict[str, float]) -> bool:
        """Determine if a change constitutes a regression."""
        if metric in ["suggestions_per_second", "specificity_ratio"]:
            return change < thresholds.get("speed", 0)
        else:
            return change > thresholds.get("memory", 0)

    def _get_latest_commit(self) -> str:
        """Get the latest commit hash from benchmark files."""
        latest_file = sorted(self.results_dir.glob("type_suggestion_benchmarks_*.json"))[-1]
        return latest_file.name.split("_")[-1].replace(".json", "")

    def generate_trend_visualization(self) -> None:
        """Generate visualization of performance trends."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Processing Speed", "Memory Usage", 
                          "Type Quality", "Scaling Factor")
        )
        
        metrics = [
            ("suggestions_per_second", 1, 1),
            ("peak_memory_mb", 1, 2),
            ("specificity_ratio", 2, 1),
            ("scaling_factor", 2, 2)
        ]
        
        for metric, row, col in metrics:
            if metric in self.historical_data:
                dates, values = zip(*self.historical_data[metric])
                
                # Add trend line
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        name=metric.replace("_", " ").title(),
                        mode='lines+markers'
                    ),
                    row=row, col=col
                )
                
                # Add regression points
                regression_dates = []
                regression_values = []
                for reg in self.regressions:
                    if reg.metric == metric:
                        regression_dates.append(reg.date)
                        regression_values.append(reg.new_value)
                
                if regression_dates:
                    fig.add_trace(
                        go.Scatter(
                            x=regression_dates,
                            y=regression_values,
                            mode='markers',
                            marker=dict(
                                symbol='x',
                                size=10,
                                color='red'
                            ),
                            name=f"{metric} regressions"
                        ),
                        row=row, col=col
                    )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="Performance Trends and Regressions"
        )
        
        output_file = self.results_dir / "regression_analysis.html"
        fig.write_html(str(output_file))

    def generate_report(self) -> str:
        """Generate markdown report of regression analysis."""
        sections = ["# Performance Regression Analysis\n"]
        
        if self.regressions:
            sections.append("## Detected Regressions\n")
            critical = [r for r in self.regressions if r.severity == "critical"]
            warnings = [r for r in self.regressions if r.severity == "warning"]
            
            if critical:
                sections.append("### Critical Regressions\n")
                for reg in critical:
                    sections.append(f"```\n{reg}\n```\n")
            
            if warnings:
                sections.append("### Warnings\n")
                for reg in warnings:
                    sections.append(f"```\n{reg}\n```\n")
        else:
            sections.append("✅ No regressions detected\n")
        
        # Add trend summary
        sections.append("## Trend Summary\n")
        for metric, data in self.historical_data.items():
            if len(data) >= 2:
                _, old_value = data[-2]
                _, new_value = data[-1]
                change = ((new_value - old_value) / old_value) * 100
                sections.append(
                    f"- {metric.replace('_', ' ').title()}: "
                    f"{old_value:.2f} → {new_value:.2f} ({change:+.1f}%)\n"
                )
        
        return "\n".join(sections)

def main() -> int:
    """Main entry point."""
    try:
        results_dir = Path("benchmark_results")
        if not results_dir.exists():
            logger.error("No benchmark results directory found")
            return 1

        analyzer = RegressionAnalyzer(
            results_dir,
            Path("performance-budget.json")
        )
        
        analyzer.load_historical_data()
        analyzer.detect_regressions()
        
        # Generate visualization
        analyzer.generate_trend_visualization()
        
        # Generate and save report
        report = analyzer.generate_report()
        report_file = results_dir / "regression_report.md"
        report_file.write_text(report)
        
        # Print report to console
        print(report)
        
        # Create GitHub annotations if in CI
        if "GITHUB_ACTIONS" in os.environ:
            for reg in analyzer.regressions:
                cmd = "error" if reg.severity == "critical" else "warning"
                print(f"::{cmd}::{reg}")
        
        # Exit with error if critical regressions found
        return 1 if any(r.severity == "critical" for r in analyzer.regressions) else 0
        
    except Exception as e:
        logger.error(f"Error running regression analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
