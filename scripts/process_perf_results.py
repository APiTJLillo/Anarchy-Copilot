#!/usr/bin/env python3
"""Process and aggregate performance test results."""
import argparse
import json
from pathlib import Path
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import statistics
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultsProcessor:
    """Process and analyze performance test results."""
    
    def __init__(self, results_dir: Path, output_format: str = "github"):
        self.results_dir = Path(results_dir)
        self.output_format = output_format
        self.results_cache: Dict[str, Any] = {}
    
    def process_results(self) -> Dict[str, Any]:
        """Process all test results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "tests": self._load_test_results(),
            "system": self._load_system_metrics(),
            "regressions": self._analyze_regressions(),
            "trends": self._analyze_trends()
        }
        
        # Cache results
        self.results_cache = summary
        return summary
    
    def _load_test_results(self) -> Dict[str, Any]:
        """Load and aggregate test results."""
        results = {}
        
        # Load all result files
        for result_file in self.results_dir.glob("results_*.json"):
            with open(result_file) as f:
                data = json.load(f)
                test_name = result_file.stem.replace("results_", "")
                results[test_name] = data
        
        # Aggregate metrics
        if not results:
            return {}
        
        aggregated = {
            "throughput": [],
            "latency_p95": [],
            "memory_usage": [],
            "execution_times": {}
        }
        
        for result in results.values():
            metrics = result.get("results", {})
            aggregated["throughput"].append(metrics.get("throughput", 0))
            aggregated["latency_p95"].append(metrics.get("latency_p95", 0))
            aggregated["memory_usage"].append(metrics.get("memory_usage", 0))
            
            # Aggregate execution times
            for name, times in metrics.get("execution_times", {}).items():
                if name not in aggregated["execution_times"]:
                    aggregated["execution_times"][name] = []
                aggregated["execution_times"][name].extend(times)
        
        # Calculate statistics
        return {
            "throughput": {
                "mean": statistics.mean(aggregated["throughput"]),
                "p95": statistics.quantiles(aggregated["throughput"], n=20)[18],
                "min": min(aggregated["throughput"]),
                "max": max(aggregated["throughput"])
            },
            "latency": {
                "mean": statistics.mean(aggregated["latency_p95"]),
                "p95": statistics.quantiles(aggregated["latency_p95"], n=20)[18],
                "min": min(aggregated["latency_p95"]),
                "max": max(aggregated["latency_p95"])
            },
            "memory": {
                "mean": statistics.mean(aggregated["memory_usage"]),
                "peak": max(aggregated["memory_usage"]),
                "min": min(aggregated["memory_usage"])
            },
            "execution_times": {
                name: {
                    "mean": statistics.mean(times),
                    "p95": statistics.quantiles(times, n=20)[18]
                }
                for name, times in aggregated["execution_times"].items()
            }
        }
    
    def _load_system_metrics(self) -> Dict[str, Any]:
        """Load and process system metrics."""
        try:
            # Load metrics summary
            metrics_file = self.results_dir / "metrics_summary.txt"
            if not metrics_file.exists():
                return {}
            
            with open(metrics_file) as f:
                content = f.read()
            
            # Extract metrics using regex
            metrics = {}
            patterns = {
                "cpu_peak": r"Peak CPU.*?(\d+\.?\d*)%",
                "memory_peak": r"Peak Memory.*?(\d+\.?\d*)MB",
                "io_wait_avg": r"IO Wait.*?Average: (\d+\.?\d*)%"
            }
            
            for name, pattern in patterns.items():
                if match := re.search(pattern, content):
                    metrics[name] = float(match.group(1))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Failed to load system metrics: {e}")
            return {}
    
    def _analyze_regressions(self) -> Dict[str, Any]:
        """Analyze performance regressions."""
        regressions = {}
        
        # Load all regression data
        for result_file in self.results_dir.glob("results_*.json"):
            with open(result_file) as f:
                data = json.load(f)
                if reg_data := data.get("regressions"):
                    regressions.update(reg_data)
        
        # Analyze regression severity
        severity = {
            "critical": [],
            "major": [],
            "minor": []
        }
        
        for metric, details in regressions.items():
            degradation = details.get("degradation", 0)
            if degradation > 25:
                severity["critical"].append(metric)
            elif degradation > 15:
                severity["major"].append(metric)
            else:
                severity["minor"].append(metric)
        
        return {
            "details": regressions,
            "severity": severity,
            "total_count": len(regressions)
        }
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends."""
        try:
            results_files = sorted(self.results_dir.glob("results_*.json"))
            if len(results_files) < 2:
                return {}
            
            # Extract metrics over time
            metrics = {
                "throughput": [],
                "latency": [],
                "memory": []
            }
            
            for file in results_files:
                with open(file) as f:
                    data = json.load(f)
                    results = data.get("results", {})
                    metrics["throughput"].append(results.get("throughput", 0))
                    metrics["latency"].append(results.get("latency_p95", 0))
                    metrics["memory"].append(results.get("memory_usage", 0))
            
            # Calculate trends
            return {
                metric: {
                    "direction": "improving" if values[-1] > values[0] else "degrading",
                    "change_pct": ((values[-1] - values[0]) / values[0]) * 100
                }
                for metric, values in metrics.items()
                if values
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze trends: {e}")
            return {}
    
    def generate_report(self, summary: Optional[Dict[str, Any]] = None) -> str:
        """Generate formatted report."""
        if summary is None:
            summary = self.results_cache or self.process_results()
        
        if self.output_format == "github":
            return self._generate_github_report(summary)
        else:
            return self._generate_text_report(summary)
    
    def _generate_github_report(self, summary: Dict[str, Any]) -> str:
        """Generate GitHub-flavored markdown report."""
        lines = [
            "## Performance Test Results",
            "",
            "### Summary",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        
        # Add test results
        test_results = summary.get("tests", {})
        if test_results:
            lines.extend([
                f"| Throughput | {test_results['throughput']['mean']:.2f} req/s |",
                f"| P95 Latency | {test_results['latency']['p95']*1000:.2f}ms |",
                f"| Memory Peak | {test_results['memory']['peak']/1024/1024:.2f}MB |"
            ])
        
        # Add regression summary
        regressions = summary.get("regressions", {})
        if critical := regressions.get("severity", {}).get("critical"):
            lines.extend([
                "",
                "### ⚠️ Critical Regressions",
                "```",
                *[f"- {metric}: {regressions['details'][metric]['degradation']:.1f}% degradation"
                  for metric in critical],
                "```"
            ])
        
        # Add trends
        trends = summary.get("trends", {})
        if trends:
            lines.extend([
                "",
                "### Performance Trends",
                "| Metric | Direction | Change |",
                "|--------|-----------|---------|"
            ])
            for metric, data in trends.items():
                emoji = "🔼" if data["direction"] == "improving" else "🔽"
                lines.append(
                    f"| {metric.title()} | {emoji} {data['direction']} | "
                    f"{abs(data['change_pct']):.1f}% |"
                )
        
        return "\n".join(lines)
    
    def _generate_text_report(self, summary: Dict[str, Any]) -> str:
        """Generate plain text report."""
        lines = [
            "Performance Test Results",
            "=" * 50,
            "",
            "Test Results:",
            "-" * 20
        ]
        
        # Add test results
        test_results = summary.get("tests", {})
        if test_results:
            lines.extend([
                f"Throughput: {test_results['throughput']['mean']:.2f} req/s",
                f"P95 Latency: {test_results['latency']['p95']*1000:.2f}ms",
                f"Memory Peak: {test_results['memory']['peak']/1024/1024:.2f}MB"
            ])
        
        # Add regressions
        regressions = summary.get("regressions", {})
        if regressions.get("total_count"):
            lines.extend([
                "",
                "Regressions:",
                "-" * 20
            ])
            for severity, metrics in regressions["severity"].items():
                if metrics:
                    lines.append(f"{severity.title()}:")
                    for metric in metrics:
                        details = regressions["details"][metric]
                        lines.append(
                            f"  - {metric}: {details['degradation']:.1f}% degradation"
                        )
        
        # Add trends
        trends = summary.get("trends", {})
        if trends:
            lines.extend([
                "",
                "Performance Trends:",
                "-" * 20
            ])
            for metric, data in trends.items():
                lines.append(
                    f"{metric.title()}: {data['direction']} "
                    f"({abs(data['change_pct']):.1f}% change)"
                )
        
        return "\n".join(lines)

def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Process performance test results")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Results directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output report path"
    )
    parser.add_argument(
        "--format",
        choices=["text", "github"],
        default="text",
        help="Report format"
    )
    
    args = parser.parse_args()
    
    try:
        processor = ResultsProcessor(args.results, args.format)
        summary = processor.process_results()
        report = processor.generate_report(summary)
        
        with open(args.output, "w") as f:
            f.write(report)
        
        logger.info(f"Report generated: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to process results: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
