#!/usr/bin/env python3
"""Check benchmark results against performance budget."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BudgetViolation:
    """Represents a performance budget violation."""
    def __init__(self, metric: str, value: float, threshold: float, severity: str):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.severity = severity
        self.timestamp = datetime.now()

    def __str__(self) -> str:
        return (f"{self.severity.upper()}: {self.metric} = {self.value:.2f} "
                f"(threshold: {self.threshold:.2f})")

class PerformanceBudget:
    """Manages and enforces performance budgets."""
    
    def __init__(self, budget_file: Path):
        self.budget = self._load_budget(budget_file)
        self.violations: List[BudgetViolation] = []
        self.warnings: List[BudgetViolation] = []

    def _load_budget(self, budget_file: Path) -> Dict[str, Any]:
        """Load performance budget configuration."""
        try:
            return json.loads(budget_file.read_text())
        except Exception as e:
            logger.error(f"Error loading budget file: {e}")
            sys.exit(1)

    def check_processing_speed(self, metrics: Dict[str, Any]) -> None:
        """Check processing speed metrics."""
        budget = self.budget["processing_speed"]["suggestions_per_second"]
        value = metrics.get("medium", {}).get("data", {}).get("stats", {}).get("suggestions_per_second", 0)
        
        if value < budget["min"]:
            self.violations.append(BudgetViolation(
                "Processing Speed",
                value,
                budget["min"],
                "critical"
            ))
        elif value < budget["target"]:
            self.warnings.append(BudgetViolation(
                "Processing Speed",
                value,
                budget["target"],
                "warning"
            ))

    def check_memory_usage(self, metrics: Dict[str, Any]) -> None:
        """Check memory usage metrics."""
        budget = self.budget["memory"]["peak_mb"]
        value = metrics.get("memory_usage", {}).get("data", {}).get("max_memory_mb", 0)
        
        if value > budget["max"]:
            self.violations.append(BudgetViolation(
                "Peak Memory",
                value,
                budget["max"],
                "critical"
            ))
        elif value > budget["warning"]:
            self.warnings.append(BudgetViolation(
                "Peak Memory",
                value,
                budget["warning"],
                "warning"
            ))

    def check_type_quality(self, metrics: Dict[str, Any]) -> None:
        """Check type suggestion quality metrics."""
        budget = self.budget["type_quality"]
        quality_data = metrics.get("suggestion_quality", {}).get("data", {})
        
        # Check specificity ratio
        specificity = quality_data.get("specificity_ratio", 0)
        if specificity < budget["specificity_ratio"]["min"]:
            self.violations.append(BudgetViolation(
                "Type Specificity",
                specificity,
                budget["specificity_ratio"]["min"],
                "critical"
            ))
        
        # Check Any type usage
        total = quality_data.get("total_suggestions", 1)
        any_ratio = quality_data.get("any_suggestions", 0) / total
        if any_ratio > budget["any_type_ratio"]["max"]:
            self.violations.append(BudgetViolation(
                "Any Type Usage",
                any_ratio,
                budget["any_type_ratio"]["max"],
                "critical"
            ))

    def check_scaling(self, metrics: Dict[str, Any]) -> None:
        """Check scaling performance metrics."""
        budget = self.budget["scaling"]
        
        # Check large file processing time
        large_time = metrics.get("large", {}).get("data", {}).get("duration", 0)
        if large_time > budget["large_file_processing_seconds"]["max"]:
            self.violations.append(BudgetViolation(
                "Large File Processing",
                large_time,
                budget["large_file_processing_seconds"]["max"],
                "critical"
            ))
        
        # Check linear scaling
        sizes = ["small", "medium", "large"]
        times = []
        for size in sizes:
            if size in metrics:
                times.append(metrics[size]["data"]["duration"])
        
        if len(times) >= 3:
            # Check if scaling is roughly linear
            ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
            avg_ratio = sum(ratios) / len(ratios)
            max_deviation = max(abs(r - avg_ratio) for r in ratios)
            
            if max_deviation > budget["linear_scaling_threshold"]["max_deviation"] / 100:
                self.violations.append(BudgetViolation(
                    "Scaling Linearity",
                    max_deviation * 100,
                    budget["linear_scaling_threshold"]["max_deviation"],
                    "warning"
                ))

    def check_feature_performance(self, metrics: Dict[str, Any]) -> None:
        """Check feature-specific performance metrics."""
        budget = self.budget["feature_performance"]
        
        for feature, limits in budget.items():
            key = f"feature_{feature}"
            if key in metrics:
                duration_ms = metrics[key]["data"]["duration"] * 1000
                if duration_ms > limits["max_processing_ms"]:
                    self.violations.append(BudgetViolation(
                        f"{feature.title()} Processing",
                        duration_ms,
                        limits["max_processing_ms"],
                        "warning"
                    ))

    def generate_report(self) -> str:
        """Generate a markdown report of budget checks."""
        sections = []
        
        # Add summary section
        total_violations = len(self.violations)
        total_warnings = len(self.warnings)
        status = "✅ PASS" if total_violations == 0 else "❌ FAIL"
        
        sections.append(f"""
# Performance Budget Check Report

Status: {status}
- Critical Violations: {total_violations}
- Warnings: {total_warnings}
""")

        # Add violations section if any
        if self.violations:
            sections.append("\n## Critical Violations\n")
            for v in self.violations:
                sections.append(f"- {v}")

        # Add warnings section if any
        if self.warnings:
            sections.append("\n## Warnings\n")
            for w in self.warnings:
                sections.append(f"- {w}")

        return "\n".join(sections)

def main() -> int:
    """Main entry point."""
    try:
        # Load benchmark results
        results_file = Path("benchmark_results/type_suggestion_benchmarks.json")
        if not results_file.exists():
            logger.error("No benchmark results found")
            return 1

        metrics = json.loads(results_file.read_text())
        
        # Load and check budget
        budget = PerformanceBudget(Path("performance-budget.json"))
        
        # Run all checks
        budget.check_processing_speed(metrics)
        budget.check_memory_usage(metrics)
        budget.check_type_quality(metrics)
        budget.check_scaling(metrics)
        budget.check_feature_performance(metrics)
        
        # Generate and save report
        report = budget.generate_report()
        report_file = Path("benchmark_results/budget_report.md")
        report_file.write_text(report)
        
        # Print report to console
        print(report)
        
        # Create GitHub annotation if running in CI
        if "GITHUB_ACTIONS" in os.environ:
            for violation in budget.violations:
                print(f"::error::{violation}")
            for warning in budget.warnings:
                print(f"::warning::{warning}")
        
        # Exit with failure if there are violations
        return 1 if budget.violations else 0
        
    except Exception as e:
        logger.error(f"Error running budget check: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
