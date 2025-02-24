#!/usr/bin/env python3
"""Check performance test results against thresholds."""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdChecker:
    """Check performance metrics against defined thresholds."""
    
    def __init__(self, profile: str):
        self.profile = profile
        self.thresholds = self._load_thresholds()
        
    def _load_thresholds(self) -> Dict[str, Any]:
        """Load thresholds for the current profile."""
        config_path = Path(__file__).parents[1] / "tests/proxy/server/handlers/config_examples" / f"{self.profile}.yml"
        
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
                return {
                    "throughput": config.get("min_throughput", 1000.0),
                    "latency_p95": config.get("max_latency_p95", 0.01),
                    "memory_mb": config.get("max_memory_mb", 1024.0),
                    "cpu_percent": config.get("max_cpu_percent", 80.0),
                    "regression_threshold": config.get("throughput_regression_threshold", 0.15)
                }
        except Exception as e:
            logger.error(f"Failed to load thresholds: {e}")
            return {}
    
    def check_results(self, results_dir: Path) -> List[str]:
        """Check test results against thresholds."""
        violations = []
        
        try:
            # Load test results
            results_file = next(results_dir.glob("results_*.json"))
            with open(results_file) as f:
                data = json.load(f)
            
            results = data.get("results", {})
            
            # Check throughput
            if results.get("throughput", 0) < self.thresholds["throughput"]:
                violations.append(
                    f"❌ Throughput {results['throughput']:.1f} req/s below minimum "
                    f"threshold of {self.thresholds['throughput']} req/s"
                )
            
            # Check latency
            if results.get("latency_p95", float("inf")) > self.thresholds["latency_p95"]:
                violations.append(
                    f"❌ P95 latency {results['latency_p95']*1000:.1f}ms exceeds maximum "
                    f"threshold of {self.thresholds['latency_p95']*1000:.1f}ms"
                )
            
            # Check memory usage
            memory_mb = results.get("memory_usage", 0) / (1024 * 1024)
            if memory_mb > self.thresholds["memory_mb"]:
                violations.append(
                    f"❌ Memory usage {memory_mb:.1f}MB exceeds maximum "
                    f"threshold of {self.thresholds['memory_mb']}MB"
                )
            
            # Check regressions
            regressions = data.get("regressions", {})
            for metric, details in regressions.items():
                if details["degradation"] > self.thresholds["regression_threshold"] * 100:
                    violations.append(
                        f"❌ {metric} regression of {details['degradation']:.1f}% exceeds "
                        f"threshold of {self.thresholds['regression_threshold']*100}%"
                    )
            
            # Load system metrics
            try:
                with open(results_dir / "metrics_summary.txt") as f:
                    metrics_text = f.read()
                    
                # Check CPU usage
                import re
                if match := re.search(r"Peak CPU.*?(\d+\.?\d*)%", metrics_text):
                    peak_cpu = float(match.group(1))
                    if peak_cpu > self.thresholds["cpu_percent"]:
                        violations.append(
                            f"❌ Peak CPU usage {peak_cpu:.1f}% exceeds maximum "
                            f"threshold of {self.thresholds['cpu_percent']}%"
                        )
            except Exception as e:
                logger.warning(f"Failed to check system metrics: {e}")
            
            # Generate summary
            if not violations:
                logger.info("✅ All performance metrics within thresholds")
            else:
                logger.error("Performance threshold violations found:")
                for violation in violations:
                    logger.error(violation)
            
            # Save detailed report
            self._save_report(results_dir, results, violations)
            
        except Exception as e:
            logger.error(f"Failed to check results: {e}")
            violations.append(f"❌ Error checking results: {e}")
        
        return violations
    
    def _save_report(self, results_dir: Path, results: Dict[str, Any],
                    violations: List[str]):
        """Save detailed threshold report."""
        report = [
            f"Performance Threshold Report - {self.profile}",
            "=" * 50,
            "",
            "Thresholds:",
            "-" * 20
        ]
        
        for metric, value in self.thresholds.items():
            report.append(f"{metric}: {value}")
        
        report.extend([
            "",
            "Results:",
            "-" * 20
        ])
        
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                report.append(f"{metric}: {value:.2f}")
        
        if violations:
            report.extend([
                "",
                "Violations:",
                "-" * 20
            ])
            report.extend(violations)
        else:
            report.extend([
                "",
                "✅ All checks passed"
            ])
        
        report_path = results_dir / "threshold_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(report))
        
        logger.info(f"Detailed report saved to {report_path}")

def main():
    """Script entry point."""
    parser = argparse.ArgumentParser(description="Check performance thresholds")
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Test results directory"
    )
    parser.add_argument(
        "--profile",
        choices=["ci", "development", "production", "stress"],
        default="development",
        help="Test profile to use for thresholds"
    )
    
    args = parser.parse_args()
    
    checker = ThresholdChecker(args.profile)
    violations = checker.check_results(args.results)
    
    # Exit with status code
    sys.exit(1 if violations else 0)

if __name__ == "__main__":
    main()
