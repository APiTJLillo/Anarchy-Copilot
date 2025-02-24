#!/usr/bin/env python3
"""Performance test runner with configuration management."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
import json
from datetime import datetime
import shutil

from .perf_config import PerformanceSettings, load_config
from .middleware_perf_visualizer import create_visualization
from .test_middleware_performance import run_performance_test
from .test_middleware_regression import RegressionTest

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceTestRunner:
    """Manages performance test execution and reporting."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config = load_config(config_path)
        self.results_dir = self.config.output_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize regression testing
        self.regression_tester = RegressionTest(
            self.config.baseline_dir,
            threshold=self.config.throughput_regression_threshold
        )
    
    async def run_tests(self) -> Dict[str, Any]:
        """Run the performance test suite."""
        logger.info("Starting performance test suite")
        
        # Prepare test environment
        self._prepare_environment()
        
        try:
            # Run tests
            results = await run_performance_test(self.config)
            
            # Check for regressions
            regressions = self.regression_tester.check_regression(
                results,
                self.regression_tester.get_baseline()
            )
            
            # Generate reports
            self._generate_reports(results, regressions)
            
            return {
                "results": results,
                "regressions": regressions,
                "timestamp": self.timestamp
            }
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise
    
    def _prepare_environment(self):
        """Prepare test environment."""
        if self.config.disable_gc:
            import gc
            gc.disable()
            logger.info("Garbage collection disabled")
        
        if self.config.process_priority is not None:
            try:
                import os
                os.nice(self.config.process_priority)
                logger.info(f"Process priority set to {self.config.process_priority}")
            except Exception as e:
                logger.warning(f"Failed to set process priority: {e}")
    
    def _generate_reports(self, results: Dict[str, Any], regressions: Dict[str, Any]):
        """Generate test reports."""
        # Save raw results
        results_file = self.results_dir / f"results_{self.timestamp}.json"
        with open(results_file, "w") as f:
            json.dump({"results": results, "regressions": regressions}, f, indent=2)
        
        # Create visualization report
        report_path = create_visualization(
            results,
            str(self.results_dir / f"report_{self.timestamp}")
        )
        
        # Generate summary
        self._generate_summary(results, regressions, report_path)
    
    def _generate_summary(self, results: Dict[str, Any], 
                         regressions: Dict[str, Any], report_path: Path):
        """Generate test summary."""
        summary = [
            "Performance Test Summary",
            "=" * 50,
            f"Timestamp: {self.timestamp}",
            f"Configuration: {self.config.__class__.__name__}",
            "",
            "Performance Metrics",
            "-" * 20,
            f"Throughput: {results['throughput']:.2f} req/s",
            f"P95 Latency: {results['latency_p95']*1000:.2f} ms",
            f"Memory Usage: {results['memory_usage']/1024/1024:.2f} MB",
            "",
        ]
        
        if regressions:
            summary.extend([
                "Performance Regressions",
                "-" * 20
            ])
            for metric, details in regressions.items():
                summary.append(
                    f"{metric}: {details['degradation']:.1f}% degradation"
                )
        else:
            summary.append("No performance regressions detected")
        
        summary.extend([
            "",
            "Reports",
            "-" * 20,
            f"Full Report: {report_path}",
            f"Raw Results: {self.results_dir}/results_{self.timestamp}.json"
        ])
        
        summary_path = self.results_dir / f"summary_{self.timestamp}.txt"
        with open(summary_path, "w") as f:
            f.write("\n".join(summary))
        
        logger.info("Test summary saved to %s", summary_path)

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run performance tests")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--profile",
        choices=["ci", "development", "production", "stress"],
        default="development",
        help="Use predefined configuration profile"
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Update performance baseline"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent / "config_examples" / f"{args.profile}.yml"
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Run tests
    runner = PerformanceTestRunner(config_path)
    try:
        results = asyncio.run(runner.run_tests())
        
        # Update baseline if requested
        if args.update_baseline and not results.get("regressions"):
            runner.regression_tester.save_baseline(results["results"])
            logger.info("Performance baseline updated")
        
        # Exit with status
        sys.exit(0 if not results.get("regressions") else 1)
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()
