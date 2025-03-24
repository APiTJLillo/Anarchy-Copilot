#!/usr/bin/env python3
"""Combined type analysis tool with tests, metrics, and visualizations."""

import argparse
import asyncio
import json
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_environment() -> bool:
    """Install required dependencies."""
    requirements = [
        "mypy>=1.0.0",
        "plotly>=5.0.0",
        "pytest>=7.0.0",
        "pytest-mypy>=0.10.0",
        "pytest-asyncio>=0.20.0",
        "psutil>=5.9.0"
    ]
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + requirements,
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        return False

async def run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    """Run complete type analysis."""
    from run_type_tests import run_type_tests
    
    # Run type tests and collect metrics
    metrics = run_type_tests(args)
    
    # Generate reports
    if args.report:
        report_dir = project_root / "type_report"
        report_dir.mkdir(exist_ok=True)
        
        # Save metrics JSON
        metrics_file = report_dir / "type_check_metrics.json"
        with metrics_file.open("w") as f:
            json.dump(metrics, f, indent=2)
        
        # Generate visualization
        from visualize_type_metrics import generate_html_report
        vis_file = report_dir / "type_check_visualization.html"
        generate_html_report(metrics, vis_file)
        
        if args.view:
            webbrowser.open(str(vis_file))
    
    return metrics

def print_summary(metrics: Dict[str, Any], args: argparse.Namespace) -> None:
    """Print analysis summary."""
    print("\n=== Type Analysis Summary ===")
    
    # Coverage stats
    coverage = metrics['type_coverage']
    print("\nType Coverage:")
    print(f"  Files: {coverage['files_with_types']}/{coverage['total_files']} "
          f"({coverage['files_with_types']/coverage['total_files']*100:.1f}%)")
    print(f"  Functions: {coverage['typed_functions']}/{coverage['total_functions']} "
          f"({coverage['typed_functions']/coverage['total_functions']*100:.1f}%)")
    print(f"  Variables: {coverage['typed_variables']}/{coverage['total_variables']} "
          f"({coverage['typed_variables']/coverage['total_variables']*100:.1f}%)")
    
    # Test results
    print("\nTest Results:")
    for test_name, results in metrics['tests'].items():
        status = "✓" if results['success'] else "✗"
        print(f"  {status} {test_name:<20} {results['duration']:.2f}s")
    
    # Performance
    perf = metrics['performance']
    print("\nPerformance:")
    print(f"  Total Duration: {perf['total_duration']:.2f}s")
    print(f"  Peak Memory: {perf['max_memory_delta']:.1f}MB")
    
    if args.report:
        print("\nReports generated in: type_report/")
        if args.view:
            print("Opening visualization in browser...")

def check_coverage_goals(metrics: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check if coverage meets goals."""
    goals = {
        'files': 80,  # 80% of files should have types
        'functions': 90,  # 90% of functions should have types
        'variables': 70,  # 70% of variables should have types
    }
    
    coverage = metrics['type_coverage']
    failures = []
    
    # Check file coverage
    file_coverage = coverage['files_with_types'] / coverage['total_files'] * 100
    if file_coverage < goals['files']:
        failures.append(
            f"File coverage {file_coverage:.1f}% below goal of {goals['files']}%"
        )
    
    # Check function coverage
    func_coverage = coverage['typed_functions'] / coverage['total_functions'] * 100
    if func_coverage < goals['functions']:
        failures.append(
            f"Function coverage {func_coverage:.1f}% below goal of {goals['functions']}%"
        )
    
    # Check variable coverage
    var_coverage = coverage['typed_variables'] / coverage['total_variables'] * 100
    if var_coverage < goals['variables']:
        failures.append(
            f"Variable coverage {var_coverage:.1f}% below goal of {goals['variables']}%"
        )
    
    return len(failures) == 0, failures

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run complete type analysis with reporting"
    )
    parser.add_argument("--strict", action="store_true",
                      help="Enable strict type checking")
    parser.add_argument("--report", action="store_true",
                      help="Generate detailed reports")
    parser.add_argument("--view", action="store_true",
                      help="Open visualization in browser")
    parser.add_argument("--install", action="store_true",
                      help="Install dependencies")
    parser.add_argument("--check-coverage", action="store_true",
                      help="Check coverage goals")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Show verbose output")
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install:
        if not setup_environment():
            return 1
    
    try:
        # Run analysis
        metrics = asyncio.run(run_analysis(args))
        
        # Print results
        print_summary(metrics, args)
        
        # Check coverage goals
        if args.check_coverage:
            success, failures = check_coverage_goals(metrics)
            if not success:
                print("\nCoverage goals not met:")
                for failure in failures:
                    print(f"  ! {failure}")
                return 1
        
        # Check overall success
        return 0 if all(t['success'] for t in metrics['tests'].values()) else 1
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted")
        return 1
    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
