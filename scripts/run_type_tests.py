#!/usr/bin/env python3
"""Run type checking tests with performance metrics."""

import time
import argparse
import subprocess
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

def measure_execution_time(cmd: List[str], cwd: Optional[Path] = None) -> Tuple[bool, str, float, Dict[str, Any]]:
    """Run command and measure execution time and memory usage."""
    try:
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss
    except ImportError:
        initial_memory = 0

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd
        )
        success = True
        output = result.stdout
    except subprocess.CalledProcessError as e:
        success = False
        output = e.stdout + e.stderr

    duration = time.time() - start_time

    try:
        final_memory = process.memory_info().rss if 'process' in locals() else 0
        memory_delta = (final_memory - initial_memory) / 1024 / 1024  # MB
    except Exception:
        memory_delta = 0

    metrics = {
        "duration_seconds": duration,
        "memory_delta_mb": memory_delta,
        "timestamp": datetime.now().isoformat()
    }

    return success, output, duration, metrics

def count_type_annotations(path: Path) -> Dict[str, int]:
    """Count type annotations in Python files."""
    counts = {
        "total_files": 0,
        "files_with_types": 0,
        "total_functions": 0,
        "typed_functions": 0,
        "total_variables": 0,
        "typed_variables": 0,
        "stub_files": 0
    }

    import ast
    
    def has_type_annotation(node: ast.AST) -> bool:
        return hasattr(node, 'annotation') and node.annotation is not None

    class TypeCounter(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            counts["total_functions"] += 1
            if has_type_annotation(node.returns) or any(
                has_type_annotation(arg) for arg in node.args.args
            ):
                counts["typed_functions"] += 1
            self.generic_visit(node)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            counts["total_variables"] += 1
            if node.annotation:
                counts["typed_variables"] += 1
            self.generic_visit(node)

    for file in path.rglob("*.py"):
        if file.name.startswith("__"):
            continue
            
        counts["total_files"] += 1
        
        # Check for stub file
        if file.with_suffix('.pyi').exists():
            counts["stub_files"] += 1
        
        try:
            with open(file) as f:
                tree = ast.parse(f.read())
                
            visitor = TypeCounter()
            visitor.visit(tree)
            
            # Check if file has any type annotations
            if any([
                visitor.typed_functions > 0,
                visitor.typed_variables > 0
            ]):
                counts["files_with_types"] += 1
                
        except Exception as e:
            print(f"Error processing {file}: {e}", file=sys.stderr)

    return counts

def run_type_tests(args: argparse.Namespace) -> Dict[str, Any]:
    """Run type checking tests and collect metrics."""
    project_root = Path(__file__).parent.parent
    metrics: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "tests": {},
        "type_coverage": {},
        "performance": {}
    }

    # Count type annotations
    metrics["type_coverage"] = count_type_annotations(project_root / "proxy")

    # Run mypy checks
    mypy_cmd = [
        "mypy",
        "--config-file", "mypy.ini",
        "proxy"
    ]
    if args.strict:
        mypy_cmd.append("--strict")
    
    success, output, duration, perf = measure_execution_time(mypy_cmd, project_root)
    metrics["tests"]["mypy"] = {
        "success": success,
        "duration": duration,
        "output": output if not success or args.verbose else None,
        "performance": perf
    }

    # Run pytest type tests
    pytest_cmd = [
        "pytest",
        "tests/test_types.py",
        "-v",
        "--durations=0"
    ]
    success, output, duration, perf = measure_execution_time(pytest_cmd, project_root)
    metrics["tests"]["pytest"] = {
        "success": success,
        "duration": duration,
        "output": output if not success or args.verbose else None,
        "performance": perf
    }

    # Run stubtest
    stubtest_cmd = [
        "stubtest",
        "proxy",
        "--ignore-missing-stub"
    ]
    success, output, duration, perf = measure_execution_time(stubtest_cmd, project_root)
    metrics["tests"]["stubtest"] = {
        "success": success,
        "duration": duration,
        "output": output if not success or args.verbose else None,
        "performance": perf
    }

    # Calculate overall metrics
    metrics["performance"] = {
        "total_duration": sum(t["duration"] for t in metrics["tests"].values()),
        "max_memory_delta": max(t["performance"]["memory_delta_mb"] 
                              for t in metrics["tests"].values()),
        "type_coverage_percent": (
            metrics["type_coverage"]["typed_functions"] /
            metrics["type_coverage"]["total_functions"] * 100
            if metrics["type_coverage"]["total_functions"] > 0 else 0
        )
    }

    # Generate report
    if args.report:
        report_dir = project_root / "type_report"
        report_dir.mkdir(exist_ok=True)
        
        # Save JSON report
        report_path = report_dir / "type_check_metrics.json"
        with report_path.open("w") as f:
            json.dump(metrics, f, indent=2)

        # Generate HTML report
        html_report = report_dir / "type_check_report.html"
        generate_html_report(metrics, html_report)
        
        print(f"\nReports saved to {report_dir}")

    return metrics

def generate_html_report(metrics: Dict[str, Any], output_path: Path) -> None:
    """Generate HTML report from metrics."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Type Checking Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .success {{ color: green; }}
            .failure {{ color: red; }}
            .metric {{ margin: 10px 0; }}
            .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ccc; }}
            pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>Type Checking Report</h1>
        <p>Generated on: {metrics['timestamp']}</p>

        <div class="section">
            <h2>Type Coverage</h2>
            <div class="metric">Files with types: {metrics['type_coverage']['files_with_types']}/{metrics['type_coverage']['total_files']}</div>
            <div class="metric">Typed functions: {metrics['type_coverage']['typed_functions']}/{metrics['type_coverage']['total_functions']}</div>
            <div class="metric">Type coverage: {metrics['performance']['type_coverage_percent']:.1f}%</div>
            <div class="metric">Stub files: {metrics['type_coverage']['stub_files']}</div>
        </div>

        <div class="section">
            <h2>Test Results</h2>
            {_generate_test_section(metrics['tests'])}
        </div>

        <div class="section">
            <h2>Performance</h2>
            <div class="metric">Total duration: {metrics['performance']['total_duration']:.2f}s</div>
            <div class="metric">Max memory delta: {metrics['performance']['max_memory_delta']:.1f}MB</div>
        </div>
    </body>
    </html>
    """
    
    output_path.write_text(html)

def _generate_test_section(tests: Dict[str, Any]) -> str:
    """Generate HTML for test results section."""
    html = []
    for test_name, results in tests.items():
        status = "success" if results["success"] else "failure"
        html.append(f"""
            <div class="metric">
                <h3>{test_name}</h3>
                <p class="{status}">Status: {'✓ Passed' if results['success'] else '✗ Failed'}</p>
                <p>Duration: {results['duration']:.2f}s</p>
                {f"<pre>{results['output']}</pre>" if results.get('output') else ''}
            </div>
        """)
    return "\n".join(html)

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run type checking tests")
    parser.add_argument("--strict", action="store_true",
                      help="Enable strict type checking")
    parser.add_argument("--report", action="store_true",
                      help="Generate detailed reports")
    parser.add_argument("--verbose", "-v", action="store_true",
                      help="Show detailed output")
    
    args = parser.parse_args()
    metrics = run_type_tests(args)
    
    # Print summary
    print("\nType Checking Results:")
    for test_name, results in metrics["tests"].items():
        print(f"{test_name}: {'✓' if results['success'] else '✗'}")
    
    print(f"\nType Coverage: {metrics['performance']['type_coverage_percent']:.1f}%")
    print(f"Total Duration: {metrics['performance']['total_duration']:.2f}s")
    
    return 0 if all(t["success"] for t in metrics["tests"].values()) else 1

if __name__ == "__main__":
    sys.exit(main())
