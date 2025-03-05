#!/usr/bin/env python3
"""Type checking script with reporting and automatic fixes."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

def run_command(cmd: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return (success, output)."""
    print(f"\n=== Running {description} ===")
    try:
        result = subprocess.run(
            cmd, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} passed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False, e.stdout + e.stderr

def check_types(fix: bool = False, 
                strict: bool = False, 
                paths: Optional[List[str]] = None) -> bool:
    """Run type checking with optional fixing."""
    success = True
    project_root = Path(__file__).parent.parent

    # Default paths to check
    if not paths:
        paths = ['proxy']

    # Install dependencies if needed
    requirements_file = project_root / 'requirements-typing.txt'
    if requirements_file.exists():
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
        success, _ = run_command(cmd, "Installing type checking dependencies")
        if not success:
            return False

    # Format code if fixing is enabled
    if fix:
        # Run isort
        cmd = ['isort'] + paths
        success, _ = run_command(cmd, "Sorting imports")

        # Run black
        cmd = ['black'] + paths
        success, _ = run_command(cmd, "Formatting code")

    # Run mypy
    mypy_args = ['--config-file', str(project_root / 'mypy.ini')]
    if strict:
        mypy_args.extend(['--strict'])
    cmd = ['mypy'] + mypy_args + paths
    success, mypy_output = run_command(cmd, "Type checking with mypy")

    # Run flake8-pyi on stub files
    stub_files = []
    for path in paths:
        stub_files.extend(str(p) for p in Path(path).rglob('*.pyi'))
    if stub_files:
        cmd = ['flake8'] + stub_files
        success, flake_output = run_command(cmd, "Checking stub files")

    # Run pytype as additional check
    if strict:
        cmd = ['pytype'] + paths
        success, pytype_output = run_command(cmd, "Additional type checking with pytype")

    return success

def create_report(output_dir: Path) -> None:
    """Create an HTML report of type checking results."""
    try:
        import mypy.api
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run mypy with HTML report
        report_file = output_dir / 'type_check_report.html'
        args = [
            '--html-report', str(output_dir),
            '--txt-report', str(output_dir / 'type_check_report.txt'),
            'proxy'
        ]
        mypy.api.run(args)
        
        print(f"\nType check report generated at {report_file}")
    except ImportError:
        print("mypy not installed. Run with --install first.")

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Type checking tools')
    parser.add_argument('paths', nargs='*', help='Paths to check')
    parser.add_argument('--fix', action='store_true', help='Fix issues automatically')
    parser.add_argument('--strict', action='store_true', help='Use strict type checking')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    
    args = parser.parse_args()
    
    if args.install:
        cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-typing.txt']
        success, _ = run_command(cmd, "Installing dependencies")
        return 0 if success else 1

    if args.report:
        create_report(Path('type_report'))
        return 0

    success = check_types(
        fix=args.fix,
        strict=args.strict,
        paths=args.paths
    )
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
