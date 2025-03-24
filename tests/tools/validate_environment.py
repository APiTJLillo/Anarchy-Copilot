#!/usr/bin/env python3
"""Validate test environment setup and requirements."""

import sys
import os
import subprocess
import importlib.metadata
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import json
from dataclasses import dataclass
import shutil
import pkg_resources
from packaging.version import parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("env-validator")

@dataclass
class RequirementCheck:
    """Result of a requirement check."""
    name: str
    required_version: str
    installed_version: Optional[str]
    status: str
    message: str

class EnvironmentValidator:
    """Test environment validator."""

    def __init__(self):
        """Initialize validator."""
        self.project_root = Path(__file__).parent.parent.parent
        self.test_dir = self.project_root / "tests"
        self.results: List[RequirementCheck] = []

    def check_python_version(self) -> RequirementCheck:
        """Validate Python version."""
        required = ">=3.8.0"
        installed = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        is_valid = (sys.version_info.major == 3 and sys.version_info.minor >= 8)
        status = "ok" if is_valid else "error"
        message = (f"Python {installed} found" if is_valid else
                  f"Python {required} required, found {installed}")
        
        return RequirementCheck(
            name="python",
            required_version=required,
            installed_version=installed,
            status=status,
            message=message
        )

    def check_pip_packages(self) -> List[RequirementCheck]:
        """Verify installed pip packages against requirements."""
        results = []
        req_file = self.test_dir / "requirements-test.txt"
        
        if not req_file.exists():
            return [RequirementCheck(
                name="requirements-test.txt",
                required_version="N/A",
                installed_version=None,
                status="error",
                message=f"Requirements file not found: {req_file}"
            )]

        with open(req_file) as f:
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)

        for req in requirements:
            try:
                req_name = req.split(">=")[0] if ">=" in req else req.split("==")[0]
                req_version = req.split(">=")[1] if ">=" in req else req.split("==")[1]
                
                installed = importlib.metadata.version(req_name)
                
                is_valid = parse(installed) >= parse(req_version)
                status = "ok" if is_valid else "warning"
                message = (f"Version {installed} installed" if is_valid else
                          f"Version {req_version} required, found {installed}")
            except DistributionNotFound:
                status = "error"
                message = "Package not installed"
                installed = None
            except pkg_resources.DistributionNotFound:
                status = "error"
                message = "Package not installed"
                installed = None
                status = "error"
                message = "Package not installed"
                installed = None
            
            results.append(RequirementCheck(
                name=req_name,
                required_version=req_version,
                installed_version=installed,
                status=status,
                message=message
            ))
        
        return results

    def check_nuclei(self) -> RequirementCheck:
        """Check Nuclei installation and version."""
        try:
            result = subprocess.run(
                ["nuclei", "-version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.strip().split()[2]
            
            status = "ok"
            message = f"Nuclei {version} found"
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            version = None
            status = "error"
            message = "Nuclei not found or not working"
        
        return RequirementCheck(
            name="nuclei",
            required_version=">=2.8.0",
            installed_version=version,
            status=status,
            message=message
        )

    def check_test_assets(self) -> List[RequirementCheck]:
        """Verify test assets and directories."""
        results = []
        required_dirs = [
            ("test_data", self.test_dir / "data"),
            ("nuclei_templates", self.test_dir / "data" / "nuclei_templates"),
            ("test_reports", self.project_root / "test-reports"),
        ]
        
        for name, path in required_dirs:
            status = "ok" if path.exists() else "error"
            message = f"Directory exists" if path.exists() else "Directory not found"
            
            results.append(RequirementCheck(
                name=f"directory_{name}",
                required_version="N/A",
                installed_version=str(path) if path.exists() else None,
                status=status,
                message=message
            ))
        
        return results

    def check_permissions(self) -> List[RequirementCheck]:
        """Check file and directory permissions."""
        results = []
        paths_to_check = [
            (self.test_dir, "read"),
            (self.project_root / "test-reports", "write")
        ]
        
        for path, perm in paths_to_check:
            if not path.exists():
                status = "error"
                message = f"Path not found: {path}"
            elif perm == "read" and not os.access(path, os.R_OK):
                status = "error"
                message = f"No read permission: {path}"
            elif perm == "write" and not os.access(path, os.W_OK):
                status = "error"
                message = f"No write permission: {path}"
            else:
                status = "ok"
                message = f"{perm.capitalize()} permission OK"
            
            results.append(RequirementCheck(
                name=f"permission_{path.name}_{perm}",
                required_version=perm,
                installed_version=str(path),
                status=status,
                message=message
            ))
        
        return results

    def check_docker(self) -> RequirementCheck:
        """Check Docker availability."""
        try:
            result = subprocess.run(
                ["docker", "version"],
                capture_output=True,
                text=True,
                check=True
            )
            version = result.stdout.split("Version:")[1].split()[0]
            
            status = "ok"
            message = f"Docker {version} found"
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            version = None
            status = "warning"
            message = "Docker not found (optional)"
        
        return RequirementCheck(
            name="docker",
            required_version="optional",
            installed_version=version,
            status=status,
            message=message
        )

    def validate(self) -> bool:
        """Run all validation checks."""
        self.results.extend([
            self.check_python_version(),
            *self.check_pip_packages(),
            self.check_nuclei(),
            *self.check_test_assets(),
            *self.check_permissions(),
            self.check_docker()
        ])
        
        # Check for any errors
        has_errors = any(r.status == "error" for r in self.results)
        return not has_errors

    def print_report(self):
        """Print validation report."""
        print("\nTest Environment Validation Report")
        print("=================================")
        
        categories = {
            "System Requirements": ["python", "docker"],
            "Package Dependencies": [r.name for r in self.results if r.name not in ["python", "docker", "nuclei"] and not r.name.startswith(("directory_", "permission_"))],
            "External Tools": ["nuclei"],
            "Test Assets": [r.name for r in self.results if r.name.startswith("directory_")],
            "Permissions": [r.name for r in self.results if r.name.startswith("permission_")]
        }
        
        for category, items in categories.items():
            print(f"\n{category}:")
            print("-" * len(category) + "-")
            
            for item in items:
                result = next((r for r in self.results if r.name == item), None)
                if result:
                    status_symbol = "✓" if result.status == "ok" else "⚠" if result.status == "warning" else "✗"
                    print(f"{status_symbol} {result.name}: {result.message}")

        print("\nSummary:")
        print(f"Total Checks: {len(self.results)}")
        print(f"Passed: {sum(1 for r in self.results if r.status == 'ok')}")
        print(f"Warnings: {sum(1 for r in self.results if r.status == 'warning')}")
        print(f"Errors: {sum(1 for r in self.results if r.status == 'error')}")

def main():
    """Run environment validation."""
    validator = EnvironmentValidator()
    is_valid = validator.validate()
    validator.print_report()
    
    sys.exit(0 if is_valid else 1)

if __name__ == "__main__":
    main()
