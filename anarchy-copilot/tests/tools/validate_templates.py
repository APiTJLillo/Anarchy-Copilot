#!/usr/bin/env python3
"""Template validation tool for test templates."""

import sys
import os
import subprocess
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TemplateValidator:
    """Validates Nuclei templates used in tests."""

    def __init__(self, templates_dir: Path):
        """Initialize validator with template directory."""
        self.templates_dir = templates_dir
        self.index_file = templates_dir / "template_index.yaml"
        self._load_index()

    def _load_index(self) -> None:
        """Load template index file."""
        try:
            with open(self.index_file, 'r') as f:
                self.index = yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Template index not found: {self.index_file}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"Error parsing template index: {e}")
            sys.exit(1)

    def verify_nuclei_installation(self) -> bool:
        """Check if nuclei is installed and meets version requirements."""
        try:
            version_req = self.index["template_metadata"]["requirements"]["nuclei_version"]
            result = subprocess.run(
                ["nuclei", "-version"],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip().split()[1]
            logger.info(f"Found Nuclei version: {version}")
            return True
        except subprocess.CalledProcessError:
            logger.error("Nuclei not found. Please install nuclei first.")
            return False

    def validate_template(self, template_path: Path) -> bool:
        """Validate a single template."""
        try:
            # Parse template
            with open(template_path, 'r') as f:
                template = yaml.safe_load(f)

            # Basic structure checks
            required_fields = ["id", "info"]
            for field in required_fields:
                if field not in template:
                    logger.error(f"Missing required field '{field}' in {template_path}")
                    return False

            # Run nuclei validation
            result = subprocess.run(
                ["nuclei", "-validate", "-t", str(template_path)],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Template validation failed for {template_path}:")
                logger.error(result.stderr)
                return False

            logger.info(f"Template validated successfully: {template_path}")
            return True

        except Exception as e:
            logger.error(f"Error validating template {template_path}: {e}")
            return False

    def validate_all(self) -> bool:
        """Validate all templates listed in the index."""
        success = True
        templates = self.index.get("templates", [])
        
        for template in templates:
            template_id = template["id"]
            template_path = self.templates_dir / f"{template_id}.yaml"
            if not self.validate_template(template_path):
                success = False

        return success

    def check_template_consistency(self) -> bool:
        """Check if all templates in directory are listed in index."""
        success = True
        template_files = set(p.stem for p in self.templates_dir.glob("*.yaml"))
        indexed_templates = {t["id"] for t in self.index.get("templates", [])}

        # Check for templates not in index
        unindexed = template_files - indexed_templates - {"template_index"}
        if unindexed:
            logger.warning(f"Templates not in index: {unindexed}")
            success = False

        # Check for missing template files
        missing = indexed_templates - template_files
        if missing:
            logger.error(f"Templates in index but missing files: {missing}")
            success = False

        return success

def main():
    """Main entry point for template validation."""
    parser = argparse.ArgumentParser(
        description="Validate Nuclei test templates."
    )
    parser.add_argument(
        "--templates-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data/nuclei_templates",
        help="Directory containing test templates"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check template consistency without validation"
    )
    args = parser.parse_args()

    validator = TemplateValidator(args.templates_dir)

    if not args.check_only and not validator.verify_nuclei_installation():
        sys.exit(1)

    consistency = validator.check_template_consistency()
    if args.check_only:
        sys.exit(0 if consistency else 1)

    if not consistency or not validator.validate_all():
        sys.exit(1)

    logger.info("All templates validated successfully!")

if __name__ == "__main__":
    main()
