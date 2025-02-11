"""Test tools and utilities."""

from .validate_templates import TemplateValidator
from .validate_environment import EnvironmentValidator, RequirementCheck
from .test_summary import TestSummarizer

__all__ = [
    'TemplateValidator',
    'EnvironmentValidator',
    'RequirementCheck',
    'TestSummarizer'
]

# Version information
__version__ = "0.1.0"

# Make the validate_environment function available at package level
def validate_environment(print_report: bool = True) -> bool:
    """
    Validate the test environment setup.

    Args:
        print_report: Whether to print the validation report to stdout

    Returns:
        bool: True if environment is valid, False otherwise
    """
    validator = EnvironmentValidator()
    is_valid = validator.validate()
    if print_report:
        validator.print_report()
    return is_valid

# Make test summary generation available at package level
def generate_test_summary(
    junit_path: str,
    coverage_path: str,
    output_dir: str,
    environment: dict = None
) -> bool:
    """
    Generate test summary report.

    Args:
        junit_path: Path to JUnit XML test results
        coverage_path: Path to coverage XML report
        output_dir: Directory to write reports to
        environment: Optional environment information to include in the report

    Returns:
        bool: True if report generation succeeded, False otherwise
    """
    from pathlib import Path
    from .test_summary import TestSummarizer

    try:
        summarizer = TestSummarizer(Path(output_dir))
        summary = summarizer.generate_suite_summary(
            Path(junit_path),
            Path(coverage_path),
            environment or {}
        )
        summarizer.generate_html_report(summary)
        return True
    except Exception as e:
        import logging
        logging.error(f"Failed to generate test summary: {e}")
        return False

# Make template validation available at package level
def validate_templates(
    template_dir: str,
    patterns: list = None,
    strict: bool = False
) -> bool:
    """
    Validate Nuclei templates.

    Args:
        template_dir: Directory containing templates
        patterns: Optional list of filename patterns to match
        strict: Whether to enforce strict validation

    Returns:
        bool: True if all templates are valid, False otherwise
    """
    from pathlib import Path
    validator = TemplateValidator(Path(template_dir))
    return validator.validate_all(patterns=patterns, strict=strict)
