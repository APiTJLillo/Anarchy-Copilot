"""Test summary generation and coverage tracking."""

import pytest
from pathlib import Path
import json
from typing import Dict, Any, List
import logging
from datetime import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class TestSummaryStats:
    """Test execution statistics."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration: float = 0.0
    coverage: float = 0.0

@dataclass
class ModuleSummary:
    """Summary for a test module."""
    name: str
    tests: TestSummaryStats
    coverage: Dict[str, float]
    last_run: str

@dataclass
class TestSuiteResult:
    """Overall test suite execution results."""
    modules: List[ModuleSummary]
    total_stats: TestSummaryStats
    timestamp: str
    environment: Dict[str, str]

class TestSummarizer:
    """Generate test summaries and reports."""

    def __init__(self, output_dir: Path):
        """Initialize test summarizer."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_junit_results(self, junit_path: Path) -> Dict[str, Any]:
        """Parse JUnit XML test results."""
        tree = ET.parse(junit_path)
        root = tree.getroot()

        results = defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "duration": 0.0
        })

        for testsuite in root.findall(".//testsuite"):
            module_name = testsuite.get("name", "unknown")
            results[module_name]["total"] += int(testsuite.get("tests", 0))
            results[module_name]["failed"] += int(testsuite.get("failures", 0))
            results[module_name]["skipped"] += int(testsuite.get("skipped", 0))
            results[module_name]["duration"] += float(testsuite.get("time", 0))
            results[module_name]["passed"] = (
                results[module_name]["total"]
                - results[module_name]["failed"]
                - results[module_name]["skipped"]
            )

        return dict(results)

    def parse_coverage_data(self, coverage_path: Path) -> Dict[str, float]:
        """Parse coverage data from coverage XML."""
        tree = ET.parse(coverage_path)
        root = tree.getroot()

        coverage = {}
        for package in root.findall(".//package"):
            name = package.get("name", "unknown")
            covered_lines = sum(1 for line in package.findall(".//line")
                              if int(line.get("hits", 0)) > 0)
            total_lines = len(package.findall(".//line"))
            coverage[name] = (covered_lines / total_lines * 100) if total_lines else 0

        return coverage

    def generate_module_summary(
        self,
        module_name: str,
        test_results: Dict[str, Any],
        coverage_data: Dict[str, float]
    ) -> ModuleSummary:
        """Generate summary for a test module."""
        stats = TestSummaryStats(
            total_tests=test_results["total"],
            passed=test_results["passed"],
            failed=test_results["failed"],
            skipped=test_results["skipped"],
            duration=test_results["duration"],
            coverage=coverage_data.get(module_name, 0.0)
        )

        return ModuleSummary(
            name=module_name,
            tests=stats,
            coverage=coverage_data,
            last_run=datetime.now().isoformat()
        )

    def generate_suite_summary(
        self,
        junit_path: Path,
        coverage_path: Path,
        environment: Dict[str, str]
    ) -> TestSuiteResult:
        """Generate summary for entire test suite."""
        test_results = self.parse_junit_results(junit_path)
        coverage_data = self.parse_coverage_data(coverage_path)

        modules = []
        total_stats = TestSummaryStats()

        for module_name, results in test_results.items():
            module_summary = self.generate_module_summary(
                module_name, results, coverage_data
            )
            modules.append(module_summary)

            # Update total stats
            total_stats.total_tests += module_summary.tests.total_tests
            total_stats.passed += module_summary.tests.passed
            total_stats.failed += module_summary.tests.failed
            total_stats.skipped += module_summary.tests.skipped
            total_stats.duration += module_summary.tests.duration

        # Calculate overall coverage
        if coverage_data:
            total_stats.coverage = sum(coverage_data.values()) / len(coverage_data)

        return TestSuiteResult(
            modules=modules,
            total_stats=total_stats,
            timestamp=datetime.now().isoformat(),
            environment=environment
        )

    def save_summary(self, summary: TestSuiteResult, name: str = "test_summary"):
        """Save test summary to file."""
        output_path = self.output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(asdict(summary), f, indent=2)
        return output_path

    def generate_html_report(
        self,
        summary: TestSuiteResult,
        template_path: Optional[Path] = None,
        output_name: str = "test_report"
    ) -> Path:
        """Generate HTML test report."""
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "test_report.html"

        if not template_path.exists():
            raise FileNotFoundError(f"Report template not found: {template_path}")

        import jinja2
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_path.parent)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        template = env.get_template(template_path.name)

        report_html = template.render(
            summary=summary,
            report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        output_path = self.output_dir / f"{output_name}.html"
        output_path.write_text(report_html)
        return output_path

@pytest.fixture
def test_summarizer(tmp_path: Path) -> TestSummarizer:
    """Provide test summarizer instance."""
    return TestSummarizer(tmp_path / "test_reports")

@pytest.fixture
def sample_junit_xml(tmp_path: Path) -> Path:
    """Create sample JUnit XML file."""
    content = """<?xml version="1.0" encoding="utf-8"?>
    <testsuites>
        <testsuite name="example_module" tests="10" failures="1" skipped="1" time="1.5">
            <testcase name="test_1" time="0.1"/>
            <testcase name="test_2" time="0.2">
                <failure message="Test failed">Stack trace</failure>
            </testcase>
            <testcase name="test_3" time="0.1">
                <skipped message="Test skipped"/>
            </testcase>
        </testsuite>
    </testsuites>
    """
    junit_path = tmp_path / "junit.xml"
    junit_path.write_text(content)
    return junit_path

@pytest.fixture
def sample_coverage_xml(tmp_path: Path) -> Path:
    """Create sample coverage XML file."""
    content = """<?xml version="1.0" ?>
    <coverage>
        <package name="example_module">
            <line hits="1" number="1"/>
            <line hits="1" number="2"/>
            <line hits="0" number="3"/>
            <line hits="1" number="4"/>
        </package>
    </coverage>
    """
    coverage_path = tmp_path / "coverage.xml"
    coverage_path.write_text(content)
    return coverage_path

def test_parse_junit_results(test_summarizer: TestSummarizer, sample_junit_xml: Path):
    """Test parsing JUnit XML results."""
    results = test_summarizer.parse_junit_results(sample_junit_xml)
    assert "example_module" in results
    assert results["example_module"]["total"] == 10
    assert results["example_module"]["failed"] == 1
    assert results["example_module"]["skipped"] == 1
    assert results["example_module"]["duration"] == 1.5

def test_parse_coverage_data(test_summarizer: TestSummarizer, sample_coverage_xml: Path):
    """Test parsing coverage data."""
    coverage = test_summarizer.parse_coverage_data(sample_coverage_xml)
    assert "example_module" in coverage
    assert coverage["example_module"] == 75.0  # 3 out of 4 lines covered

def test_generate_suite_summary(
    test_summarizer: TestSummarizer,
    sample_junit_xml: Path,
    sample_coverage_xml: Path
):
    """Test generating full test suite summary."""
    environment = {"python": "3.8.0", "os": "Linux"}
    summary = test_summarizer.generate_suite_summary(
        sample_junit_xml,
        sample_coverage_xml,
        environment
    )

    assert len(summary.modules) == 1
    assert summary.total_stats.total_tests == 10
    assert summary.total_stats.failed == 1
    assert summary.total_stats.coverage == 75.0
    assert summary.environment == environment

def test_save_summary(test_summarizer: TestSummarizer):
    """Test saving test summary."""
    summary = TestSuiteResult(
        modules=[],
        total_stats=TestSummaryStats(),
        timestamp=datetime.now().isoformat(),
        environment={}
    )

    output_path = test_summarizer.save_summary(summary)
    assert output_path.exists()
    with open(output_path) as f:
        saved_data = json.load(f)
        assert saved_data["timestamp"] == summary.timestamp
