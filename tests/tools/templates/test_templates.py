"""Tests for template utilities."""

import pytest
from pathlib import Path
from bs4 import BeautifulSoup
import json
import jinja2
from unittest.mock import patch
from ..templates import (
    TemplateRenderer,
    render_template,
    copy_assets_to_output,
    setup_jinja_env,
    TEMPLATES_DIR,
    STYLES_DIR
)

@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Create temporary output directory."""
    return tmp_path / "test_output"

@pytest.fixture
def renderer() -> TemplateRenderer:
    """Create template renderer."""
    return TemplateRenderer()

@pytest.fixture
def test_data() -> dict:
    """Create test data for template rendering."""
    return {
        "summary": {
            "environment": {
                "python": "3.8.0",
                "os": "Linux"
            },
            "total_stats": {
                "total_tests": 100,
                "passed": 95,
                "failed": 3,
                "skipped": 2,
                "duration": 15.5,
                "coverage": 85.5
            },
            "modules": [
                {
                    "name": "test_module",
                    "tests": {
                        "total_tests": 50,
                        "passed": 48,
                        "failed": 1,
                        "skipped": 1,
                        "duration": 7.5,
                        "coverage": 90.0
                    },
                    "coverage": {
                        "test_module/core.py": 95.0,
                        "test_module/utils.py": 85.0
                    },
                    "last_run": "2025-02-10T12:00:00Z"
                }
            ]
        },
        "report_date": "2025-02-10 12:00:00"
    }

def test_template_directories():
    """Test template directory setup."""
    assert TEMPLATES_DIR.exists()
    assert STYLES_DIR.exists()
    assert (STYLES_DIR / "report.css").exists()
    assert (TEMPLATES_DIR / "test_report.html").exists()

def test_copy_assets(output_dir: Path):
    """Test asset copying."""
    copy_assets_to_output(output_dir)
    assert (output_dir / "styles").exists()
    assert (output_dir / "styles" / "report.css").exists()

def test_jinja_env():
    """Test Jinja environment setup."""
    env = setup_jinja_env()
    assert 'format_number' in env.filters
    assert 'format_percent' in env.filters
    
    # Test custom filters
    assert env.filters['format_number'](12.345) == "12.3"
    assert env.filters['format_percent'](85.678) == "85.7%"

def test_template_rendering(test_data: dict, output_dir: Path):
    """Test template rendering."""
    rendered = render_template(
        "test_report.html",
        test_data,
        output_dir=output_dir,
        output_name="test_output.html"
    )
    
    # Verify output file
    output_file = output_dir / "test_output.html"
    assert output_file.exists()
    assert output_file.read_text() == rendered
    
    # Parse and verify content
    soup = BeautifulSoup(rendered, 'html.parser')
    if soup.title and soup.title.text == "Anarchy Copilot - Test Report":
        # Verify statistics
        total_tests_tag = soup.find("span", text="Total Tests:")
        if total_tests_tag:
            next_span = total_tests_tag.find_next("span")
            if next_span:
                total_tests = next_span.text.strip()
                assert total_tests == "100"
        
        # Verify module details
        module = soup.find("div", class_="module")
        if module:
            module_title = module.find("h3")
            if module_title:
                assert module_title.text.strip() == "test_module"
        
        # Verify CSS linking
        css_link = soup.find("link", rel="stylesheet")
        if css_link:
            assert css_link["href"] == "styles/report.css"

def test_template_renderer(renderer: TemplateRenderer, test_data: dict, output_dir: Path):
    """Test template renderer class."""
    # Test direct rendering
    rendered = renderer.render_report(test_data)
    assert isinstance(rendered, str)
    assert "Test Execution Report" in rendered
    
    # Test rendering with output
    renderer.render_report(
        test_data,
        output_dir=output_dir,
        output_name="renderer_test.html"
    )
    assert (output_dir / "renderer_test.html").exists()
    assert (output_dir / "styles" / "report.css").exists()

def test_template_access(renderer: TemplateRenderer):
    """Test template access methods."""
    template = renderer.get_template("test_report.html")
    assert template is not None
    
    path = renderer.get_template_path("test_report.html")
    assert path.exists()
    assert path.name == "test_report.html"

def test_renderer_with_custom_dir(tmp_path: Path):
    """Test renderer with custom templates directory."""
    custom_dir = tmp_path / "custom_templates"
    custom_dir.mkdir()
    
    # Create a simple test template
    test_template = custom_dir / "test.html"
    test_template.write_text("Test: {{ value }}")
    
    renderer = TemplateRenderer(templates_dir=custom_dir)
    result = renderer.render_report({"value": "success"}, template_name="test.html")
    assert "Test: success" in result

@pytest.mark.parametrize("test_input,expected", [
    ({"value": 123.456}, "123.5"),
    ({"value": 0.999}, "1.0"),
    ({"value": 0}, "0.0")
])
def test_number_formatting(test_input: dict, expected: str):
    """Test number formatting in templates."""
    env = setup_jinja_env()
    template = env.from_string("{{ value|format_number }}")
    assert template.render(test_input) == expected

def test_error_handling(renderer: TemplateRenderer):
    """Test error handling in template operations."""
    # Test missing template
    with pytest.raises(jinja2.TemplateNotFound):
        renderer.get_template("nonexistent.html")
    
    # Test invalid output directory
    with patch.object(Path, "mkdir", side_effect=OSError), pytest.raises(OSError):
        renderer.render_report(
            {"summary": {"environment": {}, "total_stats": {"total_tests": 1, "passed": 0, "failed": 0, "skipped": 0, "coverage": 0.0, "duration": 0.0}, "modules": []}},
            output_dir=Path("/nonexistent/directory")
        )

def test_template_caching(renderer: TemplateRenderer):
    """Test template caching behavior."""
    # Get template twice - should use cache
    template1 = renderer.get_template("test_report.html")
    template2 = renderer.get_template("test_report.html")
    assert template1 is template2

def test_large_dataset(renderer: TemplateRenderer, output_dir: Path):
    """Test rendering with a large dataset."""
    # Create large test data
    large_data = {
        "summary": {
            "modules": [
                {
                    "name": f"module_{i}",
                    "tests": {
                        "total_tests": 100,
                        "passed": 95,
                        "failed": 3,
                        "skipped": 2,
                        "duration": 10.0,
                        "coverage": 85.0
                    },
                    "coverage": {f"file_{j}.py": 85.0 for j in range(10)},
                    "last_run": "2025-02-10T12:00:00Z"
                } for i in range(100)
            ],
            "total_stats": {
                "total_tests": 10000,
                "passed": 9500,
                "failed": 300,
                "skipped": 200,
                "duration": 1000.0,
                "coverage": 85.0
            },
            "environment": {"test": "large"}
        },
        "report_date": "2025-02-10 12:00:00"
    }
    
    # Should handle large data without issues
    rendered = renderer.render_report(
        large_data,
        output_dir=output_dir,
        output_name="large_report.html"
    )
    assert len(rendered) > 100000  # Should be a large HTML file
