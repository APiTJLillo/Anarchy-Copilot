"""Tests for example asset utilities."""

import pytest
from pathlib import Path
import json
import yaml
from typing import Dict, Any
from bs4 import BeautifulSoup

from .. import assets
from ..assets import (
    ExampleAssets,
    load_example_config,
    load_example_findings,
    generate_report,
    merge_configs,
    is_asset_file
)

@pytest.fixture
def example_report_dir(temp_dir: Path) -> Path:
    """Create temporary directory for test reports."""
    report_dir = temp_dir / "reports"
    report_dir.mkdir()
    return report_dir

def test_asset_paths():
    """Test that asset paths are correct."""
    assert assets.ASSETS_DIR.exists()
    assert assets.CONFIGS_DIR.exists()
    assert assets.DATA_DIR.exists()
    assert assets.TEMPLATES_DIR.exists()

def test_load_example_config():
    """Test loading example scan configuration."""
    config = load_example_config()
    assert isinstance(config, dict)
    assert "name" in config
    assert "targets" in config
    assert "scan" in config

def test_load_nonexistent_config():
    """Test loading non-existent configuration."""
    with pytest.raises(FileNotFoundError):
        load_example_config("nonexistent.yaml")

def test_load_example_findings():
    """Test loading example findings data."""
    findings = load_example_findings()
    assert isinstance(findings, dict)
    assert "scan_id" in findings
    assert "findings" in findings
    assert isinstance(findings["findings"], list)

def test_generate_report(example_report_dir: Path):
    """Test report generation."""
    findings = load_example_findings()
    report_path = generate_report(
        findings,
        output_dir=example_report_dir,
        output_name="test_report.html"
    )
    
    assert report_path.exists()
    
    # Verify report content
    content = report_path.read_text()
    soup = BeautifulSoup(content, 'html.parser')
    
    # Check basic structure
    assert soup.title.text.strip() == f"{findings['metadata']['scan_config']['name']} - Scan Report"
    assert soup.find("div", class_="header")
    
    # Check findings are rendered
    finding_divs = soup.find_all("div", class_="finding")
    assert len(finding_divs) == len(findings["findings"])
    
    # Check severity classes
    severity_classes = [div['class'] for div in finding_divs]
    assert any('critical' in classes for classes in severity_classes)
    assert any('high' in classes for classes in severity_classes)

def test_merge_configs():
    """Test configuration merging."""
    base_config = {
        "name": "base",
        "settings": {
            "timeout": 30,
            "retries": 3
        }
    }
    
    override_config = {
        "settings": {
            "timeout": 60,
            "new_setting": "value"
        },
        "additional": "setting"
    }
    
    merged = merge_configs(base_config, override_config)
    assert merged["name"] == "base"
    assert merged["settings"]["timeout"] == 60
    assert merged["settings"]["retries"] == 3
    assert merged["settings"]["new_setting"] == "value"
    assert merged["additional"] == "setting"

def test_example_assets_helper():
    """Test ExampleAssets helper class."""
    helper = ExampleAssets()
    
    # Test asset path retrieval
    config_path = helper.get_asset_path("configs", "example_scan.yaml")
    assert config_path.exists()
    assert config_path.name == "example_scan.yaml"
    
    # Test asset listing
    assets = helper.list_assets()
    assert "configs" in assets
    assert "data" in assets
    assert "templates" in assets
    assert "example_scan.yaml" in assets["configs"]
    
    # Test asset creation
    test_data = helper.create_example_data()
    assert test_data["metadata"]["test"] is True
    assert test_data["metadata"]["generated"] is True

def test_invalid_asset_category():
    """Test handling of invalid asset category."""
    with pytest.raises(ValueError):
        ExampleAssets.get_asset_path("invalid", "file.txt")

def test_is_asset_file():
    """Test asset file detection."""
    # Test valid asset paths
    assert is_asset_file(assets.CONFIGS_DIR / "example_scan.yaml")
    assert is_asset_file(assets.DATA_DIR / "example_findings.json")
    
    # Test invalid paths
    assert not is_asset_file(Path("/tmp/not_an_asset.txt"))
    assert not is_asset_file(Path("relative/path.txt"))

def test_report_generation_with_missing_template():
    """Test report generation with missing template."""
    findings = load_example_findings()
    with pytest.raises(FileNotFoundError):
        generate_report(findings, template_name="nonexistent.html")

def test_empty_findings_report(example_report_dir: Path):
    """Test report generation with empty findings."""
    findings = {
        "scan_id": "empty-test",
        "findings": [],
        "metadata": {
            "scan_config": {
                "name": "empty-scan"
            }
        }
    }
    
    report_path = generate_report(findings, output_dir=example_report_dir)
    assert report_path.exists()
    
    content = report_path.read_text()
    assert "empty-scan" in content
    assert "Total Findings: 0" in content

@pytest.mark.parametrize("config_updates,expected", [
    ({"scan": {"timeout": 60}}, 60),
    ({"scan": {"concurrency": 20}}, 20),
    ({"scan": {"templates": ["new"]}}, ["new"]),
])
def test_config_updates(config_updates: Dict[str, Any], expected: Any):
    """Test updating configurations with different values."""
    base_config = load_example_config()
    merged = merge_configs(base_config, config_updates)
    
    # Navigate to the updated value
    value = merged
    for key in config_updates.keys():
        value = value[key]
    for key in list(config_updates.values())[0].keys():
        actual = value[key]
        
    assert actual == expected
