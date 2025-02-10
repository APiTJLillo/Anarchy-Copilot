"""Tests for template validation tool."""

import pytest
from pathlib import Path
import yaml
import tempfile
import shutil
import subprocess
from unittest.mock import patch, MagicMock

from .validate_templates import TemplateValidator

@pytest.fixture
def temp_templates_dir():
    """Create a temporary directory with test templates."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create index file
        index = {
            "templates": [
                {
                    "id": "test-template",
                    "type": "vulnerability",
                    "info": {
                        "name": "Test Template",
                        "severity": "high"
                    }
                }
            ],
            "template_metadata": {
                "requirements": {
                    "nuclei_version": ">=2.8.0"
                }
            }
        }
        with open(tmp_path / "template_index.yaml", "w") as f:
            yaml.dump(index, f)
        
        # Create test template
        template = {
            "id": "test-template",
            "info": {
                "name": "Test Template",
                "severity": "high"
            },
            "requests": [
                {
                    "method": "GET",
                    "path": ["{{BaseURL}}/test"],
                    "matchers": [{
                        "type": "word",
                        "words": ["test"]
                    }]
                }
            ]
        }
        with open(tmp_path / "test-template.yaml", "w") as f:
            yaml.dump(template, f)
            
        yield tmp_path

def test_validator_initialization(temp_templates_dir):
    """Test validator initialization."""
    validator = TemplateValidator(temp_templates_dir)
    assert validator.templates_dir == temp_templates_dir
    assert validator.index is not None
    assert "templates" in validator.index

def test_verify_nuclei_installation(temp_templates_dir):
    """Test nuclei version verification."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="nuclei version 2.9.0"
        )
        
        validator = TemplateValidator(temp_templates_dir)
        assert validator.verify_nuclei_installation()

def test_validate_template(temp_templates_dir):
    """Test template validation."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Template validated successfully"
        )
        
        validator = TemplateValidator(temp_templates_dir)
        assert validator.validate_template(
            temp_templates_dir / "test-template.yaml"
        )

def test_validate_invalid_template(temp_templates_dir):
    """Test validation of invalid template."""
    # Create invalid template
    invalid_template = {
        "invalid": "template"
    }
    invalid_path = temp_templates_dir / "invalid.yaml"
    with open(invalid_path, "w") as f:
        yaml.dump(invalid_template, f)
    
    validator = TemplateValidator(temp_templates_dir)
    assert not validator.validate_template(invalid_path)

def test_check_template_consistency(temp_templates_dir):
    """Test template consistency check."""
    validator = TemplateValidator(temp_templates_dir)
    assert validator.check_template_consistency()
    
    # Add unindexed template
    unindexed = {
        "id": "unindexed",
        "info": {"name": "Unindexed"}
    }
    with open(temp_templates_dir / "unindexed.yaml", "w") as f:
        yaml.dump(unindexed, f)
    
    assert not validator.check_template_consistency()

@pytest.mark.integration
def test_real_templates():
    """Test validation of actual test templates."""
    templates_dir = Path(__file__).parent.parent / "data/nuclei_templates"
    
    try:
        result = subprocess.run(
            ["nuclei", "-version"],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            pytest.skip("Nuclei not installed")
    except FileNotFoundError:
        pytest.skip("Nuclei not installed")
    
    validator = TemplateValidator(templates_dir)
    assert validator.check_template_consistency()
    assert validator.validate_all()

def test_invalid_index_file():
    """Test handling of invalid index file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create invalid index
        with open(tmp_path / "template_index.yaml", "w") as f:
            f.write("invalid: yaml: :")
        
        with pytest.raises(SystemExit):
            TemplateValidator(tmp_path)

def test_missing_index_file():
    """Test handling of missing index file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with pytest.raises(SystemExit):
            TemplateValidator(tmp_path)

@pytest.mark.parametrize("template_content,expected_valid", [
    # Valid template
    ({
        "id": "test",
        "info": {"name": "Test"},
        "requests": [{"method": "GET", "path": ["test"]}]
    }, True),
    # Missing ID
    ({
        "info": {"name": "Test"},
        "requests": [{"method": "GET", "path": ["test"]}]
    }, False),
    # Missing info
    ({
        "id": "test",
        "requests": [{"method": "GET", "path": ["test"]}]
    }, False),
])
def test_template_validation_cases(temp_templates_dir, template_content, expected_valid):
    """Test various template validation cases."""
    template_path = temp_templates_dir / "test-case.yaml"
    with open(template_path, "w") as f:
        yaml.dump(template_content, f)
    
    validator = TemplateValidator(temp_templates_dir)
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0 if expected_valid else 1
        )
        assert validator.validate_template(template_path) == expected_valid
