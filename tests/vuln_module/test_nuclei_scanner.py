"""Tests for Nuclei scanner implementation."""

import pytest
import os
import tempfile
from pathlib import Path
import yaml
import json
from unittest.mock import Mock, patch, AsyncMock

from anarchy_copilot.vuln_module.models import (
    ScanConfig,
    PayloadType,
    VulnSeverity,
    PayloadResult
)
from anarchy_copilot.vuln_module.scanner.nuclei import NucleiScanner

@pytest.fixture
def scan_config():
    """Create a test scan configuration."""
    return ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS, PayloadType.SQLI},
        max_depth=2,
        threads=1,
        timeout=5,
        rate_limit=10
    )

@pytest.fixture
def nuclei_scanner(scan_config):
    """Create a NucleiScanner instance."""
    scanner = NucleiScanner(scan_config)
    return scanner

@pytest.mark.asyncio
async def test_scanner_setup_cleanup(nuclei_scanner):
    """Test scanner setup and cleanup."""
    await nuclei_scanner.setup()
    assert nuclei_scanner._templates_dir is not None
    assert os.path.exists(nuclei_scanner._templates_dir)
    assert os.path.exists(nuclei_scanner._output_file)
    
    await nuclei_scanner.cleanup()
    assert not os.path.exists(nuclei_scanner._templates_dir)

@pytest.mark.asyncio
async def test_template_generation(nuclei_scanner):
    """Test custom template generation."""
    await nuclei_scanner.setup()
    try:
        templates = await nuclei_scanner._generate_templates()
        assert len(templates) > 0
        
        # Check template content
        with open(templates[0], 'r') as f:
            content = yaml.safe_load(f)
            assert "id" in content
            assert "info" in content
            assert "requests" in content
    finally:
        await nuclei_scanner.cleanup()

@pytest.mark.asyncio
async def test_scan_target_process(nuclei_scanner):
    """Test scan execution process."""
    # Mock subprocess for testing
    mock_process = AsyncMock()
    mock_process.stdout.readline = AsyncMock(return_value=b'')
    mock_process.wait = AsyncMock(return_value=0)

    with patch('asyncio.create_subprocess_exec', return_value=mock_process):
        await nuclei_scanner.setup()
        try:
            # Write test result to output file
            result_data = {
                "template-id": "test-xss",
                "type": "http",
                "severity": "high",
                "info": {"description": "Test XSS"},
                "matched-at": "http://example.com/test",
                "matcher-name": "<script>alert(1)</script>",
                "extracted-values": {},
                "ip": "1.2.3.4",
                "host": "example.com",
                "request": "GET /test",
                "response": "<html>test</html>"
            }
            
            with open(nuclei_scanner._output_file, 'w') as f:
                json.dump(result_data, f)
                f.write('\n')
            
            results = []
            async for result in nuclei_scanner._scan_target():
                results.append(result)
            
            assert len(results) == 1
            result = results[0]
            assert result.name == "test-xss"
            assert result.severity == VulnSeverity.HIGH
            assert len(result.payloads) == 1
        finally:
            await nuclei_scanner.cleanup()

@pytest.mark.asyncio
async def test_payload_testing(nuclei_scanner):
    """Test individual payload testing."""
    mock_process = AsyncMock()
    mock_process.stdout.readline = AsyncMock(return_value=b'{"matcher-status": true}')
    mock_process.wait = AsyncMock(return_value=0)
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b'{"matcher-status": true}', b''))

    with patch('asyncio.create_subprocess_exec', return_value=mock_process):
        result = await nuclei_scanner.test_payload(
            "<script>alert(1)</script>",
            PayloadType.XSS
        )
        assert isinstance(result, PayloadResult)
        assert result.success
        assert result.payload.type == PayloadType.XSS

@pytest.mark.asyncio
async def test_scan_rate_limiting(nuclei_scanner):
    """Test rate limiting functionality."""
    start_time = pytest.helpers.time()
    
    # Set a very low rate limit
    nuclei_scanner.config.rate_limit = 2  # 2 requests per second
    
    # Make multiple requests
    for _ in range(3):
        await nuclei_scanner.rate_limit()
    
    duration = pytest.helpers.time() - start_time
    assert duration >= 1.0  # Should take at least 1 second due to rate limiting

def test_severity_mapping(nuclei_scanner):
    """Test severity mapping from Nuclei to internal representation."""
    assert nuclei_scanner.SEVERITY_MAP["info"] == VulnSeverity.INFO
    assert nuclei_scanner.SEVERITY_MAP["low"] == VulnSeverity.LOW
    assert nuclei_scanner.SEVERITY_MAP["medium"] == VulnSeverity.MEDIUM
    assert nuclei_scanner.SEVERITY_MAP["high"] == VulnSeverity.HIGH
    assert nuclei_scanner.SEVERITY_MAP["critical"] == VulnSeverity.CRITICAL

def test_payload_type_detection(nuclei_scanner):
    """Test payload type detection from template IDs."""
    test_cases = [
        ("wordpress-xss", PayloadType.XSS),
        ("mysql-sqli", PayloadType.SQLI),
        ("path-traversal", PayloadType.PATH_TRAVERSAL),
        ("ssrf-detect", PayloadType.SSRF),
        ("unknown-template", PayloadType.CUSTOM)
    ]
    
    for template_id, expected_type in test_cases:
        result = nuclei_scanner._get_payload_type({"template-id": template_id})
        assert result == expected_type

@pytest.mark.asyncio
async def test_scanner_error_handling(nuclei_scanner):
    """Test error handling during scanning."""
    # Simulate a process error
    mock_process = AsyncMock()
    mock_process.wait = AsyncMock(side_effect=Exception("Test error"))
    
    with patch('asyncio.create_subprocess_exec', return_value=mock_process):
        await nuclei_scanner.setup()
        try:
            results = []
            async for result in nuclei_scanner._scan_target():
                results.append(result)
            assert len(results) == 0  # Should handle error gracefully
        finally:
            await nuclei_scanner.cleanup()
