"""Tests for Nuclei scanner implementation."""

import pytest
import os
import tempfile
from pathlib import Path
import yaml
import json
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from vuln_module.models import (
    ScanConfig,
    PayloadType,
    VulnSeverity,
    PayloadResult
)
from vuln_module.scanner.nuclei.scanner import NucleiScanner

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
@pytest.mark.timeout(10)
async def test_scanner_setup_cleanup(nuclei_scanner):
    """Test scanner setup and cleanup."""
    mock_process = AsyncMock()
    mock_process.stop = AsyncMock()
    
    temp_dir = Path(tempfile.mkdtemp())
    try:
        templates_dir = temp_dir / "templates"
        templates_dir.mkdir(parents=True)

        with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
             patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(temp_dir)):

            await nuclei_scanner.setup()
            
            # Verify directories were created
            assert nuclei_scanner._templates_dir == temp_dir
            assert templates_dir.exists()
            
            # Verify process setup
            assert nuclei_scanner._process == mock_process
            
            await nuclei_scanner.cleanup()
            mock_process.stop.assert_awaited_once()
            
    finally:
        # Clean up temp directory even if test fails
        import shutil
        shutil.rmtree(str(temp_dir))

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_scan_target_process(nuclei_scanner):
    """Test scan execution process."""
    mock_process = AsyncMock()
    mock_process.run = AsyncMock()
    mock_process.stop = AsyncMock()
    async def async_result_iter():
        yield {
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

    mock_process.read_results = AsyncMock(return_value=async_result_iter())

    # Create temp output file
    output_file = Path(tempfile.mktemp())
    output_file.touch()
    process_mock = AsyncMock(return_value=output_file)

    with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
         patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(output_file.parent)), \
         patch('pathlib.Path.exists', return_value=True):
        await nuclei_scanner.setup()
        try:
            results = []
            async for result in nuclei_scanner._scan_target():
                results.append(result)
            
            assert len(results) == 1
            result = results[0]
            assert result.name == "test-xss"
            assert result.severity == VulnSeverity.HIGH
            assert len(result.payloads) == 1

            # Verify process methods were called
            mock_process.run.assert_awaited_once()
            mock_process.read_results.assert_awaited_once()
        finally:
            await nuclei_scanner.cleanup()
            mock_process.stop.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_payload_testing(nuclei_scanner):
    """Test individual payload testing."""
    mock_process = AsyncMock()
    mock_process.run = AsyncMock()
    async def async_payload_iter():
        yield {
            "template-id": "test-xss",
            "type": "http",
            "severity": "high",
            "matched": "<script>alert(1)</script>",
            "matched-at": "http://example.com/test"
        }

    mock_process.read_results = AsyncMock(return_value=async_payload_iter())
    mock_process.stop = AsyncMock()

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create template dir
        templates_dir = temp_dir / "templates"
        templates_dir.mkdir(parents=True)

        # Create output file
        output_file = temp_dir / "output.json"
        output_file.touch()

        with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
             patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(temp_dir)):

            # Mock file operations for template creation
            with patch('pathlib.Path.write_text'):
                result = await nuclei_scanner.test_payload(
                    "<script>alert(1)</script>",
                    PayloadType.XSS
                )
                assert isinstance(result, PayloadResult)
                assert result.success
                assert result.payload.type == PayloadType.XSS
                assert result.payload.content == "<script>alert(1)</script>"

                # Verify process was cleaned up
                mock_process.run.assert_awaited_once()
                mock_process.read_results.assert_awaited_once()
                mock_process.stop.assert_awaited_once()

    finally:
        # Clean up test directory
        import shutil
        shutil.rmtree(str(temp_dir))

@pytest.mark.asyncio
async def test_scan_rate_limiting(nuclei_scanner):
    """Test rate limiting functionality."""
    time_values = [0.0]

    def mock_time():
        return time_values[0]

    def mock_sleep(seconds):
        time_values[0] += seconds
        return seconds

    with patch('vuln_module.rate_limiter.datetime') as mock_datetime, \
         patch('asyncio.sleep', side_effect=mock_sleep):
        
        # Configure mock datetime
        mock_datetime.now.side_effect = lambda: type(
            'MockDateTime', (), {'timestamp': lambda: mock_time()}
        )()
        
        # With rate limit of 2/sec, 3 requests should take 1 second
        nuclei_scanner.config.rate_limit = 2
        
        # Make multiple requests - should be rate limited
        for _ in range(3):
            await nuclei_scanner.rate_limit()

        # Should have taken exactly 1.0 seconds
        assert time_values[0] == 1.0

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_scanner_error_handling(nuclei_scanner):
    """Test error handling during scanning."""
    mock_process = AsyncMock()
    mock_process.run = AsyncMock(side_effect=Exception("Test error"))
    async def empty_iter():
        if False:  # Never yield anything
            yield

    mock_process.read_results = AsyncMock(return_value=empty_iter())
    mock_process.stop = AsyncMock()

    temp_dir = Path(tempfile.mkdtemp())
    try:
        # Create template dir
        templates_dir = temp_dir / "templates"
        templates_dir.mkdir(parents=True)

        with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
             patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(temp_dir)), \
             patch('pathlib.Path.write_text'):  # Mock template writing
            await nuclei_scanner.setup()
            try:
                results = []
                async for result in nuclei_scanner._scan_target():
                    results.append(result)
                assert len(results) == 0  # Should handle error gracefully

                # Verify run was attempted and cleanup happened
                mock_process.run.assert_awaited_once()
                mock_process.stop.assert_awaited_once()
            finally:
                await nuclei_scanner.cleanup()
    finally:
        # Clean up test directory
        import shutil
        shutil.rmtree(str(temp_dir))

@pytest.mark.timeout(5)  # Enforce quick test execution
def test_payload_type_detection():
    """Test payload type detection from template IDs."""
    test_cases = [
        ("wordpress-xss", PayloadType.XSS),
        ("mysql-sqli", PayloadType.SQLI),
        ("path-traversal", PayloadType.PATH_TRAVERSAL),
        ("ssrf-detect", PayloadType.SSRF),
        ("unknown-template", PayloadType.CUSTOM)
    ]
    
    from vuln_module.scanner.nuclei.results import NucleiResultParser
    for template_id, expected_type in test_cases:
        result = NucleiResultParser._get_payload_type({"template-id": template_id})
        assert result == expected_type
