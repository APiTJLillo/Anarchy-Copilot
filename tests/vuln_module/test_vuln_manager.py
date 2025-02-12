"""Tests for vulnerability manager."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime
from typing import AsyncGenerator

from vuln_module.vuln_manager import VulnManager
from vuln_module.models import (
    VulnResult,
    VulnSeverity,
    PayloadType,
    ScanConfig,
    PayloadResult,
    Payload
)
from vuln_module.scanner.base import BaseVulnScanner

# Mock scanner for testing
class MockScanner(BaseVulnScanner):
    """Mock scanner implementation for testing."""

    def __init__(self, config: ScanConfig):
        super().__init__(config)
        self.setup_called = False
        self.cleanup_called = False
        self.test_results = []
        self.verify_results = {}

    async def setup(self) -> None:
        self.setup_called = True

    async def cleanup(self) -> None:
        self.cleanup_called = True

    async def _scan_target(self) -> AsyncGenerator[VulnResult, None]:
        """Mock scan implementation using async generator."""
        for result in self.test_results:
            yield result

    async def verify_vulnerability(self, result: VulnResult) -> bool:
        return self.verify_results.get(result.name, True)

    async def test_payload(self, payload: str, payload_type: PayloadType) -> PayloadResult:
        return PayloadResult(
            payload=Payload(content=payload, type=payload_type),
            success=True,
            response_data={"test": "response"}
        )

    async def get_supported_payloads(self):
        return {
            PayloadType.XSS: ["test"],
            PayloadType.SQLI: ["test"]
        }

@pytest.fixture
def vuln_manager():
    """Create vulnerability manager instance."""
    manager = VulnManager(project_id=1)
    # Register mock scanner
    manager.register_scanner("mock", MockScanner)
    return manager

@pytest.fixture
def mock_vuln_result():
    """Create a mock vulnerability result."""
    return VulnResult(
        name="Test XSS",
        type="xss",
        severity=VulnSeverity.HIGH,
        description="Test vulnerability",
        endpoint="http://example.com/test",
        payloads=[
            PayloadResult(
                payload=Payload(
                    content="<script>alert(1)</script>",
                    type=PayloadType.XSS
                ),
                success=True,
                response_data={}
            )
        ]
    )

@pytest.mark.asyncio
async def test_scan_target(vuln_manager, mock_vuln_result):
    """Test full vulnerability scan execution."""
    # Configure mock scanner
    scanner = MockScanner(ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS}
    ))
    scanner.test_results = [mock_vuln_result]
    
    # Replace default scanner with our configured mock
    vuln_manager.DEFAULT_SCANNERS["mock"] = lambda config: scanner
    
    # Run scan
    results = await vuln_manager.scan_target(
        target="http://example.com",
        scanner_type="mock"
    )
    
    # Verify results
    assert len(results) == 1
    assert results[0].name == "Test XSS"
    assert results[0].severity == VulnSeverity.HIGH
    assert scanner.setup_called
    assert scanner.cleanup_called

@pytest.mark.asyncio
async def test_scanner_error_handling(vuln_manager):
    """Test handling of scanner errors."""
    bad_scanner = MockScanner(ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS}
    ))

    # Make scanner raise an error using async generator
    async def error_scan() -> AsyncGenerator[VulnResult, None]:
        if False:  # Never yield anything
            yield
        raise Exception("Test error")

    bad_scanner._scan_target = error_scan
    vuln_manager.DEFAULT_SCANNERS["mock"] = lambda config: bad_scanner

    # Scan should not raise exception but return empty results
    results = await vuln_manager.scan_target(
        target="http://example.com",
        scanner_type="mock"
    )
    assert len(results) == 0

def test_scanner_registration(vuln_manager):
    """Test scanner registration process."""
    # Create custom scanner
    class CustomScanner(BaseVulnScanner):
        pass

    # Register scanner
    vuln_manager.register_scanner("custom", CustomScanner)
    
    # Verify registration
    assert "custom" in vuln_manager.DEFAULT_SCANNERS
    assert vuln_manager.DEFAULT_SCANNERS["custom"] == CustomScanner

    # Try registering invalid scanner
    with pytest.raises(ValueError):
        class InvalidScanner:
            pass
        vuln_manager.register_scanner("invalid", InvalidScanner)

@pytest.mark.asyncio
async def test_payload_testing(vuln_manager):
    """Test individual payload testing."""
    result = await vuln_manager.test_payload(
        target="http://example.com",
        payload="<script>alert(1)</script>",
        payload_type=PayloadType.XSS,
        scanner_type="mock"
    )
    
    assert isinstance(result, PayloadResult)
    assert result.success
    assert result.payload.type == PayloadType.XSS
    assert result.payload.content == "<script>alert(1)</script>"

def test_scan_status_tracking(vuln_manager, mock_vuln_result):
    """Test scan status tracking."""
    # Add some test results
    target = "http://example.com"
    vuln_manager._results_cache[target] = [mock_vuln_result]
    
    # Get status
    status = vuln_manager.get_scan_status(target)
    
    # Verify status information
    assert status["total_vulnerabilities"] == 1
    assert status["by_severity"]["HIGH"] == 1
    assert len(status["active_scans"]) == 0

def test_supported_scanners(vuln_manager):
    """Test getting supported scanner information."""
    scanners = vuln_manager.get_supported_scanners()
    
    assert "mock" in scanners
    assert "class" in scanners["mock"]
    assert "description" in scanners["mock"]
    assert "supported_payload_types" in scanners["mock"]

@pytest.mark.asyncio
async def test_concurrent_scans(vuln_manager, mock_vuln_result):
    """Test running multiple concurrent scans."""
    # Configure mock scanner
    scanner = MockScanner(ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS}
    ))
    scanner.test_results = [mock_vuln_result]
    vuln_manager.DEFAULT_SCANNERS["mock"] = lambda config: scanner
    
    # Run multiple scans
    targets = [
        "http://example1.com",
        "http://example2.com",
        "http://example3.com"
    ]
    
    # Start all scans concurrently
    scan_tasks = [
        vuln_manager.scan_target(target, scanner_type="mock")
        for target in targets
    ]
    results = await asyncio.gather(*scan_tasks)
    
    # Verify each scan completed successfully
    for result_set in results:
        assert len(result_set) == 1
        assert result_set[0].name == "Test XSS"

@pytest.mark.asyncio
async def test_scan_config_validation(vuln_manager):
    """Test scan configuration validation."""
    # Test with invalid scanner type
    with pytest.raises(ValueError):
        await vuln_manager.scan_target(
            target="http://example.com",
            scanner_type="invalid_scanner"
        )

    # Test with invalid configuration
    bad_config = ScanConfig(
        target="",  # Invalid empty target
        payload_types=set()  # No payload types specified
    )
    
    with pytest.raises(ValueError):
        await vuln_manager.scan_target(
            target="http://example.com",
            scanner_type="mock",
            config=bad_config
        )
