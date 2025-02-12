"""Tests for Nuclei scanner implementation."""
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from vuln_module.scanner.nuclei.scanner import NucleiScanner
from vuln_module.models import PayloadType, ScanConfig, VulnSeverity

@pytest.fixture
def nuclei_scanner():
    """Create test scanner instance."""
    config = ScanConfig(
        target="http://example.com",
        payload_types={PayloadType.XSS, PayloadType.SQLI}
    )
    return NucleiScanner(config)

@pytest.mark.asyncio
async def test_scanner_setup_cleanup(nuclei_scanner, test_templates_dir, test_output_file):
    """Test scanner initialization and cleanup."""
    mock_process = AsyncMock()
    mock_process.is_running = True
    mock_process.set_output_file = AsyncMock()
    mock_process.stop = AsyncMock()

    with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
         patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(test_templates_dir.parent)):
        await nuclei_scanner.setup()
        
        assert nuclei_scanner._process is not None
        assert nuclei_scanner._templates_dir is not None
        assert len(nuclei_scanner._custom_templates) > 0
        
        await nuclei_scanner.cleanup()
        mock_process.stop.assert_awaited_once()
        
        assert nuclei_scanner._process is None
        assert nuclei_scanner._templates_dir is None
        assert len(nuclei_scanner._custom_templates) == 0

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_scan_target_process(nuclei_scanner, mock_scan_results, test_templates_dir, async_gen):
    """Test scan execution process."""
    # Create mock process with async iterator
    mock_process = AsyncMock()
    mock_process.is_running = True
    mock_process.run = AsyncMock()
    mock_process.stop = AsyncMock()
    mock_process.read_results = lambda: async_gen(mock_scan_results)

    with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
         patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(test_templates_dir.parent)):
        await nuclei_scanner.setup()
        try:
            results = []
            async for result in nuclei_scanner._scan_target():
                results.append(result)
            
            assert len(results) == 1
            assert results[0].severity == VulnSeverity.HIGH
            assert "<script>alert(1)</script>" in str(results[0].payloads[0].payload.content)
            
            mock_process.run.assert_awaited_once()
            # No need to assert read_results was awaited since it returns an iterator
        finally:
            await nuclei_scanner.cleanup()

@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_payload_testing(nuclei_scanner, mock_scan_results, test_templates_dir, async_gen):
    """Test individual payload testing."""
    # Create mock process with async iterator
    mock_process = AsyncMock()
    mock_process.is_running = True
    mock_process.run = AsyncMock()
    mock_process.set_output_file = AsyncMock()
    mock_process.stop = AsyncMock()
    mock_process.read_results = lambda: async_gen(mock_scan_results)

    mock_file = AsyncMock()
    mock_file.__aenter__.return_value = mock_file
    mock_file.write = AsyncMock()
    
    with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
         patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(test_templates_dir.parent)), \
         patch('aiofiles.open', return_value=mock_file):
            
        result = await nuclei_scanner.test_payload(
            "<script>alert(1)</script>",
            PayloadType.XSS
        )
                
        assert result.success is True
        assert result.error is None
        # Verify process was used correctly
        mock_process.run.assert_awaited_once()

@pytest.mark.asyncio
async def test_scan_rate_limiting(nuclei_scanner, mock_time, mock_rate_limiter):
    """Test rate limiting functionality."""
    nuclei_scanner._rate_limiter = mock_rate_limiter(rate=2)  # 2 requests per second
    
    # Make multiple requests - should be rate limited
    for _ in range(3):
        await nuclei_scanner.rate_limit()
        mock_time.advance(0.5)  # Advance time by 0.5 seconds
        
    assert mock_time.timestamp() >= 1.0  # At least 1 second should have passed

@pytest.mark.asyncio 
@pytest.mark.timeout(10)
async def test_scanner_error_handling(nuclei_scanner, test_templates_dir):
    """Test error handling during scanning."""
    mock_process = AsyncMock()
    mock_process.is_running = True
    mock_process.run = AsyncMock(side_effect=RuntimeError("Test error"))
    mock_process.stop = AsyncMock()
    
    with patch('vuln_module.scanner.nuclei.scanner.NucleiProcess', return_value=mock_process), \
         patch('vuln_module.scanner.nuclei.scanner.tempfile.mkdtemp', return_value=str(test_templates_dir.parent)):
        await nuclei_scanner.setup()
        with pytest.raises(RuntimeError, match="Test error"):
            async for _ in nuclei_scanner._scan_target():
                pass
        await nuclei_scanner.cleanup()
        mock_process.stop.assert_awaited_once()

@pytest.mark.asyncio
async def test_payload_type_detection(nuclei_scanner):
    """Test correct template generation for different payload types."""
    test_cases = [
        (PayloadType.XSS, "<script>alert(1)</script>"),
        (PayloadType.SQLI, "' OR '1'='1"),
    ]
    
    for payload_type, payload in test_cases:
        template = nuclei_scanner._create_template(payload_type, [payload])
        assert template is not None
        template_yaml = template.to_yaml()
        assert payload in template_yaml
