"""Tests for performance test runner."""
import pytest
import asyncio
import tempfile
import json
from pathlib import Path
import yaml
from unittest.mock import Mock, patch

from .run_perf_tests import PerformanceTestRunner
from .perf_config import PerformanceSettings

@pytest.fixture
def mock_results():
    """Mock performance test results."""
    return {
        "throughput": 1000.0,
        "latency_p95": 0.01,
        "memory_usage": 100 * 1024 * 1024,  # 100MB
        "execution_times": {
            "test1": [0.001, 0.002, 0.003],
            "test2": [0.002, 0.003, 0.004]
        }
    }

@pytest.fixture
def test_config():
    """Create test configuration."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
        config = {
            "concurrent_connections": 10,
            "request_count": 100,
            "min_throughput": 500.0,
            "data_sizes": [1024],
            "save_raw_data": True
        }
        yaml.dump(config, f)
    yield Path(f.name)
    Path(f.name).unlink()

@pytest.fixture
async def runner(tmp_path, test_config):
    """Create test runner instance."""
    runner = PerformanceTestRunner(test_config)
    runner.results_dir = tmp_path / "results"
    runner.results_dir.mkdir()
    return runner

@pytest.mark.asyncio
async def test_run_tests_success(runner, mock_results):
    """Test successful test execution."""
    with patch('tests.proxy.server.handlers.test_middleware_performance.run_performance_test',
              return_value=mock_results):
        results = await runner.run_tests()
        
        assert "results" in results
        assert "timestamp" in results
        assert results["results"]["throughput"] == 1000.0
        
        # Check report generation
        results_file = list(runner.results_dir.glob("results_*.json"))
        assert len(results_file) == 1
        
        summary_file = list(runner.results_dir.glob("summary_*.txt"))
        assert len(summary_file) == 1
        
        # Verify summary content
        with open(summary_file[0]) as f:
            summary = f.read()
            assert "Performance Test Summary" in summary
            assert "Throughput: 1000.00 req/s" in summary

@pytest.mark.asyncio
async def test_run_tests_with_regressions(runner, mock_results):
    """Test handling of performance regressions."""
    # Mock baseline with better performance
    baseline_results = {
        "throughput": 2000.0,  # 2x better
        "latency_p95": 0.005,  # 2x better
        "memory_usage": 50 * 1024 * 1024  # Half the memory
    }
    
    with patch('tests.proxy.server.handlers.test_middleware_performance.run_performance_test',
              return_value=mock_results), \
         patch.object(runner.regression_tester, 'get_baseline',
                     return_value=baseline_results):
        results = await runner.run_tests()
        
        assert "regressions" in results
        assert len(results["regressions"]) > 0
        
        # Check regression reporting
        summary_file = list(runner.results_dir.glob("summary_*.txt"))
        with open(summary_file[0]) as f:
            summary = f.read()
            assert "Performance Regressions" in summary
            assert "degradation" in summary

@pytest.mark.asyncio
async def test_run_tests_error_handling(runner):
    """Test error handling during test execution."""
    with patch('tests.proxy.server.handlers.test_middleware_performance.run_performance_test',
              side_effect=Exception("Test error")):
        with pytest.raises(Exception) as exc_info:
            await runner.run_tests()
        assert "Test error" in str(exc_info.value)

def test_environment_preparation(runner):
    """Test test environment preparation."""
    with patch('gc.disable') as mock_gc_disable, \
         patch('os.nice') as mock_nice:
        
        # Test with GC disabled and process priority set
        runner.config.disable_gc = True
        runner.config.process_priority = -10
        
        runner._prepare_environment()
        
        mock_gc_disable.assert_called_once()
        mock_nice.assert_called_once_with(-10)

def test_report_generation(runner, mock_results, tmp_path):
    """Test report generation."""
    runner._generate_reports(mock_results, {})
    
    # Check for report files
    results_file = list(runner.results_dir.glob("results_*.json"))
    assert len(results_file) == 1
    
    summary_file = list(runner.results_dir.glob("summary_*.txt"))
    assert len(summary_file) == 1
    
    # Verify results content
    with open(results_file[0]) as f:
        saved_results = json.load(f)
        assert saved_results["results"] == mock_results
        assert "regressions" in saved_results

def test_command_line_interface():
    """Test CLI argument handling."""
    with patch('argparse.ArgumentParser.parse_args') as mock_parse_args, \
         patch('tests.proxy.server.handlers.run_perf_tests.PerformanceTestRunner') as MockRunner:
        
        # Test with custom config
        mock_parse_args.return_value = Mock(
            config=Path("custom_config.yml"),
            profile="development",
            update_baseline=False
        )
        
        # Mock runner instance
        mock_runner = Mock()
        MockRunner.return_value = mock_runner
        mock_runner.run_tests.return_value = {"results": {}, "regressions": {}}
        
        # Run main function
        from .run_perf_tests import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Check exit code
        assert exc_info.value.code == 0
        
        # Verify runner was created with correct config
        MockRunner.assert_called_once_with(Path("custom_config.yml"))

def test_missing_config_handling():
    """Test handling of missing configuration file."""
    with patch('argparse.ArgumentParser.parse_args') as mock_parse_args:
        mock_parse_args.return_value = Mock(
            config=Path("nonexistent.yml"),
            profile="development",
            update_baseline=False
        )
        
        # Run main function
        from .run_perf_tests import main
        with pytest.raises(SystemExit) as exc_info:
            main()
        
        # Check exit code
        assert exc_info.value.code == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
