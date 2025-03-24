"""Tests for system metrics visualization."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
from datetime import datetime, timedelta

from scripts.visualize_system_metrics import MetricsVisualizer

@pytest.fixture
def sample_cpu_data():
    """Generate sample CPU metrics data."""
    base_time = datetime.now()
    times = [base_time + timedelta(seconds=i) for i in range(100)]
    
    data = {
        "timestamp": times,
        "%user": np.random.uniform(0, 70, 100),
        "%system": np.random.uniform(0, 30, 100),
        "%iowait": np.random.uniform(0, 10, 100),
        "%idle": np.random.uniform(0, 20, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_memory_data():
    """Generate sample memory metrics data."""
    base_time = datetime.now()
    times = [base_time + timedelta(seconds=i) for i in range(100)]
    
    data = {
        "timestamp": times,
        "kbmemused": np.random.uniform(1024*1024, 4*1024*1024, 100),  # 1GB to 4GB
        "kbmemfree": np.random.uniform(1024*1024, 2*1024*1024, 100),
        "kbmemcached": np.random.uniform(512*1024, 1024*1024, 100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def temp_metrics_dir():
    """Create temporary directory for metrics files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

def save_sample_data(df: pd.DataFrame, path: Path):
    """Save sample dataframe to CSV."""
    df.to_csv(path, index=False)
    return path

def test_metrics_loading(sample_cpu_data, sample_memory_data, temp_metrics_dir):
    """Test loading metrics data."""
    # Save sample data
    cpu_path = save_sample_data(sample_cpu_data, temp_metrics_dir / "cpu.csv")
    mem_path = save_sample_data(sample_memory_data, temp_metrics_dir / "memory.csv")
    
    visualizer = MetricsVisualizer(temp_metrics_dir / "report.html")
    
    # Test loading
    cpu_df = visualizer.load_cpu_metrics(cpu_path)
    mem_df = visualizer.load_memory_metrics(mem_path)
    
    assert len(cpu_df) == len(sample_cpu_data)
    assert len(mem_df) == len(sample_memory_data)
    assert isinstance(cpu_df['timestamp'].iloc[0], pd.Timestamp)
    assert isinstance(mem_df['timestamp'].iloc[0], pd.Timestamp)

def test_cpu_heatmap_generation(sample_cpu_data):
    """Test CPU usage heatmap creation."""
    visualizer = MetricsVisualizer(Path("dummy.html"))
    matrix = visualizer._create_cpu_heatmap(sample_cpu_data)
    
    assert matrix.shape == (24, 10)  # 24 hours x 10 usage bins
    assert np.all(matrix >= 0)  # All values should be non-negative
    assert matrix.sum() > 0  # Should have some data

def test_performance_insights(sample_cpu_data, sample_memory_data):
    """Test performance insight generation."""
    import plotly.graph_objects as go
    
    # Create test data with known issues
    sample_cpu_data.loc[50:60, '%user'] = 95  # High CPU usage
    sample_memory_data.loc[30:40, 'kbmemused'] = 8 * 1024 * 1024  # High memory usage
    
    visualizer = MetricsVisualizer(Path("dummy.html"))
    fig = go.Figure()
    
    # Add insights
    visualizer._add_performance_insights(fig, sample_cpu_data, sample_memory_data)
    
    # Check annotations
    annotations = fig.layout.annotations
    assert any("High CPU usage" in ann.text for ann in annotations)
    assert any("High memory usage" in ann.text for ann in annotations)

def test_summary_generation(sample_cpu_data, sample_memory_data):
    """Test performance summary generation."""
    visualizer = MetricsVisualizer(Path("dummy.html"))
    summary = visualizer._generate_summary(sample_cpu_data, sample_memory_data)
    
    # Check summary content
    assert "System Performance Summary" in summary
    assert "CPU Usage:" in summary
    assert "Memory Usage:" in summary
    assert "IO Wait:" in summary
    
    # Verify metrics are present
    assert "Average:" in summary
    assert "Peak:" in summary
    assert "95th percentile:" in summary

def test_visualization_creation(sample_cpu_data, sample_memory_data, temp_metrics_dir):
    """Test full visualization creation."""
    output_path = temp_metrics_dir / "report.html"
    visualizer = MetricsVisualizer(output_path)
    
    # Create visualization
    result_path = visualizer.create_visualizations(sample_cpu_data, sample_memory_data)
    
    assert Path(result_path).exists()
    assert Path(result_path).suffix == ".html"
    
    # Check for summary file
    summary_path = output_path.parent / "metrics_summary.txt"
    assert summary_path.exists()

def test_error_handling(temp_metrics_dir):
    """Test error handling in visualization."""
    visualizer = MetricsVisualizer(temp_metrics_dir / "report.html")
    
    # Test with invalid data
    invalid_df = pd.DataFrame({"invalid": [1, 2, 3]})
    
    with pytest.raises(KeyError):
        visualizer.create_visualizations(invalid_df, invalid_df)

def test_large_dataset_handling(temp_metrics_dir):
    """Test handling of large datasets."""
    # Generate large dataset
    base_time = datetime.now()
    times = [base_time + timedelta(seconds=i) for i in range(10000)]
    
    large_cpu_data = pd.DataFrame({
        "timestamp": times,
        "%user": np.random.uniform(0, 100, 10000),
        "%system": np.random.uniform(0, 100, 10000),
        "%iowait": np.random.uniform(0, 100, 10000),
        "%idle": np.random.uniform(0, 100, 10000)
    })
    
    large_memory_data = pd.DataFrame({
        "timestamp": times,
        "kbmemused": np.random.uniform(1024*1024, 8*1024*1024, 10000),
        "kbmemfree": np.random.uniform(1024*1024, 4*1024*1024, 10000),
        "kbmemcached": np.random.uniform(512*1024, 2*1024*1024, 10000)
    })
    
    visualizer = MetricsVisualizer(temp_metrics_dir / "report.html")
    result_path = visualizer.create_visualizations(large_cpu_data, large_memory_data)
    
    assert Path(result_path).exists()
    assert Path(result_path).stat().st_size > 0

def test_resource_cleanup(temp_metrics_dir):
    """Test resource cleanup after visualization."""
    import gc
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    visualizer = MetricsVisualizer(temp_metrics_dir / "report.html")
    result_path = visualizer.create_visualizations(
        pd.DataFrame(np.random.rand(1000, 4)),
        pd.DataFrame(np.random.rand(1000, 4))
    )
    
    # Force cleanup
    del visualizer
    gc.collect()
    
    # Check memory usage
    final_memory = process.memory_info().rss
    memory_diff_mb = (final_memory - initial_memory) / (1024 * 1024)
    
    assert memory_diff_mb < 100, "Memory usage increased significantly"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
