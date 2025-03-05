"""Performance tests for cost-benefit analysis."""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List

import pytest
import numpy as np
from unittest.mock import Mock

from proxy.analysis.monitoring.costbenefit_analysis import (
    CostBenefitAnalyzer,
    CostConfig,
    CostComponent,
    BenefitComponent,
    VisualizationConfig,
    PlotControls
)

def generate_test_data(num_components: int, num_scenarios: int):
    """Generate test data with specified scale."""
    scenarios = {}
    
    for s in range(num_scenarios):
        scenario_name = f"scenario_{s}"
        interventions = {}
        effects = {}
        optimization = {}
        
        for i in range(num_components):
            intervention_name = f"intervention_{i}"
            interventions[intervention_name] = [
                Mock(
                    magnitude=np.random.uniform(0.5, 2.0),
                    duration=timedelta(hours=int(np.random.uniform(1, 48))),
                    target=f"target_{i}",
                    priority=np.random.randint(1, 5),
                    cost=np.random.uniform(100, 1000),
                    constraints={"max_cost": np.random.uniform(1000, 5000)}
                )
            ]
            
            effects[intervention_name] = {
                f"target_{i}": Mock(
                    direct_impact=np.random.uniform(0.5, 1.0),
                    indirect_impacts={
                        "system": np.random.uniform(0.1, 0.5),
                        "related": np.random.uniform(0.1, 0.3)
                    },
                    success_probability=np.random.uniform(0.7, 1.0),
                    stability_score=np.random.uniform(0.8, 1.0),
                    recovery_time=np.random.uniform(1, 5)
                )
            }
            
            optimization[intervention_name] = {
                f"target_{i}": {
                    "optimal_impact": np.random.uniform(0.8, 1.0),
                    "optimal_magnitude": np.random.uniform(1.0, 1.5)
                }
            }
        
        scenarios[scenario_name] = Mock(
            interventions=interventions,
            effects=effects,
            optimization=optimization
        )
    
    return scenarios

@pytest.fixture
def large_scale_analyzer():
    """Create analyzer with large-scale test data."""
    analyzer = Mock()
    analyzer.results = generate_test_data(num_components=50, num_scenarios=10)
    
    config = CostConfig(
        enabled=True,
        visualization=VisualizationConfig(
            interactive=True,
            controls=PlotControls(
                enable_range_selector=True,
                enable_compare_mode=True
            )
        )
    )
    
    return CostBenefitAnalyzer(analyzer, config)

@pytest.mark.performance
@pytest.mark.asyncio
async def test_plot_creation_performance(large_scale_analyzer):
    """Test performance of plot creation with large datasets."""
    start_time = time.time()
    
    # Create plots
    plots = await large_scale_analyzer.create_costbenefit_plots()
    
    duration = time.time() - start_time
    
    # Performance assertions
    assert duration < 5.0  # Should complete within 5 seconds
    assert len(plots) >= 30  # At least 3 plots per scenario

@pytest.mark.performance
@pytest.mark.asyncio
@pytest.mark.parametrize("num_components", [10, 50, 100])
async def test_analysis_scalability(num_components):
    """Test analysis scalability with increasing data size."""
    # Create analyzer with varying data sizes
    analyzer = Mock()
    analyzer.results = generate_test_data(num_components=num_components, num_scenarios=1)
    
    config = CostConfig(enabled=True)
    cost_benefit_analyzer = CostBenefitAnalyzer(analyzer, config)
    
    start_time = time.time()
    
    # Run analysis
    await cost_benefit_analyzer.analyze_costs_benefits("scenario_0")
    
    duration = time.time() - start_time
    
    # Scale factor assertions
    if num_components <= 10:
        assert duration < 0.5
    elif num_components <= 50:
        assert duration < 2.0
    else:
        assert duration < 5.0

@pytest.mark.performance
@pytest.mark.asyncio
async def test_memory_usage(large_scale_analyzer):
    """Test memory usage during analysis."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run analysis
    for scenario in large_scale_analyzer.intervention_analyzer.results:
        await large_scale_analyzer.analyze_costs_benefits(scenario)
    
    # Create plots
    plots = await large_scale_analyzer.create_costbenefit_plots()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Memory usage assertions
    assert memory_increase < 500  # Should use less than 500MB additional memory

@pytest.mark.performance
@pytest.mark.asyncio
async def test_concurrent_analysis(large_scale_analyzer):
    """Test performance with concurrent analysis requests."""
    scenarios = list(large_scale_analyzer.intervention_analyzer.results.keys())
    
    start_time = time.time()
    
    # Run analyses concurrently
    tasks = [
        large_scale_analyzer.analyze_costs_benefits(scenario)
        for scenario in scenarios
    ]
    await asyncio.gather(*tasks)
    
    duration = time.time() - start_time
    
    # Performance assertions
    assert duration < len(scenarios) * 2  # Should take less than 2s per scenario
    assert len(large_scale_analyzer.results) == len(scenarios)

@pytest.mark.performance
@pytest.mark.asyncio
async def test_visualization_caching(large_scale_analyzer):
    """Test visualization caching performance."""
    # First run - should take longer
    start_time = time.time()
    plots1 = await large_scale_analyzer.create_costbenefit_plots()
    first_duration = time.time() - start_time
    
    # Second run - should be faster due to caching
    start_time = time.time()
    plots2 = await large_scale_analyzer.create_costbenefit_plots()
    second_duration = time.time() - start_time
    
    # Cache effectiveness assertions
    assert second_duration < first_duration * 0.5  # Should be at least 50% faster
    assert plots1.keys() == plots2.keys()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])
