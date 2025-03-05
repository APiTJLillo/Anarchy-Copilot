"""Unit tests for cost-benefit analysis."""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import numpy as np
import plotly.graph_objects as go

from proxy.analysis.monitoring.costbenefit_analysis import (
    CostBenefitAnalyzer,
    CostConfig,
    CostComponent,
    BenefitComponent,
    VisualizationConfig,
    PlotControls
)

@pytest.fixture
def mock_intervention_analyzer():
    """Create mock intervention analyzer."""
    analyzer = Mock()
    analyzer.results = {
        "test_scenario": Mock(
            interventions={
                "test_intervention": [
                    Mock(
                        magnitude=1.0,
                        duration=timedelta(hours=24),
                        target="test_target",
                        priority=1,
                        cost=100.0,
                        constraints={"max_cost": 1000.0}
                    )
                ]
            },
            effects={
                "test_intervention": {
                    "test_target": Mock(
                        direct_impact=0.8,
                        indirect_impacts={"system": 0.2},
                        success_probability=0.9,
                        stability_score=0.95,
                        recovery_time=2.0
                    )
                }
            },
            optimization={
                "test_intervention": {
                    "test_target": {
                        "optimal_impact": 0.9,
                        "optimal_magnitude": 1.2
                    }
                }
            }
        )
    }
    return analyzer

@pytest.fixture
def analyzer(mock_intervention_analyzer):
    """Create cost-benefit analyzer instance."""
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
    return CostBenefitAnalyzer(mock_intervention_analyzer, config)

@pytest.mark.asyncio
async def test_cost_breakdown_plot(analyzer):
    """Test cost breakdown plot creation."""
    # Add test data
    costs = {
        "test": [
            CostComponent(
                name="implementation",
                fixed_cost=100.0,
                variable_cost=50.0,
                uncertainty=0.2
            ),
            CostComponent(
                name="operational",
                fixed_cost=200.0,
                variable_cost=75.0,
                uncertainty=0.3
            )
        ]
    }
    
    # Create plot
    fig = analyzer._create_cost_breakdown_plot(
        Mock(costs=costs),
        "test_scenario"
    )
    
    # Verify plot
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # Fixed and variable costs
    assert fig.data[0].name == "test Fixed"
    assert fig.data[1].name == "test Variable"

@pytest.mark.asyncio
async def test_financial_metrics_plot(analyzer):
    """Test financial metrics plot creation."""
    # Add test data
    result = Mock(
        net_present_value={"test": 1000.0},
        roi={"test": 50.0},
        benefit_cost_ratio={"test": 1.5}
    )
    
    # Create plot
    fig = analyzer._create_financial_metrics_plot(result, "test_scenario")
    
    # Verify plot
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 3  # NPV, ROI, BCR
    assert all(trace.name in ["Net Present Value", "ROI (%)", "Benefit-Cost Ratio"] 
              for trace in fig.data)

@pytest.mark.asyncio
async def test_risk_analysis_plot(analyzer):
    """Test risk analysis plot creation."""
    # Add test data
    result = Mock(
        uncertainty_ranges={
            "test_scenario": {
                "cost": (800.0, 1200.0),
                "benefit": (1500.0, 2000.0)
            }
        },
        sensitivity_scores={
            "test_scenario": {
                "cost": 0.7,
                "benefit": 0.8
            }
        },
        optimal_allocation={
            "test_scenario": {
                "implementation": 0.6,
                "operational": 0.4
            }
        },
        risk_adjusted_values={
            "test_scenario": {
                "net_value_test": 900.0
            }
        }
    )
    
    # Create plot
    fig = analyzer._create_risk_analysis_plot(result, "test_scenario")
    
    # Verify plot
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 4  # One trace per subplot

@pytest.mark.asyncio
async def test_interactive_controls(analyzer):
    """Test interactive control addition."""
    fig = go.Figure()
    plot_type = "financial_metrics"
    
    # Add controls
    fig = analyzer._add_interactive_controls(fig, plot_type)
    
    # Verify controls
    assert fig.layout.xaxis.rangeslider.visible
    assert "updatemenus" in fig.layout
    assert "modebar" in fig.layout

@pytest.mark.asyncio
async def test_event_handler_registration(analyzer):
    """Test event handler registration."""
    def test_handler(event_data):
        pass
    
    # Register handlers
    analyzer.register_event_handler("click", "cost", test_handler)
    analyzer.register_event_handler("hover", "benefit", test_handler)
    
    # Verify registration
    assert len(analyzer.click_handlers["cost"]) == 1
    assert len(analyzer.hover_handlers["benefit"]) == 1
    
    # Test invalid event type
    with pytest.raises(ValueError):
        analyzer.register_event_handler("invalid", "cost", test_handler)

@pytest.mark.asyncio
async def test_full_analysis_workflow(analyzer):
    """Test complete analysis workflow."""
    # Run analysis
    await analyzer.analyze_costs_benefits("test_scenario")
    
    # Create plots
    plots = await analyzer.create_costbenefit_plots()
    
    # Verify results
    assert "test_scenario_costs" in plots
    assert "test_scenario_metrics" in plots
    assert "test_scenario_risk" in plots
    
    # Verify plot configurations
    for plot in plots.values():
        assert plot.layout.template == analyzer.config.visualization.theme
        assert plot.layout.width == analyzer.config.visualization.width
        assert plot.layout.height == analyzer.config.visualization.height

if __name__ == "__main__":
    pytest.main([__file__])
