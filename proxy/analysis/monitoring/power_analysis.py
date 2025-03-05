"""Power analysis tools for statistical comparisons."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestPower, GofChisquarePower, FTestPower
from statsmodels.stats.proportion import proportion_effectsize
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .statistical_comparison import ChainStatistician, StatisticalConfig

logger = logging.getLogger(__name__)

@dataclass
class PowerConfig:
    """Configuration for power analysis."""
    target_power: float = 0.8
    alpha: float = 0.05
    min_effect_size: float = 0.2
    max_effect_size: float = 1.0
    effect_size_steps: int = 20
    min_samples: int = 10
    max_samples: int = 1000
    sample_steps: int = 50
    output_path: Optional[Path] = None

class ChainPowerAnalyzer:
    """Power analysis for chain comparisons."""
    
    def __init__(
        self,
        statistician: ChainStatistician,
        config: PowerConfig
    ):
        self.statistician = statistician
        self.config = config
        
        self.ttest_power = TTestPower()
        self.chisq_power = GofChisquarePower()
        self.ftest_power = FTestPower()
    
    def analyze_power(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive power analysis."""
        analysis = self.statistician.analyze_chains(names, data)
        
        results = {
            "sample_size": self._analyze_sample_size(analysis),
            "effect_size": self._analyze_effect_size(analysis),
            "sensitivity": self._analyze_sensitivity(analysis),
            "equivalence": self._analyze_equivalence(analysis)
        }
        
        return results
    
    def visualize_power(
        self,
        analysis: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of power analysis."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Sample Size Analysis",
                "Effect Size Analysis",
                "Sensitivity Analysis",
                "Equivalence Testing"
            ]
        )
        
        # Sample size plot
        sample_data = analysis["sample_size"]
        for test_type, data in sample_data["power_curves"].items():
            fig.add_trace(
                go.Scatter(
                    x=data["n"],
                    y=data["power"],
                    name=f"{test_type} Power",
                    mode="lines"
                ),
                row=1,
                col=1
            )
        
        # Effect size plot
        effect_data = analysis["effect_size"]
        fig.add_trace(
            go.Scatter(
                x=effect_data["effect_sizes"],
                y=effect_data["power"],
                name="Effect Size Power",
                mode="lines"
            ),
            row=1,
            col=2
        )
        
        # Sensitivity analysis
        sens_data = analysis["sensitivity"]
        fig.add_trace(
            go.Heatmap(
                z=sens_data["power_matrix"],
                x=sens_data["sample_sizes"],
                y=sens_data["effect_sizes"],
                colorscale="Viridis",
                name="Power Sensitivity"
            ),
            row=2,
            col=1
        )
        
        # Equivalence testing
        equiv_data = analysis["equivalence"]
        fig.add_trace(
            go.Scatter(
                x=equiv_data["margins"],
                y=equiv_data["power"],
                name="Equivalence Power",
                mode="lines"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title="Power Analysis Results",
            showlegend=True
        )
        
        return fig
    
    def _analyze_sample_size(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze required sample sizes."""
        results = {
            "power_curves": {},
            "recommended": {}
        }
        
        # Get effect sizes from analysis
        effect_sizes = np.abs(analysis["effect_sizes"]["matrix"]).flatten()
        effect_sizes = effect_sizes[effect_sizes > 0]
        
        if len(effect_sizes) == 0:
            return results
        
        mean_effect = np.mean(effect_sizes)
        
        # Calculate power curves for different tests
        n_range = np.linspace(
            self.config.min_samples,
            self.config.max_samples,
            self.config.sample_steps
        )
        
        # T-test power curve
        ttest_power = [
            self.ttest_power.power(
                effect_size=mean_effect,
                nobs=n,
                alpha=self.config.alpha
            )
            for n in n_range
        ]
        results["power_curves"]["t_test"] = {
            "n": n_range.tolist(),
            "power": ttest_power
        }
        
        # Calculate recommended sample sizes
        results["recommended"]["t_test"] = self.ttest_power.solve_power(
            effect_size=mean_effect,
            power=self.config.target_power,
            alpha=self.config.alpha
        )
        
        # ANOVA power curve
        groups = len(analysis["descriptive"]["per_chain"])
        if groups > 2:
            ftest_power = [
                self.ftest_power.power(
                    effect_size=mean_effect,
                    nobs=n * groups,
                    alpha=self.config.alpha
                )
                for n in n_range
            ]
            results["power_curves"]["anova"] = {
                "n": n_range.tolist(),
                "power": ftest_power
            }
            
            results["recommended"]["anova"] = self.ftest_power.solve_power(
                effect_size=mean_effect,
                power=self.config.target_power,
                alpha=self.config.alpha
            ) / groups
        
        return results
    
    def _analyze_effect_size(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze effect size sensitivity."""
        results = {
            "effect_sizes": [],
            "power": [],
            "required_n": []
        }
        
        # Calculate power across effect sizes
        effect_sizes = np.linspace(
            self.config.min_effect_size,
            self.config.max_effect_size,
            self.config.effect_size_steps
        )
        
        # Use median sample size from chains
        median_n = np.median([
            len(data["values"])
            for data in analysis["distributions"]["per_chain"].values()
        ])
        
        for effect_size in effect_sizes:
            power = self.ttest_power.power(
                effect_size=effect_size,
                nobs=median_n,
                alpha=self.config.alpha
            )
            
            required_n = self.ttest_power.solve_power(
                effect_size=effect_size,
                power=self.config.target_power,
                alpha=self.config.alpha
            )
            
            results["effect_sizes"].append(effect_size)
            results["power"].append(power)
            results["required_n"].append(required_n)
        
        return results
    
    def _analyze_sensitivity(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze sensitivity to parameters."""
        results = {
            "sample_sizes": [],
            "effect_sizes": [],
            "power_matrix": []
        }
        
        # Create parameter grids
        sample_sizes = np.linspace(
            self.config.min_samples,
            self.config.max_samples,
            self.config.sample_steps
        )
        
        effect_sizes = np.linspace(
            self.config.min_effect_size,
            self.config.max_effect_size,
            self.config.effect_size_steps
        )
        
        # Calculate power for each combination
        power_matrix = np.zeros((len(effect_sizes), len(sample_sizes)))
        
        for i, effect in enumerate(effect_sizes):
            for j, n in enumerate(sample_sizes):
                power = self.ttest_power.power(
                    effect_size=effect,
                    nobs=n,
                    alpha=self.config.alpha
                )
                power_matrix[i, j] = power
        
        results["sample_sizes"] = sample_sizes.tolist()
        results["effect_sizes"] = effect_sizes.tolist()
        results["power_matrix"] = power_matrix.tolist()
        
        return results
    
    def _analyze_equivalence(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze equivalence testing power."""
        results = {
            "margins": [],
            "power": [],
            "recommended_margins": {}
        }
        
        # Get standard deviation from analysis
        std = np.mean([
            metrics["std"]
            for metrics in analysis["descriptive"]["per_chain"].values()
        ])
        
        # Calculate power for different equivalence margins
        margins = np.linspace(
            0.1 * std,
            2 * std,
            self.config.effect_size_steps
        )
        
        median_n = np.median([
            len(data["values"])
            for data in analysis["distributions"]["per_chain"].values()
        ])
        
        for margin in margins:
            # Convert margin to effect size
            effect_size = margin / std
            
            # Calculate power for TOST
            power = self.ttest_power.power(
                effect_size=effect_size,
                nobs=median_n,
                alpha=self.config.alpha
            )
            
            results["margins"].append(margin)
            results["power"].append(power)
        
        # Find recommended margins for target power
        try:
            recommended_margin = (
                std * self.ttest_power.solve_power(
                    power=self.config.target_power,
                    nobs=median_n,
                    alpha=self.config.alpha
                )
            )
            
            results["recommended_margins"] = {
                "absolute": recommended_margin,
                "relative": recommended_margin / std
            }
            
        except Exception:
            pass
        
        return results
    
    def save_analysis(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save power analysis results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save analysis results
            analysis_file = path / "power_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(
                    {
                        k: v for k, v in analysis.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_power(analysis)
            viz.write_html(str(path / "power_analysis.html"))
            
            logger.info(f"Saved power analysis to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_chain_power_analyzer(
    statistician: ChainStatistician,
    output_path: Optional[Path] = None
) -> ChainPowerAnalyzer:
    """Create chain power analyzer."""
    config = PowerConfig(output_path=output_path)
    return ChainPowerAnalyzer(statistician, config)

if __name__ == "__main__":
    # Example usage
    from .statistical_comparison import create_chain_statistician
    from .comparison_animation import create_chain_comparator
    from .chain_animation import create_chain_animator
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    from .interactive_learning import create_interactive_learning
    from .learning_visualization import create_learning_visualizer
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(learner)
    interactive_learning = create_interactive_learning(viz)
    filters = create_learning_filter(interactive_learning)
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(animator)
    statistician = create_chain_statistician(comparator)
    power_analyzer = create_chain_power_analyzer(
        statistician,
        output_path=Path("power_analysis")
    )
    
    # Create example chains
    chain.create_chain(
        "preprocessing_a",
        [
            {
                "filter": "time_range",
                "params": {"window": 30}
            },
            {
                "filter": "confidence",
                "params": {"threshold": 0.7}
            }
        ]
    )
    
    chain.create_chain(
        "preprocessing_b",
        [
            {
                "filter": "time_range",
                "params": {"window": 60}
            },
            {
                "filter": "complexity",
                "params": {"max_complexity": 5}
            }
        ]
    )
    
    # Example data
    data = {
        "timestamp": pd.date_range(start="2025-01-01", periods=1000, freq="H"),
        "confidence": np.random.uniform(0, 1, 1000),
        "success": np.random.choice([True, False], 1000),
        "complexity": np.random.randint(1, 20, 1000),
        "features": pd.DataFrame(
            np.random.randn(1000, 5),
            columns=["f1", "f2", "f3", "f4", "f5"]
        )
    }
    
    # Perform power analysis
    power_results = power_analyzer.analyze_power(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save results
    power_analyzer.save_analysis(power_results)
