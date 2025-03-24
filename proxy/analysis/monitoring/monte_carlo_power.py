"""Monte Carlo simulation for power analysis."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from tqdm import tqdm

from .power_analysis import ChainPowerAnalyzer, PowerConfig
from .statistical_comparison import ChainStatistician

logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""
    n_simulations: int = 1000
    parallel_jobs: int = mp.cpu_count()
    random_seed: Optional[int] = None
    confidence_level: float = 0.95
    effect_sizes: List[float] = None
    sample_sizes: List[int] = None
    distributions: List[str] = None
    output_path: Optional[Path] = None

class MonteCarloAnalyzer:
    """Monte Carlo simulation for power analysis."""
    
    def __init__(
        self,
        power_analyzer: ChainPowerAnalyzer,
        config: SimulationConfig
    ):
        self.power_analyzer = power_analyzer
        self.config = config
        
        # Set default values if not provided
        if not config.effect_sizes:
            self.config.effect_sizes = [0.2, 0.5, 0.8]
        
        if not config.sample_sizes:
            self.config.sample_sizes = [30, 50, 100, 200, 500]
        
        if not config.distributions:
            self.config.distributions = ["normal", "uniform", "gamma"]
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def simulate_power(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform Monte Carlo power analysis."""
        results = {
            "empirical_power": self._simulate_empirical_power(names, data),
            "robustness": self._simulate_robustness(names, data),
            "sensitivity": self._simulate_sensitivity(names, data),
            "distribution_effects": self._simulate_distribution_effects(names, data)
        }
        
        return results
    
    def visualize_simulation(
        self,
        simulation: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of simulation results."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Empirical Power",
                "Robustness Analysis",
                "Sensitivity Analysis",
                "Distribution Effects"
            ]
        )
        
        # Empirical power plot
        emp_power = simulation["empirical_power"]
        for effect, data in emp_power["results"].items():
            fig.add_trace(
                go.Scatter(
                    x=data["sample_sizes"],
                    y=data["power"],
                    name=f"Effect={effect}",
                    mode="lines+markers"
                ),
                row=1,
                col=1
            )
        
        # Robustness analysis plot
        robustness = simulation["robustness"]
        fig.add_trace(
            go.Violin(
                x=robustness["effect_sizes"],
                y=robustness["power_distributions"],
                name="Power Distribution",
                box_visible=True,
                meanline_visible=True
            ),
            row=1,
            col=2
        )
        
        # Sensitivity analysis plot
        sensitivity = simulation["sensitivity"]
        fig.add_trace(
            go.Heatmap(
                z=sensitivity["power_matrix"],
                x=sensitivity["noise_levels"],
                y=sensitivity["effect_sizes"],
                colorscale="Viridis",
                name="Power Sensitivity"
            ),
            row=2,
            col=1
        )
        
        # Distribution effects plot
        dist_effects = simulation["distribution_effects"]
        for dist, data in dist_effects["results"].items():
            fig.add_trace(
                go.Box(
                    y=data["power"],
                    name=dist,
                    boxpoints="outliers"
                ),
                row=2,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title="Monte Carlo Power Analysis",
            showlegend=True
        )
        
        return fig
    
    def _simulate_empirical_power(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate empirical power via Monte Carlo."""
        results = {
            "results": {},
            "confidence_intervals": {}
        }
        
        # Parallel simulation for each effect size
        with ProcessPoolExecutor(max_workers=self.config.parallel_jobs) as executor:
            futures = []
            
            for effect_size in self.config.effect_sizes:
                futures.append(
                    executor.submit(
                        self._run_power_simulation,
                        effect_size,
                        names,
                        data
                    )
                )
            
            # Collect results
            for future, effect_size in zip(futures, self.config.effect_sizes):
                power_results = future.result()
                results["results"][effect_size] = power_results
                
                # Calculate confidence intervals
                ci = self._calculate_confidence_intervals(
                    power_results["power"],
                    self.config.confidence_level
                )
                results["confidence_intervals"][effect_size] = ci
        
        return results
    
    def _simulate_robustness(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate robustness to violations of assumptions."""
        results = {
            "effect_sizes": [],
            "power_distributions": [],
            "outlier_effects": {},
            "heterogeneity_effects": {}
        }
        
        # Simulate with different data conditions
        for effect_size in self.config.effect_sizes:
            powers = []
            
            # Normal simulation
            powers.extend(self._run_robustness_simulation(
                effect_size,
                names,
                data
            ))
            
            # Add outliers
            outlier_powers = self._run_robustness_simulation(
                effect_size,
                names,
                data,
                add_outliers=True
            )
            results["outlier_effects"][effect_size] = outlier_powers
            
            # Add heterogeneity
            heterog_powers = self._run_robustness_simulation(
                effect_size,
                names,
                data,
                add_heterogeneity=True
            )
            results["heterogeneity_effects"][effect_size] = heterog_powers
            
            results["effect_sizes"].extend([effect_size] * len(powers))
            results["power_distributions"].extend(powers)
        
        return results
    
    def _simulate_sensitivity(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate sensitivity to parameter variations."""
        results = {
            "effect_sizes": self.config.effect_sizes,
            "noise_levels": np.linspace(0, 1, 10),
            "power_matrix": []
        }
        
        # Calculate power for each combination
        power_matrix = np.zeros(
            (len(self.config.effect_sizes), 10)
        )
        
        for i, effect_size in enumerate(self.config.effect_sizes):
            for j, noise in enumerate(results["noise_levels"]):
                power = np.mean(
                    self._run_sensitivity_simulation(
                        effect_size,
                        names,
                        data,
                        noise_level=noise
                    )
                )
                power_matrix[i, j] = power
        
        results["power_matrix"] = power_matrix.tolist()
        return results
    
    def _simulate_distribution_effects(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate effects of different distributions."""
        results = {
            "results": {},
            "comparative_stats": {}
        }
        
        for dist in self.config.distributions:
            powers = []
            
            for effect_size in self.config.effect_sizes:
                power = self._run_distribution_simulation(
                    effect_size,
                    names,
                    data,
                    distribution=dist
                )
                powers.extend(power)
            
            results["results"][dist] = {
                "power": powers,
                "mean": np.mean(powers),
                "std": np.std(powers)
            }
        
        # Calculate comparative statistics
        base_dist = self.config.distributions[0]
        base_powers = results["results"][base_dist]["power"]
        
        for dist in self.config.distributions[1:]:
            comp_powers = results["results"][dist]["power"]
            
            # Perform statistical tests
            t_stat, p_val = stats.ttest_ind(base_powers, comp_powers)
            effect_size = (
                np.mean(comp_powers) - np.mean(base_powers)
            ) / np.sqrt(
                (np.var(base_powers) + np.var(comp_powers)) / 2
            )
            
            results["comparative_stats"][f"{base_dist}_vs_{dist}"] = {
                "t_statistic": t_stat,
                "p_value": p_val,
                "effect_size": effect_size
            }
        
        return results
    
    def _run_power_simulation(
        self,
        effect_size: float,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run single power simulation."""
        sample_powers = []
        sample_sizes = []
        
        for n in self.config.sample_sizes:
            powers = []
            
            for _ in range(self.config.n_simulations):
                # Generate simulated data
                sim_data = self._generate_simulated_data(
                    data,
                    n,
                    effect_size
                )
                
                # Calculate power
                analysis = self.power_analyzer.analyze_power(
                    names,
                    sim_data
                )
                
                powers.append(
                    analysis["sample_size"]["power_curves"]["t_test"]["power"][-1]
                )
            
            sample_sizes.append(n)
            sample_powers.append(np.mean(powers))
        
        return {
            "sample_sizes": sample_sizes,
            "power": sample_powers
        }
    
    def _run_robustness_simulation(
        self,
        effect_size: float,
        names: List[str],
        data: Dict[str, Any],
        add_outliers: bool = False,
        add_heterogeneity: bool = False
    ) -> List[float]:
        """Run robustness simulation."""
        powers = []
        
        for _ in range(self.config.n_simulations):
            sim_data = self._generate_simulated_data(
                data,
                self.config.sample_sizes[0],
                effect_size
            )
            
            if add_outliers:
                sim_data = self._add_outliers(sim_data)
            
            if add_heterogeneity:
                sim_data = self._add_heterogeneity(sim_data)
            
            analysis = self.power_analyzer.analyze_power(
                names,
                sim_data
            )
            
            powers.append(
                analysis["sample_size"]["power_curves"]["t_test"]["power"][-1]
            )
        
        return powers
    
    def _run_sensitivity_simulation(
        self,
        effect_size: float,
        names: List[str],
        data: Dict[str, Any],
        noise_level: float
    ) -> List[float]:
        """Run sensitivity simulation."""
        powers = []
        
        for _ in range(self.config.n_simulations):
            sim_data = self._generate_simulated_data(
                data,
                self.config.sample_sizes[0],
                effect_size
            )
            
            # Add noise
            sim_data = self._add_noise(sim_data, noise_level)
            
            analysis = self.power_analyzer.analyze_power(
                names,
                sim_data
            )
            
            powers.append(
                analysis["sample_size"]["power_curves"]["t_test"]["power"][-1]
            )
        
        return powers
    
    def _run_distribution_simulation(
        self,
        effect_size: float,
        names: List[str],
        data: Dict[str, Any],
        distribution: str
    ) -> List[float]:
        """Run distribution simulation."""
        powers = []
        
        for _ in range(self.config.n_simulations):
            sim_data = self._generate_simulated_data(
                data,
                self.config.sample_sizes[0],
                effect_size,
                distribution=distribution
            )
            
            analysis = self.power_analyzer.analyze_power(
                names,
                sim_data
            )
            
            powers.append(
                analysis["sample_size"]["power_curves"]["t_test"]["power"][-1]
            )
        
        return powers
    
    def _generate_simulated_data(
        self,
        data: Dict[str, Any],
        n: int,
        effect_size: float,
        distribution: str = "normal"
    ) -> Dict[str, Any]:
        """Generate simulated data."""
        sim_data = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                sim_data[key] = self._generate_simulated_df(
                    value,
                    n,
                    effect_size,
                    distribution
                )
            elif isinstance(value, np.ndarray):
                sim_data[key] = self._generate_simulated_array(
                    value,
                    n,
                    effect_size,
                    distribution
                )
            else:
                sim_data[key] = value
        
        return sim_data
    
    def _generate_simulated_df(
        self,
        df: pd.DataFrame,
        n: int,
        effect_size: float,
        distribution: str
    ) -> pd.DataFrame:
        """Generate simulated DataFrame."""
        sim_df = pd.DataFrame()
        
        for col in df.columns:
            sim_df[col] = self._generate_simulated_array(
                df[col].values,
                n,
                effect_size,
                distribution
            )
        
        return sim_df
    
    def _generate_simulated_array(
        self,
        arr: np.ndarray,
        n: int,
        effect_size: float,
        distribution: str
    ) -> np.ndarray:
        """Generate simulated array."""
        mean = np.mean(arr)
        std = np.std(arr)
        
        if distribution == "normal":
            return np.random.normal(
                mean + effect_size * std,
                std,
                n
            )
        elif distribution == "uniform":
            return np.random.uniform(
                mean - std,
                mean + std + effect_size * std,
                n
            )
        elif distribution == "gamma":
            shape = (mean / std) ** 2
            scale = std ** 2 / mean
            return np.random.gamma(
                shape + effect_size,
                scale,
                n
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
    
    def _add_outliers(
        self,
        data: Dict[str, Any],
        proportion: float = 0.1
    ) -> Dict[str, Any]:
        """Add outliers to data."""
        outlier_data = data.copy()
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                for col in value.columns:
                    outlier_data[key][col] = self._add_outliers_to_array(
                        value[col].values,
                        proportion
                    )
            elif isinstance(value, np.ndarray):
                outlier_data[key] = self._add_outliers_to_array(
                    value,
                    proportion
                )
        
        return outlier_data
    
    def _add_outliers_to_array(
        self,
        arr: np.ndarray,
        proportion: float
    ) -> np.ndarray:
        """Add outliers to array."""
        n_outliers = int(len(arr) * proportion)
        indices = np.random.choice(len(arr), n_outliers, replace=False)
        
        outlier_arr = arr.copy()
        outlier_arr[indices] += np.random.normal(0, 5 * np.std(arr), n_outliers)
        
        return outlier_arr
    
    def _add_heterogeneity(
        self,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add heterogeneity to data."""
        hetero_data = data.copy()
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                for col in value.columns:
                    hetero_data[key][col] = self._add_heterogeneity_to_array(
                        value[col].values
                    )
            elif isinstance(value, np.ndarray):
                hetero_data[key] = self._add_heterogeneity_to_array(value)
        
        return hetero_data
    
    def _add_heterogeneity_to_array(
        self,
        arr: np.ndarray
    ) -> np.ndarray:
        """Add heterogeneity to array."""
        split = len(arr) // 2
        hetero_arr = arr.copy()
        
        hetero_arr[:split] *= np.random.uniform(0.5, 1.5, split)
        hetero_arr[split:] *= np.random.uniform(1.5, 2.5, len(arr) - split)
        
        return hetero_arr
    
    def _add_noise(
        self,
        data: Dict[str, Any],
        level: float
    ) -> Dict[str, Any]:
        """Add noise to data."""
        noisy_data = data.copy()
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                for col in value.columns:
                    noisy_data[key][col] = self._add_noise_to_array(
                        value[col].values,
                        level
                    )
            elif isinstance(value, np.ndarray):
                noisy_data[key] = self._add_noise_to_array(value, level)
        
        return noisy_data
    
    def _add_noise_to_array(
        self,
        arr: np.ndarray,
        level: float
    ) -> np.ndarray:
        """Add noise to array."""
        noise = np.random.normal(0, level * np.std(arr), len(arr))
        return arr + noise
    
    def _calculate_confidence_intervals(
        self,
        data: List[float],
        confidence: float
    ) -> Tuple[float, float]:
        """Calculate confidence intervals."""
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = stats.t.interval(
            confidence,
            len(data) - 1,
            loc=mean,
            scale=sem
        )
        return ci
    
    def save_simulation(
        self,
        simulation: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save simulation results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save simulation results
            sim_file = path / "monte_carlo_simulation.json"
            with open(sim_file, "w") as f:
                json.dump(
                    {
                        k: v for k, v in simulation.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_simulation(simulation)
            viz.write_html(str(path / "monte_carlo_simulation.html"))
            
            logger.info(f"Saved Monte Carlo simulation to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save simulation: {e}")

def create_monte_carlo_analyzer(
    power_analyzer: ChainPowerAnalyzer,
    output_path: Optional[Path] = None
) -> MonteCarloAnalyzer:
    """Create Monte Carlo analyzer."""
    config = SimulationConfig(output_path=output_path)
    return MonteCarloAnalyzer(power_analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .power_analysis import create_chain_power_analyzer
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
    power_analyzer = create_chain_power_analyzer(statistician)
    mc_analyzer = create_monte_carlo_analyzer(
        power_analyzer,
        output_path=Path("monte_carlo_simulation")
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
    
    # Perform Monte Carlo simulation
    simulation_results = mc_analyzer.simulate_power(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save results
    mc_analyzer.save_simulation(simulation_results)
