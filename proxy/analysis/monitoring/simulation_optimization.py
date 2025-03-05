"""Optimization methods for Monte Carlo simulation parameters."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy import stats, optimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import multiprocessing as mp
from functools import partial

from .monte_carlo_power import MonteCarloAnalyzer, SimulationConfig
from .power_analysis import ChainPowerAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for simulation optimization."""
    max_iterations: int = 50
    convergence_threshold: float = 0.001
    exploration_ratio: float = 0.2
    parallel_trials: int = mp.cpu_count()
    random_seed: Optional[int] = None
    target_power: float = 0.8
    output_path: Optional[Path] = None

class SimulationOptimizer:
    """Optimize Monte Carlo simulation parameters."""
    
    def __init__(
        self,
        mc_analyzer: MonteCarloAnalyzer,
        config: OptimizationConfig
    ):
        self.mc_analyzer = mc_analyzer
        self.config = config
        
        # Parameters to optimize
        self.param_ranges = {
            "n_simulations": (100, 10000),
            "effect_size": (0.1, 2.0),
            "sample_size": (10, 1000),
            "noise_level": (0.0, 1.0)
        }
        
        # Initialize GP regressor for Bayesian optimization
        self.gp = GaussianProcessRegressor(
            kernel=C(1.0) * RBF([1.0] * len(self.param_ranges)),
            n_restarts_optimizer=10,
            random_state=config.random_seed
        )
        
        # Set random seed if provided
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
    
    def optimize_simulation(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize simulation parameters via Bayesian optimization."""
        results = {
            "parameters": self._optimize_parameters(names, data),
            "convergence": self._analyze_convergence(names, data),
            "exploration": self._analyze_exploration(names, data),
            "validation": self._validate_optimization(names, data)
        }
        
        return results
    
    def visualize_optimization(
        self,
        optimization: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of optimization results."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Parameter Evolution",
                "Convergence Analysis",
                "Exploration vs Exploitation",
                "Validation Results"
            ]
        )
        
        # Parameter evolution plot
        params = optimization["parameters"]
        for param, values in params["evolution"].items():
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(values))),
                    y=values,
                    name=param,
                    mode="lines+markers"
                ),
                row=1,
                col=1
            )
        
        # Convergence plot
        conv = optimization["convergence"]
        fig.add_trace(
            go.Scatter(
                x=conv["iterations"],
                y=conv["objective"],
                name="Convergence",
                mode="lines"
            ),
            row=1,
            col=2
        )
        
        # Exploration plot
        explore = optimization["exploration"]
        fig.add_trace(
            go.Scatter(
                x=explore["trials"],
                y=explore["exploration_ratio"],
                name="Exploration Ratio",
                mode="lines"
            ),
            row=2,
            col=1
        )
        
        # Validation plot
        valid = optimization["validation"]
        fig.add_trace(
            go.Bar(
                x=list(valid["metrics"].keys()),
                y=list(valid["metrics"].values()),
                name="Validation Metrics"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title="Simulation Optimization Results",
            showlegend=True
        )
        
        return fig
    
    def _optimize_parameters(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform Bayesian optimization of parameters."""
        results = {
            "optimal_params": {},
            "evolution": defaultdict(list),
            "objective_history": []
        }
        
        # Initial random trials
        X_init = self._generate_initial_points()
        y_init = self._evaluate_points(X_init, names, data)
        
        # Fit GP to initial data
        self.gp.fit(X_init, y_init)
        
        # Start optimization loop
        best_score = max(y_init)
        best_params = X_init[np.argmax(y_init)]
        
        for i in range(self.config.max_iterations):
            # Get next point to evaluate
            next_point = self._get_next_point(X_init, y_init)
            
            # Evaluate point
            score = self._evaluate_single_point(next_point, names, data)
            
            # Update data
            X_init = np.vstack((X_init, [next_point]))
            y_init = np.append(y_init, score)
            
            # Update GP
            self.gp.fit(X_init, y_init)
            
            # Update best result
            if score > best_score:
                best_score = score
                best_params = next_point
            
            # Track evolution
            for j, param in enumerate(self.param_ranges.keys()):
                results["evolution"][param].append(best_params[j])
            results["objective_history"].append(best_score)
            
            # Check convergence
            if i > 10 and np.std(results["objective_history"][-10:]) < self.config.convergence_threshold:
                break
        
        # Convert optimal parameters to dictionary
        for param, value in zip(self.param_ranges.keys(), best_params):
            results["optimal_params"][param] = value
        
        return results
    
    def _analyze_convergence(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze optimization convergence."""
        results = {
            "iterations": [],
            "objective": [],
            "gradients": [],
            "stability": {}
        }
        
        # Use optimal parameters from previous optimization
        optimal_params = self._optimize_parameters(names, data)["optimal_params"]
        
        # Run multiple trials to analyze convergence
        for i in range(10):
            trial_results = []
            
            for j in range(self.config.max_iterations):
                # Add small random perturbation
                perturbed_params = {
                    k: v + np.random.normal(0, 0.1 * v)
                    for k, v in optimal_params.items()
                }
                
                # Evaluate
                score = self._evaluate_parameters(perturbed_params, names, data)
                trial_results.append(score)
                
                results["iterations"].append(j)
                results["objective"].append(score)
                
                if j > 0:
                    results["gradients"].append(
                        trial_results[-1] - trial_results[-2]
                    )
            
            # Analyze stability
            results["stability"][f"trial_{i}"] = {
                "mean": np.mean(trial_results),
                "std": np.std(trial_results),
                "convergence_iter": self._find_convergence_point(trial_results)
            }
        
        return results
    
    def _analyze_exploration(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze exploration vs exploitation balance."""
        results = {
            "trials": [],
            "exploration_ratio": [],
            "parameter_coverage": {},
            "acquisition_values": []
        }
        
        # Track parameter space coverage
        for param in self.param_ranges.keys():
            results["parameter_coverage"][param] = []
        
        # Run exploration analysis
        explored_points = set()
        total_points = 0
        
        for i in range(self.config.max_iterations):
            # Get next point with exploration ratio
            if np.random.random() < self.config.exploration_ratio:
                # Exploration
                point = self._generate_random_point()
            else:
                # Exploitation
                point = self._get_next_point(
                    np.array(list(explored_points)),
                    np.zeros(len(explored_points))
                )
            
            point_tuple = tuple(point)
            explored_points.add(point_tuple)
            total_points += 1
            
            # Calculate exploration ratio
            coverage_ratio = len(explored_points) / total_points
            results["trials"].append(i)
            results["exploration_ratio"].append(coverage_ratio)
            
            # Track parameter coverage
            for j, param in enumerate(self.param_ranges.keys()):
                coverage = (
                    point[j] - self.param_ranges[param][0]
                ) / (
                    self.param_ranges[param][1] - self.param_ranges[param][0]
                )
                results["parameter_coverage"][param].append(coverage)
            
            # Track acquisition function values
            acq_value = self._acquisition_function(point)
            results["acquisition_values"].append(acq_value)
        
        return results
    
    def _validate_optimization(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate optimization results."""
        results = {
            "metrics": {},
            "cross_validation": {},
            "robustness": {}
        }
        
        # Get optimal parameters
        optimal_params = self._optimize_parameters(names, data)["optimal_params"]
        
        # Calculate validation metrics
        validation_score = self._evaluate_parameters(optimal_params, names, data)
        results["metrics"]["validation_score"] = validation_score
        
        # Cross-validation
        cv_scores = []
        for i in range(5):
            # Split data
            train_data, test_data = self._split_data(data, 0.8)
            
            # Optimize on train
            train_params = self._optimize_parameters(names, train_data)["optimal_params"]
            
            # Evaluate on test
            test_score = self._evaluate_parameters(train_params, names, test_data)
            cv_scores.append(test_score)
        
        results["cross_validation"] = {
            "mean": np.mean(cv_scores),
            "std": np.std(cv_scores),
            "scores": cv_scores
        }
        
        # Robustness analysis
        robustness_scores = []
        for _ in range(10):
            # Add noise to parameters
            noisy_params = {
                k: v * (1 + np.random.normal(0, 0.1))
                for k, v in optimal_params.items()
            }
            
            # Evaluate
            score = self._evaluate_parameters(noisy_params, names, data)
            robustness_scores.append(score)
        
        results["robustness"] = {
            "mean": np.mean(robustness_scores),
            "std": np.std(robustness_scores),
            "scores": robustness_scores
        }
        
        return results
    
    def _generate_initial_points(
        self,
        n_points: int = 10
    ) -> np.ndarray:
        """Generate initial points for optimization."""
        points = []
        
        for _ in range(n_points):
            point = self._generate_random_point()
            points.append(point)
        
        return np.array(points)
    
    def _generate_random_point(self) -> np.ndarray:
        """Generate random point in parameter space."""
        point = []
        
        for param, (low, high) in self.param_ranges.items():
            if param == "n_simulations":
                value = np.random.randint(low, high)
            else:
                value = np.random.uniform(low, high)
            point.append(value)
        
        return np.array(point)
    
    def _evaluate_points(
        self,
        points: np.ndarray,
        names: List[str],
        data: Dict[str, Any]
    ) -> np.ndarray:
        """Evaluate multiple points in parallel."""
        with mp.Pool(self.config.parallel_trials) as pool:
            scores = pool.map(
                partial(
                    self._evaluate_single_point,
                    names=names,
                    data=data
                ),
                points
            )
        
        return np.array(scores)
    
    def _evaluate_single_point(
        self,
        point: np.ndarray,
        names: List[str],
        data: Dict[str, Any]
    ) -> float:
        """Evaluate single point."""
        # Convert point to parameters
        params = {
            name: value
            for name, value in zip(self.param_ranges.keys(), point)
        }
        
        return self._evaluate_parameters(params, names, data)
    
    def _evaluate_parameters(
        self,
        params: Dict[str, Any],
        names: List[str],
        data: Dict[str, Any]
    ) -> float:
        """Evaluate parameter set."""
        # Update simulation configuration
        sim_config = SimulationConfig(
            n_simulations=int(params["n_simulations"]),
            effect_sizes=[params["effect_size"]],
            sample_sizes=[int(params["sample_size"])],
            random_seed=self.config.random_seed
        )
        
        # Create new analyzer with updated config
        analyzer = MonteCarloAnalyzer(
            self.mc_analyzer.power_analyzer,
            sim_config
        )
        
        # Run simulation
        sim_results = analyzer.simulate_power(names, data)
        
        # Calculate objective (difference from target power)
        power = np.mean(
            sim_results["empirical_power"]["results"][params["effect_size"]]["power"]
        )
        
        return -abs(power - self.config.target_power)
    
    def _get_next_point(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Get next point using Bayesian optimization."""
        # Define acquisition function (Expected Improvement)
        best_f = np.max(y)
        
        def acquisition(x):
            x = x.reshape(1, -1)
            mu, std = self.gp.predict(x, return_std=True)
            z = (mu - best_f) / std
            return -(mu + self.config.exploration_ratio * std * stats.norm.cdf(z))
        
        # Optimize acquisition function
        x0 = self._generate_random_point()
        
        bounds = [
            (low, high)
            for low, high in self.param_ranges.values()
        ]
        
        result = optimize.minimize(
            acquisition,
            x0,
            bounds=bounds,
            method="L-BFGS-B"
        )
        
        return result.x
    
    def _acquisition_function(
        self,
        x: np.ndarray
    ) -> float:
        """Calculate acquisition function value."""
        x = x.reshape(1, -1)
        mu, std = self.gp.predict(x, return_std=True)
        return mu + self.config.exploration_ratio * std
    
    def _find_convergence_point(
        self,
        values: List[float],
        window: int = 10
    ) -> int:
        """Find point of convergence in series."""
        if len(values) < window:
            return len(values)
        
        # Use rolling standard deviation
        rolling_std = pd.Series(values).rolling(window).std()
        
        # Find first point where std is below threshold
        for i, std in enumerate(rolling_std[window:], window):
            if std < self.config.convergence_threshold:
                return i
        
        return len(values)
    
    def _split_data(
        self,
        data: Dict[str, Any],
        train_ratio: float
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Split data for cross-validation."""
        train_data = {}
        test_data = {}
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                split_idx = int(len(value) * train_ratio)
                train_data[key] = value.iloc[:split_idx]
                test_data[key] = value.iloc[split_idx:]
            elif isinstance(value, np.ndarray):
                split_idx = int(len(value) * train_ratio)
                train_data[key] = value[:split_idx]
                test_data[key] = value[split_idx:]
            else:
                train_data[key] = value
                test_data[key] = value
        
        return train_data, test_data
    
    def save_optimization(
        self,
        optimization: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save optimization results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save optimization results
            opt_file = path / "simulation_optimization.json"
            with open(opt_file, "w") as f:
                json.dump(
                    {
                        k: v for k, v in optimization.items()
                        if isinstance(v, (dict, list, str, int, float, bool))
                    },
                    f,
                    indent=2
                )
            
            # Save visualization
            viz = self.visualize_optimization(optimization)
            viz.write_html(str(path / "simulation_optimization.html"))
            
            logger.info(f"Saved optimization results to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization: {e}")

def create_simulation_optimizer(
    mc_analyzer: MonteCarloAnalyzer,
    output_path: Optional[Path] = None
) -> SimulationOptimizer:
    """Create simulation optimizer."""
    config = OptimizationConfig(output_path=output_path)
    return SimulationOptimizer(mc_analyzer, config)

if __name__ == "__main__":
    # Example usage
    from .monte_carlo_power import create_monte_carlo_analyzer
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
    mc_analyzer = create_monte_carlo_analyzer(power_analyzer)
    sim_optimizer = create_simulation_optimizer(
        mc_analyzer,
        output_path=Path("simulation_optimization")
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
    
    # Optimize simulation parameters
    optimization_results = sim_optimizer.optimize_simulation(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save results
    sim_optimizer.save_optimization(optimization_results)
