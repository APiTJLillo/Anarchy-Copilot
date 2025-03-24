"""Multi-objective optimization for simulation parameters."""

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
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize

from .simulation_optimization import SimulationOptimizer, OptimizationConfig
from .monte_carlo_power import MonteCarloAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    pop_size: int = 100
    n_generations: int = 50
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    objective_weights: Dict[str, float] = None
    output_path: Optional[Path] = None

class SimulationProblem(Problem):
    """Multi-objective optimization problem definition."""
    
    def __init__(
        self,
        optimizer: SimulationOptimizer,
        names: List[str],
        data: Dict[str, Any]
    ):
        super().__init__(
            n_var=len(optimizer.param_ranges),
            n_obj=3,
            n_constr=0,
            xl=np.array([r[0] for r in optimizer.param_ranges.values()]),
            xu=np.array([r[1] for r in optimizer.param_ranges.values()])
        )
        
        self.optimizer = optimizer
        self.names = names
        self.data = data
        self.scaler = StandardScaler()
    
    def _evaluate(
        self,
        x: np.ndarray,
        out: Dict[str, np.ndarray],
        *args,
        **kwargs
    ):
        """Evaluate objectives."""
        f = np.zeros((x.shape[0], self.n_obj))
        
        for i in range(x.shape[0]):
            # Convert parameters
            params = {
                name: value
                for name, value in zip(self.optimizer.param_ranges.keys(), x[i])
            }
            
            # Calculate objectives
            power_error = -self.optimizer._evaluate_parameters(
                params,
                self.names,
                self.data
            )
            
            computation_cost = np.log(params["n_simulations"] * params["sample_size"])
            
            stability = self._calculate_stability(params)
            
            f[i, 0] = power_error
            f[i, 1] = computation_cost
            f[i, 2] = -stability
        
        out["F"] = f
    
    def _calculate_stability(
        self,
        params: Dict[str, Any]
    ) -> float:
        """Calculate solution stability."""
        stability_trials = []
        
        for _ in range(5):
            noisy_params = {
                k: v * (1 + np.random.normal(0, 0.05))
                for k, v in params.items()
            }
            
            score = self.optimizer._evaluate_parameters(
                noisy_params,
                self.names,
                self.data
            )
            
            stability_trials.append(score)
        
        return -np.std(stability_trials)

class MultiObjectiveOptimizer:
    """Multi-objective optimization for simulation parameters."""
    
    def __init__(
        self,
        optimizer: SimulationOptimizer,
        config: MultiObjectiveConfig
    ):
        self.optimizer = optimizer
        self.config = config
        
        if not config.objective_weights:
            self.config.objective_weights = {
                "power_error": 1.0,
                "computation_cost": 0.5,
                "stability": 0.8
            }
    
    def optimize_multi_objective(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform multi-objective optimization."""
        results = {
            "pareto_front": self._optimize_pareto(names, data),
            "trade_offs": self._analyze_trade_offs(names, data),
            "sensitivity": self._analyze_sensitivity(names, data),
            "robustness": self._analyze_robustness(names, data)
        }
        
        return results
    
    def visualize_optimization(
        self,
        optimization: Dict[str, Any]
    ) -> go.Figure:
        """Create visualization of multi-objective optimization."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Pareto Front",
                "Trade-off Analysis",
                "Sensitivity Analysis",
                "Robustness Analysis"
            ]
        )
        
        # Pareto front plot
        pareto = optimization["pareto_front"]
        for sol in pareto["solutions"]:
            fig.add_trace(
                go.Scatter3d(
                    x=[sol["objectives"]["power_error"]],
                    y=[sol["objectives"]["computation_cost"]],
                    z=[sol["objectives"]["stability"]],
                    mode="markers",
                    marker=dict(size=5),
                    name=f"Solution {sol['id']}"
                ),
                row=1,
                col=1
            )
        
        # Trade-off plot
        trade_offs = optimization["trade_offs"]
        fig.add_trace(
            go.Heatmap(
                z=trade_offs["correlation_matrix"],
                x=trade_offs["objectives"],
                y=trade_offs["objectives"],
                colorscale="RdBu",
                name="Trade-offs"
            ),
            row=1,
            col=2
        )
        
        # Sensitivity plot
        sensitivity = optimization["sensitivity"]
        fig.add_trace(
            go.Heatmap(
                z=sensitivity["sensitivity_matrix"],
                x=sensitivity["parameters"],
                y=sensitivity["objectives"],
                colorscale="Viridis",
                name="Sensitivity"
            ),
            row=2,
            col=1
        )
        
        # Robustness plot
        robustness = optimization["robustness"]
        for obj in robustness["stability"]:
            fig.add_trace(
                go.Box(
                    y=robustness["stability"][obj],
                    name=obj,
                    boxpoints="outliers"
                ),
                row=2,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            title="Multi-objective Optimization Results",
            showlegend=True,
            scene=dict(
                xaxis_title="Power Error",
                yaxis_title="Computation Cost",
                zaxis_title="Stability"
            )
        )
        
        return fig
    
    def _optimize_pareto(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Find Pareto optimal solutions."""
        problem = SimulationProblem(self.optimizer, names, data)
        
        algorithm = NSGA2(
            pop_size=self.config.pop_size,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=self.config.crossover_prob),
            mutation=get_mutation("real_pm", prob=self.config.mutation_prob)
        )
        
        result = minimize(
            problem,
            algorithm,
            ("n_gen", self.config.n_generations),
            seed=self.optimizer.config.random_seed,
            verbose=True
        )
        
        # Process results
        solutions = []
        for i, x in enumerate(result.X):
            f = result.F[i]
            
            params = {
                name: value
                for name, value in zip(self.optimizer.param_ranges.keys(), x)
            }
            
            solution = {
                "id": i,
                "parameters": params,
                "objectives": {
                    "power_error": float(f[0]),
                    "computation_cost": float(f[1]),
                    "stability": float(f[2])
                }
            }
            
            solutions.append(solution)
        
        return {
            "solutions": solutions,
            "metrics": {
                "hypervolume": result.algorithm.indicator_hypervolume,
                "spread": result.algorithm.indicator_spread
            }
        }
    
    def _analyze_trade_offs(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze trade-offs between objectives."""
        results = {
            "objectives": ["power_error", "computation_cost", "stability"],
            "correlation_matrix": None,
            "pareto_dominance": {}
        }
        
        # Get Pareto solutions
        pareto = self._optimize_pareto(names, data)["solutions"]
        
        # Calculate correlation matrix
        obj_values = np.array([
            [s["objectives"][obj] for obj in results["objectives"]]
            for s in pareto
        ])
        
        results["correlation_matrix"] = np.corrcoef(obj_values.T)
        
        # Analyze Pareto dominance
        for i, sol1 in enumerate(pareto):
            dominates = []
            dominated_by = []
            
            for j, sol2 in enumerate(pareto):
                if i != j:
                    if self._dominates(sol1, sol2):
                        dominates.append(j)
                    elif self._dominates(sol2, sol1):
                        dominated_by.append(j)
            
            results["pareto_dominance"][i] = {
                "dominates": dominates,
                "dominated_by": dominated_by
            }
        
        return results
    
    def _analyze_sensitivity(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze parameter sensitivity."""
        results = {
            "parameters": list(self.optimizer.param_ranges.keys()),
            "objectives": ["power_error", "computation_cost", "stability"],
            "sensitivity_matrix": None,
            "critical_parameters": []
        }
        
        # Get baseline solution
        baseline = self._optimize_pareto(names, data)["solutions"][0]
        
        # Calculate sensitivity matrix
        n_params = len(results["parameters"])
        n_obj = len(results["objectives"])
        sensitivity = np.zeros((n_obj, n_params))
        
        for i, param in enumerate(results["parameters"]):
            # Perturb parameter
            delta = 0.1 * (
                self.optimizer.param_ranges[param][1] -
                self.optimizer.param_ranges[param][0]
            )
            
            perturbed = baseline["parameters"].copy()
            perturbed[param] += delta
            
            # Evaluate perturbation
            problem = SimulationProblem(self.optimizer, names, data)
            f_perturbed = problem._evaluate(
                np.array([list(perturbed.values())]),
                {"F": np.zeros((1, 3))}
            )
            
            # Calculate sensitivity
            sensitivity[:, i] = (
                f_perturbed["F"][0] -
                np.array(list(baseline["objectives"].values()))
            ) / delta
        
        results["sensitivity_matrix"] = sensitivity
        
        # Identify critical parameters
        sensitivity_norm = np.linalg.norm(sensitivity, axis=0)
        critical_idx = np.where(sensitivity_norm > np.mean(sensitivity_norm))[0]
        results["critical_parameters"] = [
            results["parameters"][i] for i in critical_idx
        ]
        
        return results
    
    def _analyze_robustness(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze solution robustness."""
        results = {
            "stability": defaultdict(list),
            "variation": {},
            "ranking": []
        }
        
        # Get Pareto solutions
        pareto = self._optimize_pareto(names, data)["solutions"]
        
        # Analyze stability
        for solution in pareto:
            params = solution["parameters"]
            
            for _ in range(20):
                # Add noise
                noisy_params = {
                    k: v * (1 + np.random.normal(0, 0.1))
                    for k, v in params.items()
                }
                
                # Evaluate
                problem = SimulationProblem(self.optimizer, names, data)
                f_noisy = problem._evaluate(
                    np.array([list(noisy_params.values())]),
                    {"F": np.zeros((1, 3))}
                )
                
                # Record variations
                for i, obj in enumerate(["power_error", "computation_cost", "stability"]):
                    results["stability"][obj].append(f_noisy["F"][0, i])
        
        # Calculate variation statistics
        for obj in results["stability"]:
            values = results["stability"][obj]
            results["variation"][obj] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "cv": np.std(values) / np.abs(np.mean(values))
            }
        
        # Rank solutions by robustness
        robustness_scores = [
            -np.mean([v["cv"] for v in results["variation"].values()])
            for _ in pareto
        ]
        
        ranked_indices = np.argsort(robustness_scores)
        results["ranking"] = [
            {
                "solution_id": int(i),
                "robustness_score": float(robustness_scores[i])
            }
            for i in ranked_indices
        ]
        
        return results
    
    def _dominates(
        self,
        sol1: Dict[str, Any],
        sol2: Dict[str, Any]
    ) -> bool:
        """Check if solution 1 dominates solution 2."""
        better_in_any = False
        
        for obj, weight in self.config.objective_weights.items():
            val1 = sol1["objectives"][obj] * weight
            val2 = sol2["objectives"][obj] * weight
            
            if val1 > val2:
                return False
            elif val1 < val2:
                better_in_any = True
        
        return better_in_any
    
    def save_optimization(
        self,
        optimization: Dict[str, Any],
        output_path: Optional[Path] = None
    ):
        """Save multi-objective optimization results."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save optimization results
            opt_file = path / "multi_objective_optimization.json"
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
            viz.write_html(str(path / "multi_objective_optimization.html"))
            
            logger.info(f"Saved multi-objective optimization to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization: {e}")

def create_multi_objective_optimizer(
    optimizer: SimulationOptimizer,
    output_path: Optional[Path] = None
) -> MultiObjectiveOptimizer:
    """Create multi-objective optimizer."""
    config = MultiObjectiveConfig(output_path=output_path)
    return MultiObjectiveOptimizer(optimizer, config)

if __name__ == "__main__":
    # Example usage
    from .simulation_optimization import create_simulation_optimizer
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
    sim_optimizer = create_simulation_optimizer(mc_analyzer)
    mo_optimizer = create_multi_objective_optimizer(
        sim_optimizer,
        output_path=Path("multi_objective_optimization")
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
    
    # Perform multi-objective optimization
    optimization_results = mo_optimizer.optimize_multi_objective(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save results
    mo_optimizer.save_optimization(optimization_results)
