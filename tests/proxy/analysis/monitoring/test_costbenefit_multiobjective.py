"""Multi-objective adaptation for visualization performance."""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import logging
import json

from .test_costbenefit_adaptation import (
    RealTimeAdapter,
    AdaptationConfig,
    AdaptationMetrics
)

@dataclass
class ObjectiveConfig:
    """Configuration for optimization objectives."""
    name: str
    weight: float
    target: float
    threshold: float
    minimize: bool = True
    constraint_type: str = "soft"  # soft, hard
    penalty_factor: float = 1.0

@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective optimization."""
    objectives: List[ObjectiveConfig]
    pareto_front_size: int = 100
    population_size: int = 200
    generations: int = 50
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 10
    convergence_threshold: float = 0.001
    constraint_tolerance: float = 0.05
    enable_preference_learning: bool = True
    preference_history_size: int = 1000

class MultiObjectiveAdapter:
    """Multi-objective performance adaptation."""

    def __init__(
        self,
        base_adapter: RealTimeAdapter,
        config: MultiObjectiveConfig
    ):
        self.base_adapter = base_adapter
        self.config = config
        
        # State
        self.pareto_front: List[Dict[str, float]] = []
        self.objective_history: Dict[str, List[float]] = {}
        self.preference_history: List[Dict[str, float]] = []
        self.current_solution: Optional[Dict[str, float]] = None
        self.generation: int = 0
        self.population: List[Dict[str, float]] = []
        
        # Initialize tracking
        for objective in self.config.objectives:
            self.objective_history[objective.name] = []

    async def optimize(self) -> Dict[str, float]:
        """Run multi-objective optimization."""
        # Initialize population
        if not self.population:
            self.population = await self._initialize_population()
        
        # Run genetic algorithm
        for _ in range(self.config.generations):
            # Evaluate population
            objectives = await self._evaluate_population()
            
            # Update Pareto front
            self.pareto_front = self._update_pareto_front(
                self.population,
                objectives
            )
            
            # Check convergence
            if self._check_convergence():
                break
            
            # Create next generation
            self.population = await self._create_next_generation()
            self.generation += 1
        
        # Select best solution
        solution = await self._select_solution()
        self.current_solution = solution
        
        # Update preference history
        if self.config.enable_preference_learning:
            self._update_preferences(solution)
        
        return solution

    async def _initialize_population(self) -> List[Dict[str, float]]:
        """Initialize optimization population."""
        population = []
        current_params = self.base_adapter.tuner._get_current_parameters()
        bounds = dict(zip(
            current_params.keys(),
            self.base_adapter.tuner._get_parameter_bounds()
        ))
        
        for _ in range(self.config.population_size):
            # Create random solution within bounds
            solution = {
                param: np.random.uniform(low, high)
                for param, (low, high) in bounds.items()
            }
            population.append(solution)
        
        return population

    async def _evaluate_population(self) -> List[Dict[str, float]]:
        """Evaluate objectives for population."""
        objectives = []
        
        for solution in self.population:
            # Apply solution parameters
            await self.base_adapter.tuner._update_parameters(solution)
            
            # Collect metrics
            metrics = await self.base_adapter.tuner._collect_metrics()
            
            # Calculate objective values
            objective_values = {}
            for objective in self.config.objectives:
                value = self._calculate_objective(objective, metrics)
                objective_values[objective.name] = value
            
            objectives.append(objective_values)
        
        return objectives

    def _calculate_objective(
        self,
        objective: ObjectiveConfig,
        metrics: Any
    ) -> float:
        """Calculate single objective value."""
        # Get metric value
        value = getattr(metrics, objective.name.lower())
        
        # Apply normalization
        normalized = (value - objective.target) / objective.threshold
        
        # Apply penalty for constraint violation
        if objective.constraint_type == "hard":
            if objective.minimize and value > objective.target + objective.threshold:
                return float('inf')
            if not objective.minimize and value < objective.target - objective.threshold:
                return float('inf')
        else:  # soft constraint
            if objective.minimize:
                if value > objective.target:
                    normalized *= (1 + objective.penalty_factor)
            else:
                if value < objective.target:
                    normalized *= (1 + objective.penalty_factor)
        
        return normalized if objective.minimize else -normalized

    def _update_pareto_front(
        self,
        population: List[Dict[str, float]],
        objectives: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """Update Pareto front with new solutions."""
        front = []
        
        for i, solution in enumerate(population):
            dominated = False
            
            # Check if solution is dominated
            for j, other_solution in enumerate(population):
                if i == j:
                    continue
                
                if self._dominates(objectives[j], objectives[i]):
                    dominated = True
                    break
            
            if not dominated:
                front.append(solution)
        
        # Trim front size if needed
        if len(front) > self.config.pareto_front_size:
            # Keep diverse solutions
            front = self._select_diverse_solutions(
                front,
                self.config.pareto_front_size
            )
        
        return front

    def _dominates(
        self,
        objectives1: Dict[str, float],
        objectives2: Dict[str, float]
    ) -> bool:
        """Check if objectives1 dominates objectives2."""
        at_least_one_better = False
        
        for name, value1 in objectives1.items():
            value2 = objectives2[name]
            if value1 > value2:
                return False
            if value1 < value2:
                at_least_one_better = True
        
        return at_least_one_better

    def _select_diverse_solutions(
        self,
        solutions: List[Dict[str, float]],
        n: int
    ) -> List[Dict[str, float]]:
        """Select diverse solutions using crowding distance."""
        if len(solutions) <= n:
            return solutions
        
        # Calculate crowding distances
        distances = self._calculate_crowding_distances(solutions)
        
        # Sort by distance and select top n
        sorted_indices = np.argsort(distances)[-n:]
        return [solutions[i] for i in sorted_indices]

    def _calculate_crowding_distances(
        self,
        solutions: List[Dict[str, float]]
    ) -> np.ndarray:
        """Calculate crowding distances for solutions."""
        n = len(solutions)
        distances = np.zeros(n)
        
        for param in solutions[0].keys():
            # Get values for parameter
            values = [s[param] for s in solutions]
            
            # Sort solutions by parameter
            sorted_indices = np.argsort(values)
            sorted_values = np.array(values)[sorted_indices]
            
            # Calculate distances
            value_range = sorted_values[-1] - sorted_values[0]
            if value_range == 0:
                continue
            
            # Add normalized distances
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            for i in range(1, n-1):
                distances[sorted_indices[i]] += (
                    sorted_values[i+1] - sorted_values[i-1]
                ) / value_range
        
        return distances

    async def _create_next_generation(self) -> List[Dict[str, float]]:
        """Create next generation using genetic operators."""
        new_population = []
        
        # Add elite solutions
        elite = self.population[:self.config.elite_size]
        new_population.extend(elite)
        
        # Fill rest of population
        while len(new_population) < self.config.population_size:
            if random.random() < self.config.crossover_rate:
                # Crossover
                parent1 = random.choice(self.population)
                parent2 = random.choice(self.population)
                child = self._crossover(parent1, parent2)
            else:
                # Mutation
                parent = random.choice(self.population)
                child = self._mutate(parent)
            
            new_population.append(child)
        
        return new_population

    def _crossover(
        self,
        parent1: Dict[str, float],
        parent2: Dict[str, float]
    ) -> Dict[str, float]:
        """Perform crossover between parents."""
        child = {}
        
        for param in parent1.keys():
            # Blend crossover
            alpha = random.random()
            value = alpha * parent1[param] + (1 - alpha) * parent2[param]
            
            # Ensure value is within bounds
            bounds = dict(zip(
                self.base_adapter.tuner._get_current_parameters().keys(),
                self.base_adapter.tuner._get_parameter_bounds()
            ))
            low, high = bounds[param]
            child[param] = np.clip(value, low, high)
        
        return child

    def _mutate(self, solution: Dict[str, float]) -> Dict[str, float]:
        """Mutate solution."""
        mutated = solution.copy()
        bounds = dict(zip(
            self.base_adapter.tuner._get_current_parameters().keys(),
            self.base_adapter.tuner._get_parameter_bounds()
        ))
        
        for param in mutated.keys():
            if random.random() < self.config.mutation_rate:
                # Add random noise
                noise = np.random.normal(0, 0.1)
                mutated[param] += noise
                
                # Clip to bounds
                low, high = bounds[param]
                mutated[param] = np.clip(mutated[param], low, high)
        
        return mutated

    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.objective_history[self.config.objectives[0].name]) < 2:
            return False
        
        # Check change in objective values
        for objective in self.config.objectives:
            history = self.objective_history[objective.name]
            change = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-10)
            
            if change > self.config.convergence_threshold:
                return False
        
        return True

    async def _select_solution(self) -> Dict[str, float]:
        """Select best solution from Pareto front."""
        if not self.pareto_front:
            return self.base_adapter.tuner._get_current_parameters()
        
        if self.config.enable_preference_learning and self.preference_history:
            # Use learned preferences
            return self._select_with_preferences()
        else:
            # Use weighted sum
            return self._select_with_weights()

    def _select_with_preferences(self) -> Dict[str, float]:
        """Select solution using learned preferences."""
        # Calculate preference vector from history
        preferences = {}
        for objective in self.config.objectives:
            values = [
                h[objective.name]
                for h in self.preference_history
            ]
            preferences[objective.name] = np.mean(values)
        
        # Find closest solution to preferences
        min_distance = float('inf')
        best_solution = None
        
        for solution in self.pareto_front:
            distance = sum(
                (solution[name] - pref) ** 2
                for name, pref in preferences.items()
            )
            
            if distance < min_distance:
                min_distance = distance
                best_solution = solution
        
        return best_solution

    def _select_with_weights(self) -> Dict[str, float]:
        """Select solution using objective weights."""
        best_score = float('inf')
        best_solution = None
        
        for solution in self.pareto_front:
            score = sum(
                obj.weight * solution[obj.name]
                for obj in self.config.objectives
            )
            
            if score < best_score:
                best_score = score
                best_solution = solution
        
        return best_solution

    def _update_preferences(self, solution: Dict[str, float]) -> None:
        """Update preference history with selected solution."""
        self.preference_history.append(solution)
        
        # Maintain history size
        if len(self.preference_history) > self.config.preference_history_size:
            self.preference_history = self.preference_history[
                -self.config.preference_history_size:
            ]

@pytest.fixture
def multi_objective_adapter(real_time_adapter):
    """Create multi-objective adapter for testing."""
    objectives = [
        ObjectiveConfig(
            name="response_time",
            weight=0.4,
            target=0.1,
            threshold=0.05
        ),
        ObjectiveConfig(
            name="memory_usage",
            weight=0.3,
            target=500,
            threshold=100
        ),
        ObjectiveConfig(
            name="error_rate",
            weight=0.2,
            target=0.01,
            threshold=0.005
        ),
        ObjectiveConfig(
            name="cpu_usage",
            weight=0.1,
            target=0.5,
            threshold=0.1
        )
    ]
    
    config = MultiObjectiveConfig(objectives=objectives)
    return MultiObjectiveAdapter(real_time_adapter, config)

@pytest.mark.asyncio
async def test_multi_objective_optimization(multi_objective_adapter):
    """Test multi-objective optimization."""
    # Run optimization
    solution = await multi_objective_adapter.optimize()
    
    # Verify solution
    assert solution is not None
    assert len(multi_objective_adapter.pareto_front) > 0
    
    # Verify objectives
    objectives = await multi_objective_adapter._evaluate_population()
    assert len(objectives) > 0
    for obj in objectives:
        for name in ["response_time", "memory_usage", "error_rate", "cpu_usage"]:
            assert name in obj

@pytest.mark.asyncio
async def test_preference_learning(multi_objective_adapter):
    """Test preference learning."""
    # Enable preference learning
    multi_objective_adapter.config.enable_preference_learning = True
    
    # Run multiple optimizations
    for _ in range(3):
        solution = await multi_objective_adapter.optimize()
        assert solution is not None
    
    # Verify preference history
    assert len(multi_objective_adapter.preference_history) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
