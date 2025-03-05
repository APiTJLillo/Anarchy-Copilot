"""Interactive preference learning for multi-objective optimization."""

import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from scipy.stats import norm
import logging

from .test_costbenefit_multiobjective import (
    MultiObjectiveAdapter,
    MultiObjectiveConfig,
    ObjectiveConfig
)

@dataclass
class PreferenceConfig:
    """Configuration for interactive preference learning."""
    feedback_batch_size: int = 5
    min_confidence: float = 0.7
    exploration_factor: float = 0.2
    learning_rate: float = 0.1
    update_interval: float = 60.0  # seconds
    history_window: int = 100
    similarity_threshold: float = 0.8
    feedback_timeout: float = 30.0
    max_comparisons: int = 10
    enable_active_learning: bool = True
    enable_preference_drift: bool = True
    drift_detection_window: int = 20

@dataclass
class Comparison:
    """Pair of solutions for comparison."""
    solution_a: Dict[str, float]
    solution_b: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    preference: Optional[str] = None  # "A", "B", or None for equal
    confidence: float = 1.0

class InteractivePreferenceLearner:
    """Interactive preference learning for optimization."""

    def __init__(
        self,
        adapter: MultiObjectiveAdapter,
        config: PreferenceConfig
    ):
        self.adapter = adapter
        self.config = config
        
        # State
        self.comparison_history: List[Comparison] = []
        self.preference_model: Optional[Any] = None
        self.pending_comparisons: List[Comparison] = []
        self.active_query: Optional[Comparison] = None
        self.feedback_handlers: List[Callable] = []
        self.drift_detected: bool = False
        self.last_update: Optional[datetime] = None
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    async def start_learning(self) -> None:
        """Start interactive preference learning."""
        self.learning_task = asyncio.create_task(self._run_learning())

    async def stop_learning(self) -> None:
        """Stop interactive preference learning."""
        if hasattr(self, 'learning_task'):
            self.learning_task.cancel()
            try:
                await self.learning_task
            except asyncio.CancelledError:
                pass

    async def _run_learning(self) -> None:
        """Run continuous preference learning."""
        while True:
            try:
                current_time = datetime.now()
                
                # Check if update is needed
                if self.last_update and (
                    current_time - self.last_update
                ).total_seconds() < self.config.update_interval:
                    await asyncio.sleep(1.0)
                    continue
                
                # Generate comparison batch
                await self._generate_comparisons()
                
                # Process pending comparisons
                await self._process_comparisons()
                
                # Update preference model
                if self.comparison_history:
                    await self._update_model()
                
                # Check for preference drift
                if self.config.enable_preference_drift:
                    self.drift_detected = self._detect_drift()
                
                self.last_update = current_time
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Learning error: {e}")
                await asyncio.sleep(5.0)

    async def _generate_comparisons(self) -> None:
        """Generate new solution comparisons."""
        if len(self.pending_comparisons) >= self.config.feedback_batch_size:
            return
        
        # Get current Pareto front
        solutions = self.adapter.pareto_front
        if len(solutions) < 2:
            return
        
        # Generate comparisons using active learning if enabled
        if self.config.enable_active_learning and self.preference_model:
            pairs = self._select_informative_pairs(solutions)
        else:
            # Random selection
            pairs = self._select_random_pairs(solutions)
        
        # Create comparisons
        for sol_a, sol_b in pairs:
            comparison = Comparison(
                solution_a=sol_a,
                solution_b=sol_b
            )
            self.pending_comparisons.append(comparison)

    def _select_informative_pairs(
        self,
        solutions: List[Dict[str, float]]
    ) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        """Select most informative solution pairs."""
        pairs = []
        n = min(len(solutions), self.config.max_comparisons)
        
        # Calculate solution utilities
        utilities = self._predict_utilities(solutions)
        
        # Find pairs with highest uncertainty
        uncertainties = []
        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                utility_diff = abs(utilities[i] - utilities[j])
                uncertainty = 1 - norm.cdf(utility_diff)
                uncertainties.append((uncertainty, i, j))
        
        # Select most uncertain pairs
        for uncertainty, i, j in sorted(uncertainties, reverse=True)[:n]:
            pairs.append((solutions[i], solutions[j]))
        
        return pairs

    def _select_random_pairs(
        self,
        solutions: List[Dict[str, float]]
    ) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        """Select random solution pairs."""
        pairs = []
        n = min(len(solutions), self.config.max_comparisons)
        
        for _ in range(n):
            i, j = np.random.choice(len(solutions), 2, replace=False)
            pairs.append((solutions[i], solutions[j]))
        
        return pairs

    async def _process_comparisons(self) -> None:
        """Process pending comparisons."""
        timeout = self.config.feedback_timeout
        
        for comparison in self.pending_comparisons[:]:
            self.active_query = comparison
            
            # Get feedback from handlers
            feedback = await self._collect_feedback(timeout)
            
            if feedback:
                comparison.preference = feedback["preference"]
                comparison.confidence = feedback["confidence"]
                self.comparison_history.append(comparison)
                self.pending_comparisons.remove(comparison)
            
            self.active_query = None
        
        # Trim history if needed
        if len(self.comparison_history) > self.config.history_window:
            self.comparison_history = self.comparison_history[
                -self.config.history_window:
            ]

    async def _collect_feedback(
        self,
        timeout: float
    ) -> Optional[Dict[str, Union[str, float]]]:
        """Collect feedback from handlers."""
        try:
            # Create future for feedback
            feedback_future = asyncio.Future()
            
            def feedback_callback(result: Dict[str, Union[str, float]]):
                if not feedback_future.done():
                    feedback_future.set_result(result)
            
            # Register temporary handler
            self.feedback_handlers.append(feedback_callback)
            
            # Wait for feedback with timeout
            try:
                return await asyncio.wait_for(feedback_future, timeout)
            except asyncio.TimeoutError:
                return None
            
        finally:
            # Clean up handler
            if feedback_callback in self.feedback_handlers:
                self.feedback_handlers.remove(feedback_callback)

    async def _update_model(self) -> None:
        """Update preference learning model."""
        if len(self.comparison_history) < 2:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for comparison in self.comparison_history:
            if comparison.preference in ["A", "B"]:
                # Feature vector is difference between solutions
                features = self._extract_features(
                    comparison.solution_a,
                    comparison.solution_b
                )
                
                X.append(features)
                y.append(1 if comparison.preference == "A" else 0)
        
        if not X:
            return
        
        # Train model
        X = np.array(X)
        y = np.array(y)
        
        if self.preference_model is None:
            from sklearn.linear_model import LogisticRegression
            self.preference_model = LogisticRegression(
                class_weight="balanced"
            )
        
        self.preference_model.fit(X, y)

    def _extract_features(
        self,
        solution_a: Dict[str, float],
        solution_b: Dict[str, float]
    ) -> np.ndarray:
        """Extract features from solution pair."""
        features = []
        
        # Add objective differences
        for objective in self.adapter.config.objectives:
            diff = solution_a[objective.name] - solution_b[objective.name]
            features.append(diff)
        
        return np.array(features)

    def _predict_utilities(
        self,
        solutions: List[Dict[str, float]]
    ) -> np.ndarray:
        """Predict utilities for solutions."""
        if not self.preference_model:
            # Return random utilities if no model
            return np.random.random(len(solutions))
        
        # Calculate pairwise comparisons
        utilities = np.zeros(len(solutions))
        
        for i, sol_i in enumerate(solutions):
            wins = 0
            for j, sol_j in enumerate(solutions):
                if i == j:
                    continue
                
                features = self._extract_features(sol_i, sol_j)
                prob = self.preference_model.predict_proba([features])[0][1]
                wins += prob
            
            utilities[i] = wins / (len(solutions) - 1)
        
        return utilities

    def _detect_drift(self) -> bool:
        """Detect changes in preferences."""
        if len(self.comparison_history) < self.config.drift_detection_window:
            return False
        
        recent = self.comparison_history[-self.config.drift_detection_window:]
        old = self.comparison_history[:-self.config.drift_detection_window]
        
        if not old:
            return False
        
        # Compare preference distributions
        def get_preference_dist(comparisons):
            counts = {"A": 0, "B": 0, None: 0}
            for comp in comparisons:
                counts[comp.preference] = counts.get(comp.preference, 0) + 1
            total = sum(counts.values())
            return {k: v/total for k, v in counts.items()}
        
        recent_dist = get_preference_dist(recent)
        old_dist = get_preference_dist(old)
        
        # Calculate distribution similarity
        similarity = sum(
            min(recent_dist.get(k, 0), old_dist.get(k, 0))
            for k in set(recent_dist) | set(old_dist)
        )
        
        return similarity < self.config.similarity_threshold

    def provide_feedback(
        self,
        preference: str,
        confidence: float = 1.0
    ) -> None:
        """Provide feedback for active comparison."""
        if not self.active_query:
            return
        
        # Validate input
        if preference not in ["A", "B", None]:
            raise ValueError("Preference must be 'A', 'B' or None")
        
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        
        # Notify handlers
        feedback = {
            "preference": preference,
            "confidence": confidence
        }
        
        for handler in self.feedback_handlers:
            handler(feedback)

@pytest.fixture
def preference_learner(multi_objective_adapter):
    """Create preference learner for testing."""
    config = PreferenceConfig()
    return InteractivePreferenceLearner(multi_objective_adapter, config)

@pytest.mark.asyncio
async def test_preference_learning(preference_learner):
    """Test preference learning process."""
    # Start learning
    await preference_learner.start_learning()
    
    # Generate some test comparisons
    for _ in range(5):
        await preference_learner._generate_comparisons()
        
        # Provide random feedback
        if preference_learner.active_query:
            preference_learner.provide_feedback(
                np.random.choice(["A", "B"]),
                confidence=0.8
            )
        
        await asyncio.sleep(0.1)
    
    # Allow time for processing
    await asyncio.sleep(1)
    
    # Verify learning occurred
    assert len(preference_learner.comparison_history) > 0
    if len(preference_learner.comparison_history) >= 2:
        assert preference_learner.preference_model is not None
    
    # Stop learning
    await preference_learner.stop_learning()

@pytest.mark.asyncio
async def test_drift_detection(preference_learner):
    """Test preference drift detection."""
    # Add consistent preferences
    for i in range(20):
        comparison = Comparison(
            solution_a={"test": i},
            solution_b={"test": i+1},
            preference="A"
        )
        preference_learner.comparison_history.append(comparison)
    
    # Add different preferences
    for i in range(20):
        comparison = Comparison(
            solution_a={"test": i},
            solution_b={"test": i+1},
            preference="B"
        )
        preference_learner.comparison_history.append(comparison)
    
    # Check drift detection
    assert preference_learner._detect_drift()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
