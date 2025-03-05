"""Reinforcement learning for optimization strategy selection."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import random
from collections import defaultdict
import pickle
import threading
import queue

from .adaptive_optimization import AdaptiveOptimizer, OptimizationStrategy
from .filter_optimization import FilterOptimizer

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    """Configuration for reinforcement learning."""
    epsilon: float = 0.1  # Exploration rate
    gamma: float = 0.9   # Discount factor
    alpha: float = 0.1   # Learning rate
    min_epsilon: float = 0.01
    epsilon_decay: float = 0.995
    batch_size: int = 32
    memory_size: int = 10000
    target_update: int = 100
    save_interval: int = 1000
    output_path: Optional[Path] = None

class ReplayMemory:
    """Experience replay memory."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory: List[Tuple[Any, ...]] = []
        self.position = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray
    ):
        """Save experience."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        """Sample random batch of experiences."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(self) -> int:
        return len(self.memory)

class QNetwork:
    """Q-learning neural network."""
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 64
    ):
        # Simple feedforward network
        self.model = {
            "w1": np.random.randn(state_size, hidden_size) / np.sqrt(state_size),
            "b1": np.zeros(hidden_size),
            "w2": np.random.randn(hidden_size, action_size) / np.sqrt(hidden_size),
            "b2": np.zeros(action_size)
        }
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = np.dot(state, self.model["w1"]) + self.model["b1"]
        h = np.maximum(0, h)  # ReLU
        q_values = np.dot(h, self.model["w2"]) + self.model["b2"]
        return q_values
    
    def update(
        self,
        state: np.ndarray,
        target: np.ndarray,
        learning_rate: float
    ):
        """Update network weights."""
        # Forward pass
        h = np.dot(state, self.model["w1"]) + self.model["b1"]
        h_relu = np.maximum(0, h)
        output = np.dot(h_relu, self.model["w2"]) + self.model["b2"]
        
        # Backward pass
        grad_output = output - target
        grad_w2 = np.outer(h_relu, grad_output)
        grad_b2 = grad_output
        
        grad_h_relu = np.dot(grad_output, self.model["w2"].T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        
        grad_w1 = np.outer(state, grad_h)
        grad_b1 = grad_h
        
        # Update weights
        self.model["w2"] -= learning_rate * grad_w2
        self.model["b2"] -= learning_rate * grad_b2
        self.model["w1"] -= learning_rate * grad_w1
        self.model["b1"] -= learning_rate * grad_b1

class RLOptimizer:
    """Reinforcement learning optimization strategy selector."""
    
    def __init__(
        self,
        adaptive_optimizer: AdaptiveOptimizer,
        config: RLConfig
    ):
        self.adaptive_optimizer = adaptive_optimizer
        self.config = config
        self.state_size = self._get_state_size()
        self.action_size = len(self.adaptive_optimizer.strategies)
        
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.memory = ReplayMemory(self.config.memory_size)
        
        self.steps = 0
        self.epsilon = self.config.epsilon
        self.learning_thread: Optional[threading.Thread] = None
        self.running = False
        self.experience_queue = queue.Queue()
    
    def start_learning(self):
        """Start learning thread."""
        if self.learning_thread is not None:
            return
        
        self.running = True
        self.learning_thread = threading.Thread(
            target=self._learning_loop,
            daemon=True
        )
        self.learning_thread.start()
    
    def stop_learning(self):
        """Stop learning thread."""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join()
            self.learning_thread = None
    
    def select_strategy(
        self,
        filter_name: str,
        state: Dict[str, float]
    ) -> OptimizationStrategy:
        """Select strategy using epsilon-greedy policy."""
        state_vector = self._encode_state(state)
        
        # Explore
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        # Exploit
        else:
            q_values = self.q_network.forward(state_vector)
            action = np.argmax(q_values)
        
        # Get corresponding strategy
        strategies = list(
            self.adaptive_optimizer.strategies[filter_name]
        )
        return strategies[action % len(strategies)]
    
    def update(
        self,
        filter_name: str,
        state: Dict[str, float],
        strategy: OptimizationStrategy,
        reward: float,
        next_state: Dict[str, float]
    ):
        """Update Q-network with experience."""
        state_vector = self._encode_state(state)
        next_state_vector = self._encode_state(next_state)
        
        # Get action index
        action = list(
            self.adaptive_optimizer.strategies[filter_name]
        ).index(strategy)
        
        # Store experience
        self.experience_queue.put((
            state_vector,
            action,
            reward,
            next_state_vector
        ))
        
        # Decay exploration rate
        self.epsilon = max(
            self.config.min_epsilon,
            self.epsilon * self.config.epsilon_decay
        )
    
    def _learning_loop(self):
        """Background learning loop."""
        while self.running:
            try:
                # Process experiences
                while not self.experience_queue.empty():
                    experience = self.experience_queue.get_nowait()
                    self.memory.push(*experience)
                
                # Train on batch
                if len(self.memory) >= self.config.batch_size:
                    self._train_batch()
                
                # Update target network
                self.steps += 1
                if self.steps % self.config.target_update == 0:
                    self._update_target_network()
                
                # Save model
                if (
                    self.config.save_interval > 0 and
                    self.steps % self.config.save_interval == 0
                ):
                    self.save_model()
                
            except Exception as e:
                logger.error(f"Learning loop error: {e}")
    
    def _train_batch(self):
        """Train on batch of experiences."""
        batch = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to arrays
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        rewards = np.array(rewards)
        
        # Get current Q values
        current_q = self.q_network.forward(states)
        
        # Get next Q values from target network
        next_q = self.target_network.forward(next_states)
        max_next_q = np.max(next_q, axis=1)
        
        # Calculate targets
        targets = current_q.copy()
        for i, action in enumerate(actions):
            targets[i, action] = rewards[i] + self.config.gamma * max_next_q[i]
        
        # Update network
        self.q_network.update(states, targets, self.config.alpha)
    
    def _update_target_network(self):
        """Update target network weights."""
        self.target_network.model = {
            k: v.copy()
            for k, v in self.q_network.model.items()
        }
    
    def _get_state_size(self) -> int:
        """Get state vector size."""
        # Basic metrics + strategy indicators
        return (
            4 +  # avg_time, throughput, cache_hit_rate, result_count
            len(self.adaptive_optimizer.strategies)
        )
    
    def _encode_state(self, state: Dict[str, float]) -> np.ndarray:
        """Encode state dictionary to vector."""
        vector = [
            state.get("avg_time", 0),
            state.get("avg_throughput", 0),
            state.get("cache_hit_rate", 0),
            state.get("result_counts", 0)
        ]
        
        # Add strategy indicators
        current_strategy = state.get("strategy", "none")
        for strategy in self.adaptive_optimizer.strategies:
            vector.append(float(strategy == current_strategy))
        
        return np.array(vector, dtype=np.float32)
    
    def calculate_reward(
        self,
        metrics: Dict[str, float],
        previous_metrics: Dict[str, float]
    ) -> float:
        """Calculate reward from metrics."""
        if not previous_metrics:
            return 0.0
        
        # Calculate performance improvement
        time_improvement = (
            previous_metrics.get("avg_time", 0) -
            metrics.get("avg_time", 0)
        ) / max(previous_metrics.get("avg_time", 1), 1e-6)
        
        throughput_improvement = (
            metrics.get("avg_throughput", 0) -
            previous_metrics.get("avg_throughput", 0)
        ) / max(previous_metrics.get("avg_throughput", 1), 1e-6)
        
        cache_improvement = (
            metrics.get("cache_hit_rate", 0) -
            previous_metrics.get("cache_hit_rate", 0)
        )
        
        # Weighted combination
        reward = (
            0.4 * time_improvement +
            0.4 * throughput_improvement +
            0.2 * cache_improvement
        )
        
        return float(reward)
    
    def save_model(self):
        """Save model state."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            state = {
                "q_network": self.q_network.model,
                "target_network": self.target_network.model,
                "steps": self.steps,
                "epsilon": self.epsilon
            }
            
            with open(output_path / "rl_model.pkl", "wb") as f:
                pickle.dump(state, f)
            
            logger.info(f"Saved model to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """Load model state."""
        if not self.config.output_path:
            return
        
        try:
            model_file = self.config.output_path / "rl_model.pkl"
            if not model_file.exists():
                return
            
            with open(model_file, "rb") as f:
                state = pickle.load(f)
            
            self.q_network.model = state["q_network"]
            self.target_network.model = state["target_network"]
            self.steps = state["steps"]
            self.epsilon = state["epsilon"]
            
            logger.info(f"Loaded model from {model_file}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

def create_rl_optimizer(
    adaptive_optimizer: AdaptiveOptimizer,
    output_path: Optional[Path] = None
) -> RLOptimizer:
    """Create reinforcement learning optimizer."""
    config = RLConfig(output_path=output_path)
    return RLOptimizer(adaptive_optimizer, config)

if __name__ == "__main__":
    # Example usage
    from .adaptive_optimization import create_adaptive_optimizer
    from .filter_optimization import create_filter_optimizer
    from .validation_filters import create_filter_manager
    from .preset_validation import create_preset_validator
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(preset_manager)
    filter_manager = create_filter_manager(validator)
    optimizer = create_filter_optimizer(filter_manager)
    adaptive = create_adaptive_optimizer(optimizer)
    
    # Create RL optimizer
    rl_optimizer = create_rl_optimizer(
        adaptive,
        output_path=Path("rl_optimization")
    )
    
    # Start learning
    rl_optimizer.start_learning()
    
    # Create filter
    filter_manager.create_error_filter(
        "critical_errors",
        ["schema", "value_range"]
    )
    
    # Get current state
    metrics = adaptive.metrics["critical_errors"].get_recent_performance()
    
    # Select and apply strategy
    strategy = rl_optimizer.select_strategy("critical_errors", metrics)
    strategy.apply(optimizer, "critical_errors")
    
    # Apply filter
    results = adaptive.apply_filter(
        "critical_errors",
        validator.validation_results.values()
    )
    
    # Get new state and update
    new_metrics = adaptive.metrics["critical_errors"].get_recent_performance()
    reward = rl_optimizer.calculate_reward(new_metrics, metrics)
    rl_optimizer.update(
        "critical_errors",
        metrics,
        strategy,
        reward,
        new_metrics
    )
    
    # Stop learning
    rl_optimizer.stop_learning()
    rl_optimizer.save_model()
