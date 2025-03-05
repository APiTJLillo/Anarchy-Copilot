"""Learning capabilities for optimization suggestions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import joblib

from .composition_optimization import CompositionOptimizer, OptimizationSuggestion
from .event_scheduler import ScheduledEvent

logger = logging.getLogger(__name__)

@dataclass
class LearningConfig:
    """Configuration for optimization learning."""
    min_samples: int = 100
    retrain_interval: int = 50
    feature_threshold: float = 0.1
    max_history: int = 1000
    auto_retrain: bool = True
    store_models: bool = True
    output_path: Optional[Path] = None

class OptimizationLearner:
    """Learn from optimization history to improve suggestions."""
    
    def __init__(
        self,
        optimizer: CompositionOptimizer,
        config: LearningConfig
    ):
        self.optimizer = optimizer
        self.config = config
        self.samples = 0
        self.last_training = None
        self.feature_importances: Dict[str, float] = {}
        
        # Initialize models
        self.impact_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        self.priority_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.success_model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        
        # Load models if available
        if config.store_models:
            self.load_models()
    
    def process_suggestion(
        self,
        suggestion: OptimizationSuggestion,
        composition: List[ScheduledEvent],
        features: Dict[str, Any]
    ):
        """Process and potentially update suggestion based on learning."""
        if not self._has_sufficient_data():
            return suggestion
        
        # Extract features
        X = self._extract_features(suggestion, composition, features)
        
        try:
            # Predict impact and priority
            predicted_impact = float(self.impact_model.predict([X])[0])
            predicted_priority = int(self.priority_model.predict([X])[0])
            success_prob = float(self.success_model.predict_proba([X])[0][1])
            
            # Adjust suggestion based on predictions
            suggestion.impact = (suggestion.impact + predicted_impact) / 2
            suggestion.priority = predicted_priority
            
            # Add prediction confidence info
            suggestion.rationale = (
                f"{suggestion.rationale}\n"
                f"Success probability: {success_prob:.2%}"
            )
            
        except Exception as e:
            logger.warning(f"Failed to apply learning: {e}")
        
        return suggestion
    
    def record_feedback(
        self,
        suggestion: OptimizationSuggestion,
        composition: List[ScheduledEvent],
        features: Dict[str, Any],
        success: bool,
        actual_impact: Optional[float] = None
    ):
        """Record feedback on suggestion effectiveness."""
        try:
            # Store feedback
            feedback = {
                "timestamp": datetime.now().isoformat(),
                "suggestion_type": suggestion.type,
                "original_impact": suggestion.impact,
                "actual_impact": actual_impact or suggestion.impact,
                "priority": suggestion.priority,
                "success": success,
                "composition_size": len(composition),
                "features": features
            }
            
            self._store_feedback(feedback)
            self.samples += 1
            
            # Check if retraining needed
            if (
                self.config.auto_retrain and
                self.samples % self.config.retrain_interval == 0
            ):
                self.retrain_models()
            
        except Exception as e:
            logger.error(f"Failed to record feedback: {e}")
    
    def retrain_models(self):
        """Retrain models on accumulated data."""
        try:
            # Load feedback data
            feedback_data = self._load_feedback_data()
            if not feedback_data:
                return
            
            # Prepare training data
            X, y_impact, y_priority, y_success = self._prepare_training_data(
                feedback_data
            )
            
            if len(X) < self.config.min_samples:
                return
            
            # Train models
            self.impact_model.fit(X, y_impact)
            self.priority_model.fit(X, y_priority)
            self.success_model.fit(X, y_success)
            
            # Update feature importances
            self.feature_importances = self._calculate_feature_importances()
            
            # Save models
            if self.config.store_models:
                self.save_models()
            
            self.last_training = datetime.now()
            logger.info("Successfully retrained optimization models")
            
        except Exception as e:
            logger.error(f"Failed to retrain models: {e}")
    
    def _extract_features(
        self,
        suggestion: OptimizationSuggestion,
        composition: List[ScheduledEvent],
        features: Dict[str, Any]
    ) -> List[float]:
        """Extract features for prediction."""
        suggestion_features = {
            "type": suggestion.type,
            "complexity": suggestion.complexity
        }
        
        composition_features = {
            "size": len(composition),
            "unique_events": len(set(
                event.event.name if hasattr(event.event, "name")
                else event.event
                for event in composition
            )),
            "has_conditions": any(event.condition for event in composition),
            "has_repeats": any(event.repeat for event in composition)
        }
        
        # Combine all features
        all_features = {
            **suggestion_features,
            **composition_features,
            **features
        }
        
        # Encode categorical features
        encoded_features = []
        for key, value in sorted(all_features.items()):
            if isinstance(value, (str, bool)):
                if key not in self.label_encoders:
                    self.label_encoders[key] = LabelEncoder()
                    self.label_encoders[key].fit([value])
                try:
                    encoded_value = self.label_encoders[key].transform([value])[0]
                except ValueError:
                    # Handle unseen categories
                    self.label_encoders[key].fit([value])
                    encoded_value = self.label_encoders[key].transform([value])[0]
                encoded_features.append(float(encoded_value))
            else:
                encoded_features.append(float(value))
        
        return encoded_features
    
    def _prepare_training_data(
        self,
        feedback_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from feedback."""
        X = []
        y_impact = []
        y_priority = []
        y_success = []
        
        for feedback in feedback_data:
            try:
                features = self._extract_features(
                    OptimizationSuggestion(
                        type=feedback["suggestion_type"],
                        description="",
                        impact=feedback["original_impact"],
                        complexity=1,
                        priority=feedback["priority"]
                    ),
                    [],  # Placeholder for composition
                    feedback["features"]
                )
                
                X.append(features)
                y_impact.append(feedback["actual_impact"])
                y_priority.append(feedback["priority"])
                y_success.append(feedback["success"])
                
            except Exception as e:
                logger.warning(f"Failed to process feedback entry: {e}")
        
        return np.array(X), np.array(y_impact), np.array(y_priority), np.array(y_success)
    
    def _calculate_feature_importances(self) -> Dict[str, float]:
        """Calculate feature importances from models."""
        importances = defaultdict(float)
        
        # Combine importances from all models
        models = {
            "impact": self.impact_model,
            "priority": self.priority_model,
            "success": self.success_model
        }
        
        for name, model in models.items():
            for i, importance in enumerate(model.feature_importances_):
                importances[f"feature_{i}"] += importance / len(models)
        
        return dict(importances)
    
    def _has_sufficient_data(self) -> bool:
        """Check if sufficient data for predictions."""
        return self.samples >= self.config.min_samples
    
    def _store_feedback(
        self,
        feedback: Dict[str, Any]
    ):
        """Store optimization feedback."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            feedback_file = output_path / "optimization_feedback.jsonl"
            with open(feedback_file, "a") as f:
                f.write(json.dumps(feedback) + "\n")
            
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
    
    def _load_feedback_data(self) -> List[Dict[str, Any]]:
        """Load historical feedback data."""
        if not self.config.output_path:
            return []
        
        feedback_data = []
        try:
            feedback_file = self.config.output_path / "optimization_feedback.jsonl"
            if not feedback_file.exists():
                return []
            
            with open(feedback_file) as f:
                for line in f:
                    feedback_data.append(json.loads(line.strip()))
            
            # Limit history size
            if len(feedback_data) > self.config.max_history:
                feedback_data = feedback_data[-self.config.max_history:]
            
        except Exception as e:
            logger.error(f"Failed to load feedback data: {e}")
        
        return feedback_data
    
    def save_models(self):
        """Save trained models."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save models
            joblib.dump(
                self.impact_model,
                output_path / "impact_model.joblib"
            )
            joblib.dump(
                self.priority_model,
                output_path / "priority_model.joblib"
            )
            joblib.dump(
                self.success_model,
                output_path / "success_model.joblib"
            )
            
            # Save label encoders
            joblib.dump(
                self.label_encoders,
                output_path / "label_encoders.joblib"
            )
            
            # Save feature importances
            with open(output_path / "feature_importances.json", "w") as f:
                json.dump(self.feature_importances, f, indent=2)
            
            logger.info(f"Saved models to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def load_models(self):
        """Load trained models."""
        if not self.config.output_path:
            return
        
        try:
            # Check if model files exist
            model_files = [
                "impact_model.joblib",
                "priority_model.joblib",
                "success_model.joblib",
                "label_encoders.joblib"
            ]
            
            if not all(
                (self.config.output_path / f).exists()
                for f in model_files
            ):
                return
            
            # Load models
            self.impact_model = joblib.load(
                self.config.output_path / "impact_model.joblib"
            )
            self.priority_model = joblib.load(
                self.config.output_path / "priority_model.joblib"
            )
            self.success_model = joblib.load(
                self.config.output_path / "success_model.joblib"
            )
            
            # Load label encoders
            self.label_encoders = joblib.load(
                self.config.output_path / "label_encoders.joblib"
            )
            
            # Load feature importances
            importances_file = self.config.output_path / "feature_importances.json"
            if importances_file.exists():
                with open(importances_file) as f:
                    self.feature_importances = json.load(f)
            
            logger.info(f"Loaded models from {self.config.output_path}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")

def create_optimization_learner(
    optimizer: CompositionOptimizer,
    output_path: Optional[Path] = None
) -> OptimizationLearner:
    """Create optimization learner."""
    config = LearningConfig(output_path=output_path)
    return OptimizationLearner(optimizer, config)

if __name__ == "__main__":
    # Example usage
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
    learner = create_optimization_learner(
        optimizer,
        output_path=Path("optimization_learning")
    )
    
    # Example composition
    events_a = ["animation:start", "animation:start", "progress:update"]
    events_b = ["animation:pause", "animation:resume", "animation:pause"]
    
    sequence_a = pattern.sequence(events_a)
    sequence_b = pattern.sequence(events_b)
    
    composition = composer.compose(
        "chain",
        [sequence_a, sequence_b],
        delay=1.0,
        gap=0.5
    )
    
    # Get suggestions with learning
    suggestions = optimizer.analyze_and_suggest(composition)
    
    # Process suggestions
    features = {
        "timing_efficiency": 0.8,
        "resource_utilization": 0.7,
        "complexity_score": 0.5
    }
    
    for suggestion in suggestions:
        # Apply learning
        enhanced_suggestion = learner.process_suggestion(
            suggestion,
            composition,
            features
        )
        
        # Record feedback
        learner.record_feedback(
            enhanced_suggestion,
            composition,
            features,
            success=True,
            actual_impact=0.9
        )
    
    # Retrain models
    learner.retrain_models()
