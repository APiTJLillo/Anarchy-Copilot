"""Dynamic priority adjustment for notifications."""

import asyncio
from typing import Dict, List, Any, Optional, Set, DefaultDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import numpy as np
from collections import defaultdict
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

from .notification_priority import Priority, PriorityRouter, PrioritizedNotification
from .notification_throttling import NotificationThrottler

logger = logging.getLogger(__name__)

@dataclass
class PriorityAdjustmentConfig:
    """Configuration for dynamic priority adjustment."""
    learning_rate: float = 0.1  # Rate of priority adjustment
    history_window: int = 1000  # Max historical notifications to consider
    min_samples: int = 50  # Min samples before making predictions
    update_interval: int = 100  # Updates per N notifications
    feature_weights: Dict[str, float] = field(default_factory=lambda: {
        "time_of_day": 1.0,
        "day_of_week": 0.8,
        "notification_rate": 1.2,
        "response_time": 1.5,
        "feedback_score": 2.0
    })
    model_path: Optional[Path] = None

@dataclass
class NotificationFeedback:
    """Feedback for sent notification."""
    notification_id: str
    priority: Priority
    timestamp: datetime
    response_time: Optional[float] = None
    acknowledged: bool = False
    action_taken: bool = False
    feedback_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PriorityLearner:
    """Learn and adjust notification priorities."""
    
    def __init__(
        self,
        router: PriorityRouter,
        config: PriorityAdjustmentConfig = None
    ):
        self.router = router
        self.config = config or PriorityAdjustmentConfig()
        
        # History tracking
        self.notification_history: List[PrioritizedNotification] = []
        self.feedback_history: Dict[str, NotificationFeedback] = {}
        
        # Pattern recognition
        self.pattern_features: Dict[str, List[float]] = defaultdict(list)
        self.priority_adjustments: Dict[str, float] = defaultdict(float)
        
        # ML components
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100)
        self.model_trained = False
        
        # Load existing model if available
        self._load_model()
    
    def _load_model(self):
        """Load trained model if available."""
        if self.config.model_path and self.config.model_path.exists():
            try:
                model_data = joblib.load(self.config.model_path)
                self.model = model_data["model"]
                self.scaler = model_data["scaler"]
                self.model_trained = True
                logger.info("Loaded existing priority model")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    def _save_model(self):
        """Save trained model."""
        if self.config.model_path:
            try:
                model_data = {
                    "model": self.model,
                    "scaler": self.scaler,
                    "timestamp": datetime.now()
                }
                joblib.dump(model_data, self.config.model_path)
                logger.info("Saved priority model")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
    
    def _extract_features(
        self,
        notification: PrioritizedNotification,
        feedback: Optional[NotificationFeedback] = None
    ) -> Dict[str, float]:
        """Extract features for priority prediction."""
        timestamp = notification.timestamp
        features = {
            "time_of_day": timestamp.hour + timestamp.minute / 60.0,
            "day_of_week": timestamp.weekday(),
            "notification_rate": self._get_notification_rate(),
            "content_length": len(json.dumps(notification.content)),
            "channel_count": len(notification.channels) if notification.channels else 0
        }
        
        if feedback:
            features.update({
                "response_time": feedback.response_time or -1,
                "acknowledged": float(feedback.acknowledged),
                "action_taken": float(feedback.action_taken),
                "feedback_score": feedback.feedback_score or 0.0
            })
        
        return features
    
    def _get_notification_rate(
        self,
        window: int = 3600
    ) -> float:
        """Calculate recent notification rate."""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window)
        
        recent = [
            n for n in self.notification_history
            if n.timestamp >= cutoff
        ]
        
        return len(recent) / (window / 3600)  # Notifications per hour
    
    def _adjust_priority(
        self,
        original: Priority,
        adjustment: float
    ) -> Priority:
        """Adjust priority based on learned patterns."""
        current_value = original.value
        adjusted_value = current_value + adjustment
        
        # Find closest priority
        priorities = list(Priority)
        closest = min(
            priorities,
            key=lambda p: abs(p.value - adjusted_value)
        )
        
        return closest
    
    async def record_notification(
        self,
        notification: PrioritizedNotification
    ):
        """Record sent notification."""
        self.notification_history.append(notification)
        
        # Trim history if needed
        if len(self.notification_history) > self.config.history_window:
            self.notification_history = self.notification_history[
                -self.config.history_window:
            ]
        
        # Update patterns periodically
        if len(self.notification_history) % self.config.update_interval == 0:
            await self._update_patterns()
    
    async def record_feedback(
        self,
        feedback: NotificationFeedback
    ):
        """Record notification feedback."""
        self.feedback_history[feedback.notification_id] = feedback
        
        # Update model with new feedback
        if len(self.feedback_history) >= self.config.min_samples:
            await self._update_model()
    
    async def _update_patterns(self):
        """Update notification patterns."""
        if not self.notification_history:
            return
        
        # Analyze temporal patterns
        hours = [n.timestamp.hour for n in self.notification_history]
        days = [n.timestamp.weekday() for n in self.notification_history]
        
        # Calculate priority adjustments
        for notification in self.notification_history[-self.config.update_interval:]:
            features = self._extract_features(notification)
            
            # Apply feature weights
            weighted_features = {
                k: v * self.config.feature_weights.get(k, 1.0)
                for k, v in features.items()
            }
            
            # Calculate adjustment
            adjustment = sum(weighted_features.values()) / len(weighted_features)
            adjustment *= self.config.learning_rate
            
            key = (
                notification.title,
                tuple(sorted(notification.channels or []))
            )
            self.priority_adjustments[key] = adjustment
    
    async def _update_model(self):
        """Update prediction model."""
        if len(self.feedback_history) < self.config.min_samples:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for notification in self.notification_history:
            feedback = self.feedback_history.get(notification.title)
            if not feedback:
                continue
            
            features = self._extract_features(notification, feedback)
            X.append(list(features.values()))
            y.append(notification.priority.value)
        
        if not X:
            return
        
        # Train model
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.model_trained = True
        
        # Save updated model
        self._save_model()
    
    async def suggest_priority(
        self,
        title: str,
        content: Dict[str, Any],
        channels: Optional[List[str]] = None
    ) -> Priority:
        """Suggest notification priority."""
        # Use ML model if available
        if self.model_trained:
            features = self._extract_features(
                PrioritizedNotification(
                    priority=Priority.MEDIUM,
                    timestamp=datetime.now(),
                    title=title,
                    content=content,
                    channels=channels
                )
            )
            
            X = self.scaler.transform([list(features.values())])
            predicted_value = self.model.predict(X)[0]
            
            # Find closest priority
            suggested = min(
                Priority,
                key=lambda p: abs(p.value - predicted_value)
            )
        else:
            # Use pattern-based adjustment
            key = (title, tuple(sorted(channels or [])))
            adjustment = self.priority_adjustments.get(key, 0)
            suggested = self._adjust_priority(Priority.MEDIUM, adjustment)
        
        return suggested
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get priority learning metrics."""
        return {
            "notification_count": len(self.notification_history),
            "feedback_count": len(self.feedback_history),
            "model_trained": self.model_trained,
            "priority_distribution": {
                p.name: len([
                    n for n in self.notification_history
                    if n.priority == p
                ])
                for p in Priority
            },
            "average_feedback_score": np.mean([
                f.feedback_score
                for f in self.feedback_history.values()
                if f.feedback_score is not None
            ]) if self.feedback_history else None,
            "pattern_strength": {
                key: abs(adj)
                for key, adj in self.priority_adjustments.items()
            }
        }

def create_priority_learner(
    router: PriorityRouter,
    config: Optional[PriorityAdjustmentConfig] = None
) -> PriorityLearner:
    """Create priority learner."""
    return PriorityLearner(router, config)

if __name__ == "__main__":
    # Example usage
    from .notification_priority import create_priority_router
    from .notification_throttling import create_throttled_manager
    from .notification_channels import create_notification_manager
    
    async def main():
        # Create notification stack
        manager = create_notification_manager()
        throttler = create_throttled_manager(manager)
        router = create_priority_router(throttler)
        learner = create_priority_learner(router)
        
        # Simulate notifications and feedback
        for i in range(100):
            # Create notification
            title = f"Test Notification {i}"
            content = {"value": i}
            
            # Get suggested priority
            priority = await learner.suggest_priority(title, content)
            
            # Send notification
            notification = PrioritizedNotification(
                priority=priority,
                timestamp=datetime.now(),
                title=title,
                content=content,
                channels=None
            )
            await learner.record_notification(notification)
            
            # Simulate feedback
            if i % 3 == 0:
                feedback = NotificationFeedback(
                    notification_id=title,
                    priority=priority,
                    timestamp=datetime.now(),
                    response_time=np.random.exponential(300),
                    acknowledged=np.random.random() > 0.3,
                    action_taken=np.random.random() > 0.5,
                    feedback_score=np.random.normal(0.7, 0.2)
                )
                await learner.record_feedback(feedback)
            
            await asyncio.sleep(0.1)
        
        # Check metrics
        metrics = learner.get_metrics()
        print("Learning Metrics:", json.dumps(metrics, indent=2))
    
    asyncio.run(main())
