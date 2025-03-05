"""Model validation and analysis for priority learning."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
from pathlib import Path
import joblib

from .adaptive_priority import PriorityLearner, PriorityAdjustmentConfig
from .notification_priority import Priority

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    n_splits: int = 5  # Number of CV folds
    test_size: float = 0.2  # Size of test set
    min_confidence: float = 0.7  # Minimum prediction confidence
    validation_interval: int = 1000  # Validate every N notifications
    export_path: Optional[Path] = None  # Path for validation reports

class PriorityValidator:
    """Validate priority prediction model."""
    
    def __init__(
        self,
        learner: PriorityLearner,
        config: ValidationConfig = None
    ):
        self.learner = learner
        self.config = config or ValidationConfig()
        self.validation_history: List[Dict[str, Any]] = []
        self.feature_importance: Dict[str, float] = {}
    
    async def validate_model(self) -> Dict[str, Any]:
        """Run model validation."""
        if not self.learner.model_trained:
            return {}
        
        # Prepare validation data
        X = []
        y = []
        
        for notification in self.learner.notification_history:
            feedback = self.learner.feedback_history.get(notification.title)
            if not feedback:
                continue
            
            features = self.learner._extract_features(notification, feedback)
            X.append(list(features.values()))
            y.append(notification.priority.value)
        
        if not X:
            return {}
        
        X = np.array(X)
        y = np.array(y)
        
        # Cross validation
        cv_results = await self._run_cross_validation(X, y)
        
        # Feature importance
        feature_importance = await self._analyze_feature_importance(X, y)
        
        # Performance metrics
        performance = await self._calculate_performance_metrics(X, y)
        
        # Confusion matrix
        confusion = await self._calculate_confusion_matrix(X, y)
        
        # Combine results
        validation_result = {
            "timestamp": datetime.now(),
            "cross_validation": cv_results,
            "feature_importance": feature_importance,
            "performance": performance,
            "confusion_matrix": confusion
        }
        
        self.validation_history.append(validation_result)
        self.feature_importance = feature_importance
        
        # Export validation report
        await self._export_validation_report(validation_result)
        
        return validation_result
    
    async def _run_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Run k-fold cross validation."""
        kf = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=42
        )
        
        # Calculate scores
        scores = cross_val_score(
            self.learner.model,
            X,
            y,
            cv=kf,
            scoring="accuracy"
        )
        
        return {
            "mean_accuracy": float(np.mean(scores)),
            "std_accuracy": float(np.std(scores)),
            "fold_scores": [float(s) for s in scores]
        }
    
    async def _analyze_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Analyze feature importance."""
        feature_names = list(
            self.learner._extract_features(
                next(iter(self.learner.notification_history))
            ).keys()
        )
        
        # Get feature importance from model
        importance = self.learner.model.feature_importances_
        
        return {
            name: float(imp)
            for name, imp in zip(feature_names, importance)
        }
    
    async def _calculate_performance_metrics(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        # Split data
        n_test = int(len(X) * self.config.test_size)
        X_test = X[-n_test:]
        y_test = y[-n_test:]
        
        # Get predictions
        y_pred = self.learner.model.predict(X_test)
        
        # Calculate metrics
        report = classification_report(
            y_test,
            y_pred,
            target_names=[p.name for p in Priority],
            output_dict=True
        )
        
        return report
    
    async def _calculate_confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, List[List[int]]]:
        """Calculate confusion matrix."""
        # Split data
        n_test = int(len(X) * self.config.test_size)
        X_test = X[-n_test:]
        y_test = y[-n_test:]
        
        # Get predictions
        y_pred = self.learner.model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            "matrix": cm.tolist(),
            "labels": [p.name for p in Priority]
        }
    
    async def _export_validation_report(
        self,
        result: Dict[str, Any]
    ):
        """Export validation report."""
        if not self.config.export_path:
            return
        
        try:
            # Create export directory
            report_dir = Path(self.config.export_path)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Export metrics
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = report_dir / f"validation_metrics_{timestamp}.json"
            with open(metrics_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
            
            # Create visualizations
            await self.create_validation_plots(
                report_dir / f"validation_plots_{timestamp}.html"
            )
            
            logger.info(f"Exported validation report to {report_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")
    
    async def create_validation_plots(
        self,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Create validation visualizations."""
        if not self.validation_history:
            return None
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Cross Validation Scores",
                "Feature Importance",
                "Performance Metrics",
                "Confusion Matrix"
            ]
        )
        
        # CV scores plot
        scores = [
            v["cross_validation"]["fold_scores"]
            for v in self.validation_history
        ]
        
        fig.add_trace(
            go.Box(
                y=np.array(scores).flatten(),
                name="CV Scores"
            ),
            row=1,
            col=1
        )
        
        # Feature importance plot
        importance = pd.DataFrame(self.feature_importance.items())
        importance.columns = ["Feature", "Importance"]
        importance = importance.sort_values("Importance", ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=importance["Importance"],
                y=importance["Feature"],
                orientation="h",
                name="Importance"
            ),
            row=1,
            col=2
        )
        
        # Performance metrics plot
        latest = self.validation_history[-1]
        metrics = pd.DataFrame(latest["performance"]).T
        metrics = metrics[metrics.index.isin(Priority.__members__.keys())]
        
        fig.add_trace(
            go.Heatmap(
                z=metrics[["precision", "recall", "f1-score"]].values,
                x=["Precision", "Recall", "F1"],
                y=metrics.index,
                colorscale="Viridis"
            ),
            row=2,
            col=1
        )
        
        # Confusion matrix plot
        cm = np.array(latest["confusion_matrix"]["matrix"])
        labels = latest["confusion_matrix"]["labels"]
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=labels,
                y=labels,
                colorscale="Reds"
            ),
            row=2,
            col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title="Priority Model Validation"
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get validation metrics summary."""
        if not self.validation_history:
            return {}
        
        latest = self.validation_history[-1]
        cv = latest["cross_validation"]
        perf = latest["performance"]
        
        return {
            "accuracy": {
                "mean": cv["mean_accuracy"],
                "std": cv["std_accuracy"]
            },
            "top_features": dict(
                sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ),
            "performance": {
                priority: {
                    "precision": perf[priority]["precision"],
                    "recall": perf[priority]["recall"],
                    "f1": perf[priority]["f1-score"]
                }
                for priority in Priority.__members__.keys()
            }
        }

def create_priority_validator(
    learner: PriorityLearner,
    config: Optional[ValidationConfig] = None
) -> PriorityValidator:
    """Create priority validator."""
    return PriorityValidator(learner, config)

if __name__ == "__main__":
    # Example usage
    from .adaptive_priority import create_priority_learner
    from .notification_priority import create_priority_router
    from .notification_throttling import create_throttled_manager
    from .notification_channels import create_notification_manager
    
    async def main():
        # Create notification stack
        manager = create_notification_manager()
        throttler = create_throttled_manager(manager)
        router = create_priority_router(throttler)
        learner = create_priority_learner(router)
        validator = create_priority_validator(
            learner,
            ValidationConfig(export_path=Path("validation_reports"))
        )
        
        # Train model with example data
        # ... (training code) ...
        
        # Run validation
        result = await validator.validate_model()
        print("Validation Summary:", json.dumps(
            validator.get_validation_summary(),
            indent=2
        ))
        
        # Create plots
        fig = await validator.create_validation_plots()
        fig.show()
    
    asyncio.run(main())
