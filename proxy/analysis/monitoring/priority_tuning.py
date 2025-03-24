"""Automated hyperparameter tuning for priority prediction."""

import asyncio
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import optuna
import joblib
from pathlib import Path
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .priority_validation import PriorityValidator, ValidationConfig
from .adaptive_priority import PriorityLearner, PriorityAdjustmentConfig

logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 100  # Number of optimization trials
    cv_folds: int = 5  # Cross-validation folds
    timeout: int = 3600  # Max tuning time in seconds
    study_name: str = "priority_tuning"
    storage: Optional[str] = None  # Optuna storage URL
    metric: str = "accuracy"  # Optimization metric
    export_path: Optional[Path] = None

@dataclass
class TuningResult:
    """Results from hyperparameter tuning."""
    best_params: Dict[str, Any]
    best_score: float
    all_trials: List[Dict[str, Any]]
    study_info: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)

class PriorityTuner:
    """Tune priority prediction hyperparameters."""
    
    def __init__(
        self,
        validator: PriorityValidator,
        config: TuningConfig = None
    ):
        self.validator = validator
        self.config = config or TuningConfig()
        self.tuning_history: List[TuningResult] = []
        self.current_study: Optional[optuna.Study] = None
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study."""
        storage = self.config.storage
        if storage and not storage.startswith(("sqlite:///", "postgresql://")):
            storage = f"sqlite:///{storage}"
        
        return optuna.create_study(
            study_name=self.config.study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True
        )
    
    def _suggest_params(
        self,
        trial: optuna.Trial
    ) -> Dict[str, Any]:
        """Suggest hyperparameters for trial."""
        params = {
            # RandomForest params
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            
            # Priority adjustment params
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "batch_size": trial.suggest_int("batch_size", 32, 256),
            
            # Feature weights
            "time_weight": trial.suggest_float("time_weight", 0.1, 2.0),
            "rate_weight": trial.suggest_float("rate_weight", 0.1, 2.0),
            "feedback_weight": trial.suggest_float("feedback_weight", 0.5, 3.0)
        }
        
        return params
    
    async def _objective(
        self,
        trial: optuna.Trial
    ) -> float:
        """Optimization objective function."""
        # Get hyperparameter suggestions
        params = self._suggest_params(trial)
        
        # Update model parameters
        self.validator.learner.model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"]
        )
        
        # Update learning parameters
        self.validator.learner.config.learning_rate = params["learning_rate"]
        
        # Update feature weights
        self.validator.learner.config.feature_weights.update({
            "time_of_day": params["time_weight"],
            "notification_rate": params["rate_weight"],
            "feedback_score": params["feedback_weight"]
        })
        
        # Run validation
        result = await self.validator.validate_model()
        if not result:
            return 0.0
        
        # Get optimization metric
        if self.config.metric == "accuracy":
            score = result["cross_validation"]["mean_accuracy"]
        elif self.config.metric == "f1":
            scores = [
                m["f1-score"]
                for m in result["performance"].values()
                if isinstance(m, dict)
            ]
            score = np.mean(scores)
        else:
            raise ValueError(f"Unknown metric: {self.config.metric}")
        
        return score
    
    async def tune_model(self) -> TuningResult:
        """Run hyperparameter tuning."""
        study = self._create_study()
        self.current_study = study
        
        try:
            # Run optimization
            await optuna.study.run_study(
                study,
                self._objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout
            )
            
            # Collect results
            result = TuningResult(
                best_params=study.best_params,
                best_score=study.best_value,
                all_trials=[
                    {
                        "number": t.number,
                        "params": t.params,
                        "value": t.value
                    }
                    for t in study.trials
                ],
                study_info={
                    "n_trials": len(study.trials),
                    "duration": study.trials[-1].datetime_complete - study.trials[0].datetime_start,
                    "state": study.system_attrs
                }
            )
            
            # Apply best parameters
            await self._apply_best_params(result.best_params)
            
            # Save results
            self.tuning_history.append(result)
            await self._export_tuning_report(result)
            
            return result
            
        finally:
            self.current_study = None
    
    async def _apply_best_params(
        self,
        params: Dict[str, Any]
    ):
        """Apply best hyperparameters."""
        # Update model
        self.validator.learner.model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"]
        )
        
        # Update learning config
        self.validator.learner.config.learning_rate = params["learning_rate"]
        
        # Update feature weights
        self.validator.learner.config.feature_weights.update({
            "time_of_day": params["time_weight"],
            "notification_rate": params["rate_weight"],
            "feedback_score": params["feedback_weight"]
        })
        
        # Retrain model
        await self.validator.validate_model()
    
    async def _export_tuning_report(
        self,
        result: TuningResult
    ):
        """Export tuning results."""
        if not self.config.export_path:
            return
        
        try:
            # Create export directory
            report_dir = Path(self.config.export_path)
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Export results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = report_dir / f"tuning_results_{timestamp}.json"
            
            with open(results_file, "w") as f:
                json.dump({
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                    "study_info": result.study_info,
                    "timestamp": result.timestamp.isoformat()
                }, f, indent=2)
            
            # Create visualizations
            await self.create_tuning_plots(
                report_dir / f"tuning_plots_{timestamp}.html"
            )
            
            logger.info(f"Exported tuning report to {report_dir}")
            
        except Exception as e:
            logger.error(f"Failed to export tuning report: {e}")
    
    async def create_tuning_plots(
        self,
        output_path: Optional[Path] = None
    ) -> go.Figure:
        """Create tuning visualizations."""
        if not self.tuning_history:
            return None
        
        latest = self.tuning_history[-1]
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Parameter Importance",
                "Optimization History",
                "Parallel Coordinates",
                "Contour Plot"
            ]
        )
        
        # Parameter importance
        study = self.current_study or self._create_study()
        importances = optuna.importance.get_param_importances(study)
        
        fig.add_trace(
            go.Bar(
                x=list(importances.values()),
                y=list(importances.keys()),
                orientation="h"
            ),
            row=1,
            col=1
        )
        
        # Optimization history
        values = [t["value"] for t in latest.all_trials]
        best_values = np.maximum.accumulate(values)
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode="markers",
                name="Trial"
            ),
            row=1,
            col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(best_values))),
                y=best_values,
                mode="lines",
                name="Best"
            ),
            row=1,
            col=2
        )
        
        # Parallel coordinates
        dims = []
        for param in latest.best_params:
            values = [t["params"][param] for t in latest.all_trials]
            dims.append(
                dict(
                    range=[min(values), max(values)],
                    label=param,
                    values=values
                )
            )
        
        dims.append(
            dict(
                range=[min(values), max(values)],
                label="Score",
                values=[t["value"] for t in latest.all_trials]
            )
        )
        
        fig.add_trace(
            go.Parcoords(
                line=dict(
                    color=[t["value"] for t in latest.all_trials],
                    colorscale="Viridis"
                ),
                dimensions=dims
            ),
            row=2,
            col=1
        )
        
        # Contour plot of top 2 parameters
        top_params = sorted(
            importances.items(),
            key=lambda x: x[1],
            reverse=True
        )[:2]
        
        if len(top_params) >= 2:
            param1, param2 = top_params[0][0], top_params[1][0]
            x = [t["params"][param1] for t in latest.all_trials]
            y = [t["params"][param2] for t in latest.all_trials]
            z = [t["value"] for t in latest.all_trials]
            
            fig.add_trace(
                go.Contour(
                    x=np.unique(x),
                    y=np.unique(y),
                    z=np.reshape(z, (len(np.unique(x)), len(np.unique(y)))),
                    colorscale="Viridis",
                    contours=dict(
                        coloring="heatmap"
                    )
                ),
                row=2,
                col=2
            )
        
        fig.update_layout(
            height=800,
            title="Hyperparameter Tuning Analysis"
        )
        
        if output_path:
            fig.write_html(str(output_path))
        
        return fig

def create_priority_tuner(
    validator: PriorityValidator,
    config: Optional[TuningConfig] = None
) -> PriorityTuner:
    """Create priority tuner."""
    return PriorityTuner(validator, config)

if __name__ == "__main__":
    # Example usage
    from .priority_validation import create_priority_validator
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
        tuner = create_priority_tuner(
            validator,
            TuningConfig(export_path=Path("tuning_reports"))
        )
        
        # Run tuning
        result = await tuner.tune_model()
        print("Best Parameters:", json.dumps(result.best_params, indent=2))
        print(f"Best Score: {result.best_score:.4f}")
        
        # Create plots
        fig = await tuner.create_tuning_plots()
        fig.show()
    
    asyncio.run(main())
