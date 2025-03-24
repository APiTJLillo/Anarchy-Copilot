"""Hyperparameter tuning for profile learning models."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from .profile_learning import ProfileLearning, LearningConfig

logger = logging.getLogger(__name__)

@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 50
    cv_folds: int = 5
    n_jobs: int = -1
    timeout: Optional[float] = 3600
    use_ray: bool = False
    ray_cpus: int = 4
    early_stopping: bool = True
    output_path: Optional[Path] = None

class HyperparameterTuner:
    """Tune model hyperparameters."""
    
    def __init__(
        self,
        learning: ProfileLearning,
        config: TuningConfig
    ):
        self.learning = learning
        self.config = config
        self.best_params: Dict[str, Dict[str, Any]] = {}
        self.study_results: Dict[str, Any] = {}
    
    def tune_random_forest(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune Random Forest hyperparameters."""
        param_space = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["auto", "sqrt", "log2"],
            "bootstrap": [True, False]
        }
        
        model = RandomForestRegressor(random_state=42)
        
        results = self._perform_grid_search(
            model,
            param_space,
            X,
            y,
            "rf"
        )
        
        self.best_params["rf"] = results["best_params"]
        return results
    
    def tune_xgboost(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),
                "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_loguniform("gamma", 1e-3, 1.0)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            
            scores = self._cross_validate_model(model, X, y)
            return np.mean(scores["test_neg_mean_squared_error"])
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        results = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "study_statistics": {
                "completed_trials": len(study.trials),
                "pruned_trials": len([
                    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
                ]),
                "duration": sum(
                    t.duration.total_seconds()
                    for t in study.trials
                    if t.duration
                )
            }
        }
        
        self.best_params["xgb"] = results["best_params"]
        self.study_results["xgb"] = study
        
        return results
    
    def tune_neural_network(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune Neural Network hyperparameters."""
        def objective(config):
            model = MLPRegressor(
                hidden_layer_sizes=tuple(config["hidden_layers"]),
                activation=config["activation"],
                solver=config["solver"],
                alpha=config["alpha"],
                learning_rate=config["learning_rate"],
                max_iter=200,
                random_state=42
            )
            
            scores = self._cross_validate_model(model, X, y)
            tune.report(loss=np.mean(-scores["test_neg_mean_squared_error"]))
        
        search_space = {
            "hidden_layers": tune.choice([
                [64, 32],
                [128, 64],
                [256, 128],
                [64, 32, 16],
                [128, 64, 32]
            ]),
            "activation": tune.choice(["relu", "tanh"]),
            "solver": tune.choice(["adam", "sgd"]),
            "alpha": tune.loguniform(1e-5, 1e-2),
            "learning_rate": tune.choice(["constant", "adaptive"])
        }
        
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=100,
            grace_period=10
        )
        
        if self.config.use_ray:
            analysis = tune.run(
                objective,
                config=search_space,
                scheduler=scheduler,
                num_samples=self.config.n_trials,
                resources_per_trial={"cpu": self.config.ray_cpus}
            )
            
            results = {
                "best_params": analysis.best_config,
                "best_value": float(analysis.best_result["loss"]),
                "n_trials": len(analysis.trials),
                "analysis_statistics": {
                    "total_time": analysis.time_total_s,
                    "completed_trials": len(analysis.trials)
                }
            }
        else:
            # Fallback to grid search if Ray is not enabled
            param_grid = {
                "hidden_layer_sizes": [(64, 32), (128, 64), (256, 128)],
                "activation": ["relu", "tanh"],
                "solver": ["adam", "sgd"],
                "alpha": [1e-4, 1e-3, 1e-2],
                "learning_rate": ["constant", "adaptive"]
            }
            
            model = MLPRegressor(random_state=42)
            results = self._perform_grid_search(
                model,
                param_grid,
                X,
                y,
                "nn"
            )
        
        self.best_params["nn"] = results["best_params"]
        return results
    
    def tune_lstm(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """Tune LSTM hyperparameters."""
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(),
            pruner=MedianPruner()
        )
        
        def objective(trial):
            # Build model with trial parameters
            model = tf.keras.Sequential([
                tf.keras.layers.LSTM(
                    units=trial.suggest_int("lstm_units", 32, 256),
                    input_shape=(1, X.shape[1])
                ),
                tf.keras.layers.Dropout(
                    trial.suggest_uniform("dropout", 0.1, 0.5)
                ),
                tf.keras.layers.Dense(
                    trial.suggest_int("dense_units", 16, 128),
                    activation="relu"
                ),
                tf.keras.layers.Dense(1)
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=trial.suggest_loguniform(
                        "learning_rate",
                        1e-4,
                        1e-2
                    )
                ),
                loss="mse"
            )
            
            # Train with early stopping
            if self.config.early_stopping:
                callback = EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    restore_best_weights=True
                )
                callbacks = [callback]
            else:
                callbacks = []
            
            # Reshape input for LSTM
            X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
            
            history = model.fit(
                X_reshaped,
                y,
                epochs=trial.suggest_int("epochs", 50, 200),
                batch_size=trial.suggest_int("batch_size", 16, 128),
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )
            
            return history.history["val_loss"][-1]
        
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout
        )
        
        results = {
            "best_params": study.best_params,
            "best_value": float(study.best_value),
            "n_trials": len(study.trials),
            "study_statistics": {
                "completed_trials": len(study.trials),
                "pruned_trials": len([
                    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
                ]),
                "duration": sum(
                    t.duration.total_seconds()
                    for t in study.trials
                    if t.duration
                )
            }
        }
        
        self.best_params["lstm"] = results["best_params"]
        self.study_results["lstm"] = study
        
        return results
    
    def _perform_grid_search(
        self,
        model: Any,
        param_grid: Dict[str, List[Any]],
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> Dict[str, Any]:
        """Perform grid search with cross-validation."""
        search = GridSearchCV(
            model,
            param_grid,
            cv=self.config.cv_folds,
            n_jobs=self.config.n_jobs,
            scoring="neg_mean_squared_error",
            return_train_score=True
        )
        
        search.fit(X, y)
        
        return {
            "best_params": search.best_params_,
            "best_score": float(-search.best_score_),
            "cv_results": {
                "mean_test_score": (-search.cv_results_["mean_test_score"]).tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
                "mean_train_score": (-search.cv_results_["mean_train_score"]).tolist(),
                "std_train_score": search.cv_results_["std_train_score"].tolist(),
                "params": search.cv_results_["params"]
            }
        }
    
    def _cross_validate_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Perform cross-validation."""
        from sklearn.model_selection import cross_validate
        
        scores = cross_validate(
            model,
            X,
            y,
            cv=self.config.cv_folds,
            scoring=[
                "neg_mean_squared_error",
                "r2",
                "neg_mean_absolute_error"
            ],
            return_train_score=True
        )
        
        return scores
    
    def save_results(self):
        """Save tuning results."""
        if not self.config.output_path:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save best parameters
            params_file = output_path / "best_params.json"
            with open(params_file, "w") as f:
                json.dump(self.best_params, f, indent=2)
            
            # Save study statistics
            stats_file = output_path / "study_statistics.json"
            stats = {
                model: {
                    "n_trials": len(study.trials),
                    "best_value": float(study.best_value),
                    "duration": sum(
                        t.duration.total_seconds()
                        for t in study.trials
                        if t.duration
                    )
                }
                for model, study in self.study_results.items()
            }
            with open(stats_file, "w") as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Saved tuning results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def load_results(self):
        """Load saved tuning results."""
        if not self.config.output_path:
            return
        
        try:
            params_file = self.config.output_path / "best_params.json"
            if params_file.exists():
                with open(params_file) as f:
                    self.best_params = json.load(f)
            
            logger.info(f"Loaded tuning results from {params_file}")
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")

def create_hyperparameter_tuner(
    learning: ProfileLearning,
    output_path: Optional[Path] = None
) -> HyperparameterTuner:
    """Create hyperparameter tuner."""
    config = TuningConfig(output_path=output_path)
    return HyperparameterTuner(learning, config)

if __name__ == "__main__":
    # Example usage
    from .profile_learning import create_profile_learning
    from .profile_analysis import create_profile_analyzer
    from .prediction_profiling import create_prediction_profiler
    from .prediction_performance import create_prediction_performance
    from .realtime_prediction import create_realtime_prediction
    from .prediction_controls import create_interactive_controls
    from .prediction_visualization import create_prediction_visualizer
    from .easing_prediction import create_easing_predictor
    from .easing_statistics import create_easing_statistics
    from .easing_metrics import create_easing_metrics
    from .easing_functions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    metrics = create_easing_metrics(easing)
    stats = create_easing_statistics(metrics)
    predictor = create_easing_predictor(stats)
    visualizer = create_prediction_visualizer(predictor)
    controls = create_interactive_controls(visualizer)
    realtime = create_realtime_prediction(controls)
    performance = create_prediction_performance(realtime)
    profiler = create_prediction_profiler(performance)
    analyzer = create_profile_analyzer(profiler)
    learning = create_profile_learning(analyzer)
    tuner = create_hyperparameter_tuner(
        learning,
        output_path=Path("tuning_results")
    )
    
    # Get training data
    data = learning._prepare_training_data("execution_time")
    if data is not None:
        X, y = data
        
        # Tune Random Forest
        rf_results = tuner.tune_random_forest(X, y)
        print("Random Forest tuning:", json.dumps(rf_results, indent=2))
        
        # Tune XGBoost
        xgb_results = tuner.tune_xgboost(X, y)
        print("\nXGBoost tuning:", json.dumps(xgb_results, indent=2))
        
        # Tune Neural Network
        nn_results = tuner.tune_neural_network(X, y)
        print("\nNeural Network tuning:", json.dumps(nn_results, indent=2))
        
        # Tune LSTM
        lstm_results = tuner.tune_lstm(X, y)
        print("\nLSTM tuning:", json.dumps(lstm_results, indent=2))
        
        # Save results
        tuner.save_results()
