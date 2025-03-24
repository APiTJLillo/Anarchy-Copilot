"""Predictive analytics for performance metrics."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path
import json
from prophet import Prophet
import tensorflow as tf

from .performance_metrics import PerformanceMetric, MetricsConfig

logger = logging.getLogger(__name__)

@dataclass
class PredictionConfig:
    """Configuration for performance prediction."""
    forecast_horizon: int = 100
    training_window: timedelta = timedelta(hours=24)
    update_interval: timedelta = timedelta(minutes=5)
    min_samples: int = 100
    confidence_level: float = 0.95
    anomaly_threshold: float = 0.1
    feature_history: int = 10
    model_path: Optional[Path] = None
    save_predictions: bool = True
    prediction_file: Optional[Path] = None
    use_gpu: bool = True

class PerformancePredictor:
    """Predict performance trends and issues."""
    
    def __init__(
        self,
        config: PredictionConfig,
        metrics_config: MetricsConfig
    ):
        self.config = config
        self.metrics_config = metrics_config
        
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self.last_update: Optional[datetime] = None
        self.predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Initialize TensorFlow if available
        if config.use_gpu and tf.test.is_built_with_cuda():
            try:
                physical_devices = tf.config.list_physical_devices("GPU")
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except Exception as e:
                logger.warning(f"Failed to configure GPU: {e}")
    
    def update_models(
        self,
        metrics: List[PerformanceMetric]
    ):
        """Update prediction models with new data."""
        if not metrics:
            return
        
        current_time = datetime.now()
        if (
            self.last_update and 
            current_time - self.last_update < self.config.update_interval
        ):
            return
        
        try:
            # Prepare training data
            df = self._prepare_training_data(metrics)
            
            if len(df) < self.config.min_samples:
                return
            
            # Train models for each metric
            for metric in ["fps", "memory_mb", "cpu_percent"]:
                self._train_model(df, metric)
            
            self.last_update = current_time
            
            # Save models if configured
            if self.config.model_path:
                self._save_models()
                
        except Exception as e:
            logger.error(f"Failed to update models: {e}")
    
    def predict_performance(
        self,
        metrics: List[PerformanceMetric]
    ) -> Dict[str, Any]:
        """Predict future performance metrics."""
        if not metrics or not self.models:
            return {}
        
        try:
            # Prepare recent data
            df = self._prepare_training_data(metrics[-self.config.feature_history:])
            
            predictions = {}
            anomalies = {}
            
            # Make predictions for each metric
            for metric in ["fps", "memory_mb", "cpu_percent"]:
                if metric in self.models:
                    metric_predictions = self._predict_metric(df, metric)
                    predictions[metric] = metric_predictions
                    
                    # Detect anomalies
                    anomalies[metric] = self._detect_anomalies(
                        df[metric].values,
                        metric_predictions["forecast"]
                    )
            
            result = {
                "predictions": predictions,
                "anomalies": anomalies,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store predictions
            self._store_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}
    
    def _prepare_training_data(
        self,
        metrics: List[PerformanceMetric]
    ) -> pd.DataFrame:
        """Prepare metrics data for training."""
        data = []
        for metric in metrics:
            data.append({
                "timestamp": metric.timestamp,
                "fps": metric.fps,
                "frame_time": metric.frame_time,
                "memory_mb": metric.memory_mb,
                "cpu_percent": metric.cpu_percent,
                "io_read_mb": metric.io_read_mb,
                "io_write_mb": metric.io_write_mb,
                "queue_size": metric.queue_size
            })
        
        df = pd.DataFrame(data)
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        
        # Add lag features
        for col in ["fps", "memory_mb", "cpu_percent"]:
            for lag in range(1, 6):
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)
        
        # Add rolling statistics
        windows = [5, 10, 20]
        for col in ["fps", "memory_mb", "cpu_percent"]:
            for window in windows:
                df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window).mean()
                df[f"{col}_rolling_std_{window}"] = df[col].rolling(window).std()
        
        return df.dropna()
    
    def _train_model(
        self,
        df: pd.DataFrame,
        target: str
    ):
        """Train prediction model for metric."""
        feature_columns = [
            c for c in df.columns
            if c not in ["timestamp", target] and not pd.api.types.is_datetime64_any_dtype(df[c])
        ]
        
        X = df[feature_columns]
        y = df[target]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, shuffle=False
        )
        
        # Train random forest model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Train Prophet model for time series
        prophet_df = pd.DataFrame({
            "ds": df["timestamp"],
            "y": df[target]
        })
        prophet_model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_mode="multiplicative"
        )
        prophet_model.fit(prophet_df)
        
        # Store models
        self.models[target] = {
            "rf": rf_model,
            "prophet": prophet_model,
            "features": feature_columns
        }
    
    def _predict_metric(
        self,
        df: pd.DataFrame,
        metric: str
    ) -> Dict[str, Any]:
        """Make predictions for metric."""
        model = self.models[metric]
        
        # Make RF prediction
        X = df[model["features"]]
        X_scaled = self.scaler.transform(X)
        rf_pred = model["rf"].predict(X_scaled)
        
        # Make Prophet prediction
        future = model["prophet"].make_future_dataframe(
            periods=self.config.forecast_horizon,
            freq="T"
        )
        prophet_pred = model["prophet"].predict(future)
        
        # Combine predictions
        combined_pred = (rf_pred + prophet_pred["yhat"].values[-len(rf_pred):]) / 2
        
        return {
            "forecast": combined_pred.tolist(),
            "upper": prophet_pred["yhat_upper"].values[-len(rf_pred):].tolist(),
            "lower": prophet_pred["yhat_lower"].values[-len(rf_pred):].tolist(),
            "components": {
                "trend": prophet_pred["trend"].values[-len(rf_pred):].tolist(),
                "weekly": prophet_pred["weekly"].values[-len(rf_pred):].tolist()
            }
        }
    
    def _detect_anomalies(
        self,
        actual: np.ndarray,
        predicted: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in predictions."""
        anomalies = []
        
        # Calculate residuals
        residuals = actual - np.array(predicted)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        
        # Find anomalies using MAD
        threshold = self.config.anomaly_threshold * mad
        anomaly_indices = np.where(np.abs(residuals) > threshold)[0]
        
        for idx in anomaly_indices:
            anomalies.append({
                "index": int(idx),
                "actual": float(actual[idx]),
                "predicted": predicted[idx],
                "deviation": float(residuals[idx]),
                "severity": float(abs(residuals[idx]) / mad)
            })
        
        return anomalies
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """Store prediction results."""
        if self.config.save_predictions:
            for metric, pred in prediction["predictions"].items():
                self.predictions[metric].append({
                    "timestamp": prediction["timestamp"],
                    "forecast": pred["forecast"][0],
                    "upper": pred["upper"][0],
                    "lower": pred["lower"][0]
                })
            
            # Trim history
            for metric in self.predictions:
                if len(self.predictions[metric]) > self.config.min_samples:
                    self.predictions[metric] = \
                        self.predictions[metric][-self.config.min_samples:]
            
            # Save to file if configured
            if self.config.prediction_file:
                self._save_predictions()
    
    def _save_models(self):
        """Save trained models."""
        try:
            Path(self.config.model_path).mkdir(parents=True, exist_ok=True)
            
            for metric, model in self.models.items():
                # Save Random Forest model
                rf_path = Path(self.config.model_path) / f"{metric}_rf.joblib"
                joblib.dump(model["rf"], rf_path)
                
                # Save Prophet model
                prophet_path = Path(self.config.model_path) / f"{metric}_prophet.json"
                with open(prophet_path, "w") as f:
                    json.dump(
                        model["prophet"].to_json(),
                        f,
                        indent=2
                    )
                
            logger.info(f"Saved models to {self.config.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    def _save_predictions(self):
        """Save prediction history."""
        try:
            with open(self.config.prediction_file, "w") as f:
                json.dump(
                    {
                        "predictions": dict(self.predictions),
                        "last_update": self.last_update.isoformat()
                    },
                    f,
                    indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    
    def get_prediction_accuracy(
        self,
        metric: str,
        window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Calculate prediction accuracy metrics."""
        if metric not in self.predictions:
            return {}
        
        predictions = self.predictions[metric]
        if not predictions:
            return {}
        
        if window:
            cutoff = datetime.now() - window
            predictions = [
                p for p in predictions
                if datetime.fromisoformat(p["timestamp"]) > cutoff
            ]
        
        actuals = [p.get("actual", 0) for p in predictions]
        forecasts = [p["forecast"] for p in predictions]
        
        if not actuals or not forecasts:
            return {}
        
        mse = np.mean((np.array(actuals) - np.array(forecasts)) ** 2)
        mae = np.mean(np.abs(np.array(actuals) - np.array(forecasts)))
        mape = np.mean(
            np.abs((np.array(actuals) - np.array(forecasts)) / np.array(actuals))
        ) * 100
        
        return {
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "mae": float(mae),
            "mape": float(mape)
        }

def create_predictor(
    metrics_config: MetricsConfig,
    model_path: Optional[Path] = None,
    prediction_file: Optional[Path] = None
) -> PerformancePredictor:
    """Create performance predictor."""
    config = PredictionConfig(
        model_path=model_path,
        prediction_file=prediction_file
    )
    return PerformancePredictor(config, metrics_config)

if __name__ == "__main__":
    # Example usage
    from .performance_metrics import monitor_performance
    
    # Create monitor and predictor
    monitor = monitor_performance()
    predictor = create_predictor(
        monitor.config,
        model_path=Path("models"),
        prediction_file=Path("predictions.json")
    )
    
    # Simulate metrics collection
    for _ in range(1000):
        time.sleep(0.1)
        monitor.record_frame(time.perf_counter())
    
    # Update models and make predictions
    predictor.update_models(monitor.metrics)
    predictions = predictor.predict_performance(monitor.metrics)
    
    print(json.dumps(predictions, indent=2))
