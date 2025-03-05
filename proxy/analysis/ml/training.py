"""Training utilities for ML models."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from .models import ThrottlingModel, PerformancePredictor
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

def train_model(
    data: np.ndarray,
    labels: np.ndarray,
    model_params: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[ThrottlingModel, Dict[str, float]]:
    """Train a throttling model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=test_size,
        random_state=random_state
    )
    
    # Initialize and train model
    model = ThrottlingModel(**(model_params or {}))
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    return model, metrics

def validate_model(
    model: ThrottlingModel,
    validation_data: np.ndarray,
    validation_labels: np.ndarray,
    threshold: float = 0.8
) -> Tuple[bool, Dict[str, float]]:
    """Validate model performance."""
    predictions = model.predict(validation_data)
    
    # Calculate validation metrics
    metrics = {
        "mse": mean_squared_error(validation_labels, predictions),
        "r2": r2_score(validation_labels, predictions),
        "accuracy": np.mean(
            np.abs(predictions - validation_labels) < threshold
        )
    }
    
    # Determine if model meets validation criteria
    is_valid = (
        metrics["r2"] > 0.7 and
        metrics["accuracy"] > 0.8
    )
    
    return is_valid, metrics

def update_model(
    model: ThrottlingModel,
    new_data: np.ndarray,
    new_labels: np.ndarray,
    learning_rate: float = 0.1
) -> Tuple[ThrottlingModel, Dict[str, float]]:
    """Update model with new data."""
    # Combine with small sample of previous data if available
    if hasattr(model, "previous_data"):
        data = np.vstack([
            model.previous_data[:100],
            new_data
        ])
        labels = np.concatenate([
            model.previous_labels[:100],
            new_labels
        ])
    else:
        data = new_data
        labels = new_labels
    
    # Update model
    model.fit(data, labels)
    
    # Store sample of data for future updates
    model.previous_data = data[-1000:]  # Keep last 1000 samples
    model.previous_labels = labels[-1000:]
    
    # Evaluate updated model
    predictions = model.predict(data)
    metrics = {
        "mse": mean_squared_error(labels, predictions),
        "r2": r2_score(labels, predictions)
    }
    
    return model, metrics

def cross_validate_model(
    model: ThrottlingModel,
    data: np.ndarray,
    labels: np.ndarray,
    cv_folds: int = 5,
    scoring: List[str] = None
) -> Dict[str, List[float]]:
    """Perform cross-validation."""
    if scoring is None:
        scoring = ['neg_mean_squared_error', 'r2']
    
    # Perform cross-validation
    cv_results = cross_validate(
        model,
        data,
        labels,
        cv=cv_folds,
        scoring=scoring,
        return_train_score=True
    )
    
    return {
        metric: scores.tolist()
        for metric, scores in cv_results.items()
    }

def save_model_artifacts(
    model: ThrottlingModel,
    metrics: Dict[str, float],
    base_path: Path,
    model_name: str
):
    """Save model and associated artifacts."""
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = base_path / f"{model_name}.joblib"
    joblib.dump(model, model_path)
    
    # Save metrics
    metrics_path = base_path / f"{model_name}_metrics.json"
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model.__class__.__name__,
        "parameters": model.__dict__,
        "features": model.features
    }
    metadata_path = base_path / f"{model_name}_metadata.json"
    with metadata_path.open('w') as f:
        json.dump(metadata, f, indent=2)

def load_latest_model(
    base_path: Path,
    model_name: str
) -> Tuple[Optional[ThrottlingModel], Optional[Dict[str, Any]]]:
    """Load the latest model and its metadata."""
    model_path = base_path / f"{model_name}.joblib"
    metadata_path = base_path / f"{model_name}_metadata.json"
    
    if not model_path.exists() or not metadata_path.exists():
        return None, None
    
    try:
        model = joblib.load(model_path)
        with metadata_path.open() as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None
