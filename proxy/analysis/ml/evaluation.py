"""Evaluation utilities for ML models."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_recall_curve,
    roc_curve,
    auc
)
from .models import ThrottlingModel, PerformancePredictor

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate model performance and generate reports."""
    
    def __init__(
        self,
        output_dir: Path,
        metrics: List[str] = None
    ):
        self.output_dir = output_dir
        self.metrics = metrics or [
            "mse",
            "rmse",
            "r2",
            "mae",
            "mape"
        ]
        self.history: Dict[str, List[Dict[str, float]]] = {}

    def evaluate_performance(
        self,
        model: ThrottlingModel,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        model_name: str = "default"
    ) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        predictions = model.predict(test_data)
        
        metrics = {
            "mse": mean_squared_error(test_labels, predictions),
            "rmse": np.sqrt(mean_squared_error(test_labels, predictions)),
            "r2": r2_score(test_labels, predictions),
            "mae": np.mean(np.abs(test_labels - predictions)),
            "mape": np.mean(np.abs((test_labels - predictions) / test_labels)) * 100
        }
        
        if model_name not in self.history:
            self.history[model_name] = []
        self.history[model_name].append({
            "timestamp": datetime.now().timestamp(),
            **metrics
        })
        
        return metrics

    def analyze_errors(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze prediction errors."""
        errors = predictions - true_values
        
        analysis = {
            "error_mean": float(np.mean(errors)),
            "error_std": float(np.std(errors)),
            "error_median": float(np.median(errors)),
            "error_quartiles": [
                float(np.percentile(errors, 25)),
                float(np.percentile(errors, 75))
            ],
            "outliers": float(np.sum(np.abs(errors) > 2 * np.std(errors)))
        }
        
        return analysis

    def calculate_metrics(
        self,
        model: ThrottlingModel,
        data: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.8
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        predictions = model.predict(data)
        
        # Basic metrics
        metrics = {
            "mse": mean_squared_error(labels, predictions),
            "rmse": np.sqrt(mean_squared_error(labels, predictions)),
            "r2": r2_score(labels, predictions),
            "mae": np.mean(np.abs(labels - predictions)),
            "mape": np.mean(np.abs((labels - predictions) / labels)) * 100,
            "accuracy": np.mean(np.abs(predictions - labels) < threshold)
        }
        
        # Classification metrics if using threshold
        binary_predictions = predictions >= threshold
        binary_labels = labels >= threshold
        
        # Precision-Recall
        precision, recall, _ = precision_recall_curve(binary_labels, predictions)
        metrics["pr_auc"] = auc(recall, precision)
        
        # ROC
        fpr, tpr, _ = roc_curve(binary_labels, predictions)
        metrics["roc_auc"] = auc(fpr, tpr)
        
        return metrics

    def compare_models(
        self,
        models: Dict[str, ThrottlingModel],
        test_data: np.ndarray,
        test_labels: np.ndarray
    ) -> pd.DataFrame:
        """Compare multiple models."""
        results = []
        
        for name, model in models.items():
            metrics = self.calculate_metrics(model, test_data, test_labels)
            metrics["model"] = name
            results.append(metrics)
        
        return pd.DataFrame(results)

    def plot_performance_history(
        self,
        model_name: str = "default",
        metrics: List[str] = None
    ) -> None:
        """Plot model performance history."""
        if model_name not in self.history:
            logger.warning(f"No history for model {model_name}")
            return
        
        metrics = metrics or self.metrics
        history = pd.DataFrame(self.history[model_name])
        
        plt.figure(figsize=(12, 6))
        for metric in metrics:
            if metric in history:
                plt.plot(
                    history["timestamp"],
                    history[metric],
                    label=metric
                )
        
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title(f"Model Performance History: {model_name}")
        plt.legend()
        
        output_path = self.output_dir / f"{model_name}_history.png"
        plt.savefig(output_path)
        plt.close()

    def generate_report(
        self,
        model: ThrottlingModel,
        test_data: np.ndarray,
        test_labels: np.ndarray,
        model_name: str = "default"
    ) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        predictions = model.predict(test_data)
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "metrics": self.calculate_metrics(model, test_data, test_labels),
            "error_analysis": self.analyze_errors(predictions, test_labels),
            "data_stats": {
                "n_samples": len(test_data),
                "n_features": test_data.shape[1],
                "value_range": {
                    "min": float(test_labels.min()),
                    "max": float(test_labels.max()),
                    "mean": float(test_labels.mean()),
                    "std": float(test_labels.std())
                }
            }
        }
        
        # Save report
        report_path = self.output_dir / f"{model_name}_report.json"
        with report_path.open('w') as f:
            json.dump(report, f, indent=2)
        
        return report

    def plot_error_distribution(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        model_name: str = "default"
    ) -> None:
        """Plot error distribution."""
        errors = predictions - true_values
        
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True)
        plt.title(f"Error Distribution: {model_name}")
        plt.xlabel("Error")
        plt.ylabel("Count")
        
        output_path = self.output_dir / f"{model_name}_error_dist.png"
        plt.savefig(output_path)
        plt.close()

    def plot_prediction_scatter(
        self,
        predictions: np.ndarray,
        true_values: np.ndarray,
        model_name: str = "default"
    ) -> None:
        """Plot predicted vs actual values."""
        plt.figure(figsize=(10, 10))
        plt.scatter(true_values, predictions, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(predictions.min(), true_values.min())
        max_val = max(predictions.max(), true_values.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title(f"Predicted vs Actual: {model_name}")
        plt.xlabel("Actual Values")
        plt.ylabel("Predictions")
        
        output_path = self.output_dir / f"{model_name}_predictions.png"
        plt.savefig(output_path)
        plt.close()

def analyze_results(
    evaluator: ModelEvaluator,
    model: ThrottlingModel,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    model_name: str = "default"
) -> Dict[str, Any]:
    """Perform comprehensive results analysis."""
    # Generate predictions
    predictions = model.predict(test_data)
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(model, test_data, test_labels)
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(predictions, test_labels)
    
    # Generate plots
    evaluator.plot_error_distribution(predictions, test_labels, model_name)
    evaluator.plot_prediction_scatter(predictions, test_labels, model_name)
    evaluator.plot_performance_history(model_name)
    
    # Generate full report
    report = evaluator.generate_report(model, test_data, test_labels, model_name)
    
    return {
        "metrics": metrics,
        "error_analysis": error_analysis,
        "report": report
    }
