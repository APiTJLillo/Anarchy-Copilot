#!/usr/bin/env python3
"""Automated retraining system for throttling performance models."""

import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import logging
from dataclasses import dataclass
import threading
import schedule
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Metrics for model performance."""
    mse: float
    r2: float
    prediction_error: float
    drift_score: float
    last_retrain: datetime
    samples_since_retrain: int

class ModelMonitor:
    """Monitor model performance and trigger retraining."""
    
    def __init__(self, 
                 model_dir: Path,
                 history_file: Path,
                 metrics_file: Optional[Path] = None):
        self.model_dir = model_dir
        self.history_file = history_file
        self.metrics_file = metrics_file or (model_dir / "model_metrics.json")
        self.metrics: Dict[str, ModelMetrics] = {}
        self.retrain_thresholds = {
            'min_r2': 0.7,
            'max_mse': 0.3,
            'max_drift': 0.2,
            'min_samples': 100,
            'max_age_days': 7
        }
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load saved model metrics."""
        if self.metrics_file.exists():
            try:
                data = json.loads(self.metrics_file.read_text())
                self.metrics = {
                    metric: ModelMetrics(
                        mse=m['mse'],
                        r2=m['r2'],
                        prediction_error=m['prediction_error'],
                        drift_score=m['drift_score'],
                        last_retrain=datetime.fromisoformat(m['last_retrain']),
                        samples_since_retrain=m['samples_since_retrain']
                    )
                    for metric, m in data.items()
                }
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")

    def _save_metrics(self) -> None:
        """Save current model metrics."""
        try:
            data = {
                metric: {
                    'mse': m.mse,
                    'r2': m.r2,
                    'prediction_error': m.prediction_error,
                    'drift_score': m.drift_score,
                    'last_retrain': m.last_retrain.isoformat(),
                    'samples_since_retrain': m.samples_since_retrain
                }
                for metric, m in self.metrics.items()
            }
            self.metrics_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def calculate_drift(self, metric: str, recent_data: pd.DataFrame) -> float:
        """Calculate distribution drift for a metric."""
        try:
            # Load historical predictions
            predictions_file = self.model_dir / f"{metric}_predictions.npy"
            if not predictions_file.exists():
                return 0.0
            
            historical_preds = np.load(predictions_file)
            recent_values = recent_data[metric].values
            
            # Use KS test to measure distribution shift
            from scipy import stats
            ks_statistic, _ = stats.ks_2samp(historical_preds, recent_values)
            return ks_statistic
            
        except Exception as e:
            logger.error(f"Error calculating drift for {metric}: {e}")
            return 0.0

    def evaluate_model(self, metric: str, recent_data: pd.DataFrame) -> ModelMetrics:
        """Evaluate model performance on recent data."""
        try:
            # Load model
            regressor = joblib.load(self.model_dir / f"{metric}_regressor.pkl")
            scaler = joblib.load(self.model_dir / f"{metric}_scaler.pkl")
            
            # Prepare features
            from scripts.predict_throttling_performance import PerformancePredictor
            predictor = PerformancePredictor(self.history_file)
            features = predictor._prepare_features(recent_data)
            
            # Make predictions
            X = scaler.transform(features)
            y_true = recent_data[metric].values
            y_pred = regressor.predict(X)
            
            # Calculate metrics
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            prediction_error = np.mean(np.abs(y_true - y_pred) / y_true)
            drift_score = self.calculate_drift(metric, recent_data)
            
            # Get or create metrics record
            if metric in self.metrics:
                metrics = self.metrics[metric]
                metrics.samples_since_retrain += len(recent_data)
            else:
                metrics = ModelMetrics(
                    mse=mse,
                    r2=r2,
                    prediction_error=prediction_error,
                    drift_score=drift_score,
                    last_retrain=datetime.now(),
                    samples_since_retrain=len(recent_data)
                )
            
            # Update metrics
            metrics.mse = mse
            metrics.r2 = r2
            metrics.prediction_error = prediction_error
            metrics.drift_score = drift_score
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model for {metric}: {e}")
            return None

    def needs_retraining(self, metric: str, metrics: ModelMetrics) -> Tuple[bool, str]:
        """Determine if model needs retraining."""
        reasons = []
        
        # Check model age
        age_days = (datetime.now() - metrics.last_retrain).days
        if age_days > self.retrain_thresholds['max_age_days']:
            reasons.append(f"Model age ({age_days} days) exceeds threshold")
        
        # Check performance metrics
        if metrics.r2 < self.retrain_thresholds['min_r2']:
            reasons.append(f"R² score ({metrics.r2:.3f}) below threshold")
        
        if metrics.mse > self.retrain_thresholds['max_mse']:
            reasons.append(f"MSE ({metrics.mse:.3f}) above threshold")
        
        if metrics.drift_score > self.retrain_thresholds['max_drift']:
            reasons.append(f"Drift score ({metrics.drift_score:.3f}) above threshold")
        
        # Check sample count
        if metrics.samples_since_retrain > self.retrain_thresholds['min_samples']:
            reasons.append(f"Sufficient new samples ({metrics.samples_since_retrain})")
        
        return bool(reasons), "; ".join(reasons)

    def retrain_model(self, metric: str) -> bool:
        """Retrain model for specified metric."""
        try:
            logger.info(f"Retraining model for {metric}...")
            
            # Load data and train model
            from scripts.predict_throttling_performance import PerformancePredictor
            predictor = PerformancePredictor(self.history_file)
            predictor.train_models()
            
            # Update metrics
            self.metrics[metric].last_retrain = datetime.now()
            self.metrics[metric].samples_since_retrain = 0
            self._save_metrics()
            
            logger.info(f"Successfully retrained model for {metric}")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model for {metric}: {e}")
            return False

    def create_monitoring_visualization(self) -> go.Figure:
        """Create visualization of model monitoring metrics."""
        metrics = list(self.metrics.keys())
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Model Performance (R²)",
                "Prediction Error",
                "Distribution Drift",
                "Samples Since Retraining"
            )
        )
        
        # R² scores
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[m.r2 for m in self.metrics.values()],
                name="R² Score"
            ),
            row=1, col=1
        )
        
        # Prediction error
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[m.prediction_error for m in self.metrics.values()],
                name="Prediction Error"
            ),
            row=1, col=2
        )
        
        # Drift scores
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[m.drift_score for m in self.metrics.values()],
                name="Drift Score"
            ),
            row=2, col=1
        )
        
        # Sample counts
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[m.samples_since_retrain for m in self.metrics.values()],
                name="Samples"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Model Monitoring Metrics",
            showlegend=True
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate model monitoring report."""
        fig = self.create_monitoring_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .monitoring-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-status {{
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .needs-retraining {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                }}
                .good {{
                    background: #d4edda;
                    border-left: 4px solid #28a745;
                }}
            </style>
        </head>
        <body>
            <h1>Model Monitoring Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="monitoring-container">
        """
        
        for metric, metrics in self.metrics.items():
            needs_retrain, reasons = self.needs_retraining(metric, metrics)
            status_class = "needs-retraining" if needs_retrain else "good"
            
            html += f"""
                <div class="metric-status {status_class}">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>Last Retrained: {metrics.last_retrain.strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <ul>
                        <li>R² Score: {metrics.r2:.3f}</li>
                        <li>MSE: {metrics.mse:.3f}</li>
                        <li>Prediction Error: {metrics.prediction_error:.1%}</li>
                        <li>Drift Score: {metrics.drift_score:.3f}</li>
                        <li>Samples Since Retrain: {metrics.samples_since_retrain}</li>
                    </ul>
                    {f'<p><strong>Needs Retraining:</strong> {reasons}</p>' if needs_retrain else ''}
                </div>
            """
        
        html += f"""
            </div>

            <div class="monitoring-container">
                {fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html)

def monitor_and_retrain() -> None:
    """Main monitoring function."""
    try:
        results_dir = Path("benchmark_results")
        history_file = results_dir / "performance_history.json"
        model_dir = results_dir / "models"
        
        if not all(p.exists() for p in [results_dir, history_file, model_dir]):
            logger.error("Required files not found")
            return
        
        # Load recent data
        data = pd.read_json(history_file)
        recent_data = data[data['timestamp'] >= datetime.now() - timedelta(days=1)]
        
        if recent_data.empty:
            logger.info("No recent data available")
            return
        
        # Initialize monitor
        monitor = ModelMonitor(model_dir, history_file)
        
        # Check each metric
        for metric in monitor.metrics:
            # Evaluate current performance
            metrics = monitor.evaluate_model(metric, recent_data)
            if not metrics:
                continue
            
            # Check if retraining needed
            needs_retrain, reasons = monitor.needs_retraining(metric, metrics)
            
            if needs_retrain:
                logger.info(f"Model for {metric} needs retraining: {reasons}")
                if monitor.retrain_model(metric):
                    logger.info(f"Successfully retrained model for {metric}")
                else:
                    logger.error(f"Failed to retrain model for {metric}")
        
        # Generate report
        monitor.generate_report(results_dir / "model_monitoring.html")
        
    except Exception as e:
        logger.error(f"Error in monitoring cycle: {e}")

def run_monitoring_schedule() -> None:
    """Run monitoring on a schedule."""
    schedule.every(6).hours.do(monitor_and_retrain)
    
    while True:
        schedule.run_pending()
        time.sleep(60)

def main() -> int:
    """Main entry point."""
    try:
        # Run initial monitoring
        monitor_and_retrain()
        
        # Start scheduled monitoring in background
        if os.environ.get("ENABLE_SCHEDULED_MONITORING"):
            thread = threading.Thread(target=run_monitoring_schedule, daemon=True)
            thread.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
