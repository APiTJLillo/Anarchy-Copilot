#!/usr/bin/env python3
"""ML-based performance prediction for alert throttling system."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from scipy.stats import norm

@dataclass
class PredictionResults:
    """Results of performance predictions."""
    metric: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    anomaly_probability: float
    trend_forecast: List[float]
    feature_importance: Dict[str, float]

class PerformancePredictor:
    """ML-based performance prediction and analysis."""
    
    def __init__(self, history_file: Path, model_dir: Optional[Path] = None):
        self.history_file = history_file
        self.model_dir = model_dir or Path("benchmark_results/models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data = self._load_data()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        
        self.metrics = [
            'throughput', 'memory_usage', 'storage_size',
            'cleanup_time', 'alerts_per_second'
        ]

    def _load_data(self) -> pd.DataFrame:
        """Load and prepare historical data."""
        if not self.history_file.exists():
            return pd.DataFrame()
        
        try:
            data = json.loads(self.history_file.read_text())
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Error loading data: {e}", file=sys.stderr)
            return pd.DataFrame()

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for ML models."""
        features = pd.DataFrame()
        
        # Time-based features
        features['hour'] = df['timestamp'].dt.hour
        features['day_of_week'] = df['timestamp'].dt.dayofweek
        features['day_of_month'] = df['timestamp'].dt.day
        
        # Rolling statistics
        for metric in self.metrics:
            if metric in df.columns:
                features[f'{metric}_rolling_mean'] = df[metric].rolling(3).mean()
                features[f'{metric}_rolling_std'] = df[metric].rolling(3).std()
                features[f'{metric}_diff'] = df[metric].diff()
        
        # Fill missing values
        features = features.fillna(method='bfill')
        return features

    def train_models(self) -> None:
        """Train prediction models for each metric."""
        if self.data.empty:
            print("No data available for training")
            return
        
        features = self._prepare_features(self.data)
        
        for metric in self.metrics:
            if metric not in self.data.columns:
                continue
            
            # Prepare data
            X = features.values
            y = self.data[metric].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            # Regression model
            regressor = RandomForestRegressor(n_estimators=100, random_state=42)
            regressor.fit(X_train_scaled, y_train)
            
            # Anomaly detection model
            anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            anomaly_detector.fit(X_train_scaled)
            
            # Save models and scaler
            self.models[f"{metric}_regressor"] = regressor
            self.models[f"{metric}_anomaly"] = anomaly_detector
            self.scalers[metric] = scaler
            
            # Save to disk
            joblib.dump(regressor, self.model_dir / f"{metric}_regressor.pkl")
            joblib.dump(anomaly_detector, self.model_dir / f"{metric}_anomaly.pkl")
            joblib.dump(scaler, self.model_dir / f"{metric}_scaler.pkl")
            
            # Evaluate
            y_pred = regressor.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"\nModel performance for {metric}:")
            print(f"MSE: {mse:.4f}")
            print(f"R2 Score: {r2:.4f}")

    def predict_metric(self, metric: str, current_features: Optional[Dict[str, float]] = None) -> PredictionResults:
        """Make predictions for a metric."""
        if metric not in self.metrics:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Load models if not in memory
        if f"{metric}_regressor" not in self.models:
            try:
                self.models[f"{metric}_regressor"] = joblib.load(
                    self.model_dir / f"{metric}_regressor.pkl"
                )
                self.models[f"{metric}_anomaly"] = joblib.load(
                    self.model_dir / f"{metric}_anomaly.pkl"
                )
                self.scalers[metric] = joblib.load(
                    self.model_dir / f"{metric}_scaler.pkl"
                )
            except Exception as e:
                raise RuntimeError(f"Error loading models for {metric}: {e}")
        
        # Prepare features
        if current_features is None:
            features = self._prepare_features(self.data).iloc[-1:]
        else:
            features = pd.DataFrame([current_features])
        
        # Scale features
        X = self.scalers[metric].transform(features)
        
        # Make predictions
        regressor = self.models[f"{metric}_regressor"]
        anomaly_detector = self.models[f"{metric}_anomaly"]
        
        # Point prediction
        prediction = regressor.predict(X)[0]
        
        # Confidence interval using bootstrapping
        predictions = []
        n_estimators = len(regressor.estimators_)
        for estimator in regressor.estimators_:
            predictions.append(estimator.predict(X)[0])
        
        ci_lower = np.percentile(predictions, 2.5)
        ci_upper = np.percentile(predictions, 97.5)
        
        # Anomaly score
        anomaly_score = anomaly_detector.score_samples(X)[0]
        anomaly_prob = 1 - norm.cdf(anomaly_score)
        
        # Feature importance
        importance = dict(zip(
            features.columns,
            regressor.feature_importances_
        ))
        
        # Trend forecast
        forecast_horizon = 5
        trend = []
        current = features.copy()
        for _ in range(forecast_horizon):
            pred = regressor.predict(self.scalers[metric].transform(current))[0]
            trend.append(pred)
            # Update features for next prediction
            current[f'{metric}_rolling_mean'] = pred
            current[f'{metric}_diff'] = pred - current[f'{metric}_rolling_mean'].iloc[0]
        
        return PredictionResults(
            metric=metric,
            predicted_value=prediction,
            confidence_interval=(ci_lower, ci_upper),
            anomaly_probability=anomaly_prob,
            trend_forecast=trend,
            feature_importance=importance
        )

    def create_prediction_visualization(self, predictions: Dict[str, PredictionResults]) -> go.Figure:
        """Create visualization of predictions."""
        fig = make_subplots(
            rows=len(predictions), cols=2,
            subplot_titles=[
                f"{m.replace('_', ' ').title()} Forecast" for m in predictions.keys()
            ] * 2,
            specs=[[{"type": "scatter"}, {"type": "bar"}]] * len(predictions),
            vertical_spacing=0.1
        )
        
        for i, (metric, pred) in enumerate(predictions.items(), 1):
            # Historical values
            historical = self.data[metric].values[-10:]
            historical_dates = self.data['timestamp'].values[-10:]
            future_dates = [
                historical_dates[-1] + timedelta(days=i+1)
                for i in range(len(pred.trend_forecast))
            ]
            
            # Forecast plot
            fig.add_trace(
                go.Scatter(
                    x=historical_dates,
                    y=historical,
                    name=f"{metric} historical",
                    line=dict(color='#3498db')
                ),
                row=i, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates,
                    y=pred.trend_forecast,
                    name=f"{metric} forecast",
                    line=dict(
                        color='#e74c3c',
                        dash='dash'
                    )
                ),
                row=i, col=1
            )
            
            # Confidence interval
            fig.add_trace(
                go.Scatter(
                    x=[future_dates[0]],
                    y=[pred.predicted_value],
                    error_y=dict(
                        type='data',
                        symmetric=False,
                        array=[pred.confidence_interval[1] - pred.predicted_value],
                        arrayminus=[pred.predicted_value - pred.confidence_interval[0]]
                    ),
                    name=f"{metric} confidence",
                    mode='markers'
                ),
                row=i, col=1
            )
            
            # Feature importance
            importance_items = sorted(
                pred.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            fig.add_trace(
                go.Bar(
                    x=[item[0] for item in importance_items],
                    y=[item[1] for item in importance_items],
                    name=f"{metric} importance"
                ),
                row=i, col=2
            )
        
        fig.update_layout(
            height=400 * len(predictions),
            showlegend=True,
            title_text="Performance Predictions and Feature Importance"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate prediction report."""
        predictions = {
            metric: self.predict_metric(metric)
            for metric in self.metrics
            if metric in self.data.columns
        }
        
        fig = self.create_prediction_visualization(predictions)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Predictions Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .prediction-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-prediction {{
                    margin: 20px 0;
                    padding: 10px;
                    border-left: 4px solid #3498db;
                }}
                .warning {{
                    background: #fff3cd;
                    border-color: #ffeeba;
                }}
                .prediction {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .anomaly-score {{
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-weight: bold;
                }}
                .high-risk {{ color: #e74c3c; }}
                .medium-risk {{ color: #f39c12; }}
                .low-risk {{ color: #27ae60; }}
            </style>
        </head>
        <body>
            <h1>Performance Predictions Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="prediction-container">
        """
        
        for metric, pred in predictions.items():
            risk_class = (
                "high-risk" if pred.anomaly_probability > 0.7
                else "medium-risk" if pred.anomaly_probability > 0.3
                else "low-risk"
            )
            
            html += f"""
                <div class="metric-prediction{'warning' if pred.anomaly_probability > 0.7 else ''}">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>
                        Predicted Value: 
                        <span class="prediction">{pred.predicted_value:.2f}</span>
                        (95% CI: [{pred.confidence_interval[0]:.2f}, {pred.confidence_interval[1]:.2f}])
                    </p>
                    <p>
                        Anomaly Probability: 
                        <span class="anomaly-score {risk_class}">
                            {pred.anomaly_probability:.1%}
                        </span>
                    </p>
                    <p>Top Features:</p>
                    <ul>
            """
            
            for feature, importance in sorted(
                pred.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]:
                html += f"""
                    <li>{feature}: {importance:.3f}</li>
                """
            
            html += """
                    </ul>
                </div>
            """
        
        html += f"""
            </div>

            <div class="prediction-container">
                {fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html)

def main() -> int:
    """Main entry point."""
    try:
        results_dir = Path("benchmark_results")
        if not results_dir.exists():
            print("No benchmark results directory found.")
            return 1
        
        history_file = results_dir / "performance_history.json"
        if not history_file.exists():
            print("No performance history found.")
            return 1
        
        predictor = PerformancePredictor(history_file)
        predictor.train_models()
        
        output_path = results_dir / "performance_predictions.html"
        predictor.generate_report(output_path)
        
        print(f"\nPrediction report generated at: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error generating predictions: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
