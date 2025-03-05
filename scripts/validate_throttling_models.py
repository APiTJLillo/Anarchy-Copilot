#!/usr/bin/env python3
"""Validate throttling model quality before deployment."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score
)
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResults:
    """Results of model validation."""
    metric: str
    passed: bool
    r2_score: float
    mse: float
    stability_score: float
    residual_normality: float
    predictive_power: float
    issues: List[str]

class ModelValidator:
    """Validate ML models before deployment."""
    
    def __init__(self, 
                 model_dir: Path,
                 history_file: Path,
                 validation_thresholds: Optional[Dict[str, float]] = None):
        self.model_dir = model_dir
        self.history_file = history_file
        self.thresholds = validation_thresholds or {
            'min_r2': 0.7,
            'max_mse': 0.3,
            'min_stability': 0.8,
            'min_predictive_power': 0.7,
            'residual_normality_alpha': 0.05
        }
        self.results: Dict[str, ValidationResults] = {}

    def validate_all_models(self) -> bool:
        """Validate all available models."""
        try:
            logger.info("Starting model validation...")
            data = pd.read_json(self.history_file)
            
            # Get all model files
            model_files = list(self.model_dir.glob("*_regressor.pkl"))
            all_passed = True
            
            for model_file in model_files:
                metric = model_file.name.replace("_regressor.pkl", "")
                if metric not in data.columns:
                    continue
                
                results = self.validate_model(metric, data)
                self.results[metric] = results
                all_passed &= results.passed
            
            return all_passed
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return False

    def validate_model(self, metric: str, data: pd.DataFrame) -> ValidationResults:
        """Validate a specific model."""
        issues = []
        try:
            # Load model and scaler
            regressor = joblib.load(self.model_dir / f"{metric}_regressor.pkl")
            scaler = joblib.load(self.model_dir / f"{metric}_scaler.pkl")
            
            # Prepare features
            from scripts.predict_throttling_performance import PerformancePredictor
            predictor = PerformancePredictor(self.history_file)
            features = predictor._prepare_features(data)
            
            # Split data for validation
            validation_size = min(len(data) // 3, 100)
            validation_data = data.iloc[-validation_size:]
            validation_features = features.iloc[-validation_size:]
            
            X = scaler.transform(validation_features)
            y_true = validation_data[metric].values
            y_pred = regressor.predict(X)
            
            # Calculate metrics
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            
            if r2 < self.thresholds['min_r2']:
                issues.append(f"R² score ({r2:.3f}) below threshold")
            
            if mse > self.thresholds['max_mse']:
                issues.append(f"MSE ({mse:.3f}) above threshold")
            
            # Check prediction stability
            stability_score = self._check_stability(y_true, y_pred)
            if stability_score < self.thresholds['min_stability']:
                issues.append(f"Stability score ({stability_score:.3f}) below threshold")
            
            # Check residual normality
            residuals = y_true - y_pred
            _, normality_pvalue = stats.normaltest(residuals)
            if normality_pvalue < self.thresholds['residual_normality_alpha']:
                issues.append("Residuals not normally distributed")
            
            # Check predictive power
            predictive_power = self._assess_predictive_power(y_true, y_pred)
            if predictive_power < self.thresholds['min_predictive_power']:
                issues.append(f"Predictive power ({predictive_power:.3f}) below threshold")
            
            return ValidationResults(
                metric=metric,
                passed=len(issues) == 0,
                r2_score=r2,
                mse=mse,
                stability_score=stability_score,
                residual_normality=normality_pvalue,
                predictive_power=predictive_power,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Error validating {metric}: {e}")
            return ValidationResults(
                metric=metric,
                passed=False,
                r2_score=0.0,
                mse=float('inf'),
                stability_score=0.0,
                residual_normality=0.0,
                predictive_power=0.0,
                issues=[str(e)]
            )

    def _check_stability(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Check prediction stability."""
        # Calculate coefficient of variation of prediction error
        errors = np.abs(y_true - y_pred)
        cv = np.std(errors) / np.mean(errors) if np.mean(errors) > 0 else float('inf')
        
        # Transform to [0, 1] scale
        stability = 1 / (1 + cv)
        return stability

    def _assess_predictive_power(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Assess model's predictive power."""
        # Calculate correlation
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Calculate relative error
        rel_error = np.mean(np.abs(y_true - y_pred) / y_true)
        
        # Combine metrics
        predictive_power = (correlation + (1 - rel_error)) / 2
        return predictive_power

    def create_validation_visualization(self) -> go.Figure:
        """Create visualization of validation results."""
        metrics = list(self.results.keys())
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "R² Score",
                "MSE",
                "Stability Score",
                "Residual Normality",
                "Predictive Power",
                "Validation Status"
            )
        )
        
        # Helper function for color
        def get_color(value: float, threshold: float, inverse: bool = False) -> str:
            if inverse:
                return '#e74c3c' if value > threshold else '#2ecc71'
            return '#2ecc71' if value > threshold else '#e74c3c'
        
        # R² scores
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.r2_score for r in self.results.values()],
                marker_color=[
                    get_color(r.r2_score, self.thresholds['min_r2'])
                    for r in self.results.values()
                ],
                name="R² Score"
            ),
            row=1, col=1
        )
        
        # MSE
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.mse for r in self.results.values()],
                marker_color=[
                    get_color(r.mse, self.thresholds['max_mse'], True)
                    for r in self.results.values()
                ],
                name="MSE"
            ),
            row=1, col=2
        )
        
        # Stability
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.stability_score for r in self.results.values()],
                marker_color=[
                    get_color(r.stability_score, self.thresholds['min_stability'])
                    for r in self.results.values()
                ],
                name="Stability"
            ),
            row=2, col=1
        )
        
        # Residual normality
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.residual_normality for r in self.results.values()],
                marker_color=[
                    get_color(r.residual_normality, 
                             self.thresholds['residual_normality_alpha'])
                    for r in self.results.values()
                ],
                name="Normality"
            ),
            row=2, col=2
        )
        
        # Predictive power
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.predictive_power for r in self.results.values()],
                marker_color=[
                    get_color(r.predictive_power, 
                             self.thresholds['min_predictive_power'])
                    for r in self.results.values()
                ],
                name="Predictive Power"
            ),
            row=3, col=1
        )
        
        # Validation status
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[len(r.issues) for r in self.results.values()],
                marker_color=[
                    '#2ecc71' if r.passed else '#e74c3c'
                    for r in self.results.values()
                ],
                name="Issues"
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Model Validation Results"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate validation report."""
        fig = self.create_validation_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Validation Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .validation-container {{
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
                .failed {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                }}
                .passed {{
                    background: #d4edda;
                    border-left: 4px solid #28a745;
                }}
                .threshold {{
                    color: #666;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <h1>Model Validation Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="validation-container">
                <h2>Validation Summary</h2>
                <p>
                    Models Validated: {len(self.results)}<br>
                    Models Passed: {sum(1 for r in self.results.values() if r.passed)}<br>
                    Models Failed: {sum(1 for r in self.results.values() if not r.passed)}
                </p>
                
                <h3>Validation Thresholds</h3>
                <ul>
        """
        
        for name, value in self.thresholds.items():
            html += f"<li>{name}: {value}</li>"
        
        html += """
                </ul>
            </div>
            
            <div class="validation-container">
        """
        
        for metric, result in self.results.items():
            status_class = "passed" if result.passed else "failed"
            
            html += f"""
                <div class="metric-status {status_class}">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>Status: {'✅ Passed' if result.passed else '❌ Failed'}</p>
                    <ul>
                        <li>R² Score: {result.r2_score:.3f}</li>
                        <li>MSE: {result.mse:.3f}</li>
                        <li>Stability Score: {result.stability_score:.3f}</li>
                        <li>Residual Normality p-value: {result.residual_normality:.3f}</li>
                        <li>Predictive Power: {result.predictive_power:.3f}</li>
                    </ul>
            """
            
            if result.issues:
                html += """
                    <p><strong>Issues:</strong></p>
                    <ul>
                """
                for issue in result.issues:
                    html += f"<li>{issue}</li>"
                html += "</ul>"
            
            html += "</div>"
        
        html += f"""
            </div>

            <div class="validation-container">
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
        model_dir = results_dir / "models"
        history_file = results_dir / "performance_history.json"
        
        if not all(p.exists() for p in [results_dir, model_dir, history_file]):
            logger.error("Required files not found")
            return 1
        
        validator = ModelValidator(model_dir, history_file)
        passed = validator.validate_all_models()
        
        output_path = results_dir / "validation_report.html"
        validator.generate_report(output_path)
        
        print(f"\nValidation report generated at: {output_path}")
        return 0 if passed else 1
        
    except Exception as e:
        logger.error(f"Error during validation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
