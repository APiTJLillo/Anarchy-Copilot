#!/usr/bin/env python3
"""A/B testing for throttling model improvements."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import logging
from scipy import stats
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ABTestResults:
    """Results of A/B test comparison."""
    metric: str
    variant_a: str
    variant_b: str
    sample_size: int
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    significant: bool
    improvement: float
    metrics_comparison: Dict[str, Tuple[float, float]]

class ModelABTester:
    """Compare model variants using A/B testing."""
    
    def __init__(self, 
                 model_dir: Path,
                 history_file: Path,
                 test_config: Optional[Dict[str, Any]] = None):
        self.model_dir = model_dir
        self.history_file = history_file
        self.config = test_config or {
            'significance_level': 0.05,
            'min_effect_size': 0.1,
            'sample_size': 1000,
            'crossover_ratio': 0.2
        }
        self.results: Dict[str, ABTestResults] = {}

    def prepare_test_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for A/B testing using crossover sampling."""
        # Split data into time periods
        midpoint = len(data) // 2
        period_a = data.iloc[:midpoint]
        period_b = data.iloc[midpoint:]
        
        # Sample with crossover
        crossover_size = int(len(data) * self.config['crossover_ratio'])
        
        a_indices = np.random.choice(
            period_a.index, 
            size=min(self.config['sample_size'] - crossover_size, len(period_a)),
            replace=False
        )
        b_indices = np.random.choice(
            period_b.index,
            size=min(crossover_size, len(period_b)),
            replace=False
        )
        sample_a = pd.concat([data.loc[a_indices], data.loc[b_indices]])
        
        b_indices = np.random.choice(
            period_b.index,
            size=min(self.config['sample_size'] - crossover_size, len(period_b)),
            replace=False
        )
        a_indices = np.random.choice(
            period_a.index,
            size=min(crossover_size, len(period_a)),
            replace=False
        )
        sample_b = pd.concat([data.loc[b_indices], data.loc[a_indices]])
        
        return sample_a, sample_b

    def compare_models(self, metric: str, variant_a: str, variant_b: str) -> ABTestResults:
        """Compare two model variants."""
        try:
            # Load data and models
            data = pd.read_json(self.history_file)
            
            # Load models
            model_a = joblib.load(self.model_dir / f"{metric}_{variant_a}_regressor.pkl")
            model_b = joblib.load(self.model_dir / f"{metric}_{variant_b}_regressor.pkl")
            scaler = joblib.load(self.model_dir / f"{metric}_scaler.pkl")
            
            # Prepare features
            from scripts.predict_throttling_performance import PerformancePredictor
            predictor = PerformancePredictor(self.history_file)
            features = predictor._prepare_features(data)
            
            # Prepare test data
            sample_a, sample_b = self.prepare_test_data(data)
            X_a = scaler.transform(predictor._prepare_features(sample_a))
            X_b = scaler.transform(predictor._prepare_features(sample_b))
            y_a = sample_a[metric].values
            y_b = sample_b[metric].values
            
            # Make predictions
            pred_a = model_a.predict(X_a)
            pred_b = model_b.predict(X_b)
            
            # Calculate metrics
            metrics_a = {
                'mse': mean_squared_error(y_a, pred_a),
                'r2': r2_score(y_a, pred_a),
                'error': np.mean(np.abs(y_a - pred_a) / y_a)
            }
            
            metrics_b = {
                'mse': mean_squared_error(y_b, pred_b),
                'r2': r2_score(y_b, pred_b),
                'error': np.mean(np.abs(y_b - pred_b) / y_b)
            }
            
            # Statistical tests
            _, p_value = stats.ttest_ind(
                np.abs(y_a - pred_a),
                np.abs(y_b - pred_b)
            )
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt((np.std(pred_a)**2 + np.std(pred_b)**2) / 2)
            effect_size = (np.mean(pred_b) - np.mean(pred_a)) / pooled_std
            
            # Calculate confidence interval
            ci = stats.t.interval(
                0.95,
                len(pred_b) - 1,
                loc=np.mean(pred_b - pred_a),
                scale=stats.sem(pred_b - pred_a)
            )
            
            # Calculate overall improvement
            improvement = (
                (metrics_b['r2'] - metrics_a['r2']) / metrics_a['r2'] +
                (metrics_a['mse'] - metrics_b['mse']) / metrics_a['mse']
            ) / 2 * 100
            
            # Determine significance
            significant = (
                p_value < self.config['significance_level'] and
                abs(effect_size) > self.config['min_effect_size']
            )
            
            return ABTestResults(
                metric=metric,
                variant_a=variant_a,
                variant_b=variant_b,
                sample_size=len(y_a),
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ci,
                significant=significant,
                improvement=improvement,
                metrics_comparison={
                    'mse': (metrics_a['mse'], metrics_b['mse']),
                    'r2': (metrics_a['r2'], metrics_b['r2']),
                    'error': (metrics_a['error'], metrics_b['error'])
                }
            )
            
        except Exception as e:
            logger.error(f"Error comparing models for {metric}: {e}")
            return None

    def create_comparison_visualization(self) -> go.Figure:
        """Create visualization of A/B test results."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Model Improvements",
                "Statistical Significance",
                "Effect Sizes",
                "Metric Comparisons"
            )
        )
        
        metrics = list(self.results.keys())
        
        # Improvements
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.improvement for r in self.results.values()],
                name="Improvement %",
                marker_color=[
                    '#2ecc71' if r.improvement > 0 else '#e74c3c'
                    for r in self.results.values()
                ]
            ),
            row=1, col=1
        )
        
        # P-values
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[-np.log10(r.p_value) for r in self.results.values()],
                name="-log10(p-value)",
                marker_color=[
                    '#2ecc71' if r.significant else '#95a5a6'
                    for r in self.results.values()
                ]
            ),
            row=1, col=2
        )
        
        # Effect sizes
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=[r.effect_size for r in self.results.values()],
                name="Effect Size",
                marker_color='#3498db'
            ),
            row=2, col=1
        )
        
        # Metric comparisons
        for metric in metrics:
            result = self.results[metric]
            for i, (metric_name, (old_val, new_val)) in enumerate(result.metrics_comparison.items()):
                fig.add_trace(
                    go.Bar(
                        x=[f"{metric}-{metric_name}"],
                        y=[new_val - old_val],
                        name=f"{metric} {metric_name}",
                        marker_color='#e67e22'
                    ),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="A/B Test Results"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate A/B test report."""
        fig = self.create_comparison_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model A/B Test Results</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .test-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-result {{
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                }}
                .significant {{
                    background: #d4edda;
                    border-left: 4px solid #28a745;
                }}
                .not-significant {{
                    background: #fff3cd;
                    border-left: 4px solid #ffc107;
                }}
                .negative {{
                    background: #f8d7da;
                    border-left: 4px solid #dc3545;
                }}
                .metric-value {{
                    font-weight: bold;
                }}
                .improvement {{
                    color: #28a745;
                }}
                .degradation {{
                    color: #dc3545;
                }}
            </style>
        </head>
        <body>
            <h1>Model A/B Test Results</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="test-container">
                <h2>Test Configuration</h2>
                <ul>
        """
        
        for name, value in self.config.items():
            html += f"<li>{name}: {value}</li>"
        
        html += """
                </ul>
            </div>
            
            <div class="test-container">
        """
        
        for metric, result in self.results.items():
            status_class = (
                "significant" if result.significant and result.improvement > 0
                else "negative" if result.improvement < 0
                else "not-significant"
            )
            
            html += f"""
                <div class="metric-result {status_class}">
                    <h3>{metric.replace('_', ' ').title()}</h3>
                    <p>
                        Improvement: 
                        <span class="metric-value {'improvement' if result.improvement > 0 else 'degradation'}">
                            {result.improvement:+.1f}%
                        </span>
                    </p>
                    <ul>
                        <li>Sample Size: {result.sample_size}</li>
                        <li>P-value: {result.p_value:.4f}</li>
                        <li>Effect Size: {result.effect_size:.3f}</li>
                        <li>95% CI: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]</li>
                    </ul>
                    
                    <h4>Metric Comparison (A → B)</h4>
                    <ul>
            """
            
            for metric_name, (old_val, new_val) in result.metrics_comparison.items():
                change = (new_val - old_val) / old_val * 100
                html += f"""
                    <li>{metric_name.upper()}: {old_val:.3f} → {new_val:.3f} 
                        <span class="{'improvement' if change > 0 else 'degradation'}">
                            ({change:+.1f}%)
                        </span>
                    </li>
                """
            
            html += """
                    </ul>
                </div>
            """
        
        html += f"""
            </div>

            <div class="test-container">
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
        
        tester = ModelABTester(model_dir, history_file)
        
        # Compare current vs new models
        for metric_file in model_dir.glob("*_regressor.pkl"):
            metric = metric_file.name.split('_')[0]
            if (model_dir / f"{metric}_previous_regressor.pkl").exists():
                result = tester.compare_models(metric, "previous", "current")
                if result:
                    tester.results[metric] = result
        
        output_path = results_dir / "ab_test_results.html"
        tester.generate_report(output_path)
        
        print(f"\nA/B test report generated at: {output_path}")
        
        # Exit with failure if any significant degradation
        has_degradation = any(
            r.significant and r.improvement < 0
            for r in tester.results.values()
        )
        return 1 if has_degradation else 0
        
    except Exception as e:
        logger.error(f"Error during A/B testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
