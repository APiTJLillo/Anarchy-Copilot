#!/usr/bin/env python3
"""Multi-variant testing for throttling models."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import numpy as np
import pandas as pd
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import logging
from scipy import stats
from itertools import combinations
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VariantComparison:
    """Results of comparing two variants."""
    variant_a: str
    variant_b: str
    p_value: float
    effect_size: float
    relative_improvement: float
    winner: Optional[str]

@dataclass
class MetricResults:
    """Results for a specific metric."""
    metric: str
    variants: List[str]
    best_variant: str
    significances: Dict[Tuple[str, str], float]
    effects: Dict[Tuple[str, str], float]
    improvements: Dict[str, float]
    metrics: Dict[str, Dict[str, float]]
    comparisons: List[VariantComparison]

class MultiVariantTester:
    """Compare multiple model variants."""
    
    def __init__(self, 
                 model_dir: Path,
                 history_file: Path,
                 test_config: Optional[Dict[str, Any]] = None):
        self.model_dir = model_dir
        self.history_file = history_file
        self.config = test_config or {
            'significance_level': 0.05,
            'min_effect_size': 0.1,
            'min_sample_size': 1000,
            'crossover_ratio': 0.2,
            'multiple_testing_correction': 'bonferroni'
        }
        self.results: Dict[str, MetricResults] = {}

    def find_variants(self, metric: str) -> List[str]:
        """Find all variants for a metric."""
        variants = set()
        for model_file in self.model_dir.glob(f"{metric}_*_regressor.pkl"):
            variant = model_file.name.split('_')[1]
            if variant != 'scaler':
                variants.add(variant)
        return sorted(variants)

    def prepare_variant_data(self, data: pd.DataFrame, num_variants: int) -> List[pd.DataFrame]:
        """Prepare test data for multiple variants."""
        # Ensure balanced samples
        sample_size = min(
            len(data) // num_variants,
            self.config['min_sample_size']
        )
        
        # Create overlapping samples
        samples = []
        for i in range(num_variants):
            # Base sample
            base_indices = np.random.choice(
                data.index,
                size=int(sample_size * (1 - self.config['crossover_ratio'])),
                replace=False
            )
            
            # Crossover samples from other variants
            crossover_size = int(sample_size * self.config['crossover_ratio'] / (num_variants - 1))
            crossover_indices = []
            
            for j in range(num_variants):
                if j != i:
                    indices = np.random.choice(
                        data.index,
                        size=crossover_size,
                        replace=False
                    )
                    crossover_indices.extend(indices)
            
            # Combine samples
            sample = pd.concat([
                data.loc[base_indices],
                data.loc[crossover_indices]
            ])
            samples.append(sample)
        
        return samples

    def compare_variants(self, metric: str) -> MetricResults:
        """Compare all variants for a metric."""
        try:
            # Load data
            data = pd.read_json(self.history_file)
            variants = self.find_variants(metric)
            
            if len(variants) < 2:
                logger.warning(f"Not enough variants for {metric}")
                return None
            
            # Prepare samples
            samples = self.prepare_variant_data(data, len(variants))
            
            # Load models and make predictions
            predictions: Dict[str, np.ndarray] = {}
            metrics: Dict[str, Dict[str, float]] = {}
            scaler = joblib.load(self.model_dir / f"{metric}_scaler.pkl")
            
            for variant, sample in zip(variants, samples):
                model = joblib.load(self.model_dir / f"{metric}_{variant}_regressor.pkl")
                
                # Prepare features
                from scripts.predict_throttling_performance import PerformancePredictor
                predictor = PerformancePredictor(self.history_file)
                features = predictor._prepare_features(sample)
                X = scaler.transform(features)
                y_true = sample[metric].values
                
                # Make predictions
                y_pred = model.predict(X)
                predictions[variant] = y_pred
                
                # Calculate metrics
                metrics[variant] = {
                    'mse': mean_squared_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred),
                    'error': np.mean(np.abs(y_true - y_pred) / y_true)
                }
            
            # Perform pairwise comparisons
            comparisons = []
            significances = {}
            effects = {}
            improvements = {}
            
            for variant_a, variant_b in combinations(variants, 2):
                # Statistical test
                _, p_value = stats.ttest_ind(
                    predictions[variant_a],
                    predictions[variant_b]
                )
                
                # Effect size
                pooled_std = np.sqrt(
                    (np.std(predictions[variant_a])**2 + 
                     np.std(predictions[variant_b])**2) / 2
                )
                effect_size = (
                    np.mean(predictions[variant_b]) - 
                    np.mean(predictions[variant_a])
                ) / pooled_std
                
                # Relative improvement
                improvement = (
                    metrics[variant_b]['r2'] - metrics[variant_a]['r2']
                ) / metrics[variant_a]['r2'] * 100
                
                # Determine winner
                winner = None
                if (p_value < self.config['significance_level'] and 
                    abs(effect_size) > self.config['min_effect_size']):
                    winner = variant_b if improvement > 0 else variant_a
                
                comparison = VariantComparison(
                    variant_a=variant_a,
                    variant_b=variant_b,
                    p_value=p_value,
                    effect_size=effect_size,
                    relative_improvement=improvement,
                    winner=winner
                )
                comparisons.append(comparison)
                
                # Store results
                significances[(variant_a, variant_b)] = p_value
                effects[(variant_a, variant_b)] = effect_size
                improvements[variant_b] = improvement
            
            # Find best variant using graph analysis
            G = nx.DiGraph()
            for comp in comparisons:
                if comp.winner:
                    G.add_edge(comp.variant_a, comp.variant_b, weight=comp.effect_size)
            
            # Best variant has highest PageRank
            if len(G.nodes) > 0:
                pageranks = nx.pagerank(G)
                best_variant = max(pageranks.items(), key=lambda x: x[1])[0]
            else:
                best_variant = variants[0]
            
            return MetricResults(
                metric=metric,
                variants=variants,
                best_variant=best_variant,
                significances=significances,
                effects=effects,
                improvements=improvements,
                metrics=metrics,
                comparisons=comparisons
            )
            
        except Exception as e:
            logger.error(f"Error comparing variants for {metric}: {e}")
            return None

    def create_comparison_visualization(self, results: MetricResults) -> go.Figure:
        """Create visualization of variant comparisons."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "R² Scores by Variant",
                "Pairwise Improvements",
                "Effect Sizes",
                "Statistical Significance"
            )
        )
        
        # R² scores
        r2_scores = [
            results.metrics[variant]['r2']
            for variant in results.variants
        ]
        
        fig.add_trace(
            go.Bar(
                x=results.variants,
                y=r2_scores,
                name="R² Score",
                marker_color=[
                    '#2ecc71' if v == results.best_variant else '#3498db'
                    for v in results.variants
                ]
            ),
            row=1, col=1
        )
        
        # Pairwise improvements
        improvements = []
        pairs = []
        for comp in results.comparisons:
            improvements.append(comp.relative_improvement)
            pairs.append(f"{comp.variant_a} vs {comp.variant_b}")
        
        fig.add_trace(
            go.Bar(
                x=pairs,
                y=improvements,
                name="Improvement %",
                marker_color=[
                    '#2ecc71' if i > 0 else '#e74c3c'
                    for i in improvements
                ]
            ),
            row=1, col=2
        )
        
        # Effect sizes
        effect_sizes = [comp.effect_size for comp in results.comparisons]
        
        fig.add_trace(
            go.Bar(
                x=pairs,
                y=effect_sizes,
                name="Effect Size",
                marker_color='#3498db'
            ),
            row=2, col=1
        )
        
        # Statistical significance
        p_values = [-np.log10(comp.p_value) for comp in results.comparisons]
        significance_threshold = -np.log10(self.config['significance_level'])
        
        fig.add_trace(
            go.Bar(
                x=pairs,
                y=p_values,
                name="-log10(p-value)",
                marker_color=[
                    '#2ecc71' if p > significance_threshold else '#95a5a6'
                    for p in p_values
                ]
            ),
            row=2, col=2
        )
        
        # Add significance threshold line
        fig.add_hline(
            y=significance_threshold,
            line_dash="dash",
            line_color="red",
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Multi-Variant Comparison for {results.metric}"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate multi-variant test report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multi-Variant Test Results</title>
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
                .variant {{
                    display: inline-block;
                    padding: 4px 8px;
                    margin: 2px;
                    border-radius: 4px;
                }}
                .best-variant {{
                    background: #d4edda;
                    border: 1px solid #28a745;
                }}
                .other-variant {{
                    background: #e9ecef;
                    border: 1px solid #6c757d;
                }}
                .comparison {{
                    margin: 5px 0;
                    padding: 5px;
                    border-left: 4px solid #6c757d;
                }}
                .significant {{
                    border-left-color: #28a745;
                }}
            </style>
        </head>
        <body>
            <h1>Multi-Variant Test Results</h1>
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
        """
        
        for metric, results in self.results.items():
            if not results:
                continue
            
            fig = self.create_comparison_visualization(results)
            
            html += f"""
            <div class="test-container">
                <h2>{metric.replace('_', ' ').title()}</h2>
                
                <h3>Variants</h3>
                <div>
            """
            
            for variant in results.variants:
                variant_class = "best-variant" if variant == results.best_variant else "other-variant"
                html += f"""
                    <span class="variant {variant_class}">{variant}</span>
                """
            
            html += """
                </div>
                
                <h3>Pairwise Comparisons</h3>
            """
            
            for comp in results.comparisons:
                significant = (
                    comp.p_value < self.config['significance_level'] and
                    abs(comp.effect_size) > self.config['min_effect_size']
                )
                
                html += f"""
                <div class="comparison{'significant' if significant else ''}">
                    <strong>{comp.variant_a} vs {comp.variant_b}</strong><br>
                    Improvement: <span style="color: {'#28a745' if comp.relative_improvement > 0 else '#dc3545'}">
                        {comp.relative_improvement:+.1f}%
                    </span><br>
                    Effect Size: {comp.effect_size:.3f}<br>
                    P-value: {comp.p_value:.4f}
                    {f'<br>Winner: {comp.winner}' if comp.winner else ''}
                </div>
                """
            
            html += f"""
                <div class="chart-container">
                    {fig.to_html(full_html=False)}
                </div>
            </div>
            """
        
        html += """
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
        
        tester = MultiVariantTester(model_dir, history_file)
        
        # Test all metrics
        metrics = set()
        for model_file in model_dir.glob("*_regressor.pkl"):
            metric = model_file.name.split('_')[0]
            metrics.add(metric)
        
        for metric in metrics:
            results = tester.compare_variants(metric)
            if results:
                tester.results[metric] = results
        
        output_path = results_dir / "multi_variant_results.html"
        tester.generate_report(output_path)
        
        print(f"\nMulti-variant test report generated at: {output_path}")
        
        # Exit with failure if no clear winners
        has_winners = any(
            r and r.best_variant for r in tester.results.values()
        )
        return 0 if has_winners else 1
        
    except Exception as e:
        logger.error(f"Error during multi-variant testing: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
