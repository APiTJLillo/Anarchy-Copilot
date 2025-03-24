#!/usr/bin/env python3
"""Meta-analysis of model experiments and improvement patterns."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from collections import defaultdict
from scipy import stats
import networkx as nx
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MetaAnalysisResults:
    """Results of experiment meta-analysis."""
    total_experiments: int
    improvement_patterns: Dict[str, Any]
    success_factors: Dict[str, float]
    feature_importance: Dict[str, float]
    correlations: Dict[str, float]
    recommendations: List[str]

class ExperimentAnalyzer:
    """Analyze patterns and insights across experiments."""
    
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments = self._load_experiments()
        self.results: Optional[MetaAnalysisResults] = None

    def _load_experiments(self) -> pd.DataFrame:
        """Load all experiments into a DataFrame."""
        records = []
        for file in self.experiments_dir.glob("experiment_*.json"):
            try:
                data = json.loads(file.read_text())
                records.append({
                    "timestamp": datetime.fromisoformat(data["timestamp"]),
                    "metric": data["metric"],
                    "variants": len(data["variants"]),
                    "winner": data["winner"] is not None,
                    "improvement": data["improvement"],
                    "commit": data["commit"],
                    "config": data["config"],
                    "significance": data["significance"],
                    "effect_size": data["effect_size"]
                })
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        return pd.DataFrame(records)

    def analyze_improvement_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in model improvements."""
        if self.experiments.empty:
            return {}
        
        patterns = {
            "success_rate_by_metric": {},
            "avg_improvement_by_metric": {},
            "time_based_patterns": {},
            "variant_count_impact": {}
        }
        
        # Success rate by metric
        patterns["success_rate_by_metric"] = (
            self.experiments.groupby("metric")["winner"]
            .mean()
            .to_dict()
        )
        
        # Average improvement by metric
        patterns["avg_improvement_by_metric"] = (
            self.experiments.groupby("metric")["improvement"]
            .agg(["mean", "std"])
            .to_dict("index")
        )
        
        # Time-based patterns
        self.experiments["month"] = self.experiments["timestamp"].dt.to_period("M")
        patterns["time_based_patterns"] = (
            self.experiments.groupby("month")["improvement"]
            .mean()
            .to_dict()
        )
        
        # Impact of variant count
        patterns["variant_count_impact"] = (
            self.experiments.groupby("variants")[["improvement", "winner"]]
            .agg({
                "improvement": "mean",
                "winner": "mean"
            })
            .to_dict("index")
        )
        
        return patterns

    def analyze_success_factors(self) -> Dict[str, float]:
        """Analyze factors contributing to successful experiments."""
        factors = {}
        
        if not self.experiments.empty:
            # Correlation between effect size and improvement
            factors["effect_size_correlation"] = self.experiments[
                ["effect_size", "improvement"]
            ].corr().iloc[0, 1]
            
            # Impact of significance level
            significance_groups = self.experiments.groupby(
                pd.qcut(self.experiments["significance"], 4)
            )["winner"].mean()
            factors["significance_impact"] = (
                significance_groups.iloc[-1] - significance_groups.iloc[0]
            )
            
            # Success rate with different variant counts
            variant_success = self.experiments.groupby("variants")["winner"].mean()
            factors["optimal_variant_count"] = variant_success.idxmax()
            
            # Time-based success patterns
            self.experiments["hour"] = self.experiments["timestamp"].dt.hour
            hour_success = self.experiments.groupby("hour")["winner"].mean()
            factors["best_time_of_day"] = hour_success.idxmax()
        
        return factors

    def extract_feature_importance(self) -> Dict[str, float]:
        """Extract importance of different features for success."""
        if self.experiments.empty:
            return {}
        
        # Prepare feature matrix
        features = pd.DataFrame()
        features["effect_size"] = self.experiments["effect_size"]
        features["variants"] = self.experiments["variants"]
        features["significance"] = self.experiments["significance"]
        features["hour"] = self.experiments["timestamp"].dt.hour
        features["day_of_week"] = self.experiments["timestamp"].dt.dayofweek
        
        # Target variable
        y = self.experiments["winner"]
        
        # Train a simple model
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(features, y)
        
        # Extract feature importance
        importance = dict(zip(features.columns, model.feature_importances_))
        
        return importance

    def analyze_correlations(self) -> Dict[str, float]:
        """Analyze correlations between different metrics."""
        if self.experiments.empty:
            return {}
        
        correlations = {}
        
        # Correlation between improvement and other metrics
        for col in ["effect_size", "significance", "variants"]:
            correlations[f"improvement_vs_{col}"] = self.experiments[
                ["improvement", col]
            ].corr().iloc[0, 1]
        
        # Time-based correlations
        self.experiments["days_since_start"] = (
            self.experiments["timestamp"] - self.experiments["timestamp"].min()
        ).dt.days
        
        correlations["improvement_over_time"] = self.experiments[
            ["improvement", "days_since_start"]
        ].corr().iloc[0, 1]
        
        return correlations

    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        if not self.results:
            return recommendations
        
        # Based on success factors
        if "optimal_variant_count" in self.results.success_factors:
            recommendations.append(
                f"Use {self.results.success_factors['optimal_variant_count']} "
                "variants for optimal results"
            )
        
        # Based on correlations
        for metric, corr in self.results.correlations.items():
            if abs(corr) > 0.5:
                direction = "positively" if corr > 0 else "negatively"
                recommendations.append(
                    f"{metric.replace('_', ' ').title()} is {direction} "
                    "correlated with success"
                )
        
        # Based on improvement patterns
        patterns = self.results.improvement_patterns
        if patterns.get("time_based_patterns"):
            best_month = max(
                patterns["time_based_patterns"].items(),
                key=lambda x: x[1]
            )[0]
            recommendations.append(
                f"Historical performance is best during {best_month}"
            )
        
        # Based on feature importance
        if self.results.feature_importance:
            top_feature = max(
                self.results.feature_importance.items(),
                key=lambda x: x[1]
            )[0]
            recommendations.append(
                f"Focus on optimizing {top_feature.replace('_', ' ')} "
                "for better results"
            )
        
        return recommendations

    def run_analysis(self) -> MetaAnalysisResults:
        """Run complete meta-analysis."""
        try:
            improvement_patterns = self.analyze_improvement_patterns()
            success_factors = self.analyze_success_factors()
            feature_importance = self.extract_feature_importance()
            correlations = self.analyze_correlations()
            recommendations = self.generate_recommendations()
            
            self.results = MetaAnalysisResults(
                total_experiments=len(self.experiments),
                improvement_patterns=improvement_patterns,
                success_factors=success_factors,
                feature_importance=feature_importance,
                correlations=correlations,
                recommendations=recommendations
            )
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error during meta-analysis: {e}")
            raise

    def create_analysis_visualization(self) -> go.Figure:
        """Create visualization of meta-analysis results."""
        if not self.results:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Success Rate by Metric",
                "Improvement Distribution",
                "Time-Based Patterns",
                "Feature Importance",
                "Correlation Matrix",
                "Variant Count Impact"
            )
        )
        
        # Success rate by metric
        metrics = list(self.results.improvement_patterns["success_rate_by_metric"].keys())
        success_rates = list(self.results.improvement_patterns["success_rate_by_metric"].values())
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=success_rates,
                name="Success Rate",
                marker_color='#2ecc71'
            ),
            row=1, col=1
        )
        
        # Improvement distribution
        improvements = self.experiments["improvement"]
        fig.add_trace(
            go.Histogram(
                x=improvements,
                name="Improvements",
                marker_color='#3498db'
            ),
            row=1, col=2
        )
        
        # Time-based patterns
        time_patterns = self.results.improvement_patterns["time_based_patterns"]
        dates = [str(k) for k in time_patterns.keys()]
        values = list(time_patterns.values())
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode='lines+markers',
                name='Time Patterns',
                line=dict(color='#9b59b6')
            ),
            row=2, col=1
        )
        
        # Feature importance
        features = list(self.results.feature_importance.keys())
        importance = list(self.results.feature_importance.values())
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=importance,
                name="Feature Importance",
                marker_color='#e74c3c'
            ),
            row=2, col=2
        )
        
        # Correlation matrix
        correlations = pd.DataFrame([self.results.correlations]).corr()
        
        fig.add_trace(
            go.Heatmap(
                z=correlations,
                x=correlations.columns,
                y=correlations.columns,
                colorscale='RdBu',
                name="Correlations"
            ),
            row=3, col=1
        )
        
        # Variant count impact
        variant_impact = self.results.improvement_patterns["variant_count_impact"]
        variants = list(variant_impact.keys())
        improvements = [v["improvement"] for v in variant_impact.values()]
        
        fig.add_trace(
            go.Bar(
                x=[str(v) for v in variants],
                y=improvements,
                name="Variant Impact",
                marker_color='#f1c40f'
            ),
            row=3, col=2
        )
        
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Experiment Meta-Analysis"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate meta-analysis report."""
        if not self.results:
            self.run_analysis()
        
        fig = self.create_analysis_visualization()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Meta-Analysis Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .analysis-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .insight {{
                    margin: 10px 0;
                    padding: 10px;
                    background: #f8f9fa;
                    border-radius: 5px;
                }}
                .recommendation {{
                    margin: 10px 0;
                    padding: 10px;
                    background: #d4edda;
                    border-radius: 5px;
                    color: #155724;
                }}
                .metric {{
                    display: inline-block;
                    margin: 5px;
                    padding: 10px;
                    background: #e9ecef;
                    border-radius: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>Experiment Meta-Analysis Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="analysis-container">
                <h2>Overview</h2>
                <div class="insight">
                    <p>Total Experiments: {self.results.total_experiments}</p>
                    <p>Success Rate: {
                        np.mean(list(
                            self.results.improvement_patterns["success_rate_by_metric"].values()
                        )):.1%}
                    </p>
                    <p>Average Improvement: {
                        np.mean([
                            v["mean"] for v in 
                            self.results.improvement_patterns["avg_improvement_by_metric"].values()
                        ]):.1f}%
                    </p>
                </div>
            </div>
            
            <div class="analysis-container">
                <h2>Key Success Factors</h2>
                {self._format_success_factors()}
            </div>
            
            <div class="analysis-container">
                <h2>Recommendations</h2>
                {self._format_recommendations()}
            </div>
            
            <div class="analysis-container">
                <h2>Detailed Analysis</h2>
                {fig.to_html(full_html=False) if fig else ''}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html)

    def _format_success_factors(self) -> str:
        """Format success factors for HTML display."""
        html = "<div class='insight'><ul>"
        
        for factor, value in self.results.success_factors.items():
            html += f"<li>{factor.replace('_', ' ').title()}: {value:.2f}</li>"
        
        html += "</ul></div>"
        return html

    def _format_recommendations(self) -> str:
        """Format recommendations for HTML display."""
        html = ""
        
        for rec in self.results.recommendations:
            html += f"<div class='recommendation'>{rec}</div>"
        
        return html

def main() -> int:
    """Main entry point."""
    try:
        experiments_dir = Path("benchmark_results/experiments")
        if not experiments_dir.exists():
            logger.error("No experiments directory found")
            return 1
        
        analyzer = ExperimentAnalyzer(experiments_dir)
        analyzer.run_analysis()
        
        output_path = experiments_dir.parent / "meta_analysis.html"
        analyzer.generate_report(output_path)
        
        print(f"\nMeta-analysis report generated at: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error during meta-analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
