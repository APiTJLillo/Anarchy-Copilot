#!/usr/bin/env python3
"""Track and analyze model experiments and variant tests."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentResult:
    """Results of a model experiment."""
    timestamp: datetime
    metric: str
    variants: List[str]
    winner: Optional[str]
    improvement: float
    commit: str
    config: Dict[str, Any]
    metrics: Dict[str, Dict[str, float]]
    significance: float
    effect_size: float

class ExperimentTracker:
    """Track and analyze model experiments."""
    
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, List[ExperimentResult]] = defaultdict(list)
        self._load_history()

    def _load_history(self) -> None:
        """Load experiment history."""
        try:
            for file in sorted(self.experiments_dir.glob("experiment_*.json")):
                with open(file) as f:
                    data = json.load(f)
                    result = ExperimentResult(
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        metric=data["metric"],
                        variants=data["variants"],
                        winner=data["winner"],
                        improvement=data["improvement"],
                        commit=data["commit"],
                        config=data["config"],
                        metrics=data["metrics"],
                        significance=data["significance"],
                        effect_size=data["effect_size"]
                    )
                    self.history[result.metric].append(result)
        except Exception as e:
            logger.error(f"Error loading experiment history: {e}")

    def record_experiment(self, result: ExperimentResult) -> None:
        """Record a new experiment result."""
        try:
            # Save to history
            self.history[result.metric].append(result)
            
            # Save to file
            timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
            file_path = self.experiments_dir / f"experiment_{timestamp}_{result.metric}.json"
            
            data = {
                "timestamp": result.timestamp.isoformat(),
                "metric": result.metric,
                "variants": result.variants,
                "winner": result.winner,
                "improvement": result.improvement,
                "commit": result.commit,
                "config": result.config,
                "metrics": result.metrics,
                "significance": result.significance,
                "effect_size": result.effect_size
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error recording experiment: {e}")

    def analyze_progression(self, metric: str) -> Dict[str, Any]:
        """Analyze progression of model improvements."""
        if metric not in self.history:
            return {}
        
        experiments = sorted(self.history[metric], key=lambda x: x.timestamp)
        
        # Track metrics over time
        progression = {
            "timestamps": [],
            "improvements": [],
            "effect_sizes": [],
            "winners": [],
            "cumulative_improvement": 0.0,
            "success_rate": 0.0,
            "significant_changes": []
        }
        
        successful_experiments = 0
        
        for exp in experiments:
            progression["timestamps"].append(exp.timestamp)
            progression["improvements"].append(exp.improvement)
            progression["effect_sizes"].append(exp.effect_size)
            progression["winners"].append(exp.winner)
            
            if exp.winner and exp.improvement > 0:
                successful_experiments += 1
                progression["significant_changes"].append({
                    "date": exp.timestamp,
                    "improvement": exp.improvement,
                    "commit": exp.commit
                })
            
            progression["cumulative_improvement"] += max(0, exp.improvement)
        
        if experiments:
            progression["success_rate"] = successful_experiments / len(experiments)
        
        return progression

    def create_progression_visualization(self, metric: str) -> go.Figure:
        """Create visualization of experiment progression."""
        progression = self.analyze_progression(metric)
        if not progression:
            return None
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Improvements Over Time",
                "Effect Sizes",
                "Success Rate",
                "Cumulative Improvement"
            )
        )
        
        # Improvements over time
        fig.add_trace(
            go.Scatter(
                x=progression["timestamps"],
                y=progression["improvements"],
                mode='lines+markers',
                name='Improvement %',
                marker=dict(
                    color=[
                        '#2ecc71' if imp > 0 else '#e74c3c'
                        for imp in progression["improvements"]
                    ]
                )
            ),
            row=1, col=1
        )
        
        # Effect sizes
        fig.add_trace(
            go.Scatter(
                x=progression["timestamps"],
                y=progression["effect_sizes"],
                mode='lines+markers',
                name='Effect Size',
                marker=dict(color='#3498db')
            ),
            row=1, col=2
        )
        
        # Success rate over time
        cumulative_successes = np.cumsum([
            1 if imp > 0 and winner else 0
            for imp, winner in zip(
                progression["improvements"],
                progression["winners"]
            )
        ])
        experiment_counts = np.arange(1, len(progression["timestamps"]) + 1)
        
        fig.add_trace(
            go.Scatter(
                x=progression["timestamps"],
                y=cumulative_successes / experiment_counts,
                mode='lines',
                name='Success Rate',
                line=dict(color='#2ecc71')
            ),
            row=2, col=1
        )
        
        # Cumulative improvement
        cumulative_improvements = np.cumsum([
            max(0, imp) for imp in progression["improvements"]
        ])
        
        fig.add_trace(
            go.Scatter(
                x=progression["timestamps"],
                y=cumulative_improvements,
                mode='lines',
                name='Cumulative Improvement',
                fill='tozeroy',
                line=dict(color='#9b59b6')
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Experiment Progression for {metric}"
        )
        
        return fig

    def generate_report(self, output_path: Path) -> None:
        """Generate experiment tracking report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Tracking Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .report-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric-summary {{
                    margin: 10px 0;
                    padding: 10px;
                    border-radius: 5px;
                    background: #f8f9fa;
                }}
                .highlight {{
                    color: #28a745;
                    font-weight: bold;
                }}
                .timeline-item {{
                    margin: 10px 0;
                    padding: 10px;
                    border-left: 4px solid #3498db;
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
            <h1>Experiment Tracking Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for metric in sorted(self.history.keys()):
            progression = self.analyze_progression(metric)
            fig = self.create_progression_visualization(metric)
            
            html += f"""
            <div class="report-container">
                <h2>{metric.replace('_', ' ').title()}</h2>
                
                <div class="metric-summary">
                    <h3>Summary</h3>
                    <ul>
                        <li>Total Experiments: {len(self.history[metric])}</li>
                        <li>Success Rate: <span class="highlight">{progression['success_rate']:.1%}</span></li>
                        <li>Cumulative Improvement: <span class="highlight">{progression['cumulative_improvement']:.1f}%</span></li>
                        <li>Significant Changes: {len(progression['significant_changes'])}</li>
                    </ul>
                </div>
                
                <h3>Significant Changes Timeline</h3>
            """
            
            for change in progression["significant_changes"]:
                html += f"""
                <div class="timeline-item">
                    <strong>{change['date'].strftime('%Y-%m-%d %H:%M:%S')}</strong><br>
                    Improvement: <span class="improvement">+{change['improvement']:.1f}%</span><br>
                    Commit: {change['commit']}
                </div>
                """
            
            if fig:
                html += f"""
                <div class="chart-container">
                    {fig.to_html(full_html=False)}
                </div>
                """
            
            html += """
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
        experiments_dir = Path("benchmark_results/experiments")
        tracker = ExperimentTracker(experiments_dir)
        
        # Generate report
        output_path = experiments_dir.parent / "experiment_tracking.html"
        tracker.generate_report(output_path)
        
        print(f"\nExperiment tracking report generated at: {output_path}")
        return 0
        
    except Exception as e:
        logger.error(f"Error generating experiment tracking report: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
