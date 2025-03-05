#!/usr/bin/env python3
"""Visualize historical trends in type suggestion benchmark results."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
except ImportError:
    print("Please install required packages:")
    print("  pip install plotly numpy")
    sys.exit(1)

class BenchmarkTrends:
    """Analyze and visualize benchmark trends over time."""
    
    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.results: List[Dict[str, Any]] = []
        self.load_historical_results()

    def load_historical_results(self) -> None:
        """Load all historical benchmark results."""
        if not self.results_dir.exists():
            return

        for result_file in sorted(self.results_dir.glob("type_suggestion_benchmarks_*.json")):
            try:
                data = json.loads(result_file.read_text())
                # Add timestamp from filename if not in data
                if 'timestamp' not in data:
                    timestamp = result_file.name.split('_')[-1].replace('.json', '')
                    data['timestamp'] = timestamp
                self.results.append(data)
            except Exception as e:
                print(f"Error loading {result_file}: {e}")

    def create_scaling_trend_chart(self) -> go.Figure:
        """Create visualization of scaling performance over time."""
        timestamps = []
        small_times = []
        medium_times = []
        large_times = []
        
        for result in self.results:
            timestamp = datetime.fromtimestamp(result['timestamp'])
            timestamps.append(timestamp)
            
            for size in ['small', 'medium', 'large']:
                if size in result:
                    duration = result[size]['data']['duration']
                    if size == 'small':
                        small_times.append(duration)
                    elif size == 'medium':
                        medium_times.append(duration)
                    elif size == 'large':
                        large_times.append(duration)
        
        fig = go.Figure()
        
        for size, times in [
            ('Small', small_times),
            ('Medium', medium_times),
            ('Large', large_times)
        ]:
            if times:
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=times,
                    name=f"{size} Files",
                    mode='lines+markers'
                ))
        
        fig.update_layout(
            title_text='Processing Time Trends by File Size',
            xaxis_title='Date',
            yaxis_title='Processing Time (seconds)',
            showlegend=True
        )
        
        return fig

    def create_memory_trend_chart(self) -> go.Figure:
        """Create visualization of memory usage trends."""
        timestamps = []
        mean_memory = []
        peak_memory = []
        
        for result in self.results:
            if 'memory_usage' in result:
                timestamp = datetime.fromtimestamp(result['timestamp'])
                memory_data = result['memory_usage']['data']
                
                timestamps.append(timestamp)
                mean_memory.append(memory_data['mean_memory_mb'])
                peak_memory.append(memory_data['max_memory_mb'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=mean_memory,
            name='Mean Memory Usage',
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=peak_memory,
            name='Peak Memory Usage',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title_text='Memory Usage Trends',
            xaxis_title='Date',
            yaxis_title='Memory Usage (MB)',
            showlegend=True
        )
        
        return fig

    def create_quality_trend_chart(self) -> go.Figure:
        """Create visualization of suggestion quality trends."""
        timestamps = []
        specificity_ratios = []
        any_ratios = []
        
        for result in self.results:
            if 'suggestion_quality' in result:
                timestamp = datetime.fromtimestamp(result['timestamp'])
                quality_data = result['suggestion_quality']['data']
                
                total = quality_data['total_suggestions']
                timestamps.append(timestamp)
                specificity_ratios.append(quality_data['specificity_ratio'] * 100)
                any_ratios.append(quality_data['any_suggestions'] / total * 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=specificity_ratios,
            name='Type Specificity',
            mode='lines+markers'
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=any_ratios,
            name='Any Type Usage',
            mode='lines+markers'
        ))
        
        fig.update_layout(
            title_text='Type Suggestion Quality Trends',
            xaxis_title='Date',
            yaxis_title='Percentage',
            showlegend=True
        )
        
        return fig

    def create_performance_trend_chart(self) -> go.Figure:
        """Create visualization of performance trends."""
        timestamps = []
        suggestions_per_sec = []
        feature_times = {'functions': [], 'classes': [], 'variables': []}
        
        for result in self.results:
            timestamp = datetime.fromtimestamp(result['timestamp'])
            
            # Get suggestion rate from medium size test
            if 'medium' in result:
                stats = result['medium']['data']['stats']
                suggestions_per_sec.append(stats['suggestions_per_second'])
                timestamps.append(timestamp)
            
            # Get feature-specific times
            for feature in feature_times.keys():
                key = f'feature_{feature}'
                if key in result:
                    feature_times[feature].append(result[key]['data']['duration'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Suggestions per Second',
                'Feature Processing Times'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=suggestions_per_sec,
                name='Suggestions/Second',
                mode='lines+markers'
            ),
            row=1, col=1
        )
        
        for feature, times in feature_times.items():
            if times:
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=times,
                        name=feature.title(),
                        mode='lines+markers'
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text='Performance Trends'
        )
        
        return fig

    def create_html_report(self, output_path: Path) -> None:
        """Generate complete HTML trend report."""
        scaling_fig = self.create_scaling_trend_chart()
        memory_fig = self.create_memory_trend_chart()
        quality_fig = self.create_quality_trend_chart()
        performance_fig = self.create_performance_trend_chart()
        
        # Calculate trend indicators
        latest_idx = -1
        week_ago_idx = -1
        for i, result in enumerate(self.results):
            timestamp = datetime.fromtimestamp(result['timestamp'])
            if i == len(self.results) - 1:
                latest_idx = i
            if timestamp < datetime.now() - timedelta(days=7):
                week_ago_idx = i
                break
        
        if latest_idx >= 0 and week_ago_idx >= 0:
            latest = self.results[latest_idx]
            week_ago = self.results[week_ago_idx]
            
            perf_change = (
                latest.get('medium', {}).get('data', {}).get('stats', {}).get('suggestions_per_second', 0) /
                week_ago.get('medium', {}).get('data', {}).get('stats', {}).get('suggestions_per_second', 1) - 1
            ) * 100
            
            quality_change = (
                latest.get('suggestion_quality', {}).get('data', {}).get('specificity_ratio', 0) /
                week_ago.get('suggestion_quality', {}).get('data', {}).get('specificity_ratio', 1) - 1
            ) * 100
        else:
            perf_change = 0
            quality_change = 0

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Type Suggestion Benchmark Trends</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 40px;
                    background: #f5f5f5;
                }}
                .chart-container {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .summary {{
                    background: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .trend-indicator {{
                    display: inline-block;
                    padding: 4px 8px;
                    border-radius: 4px;
                    color: white;
                    font-weight: bold;
                }}
                .trend-up {{ background-color: #2ecc71; }}
                .trend-down {{ background-color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Type Suggestion Benchmark Trends</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="summary">
                <h2>Weekly Changes</h2>
                <p>
                    Performance: 
                    <span class="trend-indicator {'trend-up' if perf_change >= 0 else 'trend-down'}">
                        {perf_change:+.1f}%
                    </span>
                </p>
                <p>
                    Quality: 
                    <span class="trend-indicator {'trend-up' if quality_change >= 0 else 'trend-down'}">
                        {quality_change:+.1f}%
                    </span>
                </p>
            </div>

            <div class="chart-container">
                {performance_fig.to_html(full_html=False)}
            </div>

            <div class="chart-container">
                {scaling_fig.to_html(full_html=False)}
            </div>

            <div class="chart-container">
                {memory_fig.to_html(full_html=False)}
            </div>

            <div class="chart-container">
                {quality_fig.to_html(full_html=False)}
            </div>
        </body>
        </html>
        """
        
        output_path.write_text(html)

def main() -> int:
    """Main entry point."""
    results_dir = Path("benchmark_results")
    if not results_dir.exists():
        print("No benchmark results directory found.")
        return 1
    
    trends = BenchmarkTrends(results_dir)
    if not trends.results:
        print("No benchmark results found.")
        return 1
    
    output_path = results_dir / "benchmark_trends.html"
    trends.create_html_report(output_path)
    
    print(f"\nTrend visualization generated at: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
