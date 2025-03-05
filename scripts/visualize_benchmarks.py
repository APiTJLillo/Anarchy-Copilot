#!/usr/bin/env python3
"""Generate visualizations for type suggestion benchmarks."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Please install plotly: pip install plotly")
    sys.exit(1)

def load_benchmark_results() -> Dict[str, Any]:
    """Load benchmark results from JSON file."""
    results_file = Path("benchmark_results/type_suggestion_benchmarks.json")
    if not results_file.exists():
        print("No benchmark results found. Run benchmarks first:")
        print("  pytest tests/benchmark_type_suggestions.py")
        sys.exit(1)
    
    return json.loads(results_file.read_text())

def create_scaling_chart(results: Dict[str, Any]) -> go.Figure:
    """Create visualization of scaling performance."""
    sizes = ['tiny', 'small', 'medium', 'large', 'xlarge']
    durations = []
    suggestions_per_sec = []
    
    for size in sizes:
        if size in results:
            durations.append(results[size]['data']['duration'])
            stats = results[size]['data']['stats']
            suggestions_per_sec.append(stats['suggestions_per_second'])
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Processing Time by File Size', 'Suggestions/Second by File Size')
    )
    
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=durations,
            mode='lines+markers',
            name='Duration'
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=suggestions_per_sec,
            mode='lines+markers',
            name='Suggestions/Second'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Type Suggestion Scaling Performance"
    )
    
    return fig

def create_memory_chart(results: Dict[str, Any]) -> go.Figure:
    """Create memory usage visualization."""
    if 'memory_usage' not in results:
        return None
    
    memory_data = results['memory_usage']['data']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Mean', 'Max'],
        y=[
            memory_data['mean_memory_mb'],
            memory_data['max_memory_mb']
        ],
        error_y=dict(
            type='data',
            array=[memory_data['stddev_memory_mb'], 0],
            visible=True
        )
    ))
    
    fig.update_layout(
        title_text='Memory Usage Analysis',
        yaxis_title='Memory Usage (MB)',
        showlegend=False
    )
    
    return fig

def create_feature_comparison(results: Dict[str, Any]) -> go.Figure:
    """Create feature performance comparison visualization."""
    features = []
    durations = []
    stddevs = []
    
    for key, data in results.items():
        if key.startswith('feature_'):
            feature = key.replace('feature_', '')
            features.append(feature)
            durations.append(data['data']['duration'])
            stddevs.append(data['data']['stddev'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=features,
        y=durations,
        error_y=dict(
            type='data',
            array=stddevs,
            visible=True
        ),
        name='Processing Time'
    ))
    
    fig.update_layout(
        title_text='Performance by Feature Type',
        yaxis_title='Processing Time (seconds)',
        showlegend=False
    )
    
    return fig

def create_quality_metrics(results: Dict[str, Any]) -> go.Figure:
    """Create suggestion quality visualization."""
    if 'suggestion_quality' not in results:
        return None
    
    metrics = results['suggestion_quality']['data']
    
    # Calculate percentages
    total = metrics['total_suggestions']
    percentages = {
        'Any Types': (metrics['any_suggestions'] / total) * 100,
        'Optional Types': (metrics['optional_suggestions'] / total) * 100,
        'Specific Types': (metrics['specific_types'] / total) * 100
    }
    
    fig = go.Figure(data=[
        go.Pie(
            labels=list(percentages.keys()),
            values=list(percentages.values()),
            hole=.3
        )
    ])
    
    fig.update_layout(
        title_text='Type Suggestion Quality Metrics',
        annotations=[{
            'text': f'Specificity: {metrics["specificity_ratio"]:.2%}',
            'x': 0.5, 'y': 0.5,
            'font_size': 12,
            'showarrow': False
        }]
    )
    
    return fig

def create_html_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate complete HTML report with all visualizations."""
    scaling_fig = create_scaling_chart(results)
    memory_fig = create_memory_chart(results)
    feature_fig = create_feature_comparison(results)
    quality_fig = create_quality_metrics(results)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Type Suggestion Benchmark Results</title>
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
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
            }}
        </style>
    </head>
    <body>
        <h1>Type Suggestion Benchmark Results</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Summary</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Largest File Tested</td>
                    <td>{results.get('xlarge', {}).get('data', {}).get('stats', {}).get('num_lines', 'N/A')} lines</td>
                </tr>
                <tr>
                    <td>Peak Memory Usage</td>
                    <td>{results.get('memory_usage', {}).get('data', {}).get('max_memory_mb', 'N/A')} MB</td>
                </tr>
                <tr>
                    <td>Quality Score</td>
                    <td>{results.get('suggestion_quality', {}).get('data', {}).get('specificity_ratio', 'N/A'):.2%}</td>
                </tr>
            </table>
        </div>

        <div class="chart-container">
            {scaling_fig.to_html(full_html=False) if scaling_fig else ''}
        </div>

        <div class="chart-container">
            {memory_fig.to_html(full_html=False) if memory_fig else ''}
        </div>

        <div class="chart-container">
            {feature_fig.to_html(full_html=False) if feature_fig else ''}
        </div>

        <div class="chart-container">
            {quality_fig.to_html(full_html=False) if quality_fig else ''}
        </div>
    </body>
    </html>
    """
    
    output_path.write_text(html)

def main() -> int:
    """Main entry point."""
    results = load_benchmark_results()
    
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "benchmark_visualization.html"
    create_html_report(results, output_path)
    
    print(f"\nVisualization generated at: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
