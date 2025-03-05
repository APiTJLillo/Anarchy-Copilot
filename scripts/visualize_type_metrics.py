#!/usr/bin/env python3
"""Generate interactive visualizations for type checking metrics."""

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

def load_metrics(report_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON report."""
    try:
        return json.loads(report_path.read_text())
    except Exception as e:
        print(f"Error loading metrics from {report_path}: {e}")
        sys.exit(1)

def create_coverage_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create type coverage visualization."""
    coverage = metrics['type_coverage']
    
    # Create donut charts
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            'Files with Types',
            'Function Type Coverage',
            'Variable Type Coverage'
        ),
        specs=[[{'type': 'domain'}, {'type': 'domain'}, {'type': 'domain'}]]
    )

    # Files coverage
    fig.add_trace(go.Pie(
        values=[
            coverage['files_with_types'],
            coverage['total_files'] - coverage['files_with_types']
        ],
        labels=['Typed', 'Untyped'],
        name='Files',
        hole=.4,
        marker_colors=['#2ecc71', '#e74c3c']
    ), row=1, col=1)

    # Function coverage
    fig.add_trace(go.Pie(
        values=[
            coverage['typed_functions'],
            coverage['total_functions'] - coverage['typed_functions']
        ],
        labels=['Typed', 'Untyped'],
        name='Functions',
        hole=.4,
        marker_colors=['#2ecc71', '#e74c3c']
    ), row=1, col=2)

    # Variable coverage
    fig.add_trace(go.Pie(
        values=[
            coverage['typed_variables'],
            coverage['total_variables'] - coverage['typed_variables']
        ],
        labels=['Typed', 'Untyped'],
        name='Variables',
        hole=.4,
        marker_colors=['#2ecc71', '#e74c3c']
    ), row=1, col=3)

    fig.update_layout(
        title_text='Type Coverage Overview',
        showlegend=True,
        height=400
    )

    return fig

def create_performance_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create performance visualization."""
    perf = metrics['tests']
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Test Duration', 'Memory Usage'),
        specs=[[{"type": "bar"}], [{"type": "bar"}]]
    )

    # Duration chart
    durations = [t['duration'] for t in perf.values()]
    names = list(perf.keys())
    
    fig.add_trace(
        go.Bar(
            x=names,
            y=durations,
            name='Duration (s)',
            marker_color='#3498db'
        ),
        row=1, col=1
    )

    # Memory chart
    memory = [t['performance']['memory_delta_mb'] for t in perf.values()]
    
    fig.add_trace(
        go.Bar(
            x=names,
            y=memory,
            name='Memory (MB)',
            marker_color='#9b59b6'
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        title_text='Performance Metrics',
        showlegend=True
    )

    return fig

def create_test_results_chart(metrics: Dict[str, Any]) -> go.Figure:
    """Create test results visualization."""
    tests = metrics['tests']
    
    fig = go.Figure()

    # Add test results as a timeline
    for test_name, results in tests.items():
        color = '#2ecc71' if results['success'] else '#e74c3c'
        fig.add_trace(go.Scatter(
            x=[0, results['duration']],
            y=[test_name, test_name],
            mode='lines',
            name=test_name,
            line=dict(color=color, width=20),
            hovertemplate=(
                f"{test_name}<br>"
                f"Duration: %{x:.2f}s<br>"
                f"Status: {'✓ Success' if results['success'] else '✗ Failed'}"
            )
        ))

    fig.update_layout(
        title_text='Test Execution Timeline',
        xaxis_title='Duration (seconds)',
        yaxis_title='Test',
        height=300,
        showlegend=False
    )

    return fig

def generate_html_report(metrics: Dict[str, Any], output_path: Path) -> None:
    """Generate interactive HTML report."""
    coverage_fig = create_coverage_chart(metrics)
    performance_fig = create_performance_chart(metrics)
    results_fig = create_test_results_chart(metrics)

    # Create HTML with multiple charts
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Type Check Visualization</title>
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
            .metric {{
                display: inline-block;
                margin: 10px 20px;
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }}
            .metric-label {{
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <h1>Type Check Analysis</h1>
        <p>Generated on: {metrics['timestamp']}</p>

        <div class="summary">
            <h2>Summary</h2>
            <div class="metric">
                <div class="metric-value">{metrics['performance']['type_coverage_percent']:.1f}%</div>
                <div class="metric-label">Type Coverage</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['performance']['total_duration']:.2f}s</div>
                <div class="metric-label">Total Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['type_coverage']['stub_files']}</div>
                <div class="metric-label">Stub Files</div>
            </div>
        </div>

        <div class="chart-container">
            {coverage_fig.to_html(full_html=False)}
        </div>

        <div class="chart-container">
            {results_fig.to_html(full_html=False)}
        </div>

        <div class="chart-container">
            {performance_fig.to_html(full_html=False)}
        </div>
    </body>
    </html>
    """

    output_path.write_text(html)

def main() -> int:
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    report_file = project_root / "type_report" / "type_check_metrics.json"
    
    if not report_file.exists():
        print("No metrics file found. Run type tests first:")
        print("  python scripts/run_type_tests.py --report")
        return 1

    metrics = load_metrics(report_file)
    
    # Generate visualization
    output_path = project_root / "type_report" / "type_check_visualization.html"
    generate_html_report(metrics, output_path)
    
    print(f"\nVisualization generated at: {output_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
