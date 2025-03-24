#!/usr/bin/env python3
"""Visualize performance test results for alert throttling system."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def load_performance_results(results_dir: Path) -> Dict[str, Any]:
    """Load all performance test results."""
    results = {}
    
    for file_name in [
        "throttling_performance.json",
        "concurrent_performance.json",
        "storage_performance.json",
        "cleanup_performance.json"
    ]:
        file_path = results_dir / file_name
        if file_path.exists():
            try:
                results[file_name.replace('.json', '')] = json.loads(file_path.read_text())
            except Exception as e:
                print(f"Error loading {file_name}: {e}", file=sys.stderr)
    
    return results

def create_throughput_chart(data: Dict[str, Any]) -> go.Figure:
    """Create throughput visualization."""
    sizes = sorted(map(int, data.keys()))
    alerts_per_second = [data[str(size)]["alerts_per_second"] for size in sizes]
    memory_usage = [data[str(size)]["average_memory_mb"] for size in sizes]
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Throughput by Load Size", "Memory Usage by Load Size"),
        vertical_spacing=0.15
    )
    
    # Throughput line
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=alerts_per_second,
            mode='lines+markers',
            name='Alerts/Second',
            line=dict(color='#2ecc71')
        ),
        row=1, col=1
    )
    
    # Memory usage line
    fig.add_trace(
        go.Scatter(
            x=sizes,
            y=memory_usage,
            mode='lines+markers',
            name='Memory (MB)',
            line=dict(color='#3498db')
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Alert Throttling Performance Metrics",
        showlegend=True
    )
    
    fig.update_xaxes(type="log", title_text="Number of Alerts", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Number of Alerts", row=2, col=1)
    fig.update_yaxes(title_text="Alerts per Second", row=1, col=1)
    fig.update_yaxes(title_text="Memory Usage (MB)", row=2, col=1)
    
    return fig

def create_concurrent_chart(data: Dict[str, Any]) -> go.Figure:
    """Create concurrent performance visualization."""
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=data["alerts_per_second"],
        domain={'x': [0, 0.5], 'y': [0, 1]},
        title={'text': "Alerts/Second"},
        gauge={
            'axis': {'range': [0, 1000]},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 500
            },
            'steps': [
                {'range': [0, 250], 'color': "#ff7675"},
                {'range': [250, 500], 'color': "#fdcb6e"},
                {'range': [500, 1000], 'color': "#00b894"}
            ]
        }
    ))
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=data["total_memory_mb"],
        domain={'x': [0.6, 1], 'y': [0, 1]},
        title={'text': "Memory Usage (MB)"},
        gauge={
            'axis': {'range': [0, 200]},
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 100
            },
            'steps': [
                {'range': [0, 50], 'color': "#00b894"},
                {'range': [50, 100], 'color': "#fdcb6e"},
                {'range': [100, 200], 'color': "#ff7675"}
            ]
        }
    ))
    
    fig.update_layout(
        title_text="Concurrent Performance Metrics",
        height=400
    )
    
    return fig

def create_storage_chart(data: Dict[str, Any]) -> go.Figure:
    """Create storage performance visualization."""
    metrics = ['avg_write_ms', 'max_write_ms', 'avg_read_ms', 'max_read_ms']
    values = [data[metric] for metric in metrics]
    labels = ['Avg Write', 'Max Write', 'Avg Read', 'Max Read']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("I/O Latency", "File Size"),
        specs=[[{"type": "bar"}, {"type": "indicator"}]]
    )
    
    # I/O latency bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=['#2ecc71', '#e74c3c', '#3498db', '#e67e22']
        ),
        row=1, col=1
    )
    
    # File size gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=data["file_size_kb"],
            title={'text': "Storage Size (KB)"},
            gauge={
                'axis': {'range': [0, 2000]},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1000
                }
            },
            domain={'x': [0.6, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Storage Performance Metrics",
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Milliseconds", row=1, col=1)
    
    return fig

def create_cleanup_chart(data: Dict[str, Any]) -> go.Figure:
    """Create cleanup performance visualization."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cleanup Duration", "Memory Impact"),
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Cleanup duration gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=data["avg_cleanup_ms"],
            title={'text': "Avg Cleanup Time (ms)"},
            gauge={
                'axis': {'range': [0, 200]},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 100
                }
            },
            domain={'x': [0, 0.4], 'y': [0, 1]}
        ),
        row=1, col=1
    )
    
    # Memory impact gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=data["memory_impact_mb"],
            title={'text': "Memory Impact (MB)"},
            gauge={
                'axis': {'range': [0, 20]},
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 10
                }
            },
            domain={'x': [0.6, 1], 'y': [0, 1]}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Cleanup Performance Metrics"
    )
    
    return fig

def create_html_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate complete HTML performance report."""
    throughput_fig = create_throughput_chart(results["throttling_performance"])
    concurrent_fig = create_concurrent_chart(results["concurrent_performance"])
    storage_fig = create_storage_chart(results["storage_performance"])
    cleanup_fig = create_cleanup_chart(results["cleanup_performance"])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Alert Throttling Performance Report</title>
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
            .threshold {{
                padding: 4px 8px;
                border-radius: 4px;
                color: white;
                font-weight: bold;
                display: inline-block;
            }}
            .pass {{ background-color: #2ecc71; }}
            .fail {{ background-color: #e74c3c; }}
        </style>
    </head>
    <body>
        <h1>Alert Throttling Performance Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <h2>Performance Summary</h2>
            <p>
                Throughput: 
                <span class="threshold {'pass' if results['concurrent_performance']['alerts_per_second'] >= 500 else 'fail'}">
                    {results['concurrent_performance']['alerts_per_second']:.1f} alerts/sec
                </span>
            </p>
            <p>
                Memory Usage: 
                <span class="threshold {'pass' if results['concurrent_performance']['total_memory_mb'] <= 100 else 'fail'}">
                    {results['concurrent_performance']['total_memory_mb']:.1f} MB
                </span>
            </p>
            <p>
                Storage Size: 
                <span class="threshold {'pass' if results['storage_performance']['file_size_kb'] <= 1000 else 'fail'}">
                    {results['storage_performance']['file_size_kb']:.1f} KB
                </span>
            </p>
            <p>
                Cleanup Time: 
                <span class="threshold {'pass' if results['cleanup_performance']['avg_cleanup_ms'] <= 100 else 'fail'}">
                    {results['cleanup_performance']['avg_cleanup_ms']:.1f} ms
                </span>
            </p>
        </div>

        <div class="chart-container">
            {throughput_fig.to_html(full_html=False)}
        </div>

        <div class="chart-container">
            {concurrent_fig.to_html(full_html=False)}
        </div>

        <div class="chart-container">
            {storage_fig.to_html(full_html=False)}
        </div>

        <div class="chart-container">
            {cleanup_fig.to_html(full_html=False)}
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
        
        results = load_performance_results(results_dir)
        if not results:
            print("No performance results found.")
            return 1
        
        output_path = results_dir / "throttling_performance_report.html"
        create_html_report(results, output_path)
        
        print(f"\nPerformance report generated at: {output_path}")
        return 0
        
    except Exception as e:
        print(f"Error generating performance report: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
