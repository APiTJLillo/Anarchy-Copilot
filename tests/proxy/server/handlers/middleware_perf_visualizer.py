"""Visualization tools for middleware performance analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

class PerformanceVisualizer:
    """Generate visualizations for middleware performance data."""
    
    def __init__(self, output_dir: str = "performance_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set style
        plt.style.use('seaborn-darkgrid')
        sns.set_palette("husl")
    
    def plot_execution_times(self, data: Dict[str, List[float]], title: str = "Execution Times"):
        """Plot execution time distributions."""
        plt.figure(figsize=(12, 6))
        
        # Create violin plots
        df = pd.DataFrame(data)
        sns.violinplot(data=df)
        
        plt.title(f"{title} Distribution")
        plt.ylabel("Time (ms)")
        plt.xticks(rotation=45)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(self.output_dir / f"execution_times_{self.timestamp}.png")
        plt.close()
    
    def plot_throughput_comparison(self, throughputs: Dict[str, float]):
        """Plot throughput comparison bar chart."""
        plt.figure(figsize=(10, 6))
        
        x = list(throughputs.keys())
        y = list(throughputs.values())
        
        sns.barplot(x=x, y=y)
        
        plt.title("Throughput Comparison")
        plt.ylabel("Requests/second")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"throughput_{self.timestamp}.png")
        plt.close()
    
    def plot_memory_profile(self, memory_data: List[Dict[str, float]]):
        """Plot memory usage over time."""
        plt.figure(figsize=(12, 6))
        
        df = pd.DataFrame(memory_data)
        
        # Plot RSS and USS if available
        if "rss" in df.columns:
            plt.plot(df.index, df["rss"] / (1024 * 1024), label="RSS")
        if "uss" in df.columns:
            plt.plot(df.index, df["uss"] / (1024 * 1024), label="USS")
        
        plt.title("Memory Usage Over Time")
        plt.xlabel("Sample")
        plt.ylabel("Memory (MB)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"memory_profile_{self.timestamp}.png")
        plt.close()
    
    def plot_latency_heatmap(self, latency_data: List[List[float]]):
        """Plot latency heatmap for concurrent requests."""
        plt.figure(figsize=(12, 8))
        
        # Create heatmap data
        data = np.array(latency_data)
        
        sns.heatmap(
            data,
            cmap="YlOrRd",
            xticklabels=False,
            yticklabels=False
        )
        
        plt.title("Request Latency Heatmap")
        plt.xlabel("Time")
        plt.ylabel("Concurrent Requests")
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"latency_heatmap_{self.timestamp}.png")
        plt.close()
    
    def create_html_report(self, test_results: Dict[str, Any]):
        """Generate HTML performance report."""
        html = f"""
        <html>
        <head>
            <title>Middleware Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .perf-good {{ color: green; }}
                .perf-warning {{ color: orange; }}
                .perf-bad {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Middleware Performance Report - {self.timestamp}</h1>
            
            <div class="section">
                <h2>Summary</h2>
                <div class="metric">
                    <p>Average Throughput: {test_results.get('avg_throughput', 0):.2f} req/s</p>
                    <p>Mean Latency: {test_results.get('mean_latency', 0)*1000:.2f} ms</p>
                    <p>Memory Usage: {test_results.get('memory_usage', 0)/1024/1024:.2f} MB</p>
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Charts</h2>
                <div class="chart">
                    <img src="execution_times_{self.timestamp}.png" alt="Execution Times">
                </div>
                <div class="chart">
                    <img src="throughput_{self.timestamp}.png" alt="Throughput">
                </div>
                <div class="chart">
                    <img src="memory_profile_{self.timestamp}.png" alt="Memory Profile">
                </div>
                <div class="chart">
                    <img src="latency_heatmap_{self.timestamp}.png" alt="Latency Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Performance Analysis</h2>
                {self._generate_analysis_html(test_results)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(test_results)}
            </div>
        </body>
        </html>
        """
        
        with open(self.output_dir / f"report_{self.timestamp}.html", "w") as f:
            f.write(html)
    
    def _generate_analysis_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for performance analysis section."""
        analysis = []
        
        # Throughput analysis
        throughput = results.get('avg_throughput', 0)
        if throughput > 5000:
            analysis.append(
                f'<p class="perf-good">Excellent throughput: {throughput:.0f} req/s</p>'
            )
        elif throughput > 1000:
            analysis.append(
                f'<p class="perf-warning">Acceptable throughput: {throughput:.0f} req/s</p>'
            )
        else:
            analysis.append(
                f'<p class="perf-bad">Poor throughput: {throughput:.0f} req/s</p>'
            )
        
        # Latency analysis
        latency = results.get('mean_latency', 0) * 1000  # Convert to ms
        if latency < 1:
            analysis.append(
                f'<p class="perf-good">Excellent latency: {latency:.2f}ms</p>'
            )
        elif latency < 10:
            analysis.append(
                f'<p class="perf-warning">Acceptable latency: {latency:.2f}ms</p>'
            )
        else:
            analysis.append(
                f'<p class="perf-bad">Poor latency: {latency:.2f}ms</p>'
            )
        
        return "\n".join(analysis)
    
    def _generate_recommendations_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML for recommendations section."""
        recommendations = []
        
        # Throughput recommendations
        if results.get('avg_throughput', 0) < 1000:
            recommendations.append(
                "<p>Consider implementing request batching to improve throughput</p>"
            )
        
        # Memory recommendations
        memory_mb = results.get('memory_usage', 0) / 1024 / 1024
        if memory_mb > 100:
            recommendations.append(
                "<p>High memory usage detected. Consider implementing memory pooling</p>"
            )
        
        # Latency recommendations
        if results.get('mean_latency', 0) * 1000 > 10:
            recommendations.append(
                "<p>High latency detected. Consider optimizing middleware chain execution</p>"
            )
        
        if not recommendations:
            recommendations.append("<p>No specific recommendations - performance is good!</p>")
        
        return "\n".join(recommendations)

def create_visualization(test_results: Dict[str, Any], output_dir: str = "performance_reports"):
    """Create performance visualizations from test results."""
    visualizer = PerformanceVisualizer(output_dir)
    
    # Plot execution times
    if "execution_times" in test_results:
        visualizer.plot_execution_times(test_results["execution_times"])
    
    # Plot throughput
    if "throughput" in test_results:
        visualizer.plot_throughput_comparison(test_results["throughput"])
    
    # Plot memory profile
    if "memory_profile" in test_results:
        visualizer.plot_memory_profile(test_results["memory_profile"])
    
    # Plot latency heatmap
    if "latency_matrix" in test_results:
        visualizer.plot_latency_heatmap(test_results["latency_matrix"])
    
    # Generate HTML report
    visualizer.create_html_report(test_results)
    
    return visualizer.output_dir / f"report_{visualizer.timestamp}.html"

if __name__ == "__main__":
    # Example usage
    test_data = {
        "avg_throughput": 2500,
        "mean_latency": 0.002,
        "memory_usage": 52428800,  # 50MB
        "execution_times": {
            "middleware1": np.random.normal(0.002, 0.0005, 1000).tolist(),
            "middleware2": np.random.normal(0.003, 0.0007, 1000).tolist()
        },
        "throughput": {
            "single": 3000,
            "chain": 2000,
            "concurrent": 2500
        },
        "memory_profile": [
            {"rss": 50*1024*1024, "uss": 30*1024*1024},
            {"rss": 55*1024*1024, "uss": 32*1024*1024},
            {"rss": 52*1024*1024, "uss": 31*1024*1024}
        ],
        "latency_matrix": np.random.normal(0.002, 0.0005, (50, 100)).tolist()
    }
    
    report_path = create_visualization(test_data)
    print(f"Report generated: {report_path}")
