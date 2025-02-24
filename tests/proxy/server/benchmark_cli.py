#!/usr/bin/env python3
"""Command-line interface for running proxy benchmarks."""
import click
import asyncio
import sys
import logging
from pathlib import Path
from typing import List
import json
from datetime import datetime
import pandas as pd
from tabulate import tabulate
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .benchmark import BenchmarkConfig, run_benchmark, BenchmarkReporter

console = Console()
logger = logging.getLogger(__name__)

def setup_logging():
    """Configure logging for the benchmark tool."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('benchmark.log')
        ]
    )

@click.group()
def cli():
    """HTTPS Proxy Benchmark Tool."""
    setup_logging()

@cli.command()
@click.option('--proxy-url', default="http://localhost:8080", help="Proxy server URL")
@click.option('--connections', default=100, help="Number of concurrent connections")
@click.option('--duration', default=60, help="Test duration in seconds")
@click.option('--data-sizes', default="1024,65536,1048576", 
              help="Comma-separated list of data sizes in bytes")
@click.option('--warmup', default=5, help="Warmup time in seconds")
@click.option('--cooldown', default=5, help="Cooldown time in seconds")
@click.option('--name', default=None, help="Benchmark run name")
@click.option('--output', default="benchmark_reports", help="Output directory")
def run(proxy_url: str, connections: int, duration: int, data_sizes: str, 
        warmup: int, cooldown: int, name: str, output: str):
    """Run a benchmark against the proxy server."""
    try:
        sizes = [int(s) for s in data_sizes.split(",")]
    except ValueError:
        console.print("[red]Error: Invalid data sizes format[/red]")
        return

    config = BenchmarkConfig(
        concurrent_connections=connections,
        duration=duration,
        data_sizes=sizes,
        warmup_time=warmup,
        cooldown_time=cooldown,
        report_dir=output
    )

    if not name:
        name = datetime.now().strftime("benchmark_%Y%m%d_%H%M%S")

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Running benchmark: {name}", total=duration + warmup + cooldown)

        async def run_with_progress():
            metrics = await run_benchmark(proxy_url, config)
            progress.update(task, completed=duration + warmup + cooldown)
            return metrics

        metrics = asyncio.run(run_with_progress())

    reporter = BenchmarkReporter(config)
    reporter.generate_report(metrics, name)
    
    # Display summary
    stats = metrics.get_stats()
    console.print("\n[green]Benchmark Complete![/green]")
    console.print(f"\nSummary for [blue]{name}[/blue]:")
    console.print(f"Total Requests: {stats['total_requests']}")
    console.print(f"Requests/second: {stats['requests_per_second']:.2f}")
    console.print(f"Mean Latency: {stats['latency']['mean']*1000:.2f}ms")
    console.print(f"P95 Latency: {stats['latency']['p95']*1000:.2f}ms")
    console.print(f"\nFull report saved to: {output}/{name}.html")

@cli.command()
@click.argument('reports', nargs=-1, type=click.Path(exists=True))
@click.option('--output', default="comparison_report.html", help="Output file for comparison")
def compare(reports: List[str], output: str):
    """Compare multiple benchmark reports."""
    if len(reports) < 2:
        console.print("[red]Error: At least two reports are required for comparison[/red]")
        return

    results = []
    for report in reports:
        with open(report) as f:
            data = json.load(f)
            name = Path(report).stem
            results.append({
                "name": name,
                "rps": data["requests_per_second"],
                "mean_latency": data["latency"]["mean"] * 1000,  # Convert to ms
                "p95_latency": data["latency"]["p95"] * 1000,
                "error_rate": (data["errors"] / data["total_requests"] * 100),
                "cpu_mean": data["system"]["cpu_usage"]["mean"],
                "memory_max": data["system"]["memory_mb"]["max"]
            })

    # Create comparison table
    df = pd.DataFrame(results)
    table = tabulate(df, headers='keys', tablefmt='pipe', floatfmt=".2f")

    # Generate comparison plots
    import matplotlib.pyplot as plt
    
    # RPS Comparison
    plt.figure(figsize=(10, 6))
    plt.bar(df['name'], df['rps'])
    plt.title("Requests Per Second Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("comparison_rps.png")
    plt.close()
    
    # Latency Comparison
    plt.figure(figsize=(10, 6))
    x = range(len(df['name']))
    width = 0.35
    plt.bar(x, df['mean_latency'], width, label='Mean')
    plt.bar([i + width for i in x], df['p95_latency'], width, label='P95')
    plt.title("Latency Comparison")
    plt.xlabel("Benchmark")
    plt.ylabel("Latency (ms)")
    plt.xticks([i + width/2 for i in x], df['name'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_latency.png")
    plt.close()

    # Generate HTML report
    html = f"""
    <html>
    <head>
        <title>Benchmark Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Comparison Report</h1>
        <h2>Summary Table</h2>
        {table}
        
        <div class="chart">
            <h2>Performance Comparisons</h2>
            <img src="comparison_rps.png" alt="RPS Comparison">
            <img src="comparison_latency.png" alt="Latency Comparison">
        </div>
        
        <div>
            <h2>Analysis</h2>
            <p>Best RPS: {df['name'][df['rps'].idxmax()]} ({df['rps'].max():.2f})</p>
            <p>Best Latency: {df['name'][df['mean_latency'].idxmin()]} ({df['mean_latency'].min():.2f}ms)</p>
            <p>Most Efficient: {df['name'][df['cpu_mean'].idxmin()]} (CPU: {df['cpu_mean'].min():.1f}%)</p>
        </div>
    </body>
    </html>
    """

    with open(output, 'w') as f:
        f.write(html)

    console.print(f"\n[green]Comparison report generated: {output}[/green]")

@cli.command()
@click.option('--proxy-url', default="http://localhost:8080", help="Proxy server URL")
@click.option('--duration', default=300, help="Profile duration in seconds")
def profile(proxy_url: str, duration: int):
    """Run a CPU/memory profile of the proxy."""
    try:
        import cProfile
        import pstats
        
        console.print("[yellow]Starting CPU profiling...[/yellow]")
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        config = BenchmarkConfig(duration=duration)
        asyncio.run(run_benchmark(proxy_url, config))
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.dump_stats('proxy_profile.stats')
        
        # Generate report
        with open('profile_report.txt', 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats('cumulative')
            stats.print_stats()
        
        console.print("[green]Profile complete![/green]")
        console.print("Results saved to: profile_report.txt and proxy_profile.stats")
        
    except Exception as e:
        console.print(f"[red]Error during profiling: {e}[/red]")

if __name__ == '__main__':
    cli()
