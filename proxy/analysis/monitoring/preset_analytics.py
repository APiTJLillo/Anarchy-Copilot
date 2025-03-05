"""Analytics for mutation filter presets."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import defaultdict

from .mutation_presets import PresetManager, FilterPreset
from tests.proxy.analysis.monitoring.test_mutation_coverage import MutationTestResult

@dataclass
class PresetStats:
    """Statistics for a preset's performance."""
    name: str
    usage_count: int = 0
    success_rate: float = 0.0
    avg_mutation_score: float = 0.0
    avg_runtime: float = 0.0
    error_rates: Dict[str, float] = field(default_factory=dict)
    operator_coverage: Dict[str, float] = field(default_factory=dict)
    last_used: Optional[datetime] = None
    performance_trend: List[float] = field(default_factory=list)

@dataclass
class AnalyticsConfig:
    """Configuration for preset analytics."""
    history_window: timedelta = timedelta(days=30)
    min_samples: int = 10
    trend_samples: int = 100
    anomaly_threshold: float = 2.0
    correlation_threshold: float = 0.7

class PresetAnalytics:
    """Analyze mutation filter preset performance."""
    
    def __init__(
        self,
        preset_manager: PresetManager,
        config: AnalyticsConfig = None
    ):
        self.preset_manager = preset_manager
        self.config = config or AnalyticsConfig()
        
        # Analytics storage
        self.stats: Dict[str, PresetStats] = {}
        self.history: List[Tuple[datetime, str, MutationTestResult]] = []
        
        # Analysis cache
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._recommendation_cache: Dict[str, List[str]] = {}
    
    async def record_result(
        self,
        preset_name: str,
        result: MutationTestResult
    ):
        """Record test result for analytics."""
        timestamp = datetime.now()
        self.history.append((timestamp, preset_name, result))
        
        # Update stats
        if preset_name not in self.stats:
            self.stats[preset_name] = PresetStats(name=preset_name)
        
        stats = self.stats[preset_name]
        stats.usage_count += 1
        stats.last_used = timestamp
        stats.avg_mutation_score = (
            (stats.avg_mutation_score * (stats.usage_count - 1) + result.mutation_score) /
            stats.usage_count
        )
        
        # Update error rates
        error_counts = defaultdict(int)
        for error in result.errors:
            error_type = error.split(":")[0]
            error_counts[error_type] += 1
        
        total_errors = sum(error_counts.values())
        if total_errors > 0:
            for error_type, count in error_counts.items():
                stats.error_rates[error_type] = count / total_errors
        
        # Update operator coverage
        for op, op_stats in result.operator_stats.items():
            total = sum(op_stats.values())
            if total > 0:
                stats.operator_coverage[op] = op_stats["killed"] / total
        
        # Update performance trend
        stats.performance_trend.append(result.mutation_score)
        if len(stats.performance_trend) > self.config.trend_samples:
            stats.performance_trend = stats.performance_trend[-self.config.trend_samples:]
        
        # Trim history
        cutoff = datetime.now() - self.config.history_window
        self.history = [
            (ts, name, res)
            for ts, name, res in self.history
            if ts > cutoff
        ]
    
    async def analyze_preset(
        self,
        preset_name: str
    ) -> Dict[str, Any]:
        """Analyze preset performance."""
        if preset_name not in self.stats:
            return {}
        
        stats = self.stats[preset_name]
        
        # Get recent results
        recent_results = [
            res for ts, name, res in self.history
            if name == preset_name
        ]
        
        if len(recent_results) < self.config.min_samples:
            return {
                "status": "insufficient_data",
                "samples_needed": self.config.min_samples - len(recent_results)
            }
        
        # Calculate statistics
        scores = [r.mutation_score for r in recent_results]
        
        analysis = {
            "summary": {
                "usage_count": stats.usage_count,
                "success_rate": stats.success_rate,
                "avg_score": stats.avg_mutation_score,
                "score_std": np.std(scores),
                "last_used": stats.last_used
            },
            "trends": {
                "score_trend": stats.performance_trend,
                "usage_trend": self._calculate_usage_trend(preset_name)
            },
            "coverage": {
                "operators": stats.operator_coverage,
                "error_types": stats.error_rates
            },
            "anomalies": self._detect_anomalies(preset_name),
            "recommendations": await self._generate_recommendations(preset_name)
        }
        
        return analysis
    
    def _calculate_usage_trend(
        self,
        preset_name: str
    ) -> List[int]:
        """Calculate usage trend over time."""
        if not self.history:
            return []
        
        # Create time buckets
        start = min(ts for ts, _, _ in self.history)
        end = max(ts for ts, _, _ in self.history)
        bucket_size = (end - start) / self.config.trend_samples
        
        buckets = defaultdict(int)
        for ts, name, _ in self.history:
            if name == preset_name:
                bucket = int((ts - start) / bucket_size)
                buckets[bucket] += 1
        
        return [
            buckets[i]
            for i in range(self.config.trend_samples)
        ]
    
    def _detect_anomalies(
        self,
        preset_name: str
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in preset performance."""
        anomalies = []
        stats = self.stats[preset_name]
        
        # Score anomalies
        if stats.performance_trend:
            mean = np.mean(stats.performance_trend)
            std = np.std(stats.performance_trend)
            threshold = self.config.anomaly_threshold * std
            
            for i, score in enumerate(stats.performance_trend):
                if abs(score - mean) > threshold:
                    anomalies.append({
                        "type": "score",
                        "index": i,
                        "value": score,
                        "expected": mean,
                        "deviation": (score - mean) / std
                    })
        
        # Coverage anomalies
        for op, coverage in stats.operator_coverage.items():
            if coverage < 0.5:  # Less than 50% coverage
                anomalies.append({
                    "type": "coverage",
                    "operator": op,
                    "value": coverage,
                    "recommendation": "Consider adjusting filters for better coverage"
                })
        
        # Error rate anomalies
        for error_type, rate in stats.error_rates.items():
            if rate > 0.3:  # More than 30% of errors
                anomalies.append({
                    "type": "error_rate",
                    "error_type": error_type,
                    "value": rate,
                    "recommendation": f"High {error_type} error rate detected"
                })
        
        return anomalies
    
    async def _calculate_correlation(
        self,
        preset1: str,
        preset2: str
    ) -> float:
        """Calculate correlation between preset performances."""
        key = tuple(sorted([preset1, preset2]))
        if key in self._correlation_cache:
            return self._correlation_cache[key]
        
        scores1 = self.stats[preset1].performance_trend
        scores2 = self.stats[preset2].performance_trend
        
        if not scores1 or not scores2:
            return 0.0
        
        # Align and pad scores
        max_len = max(len(scores1), len(scores2))
        padded1 = scores1 + [scores1[-1]] * (max_len - len(scores1))
        padded2 = scores2 + [scores2[-1]] * (max_len - len(scores2))
        
        correlation = np.corrcoef(padded1, padded2)[0, 1]
        self._correlation_cache[key] = correlation
        
        return correlation
    
    async def _generate_recommendations(
        self,
        preset_name: str
    ) -> List[Dict[str, Any]]:
        """Generate recommendations for preset improvement."""
        if preset_name in self._recommendation_cache:
            return self._recommendation_cache[preset_name]
        
        recommendations = []
        stats = self.stats[preset_name]
        
        # Performance recommendations
        if stats.avg_mutation_score < 0.7:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "message": "Consider broadening mutation operator coverage",
                "details": {
                    "current_score": stats.avg_mutation_score,
                    "target_score": 0.7
                }
            })
        
        # Coverage recommendations
        low_coverage = [
            (op, cov) for op, cov in stats.operator_coverage.items()
            if cov < 0.5
        ]
        if low_coverage:
            recommendations.append({
                "type": "coverage",
                "priority": "medium",
                "message": "Low coverage detected for some operators",
                "details": {
                    "operators": dict(low_coverage)
                }
            })
        
        # Similar preset recommendations
        similar_presets = []
        for other_name, other_stats in self.stats.items():
            if other_name != preset_name:
                correlation = await self._calculate_correlation(
                    preset_name,
                    other_name
                )
                if correlation > self.config.correlation_threshold:
                    similar_presets.append({
                        "name": other_name,
                        "correlation": correlation
                    })
        
        if similar_presets:
            recommendations.append({
                "type": "similarity",
                "priority": "low",
                "message": "Similar presets detected",
                "details": {
                    "presets": similar_presets
                }
            })
        
        self._recommendation_cache[preset_name] = recommendations
        return recommendations
    
    async def create_analytics_plots(
        self,
        preset_name: str
    ) -> Dict[str, go.Figure]:
        """Create analytics visualizations."""
        analysis = await self.analyze_preset(preset_name)
        if not analysis:
            return {}
        
        plots = {}
        
        # Performance trend plot
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                y=analysis["trends"]["score_trend"],
                mode="lines+markers",
                name="Mutation Score"
            )
        )
        
        # Add anomaly markers
        anomalies = [a for a in analysis["anomalies"] if a["type"] == "score"]
        if anomalies:
            trend_fig.add_trace(
                go.Scatter(
                    x=[a["index"] for a in anomalies],
                    y=[a["value"] for a in anomalies],
                    mode="markers",
                    marker=dict(
                        size=12,
                        symbol="x",
                        color="red"
                    ),
                    name="Anomalies"
                )
            )
        
        trend_fig.update_layout(
            title="Performance Trend",
            yaxis_title="Mutation Score"
        )
        plots["trend"] = trend_fig
        
        # Coverage plot
        coverage_fig = go.Figure(
            go.Bar(
                x=list(analysis["coverage"]["operators"].keys()),
                y=list(analysis["coverage"]["operators"].values()),
                name="Operator Coverage"
            )
        )
        coverage_fig.update_layout(
            title="Operator Coverage",
            yaxis_title="Coverage Rate"
        )
        plots["coverage"] = coverage_fig
        
        # Error distribution plot
        error_fig = go.Figure(
            go.Pie(
                labels=list(analysis["coverage"]["error_types"].keys()),
                values=list(analysis["coverage"]["error_types"].values()),
                hole=0.3,
                name="Error Distribution"
            )
        )
        error_fig.update_layout(
            title="Error Type Distribution"
        )
        plots["errors"] = error_fig
        
        # Usage heatmap
        usage_data = np.array(analysis["trends"]["usage_trend"]).reshape(-1, 10)
        heatmap_fig = go.Figure(
            go.Heatmap(
                z=usage_data,
                colorscale="Viridis",
                name="Usage Pattern"
            )
        )
        heatmap_fig.update_layout(
            title="Usage Pattern Heatmap"
        )
        plots["usage"] = heatmap_fig
        
        return plots

def create_preset_analytics(
    preset_manager: PresetManager,
    config: Optional[AnalyticsConfig] = None
) -> PresetAnalytics:
    """Create preset analytics."""
    return PresetAnalytics(preset_manager, config)

if __name__ == "__main__":
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        
        # Create test preset
        await manager.save_preset(
            "test_preset",
            "Test preset",
            {
                "operators": ["type_mutation"],
                "error_types": ["TypeError"],
                "score_range": [0.5, 1.0],
                "time_range": None
            }
        )
        
        # Record some test results
        for _ in range(20):
            result = MutationTestResult()
            result.total_mutations = 100
            result.killed_mutations = np.random.randint(50, 90)
            result.survived_mutations = 100 - result.killed_mutations
            result.operator_stats = {
                "type_mutation": {
                    "killed": result.killed_mutations,
                    "survived": result.survived_mutations
                }
            }
            
            await analytics.record_result("test_preset", result)
        
        # Get analysis
        analysis = await analytics.analyze_preset("test_preset")
        print(json.dumps(analysis, indent=2))
        
        # Generate plots
        plots = await analytics.create_analytics_plots("test_preset")
        for name, fig in plots.items():
            fig.write_html(f"test_preset_{name}.html")
    
    asyncio.run(main())
