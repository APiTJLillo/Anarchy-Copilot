"""Correlation analysis between metrics."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
import logging
from collections import defaultdict
import json
from pathlib import Path

from .metric_analytics import MetricAnalytics, AnalyticsConfig
from .adaptive_thresholds import AdaptiveThresholds

logger = logging.getLogger(__name__)

@dataclass
class CorrelationConfig:
    """Configuration for correlation analysis."""
    min_samples: int = 100
    correlation_threshold: float = 0.7
    causation_lag: int = 10
    max_lag: int = 24
    granularity: str = "1min"
    window_size: timedelta = timedelta(hours=24)
    min_mutual_info: float = 0.5
    max_correlations: int = 10

class MetricCorrelation:
    """Analyze correlations between metrics."""
    
    def __init__(self, config: CorrelationConfig):
        self.config = config
        self.correlation_cache: Dict[str, Dict[str, Any]] = {}
        self.mutual_info_cache: Dict[str, Dict[str, float]] = {}
        self.causal_graph: nx.DiGraph = nx.DiGraph()
    
    def analyze_correlations(
        self,
        metrics: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """Analyze correlations between metrics."""
        if len(metrics) < 2:
            return {"error": "Need at least two metrics"}
        
        # Create correlation matrix
        correlation_matrix = self._compute_correlations(metrics)
        
        # Find significant correlations
        significant = self._find_significant_correlations(correlation_matrix)
        
        # Compute mutual information
        mutual_info = self._compute_mutual_information(metrics)
        
        # Detect causality
        causality = self._detect_causality(metrics)
        
        # Build correlation graph
        graph = self._build_correlation_graph(
            significant,
            mutual_info,
            causality
        )
        
        # Find correlation clusters
        clusters = self._find_correlation_clusters(graph)
        
        return {
            "correlation_matrix": correlation_matrix.to_dict(),
            "significant_correlations": significant,
            "mutual_information": mutual_info,
            "causality_relationships": causality,
            "correlation_clusters": clusters,
            "summary": self._create_correlation_summary(
                correlation_matrix,
                significant,
                mutual_info,
                causality
            )
        }
    
    def _compute_correlations(
        self,
        metrics: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """Compute correlation matrix."""
        # Align timestamps
        df = pd.DataFrame(metrics)
        df = df.resample(self.config.granularity).mean()
        
        # Compute correlations
        return df.corr(method="pearson")
    
    def _find_significant_correlations(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Find significant correlations."""
        significant = []
        
        for i in range(len(correlation_matrix.index)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) >= self.config.correlation_threshold:
                    significant.append({
                        "metric1": correlation_matrix.index[i],
                        "metric2": correlation_matrix.columns[j],
                        "correlation": float(corr),
                        "strength": "strong" if abs(corr) > 0.8 else "moderate"
                    })
        
        return sorted(
            significant,
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )[:self.config.max_correlations]
    
    def _compute_mutual_information(
        self,
        metrics: Dict[str, pd.Series]
    ) -> Dict[str, Dict[str, float]]:
        """Compute mutual information between metrics."""
        mutual_info = {}
        metric_names = list(metrics.keys())
        
        for i, name1 in enumerate(metric_names):
            mutual_info[name1] = {}
            X = metrics[name1].values.reshape(-1, 1)
            
            for name2 in metric_names[i+1:]:
                y = metrics[name2].values
                mi = float(mutual_info_regression(X, y)[0])
                
                if mi >= self.config.min_mutual_info:
                    mutual_info[name1][name2] = mi
                    if name2 not in mutual_info:
                        mutual_info[name2] = {}
                    mutual_info[name2][name1] = mi
        
        return mutual_info
    
    def _detect_causality(
        self,
        metrics: Dict[str, pd.Series]
    ) -> List[Dict[str, Any]]:
        """Detect potential causal relationships."""
        causality = []
        metric_names = list(metrics.keys())
        
        for i, name1 in enumerate(metric_names):
            for name2 in metric_names[i+1:]:
                # Test different lags
                best_lag = 0
                best_corr = 0
                
                for lag in range(1, self.config.max_lag + 1):
                    series1 = metrics[name1][lag:]
                    series2 = metrics[name2][:-lag]
                    corr = series1.corr(series2)
                    
                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag
                
                if abs(best_corr) >= self.config.correlation_threshold:
                    direction = "forward" if best_corr > 0 else "inverse"
                    causality.append({
                        "cause": name1,
                        "effect": name2,
                        "lag": best_lag,
                        "correlation": float(best_corr),
                        "direction": direction
                    })
        
        return sorted(
            causality,
            key=lambda x: abs(x["correlation"]),
            reverse=True
        )
    
    def _build_correlation_graph(
        self,
        significant: List[Dict[str, Any]],
        mutual_info: Dict[str, Dict[str, float]],
        causality: List[Dict[str, Any]]
    ) -> nx.Graph:
        """Build correlation graph."""
        graph = nx.Graph()
        
        # Add correlations
        for corr in significant:
            graph.add_edge(
                corr["metric1"],
                corr["metric2"],
                weight=abs(corr["correlation"]),
                type="correlation"
            )
        
        # Add mutual information
        for m1, info in mutual_info.items():
            for m2, mi in info.items():
                if graph.has_edge(m1, m2):
                    graph[m1][m2]["mutual_info"] = mi
                else:
                    graph.add_edge(m1, m2, weight=mi, type="mutual_info")
        
        # Add causal relationships to directed graph
        for rel in causality:
            self.causal_graph.add_edge(
                rel["cause"],
                rel["effect"],
                lag=rel["lag"],
                correlation=rel["correlation"]
            )
        
        return graph
    
    def _find_correlation_clusters(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Find clusters of correlated metrics."""
        # Find communities using Louvain method
        try:
            import community
            partition = community.best_partition(graph)
            
            # Group metrics by cluster
            clusters = defaultdict(list)
            for metric, cluster_id in partition.items():
                clusters[cluster_id].append(metric)
            
            # Analyze clusters
            cluster_analysis = []
            for cluster_id, metrics in clusters.items():
                if len(metrics) > 1:
                    cluster_analysis.append({
                        "id": cluster_id,
                        "metrics": metrics,
                        "size": len(metrics),
                        "density": nx.density(graph.subgraph(metrics)),
                        "central_metric": max(
                            metrics,
                            key=lambda m: graph.degree(m)
                        )
                    })
            
            return sorted(
                cluster_analysis,
                key=lambda x: x["size"],
                reverse=True
            )
            
        except ImportError:
            logger.warning("python-louvain package not found, skipping clustering")
            return []
    
    def _create_correlation_summary(
        self,
        correlation_matrix: pd.DataFrame,
        significant: List[Dict[str, Any]],
        mutual_info: Dict[str, Dict[str, float]],
        causality: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create summary of correlation analysis."""
        return {
            "num_metrics": len(correlation_matrix),
            "num_significant": len(significant),
            "avg_correlation": float(
                np.mean([abs(c["correlation"]) for c in significant])
            ) if significant else 0,
            "num_causal": len(causality),
            "strongest_correlation": significant[0] if significant else None,
            "strongest_causality": causality[0] if causality else None,
            "most_connected": self._find_most_connected_metrics(
                correlation_matrix,
                self.config.correlation_threshold
            ),
            "isolated_metrics": self._find_isolated_metrics(correlation_matrix)
        }
    
    def _find_most_connected_metrics(
        self,
        correlation_matrix: pd.DataFrame,
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find metrics with most significant correlations."""
        connections = []
        
        for metric in correlation_matrix.index:
            corrs = correlation_matrix[metric]
            significant = corrs[abs(corrs) >= threshold]
            
            if len(significant) > 1:  # Exclude self-correlation
                connections.append({
                    "metric": metric,
                    "num_connections": len(significant) - 1,
                    "avg_correlation": float(abs(significant).mean()),
                    "connected_to": list(
                        significant.index[significant.index != metric]
                    )
                })
        
        return sorted(
            connections,
            key=lambda x: x["num_connections"],
            reverse=True
        )[:5]
    
    def _find_isolated_metrics(
        self,
        correlation_matrix: pd.DataFrame
    ) -> List[str]:
        """Find metrics with no significant correlations."""
        isolated = []
        
        for metric in correlation_matrix.index:
            corrs = correlation_matrix[metric]
            if not any(
                abs(c) >= self.config.correlation_threshold
                for c in corrs[corrs.index != metric]
            ):
                isolated.append(metric)
        
        return isolated

def analyze_metric_correlations(
    thresholds: AdaptiveThresholds,
    window: Optional[timedelta] = None
) -> Dict[str, Any]:
    """Analyze correlations between metrics in thresholds."""
    config = CorrelationConfig()
    if window:
        config.window_size = window
    
    correlation = MetricCorrelation(config)
    
    # Convert threshold history to pd.Series
    metrics = {}
    for key in thresholds.history:
        for metric_type in thresholds.history[key]:
            name = f"{key}:{metric_type}"
            values = thresholds.history[key][metric_type]
            
            if len(values) >= config.min_samples:
                # Create timestamps
                end_time = datetime.now()
                timestamps = [
                    end_time - timedelta(minutes=i)
                    for i in range(len(values)-1, -1, -1)
                ]
                
                metrics[name] = pd.Series(
                    values,
                    index=pd.DatetimeIndex(timestamps)
                )
    
    return correlation.analyze_correlations(metrics)

if __name__ == "__main__":
    # Example usage
    from .adaptive_thresholds import create_adaptive_thresholds
    
    thresholds = create_adaptive_thresholds()
    
    # Add sample data for multiple metrics
    for hour in range(24 * 7):
        # CPU metric with daily pattern
        cpu_value = 50 + 10 * np.sin(hour * np.pi / 12)
        cpu_value += np.random.normal(0, 2)
        thresholds.add_value(
            "cpu",
            "percent",
            cpu_value,
            datetime.now() - timedelta(hours=24*7-hour)
        )
        
        # Memory metric correlated with CPU
        mem_value = cpu_value * 0.8 + 20 + np.random.normal(0, 2)
        thresholds.add_value(
            "memory",
            "percent",
            mem_value,
            datetime.now() - timedelta(hours=24*7-hour)
        )
        
        # Disk I/O metric with weak correlation
        io_value = 100 + 5 * np.sin(hour * np.pi / 8)
        io_value += 0.2 * cpu_value + np.random.normal(0, 5)
        thresholds.add_value(
            "disk",
            "iops",
            io_value,
            datetime.now() - timedelta(hours=24*7-hour)
        )
    
    # Analyze correlations
    analysis = analyze_metric_correlations(thresholds)
    print(json.dumps(analysis["summary"], indent=2))
