"""Alert correlation and pattern analysis."""

import numpy as np
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Optional, Any
import json
from pathlib import Path
import logging
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

from .prediction_alerts import PredictionAlert
from .alerts import AlertSeverity

logger = logging.getLogger(__name__)

@dataclass
class AlertPattern:
    """Pattern in alert occurrences."""
    alerts: List[str]  # Alert IDs in pattern
    confidence: float
    support: float
    time_window: timedelta
    root_cause: Optional[str]
    impact_score: float

@dataclass
class AlertCorrelation:
    """Correlation between alerts."""
    source_alert: str
    target_alert: str
    correlation_type: str  # "causal", "temporal", "metric"
    strength: float
    time_lag: Optional[timedelta]
    description: str

class AlertCorrelationAnalyzer:
    """Analyze correlations and patterns in alerts."""
    
    def __init__(
        self,
        alert_dir: Path,
        output_dir: Optional[Path] = None,
        min_correlation: float = 0.7,
        max_time_window: timedelta = timedelta(hours=24)
    ):
        self.alert_dir = alert_dir
        self.output_dir = output_dir or Path("alert_analysis")
        self.output_dir.mkdir(exist_ok=True)
        self.min_correlation = min_correlation
        self.max_time_window = max_time_window
        
        self.alert_graph = nx.DiGraph()
        self._load_alerts()
    
    def _load_alerts(self):
        """Load alerts from disk."""
        self.alerts: Dict[str, PredictionAlert] = {}
        self.alert_history: Dict[str, List[PredictionAlert]] = {}
        
        # Load active alerts
        for alert_file in self.alert_dir.glob("*.json"):
            try:
                with alert_file.open() as f:
                    data = json.load(f)
                    alert = PredictionAlert(**data)
                    self.alerts[alert.id] = alert
                    
                    # Group by metric
                    if alert.metric not in self.alert_history:
                        self.alert_history[alert.metric] = []
                    self.alert_history[alert.metric].append(alert)
            except Exception as e:
                logger.error(f"Error loading alert {alert_file}: {e}")
        
        # Load resolved alerts
        resolved_dir = self.alert_dir / "resolved"
        if resolved_dir.exists():
            for alert_file in resolved_dir.glob("*.json"):
                try:
                    with alert_file.open() as f:
                        data = json.load(f)
                        if "resolved_at" in data:
                            alert = PredictionAlert(**{
                                k: v for k, v in data.items()
                                if k != "resolved_at"
                            })
                            if alert.metric not in self.alert_history:
                                self.alert_history[alert.metric] = []
                            self.alert_history[alert.metric].append(alert)
                except Exception as e:
                    logger.error(f"Error loading resolved alert {alert_file}: {e}")
    
    def find_temporal_correlations(self) -> List[AlertCorrelation]:
        """Find temporal correlations between alerts."""
        correlations = []
        
        # Convert alerts to time series
        alert_series = {}
        for metric, alerts in self.alert_history.items():
            # Create time series with 1-minute intervals
            timestamps = pd.date_range(
                min(a.timestamp for a in alerts),
                max(a.timestamp for a in alerts),
                freq="1min"
            )
            
            series = pd.Series(
                0,
                index=timestamps,
                dtype=float
            )
            
            # Mark alert occurrences
            for alert in alerts:
                series[alert.timestamp] = 1.0
            
            alert_series[metric] = series
        
        # Find correlations
        for m1 in alert_series:
            for m2 in alert_series:
                if m1 >= m2:  # Avoid duplicates
                    continue
                
                # Calculate cross-correlation
                xcorr = pd.Series(alert_series[m1]).corr(
                    pd.Series(alert_series[m2])
                )
                
                if abs(xcorr) >= self.min_correlation:
                    # Find time lag
                    lag = pd.Series(alert_series[m1]).rolling(
                        window=len(alert_series[m2]),
                        center=True
                    ).corr(pd.Series(alert_series[m2])).idxmax()
                    
                    correlations.append(AlertCorrelation(
                        source_alert=m1,
                        target_alert=m2,
                        correlation_type="temporal",
                        strength=xcorr,
                        time_lag=timedelta(minutes=int(lag)) if lag else None,
                        description=(
                            f"Temporal correlation of {xcorr:.2f} "
                            f"with lag of {lag} minutes"
                        )
                    ))
        
        return correlations
    
    def find_metric_correlations(self) -> List[AlertCorrelation]:
        """Find correlations in metric values."""
        correlations = []
        
        # Group alerts by metric
        metric_values = {}
        for metric, alerts in self.alert_history.items():
            metric_values[metric] = [
                (a.timestamp, a.current_value, a.predicted_value)
                for a in alerts
            ]
        
        # Find correlations between metrics
        for m1 in metric_values:
            for m2 in metric_values:
                if m1 >= m2:
                    continue
                
                # Get overlapping time windows
                times1 = set(t for t, _, _ in metric_values[m1])
                times2 = set(t for t, _, _ in metric_values[m2])
                common_times = times1 & times2
                
                if len(common_times) < 2:
                    continue
                
                # Get values for common times
                values1 = [
                    v for t, v, _ in metric_values[m1]
                    if t in common_times
                ]
                values2 = [
                    v for t, v, _ in metric_values[m2]
                    if t in common_times
                ]
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(values1, values2)
                
                if abs(corr) >= self.min_correlation:
                    correlations.append(AlertCorrelation(
                        source_alert=m1,
                        target_alert=m2,
                        correlation_type="metric",
                        strength=corr,
                        time_lag=None,
                        description=(
                            f"Metric correlation of {corr:.2f} "
                            f"(p-value: {p_value:.3f})"
                        )
                    ))
        
        return correlations
    
    def find_alert_patterns(self) -> List[AlertPattern]:
        """Find recurring patterns in alert sequences."""
        patterns = []
        
        # Convert alerts to sequences
        alert_sequences = []
        current_sequence = []
        last_time = None
        
        for metric, alerts in self.alert_history.items():
            sorted_alerts = sorted(alerts, key=lambda a: a.timestamp)
            
            for alert in sorted_alerts:
                if (
                    not last_time or
                    alert.timestamp - last_time <= self.max_time_window
                ):
                    current_sequence.append(alert.id)
                else:
                    if len(current_sequence) > 1:
                        alert_sequences.append(current_sequence)
                    current_sequence = [alert.id]
                
                last_time = alert.timestamp
        
        if len(current_sequence) > 1:
            alert_sequences.append(current_sequence)
        
        # Find frequent subsequences
        if alert_sequences:
            # Convert to format for pattern mining
            transactions = [set(seq) for seq in alert_sequences]
            
            # Find frequent itemsets
            min_support = 0.3  # At least 30% occurrence
            frequent_sets = self._find_frequent_itemsets(
                transactions,
                min_support
            )
            
            # Convert to patterns
            for itemset, support in frequent_sets:
                if len(itemset) > 1:
                    # Calculate confidence
                    conf = support / len(alert_sequences)
                    
                    # Calculate impact score
                    impact = sum(
                        self.alerts[alert_id].severity.value
                        for alert_id in itemset
                    ) / len(itemset)
                    
                    patterns.append(AlertPattern(
                        alerts=list(itemset),
                        confidence=conf,
                        support=support,
                        time_window=self.max_time_window,
                        root_cause=self._find_root_cause(itemset),
                        impact_score=impact
                    ))
        
        return sorted(
            patterns,
            key=lambda p: (p.support, p.confidence),
            reverse=True
        )
    
    def _find_frequent_itemsets(
        self,
        transactions: List[Set[str]],
        min_support: float
    ) -> List[Tuple[Set[str], float]]:
        """Find frequent itemsets using Apriori algorithm."""
        # Get all unique items
        items = set().union(*transactions)
        
        # Initial 1-itemsets
        itemsets = [
            ({item}, sum(1 for t in transactions if item in t))
            for item in items
        ]
        
        # Filter by support
        min_count = min_support * len(transactions)
        itemsets = [
            (s, c) for s, c in itemsets
            if c >= min_count
        ]
        
        results = []
        k = 1
        
        while itemsets:
            results.extend(itemsets)
            k += 1
            
            # Generate candidates
            candidates = []
            for i in range(len(itemsets)):
                for j in range(i + 1, len(itemsets)):
                    s1, _ = itemsets[i]
                    s2, _ = itemsets[j]
                    new_set = s1 | s2
                    if len(new_set) == k:
                        count = sum(
                            1 for t in transactions
                            if new_set.issubset(t)
                        )
                        if count >= min_count:
                            candidates.append((new_set, count))
            
            itemsets = candidates
        
        return results
    
    def _find_root_cause(self, alert_set: Set[str]) -> Optional[str]:
        """Attempt to identify root cause in alert pattern."""
        # Find alert with earliest timestamp
        earliest = min(
            (self.alerts[aid] for aid in alert_set),
            key=lambda a: a.timestamp
        )
        
        # Check if it has high severity and contributing factors
        if (
            earliest.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] and
            earliest.contributing_factors
        ):
            return (
                f"{earliest.metric} ({earliest.severity.value}) - "
                f"factors: {', '.join(f[0] for f in earliest.contributing_factors)}"
            )
        
        return None
    
    def build_correlation_graph(self):
        """Build correlation graph from alerts."""
        # Add nodes for each alert
        for alert_id, alert in self.alerts.items():
            self.alert_graph.add_node(
                alert_id,
                metric=alert.metric,
                severity=alert.severity.value,
                timestamp=alert.timestamp.isoformat()
            )
        
        # Add edges for correlations
        correlations = (
            self.find_temporal_correlations() +
            self.find_metric_correlations()
        )
        
        for corr in correlations:
            self.alert_graph.add_edge(
                corr.source_alert,
                corr.target_alert,
                correlation_type=corr.correlation_type,
                strength=corr.strength,
                time_lag=str(corr.time_lag) if corr.time_lag else None
            )
    
    def visualize_correlations(self):
        """Create correlation visualization."""
        if not self.alert_graph:
            self.build_correlation_graph()
        
        # Create node positions using spring layout
        pos = nx.spring_layout(self.alert_graph)
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in self.alert_graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            alert = self.alerts[node]
            node_text.append(
                f"Alert: {alert.metric}<br>"
                f"Severity: {alert.severity.value}<br>"
                f"Time: {alert.timestamp}"
            )
            
            # Color by severity
            severity_colors = {
                "info": "#36a64f",
                "warning": "#ffcc00",
                "error": "#ff0000",
                "critical": "#7b0000"
            }
            node_color.append(severity_colors[alert.severity.value])
        
        nodes = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(
                size=15,
                color=node_color,
                line=dict(width=2)
            ),
            text=node_text,
            hoverinfo="text"
        )
        
        # Create edge traces
        edge_traces = []
        
        for edge in self.alert_graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_trace = go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(
                    width=2,
                    color=("#ff0000" if edge[2]["correlation_type"] == "temporal"
                           else "#0000ff")
                ),
                hoverinfo="text",
                text=(
                    f"Type: {edge[2]['correlation_type']}<br>"
                    f"Strength: {edge[2]['strength']:.2f}"
                )
            )
            edge_traces.append(edge_trace)
        
        # Create figure
        fig = go.Figure(
            data=[nodes, *edge_traces],
            layout=go.Layout(
                title="Alert Correlation Graph",
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        
        # Save visualization
        fig.write_html(str(self.output_dir / "correlation_graph.html"))
    
    def generate_correlation_report(self):
        """Generate correlation analysis report."""
        # Find patterns and correlations
        patterns = self.find_alert_patterns()
        temporal_correlations = self.find_temporal_correlations()
        metric_correlations = self.find_metric_correlations()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_alerts": len(self.alerts),
                "total_patterns": len(patterns),
                "total_correlations": len(temporal_correlations) + len(metric_correlations)
            },
            "patterns": [
                {
                    "alerts": [
                        {
                            "id": aid,
                            "metric": self.alerts[aid].metric,
                            "severity": self.alerts[aid].severity.value
                        }
                        for aid in pattern.alerts
                    ],
                    "confidence": pattern.confidence,
                    "support": pattern.support,
                    "time_window": str(pattern.time_window),
                    "root_cause": pattern.root_cause,
                    "impact_score": pattern.impact_score
                }
                for pattern in patterns
            ],
            "correlations": {
                "temporal": [
                    {
                        "source": c.source_alert,
                        "target": c.target_alert,
                        "strength": c.strength,
                        "time_lag": str(c.time_lag) if c.time_lag else None,
                        "description": c.description
                    }
                    for c in temporal_correlations
                ],
                "metric": [
                    {
                        "source": c.source_alert,
                        "target": c.target_alert,
                        "strength": c.strength,
                        "description": c.description
                    }
                    for c in metric_correlations
                ]
            }
        }
        
        # Save report
        with open(self.output_dir / "correlation_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report

def analyze_alert_correlations():
    """Run correlation analysis."""
    analyzer = AlertCorrelationAnalyzer(Path("prediction_alerts"))
    
    # Generate visualizations
    analyzer.visualize_correlations()
    
    # Generate report
    report = analyzer.generate_correlation_report()
    
    # Log findings
    logger.info("Alert Correlation Analysis Results:")
    logger.info(f"Total alerts analyzed: {report['summary']['total_alerts']}")
    logger.info(f"Found {report['summary']['total_patterns']} patterns")
    logger.info(f"Found {report['summary']['total_correlations']} correlations")
    
    # Log significant patterns
    for pattern in report["patterns"]:
        if pattern["impact_score"] > 0.7:
            logger.warning(
                f"High-impact pattern detected:\n"
                f"Alerts: {[a['metric'] for a in pattern['alerts']]}\n"
                f"Impact score: {pattern['impact_score']:.2f}\n"
                f"Root cause: {pattern['root_cause'] or 'Unknown'}"
            )

if __name__ == "__main__":
    analyze_alert_correlations()
