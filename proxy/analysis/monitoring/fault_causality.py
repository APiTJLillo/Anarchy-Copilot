"""Fault causality and root cause analysis."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import networkx as nx
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from scipy.stats import chi2_contingency, entropy
from collections import defaultdict
import json
from pathlib import Path

from .fault_trends import TrendInfo, AnomalyInfo
from .test_fault_correlation import FaultEvent, FaultCorrelationAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class CausalLink:
    """Causal relationship between faults."""
    source: str
    target: str
    strength: float
    time_lag: timedelta
    probability: float
    confidence: float
    evidence: List[str]

@dataclass
class RootCause:
    """Identified root cause of fault pattern."""
    fault_type: str
    probability: float
    impact_metrics: Dict[str, float]
    cascading_effects: List[str]
    common_patterns: List[List[str]]
    mitigation_history: List[str]

class FaultCausalityAnalyzer:
    """Analyze causal relationships between faults."""
    
    def __init__(
        self,
        analyzer: FaultCorrelationAnalyzer,
        output_dir: Optional[Path] = None,
        min_confidence: float = 0.7,
        max_lag: timedelta = timedelta(minutes=30)
    ):
        self.analyzer = analyzer
        self.output_dir = output_dir or Path("causality_analysis")
        self.output_dir.mkdir(exist_ok=True)
        self.min_confidence = min_confidence
        self.max_lag = max_lag
        
        self.causal_graph = nx.DiGraph()
        self.root_causes: Dict[str, RootCause] = {}
        self._build_causal_graph()
    
    def _build_causal_graph(self):
        """Build causal graph from fault history."""
        if not self.analyzer.fault_history:
            return
        
        # Sort events by timestamp
        events = sorted(
            self.analyzer.fault_history,
            key=lambda e: e.timestamp
        )
        
        # Build temporal relationships
        for i, event in enumerate(events[:-1]):
            for j in range(i + 1, len(events)):
                target_event = events[j]
                lag = target_event.timestamp - event.timestamp
                
                if lag > self.max_lag:
                    break
                
                # Calculate causal strength
                strength = self._calculate_causal_strength(
                    event,
                    target_event
                )
                
                if strength >= self.min_confidence:
                    self.causal_graph.add_edge(
                        event.fault_type,
                        target_event.fault_type,
                        strength=strength,
                        lag=lag,
                        count=1
                    )
    
    def _calculate_causal_strength(
        self,
        source: FaultEvent,
        target: FaultEvent
    ) -> float:
        """Calculate causal strength between events."""
        # Feature-based similarity
        metric_similarity = self._calculate_metric_similarity(
            source.impact_metrics,
            target.impact_metrics
        )
        
        # Pattern similarity
        pattern_similarity = len(
            set(source.related_faults) &
            set(target.related_faults)
        ) / max(
            len(source.related_faults),
            len(target.related_faults)
        ) if source.related_faults and target.related_faults else 0
        
        # Temporal proximity
        time_diff = (target.timestamp - source.timestamp).total_seconds()
        temporal_factor = max(0, 1 - time_diff / self.max_lag.total_seconds())
        
        # Cascade relationship
        cascade_factor = 1.0 if target.cascade_depth > source.cascade_depth else 0.5
        
        # Combine factors
        return np.mean([
            metric_similarity,
            pattern_similarity,
            temporal_factor,
            cascade_factor
        ])
    
    def _calculate_metric_similarity(
        self,
        metrics1: Dict[str, float],
        metrics2: Dict[str, float]
    ) -> float:
        """Calculate similarity between metric sets."""
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        if not common_metrics:
            return 0.0
        
        similarities = []
        for metric in common_metrics:
            val1 = metrics1[metric]
            val2 = metrics2[metric]
            max_val = max(abs(val1), abs(val2))
            if max_val > 0:
                similarities.append(1 - abs(val1 - val2) / max_val)
            else:
                similarities.append(1.0)
        
        return np.mean(similarities)
    
    def find_causal_links(self) -> List[CausalLink]:
        """Find causal links between faults."""
        links = []
        
        for source, target, data in self.causal_graph.edges(data=True):
            evidence = self._collect_evidence(source, target)
            confidence = self._calculate_confidence(source, target)
            
            if confidence >= self.min_confidence:
                links.append(CausalLink(
                    source=source,
                    target=target,
                    strength=data["strength"],
                    time_lag=data["lag"],
                    probability=self._calculate_probability(source, target),
                    confidence=confidence,
                    evidence=evidence
                ))
        
        return sorted(
            links,
            key=lambda l: l.confidence,
            reverse=True
        )
    
    def _collect_evidence(
        self,
        source: str,
        target: str
    ) -> List[str]:
        """Collect evidence supporting causal link."""
        evidence = []
        
        # Check temporal ordering
        source_events = [
            e for e in self.analyzer.fault_history
            if e.fault_type == source
        ]
        target_events = [
            e for e in self.analyzer.fault_history
            if e.fault_type == target
        ]
        
        if not source_events or not target_events:
            return evidence
        
        # Check consistent time lag
        lags = []
        for s_event in source_events:
            for t_event in target_events:
                if t_event.timestamp > s_event.timestamp:
                    lags.append(t_event.timestamp - s_event.timestamp)
        
        if lags:
            mean_lag = sum(lags, timedelta()) / len(lags)
            std_lag = np.std([l.total_seconds() for l in lags])
            
            if std_lag / mean_lag.total_seconds() < 0.5:
                evidence.append(f"Consistent time lag: {mean_lag}")
        
        # Check metric correlations
        common_metrics = set()
        for s_event in source_events:
            for t_event in target_events:
                common_metrics.update(
                    set(s_event.impact_metrics.keys()) &
                    set(t_event.impact_metrics.keys())
                )
        
        for metric in common_metrics:
            correlation = np.corrcoef(
                [e.impact_metrics.get(metric, 0) for e in source_events],
                [e.impact_metrics.get(metric, 0) for e in target_events]
            )[0, 1]
            
            if abs(correlation) > 0.7:
                evidence.append(
                    f"Strong {metric} correlation: {correlation:.2f}"
                )
        
        return evidence
    
    def _calculate_confidence(
        self,
        source: str,
        target: str
    ) -> float:
        """Calculate confidence in causal relationship."""
        if not self.analyzer.fault_history:
            return 0.0
        
        # Count occurrences
        source_count = sum(
            1 for e in self.analyzer.fault_history
            if e.fault_type == source
        )
        target_count = sum(
            1 for e in self.analyzer.fault_history
            if e.fault_type == target
        )
        
        if source_count == 0 or target_count == 0:
            return 0.0
        
        # Create contingency table
        table = np.zeros((2, 2))
        for i in range(len(self.analyzer.fault_history) - 1):
            event = self.analyzer.fault_history[i]
            next_event = self.analyzer.fault_history[i + 1]
            
            if event.fault_type == source:
                if next_event.fault_type == target:
                    table[0, 0] += 1  # Source -> Target
                else:
                    table[0, 1] += 1  # Source -> Other
            else:
                if next_event.fault_type == target:
                    table[1, 0] += 1  # Other -> Target
                else:
                    table[1, 1] += 1  # Other -> Other
        
        # Calculate chi-squared statistic
        chi2, p_value, _, _ = chi2_contingency(table)
        
        # Combine with edge strength
        edge_data = self.causal_graph.get_edge_data(source, target)
        if edge_data:
            strength = edge_data["strength"]
            return np.mean([1 - p_value, strength])
        
        return 1 - p_value
    
    def _calculate_probability(
        self,
        source: str,
        target: str
    ) -> float:
        """Calculate probability of target given source."""
        source_events = [
            e for e in self.analyzer.fault_history
            if e.fault_type == source
        ]
        
        if not source_events:
            return 0.0
        
        # Count how often target follows source within max_lag
        followed_count = 0
        for event in source_events:
            future_events = [
                e for e in self.analyzer.fault_history
                if (
                    e.timestamp > event.timestamp and
                    e.timestamp - event.timestamp <= self.max_lag and
                    e.fault_type == target
                )
            ]
            if future_events:
                followed_count += 1
        
        return followed_count / len(source_events)
    
    def identify_root_causes(self) -> List[RootCause]:
        """Identify root causes of fault patterns."""
        root_causes = []
        
        # Find potential root nodes in causal graph
        in_degrees = dict(self.causal_graph.in_degree())
        root_nodes = [
            node for node, degree in in_degrees.items()
            if degree == 0 and self.causal_graph.out_degree(node) > 0
        ]
        
        for node in root_nodes:
            # Find cascading effects
            descendants = nx.descendants(self.causal_graph, node)
            
            # Collect metrics
            node_events = [
                e for e in self.analyzer.fault_history
                if e.fault_type == node
            ]
            
            if not node_events:
                continue
            
            # Calculate average impact
            impact_metrics = defaultdict(list)
            for event in node_events:
                for metric, value in event.impact_metrics.items():
                    impact_metrics[metric].append(value)
            
            avg_metrics = {
                metric: np.mean(values)
                for metric, values in impact_metrics.items()
            }
            
            # Find common patterns
            patterns = self._find_common_patterns(node)
            
            # Get mitigation history
            mitigations = self._get_mitigation_history(node)
            
            root_causes.append(RootCause(
                fault_type=node,
                probability=self._calculate_root_probability(node),
                impact_metrics=avg_metrics,
                cascading_effects=list(descendants),
                common_patterns=patterns,
                mitigation_history=mitigations
            ))
        
        return sorted(
            root_causes,
            key=lambda rc: rc.probability,
            reverse=True
        )
    
    def _find_common_patterns(
        self,
        fault_type: str
    ) -> List[List[str]]:
        """Find common patterns involving fault type."""
        patterns = []
        
        # Look for sequences starting with fault_type
        for i in range(len(self.analyzer.fault_history)):
            if self.analyzer.fault_history[i].fault_type == fault_type:
                pattern = [fault_type]
                current_time = self.analyzer.fault_history[i].timestamp
                
                # Look for following events within max_lag
                for j in range(i + 1, len(self.analyzer.fault_history)):
                    event = self.analyzer.fault_history[j]
                    if event.timestamp - current_time <= self.max_lag:
                        pattern.append(event.fault_type)
                        current_time = event.timestamp
                    else:
                        break
                
                if len(pattern) > 1:
                    patterns.append(pattern)
        
        # Keep most common patterns
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[tuple(pattern)] += 1
        
        common_patterns = [
            list(pattern)
            for pattern, count in pattern_counts.items()
            if count >= 2  # Appear at least twice
        ]
        
        return sorted(
            common_patterns,
            key=lambda p: pattern_counts[tuple(p)],
            reverse=True
        )
    
    def _get_mitigation_history(
        self,
        fault_type: str
    ) -> List[str]:
        """Get history of successful mitigations."""
        # This would integrate with remediation history
        # For now, return placeholder
        return [
            "Implemented auto-scaling",
            "Updated resource limits",
            "Added monitoring alerts"
        ]
    
    def _calculate_root_probability(
        self,
        fault_type: str
    ) -> float:
        """Calculate probability that fault is a root cause."""
        if fault_type not in self.causal_graph:
            return 0.0
        
        # Consider multiple factors:
        
        # 1. Incoming vs outgoing edges
        out_degree = self.causal_graph.out_degree(fault_type)
        total_faults = len(set(
            e.fault_type for e in self.analyzer.fault_history
        ))
        edge_factor = out_degree / total_faults if total_faults > 0 else 0
        
        # 2. Temporal positioning
        events = [
            e for e in self.analyzer.fault_history
            if e.fault_type == fault_type
        ]
        time_positions = []
        for event in events:
            related_events = [
                e for e in self.analyzer.fault_history
                if (
                    abs((e.timestamp - event.timestamp).total_seconds()) <=
                    self.max_lag.total_seconds()
                )
            ]
            if related_events:
                position = sum(
                    1 for e in related_events
                    if e.timestamp < event.timestamp
                ) / len(related_events)
                time_positions.append(position)
        
        temporal_factor = (
            np.mean(time_positions)
            if time_positions
            else 0.5
        )
        
        # 3. Impact breadth
        impact_metrics = set()
        for event in events:
            impact_metrics.update(event.impact_metrics.keys())
        
        total_metrics = set()
        for event in self.analyzer.fault_history:
            total_metrics.update(event.impact_metrics.keys())
        
        impact_factor = (
            len(impact_metrics) / len(total_metrics)
            if total_metrics
            else 0
        )
        
        # Combine factors
        return np.mean([
            edge_factor,
            1 - temporal_factor,  # Earlier events more likely to be root causes
            impact_factor
        ])
    
    def save_analysis(self):
        """Save causality analysis results."""
        causal_links = self.find_causal_links()
        root_causes = self.identify_root_causes()
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "causal_links": [
                {
                    "source": link.source,
                    "target": link.target,
                    "strength": link.strength,
                    "time_lag": str(link.time_lag),
                    "probability": link.probability,
                    "confidence": link.confidence,
                    "evidence": link.evidence
                }
                for link in causal_links
            ],
            "root_causes": [
                {
                    "fault_type": rc.fault_type,
                    "probability": rc.probability,
                    "impact_metrics": rc.impact_metrics,
                    "cascading_effects": rc.cascading_effects,
                    "common_patterns": rc.common_patterns,
                    "mitigation_history": rc.mitigation_history
                }
                for rc in root_causes
            ]
        }
        
        with open(self.output_dir / "causality_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        return analysis

def analyze_fault_causality(analyzer: FaultCorrelationAnalyzer):
    """Run causality analysis."""
    causal_analyzer = FaultCausalityAnalyzer(analyzer)
    return causal_analyzer.save_analysis()

if __name__ == "__main__":
    # Example usage
    analyzer = FaultCorrelationAnalyzer()
    analysis = analyze_fault_causality(analyzer)
    print(json.dumps(analysis, indent=2))
