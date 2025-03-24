"""Fault correlation and interaction testing."""

import pytest
import asyncio
from datetime import datetime, timedelta
import random
import numpy as np
import networkx as nx
from pathlib import Path
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple
import pandas as pd
from scipy import stats

from proxy.analysis.monitoring.auto_remediation import (
    RemediationAction,
    RemediationPlan,
    RemediationKnowledgeBase,
    AutoRemediation
)
from .test_remediation_fault_injection import FaultInjector
from .test_remediation_stress import generate_test_data, StressTestConfig

logger = logging.getLogger(__name__)

@dataclass
class FaultEvent:
    """Record of a fault occurrence."""
    fault_type: str
    timestamp: datetime
    duration: timedelta
    impact_metrics: Dict[str, float]
    related_faults: List[str]
    recovery_time: Optional[timedelta] = None
    cascade_depth: int = 0

class FaultCorrelationAnalyzer:
    """Analyze relationships between faults."""
    
    def __init__(self, time_window: timedelta = timedelta(minutes=30)):
        self.time_window = time_window
        self.fault_graph = nx.DiGraph()
        self.fault_history: List[FaultEvent] = []
    
    def add_fault_event(self, event: FaultEvent):
        """Add fault event to history."""
        self.fault_history.append(event)
        
        # Add to correlation graph
        self.fault_graph.add_node(
            event.fault_type,
            events=self.fault_graph.nodes.get(
                event.fault_type, {"count": 0}
            )["count"] + 1
        )
        
        # Add edges for related faults
        for related in event.related_faults:
            if related in self.fault_graph:
                # Check temporal relationship
                self.fault_graph.add_edge(
                    event.fault_type,
                    related,
                    weight=self.fault_graph.edges.get(
                        (event.fault_type, related),
                        {"weight": 0}
                    )["weight"] + 1
                )
    
    def find_correlated_faults(
        self,
        min_correlation: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """Find strongly correlated fault pairs."""
        correlations = []
        
        # Convert to time series
        fault_series = self._create_fault_series()
        
        for f1 in fault_series.columns:
            for f2 in fault_series.columns:
                if f1 >= f2:
                    continue
                
                correlation = fault_series[f1].corr(fault_series[f2])
                if abs(correlation) >= min_correlation:
                    correlations.append((f1, f2, correlation))
        
        return sorted(
            correlations,
            key=lambda x: abs(x[2]),
            reverse=True
        )
    
    def detect_cascading_patterns(
        self,
        min_support: float = 0.3
    ) -> List[List[str]]:
        """Detect common cascading fault patterns."""
        # Convert fault history to sequences
        sequences = []
        current_sequence = []
        last_time = None
        
        for event in sorted(
            self.fault_history,
            key=lambda e: e.timestamp
        ):
            if (
                not last_time or
                event.timestamp - last_time <= self.time_window
            ):
                current_sequence.append(event.fault_type)
            else:
                if current_sequence:
                    sequences.append(current_sequence)
                current_sequence = [event.fault_type]
            last_time = event.timestamp
        
        if current_sequence:
            sequences.append(current_sequence)
        
        # Find frequent sequences
        patterns = self._find_frequent_sequences(
            sequences,
            min_support
        )
        
        return patterns
    
    def calculate_fault_impact(
        self,
        metric: str
    ) -> Dict[str, float]:
        """Calculate average impact of each fault type on a metric."""
        impacts = {}
        
        for fault_type in set(e.fault_type for e in self.fault_history):
            # Get all events of this type
            events = [
                e for e in self.fault_history
                if e.fault_type == fault_type
            ]
            
            if not events:
                continue
            
            # Calculate average impact
            impacts[fault_type] = np.mean([
                e.impact_metrics.get(metric, 0)
                for e in events
            ])
        
        return impacts
    
    def _create_fault_series(self) -> pd.DataFrame:
        """Create time series for fault occurrences."""
        if not self.fault_history:
            return pd.DataFrame()
        
        # Create time range
        start_time = min(e.timestamp for e in self.fault_history)
        end_time = max(e.timestamp for e in self.fault_history)
        time_range = pd.date_range(
            start_time,
            end_time,
            freq="1min"
        )
        
        # Create series for each fault type
        fault_types = set(e.fault_type for e in self.fault_history)
        data = {}
        
        for fault_type in fault_types:
            series = pd.Series(0, index=time_range)
            
            # Mark fault occurrences
            for event in self.fault_history:
                if event.fault_type == fault_type:
                    end_time = event.timestamp + event.duration
                    mask = (series.index >= event.timestamp) & (series.index <= end_time)
                    series[mask] = 1
            
            data[fault_type] = series
        
        return pd.DataFrame(data)
    
    def _find_frequent_sequences(
        self,
        sequences: List[List[str]],
        min_support: float
    ) -> List[List[str]]:
        """Find frequent fault sequences."""
        # Count sequence occurrences
        sequence_counts = {}
        
        for sequence in sequences:
            for i in range(len(sequence)):
                for j in range(i + 1, len(sequence) + 1):
                    subseq = tuple(sequence[i:j])
                    sequence_counts[subseq] = sequence_counts.get(subseq, 0) + 1
        
        # Filter by support
        min_count = min_support * len(sequences)
        frequent_sequences = [
            list(seq) for seq, count in sequence_counts.items()
            if count >= min_count
        ]
        
        return sorted(
            frequent_sequences,
            key=len,
            reverse=True
        )

class TestFaultCorrelation:
    """Test fault correlation analysis."""
    
    @pytest.fixture
    def analyzer(self) -> FaultCorrelationAnalyzer:
        """Create fault analyzer."""
        return FaultCorrelationAnalyzer()
    
    @pytest.fixture
    def injector(self) -> FaultInjector:
        """Create fault injector."""
        return FaultInjector()
    
    async def test_basic_correlation(self, analyzer, injector, test_kb):
        """Test basic fault correlation detection."""
        # Inject correlated faults
        for _ in range(10):
            # Memory pressure often leads to network issues
            with injector.limit_memory(50):
                event1 = FaultEvent(
                    fault_type="memory_pressure",
                    timestamp=datetime.now(),
                    duration=timedelta(seconds=30),
                    impact_metrics={"response_time": 2.0},
                    related_faults=[]
                )
                analyzer.add_fault_event(event1)
                
                # Network issues follow
                await asyncio.sleep(1)
                with injector.simulate_network_partition():
                    event2 = FaultEvent(
                        fault_type="network_partition",
                        timestamp=datetime.now(),
                        duration=timedelta(seconds=30),
                        impact_metrics={"response_time": 3.0},
                        related_faults=["memory_pressure"]
                    )
                    analyzer.add_fault_event(event2)
        
        correlations = analyzer.find_correlated_faults()
        assert len(correlations) > 0
        assert correlations[0][0] == "memory_pressure"
        assert correlations[0][1] == "network_partition"
        assert correlations[0][2] > 0.7
    
    async def test_cascading_detection(self, analyzer, injector, test_kb):
        """Test detection of cascading fault patterns."""
        # Create cascading fault pattern
        for _ in range(5):
            # Memory -> Network -> CPU cascade
            with injector.limit_memory(50):
                analyzer.add_fault_event(FaultEvent(
                    fault_type="memory_pressure",
                    timestamp=datetime.now(),
                    duration=timedelta(seconds=10),
                    impact_metrics={"response_time": 2.0},
                    related_faults=[],
                    cascade_depth=0
                ))
                
                await asyncio.sleep(0.5)
                with injector.simulate_network_partition():
                    analyzer.add_fault_event(FaultEvent(
                        fault_type="network_partition",
                        timestamp=datetime.now(),
                        duration=timedelta(seconds=10),
                        impact_metrics={"response_time": 3.0},
                        related_faults=["memory_pressure"],
                        cascade_depth=1
                    ))
                    
                    await asyncio.sleep(0.5)
                    with injector.inject_cpu_pressure():
                        analyzer.add_fault_event(FaultEvent(
                            fault_type="cpu_pressure",
                            timestamp=datetime.now(),
                            duration=timedelta(seconds=10),
                            impact_metrics={"response_time": 4.0},
                            related_faults=["memory_pressure", "network_partition"],
                            cascade_depth=2
                        ))
        
        patterns = analyzer.detect_cascading_patterns()
        assert patterns
        assert ["memory_pressure", "network_partition", "cpu_pressure"] in patterns
    
    def test_impact_analysis(self, analyzer):
        """Test fault impact analysis."""
        # Add faults with different impacts
        analyzer.add_fault_event(FaultEvent(
            fault_type="memory_pressure",
            timestamp=datetime.now(),
            duration=timedelta(seconds=30),
            impact_metrics={
                "response_time": 2.0,
                "error_rate": 0.1
            },
            related_faults=[]
        ))
        
        analyzer.add_fault_event(FaultEvent(
            fault_type="network_partition",
            timestamp=datetime.now(),
            duration=timedelta(seconds=30),
            impact_metrics={
                "response_time": 5.0,
                "error_rate": 0.5
            },
            related_faults=[]
        ))
        
        # Analyze impacts
        response_impacts = analyzer.calculate_fault_impact("response_time")
        error_impacts = analyzer.calculate_fault_impact("error_rate")
        
        assert response_impacts["network_partition"] > response_impacts["memory_pressure"]
        assert error_impacts["network_partition"] > error_impacts["memory_pressure"]
    
    async def test_real_world_scenario(
        self,
        analyzer,
        injector,
        test_kb
    ):
        """Test analysis of complex real-world fault scenario."""
        # Simulate complex fault scenario
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.MEDIUM_LOAD
        )
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(test_kb)
        )
        
        async def inject_faults():
            # Initial memory pressure
            with injector.limit_memory(50):
                analyzer.add_fault_event(FaultEvent(
                    fault_type="memory_pressure",
                    timestamp=datetime.now(),
                    duration=timedelta(seconds=30),
                    impact_metrics={"response_time": 2.0},
                    related_faults=[]
                ))
                
                # Triggers network issues
                await asyncio.sleep(1)
                with injector.simulate_network_partition():
                    analyzer.add_fault_event(FaultEvent(
                        fault_type="network_partition",
                        timestamp=datetime.now(),
                        duration=timedelta(seconds=30),
                        impact_metrics={"response_time": 3.0},
                        related_faults=["memory_pressure"]
                    ))
                    
                    # Which leads to CPU pressure
                    await asyncio.sleep(1)
                    with injector.inject_cpu_pressure():
                        analyzer.add_fault_event(FaultEvent(
                            fault_type="cpu_pressure",
                            timestamp=datetime.now(),
                            duration=timedelta(seconds=30),
                            impact_metrics={"response_time": 4.0},
                            related_faults=["memory_pressure", "network_partition"]
                        ))
                        
                        # Try to generate remediation plan
                        try:
                            plan = remediator.generate_plan(
                                patterns,
                                correlations
                            )
                            assert isinstance(plan, RemediationPlan)
                        except Exception as e:
                            analyzer.add_fault_event(FaultEvent(
                                fault_type="remediation_failure",
                                timestamp=datetime.now(),
                                duration=timedelta(seconds=1),
                                impact_metrics={"error_rate": 1.0},
                                related_faults=[
                                    "memory_pressure",
                                    "network_partition",
                                    "cpu_pressure"
                                ]
                            ))
        
        # Run scenario multiple times
        for _ in range(5):
            await inject_faults()
            await asyncio.sleep(2)
        
        # Analyze results
        correlations = analyzer.find_correlated_faults()
        patterns = analyzer.detect_cascading_patterns()
        impacts = analyzer.calculate_fault_impact("response_time")
        
        # Verify analysis
        assert len(correlations) > 0
        assert len(patterns) > 0
        assert "remediation_failure" in impacts
        
        # Check for expected patterns
        expected_sequence = [
            "memory_pressure",
            "network_partition",
            "cpu_pressure",
            "remediation_failure"
        ]
        matching_patterns = [
            p for p in patterns
            if all(f in p for f in expected_sequence)
        ]
        assert matching_patterns

def run_fault_correlation_tests():
    """Run all fault correlation tests."""
    pytest.main([
        __file__,
        "-v",
        "--log-cli-level=INFO",
        "--durations=0"
    ])

if __name__ == "__main__":
    run_fault_correlation_tests()
