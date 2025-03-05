"""Stress tests for remediation system."""

import pytest
import asyncio
from datetime import datetime, timedelta
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil
import time
from pathlib import Path
import tempfile
import json
import logging
from typing import List, Dict, Any

from proxy.analysis.monitoring.auto_remediation import (
    RemediationAction,
    RemediationPlan,
    RemediationKnowledgeBase,
    AutoRemediation,
    suggest_remediation
)
from proxy.analysis.monitoring.alert_correlation import (
    AlertPattern,
    AlertCorrelation
)
from proxy.analysis.monitoring.alerts import AlertSeverity

logger = logging.getLogger(__name__)

class StressTestConfig:
    """Configuration for stress tests."""
    
    # Load levels
    LIGHT_LOAD = {
        "patterns": 10,
        "correlations": 20,
        "metrics": 100,
        "concurrent_requests": 5
    }
    
    MEDIUM_LOAD = {
        "patterns": 50,
        "correlations": 100,
        "metrics": 500,
        "concurrent_requests": 20
    }
    
    HEAVY_LOAD = {
        "patterns": 200,
        "correlations": 400,
        "metrics": 2000,
        "concurrent_requests": 50
    }
    
    EXTREME_LOAD = {
        "patterns": 1000,
        "correlations": 2000,
        "metrics": 10000,
        "concurrent_requests": 100
    }
    
    # Performance thresholds
    MAX_RESPONSE_TIME = 5.0  # seconds
    MAX_MEMORY_INCREASE = 100 * 1024 * 1024  # 100MB
    MAX_CPU_USAGE = 80  # percentage
    
    # Test durations
    SUSTAINED_TEST_DURATION = 300  # 5 minutes
    BURST_TEST_DURATION = 60  # 1 minute

def generate_test_data(config: Dict[str, int]) -> tuple[List[Any], List[Any], Dict[str, float]]:
    """Generate test data for stress testing."""
    # Generate patterns
    patterns = [
        AlertPattern(
            alerts=[f"alert_{i}" for i in range(random.randint(1, 5))],
            confidence=random.random(),
            support=random.random(),
            time_window=timedelta(minutes=random.randint(5, 60)),
            root_cause=f"Test cause {i}",
            impact_score=random.random()
        )
        for i in range(config["patterns"])
    ]
    
    # Generate correlations
    correlations = [
        AlertCorrelation(
            source_alert=f"alert_{i}",
            target_alert=f"alert_{i+1}",
            correlation_type="temporal" if random.random() > 0.5 else "metric",
            strength=random.random(),
            time_lag=timedelta(minutes=random.randint(1, 30)),
            description=f"Test correlation {i}"
        )
        for i in range(config["correlations"])
    ]
    
    # Generate metrics
    metrics = {
        f"metric_{i}": random.uniform(0, 100)
        for i in range(config["metrics"])
    }
    
    return patterns, correlations, metrics

@pytest.fixture
def stress_kb() -> Path:
    """Create knowledge base for stress testing."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        # Generate large knowledge base
        solutions = {
            f"test_solution_{i}": [
                {
                    "action_type": random.choice(["scale", "restart", "config"]),
                    "target": f"service_{j}",
                    "parameters": {"param": random.randint(1, 100)},
                    "priority": random.randint(1, 5),
                    "estimated_impact": random.random(),
                    "risk_level": random.choice(["low", "medium", "high"]),
                    "prerequisites": [
                        random.choice(["cpu_below_80", "memory_available"])
                        for _ in range(random.randint(1, 3))
                    ],
                    "validation_steps": [
                        random.choice(["check_redundancy", "check_backup"])
                        for _ in range(random.randint(1, 3))
                    ]
                }
                for j in range(random.randint(1, 5))
            ]
            for i in range(100)  # Large number of solutions
        }
        
        json.dump({"solutions": solutions, "history": []}, f)
        return Path(f.name)

class TestRemediationStress:
    """Stress tests for remediation system."""
    
    @pytest.mark.parametrize("load", [
        StressTestConfig.LIGHT_LOAD,
        StressTestConfig.MEDIUM_LOAD,
        StressTestConfig.HEAVY_LOAD
    ])
    async def test_load_handling(self, stress_kb, load):
        """Test handling different load levels."""
        patterns, correlations, metrics = generate_test_data(load)
        
        start_time = time.time()
        
        # Generate remediation plan
        plan = suggest_remediation(
            patterns=patterns,
            correlations=correlations,
            current_metrics=metrics
        )
        
        response_time = time.time() - start_time
        
        assert isinstance(plan, RemediationPlan)
        assert response_time < StressTestConfig.MAX_RESPONSE_TIME
    
    async def test_sustained_load(self, stress_kb):
        """Test sustained load over time."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(stress_kb)
        )
        
        start_time = time.time()
        initial_memory = psutil.Process().memory_info().rss
        response_times = []
        memory_usage = []
        cpu_usage = []
        
        while time.time() - start_time < StressTestConfig.SUSTAINED_TEST_DURATION:
            # Generate test data
            patterns, correlations, metrics = generate_test_data(
                StressTestConfig.MEDIUM_LOAD
            )
            
            # Measure request
            req_start = time.time()
            plan = remediator.generate_plan(patterns, correlations)
            response_times.append(time.time() - req_start)
            
            # Track resource usage
            memory_usage.append(psutil.Process().memory_info().rss)
            cpu_usage.append(psutil.Process().cpu_percent())
            
            await asyncio.sleep(0.1)
        
        # Verify performance
        assert max(response_times) < StressTestConfig.MAX_RESPONSE_TIME
        assert max(memory_usage) - initial_memory < StressTestConfig.MAX_MEMORY_INCREASE
        assert max(cpu_usage) < StressTestConfig.MAX_CPU_USAGE
    
    async def test_concurrent_requests(self, stress_kb):
        """Test handling concurrent requests."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.MEDIUM_LOAD
        )
        
        async def make_request():
            return suggest_remediation(
                patterns=patterns,
                correlations=correlations,
                current_metrics=metrics
            )
        
        # Make concurrent requests
        start_time = time.time()
        plans = await asyncio.gather(*[
            make_request()
            for _ in range(StressTestConfig.MEDIUM_LOAD["concurrent_requests"])
        ])
        total_time = time.time() - start_time
        
        # Verify results
        assert len(plans) == StressTestConfig.MEDIUM_LOAD["concurrent_requests"]
        assert all(isinstance(p, RemediationPlan) for p in plans)
        assert total_time < StressTestConfig.MAX_RESPONSE_TIME * 2  # Allow some overhead
    
    async def test_burst_load(self, stress_kb):
        """Test handling sudden burst of requests."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(stress_kb)
        )
        
        # Generate burst test data
        burst_data = [
            generate_test_data(StressTestConfig.HEAVY_LOAD)
            for _ in range(10)
        ]
        
        # Track metrics
        response_times = []
        errors = 0
        
        async def process_burst(data):
            try:
                start_time = time.time()
                plan = remediator.generate_plan(data[0], data[1])
                response_times.append(time.time() - start_time)
                return plan
            except Exception:
                nonlocal errors
                errors += 1
        
        # Send burst of requests
        start_time = time.time()
        plans = await asyncio.gather(*[
            process_burst(data)
            for data in burst_data
        ])
        burst_time = time.time() - start_time
        
        # Verify handling
        assert errors == 0
        assert len(plans) == len(burst_data)
        assert max(response_times) < StressTestConfig.MAX_RESPONSE_TIME * 2
    
    def test_memory_leak(self, stress_kb):
        """Test for memory leaks during extended operation."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(stress_kb)
        )
        
        initial_memory = psutil.Process().memory_info().rss
        memory_samples = []
        
        # Run multiple cycles
        for _ in range(100):
            patterns, correlations, metrics = generate_test_data(
                StressTestConfig.MEDIUM_LOAD
            )
            
            # Generate plan
            _ = remediator.generate_plan(patterns, correlations)
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Sample memory
            memory_samples.append(psutil.Process().memory_info().rss)
        
        # Calculate memory growth
        memory_growth = [
            samples[i] - samples[i-1]
            for i in range(1, len(memory_samples))
        ]
        
        # Verify no significant memory growth trend
        growth_trend = np.polyfit(
            range(len(memory_growth)),
            memory_growth,
            1
        )[0]
        
        assert growth_trend < 1000  # Less than 1KB per iteration growth
    
    @pytest.mark.parametrize("invalid_data", [
        ([], [], {}),  # Empty data
        (None, None, None),  # None values
        ([1, 2, 3], ["a", "b", "c"], {"x": "y"}),  # Invalid types
    ])
    def test_error_handling(self, stress_kb, invalid_data):
        """Test error handling under stress."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(stress_kb)
        )
        
        patterns, correlations, metrics = invalid_data
        
        # Should handle errors gracefully
        with pytest.raises(Exception) as exc_info:
            _ = remediator.generate_plan(patterns, correlations)
        
        assert str(exc_info.value)  # Should have error message
    
    async def test_recovery_from_errors(self, stress_kb):
        """Test system recovery after errors."""
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(stress_kb)
        )
        
        # Generate valid and invalid requests
        valid_data = generate_test_data(StressTestConfig.LIGHT_LOAD)
        invalid_data = (None, None, None)
        
        for _ in range(10):
            # Send invalid request
            try:
                _ = remediator.generate_plan(*invalid_data)
            except Exception:
                pass
            
            # Verify system can still handle valid requests
            plan = remediator.generate_plan(*valid_data)
            assert isinstance(plan, RemediationPlan)

def run_stress_tests():
    """Run all stress tests."""
    pytest.main([
        __file__,
        "-v",
        "--log-cli-level=INFO",
        "--durations=0"
    ])

if __name__ == "__main__":
    run_stress_tests()
