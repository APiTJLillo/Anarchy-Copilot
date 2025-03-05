"""Fault injection tests for remediation system."""

import pytest
import asyncio
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
import tempfile
import logging
from unittest.mock import Mock, patch, AsyncMock
from contextlib import contextmanager
import signal
import resource
import psutil
from typing import List, Dict, Any, Generator, Optional

from proxy.analysis.monitoring.auto_remediation import (
    RemediationAction,
    RemediationPlan,
    RemediationKnowledgeBase,
    AutoRemediation,
    suggest_remediation
)
from .test_remediation_stress import generate_test_data, StressTestConfig

logger = logging.getLogger(__name__)

class FaultInjector:
    """Inject various faults for testing."""
    
    @contextmanager
    def limit_memory(self, max_size_mb: int) -> Generator:
        """Temporarily limit process memory."""
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            # Convert MB to bytes
            new_limit = max_size_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
            yield
        finally:
            resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    
    @contextmanager
    def inject_latency(self, seconds: float) -> Generator:
        """Inject artificial latency."""
        original_sleep = asyncio.sleep
        
        async def delayed_sleep(duration: float) -> None:
            await original_sleep(duration + seconds)
        
        try:
            asyncio.sleep = delayed_sleep
            yield
        finally:
            asyncio.sleep = original_sleep
    
    @contextmanager
    def corrupt_kb(self, kb_file: Path) -> Generator:
        """Temporarily corrupt knowledge base file."""
        if not kb_file.exists():
            raise ValueError("KB file does not exist")
        
        # Backup original content
        with kb_file.open("r") as f:
            original_content = f.read()
        
        try:
            # Write corrupt JSON
            with kb_file.open("w") as f:
                f.write("{corrupt-json")
            yield
        finally:
            # Restore original content
            with kb_file.open("w") as f:
                f.write(original_content)
    
    @contextmanager
    def simulate_network_partition(
        self,
        error_rate: float = 0.5
    ) -> Generator:
        """Simulate network partition by failing some requests."""
        original_load = RemediationKnowledgeBase._load_knowledge_base
        
        def failing_load(self):
            if random.random() < error_rate:
                raise ConnectionError("Simulated network partition")
            return original_load(self)
        
        try:
            RemediationKnowledgeBase._load_knowledge_base = failing_load
            yield
        finally:
            RemediationKnowledgeBase._load_knowledge_base = original_load
    
    @contextmanager
    def inject_cpu_pressure(self, target_percent: int = 90) -> Generator:
        """Create CPU pressure."""
        stop_flag = False
        
        def cpu_worker():
            while not stop_flag:
                pass
        
        try:
            num_cpus = psutil.cpu_count()
            workers = []
            for _ in range(num_cpus):
                worker = threading.Thread(target=cpu_worker)
                worker.daemon = True
                worker.start()
                workers.append(worker)
            yield
        finally:
            stop_flag = True
            for worker in workers:
                worker.join(timeout=1)

class TestFaultInjection:
    """Test system behavior under fault conditions."""
    
    @pytest.fixture
    def injector(self) -> FaultInjector:
        """Create fault injector."""
        return FaultInjector()
    
    @pytest.fixture
    def test_kb(self) -> Path:
        """Create test knowledge base."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            json.dump({
                "solutions": {
                    "test_solution": [{
                        "action_type": "scale",
                        "target": "service",
                        "parameters": {},
                        "priority": 1,
                        "estimated_impact": 0.5,
                        "risk_level": "low",
                        "prerequisites": [],
                        "validation_steps": []
                    }]
                },
                "history": []
            }, f)
            return Path(f.name)
    
    async def test_memory_pressure(self, injector, test_kb):
        """Test behavior under memory pressure."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.MEDIUM_LOAD
        )
        
        with injector.limit_memory(50):  # Limit to 50MB
            remediator = AutoRemediation(
                kb=RemediationKnowledgeBase(test_kb)
            )
            
            # Should handle memory pressure gracefully
            plan = remediator.generate_plan(patterns, correlations)
            assert isinstance(plan, RemediationPlan)
            assert len(plan.actions) > 0
    
    async def test_high_latency(self, injector, test_kb):
        """Test behavior with high latency."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        with injector.inject_latency(2.0):  # Add 2 second delay
            remediator = AutoRemediation(
                kb=RemediationKnowledgeBase(test_kb)
            )
            
            start_time = time.time()
            plan = remediator.generate_plan(patterns, correlations)
            duration = time.time() - start_time
            
            assert isinstance(plan, RemediationPlan)
            assert duration >= 2.0  # Verify latency was injected
    
    async def test_corrupted_kb(self, injector, test_kb):
        """Test handling of corrupted knowledge base."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        with injector.corrupt_kb(test_kb):
            # Should handle corruption gracefully
            remediator = AutoRemediation(
                kb=RemediationKnowledgeBase(test_kb)
            )
            
            # Should still generate plan with default/fallback behavior
            plan = remediator.generate_plan(patterns, correlations)
            assert isinstance(plan, RemediationPlan)
    
    async def test_network_partition(self, injector, test_kb):
        """Test behavior during network partition."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        successes = 0
        failures = 0
        
        with injector.simulate_network_partition(error_rate=0.5):
            for _ in range(20):
                try:
                    remediator = AutoRemediation(
                        kb=RemediationKnowledgeBase(test_kb)
                    )
                    plan = remediator.generate_plan(patterns, correlations)
                    assert isinstance(plan, RemediationPlan)
                    successes += 1
                except ConnectionError:
                    failures += 1
        
        # Should see mix of successes and failures
        assert successes > 0
        assert failures > 0
    
    async def test_cpu_pressure(self, injector, test_kb):
        """Test behavior under CPU pressure."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        with injector.inject_cpu_pressure():
            remediator = AutoRemediation(
                kb=RemediationKnowledgeBase(test_kb)
            )
            
            # Should still complete within reasonable time
            start_time = time.time()
            plan = remediator.generate_plan(patterns, correlations)
            duration = time.time() - start_time
            
            assert isinstance(plan, RemediationPlan)
            assert duration < 30  # Should complete within 30 seconds
    
    async def test_multiple_faults(self, injector, test_kb):
        """Test handling multiple concurrent faults."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        # Inject multiple faults
        with (
            injector.limit_memory(50),
            injector.inject_latency(1.0),
            injector.simulate_network_partition(error_rate=0.3)
        ):
            try:
                remediator = AutoRemediation(
                    kb=RemediationKnowledgeBase(test_kb)
                )
                plan = remediator.generate_plan(patterns, correlations)
                assert isinstance(plan, RemediationPlan)
            except Exception as e:
                # Should either complete successfully or fail gracefully
                assert isinstance(e, (MemoryError, ConnectionError))
    
    @pytest.mark.parametrize("fault_sequence", [
        ["memory", "latency", "network"],
        ["network", "cpu", "memory"],
        ["latency", "network", "cpu"]
    ])
    async def test_cascading_faults(
        self,
        injector,
        test_kb,
        fault_sequence: List[str]
    ):
        """Test handling of cascading faults."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(test_kb)
        )
        
        # Apply faults in sequence
        for fault in fault_sequence:
            if fault == "memory":
                with injector.limit_memory(50):
                    plan = remediator.generate_plan(patterns, correlations)
                    assert isinstance(plan, RemediationPlan)
            
            elif fault == "latency":
                with injector.inject_latency(1.0):
                    plan = remediator.generate_plan(patterns, correlations)
                    assert isinstance(plan, RemediationPlan)
            
            elif fault == "network":
                with injector.simulate_network_partition(error_rate=0.3):
                    try:
                        plan = remediator.generate_plan(patterns, correlations)
                        assert isinstance(plan, RemediationPlan)
                    except ConnectionError:
                        pass  # Expected sometimes
            
            elif fault == "cpu":
                with injector.inject_cpu_pressure():
                    plan = remediator.generate_plan(patterns, correlations)
                    assert isinstance(plan, RemediationPlan)
    
    async def test_recovery_behavior(self, injector, test_kb):
        """Test system recovery after faults."""
        patterns, correlations, metrics = generate_test_data(
            StressTestConfig.LIGHT_LOAD
        )
        
        remediator = AutoRemediation(
            kb=RemediationKnowledgeBase(test_kb)
        )
        
        # Inject fault and verify recovery
        with injector.corrupt_kb(test_kb):
            try:
                plan1 = remediator.generate_plan(patterns, correlations)
            except:
                pass
        
        # Should work normally after fault is removed
        plan2 = remediator.generate_plan(patterns, correlations)
        assert isinstance(plan2, RemediationPlan)
        assert len(plan2.actions) > 0

def run_fault_injection_tests():
    """Run all fault injection tests."""
    pytest.main([
        __file__,
        "-v",
        "--log-cli-level=INFO",
        "--durations=0"
    ])

if __name__ == "__main__":
    run_fault_injection_tests()
