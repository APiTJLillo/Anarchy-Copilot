"""Load testing for cost-benefit analysis."""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any

import pytest
import numpy as np
from unittest.mock import Mock

from proxy.analysis.monitoring.costbenefit_analysis import (
    CostBenefitAnalyzer,
    CostConfig,
    VisualizationConfig,
    PlotControls
)

class LoadTestHarness:
    """Load test harness for cost-benefit analysis."""

    def __init__(self, num_clients: int, test_duration: int):
        self.num_clients = num_clients
        self.test_duration = test_duration
        self.request_times: List[float] = []
        self.error_count = 0
        self.active_clients = 0
        self.max_concurrent = 0

    async def simulate_client(
        self,
        analyzer: CostBenefitAnalyzer,
        client_id: int,
        scenario_pool: List[str]
    ):
        """Simulate a single client making requests."""
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < self.test_duration:
            try:
                self.active_clients += 1
                self.max_concurrent = max(self.max_concurrent, self.active_clients)
                
                request_start = time.time()
                
                # Randomly choose analysis type
                if random.random() < 0.7:  # 70% analysis requests
                    scenario = random.choice(scenario_pool)
                    await analyzer.analyze_costs_benefits(scenario)
                else:  # 30% visualization requests
                    await analyzer.create_costbenefit_plots()
                
                self.request_times.append(time.time() - request_start)
                request_count += 1
                
                # Random think time between requests (100-500ms)
                await asyncio.sleep(random.uniform(0.1, 0.5))
                
            except Exception as e:
                self.error_count += 1
                print(f"Client {client_id} error: {e}")
            
            finally:
                self.active_clients -= 1
        
        return request_count

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate test statistics."""
        if not self.request_times:
            return {}
        
        times = np.array(self.request_times)
        return {
            "total_requests": len(times),
            "avg_response_time": np.mean(times),
            "p50_response_time": np.percentile(times, 50),
            "p95_response_time": np.percentile(times, 95),
            "p99_response_time": np.percentile(times, 99),
            "max_response_time": np.max(times),
            "min_response_time": np.min(times),
            "error_rate": self.error_count / len(times) if len(times) > 0 else 0,
            "max_concurrent_clients": self.max_concurrent,
            "requests_per_second": len(times) / self.test_duration
        }

@pytest.fixture
def load_test_analyzer():
    """Create analyzer for load testing."""
    from tests.proxy.analysis.monitoring.test_costbenefit_performance import generate_test_data
    
    analyzer = Mock()
    analyzer.results = generate_test_data(num_components=20, num_scenarios=5)
    
    config = CostConfig(
        enabled=True,
        visualization=VisualizationConfig(
            interactive=True,
            controls=PlotControls(
                enable_range_selector=True,
                enable_compare_mode=True
            )
        )
    )
    
    return CostBenefitAnalyzer(analyzer, config)

@pytest.mark.load
@pytest.mark.asyncio
@pytest.mark.parametrize("num_clients", [5, 10, 20])
async def test_concurrent_client_load(load_test_analyzer, num_clients):
    """Test system under concurrent client load."""
    test_duration = 30  # 30 seconds
    harness = LoadTestHarness(num_clients, test_duration)
    
    # Get available scenarios
    scenario_pool = list(load_test_analyzer.intervention_analyzer.results.keys())
    
    # Create client tasks
    tasks = [
        harness.simulate_client(load_test_analyzer, i, scenario_pool)
        for i in range(num_clients)
    ]
    
    # Run load test
    total_requests = sum(await asyncio.gather(*tasks))
    stats = harness.get_statistics()
    
    # Load test assertions
    assert stats["error_rate"] < 0.01  # Less than 1% errors
    assert stats["p95_response_time"] < 1.0  # 95% requests under 1 second
    assert stats["max_concurrent_clients"] <= num_clients
    assert stats["requests_per_second"] >= num_clients * 1.0  # At least 1 RPS per client

@pytest.mark.load
@pytest.mark.asyncio
async def test_sustained_load(load_test_analyzer):
    """Test system under sustained load over time."""
    test_duration = 60  # 60 seconds
    num_clients = 5
    harness = LoadTestHarness(num_clients, test_duration)
    
    # Track metrics over time
    response_times = []
    throughput = []
    measurement_interval = 5  # 5 second intervals
    
    async def monitor_metrics():
        start_time = time.time()
        last_requests = 0
        
        while time.time() - start_time < test_duration:
            await asyncio.sleep(measurement_interval)
            current_requests = len(harness.request_times)
            interval_throughput = (current_requests - last_requests) / measurement_interval
            throughput.append(interval_throughput)
            last_requests = current_requests
    
    # Start monitoring task
    monitor_task = asyncio.create_task(monitor_metrics())
    
    # Run client load
    scenario_pool = list(load_test_analyzer.intervention_analyzer.results.keys())
    client_tasks = [
        harness.simulate_client(load_test_analyzer, i, scenario_pool)
        for i in range(num_clients)
    ]
    
    await asyncio.gather(*client_tasks)
    await monitor_task
    
    # Analyze throughput stability
    throughput_array = np.array(throughput)
    throughput_std = np.std(throughput_array)
    throughput_mean = np.mean(throughput_array)
    
    # Stability assertions
    assert throughput_std / throughput_mean < 0.3  # Less than 30% variation
    assert min(throughput) > 0.5 * max(throughput)  # No major throughput drops

@pytest.mark.load
@pytest.mark.asyncio
async def test_recovery_from_overload(load_test_analyzer):
    """Test system recovery after overload conditions."""
    harness = LoadTestHarness(num_clients=20, test_duration=45)
    
    # Create phases for normal -> overload -> recovery
    phases = [
        (5, 5),   # 5 clients for 5 seconds
        (20, 15), # Spike to 20 clients for 15 seconds
        (5, 25)   # Back to 5 clients for 25 seconds
    ]
    
    scenario_pool = list(load_test_analyzer.intervention_analyzer.results.keys())
    current_tasks = []
    start_time = time.time()
    
    for num_clients, duration in phases:
        # Start new client tasks
        new_tasks = [
            harness.simulate_client(load_test_analyzer, i, scenario_pool)
            for i in range(num_clients - len(current_tasks))
        ]
        current_tasks.extend(new_tasks)
        
        # Wait for phase duration
        await asyncio.sleep(duration)
        
        # Stop excess tasks for next phase
        while len(current_tasks) > num_clients:
            task = current_tasks.pop()
            task.cancel()
    
    stats = harness.get_statistics()
    
    # Recovery assertions
    assert stats["error_rate"] < 0.05  # Less than 5% errors during entire test
    assert stats["p99_response_time"] < 2.0  # 99% requests under 2 seconds
    
    # Verify throughput recovered after overload
    final_requests = len(harness.request_times)
    recovery_rate = final_requests / (time.time() - start_time)
    assert recovery_rate >= 2.5  # At least 2.5 RPS after recovery

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "load"])
