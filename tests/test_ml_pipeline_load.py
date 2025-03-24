#!/usr/bin/env python3
"""Load profile testing for ML pipeline components."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Generator, List, Dict, Any, Callable
import asyncio
import aiohttp
import threading
import time
import psutil
from dataclasses import dataclass
from contextlib import contextmanager

from scripts.alert_throttling import AlertThrottler, ThrottlingConfig
from scripts.predict_throttling_performance import PerformancePredictor
from scripts.validate_throttling_models import ModelValidator
from scripts.multi_variant_test import MultiVariantTester
from scripts.track_experiments import ExperimentTracker
from scripts.analyze_experiments import ExperimentAnalyzer

@dataclass
class LoadProfile:
    """Definition of a load testing profile."""
    name: str
    duration_seconds: int
    peak_rps: int
    ramp_up_seconds: int
    ramp_down_seconds: int
    pattern: str  # "constant", "sawtooth", "random", "sine"
    bursts: bool
    burst_size: int
    burst_duration: int

@dataclass
class LoadTestResult:
    """Results from a load test."""
    profile: LoadProfile
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    p95_latency: float
    p99_latency: float
    max_memory_mb: float
    avg_cpu_percent: float
    errors: Dict[str, int]

class LoadGenerator:
    """Generate load according to specified profile."""
    
    def __init__(self, profile: LoadProfile):
        self.profile = profile
        self.stats = {
            "requests": 0,
            "successes": 0,
            "failures": 0,
            "latencies": [],
            "errors": {}
        }
        self._start_time = None
        
    def _calculate_current_rate(self, elapsed: float) -> int:
        """Calculate current request rate based on pattern."""
        total_time = self.profile.duration_seconds
        ramp_up = self.profile.ramp_up_seconds
        ramp_down = self.profile.ramp_down_seconds
        peak = self.profile.peak_rps
        
        if self.profile.pattern == "constant":
            if elapsed < ramp_up:
                return int(peak * (elapsed / ramp_up))
            elif elapsed > total_time - ramp_down:
                return int(peak * ((total_time - elapsed) / ramp_down))
            return peak
            
        elif self.profile.pattern == "sawtooth":
            cycle = 60  # 1-minute cycle
            phase = elapsed % cycle
            return int(peak * (phase / cycle))
            
        elif self.profile.pattern == "random":
            base_rate = peak // 2
            variance = peak // 4
            return int(base_rate + np.random.randint(-variance, variance))
            
        elif self.profile.pattern == "sine":
            period = 2 * np.pi * (elapsed / total_time)
            return int(peak * (0.5 + 0.5 * np.sin(period)))
            
        return peak

    async def _make_request(self, session: aiohttp.ClientSession, endpoint: str) -> None:
        """Make single request and record metrics."""
        start_time = time.time()
        try:
            async with session.post(endpoint, json={"test": "data"}) as response:
                latency = time.time() - start_time
                self.stats["latencies"].append(latency)
                
                if response.status == 200:
                    self.stats["successes"] += 1
                else:
                    self.stats["failures"] += 1
                    error = f"HTTP {response.status}"
                    self.stats["errors"][error] = self.stats["errors"].get(error, 0) + 1
                    
        except Exception as e:
            self.stats["failures"] += 1
            error_type = type(e).__name__
            self.stats["errors"][error_type] = self.stats["errors"].get(error_type, 0) + 1

    async def _burst_traffic(self, session: aiohttp.ClientSession, endpoint: str) -> None:
        """Generate burst of traffic."""
        tasks = []
        for _ in range(self.profile.burst_size):
            tasks.append(self._make_request(session, endpoint))
        await asyncio.gather(*tasks)

    async def generate_load(self, endpoint: str) -> None:
        """Generate load according to profile."""
        self._start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            while time.time() - self._start_time < self.profile.duration_seconds:
                elapsed = time.time() - self._start_time
                current_rate = self._calculate_current_rate(elapsed)
                
                # Generate base load
                tasks = []
                for _ in range(current_rate):
                    tasks.append(self._make_request(session, endpoint))
                
                # Add burst if scheduled
                if (self.profile.bursts and 
                    elapsed % self.profile.burst_duration == 0):
                    tasks.append(self._burst_traffic(session, endpoint))
                
                await asyncio.gather(*tasks)
                await asyncio.sleep(1)  # Wait for next second

    def get_results(self) -> LoadTestResult:
        """Calculate test results."""
        latencies = np.array(self.stats["latencies"])
        return LoadTestResult(
            profile=self.profile,
            total_requests=self.stats["requests"],
            successful_requests=self.stats["successes"],
            failed_requests=self.stats["failures"],
            average_latency=np.mean(latencies),
            p95_latency=np.percentile(latencies, 95),
            p99_latency=np.percentile(latencies, 99),
            max_memory_mb=psutil.Process().memory_info().rss / (1024 * 1024),
            avg_cpu_percent=psutil.Process().cpu_percent(),
            errors=self.stats["errors"]
        )

# Define standard load profiles
LOAD_PROFILES = {
    "baseline": LoadProfile(
        name="baseline",
        duration_seconds=300,
        peak_rps=100,
        ramp_up_seconds=30,
        ramp_down_seconds=30,
        pattern="constant",
        bursts=False,
        burst_size=0,
        burst_duration=0
    ),
    "stress": LoadProfile(
        name="stress",
        duration_seconds=600,
        peak_rps=500,
        ramp_up_seconds=60,
        ramp_down_seconds=60,
        pattern="sawtooth",
        bursts=True,
        burst_size=100,
        burst_duration=30
    ),
    "endurance": LoadProfile(
        name="endurance",
        duration_seconds=3600,
        peak_rps=200,
        ramp_up_seconds=300,
        ramp_down_seconds=300,
        pattern="sine",
        bursts=False,
        burst_size=0,
        burst_duration=0
    ),
    "spike": LoadProfile(
        name="spike",
        duration_seconds=900,
        peak_rps=1000,
        ramp_up_seconds=10,
        ramp_down_seconds=60,
        pattern="constant",
        bursts=True,
        burst_size=500,
        burst_duration=10
    )
}

@pytest.fixture
def mock_ml_service(test_data_dir: Path) -> Generator[Callable, None, None]:
    """Create mock ML service for load testing."""
    from aiohttp import web
    import asyncio
    
    async def handle_request(request):
        # Simulate ML processing
        await asyncio.sleep(np.random.exponential(0.1))
        return web.Response(text='OK')
    
    app = web.Application()
    app.router.add_post('/predict', handle_request)
    
    runner = web.AppRunner(app)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, 'localhost', 8083)
    loop.run_until_complete(site.start())
    
    yield lambda: 'http://localhost:8083/predict'
    
    loop.run_until_complete(runner.cleanup())

@pytest.mark.asyncio
async def test_baseline_load(mock_ml_service: Callable) -> None:
    """Test baseline load handling."""
    generator = LoadGenerator(LOAD_PROFILES["baseline"])
    await generator.generate_load(mock_ml_service())
    
    results = generator.get_results()
    assert results.failed_requests / results.total_requests < 0.01  # Less than 1% failures
    assert results.p95_latency < 0.5  # 500ms p95 latency
    assert results.max_memory_mb < 512  # Less than 512MB memory usage

@pytest.mark.asyncio
async def test_stress_load(mock_ml_service: Callable) -> None:
    """Test stress load handling."""
    generator = LoadGenerator(LOAD_PROFILES["stress"])
    await generator.generate_load(mock_ml_service())
    
    results = generator.get_results()
    assert results.failed_requests / results.total_requests < 0.05  # Less than 5% failures
    assert results.p99_latency < 2.0  # 2s p99 latency
    assert results.avg_cpu_percent < 90  # Less than 90% CPU usage

@pytest.mark.asyncio
async def test_endurance_load(mock_ml_service: Callable) -> None:
    """Test endurance load handling."""
    generator = LoadGenerator(LOAD_PROFILES["endurance"])
    await generator.generate_load(mock_ml_service())
    
    results = generator.get_results()
    assert results.failed_requests / results.total_requests < 0.02  # Less than 2% failures
    assert results.max_memory_mb < 1024  # Less than 1GB memory usage
    assert not results.errors  # No error types recorded

@pytest.mark.asyncio
async def test_spike_load(mock_ml_service: Callable) -> None:
    """Test spike load handling."""
    generator = LoadGenerator(LOAD_PROFILES["spike"])
    await generator.generate_load(mock_ml_service())
    
    results = generator.get_results()
    # Allow higher failure rate during spikes
    assert results.failed_requests / results.total_requests < 0.10  # Less than 10% failures
    # Check recovery after spikes
    assert results.p95_latency < 1.0  # 1s p95 latency

def test_load_impact_on_ml_pipeline(test_data_dir: Path, mock_ml_service: Callable) -> None:
    """Test ML pipeline performance under load."""
    # Setup pipeline components
    history_file = test_data_dir / "load_test_history.json"
    predictor = PerformancePredictor(history_file)
    validator = ModelValidator(test_data_dir, history_file)
    
    # Run load test in background thread
    async def run_load():
        generator = LoadGenerator(LOAD_PROFILES["stress"])
        await generator.generate_load(mock_ml_service())
        return generator.get_results()
    
    load_thread = threading.Thread(target=lambda: asyncio.run(run_load()))
    load_thread.start()
    
    # Measure pipeline performance under load
    start_time = time.time()
    pipeline_stats = {
        "model_updates": 0,
        "validations": 0,
        "errors": 0
    }
    
    while load_thread.is_alive():
        try:
            predictor.train_models()
            pipeline_stats["model_updates"] += 1
            
            if validator.validate_all_models():
                pipeline_stats["validations"] += 1
            
        except Exception:
            pipeline_stats["errors"] += 1
        
        time.sleep(1)
    
    load_thread.join()
    duration = time.time() - start_time
    
    # Verify pipeline performance
    assert pipeline_stats["model_updates"] > 0
    assert pipeline_stats["validations"] > 0
    assert pipeline_stats["errors"] / (pipeline_stats["model_updates"] + pipeline_stats["validations"]) < 0.05

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
