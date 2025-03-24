"""Memory leak prevention and mitigation."""

import asyncio
from typing import Dict, List, Any, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import gc
import weakref
import inspect
import logging
from collections import defaultdict
import objsize
from functools import wraps

from .memory_leak_detection import LeakDetector, LeakPattern, LeakSnapshot

@dataclass
class PreventionConfig:
    """Configuration for leak prevention."""
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    gc_threshold: float = 0.8  # 80% of max memory
    collection_interval: float = 300.0  # 5 minutes
    enable_weakref: bool = True
    track_locals: bool = True
    track_closures: bool = True
    max_object_age: float = 3600.0  # 1 hour
    cleanup_batch_size: int = 1000
    logging_level: str = "INFO"
    mitigation_strategies: Set[str] = field(default_factory=lambda: {
        "gc_collect",
        "weakref_proxy",
        "object_finalization",
        "circular_breaking",
        "local_cleanup"
    })

@dataclass
class PreventionStats:
    """Statistics for prevention actions."""
    collections: int = 0
    objects_collected: int = 0
    memory_freed: int = 0
    collection_time: float = 0.0
    retry_count: int = 0
    last_collection: Optional[datetime] = None
    mitigation_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

class LeakPrevention:
    """Prevent and mitigate memory leaks."""
    
    def __init__(
        self,
        detector: LeakDetector,
        config: PreventionConfig = None
    ):
        self.detector = detector
        self.config = config or PreventionConfig()
        
        # Prevention state
        self.stats = PreventionStats()
        self.object_cache: Dict[int, Any] = {}
        self.weak_refs: Dict[int, weakref.ReferenceType] = {}
        self.cleanup_tasks: List[asyncio.Task] = []
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.config.logging_level),
            format="%(asctime)s [%(levelname)s] %(message)s"
        )
    
    async def start_prevention(self):
        """Start prevention tasks."""
        if not self.config.enabled:
            return
        
        # Start cleanup task
        asyncio.create_task(self._run_cleanup())
        
        logging.info("Leak prevention started")
    
    async def stop_prevention(self):
        """Stop prevention tasks."""
        # Cancel cleanup tasks
        for task in self.cleanup_tasks:
            task.cancel()
        
        await asyncio.gather(*self.cleanup_tasks, return_exceptions=True)
        self.cleanup_tasks.clear()
        
        logging.info("Leak prevention stopped")
    
    def wrap_function(
        self,
        func: Callable
    ) -> Callable:
        """Wrap function with leak prevention."""
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_size = self._get_memory_size()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                end_size = self._get_memory_size()
                if end_size > start_size * (1 + self.config.gc_threshold):
                    await self._cleanup_after_function(func, end_size - start_size)
        
        return wrapper
    
    async def _run_cleanup(self):
        """Run periodic cleanup."""
        while True:
            try:
                current_size = self._get_memory_size()
                
                if current_size > self.config.gc_threshold * self._get_max_memory():
                    await self._perform_cleanup()
                
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logging.error(f"Cleanup error: {e}")
                await asyncio.sleep(self.config.retry_delay)
    
    async def _perform_cleanup(self) -> bool:
        """Perform cleanup actions."""
        start_time = datetime.now()
        start_size = self._get_memory_size()
        success = False
        
        for _ in range(self.config.max_retries):
            try:
                # Run garbage collection
                if "gc_collect" in self.config.mitigation_strategies:
                    gc.collect()
                
                # Clean weak references
                if "weakref_proxy" in self.config.mitigation_strategies:
                    await self._cleanup_weak_refs()
                
                # Finalize objects
                if "object_finalization" in self.config.mitigation_strategies:
                    await self._finalize_objects()
                
                # Break circular references
                if "circular_breaking" in self.config.mitigation_strategies:
                    await self._break_cycles()
                
                # Clean local variables
                if "local_cleanup" in self.config.mitigation_strategies:
                    await self._cleanup_locals()
                
                success = True
                break
                
            except Exception as e:
                logging.error(f"Cleanup attempt failed: {e}")
                self.stats.retry_count += 1
                await asyncio.sleep(self.config.retry_delay)
        
        # Update stats
        end_time = datetime.now()
        self.stats.collections += 1
        self.stats.collection_time = (end_time - start_time).total_seconds()
        self.stats.objects_collected = len(gc.get_objects())
        self.stats.memory_freed = start_size - self._get_memory_size()
        self.stats.last_collection = end_time
        
        return success
    
    async def _cleanup_after_function(
        self,
        func: Callable,
        size_increase: int
    ):
        """Clean up after function execution."""
        frame = inspect.currentframe()
        if frame:
            # Get function locals
            local_vars = frame.f_back.f_locals if frame.f_back else {}
            
            # Track objects created by function
            for name, value in local_vars.items():
                obj_id = id(value)
                if obj_id not in self.object_cache:
                    self.object_cache[obj_id] = (
                        datetime.now(),
                        type(value).__name__,
                        objsize.get_deep_size(value)
                    )
            
            # Create weak references
            if self.config.enable_weakref:
                for value in local_vars.values():
                    try:
                        obj_id = id(value)
                        if (
                            obj_id not in self.weak_refs and
                            not isinstance(value, (int, float, str, bool))
                        ):
                            self.weak_refs[obj_id] = weakref.proxy(value)
                    except TypeError:
                        continue
        
        # Perform cleanup if needed
        if size_increase > 0:
            await self._perform_cleanup()
    
    async def _cleanup_weak_refs(self):
        """Clean up dead weak references."""
        dead_refs = set()
        
        for obj_id, ref in self.weak_refs.items():
            try:
                # Access ref to check if dead
                _ = ref.__class__
            except ReferenceError:
                dead_refs.add(obj_id)
        
        # Remove dead references
        for obj_id in dead_refs:
            del self.weak_refs[obj_id]
            if obj_id in self.object_cache:
                del self.object_cache[obj_id]
        
        self.stats.mitigation_counts["weakref_cleanup"] += len(dead_refs)
    
    async def _finalize_objects(self):
        """Finalize old objects."""
        current_time = datetime.now()
        finalized = 0
        
        for obj_id, (create_time, _, _) in list(self.object_cache.items()):
            if (current_time - create_time).total_seconds() > self.config.max_object_age:
                if obj_id in self.weak_refs:
                    del self.weak_refs[obj_id]
                del self.object_cache[obj_id]
                finalized += 1
            
            if finalized >= self.config.cleanup_batch_size:
                break
        
        self.stats.mitigation_counts["object_finalization"] += finalized
    
    async def _break_cycles(self):
        """Break circular references."""
        broken = 0
        
        for obj in gc.get_objects():
            if not isinstance(obj, (tuple, list, dict, set)):
                continue
            
            try:
                # Clear containers
                if isinstance(obj, (list, set)):
                    obj.clear()
                elif isinstance(obj, dict):
                    obj.clear()
                broken += 1
            except Exception:
                continue
            
            if broken >= self.config.cleanup_batch_size:
                break
        
        self.stats.mitigation_counts["cycle_breaking"] += broken
    
    async def _cleanup_locals(self):
        """Clean up local variables."""
        if not self.config.track_locals:
            return
        
        cleaned = 0
        current_time = datetime.now()
        
        for frame_info in inspect.stack()[1:]:
            try:
                # Clean old locals
                if hasattr(frame_info, "frame"):
                    for name, value in frame_info.frame.f_locals.items():
                        obj_id = id(value)
                        if obj_id in self.object_cache:
                            create_time, _, _ = self.object_cache[obj_id]
                            if (
                                current_time - create_time
                            ).total_seconds() > self.config.max_object_age:
                                frame_info.frame.f_locals[name] = None
                                cleaned += 1
                
                if cleaned >= self.config.cleanup_batch_size:
                    break
            finally:
                del frame_info
        
        self.stats.mitigation_counts["local_cleanup"] += cleaned
    
    def _get_memory_size(self) -> int:
        """Get current memory size."""
        return sum(
            objsize.get_deep_size(obj)
            for obj in gc.get_objects()
        )
    
    def _get_max_memory(self) -> int:
        """Get maximum allowed memory."""
        # Implementation would get system memory limits
        return sys.maxsize

def create_leak_prevention(
    detector: LeakDetector,
    config: Optional[PreventionConfig] = None
) -> LeakPrevention:
    """Create leak prevention."""
    return LeakPrevention(detector, config)

if __name__ == "__main__":
    from .memory_leak_detection import create_leak_detector
    from .scheduler_profiling import create_profiling_hook
    from .scheduler_monitoring import create_scheduler_monitor
    from .report_scheduler import create_report_scheduler
    from .validation_reports import create_validation_reporter
    from .rule_validation import create_rule_validator
    from .notification_rules import create_rule_engine
    from .alert_notifications import create_notification_manager
    from .anomaly_alerts import create_alert_manager
    from .anomaly_analysis import create_anomaly_detector
    from .trend_analysis import create_trend_analyzer
    from .adaptation_metrics import create_performance_tracker
    from .preset_adaptation import create_online_adapter
    from .preset_ensemble import create_preset_ensemble
    from .preset_predictions import create_preset_predictor
    from .preset_analytics import create_preset_analytics
    from .mutation_presets import create_preset_manager
    
    async def main():
        # Setup components
        manager = create_preset_manager()
        analytics = create_preset_analytics(manager)
        predictor = create_preset_predictor(analytics)
        ensemble = create_preset_ensemble(predictor)
        adapter = create_online_adapter(ensemble)
        tracker = create_performance_tracker(adapter)
        analyzer = create_trend_analyzer(tracker)
        detector = create_anomaly_detector(tracker, analyzer)
        alert_manager = create_alert_manager(detector)
        notifier = create_notification_manager(alert_manager)
        engine = create_rule_engine(notifier)
        validator = create_rule_validator(engine)
        reporter = create_validation_reporter(validator)
        scheduler = create_report_scheduler(reporter)
        monitor = create_scheduler_monitor(scheduler)
        profiling = create_profiling_hook()
        leak_detector = create_leak_detector(profiling)
        prevention = create_leak_prevention(leak_detector)
        
        # Start prevention
        await prevention.start_prevention()
        
        try:
            # Test leak prevention
            @prevention.wrap_function
            async def leaky_function():
                # Create some objects
                data = []
                for i in range(1000):
                    data.append([i] * 1000)
                return len(data)
            
            # Run function multiple times
            for _ in range(5):
                result = await leaky_function()
                print(f"\nFunction result: {result}")
                print("Prevention stats:")
                print(f"  Collections: {prevention.stats.collections}")
                print(f"  Objects collected: {prevention.stats.objects_collected:,}")
                print(f"  Memory freed: {prevention.stats.memory_freed:,} bytes")
                print("  Mitigation counts:", dict(prevention.stats.mitigation_counts))
                await asyncio.sleep(1)
        
        finally:
            await prevention.stop_prevention()
    
    asyncio.run(main())
