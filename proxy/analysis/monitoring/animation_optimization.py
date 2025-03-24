"""Performance optimization for animations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
from collections import defaultdict
import cProfile
import pstats
import multiprocessing as mp
import concurrent.futures
from functools import lru_cache
import queue
import threading
import psutil
import json
from pathlib import Path

from .animation_effects import TransitionConfig, AnimationEffects

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for animation optimization."""
    max_frames: int = 1000
    min_frame_duration: float = 16.7  # ~60fps
    max_memory_mb: int = 1024
    parallel_processing: bool = True
    max_workers: int = mp.cpu_count()
    frame_batch_size: int = 10
    cache_size: int = 100
    gc_threshold: int = 1000
    profiling: bool = False
    live_monitoring: bool = True
    buffer_size: int = 50
    compression: bool = True

class AnimationOptimizer:
    """Optimize animation performance."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiler = cProfile.Profile() if config.profiling else None
        self.memory_monitor = None
        self._frame_queue = queue.Queue(maxsize=config.buffer_size)
        self._processing_threads: List[threading.Thread] = []
        self._cache = {}
        self._stats: Dict[str, Any] = defaultdict(int)
    
    def optimize_frames(
        self,
        frames: List[go.Frame],
        effects: AnimationEffects
    ) -> List[go.Frame]:
        """Optimize frame generation and processing."""
        if self.config.profiling:
            self.profiler.enable()
        
        try:
            # Initialize monitoring
            if self.config.live_monitoring:
                self._start_monitoring()
            
            # Optimize frame count
            frames = self._optimize_frame_count(frames)
            
            # Apply optimizations
            if self.config.parallel_processing:
                optimized = self._parallel_process_frames(frames, effects)
            else:
                optimized = self._sequential_process_frames(frames, effects)
            
            # Apply compression if enabled
            if self.config.compression:
                optimized = self._compress_frames(optimized)
            
            return optimized
            
        finally:
            if self.config.profiling:
                self.profiler.disable()
                self._save_profile()
            
            if self.config.live_monitoring:
                self._stop_monitoring()
    
    def _optimize_frame_count(
        self,
        frames: List[go.Frame]
    ) -> List[go.Frame]:
        """Optimize number of frames."""
        if len(frames) > self.config.max_frames:
            # Calculate optimal stride
            stride = len(frames) // self.config.max_frames
            return frames[::stride]
        return frames
    
    def _parallel_process_frames(
        self,
        frames: List[go.Frame],
        effects: AnimationEffects
    ) -> List[go.Frame]:
        """Process frames in parallel."""
        optimized_frames = []
        
        # Split frames into batches
        batches = [
            frames[i:i + self.config.frame_batch_size]
            for i in range(0, len(frames), self.config.frame_batch_size)
        ]
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            # Process batches in parallel
            futures = [
                executor.submit(
                    self._process_frame_batch,
                    batch,
                    effects
                )
                for batch in batches
            ]
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_frames = future.result()
                    optimized_frames.extend(batch_frames)
                    self._stats["processed_frames"] += len(batch_frames)
                except Exception as e:
                    logger.error(f"Frame processing failed: {e}")
                    self._stats["failed_frames"] += 1
        
        return self._sort_frames(optimized_frames)
    
    def _sequential_process_frames(
        self,
        frames: List[go.Frame],
        effects: AnimationEffects
    ) -> List[go.Frame]:
        """Process frames sequentially."""
        optimized_frames = []
        
        for frame in frames:
            try:
                processed = self._process_single_frame(frame, effects)
                if processed:
                    optimized_frames.append(processed)
                    self._stats["processed_frames"] += 1
            except Exception as e:
                logger.error(f"Frame processing failed: {e}")
                self._stats["failed_frames"] += 1
        
        return optimized_frames
    
    @lru_cache(maxsize=100)
    def _process_single_frame(
        self,
        frame: go.Frame,
        effects: AnimationEffects
    ) -> Optional[go.Frame]:
        """Process and optimize a single frame."""
        try:
            # Check cache
            cache_key = self._get_frame_cache_key(frame)
            if cache_key in self._cache:
                self._stats["cache_hits"] += 1
                return self._cache[cache_key]
            
            # Process frame data
            processed = self._optimize_frame_data(frame)
            
            # Cache result
            if len(self._cache) < self.config.cache_size:
                self._cache[cache_key] = processed
            
            return processed
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return None
    
    def _process_frame_batch(
        self,
        batch: List[go.Frame],
        effects: AnimationEffects
    ) -> List[go.Frame]:
        """Process a batch of frames."""
        processed = []
        
        for frame in batch:
            result = self._process_single_frame(frame, effects)
            if result:
                processed.append(result)
        
        return processed
    
    def _optimize_frame_data(self, frame: go.Frame) -> go.Frame:
        """Optimize frame data structures."""
        # Deep copy frame to avoid modifying original
        optimized = frame
        
        # Optimize data arrays
        for trace in optimized.data:
            if hasattr(trace, "x"):
                trace.x = self._optimize_array(trace.x)
            if hasattr(trace, "y"):
                trace.y = self._optimize_array(trace.y)
            if hasattr(trace, "z"):
                trace.z = self._optimize_array(trace.z)
        
        return optimized
    
    def _optimize_array(self, arr: Any) -> np.ndarray:
        """Optimize numeric arrays."""
        if isinstance(arr, (list, tuple)):
            arr = np.array(arr)
        
        if isinstance(arr, np.ndarray):
            # Use most efficient dtype
            if np.issubdtype(arr.dtype, np.floating):
                return arr.astype(np.float32)
            elif np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.int32)
        
        return arr
    
    def _compress_frames(self, frames: List[go.Frame]) -> List[go.Frame]:
        """Apply compression to frame data."""
        compressed = []
        
        for frame in frames:
            # Store only differences from previous frame
            if compressed:
                diff_frame = self._compute_frame_diff(
                    compressed[-1],
                    frame
                )
                compressed.append(diff_frame)
            else:
                compressed.append(frame)
        
        return compressed
    
    def _compute_frame_diff(
        self,
        prev_frame: go.Frame,
        curr_frame: go.Frame
    ) -> go.Frame:
        """Compute difference between frames."""
        diff = curr_frame
        
        # Only store changed values
        for i, trace in enumerate(diff.data):
            prev_trace = prev_frame.data[i]
            
            for attr in ["x", "y", "z"]:
                if hasattr(trace, attr):
                    curr_val = getattr(trace, attr)
                    prev_val = getattr(prev_trace, attr)
                    
                    if isinstance(curr_val, np.ndarray) and isinstance(prev_val, np.ndarray):
                        # Store only changed values
                        diff_mask = curr_val != prev_val
                        if not np.any(diff_mask):
                            delattr(trace, attr)
                        else:
                            setattr(trace, attr, curr_val[diff_mask])
        
        return diff
    
    def _sort_frames(self, frames: List[go.Frame]) -> List[go.Frame]:
        """Sort frames by name to maintain order."""
        return sorted(frames, key=lambda f: f.name)
    
    def _get_frame_cache_key(self, frame: go.Frame) -> str:
        """Generate cache key for frame."""
        return f"{frame.name}_{hash(str(frame.data))}"
    
    def _start_monitoring(self):
        """Start performance monitoring."""
        self.memory_monitor = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.memory_monitor.start()
    
    def _stop_monitoring(self):
        """Stop performance monitoring."""
        if self.memory_monitor:
            self.memory_monitor = None
    
    def _monitor_memory(self):
        """Monitor memory usage."""
        process = psutil.Process()
        
        while self.memory_monitor:
            try:
                memory_mb = process.memory_info().rss / (1024 * 1024)
                if memory_mb > self.config.max_memory_mb:
                    logger.warning(
                        f"Memory usage ({memory_mb:.1f} MB) exceeds limit "
                        f"({self.config.max_memory_mb} MB)"
                    )
                    self._clear_cache()
                
                self._stats["peak_memory_mb"] = max(
                    self._stats["peak_memory_mb"],
                    memory_mb
                )
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
            
            time.sleep(1)
    
    def _clear_cache(self):
        """Clear frame cache."""
        self._cache.clear()
        self._stats["cache_clears"] += 1
    
    def _save_profile(self):
        """Save profiling results."""
        if self.profiler:
            stats = pstats.Stats(self.profiler)
            stats.sort_stats("cumulative")
            
            profile_path = Path("animation_profile.stats")
            stats.dump_stats(profile_path)
            
            logger.info(f"Saved profiling data to {profile_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        return dict(self._stats)

def optimize_animations(
    animations: Dict[str, List[go.Frame]],
    effects: AnimationEffects,
    config: Optional[OptimizationConfig] = None
) -> Dict[str, List[go.Frame]]:
    """Optimize multiple animations."""
    optimizer = AnimationOptimizer(config or OptimizationConfig())
    optimized = {}
    
    for name, frames in animations.items():
        optimized[name] = optimizer.optimize_frames(frames, effects)
    
    return optimized

if __name__ == "__main__":
    # Example usage
    from .animation_effects import enhance_animations
    
    # Create sample animations
    animations = {
        "test": [
            go.Frame(
                data=[go.Scatter(x=[1, 2, 3], y=[1, 2, 3])],
                name=f"frame_{i}"
            )
            for i in range(100)
        ]
    }
    
    # Optimize animations
    effects = AnimationEffects(TransitionConfig())
    optimized = optimize_animations(
        animations,
        effects,
        OptimizationConfig(profiling=True)
    )
    
    print("Optimization complete")
