import os
import signal
import ssl
from fastapi import FastAPI, HTTPException, BackgroundTasks
from starlette.responses import JSONResponse
import asyncio
import logging
from typing import Optional, List, Dict, Set, Any
from pathlib import Path
import time
import psutil
import gc
import sys
import resource
import aiohttp
from datetime import datetime

from .proxy_server import ProxyServer
from ..config import ProxyConfig
from .certificates import CertificateAuthority

logger = logging.getLogger(__name__)

app = FastAPI()

class SharedState:
    def __init__(self):
        self.proxy_server: Optional[ProxyServer] = None
        self.proxy_task: Optional[asyncio.Task] = None
        self.is_shutting_down = False
        self.last_gc_time: float = 0.0
        self.alert_subscribers: Set[str] = set()
        self.initialization_lock = asyncio.Lock()
        self.startup_complete = asyncio.Event()

state = SharedState()

app = FastAPI()

def parse_path_env(env_var: str) -> Optional[Path]:
    """Parse path from environment variable and validate it exists.
    
    Args:
        env_var: Environment variable name
        
    Returns:
        Optional[Path]: Path object if environment variable is set and path exists,
                       None otherwise
    """
    path_str = os.getenv(env_var)
    if not path_str:
        logger.warning(f"Environment variable {env_var} not set")
        return None
        
    path = Path(path_str)
    if not path.exists():
        logger.error(f"Path {path} from environment variable {env_var} does not exist")
        return None
    if not os.access(path, os.R_OK):
        logger.error(f"Path {path} from environment variable {env_var} is not readable")
        return None
        
    logger.info(f"Successfully validated path {path} from environment variable {env_var}")
    return path

def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"

async def shutdown():
    """Gracefully shut down the proxy server."""
    if state.is_shutting_down:
        return

    state.is_shutting_down = True
    logger.info("Initiating graceful shutdown...")

    try:
        if state.proxy_server:
            await state.proxy_server.stop()
            state.proxy_server = None
            logger.info("Proxy server stopped")

        # Cancel proxy task if running
        if state.proxy_task and not state.proxy_task.done():
            state.proxy_task.cancel()
            try:
                await state.proxy_task
            except asyncio.CancelledError:
                pass
            state.proxy_task = None
            logger.info("Proxy task cancelled")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    finally:
        state.is_shutting_down = False

@app.on_event("startup")
async def startup_event():
    """Initialize and start the proxy server."""
    async with state.initialization_lock:
        try:
            # Initialize proxy state first
            from api.proxy.state import proxy_state
            await proxy_state.initialize()
            logger.info("Proxy state initialized")

            # Get certificate paths
            ca_cert_path = parse_path_env('CA_CERT_PATH')
            ca_key_path = parse_path_env('CA_KEY_PATH')
            
            # Check if both paths are available
            if not ca_cert_path or not ca_key_path:
                logger.warning("One or both CA certificate paths are not properly configured:")
                if not ca_cert_path:
                    logger.warning("- CA_CERT_PATH is not set or invalid")
                if not ca_key_path:
                    logger.warning("- CA_KEY_PATH is not set or invalid")
                logger.warning("HTTPS interception will be disabled.")
                ca_instance = None
            else:
                try:
                    # Initialize Certificate Authority but don't start it yet
                    ca_instance = CertificateAuthority(
                        ca_cert_path=ca_cert_path,
                        ca_key_path=ca_key_path
                    )
                    logger.info("Certificate Authority initialized successfully")
                    
                    # Start the CA
                    await ca_instance.start()
                    logger.info("Certificate Authority started successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize/start Certificate Authority: {e}")
                    ca_instance = None
                    logger.warning("HTTPS interception will be disabled")
            
            # Initialize proxy configuration
            config = ProxyConfig(
                host=os.getenv('ANARCHY_PROXY_HOST', '0.0.0.0'),
                port=int(os.getenv('ANARCHY_PROXY_PORT', '8083')),
                ca_cert_path=ca_cert_path,
                ca_key_path=ca_key_path,
                intercept_requests=os.getenv('ANARCHY_PROXY_INTERCEPT_REQUESTS', 'true').lower() == 'true',
                intercept_responses=os.getenv('ANARCHY_PROXY_INTERCEPT_RESPONSES', 'true').lower() == 'true',
                allowed_hosts=os.getenv('ALLOWED_HOSTS', '').split(',') if os.getenv('ALLOWED_HOSTS') else [],
                excluded_hosts=os.getenv('EXCLUDED_HOSTS', '').split(',') if os.getenv('EXCLUDED_HOSTS') else [],
                max_connections=int(os.getenv('ANARCHY_PROXY_MAX_CONNECTIONS', '100')),
                max_keepalive_connections=int(os.getenv('ANARCHY_PROXY_MAX_KEEPALIVE_CONNECTIONS', '20')),
                keepalive_timeout=int(os.getenv('ANARCHY_PROXY_KEEPALIVE_TIMEOUT', '30')),
                memory_sample_interval=float(os.getenv('MEMORY_SAMPLE_INTERVAL', '10.0')),
                memory_growth_threshold=int(os.getenv('MEMORY_GROWTH_THRESHOLD', str(10 * 1024 * 1024))),
                memory_sample_retention=int(os.getenv('MEMORY_SAMPLE_RETENTION', '3600')),
                memory_log_level=os.getenv('MEMORY_LOG_LEVEL', 'INFO'),
                memory_alert_level=os.getenv('MEMORY_ALERT_LEVEL', 'WARNING'),
                cleanup_timeout=float(os.getenv('CLEANUP_TIMEOUT', '5.0')),
                force_cleanup_threshold=int(os.getenv('FORCE_CLEANUP_THRESHOLD', str(100 * 1024 * 1024))),
                cleanup_retry_delay=float(os.getenv('CLEANUP_RETRY_DELAY', '0.5'))
            )
            
            # Initialize proxy server with CA instance
            logger.info("Starting proxy server...")
            state.proxy_server = ProxyServer(config, ca_instance=ca_instance)
            
            # Register proxy server with proxy state
            proxy_state.set_proxy_server(state.proxy_server)
            
            # Start the proxy server and wait for it to be ready
            try:
                await state.proxy_server.start()
                logger.info("Proxy server started successfully")
            except Exception as e:
                logger.error(f"Failed to start proxy server: {e}")
                raise
            
            # Set up signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda s, f: asyncio.create_task(shutdown()))

            # Signal successful startup
            state.startup_complete.set()
            
        except Exception as e:
            logger.error(f"Failed to start proxy server: {str(e)}")
            raise

@app.on_event("shutdown")
async def shutdown_event():
    await shutdown()

@app.get("/health")
async def health_check():
    global state
    if state.proxy_server is None or state.proxy_task is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Proxy server not initialized"}
        )
    if state.proxy_task.done():
        exception = state.proxy_task.exception()
        if exception:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "message": str(exception)}
            )
            
    # Return minimal response for health checks
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "Proxy server is running"
        }
    )

@app.get("/memory/stats")
async def get_memory_stats():
    """Get detailed memory statistics."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")
        
    stats = state.proxy_server._stats
    current_time = time.time()
    
    try:
        # Calculate time ranges for different intervals
        hour_ago = current_time - 3600
        minute_ago = current_time - 60
        
        # Filter samples by time range
        hour_samples = [s for s in stats.memory_samples if s.timestamp >= hour_ago]
        minute_samples = [s for s in stats.memory_samples if s.timestamp >= minute_ago]
        
        # Calculate statistics for different time ranges
        return {
            "current": {
                "rss": format_memory_size(stats.memory_samples[-1].rss),
                "vms": format_memory_size(stats.memory_samples[-1].vms),
                "shared": format_memory_size(stats.memory_samples[-1].shared),
                "timestamp": stats.memory_samples[-1].timestamp
            },
            "last_hour": {
                "samples": len(hour_samples),
                "rss_min": format_memory_size(min(s.rss for s in hour_samples)),
                "rss_max": format_memory_size(max(s.rss for s in hour_samples)),
                "rss_avg": format_memory_size(sum(s.rss for s in hour_samples) // len(hour_samples)),
                "vms_min": format_memory_size(min(s.vms for s in hour_samples)),
                "vms_max": format_memory_size(max(s.vms for s in hour_samples)),
                "vms_avg": format_memory_size(sum(s.vms for s in hour_samples) // len(hour_samples))
            },
            "last_minute": {
                "samples": len(minute_samples),
                "rss_delta": format_memory_size(minute_samples[-1].rss - minute_samples[0].rss if minute_samples else 0),
                "vms_delta": format_memory_size(minute_samples[-1].vms - minute_samples[0].vms if minute_samples else 0)
            },
            "lifetime": {
                "total_samples": len(stats.memory_samples),
                "start_time": stats.memory_samples[0].timestamp if stats.memory_samples else None,
                "peak_rss": format_memory_size(max(s.rss for s in stats.memory_samples)),
                "peak_vms": format_memory_size(max(s.vms for s in stats.memory_samples))
            }
        }
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/samples")
async def get_memory_samples(
    start: Optional[float] = None,
    end: Optional[float] = None,
    limit: int = 1000
):
    """Get raw memory samples for a time range."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")
        
    stats = state.proxy_server._stats
    samples = stats.memory_samples
    
    # Filter by time range if specified
    if start is not None:
        samples = [s for s in samples if s.timestamp >= start]
    if end is not None:
        samples = [s for s in samples if s.timestamp <= end]
        
    # Apply limit
    samples = samples[-limit:]
    
    return {
        "samples": [
            {
                "timestamp": s.timestamp,
                "rss": format_memory_size(s.rss),
                "vms": format_memory_size(s.vms),
                "shared": format_memory_size(s.shared)
            }
            for s in samples
        ],
        "total_samples": len(samples),
        "start_time": samples[0].timestamp if samples else None,
        "end_time": samples[-1].timestamp if samples else None
    }

@app.post("/memory/force_sample")
async def force_memory_sample():
    """Force an immediate memory sample."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")
        
    try:
        current_time = time.time()
        state.proxy_server._stats.add_memory_sample(current_time)
        latest = state.proxy_server._stats.memory_samples[-1]
        
        return {
            "timestamp": latest.timestamp,
            "rss": format_memory_size(latest.rss),
            "vms": format_memory_size(latest.vms),
            "shared": format_memory_size(latest.shared)
        }
    except Exception as e:
        logger.error(f"Error forcing memory sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/config")
async def get_memory_config():
    """Get current memory monitoring configuration."""
    if not hasattr(state.proxy_server, 'config'):
        raise HTTPException(status_code=503, detail="Server configuration not available")
        
    return state.proxy_server.config.get_memory_settings()
    
@app.post("/memory/gc")
async def trigger_garbage_collection(background_tasks: BackgroundTasks):
    """Trigger manual garbage collection."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")

    # Rate limit GC triggers
    current_time = time.time()
    if current_time - state.last_gc_time < 60:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {60 - int(current_time - state.last_gc_time)} seconds before triggering GC again"
        )

    try:
        # Capture pre-GC stats
        state.proxy_server._stats.add_memory_sample(current_time)
        pre_gc = state.proxy_server._stats.memory_samples[-1]

        # Run garbage collection
        gc.collect()
        state.last_gc_time = current_time
        
        # Capture post-GC stats
        state.proxy_server._stats.add_memory_sample(current_time)
        post_gc = state.proxy_server._stats.memory_samples[-1]
        
        # Calculate memory freed
        memory_freed = {
            "rss": pre_gc.rss - post_gc.rss,
            "vms": pre_gc.vms - post_gc.vms,
            "shared": pre_gc.shared - post_gc.shared
        }
        
        return {
            "gc_completed": True,
            "memory_freed": {
                "rss": format_memory_size(memory_freed["rss"]),
                "vms": format_memory_size(memory_freed["vms"]),
                "shared": format_memory_size(memory_freed["shared"])
            },
            "current_memory": {
                "rss": format_memory_size(post_gc.rss),
                "vms": format_memory_size(post_gc.vms),
                "shared": format_memory_size(post_gc.shared)
            },
            "timestamp": current_time
        }
    except Exception as e:
        logger.error(f"Error during garbage collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/leak-check")
async def check_memory_leaks(background_tasks: BackgroundTasks):
    """Check for potential memory leaks."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")
    
    stats = state.proxy_server._stats
    config = state.proxy_server.config
    
    if len(stats.memory_samples) < config.leak_detection_samples:
        raise HTTPException(
            status_code=400, 
            detail=f"Need at least {config.leak_detection_samples} samples for leak detection"
        )

    try:
        # Analyze recent samples
        recent_samples = stats.memory_samples[-config.leak_detection_samples:]
        time_span = recent_samples[-1].timestamp - recent_samples[0].timestamp
        rss_growth = recent_samples[-1].rss - recent_samples[0].rss
        
        # Calculate growth rate
        growth_rate = (rss_growth / recent_samples[0].rss) / time_span if time_span > 0 else 0
        
        result = {
            "status": "ok",
            "analysis_timespan": f"{time_span:.1f}s",
            "samples_analyzed": len(recent_samples),
            "memory_growth": {
                "absolute": format_memory_size(rss_growth),
                "rate": f"{growth_rate*100:.2f}% per second",
                "projected_1h": format_memory_size(int(rss_growth * (3600/time_span))) if time_span > 0 else "N/A"
            }
        }
        
        # Check against threshold
        if growth_rate > config.leak_growth_rate:
            result["status"] = "warning"
            result["recommendation"] = "Consider investigating memory usage patterns"
            background_tasks.add_task(notify_subscribers, f"Potential memory leak detected: {result['memory_growth']['rate']} growth rate")
            
        return result
    except Exception as e:
        logger.error(f"Error during leak check: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/system")
async def get_system_memory():
    """Get system-wide memory statistics."""
    try:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        process = psutil.Process()
        
        return {
            "system": {
                "total": format_memory_size(vm.total),
                "available": format_memory_size(vm.available),
                "used": format_memory_size(vm.used),
                "free": format_memory_size(vm.free),
                "percent": vm.percent
            },
            "swap": {
                "total": format_memory_size(sm.total),
                "used": format_memory_size(sm.used),
                "free": format_memory_size(sm.free),
                "percent": sm.percent
            },
            "process": {
                "rss": format_memory_size(process.memory_info().rss),
                "vms": format_memory_size(process.memory_info().vms),
                "percent": process.memory_percent(),
                "threads": len(process.threads()),
                "fds": process.num_fds() if hasattr(process, 'num_fds') else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting system memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def notify_subscribers(message: str, level: str = "INFO"):
    """Notify memory alert subscribers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.log(log_level, f"Memory Alert: {message} (Subscribers: {len(state.alert_subscribers)})")
    # In a real implementation, this would send alerts via webhooks/email/etc.

@app.post("/memory/subscribe")
async def subscribe_to_alerts(webhook_url: str):
    """Subscribe to memory alerts."""
    state.alert_subscribers.add(webhook_url)
    return {
        "status": "subscribed",
        "subscribers": len(state.alert_subscribers),
        "webhook_url": webhook_url
    }

@app.post("/memory/unsubscribe")
async def unsubscribe_from_alerts(webhook_url: str):
    """Unsubscribe from memory alerts."""
    state.alert_subscribers.discard(webhook_url)
    return {
        "status": "unsubscribed",
        "subscribers": len(state.alert_subscribers),
        "webhook_url": webhook_url
    }

@app.post("/memory/optimize")
async def optimize_memory(background_tasks: BackgroundTasks):
    """Perform memory optimization including garbage collection and memory compaction."""
    if not hasattr(state.proxy_server, '_stats'):
        raise HTTPException(status_code=503, detail="Memory stats not available")
        
    try:
        # Take pre-optimization snapshot
        current_time = time.time()
        state.proxy_server._stats.add_memory_sample(current_time)
        pre_opt = state.proxy_server._stats.memory_samples[-1]
        
        optimization_steps = []
        
        # Step 1: Run garbage collection with generational cleanup
        gc.collect(0)  # Collect young generation
        gc.collect(1)  # Collect middle generation
        gc.collect(2)  # Collect old generation
        optimization_steps.append("Full generational garbage collection completed")
        
        # Step 2: Clear Python's internal free lists
        sys.set_asyncgen_hooks(firstiter=None, finalizer=None)
        optimization_steps.append("Cleared async generator hooks")
        
        # Step 3: Release memory back to OS if possible
        if hasattr(gc, 'freeze'):
            gc.freeze()
            optimization_steps.append("GC freezing performed")
        
        # Step 4: Attempt memory compaction
        try:
            import ctypes
            if hasattr(ctypes, 'CFUNCTYPE'):
                libc = ctypes.CDLL('libc.so.6')
                libc.malloc_trim(0)
                optimization_steps.append("Memory compaction via malloc_trim completed")
        except Exception as e:
            logger.debug(f"Memory compaction not available: {e}")
            optimization_steps.append("Memory compaction skipped - not supported")
            
        # Step 5: Set memory usage limits
        try:
            # Get current soft limit
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            
            # Set soft limit to 90% of hard limit if it's currently higher
            if hard != resource.RLIM_INFINITY and (soft == resource.RLIM_INFINITY or soft > hard * 0.9):
                new_soft = int(hard * 0.9)
                resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
                optimization_steps.append(f"Memory limit adjusted to {format_memory_size(new_soft)}")
        except Exception as e:
            logger.debug(f"Memory limit adjustment failed: {e}")
            optimization_steps.append("Memory limit adjustment skipped")
            
        # Take post-optimization snapshot
        state.proxy_server._stats.add_memory_sample(time.time())
        post_opt = state.proxy_server._stats.memory_samples[-1]
        
        # Calculate improvements
        memory_freed = {
            "rss": pre_opt.rss - post_opt.rss,
            "vms": pre_opt.vms - post_opt.vms,
            "shared": pre_opt.shared - post_opt.shared
        }
        
        # Get current memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Format response with detailed metrics
        result = {
            "status": "completed",
            "duration": f"{time.time() - current_time:.2f}s",
            "steps_completed": optimization_steps,
            "memory_freed": {
                "rss": format_memory_size(memory_freed["rss"]),
                "vms": format_memory_size(memory_freed["vms"]),
                "shared": format_memory_size(memory_freed["shared"])
            },
            "current_state": {
                "rss": format_memory_size(memory_info.rss),
                "vms": format_memory_size(memory_info.vms),
                "percent": process.memory_percent(),
                "gc_enabled": gc.isenabled(),
                "gc_stats": gc.get_stats(),
                "threshold": gc.get_threshold()
            }
        }
        
        # Notify subscribers if significant memory was freed
        if memory_freed["rss"] > state.proxy_server.config.memory_growth_threshold:
            background_tasks.add_task(
                notify_subscribers,
                f"Memory optimization freed {format_memory_size(memory_freed['rss'])} RSS",
                "INFO"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Error during memory optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/proxy/sessions")
async def list_proxy_sessions():
    """List all proxy sessions."""
    try:
        from api.proxy.database_models import ProxySession
        
        async with AsyncSessionLocal() as db:
            # Get all sessions, ordered by start time descending
            result = await db.execute(
                text("""
                    SELECT id, name, start_time, end_time, is_active 
                    FROM proxy_sessions 
                    ORDER BY start_time DESC
                    LIMIT 100
                """)
            )
            sessions = result.fetchall()
            
            return {
                "sessions": [
                    {
                        "id": session.id,
                        "name": session.name,
                        "start_time": session.start_time.isoformat(),
                        "end_time": session.end_time.isoformat() if session.end_time else None,
                        "is_active": session.is_active
                    }
                    for session in sessions
                ]
            }
            
    except Exception as e:
        logger.error(f"Failed to list proxy sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list proxy sessions: {str(e)}"
        )
