"""Base class for all reconnaissance tools."""

import asyncio
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple, TypeVar, Generic

from recon_module.common.progress import ReconProgress
from recon_module.rate_limiter import RateLimiter

# Generic type for scan results
T = TypeVar('T')

class BaseReconTool(ABC, Generic[T]):
    def __init__(self, progress: ReconProgress, rate_limiter: RateLimiter):
        """Initialize base recon tool."""
        self.progress = progress
        self.rate_limiter = rate_limiter
        self.tool_name = self.__class__.__name__.lower()

    async def run_command(
        self, 
        cmd: List[str], 
        domain: str, 
        estimated_size: int = 1,
        progress_start: int = 10,
        progress_end: int = 75,
        tool_name: Optional[str] = None
    ) -> Tuple[List[str], float]:
        """Run a command with rate limiting and progress tracking.
        
        Args:
            cmd: Command to execute as list of strings
            domain: Target domain
            estimated_size: Estimated size of scan for progress tracking
            progress_start: Start percentage for progress bar
            progress_end: End percentage for progress bar
            tool_name: Override default tool name for rate limiting
        """
        # Use provided tool name or default to class tool_name
        effective_tool = tool_name or self.tool_name
        
        # Check rate limits
        can_scan, message = self.rate_limiter.can_scan(domain, effective_tool)
        if not can_scan:
            raise Exception(f"Rate limit exceeded: {message}")

        self.progress.update(progress_start, 100, f"Running {self.tool_name}...")
        
        try:
            # Start tracking the scan
            self.rate_limiter.start_scan(domain, effective_tool)
            start_time = asyncio.get_event_loop().time()

            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            duration = asyncio.get_event_loop().time() - start_time

            if process.returncode != 0:
                error_msg = stderr.decode()
                print(f"Error running command {' '.join(cmd)}: {error_msg}")
                # Clean up rate limiter on error
                self.rate_limiter.end_scan(domain, effective_tool, 0, duration)
                raise Exception(error_msg)
            
            # Update progress
            self.progress.update(progress_end, 100, f"Processing {self.tool_name} results...")
            
            # Process output
            results = [line for line in stdout.decode().split('\n') if line.strip()]
            
            # Record scan completion with actual size
            actual_size = len(results) if results else estimated_size
            self.rate_limiter.end_scan(domain, effective_tool, actual_size, duration)
            
            return results, duration

        except Exception as e:
            print(f"Error executing {self.tool_name}: {str(e)}")
            # Ensure scan is removed from active scans on error
            self.rate_limiter.end_scan(domain, self.tool_name, 0, 0)
            raise

    @abstractmethod
    async def scan(self, domain: str) -> Tuple[T, float]:
        """
        Run the scan operation. Must be implemented by subclasses.
        
        Args:
            domain: Target domain to scan
            
        Returns:
            Tuple of (scan result, scan duration)
        """
        raise NotImplementedError("Subclasses must implement scan()")

    def estimate_scan_size(self, target: str) -> int:
        """
        Estimate the size/complexity of a scan.
        Override in subclasses for more accurate estimates.
        """
        return 1

    @staticmethod
    def parse_json_line(line: str) -> Optional[dict]:
        """Safely parse a JSON line."""
        try:
            import json
            return json.loads(line)
        except json.JSONDecodeError:
            return None
        except Exception as e:
            print(f"Error parsing JSON line: {e}")
            return None
