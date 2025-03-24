"""Nuclei process execution utilities."""
import asyncio 
import logging
from pathlib import Path
from typing import List, Optional, AsyncIterator, Dict, Any
import aiofiles
import json
import shutil

logger = logging.getLogger(__name__)

class NucleiProcess:
    """Manages Nuclei subprocess execution and output handling."""
    
    def __init__(self):
        """Initialize process manager."""
        self._process: Optional[asyncio.subprocess.Process] = None
        self._output_file: Optional[Path] = None
        self._running = False

    def set_output_file(self, path: Path) -> None:
        """Set the output file path."""
        self._output_file = path

    async def run(
        self,
        target: str,
        templates: List[str],
        *,
        proxy: Optional[str] = None,
        verify_ssl: bool = True,
        rate_limit: Optional[int] = None
    ) -> None:
        """Run Nuclei with specified configuration."""
        if not self._output_file:
            raise RuntimeError("Output file not configured")

        cmd = [
            shutil.which("nuclei") or "nuclei",  # Find nuclei in PATH or use default
            "-target", target,
            "-json",
            "-output", str(self._output_file),
            "-templates", *templates
        ]

        if proxy:
            cmd.extend(["-proxy", proxy])
        
        if not verify_ssl:
            cmd.append("-insecure")

        if rate_limit:
            cmd.extend(["-rate-limit", str(rate_limit)])

        logger.info(f"Starting nuclei: {' '.join(cmd)}")
        self._running = True
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError:
            logger.error("Nuclei binary not found")
            self._running = False
            raise

    async def stop(self) -> None:
        """Stop the Nuclei process."""
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
                logger.info("Nuclei process terminated")
            except ProcessLookupError:
                logger.warning("Nuclei process already terminated")
            finally:
                self._process = None

    async def _read_output(self) -> AsyncIterator[Dict[str, Any]]:
        """Read and parse output file contents."""
        if not self._output_file or not self._output_file.exists():
            return

        async with aiofiles.open(str(self._output_file), 'r') as f:
            content = await f.read()
            for line in content.splitlines():
                if not line.strip():
                    continue
                try:
                    result = json.loads(line)
                    yield result
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse nuclei result: {e}")

    async def read_results(self) -> AsyncIterator[Dict[str, Any]]:
        """Read and parse Nuclei results as they become available."""
        if not self._output_file:
            raise RuntimeError("Output file not configured")

        # Wait for file to exist
        while self._running and not self._output_file.exists():
            await asyncio.sleep(0.1)

        if not self._running:
            return

        last_size = 0
        while self._running:
            if not self._output_file.exists():
                break

            async with aiofiles.open(str(self._output_file), 'r') as f:
                await f.seek(last_size)
                lines = await f.readlines()
                
                if not lines:
                    await asyncio.sleep(0.1)
                    continue
                
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        result = json.loads(line)
                        yield result
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse nuclei result: {e}")
                
                last_size = await f.tell()

        # Read any remaining results
        async for result in self._read_output():
            yield result

    async def is_healthy(self) -> bool:
        """Check if the Nuclei process is healthy."""
        if not self._process or not self._running:
            return False

        # Check if process has terminated
        if self._process.returncode is not None:
            logger.error(f"Nuclei process terminated with code {self._process.returncode}")
            return False

        # Read stderr for any error messages
        if self._process.stderr:
            stderr = await self._process.stderr.read()
            if stderr:
                logger.error(f"Nuclei stderr: {stderr.decode()}")
                return False

        return True

    async def wait(self) -> int:
        """Wait for the Nuclei process to complete.
        
        Returns:
            The process return code
        """
        if not self._process:
            raise RuntimeError("Nuclei process not started")

        return await self._process.wait()

    @property
    def is_running(self) -> bool:
        """Check if the process is running."""
        return self._running
