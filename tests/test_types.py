"""Tests for type checking functionality."""

import os
import sys
import subprocess
import pytest
from pathlib import Path
from typing import List, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from proxy.models import ProxyServer, ServerState
from proxy.utils.constants import NetworkConfig

def run_mypy(paths: List[str], config_file: str = "mypy.ini") -> Tuple[bool, str]:
    """Run mypy on given paths and return (success, output)."""
    cmd = [
        sys.executable, 
        "-m", 
        "mypy",
        "--config-file", 
        config_file,
        *paths
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stdout + e.stderr

def test_mypy_version():
    """Verify mypy is installed and has correct version."""
    cmd = [sys.executable, "-m", "pip", "show", "mypy"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, "mypy not installed"
    
    # Extract version from pip output
    version_line = [line for line in result.stdout.split('\n') 
                   if line.startswith('Version:')][0]
    version = version_line.split(': ')[1]
    
    # Verify minimum version
    major, minor, *_ = map(int, version.split('.'))
    assert (major, minor) >= (1, 0), f"mypy version {version} is too old"

def test_stub_files_exist():
    """Verify all .py files have corresponding .pyi files."""
    python_files = set()
    stub_files = set()
    
    for root, _, files in os.walk(project_root / "proxy"):
        for file in files:
            if file.endswith('.py') and not file.startswith('__'):
                python_files.add(Path(root) / file)
            elif file.endswith('.pyi'):
                stub_files.add(Path(root) / file.replace('.pyi', '.py'))
    
    # Check that every .py file has a .pyi stub
    missing_stubs = python_files - stub_files
    assert not missing_stubs, f"Missing stubs for: {missing_stubs}"

@pytest.mark.parametrize("module_path", [
    "proxy/models/server.py",
    "proxy/models/connection.py",
    "proxy/models/server_state.py",
    "proxy/utils/constants.py",
    "proxy/utils/logging.py",
])
def test_module_type_checking(module_path: str):
    """Test individual module type checking."""
    success, output = run_mypy([module_path])
    assert success, f"Type checking failed for {module_path}:\n{output}"

def test_example_type_checking():
    """Verify examples have correct type hints."""
    success, output = run_mypy(["examples/typed_proxy_usage.py"])
    assert success, f"Example type checking failed:\n{output}"

@pytest.mark.asyncio
async def test_runtime_type_annotations():
    """Test that runtime type annotations work correctly."""
    # Create server with type annotations
    server = ProxyServer(
        host="localhost",
        port=NetworkConfig.DEFAULT_PORT
    )
    
    assert isinstance(server.state, ServerState)
    assert isinstance(server.port, int)
    
    # Test dict with type annotations
    stats = server.state.stats
    assert isinstance(stats["total_connections"], int)
    assert isinstance(stats["bytes_transferred"], int)
    assert isinstance(stats["peak_memory_mb"], (int, float))

def test_stub_consistency():
    """Verify stubs match implementation."""
    from mypy.stubtest import test_stubs
    
    # Run stubtest on our modules
    result = test_stubs(
        ["proxy"],
        options=["--ignore-missing-stub"],
        verbose=True
    )
    
    assert result == 0, "Stub files don't match implementation"

@pytest.mark.asyncio
async def test_type_narrowing():
    """Test type narrowing works correctly."""
    server = ProxyServer()
    
    # Should narrow Optional[socket.socket] to socket.socket
    assert server._socket is None
    await server.start()
    assert server._socket is not None
    
    # Cleanup
    server.close()
    await server.cleanup_resources()

def test_generic_types():
    """Test generic type handling."""
    from typing import Dict, Any
    
    # Test TypedDict
    from proxy.utils.constants import ProxyConfig
    config: ProxyConfig = {
        "host": "localhost",
        "port": 8443,
        "cert_path": "cert.pem",
        "key_path": "key.pem",
        "backlog": 100,
        "debug": True
    }
    
    # This should type check correctly
    reveal_type_output = run_mypy(
        ["tests/test_types.py"],
        config_file="tests/mypy_reveal.ini"
    )[1]
    assert "ProxyConfig" in reveal_type_output

if __name__ == "__main__":
    pytest.main([__file__])
