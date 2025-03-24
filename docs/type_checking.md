# Type Checking Guide

This project uses static type checking to catch errors early and provide better IDE support. Here's how to work with the type system.

## Quick Start

### Running Type Checks
```bash
# Basic check
python scripts/check_types.py

# Install dependencies and run strict checks
python scripts/check_types.py --install --strict

# Fix formatting and generate report
python scripts/check_types.py --fix --report

# Check specific paths
python scripts/check_types.py proxy/models proxy/utils
```

## Project Structure

### Type Stub Files
Type stubs (`.pyi` files) provide type information for modules:
```
proxy/
├── models/
│   ├── server.py          # Implementation
│   ├── server.pyi         # Type definitions
│   ├── connection.py
│   ├── connection.pyi
│   ├── server_state.py
│   └── server_state.pyi
└── utils/
    ├── constants.py
    ├── constants.pyi
    ├── logging.py
    └── logging.pyi
```

## Writing Type Hints

### Basic Examples
```python
from typing import Optional, List, Dict, Any

# Function with type hints
def process_data(items: List[str], config: Optional[Dict[str, Any]] = None) -> int:
    return len(items)

# Async function
async def fetch_data(url: str) -> bytes:
    ...

# Class with type hints
class Configuration:
    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
    
    def get_setting(self, name: str) -> Optional[str]:
        ...
```

### Using Project Types
```python
from proxy.models import ProxyServer, ServerState
from proxy.utils.constants import NetworkConfig, SSLConfig

async def start_server(
    state: ServerState,
    port: int = NetworkConfig.DEFAULT_PORT
) -> ProxyServer:
    server = ProxyServer(port=port)
    await server.start()
    return server
```

### Context Managers
```python
from proxy import create_proxy_server

# Using async context manager
async with proxy_server_context() as server:
    await server.start()

# Manual resource management
server = await create_proxy_server()
try:
    await server.start()
finally:
    server.close()
    await server.cleanup_resources()
```

## Common Issues and Best Practices

### Missing Type Hints
```python
# ❌ Bad - Missing type hints
def process(data):
    return data + 1

# ✅ Good - Complete type hints
def process(data: int) -> int:
    return data + 1
```

### Optional Types
```python
# ❌ Bad - Implicit Optional
def get_config(path: str = None):
    ...

# ✅ Good - Explicit Optional
from typing import Optional
def get_config(path: Optional[str] = None) -> Dict[str, Any]:
    ...
```

### Async Functions
```python
# ❌ Bad - Missing return type
async def fetch():
    ...

# ✅ Good - Complete type hints
async def fetch() -> bytes:
    await some_operation()
    return result
```

### Collections
```python
# ❌ Bad - Using bare types
def process_items(items: list) -> dict:
    ...

# ✅ Good - Using generic types
from typing import List, Dict, Any
def process_items(items: List[str]) -> Dict[str, Any]:
    ...
```

## Type Checking Configuration

### mypy Settings
The project uses strict type checking settings in `mypy.ini`:
- `disallow_untyped_defs = True`
- `check_untyped_defs = True`
- `warn_return_any = True`
- `strict_optional = True`

### IDE Integration

#### VSCode
1. Install Python extension
2. Enable type checking:
   ```json
   {
     "python.analysis.typeCheckingMode": "strict"
   }
   ```
3. Use Pylance for best results

#### PyCharm
1. Enable type checking in Settings:
   - Editor → Inspections → Python
   - Enable "Type checker" inspection
2. Set severity level to "Error" for strict checking

## Continuous Integration

Type checking runs automatically on pull requests and pushes:
- Multiple Python versions (3.8, 3.9, 3.10)
- Generates HTML reports
- Checks stub files
- Verifies type completeness

## Advanced Topics

### Custom Types
```python
from typing import TypedDict, Literal

# Custom type definitions
class ServerStats(TypedDict):
    total_connections: int
    active_connections: int
    bytes_transferred: int
    peak_memory_mb: float

# Literal types
LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR']
```

### Type Variables
```python
from typing import TypeVar, Generic

T = TypeVar('T')

class Cache(Generic[T]):
    def get(self, key: str) -> Optional[T]: ...
    def set(self, key: str, value: T) -> None: ...
```

## Resources

- [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html)
