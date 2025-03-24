"""Type stubs for pytest decorators and utilities."""
from typing import (
    Any, Callable, TypeVar, Type, Optional, Dict, Union, List,
    Generator, AsyncGenerator, overload, NoReturn
)
from pathlib import Path
from typing_extensions import TypeAlias, Protocol

# Type variables
_F = TypeVar("_F", bound=Callable[..., Any])
_T = TypeVar("_T")
_R = TypeVar("_R")
E = TypeVar("E", bound=BaseException)

# Basic types
Marker: TypeAlias = Callable[[_F], _F]

class FixtureFunction(Protocol[_R]):
    def __call__(self, request: Any = None, *args: Any, **kwargs: Any) -> _R: ...

class ExceptionInfo:
    type: Type[BaseException]
    value: BaseException
    traceback: Any

class FixtureRequest:
    module: Any
    function: Any
    param: Any

class TimeHelper:
    def __call__(self, *args: Any) -> float: ...
    def sleep(self, seconds: float) -> None: ...
    def monotonic(self) -> float: ...
    def time(self) -> float: ...

class Helpers:
    time: TimeHelper
    async def run_async(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any: ...

# Mark class with dynamic attribute support
class Mark:
    helpers: Helpers
    def __init__(self) -> None: ...
    def __getattr__(self, name: str) -> Marker: ...
    def __setattr__(self, name: str, value: Any) -> None: ...
    def parametrize(self, *args: Any, **kwargs: Any) -> Marker: ...
    def asyncio(self, *args: Any, **kwargs: Any) -> Marker: ...
    def skip(self, *args: Any, **kwargs: Any) -> Marker: ...

# Main exports
mark: Mark
helpers: Helpers

@overload
def fixture(*, scope: str = "function", params: Any = None, autouse: bool = False,
            ids: Any = None, name: Optional[str] = None) -> Callable[[_T], _T]: ...
@overload
def fixture(function: _T) -> _T: ...

def skip(reason: str = "") -> None: ...
def xfail(reason: str = "") -> None: ...
def fail(*args: Any, **kwargs: Any) -> None: ...

# Raises context manager
class Raises:
    def __init__(self, expected_exception: Type[E], *args: Any, **kwargs: Any) -> None: ...
    def __enter__(self) -> ExceptionInfo: ...
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool: ...

raises: Callable[..., Raises]
