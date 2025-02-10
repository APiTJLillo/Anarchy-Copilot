from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar("T")

class FastAPI:
    def __init__(
        self,
        *,
        debug: bool = False,
        title: str = "FastAPI",
        description: str = "",
        version: str = "0.1.0",
    ) -> None: ...

    def add_middleware(
        self,
        middleware_class: Type[Any],
        **options: Any
    ) -> None: ...

    def get(
        self,
        path: str,
        *,
        response_model: Optional[Type[Any]] = None,
        status_code: Optional[int] = None,
        tags: Optional[List[str]] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        **params: Any,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...

class HTTPException(Exception):
    def __init__(
        self,
        status_code: int,
        detail: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None: ...

class Depends:
    def __init__(self, dependency: Optional[Callable[..., Any]] = None) -> None: ...
    def __call__(self) -> Any: ...

class BackgroundTasks:
    def add_task(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> None: ...
