from typing import Any, Type

class Middleware:
    def __init__(self, cls: Type[Any], **options: Any) -> None: ...
