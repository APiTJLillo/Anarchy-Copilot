from typing import List, Optional, Union

class CORSMiddleware:
    def __init__(
        self,
        allow_origins: List[str] = ["*"],
        allow_methods: List[str] = ["*"],
        allow_headers: List[str] = ["*"],
        allow_credentials: bool = False,
        expose_headers: List[str] = [],
        max_age: int = 600,
    ) -> None: ...

    async def __call__(self, scope: dict, receive: callable, send: callable) -> None: ...
