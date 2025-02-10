from typing import Any, Dict, Optional, Union

class Response:
    def __init__(
        self,
        content: Any = None,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        media_type: Optional[str] = None,
    ) -> None: ...

    @property
    def body(self) -> bytes: ...
