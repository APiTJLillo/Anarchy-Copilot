"""Type stubs for pyppeteer."""

from typing import Dict, Any, List, Optional, Union
from typing_extensions import TypedDict

class ViewPort(TypedDict):
    width: int
    height: int

class ScreenshotOptions(TypedDict, total=False):
    path: str
    type: str
    quality: int
    fullPage: bool
    clip: Dict[str, float]
    encoding: str
    omitBackground: bool

class NavigationOptions(TypedDict, total=False):
    timeout: int
    waitUntil: Union[str, List[str]]
    referer: str

class Browser:
    async def newPage(self) -> 'Page': ...
    async def close(self) -> None: ...

class Page:
    async def setViewport(self, viewport: ViewPort) -> None: ...
    async def goto(self, url: str, options: Optional[NavigationOptions] = None) -> Optional[Any]: ...
    async def screenshot(self, options: Optional[ScreenshotOptions] = None) -> Optional[bytes]: ...
    async def close(self) -> None: ...

async def launch(
    ignoreHTTPSErrors: bool = False,
    headless: bool = True,
    executablePath: Optional[str] = None,
    args: Optional[List[str]] = None,
    ignoreDefaultArgs: Optional[Union[bool, List[str]]] = False,
    handleSIGINT: bool = True,
    handleSIGTERM: bool = True,
    handleSIGHUP: bool = True,
    dumpio: bool = False,
    userDataDir: Optional[str] = None,
    env: Optional[Dict[str, Union[str, int]]] = None,
    devtools: bool = False,
    pipe: bool = False,
    **kwargs: Any
) -> Browser: ...
