"""Screenshot functionality for web endpoints."""

import os
import tempfile
from typing import Optional
import pyppeteer  # type: ignore

class ScreenshotManager:
    def __init__(self):
        """Initialize screenshot manager with temporary directory."""
        self._screenshots_dir = tempfile.mkdtemp()

    async def take_screenshot(self, url: str) -> Optional[str]:
        """Take a screenshot of a URL using pyppeteer."""
        try:
            # Launch browser with security settings
            browser = await pyppeteer.launch(
                headless=True,
                args=[
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--disable-web-security'
                ]
            )
            
            # Set up page with reasonable defaults
            page = await browser.newPage()
            await page.setViewport({'width': 1280, 'height': 800})
            
            # Add timeout and error handling
            try:
                await page.goto(url, {
                    'waitUntil': 'networkidle0',
                    'timeout': 30000  # 30 seconds timeout
                })
            except Exception as e:
                print(f"Error loading page {url}: {e}")
                await browser.close()
                return None
            
            # Generate unique filename using URL hash
            filename = f"{self._screenshots_dir}/{hash(url)}.png"
            
            # Take the screenshot
            await page.screenshot({
                'path': filename,
                'fullPage': True,
                'type': 'png',
                'quality': 80
            })
            
            await browser.close()
            
            return filename if os.path.exists(filename) else None
            
        except Exception as e:
            print(f"Error taking screenshot of {url}: {e}")
            return None

    def cleanup(self):
        """Clean up screenshot directory."""
        try:
            # Remove files older than 24 hours
            import time
            current_time = time.time()
            for filename in os.listdir(self._screenshots_dir):
                filepath = os.path.join(self._screenshots_dir, filename)
                if os.path.getmtime(filepath) < current_time - 86400:  # 24 hours
                    os.remove(filepath)
        except Exception as e:
            print(f"Error cleaning up screenshots: {e}")

    @property
    def screenshots_dir(self) -> str:
        """Get the screenshots directory path."""
        return self._screenshots_dir
