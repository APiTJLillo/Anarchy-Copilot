"""Visual regression tests for monitoring dashboard."""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import shutil
import imagehash
from PIL import Image
import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from proxy.analysis.monitoring.visualization import MonitoringDashboard
from .benchmark_visualization import generate_test_data, MockStore

class VisualConfig:
    """Configuration for visual testing."""
    
    # Directories for test artifacts
    BASELINE_DIR = Path("tests/visual_regression/baseline")
    CURRENT_DIR = Path("tests/visual_regression/current")
    DIFF_DIR = Path("tests/visual_regression/diff")
    
    # Comparison thresholds
    PIXEL_THRESHOLD = 0.02  # 2% pixel difference allowed
    STRUCTURE_THRESHOLD = 5  # Perceptual hash difference threshold
    
    # Screenshot settings
    VIEWPORT_WIDTH = 1920
    VIEWPORT_HEIGHT = 1080
    
    # Components to test
    COMPONENTS = [
        "alert-severity-chart",
        "metric-timeline",
        "active-alerts",
        "recent-alerts"
    ]
    
    # States to test
    STATES = [
        "default",
        "with_alerts",
        "error_state",
        "loading_state"
    ]

@pytest.fixture
def setup_dirs():
    """Setup test directories."""
    for dir_path in [
        VisualConfig.BASELINE_DIR,
        VisualConfig.CURRENT_DIR,
        VisualConfig.DIFF_DIR
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def selenium_driver():
    """Create Selenium WebDriver with consistent settings."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={VisualConfig.VIEWPORT_WIDTH},"
                        f"{VisualConfig.VIEWPORT_HEIGHT}")
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

@pytest.fixture
async def dashboard():
    """Create dashboard with test data."""
    metrics, alerts = generate_test_data(10, 100, 5)
    store = MockStore(metrics)
    return MonitoringDashboard(
        alert_manager=None,
        metric_store=store,
        timeseries_store=store
    )

class TestVisualRegression:
    """Visual regression tests."""
    
    def capture_component(
        self,
        driver: webdriver.Chrome,
        component_id: str,
        state: str
    ) -> Image.Image:
        """Capture screenshot of specific component."""
        # Wait for component to be present and visible
        element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, component_id))
        )
        
        # Scroll element into view
        driver.execute_script("arguments[0].scrollIntoView()", element)
        
        # Wait for any animations to complete
        driver.implicitly_wait(1)
        
        # Get element location and size
        location = element.location_once_scrolled_into_view
        size = element.size
        
        # Take full page screenshot
        png = driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(png))
        
        # Crop to component
        component = screenshot.crop((
            location['x'],
            location['y'],
            location['x'] + size['width'],
            location['y'] + size['height']
        ))
        
        return component
    
    def compare_images(
        self,
        baseline: Image.Image,
        current: Image.Image,
        component: str,
        state: str
    ) -> tuple[bool, dict]:
        """Compare images and generate diff if needed."""
        # Convert to numpy arrays
        baseline_arr = np.array(baseline)
        current_arr = np.array(current)
        
        # Ensure same size
        if baseline_arr.shape != current_arr.shape:
            current_arr = cv2.resize(
                current_arr,
                (baseline_arr.shape[1], baseline_arr.shape[0])
            )
        
        # Calculate pixel difference
        diff = cv2.absdiff(baseline_arr, current_arr)
        pixel_diff = np.count_nonzero(diff) / diff.size
        
        # Calculate structural difference
        baseline_hash = imagehash.average_hash(baseline)
        current_hash = imagehash.average_hash(current)
        struct_diff = baseline_hash - current_hash
        
        # Generate diff visualization if needed
        diff_path = None
        if (pixel_diff > VisualConfig.PIXEL_THRESHOLD or
            struct_diff > VisualConfig.STRUCTURE_THRESHOLD):
            
            # Enhance diff visualization
            diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
            diff_path = (
                VisualConfig.DIFF_DIR /
                f"{component}_{state}_diff.png"
            )
            cv2.imwrite(str(diff_path), diff_color)
        
        results = {
            "pixel_diff": pixel_diff,
            "structural_diff": struct_diff,
            "diff_path": str(diff_path) if diff_path else None
        }
        
        return (
            pixel_diff <= VisualConfig.PIXEL_THRESHOLD and
            struct_diff <= VisualConfig.STRUCTURE_THRESHOLD
        ), results
    
    def save_baseline(
        self,
        image: Image.Image,
        component: str,
        state: str
    ):
        """Save baseline image."""
        path = VisualConfig.BASELINE_DIR / f"{component}_{state}.png"
        image.save(path)
    
    def save_current(
        self,
        image: Image.Image,
        component: str,
        state: str
    ):
        """Save current image."""
        path = VisualConfig.CURRENT_DIR / f"{component}_{state}.png"
        image.save(path)
    
    @pytest.mark.parametrize("component", VisualConfig.COMPONENTS)
    @pytest.mark.parametrize("state", VisualConfig.STATES)
    def test_component_appearance(
        self,
        selenium_driver,
        dashboard,
        setup_dirs,
        component: str,
        state: str
    ):
        """Test appearance of dashboard components."""
        # Start dashboard
        dashboard.run(port=8054)
        selenium_driver.get("http://localhost:8054")
        
        # Set up test state
        self.setup_test_state(selenium_driver, state)
        
        # Capture current appearance
        current = self.capture_component(selenium_driver, component, state)
        self.save_current(current, component, state)
        
        # Get baseline image
        baseline_path = (
            VisualConfig.BASELINE_DIR /
            f"{component}_{state}.png"
        )
        
        if not baseline_path.exists():
            self.save_baseline(current, component, state)
            pytest.skip(f"Baseline created for {component} in {state} state")
        
        baseline = Image.open(baseline_path)
        
        # Compare images
        match, results = self.compare_images(
            baseline,
            current,
            component,
            state
        )
        
        # Log results
        self.log_comparison_results(component, state, results)
        
        assert match, (
            f"Visual mismatch for {component} in {state} state.\n"
            f"Pixel diff: {results['pixel_diff']:.2%}\n"
            f"Structural diff: {results['structural_diff']}\n"
            f"Diff image: {results['diff_path']}"
        )
    
    def test_responsive_layout(
        self,
        selenium_driver,
        dashboard,
        setup_dirs
    ):
        """Test responsive layout at different viewport sizes."""
        dashboard.run(port=8055)
        
        # Test viewport sizes
        viewports = [
            (375, 667),   # Mobile
            (768, 1024),  # Tablet
            (1920, 1080)  # Desktop
        ]
        
        for width, height in viewports:
            selenium_driver.set_window_size(width, height)
            selenium_driver.get("http://localhost:8055")
            
            # Capture full page
            current = Image.open(io.BytesIO(
                selenium_driver.get_screenshot_as_png()
            ))
            
            # Compare with baseline
            baseline_path = (
                VisualConfig.BASELINE_DIR /
                f"responsive_{width}x{height}.png"
            )
            
            if not baseline_path.exists():
                current.save(baseline_path)
                continue
            
            baseline = Image.open(baseline_path)
            match, results = self.compare_images(
                baseline,
                current,
                f"responsive_{width}x{height}",
                "default"
            )
            
            assert match, (
                f"Layout mismatch at {width}x{height}\n"
                f"Pixel diff: {results['pixel_diff']:.2%}\n"
                f"Structural diff: {results['structural_diff']}\n"
                f"Diff image: {results['diff_path']}"
            )
    
    def test_animation_transitions(
        self,
        selenium_driver,
        dashboard,
        setup_dirs
    ):
        """Test animated transitions."""
        dashboard.run(port=8056)
        selenium_driver.get("http://localhost:8056")
        
        # Capture animation frames
        frames = []
        start_time = time.time()
        
        # Click refresh to trigger animation
        refresh_button = selenium_driver.find_element(By.ID, "refresh-button")
        refresh_button.click()
        
        # Capture frames for 1 second
        while time.time() - start_time < 1.0:
            frames.append(Image.open(io.BytesIO(
                selenium_driver.get_screenshot_as_png()
            )))
            time.sleep(0.05)
        
        # Compare consecutive frames
        for i in range(len(frames) - 1):
            match, results = self.compare_images(
                frames[i],
                frames[i + 1],
                "animation",
                f"frame_{i}"
            )
            
            # Some difference is expected between frames
            assert results["pixel_diff"] < 0.5, (
                f"Excessive change between animation frames: "
                f"{results['pixel_diff']:.2%}"
            )
    
    @staticmethod
    def setup_test_state(driver: webdriver.Chrome, state: str):
        """Set up specific test state."""
        if state == "with_alerts":
            # Trigger some alerts
            pass
        elif state == "error_state":
            # Trigger error condition
            pass
        elif state == "loading_state":
            # Trigger loading state
            pass
    
    @staticmethod
    def log_comparison_results(
        component: str,
        state: str,
        results: dict
    ):
        """Log comparison results."""
        with open("visual_regression_results.json", "a") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "component": component,
                "state": state,
                "results": results
            }, f)
            f.write("\n")

def update_baselines():
    """Update baseline images."""
    if VisualConfig.BASELINE_DIR.exists():
        shutil.rmtree(VisualConfig.BASELINE_DIR)
    
    # Run tests to generate new baselines
    pytest.main([
        __file__,
        "-v",
        "--force-baseline"
    ])

if __name__ == "__main__":
    update_baselines()
