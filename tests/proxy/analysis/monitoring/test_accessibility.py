"""Accessibility tests for monitoring dashboard."""

import pytest
from bs4 import BeautifulSoup
import axe_core_py
import json
import asyncio
from typing import List, Dict, Any
from pathlib import Path
import logging
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from proxy.analysis.monitoring.visualization import MonitoringDashboard
from .benchmark_visualization import generate_test_data, MockStore

logger = logging.getLogger(__name__)

class AccessibilityGuidelines:
    """WCAG 2.1 Guidelines for testing."""
    
    # Contrast requirements
    MIN_CONTRAST_NORMAL = 4.5
    MIN_CONTRAST_LARGE = 3.0
    
    # Interactive element requirements
    MIN_TARGET_SIZE = 44  # pixels
    
    # Color requirements
    COLORBLIND_SAFE_COLORS = [
        "#000000",  # Black
        "#FFFFFF",  # White
        "#FF0000",  # Red
        "#00FF00",  # Green
        "#0000FF",  # Blue
        "#FFFF00",  # Yellow
    ]
    
    # Aria landmark requirements
    REQUIRED_LANDMARKS = [
        "banner",
        "main",
        "navigation",
        "complementary"
    ]

@pytest.fixture
async def dashboard():
    """Create dashboard for testing."""
    metrics, alerts = generate_test_data(10, 100, 5)
    store = MockStore(metrics)
    return MonitoringDashboard(
        alert_manager=None,
        metric_store=store,
        timeseries_store=store
    )

@pytest.fixture
def selenium_driver():
    """Create Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    yield driver
    driver.quit()

class TestAccessibility:
    """Test dashboard accessibility compliance."""

    def test_color_contrast(self, dashboard):
        """Test color contrast ratios."""
        # Get dashboard HTML
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Find all elements with color styles
        elements_with_color = soup.find_all(
            lambda tag: tag.get("style") and "color" in tag["style"]
        )
        
        for element in elements_with_color:
            style = element["style"]
            color = self._extract_color(style)
            background = self._extract_background_color(style)
            
            if color and background:
                contrast_ratio = self._calculate_contrast_ratio(color, background)
                
                # Check font size for contrast requirements
                font_size = self._extract_font_size(style)
                min_contrast = (
                    AccessibilityGuidelines.MIN_CONTRAST_LARGE
                    if font_size and font_size >= 18
                    else AccessibilityGuidelines.MIN_CONTRAST_NORMAL
                )
                
                assert contrast_ratio >= min_contrast, (
                    f"Insufficient contrast ratio {contrast_ratio:.2f} "
                    f"for element {element.name}"
                )
    
    def test_interactive_elements(self, dashboard):
        """Test interactive element accessibility."""
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Check buttons
        buttons = soup.find_all(["button", "a", "input[type='submit']"])
        for button in buttons:
            # Check for aria labels
            assert (
                button.get("aria-label") or
                button.get("aria-labelledby") or
                button.string
            ), f"Button {button} missing accessible label"
            
            # Check target size
            style = button.get("style", "")
            width = self._extract_dimension(style, "width")
            height = self._extract_dimension(style, "height")
            
            if width and height:
                assert width >= AccessibilityGuidelines.MIN_TARGET_SIZE, (
                    f"Button width {width}px too small"
                )
                assert height >= AccessibilityGuidelines.MIN_TARGET_SIZE, (
                    f"Button height {height}px too small"
                )
    
    def test_form_inputs(self, dashboard):
        """Test form input accessibility."""
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Check form inputs
        inputs = soup.find_all("input")
        for input_elem in inputs:
            # Check for labels
            input_id = input_elem.get("id")
            if input_id:
                label = soup.find("label", attrs={"for": input_id})
                assert label, f"Input {input_id} missing label"
            
            # Check for aria attributes
            assert input_elem.get("aria-label") or input_elem.get("aria-labelledby"), (
                f"Input {input_elem} missing aria label"
            )
    
    def test_keyboard_navigation(self, selenium_driver, dashboard):
        """Test keyboard navigation."""
        # Start dashboard
        dashboard.run(port=8051)
        selenium_driver.get("http://localhost:8051")
        
        # Find all focusable elements
        focusable = selenium_driver.find_elements(
            By.CSS_SELECTOR,
            "a, button, input, select, textarea, [tabindex]:not([tabindex='-1'])"
        )
        
        # Check tab order
        current_element = selenium_driver.switch_to.active_element
        for element in focusable:
            # Tab to next element
            current_element.send_keys("\t")
            current_element = selenium_driver.switch_to.active_element
            
            # Verify element is focused
            assert current_element == element, (
                f"Tab order incorrect for element {element.tag_name}"
            )
    
    @pytest.mark.asyncio
    async def test_aria_landmarks(self, dashboard):
        """Test ARIA landmarks."""
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Check for required landmarks
        for landmark in AccessibilityGuidelines.REQUIRED_LANDMARKS:
            elements = soup.find_all(attrs={"role": landmark})
            assert elements, f"Missing required landmark: {landmark}"
    
    def test_axe_core_violations(self, selenium_driver, dashboard):
        """Test for accessibility violations using axe-core."""
        # Start dashboard
        dashboard.run(port=8052)
        selenium_driver.get("http://localhost:8052")
        
        # Run axe-core analysis
        results = axe_core_py.run(selenium_driver)
        
        # Check for violations
        violations = results.get("violations", [])
        if violations:
            # Log detailed violation information
            for violation in violations:
                logger.error(
                    f"Accessibility violation: {violation['help']}\n"
                    f"Impact: {violation['impact']}\n"
                    f"Elements: {violation['nodes']}"
                )
            
            assert not violations, f"Found {len(violations)} accessibility violations"
    
    def test_screen_reader_compatibility(self, dashboard):
        """Test screen reader compatibility."""
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Check images have alt text
        images = soup.find_all("img")
        for img in images:
            assert img.get("alt"), f"Image missing alt text: {img}"
        
        # Check custom controls have proper roles
        custom_controls = soup.find_all(attrs={"role": True})
        for control in custom_controls:
            role = control["role"]
            # Check required aria attributes for role
            required_attrs = self._get_required_aria_attrs(role)
            for attr in required_attrs:
                assert control.get(f"aria-{attr}"), (
                    f"Control with role {role} missing required "
                    f"aria attribute: aria-{attr}"
                )
    
    def test_dynamic_content_updates(self, selenium_driver, dashboard):
        """Test accessibility of dynamic content updates."""
        dashboard.run(port=8053)
        selenium_driver.get("http://localhost:8053")
        
        # Find refresh button
        refresh_button = selenium_driver.find_element(By.ID, "refresh-button")
        
        # Click refresh and check for ARIA live regions
        refresh_button.click()
        WebDriverWait(selenium_driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//*[@aria-live]"))
        )
        
        # Verify live regions are properly configured
        live_regions = selenium_driver.find_elements(
            By.XPATH,
            "//*[@aria-live]"
        )
        for region in live_regions:
            assert region.get_attribute("aria-live") in ["polite", "assertive"]
    
    @staticmethod
    def _extract_color(style: str) -> Optional[str]:
        """Extract color from style string."""
        # Implementation here
        pass
    
    @staticmethod
    def _extract_background_color(style: str) -> Optional[str]:
        """Extract background color from style string."""
        # Implementation here
        pass
    
    @staticmethod
    def _calculate_contrast_ratio(color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors."""
        # Implementation here
        pass
    
    @staticmethod
    def _extract_font_size(style: str) -> Optional[int]:
        """Extract font size from style string."""
        # Implementation here
        pass
    
    @staticmethod
    def _extract_dimension(style: str, property_name: str) -> Optional[int]:
        """Extract dimension value from style string."""
        # Implementation here
        pass
    
    @staticmethod
    def _get_required_aria_attrs(role: str) -> List[str]:
        """Get required ARIA attributes for role."""
        # Role requirements based on WAI-ARIA specification
        requirements = {
            "button": ["pressed"],
            "checkbox": ["checked"],
            "combobox": ["expanded", "controls"],
            "gridcell": ["selected"],
            "link": ["expanded"],
            "menuitem": ["checked"],
            "menuitemcheckbox": ["checked"],
            "menuitemradio": ["checked"],
            "option": ["selected"],
            "progressbar": ["valuenow", "valuemin", "valuemax"],
            "radio": ["checked"],
            "scrollbar": ["controls", "valuenow", "valuemin", "valuemax"],
            "search": ["expanded"],
            "slider": ["valuenow", "valuemin", "valuemax"],
            "spinbutton": ["valuenow", "valuemin", "valuemax"],
            "switch": ["checked"],
            "tab": ["selected"],
            "tabpanel": ["labelledby"],
            "textbox": ["multiline", "readonly"],
            "treeitem": ["selected", "expanded"]
        }
        return requirements.get(role, [])

def run_accessibility_tests():
    """Run all accessibility tests and generate report."""
    import pytest
    import json
    from datetime import datetime
    
    # Run tests
    results = pytest.main([__file__, "-v", "--json=accessibility_results.json"])
    
    # Load results
    with open("accessibility_results.json") as f:
        data = json.load(f)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": data["summary"]["total"],
            "passed": data["summary"]["passed"],
            "failed": data["summary"]["failed"],
            "skipped": data["summary"]["skipped"]
        },
        "violations": []
    }
    
    # Add detailed violation information
    for test in data["tests"]:
        if test["outcome"] == "failed":
            report["violations"].append({
                "name": test["nodeid"],
                "message": test["call"]["longrepr"]
            })
    
    # Save report
    with open("accessibility_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_accessibility_tests()
