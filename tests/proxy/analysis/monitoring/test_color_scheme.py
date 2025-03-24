"""Color scheme tests for monitoring dashboard."""

import pytest
import colorsys
from typing import List, Dict, Tuple
import numpy as np
from colour import Color
import json
from pathlib import Path
from bs4 import BeautifulSoup
import re
import logging

from proxy.analysis.monitoring.visualization import MonitoringDashboard
from .benchmark_visualization import generate_test_data, MockStore

logger = logging.getLogger(__name__)

class ColorSchemeConfig:
    """Configuration for color scheme testing."""
    
    # Brand colors
    PRIMARY_COLOR = "#1976D2"
    SECONDARY_COLOR = "#424242"
    
    # Alert colors
    INFO_COLOR = "#36a64f"
    WARNING_COLOR = "#ffcc00"
    ERROR_COLOR = "#ff0000"
    CRITICAL_COLOR = "#7b0000"
    
    # Chart colors
    CHART_COLORS = [
        "#2196F3",  # Blue
        "#4CAF50",  # Green
        "#FFC107",  # Amber
        "#F44336",  # Red
        "#9C27B0",  # Purple
        "#00BCD4",  # Cyan
    ]
    
    # Contrast requirements
    MIN_CONTRAST_NORMAL = 4.5
    MIN_CONTRAST_LARGE = 3.0
    
    # Color blindness simulation types
    COLOR_BLIND_TYPES = [
        "protanopia",
        "deuteranopia",
        "tritanopia"
    ]
    
    # Maximum similar colors in sequence
    MAX_SIMILAR_COLORS = 2
    SIMILARITY_THRESHOLD = 0.1

class ColorTester:
    """Utility class for color testing."""
    
    @staticmethod
    def get_relative_luminance(color: str) -> float:
        """Calculate relative luminance of color."""
        # Convert hex to RGB
        color = Color(color)
        r, g, b = color.rgb
        
        # Convert to sRGB
        r = ColorTester._to_srgb(r)
        g = ColorTester._to_srgb(g)
        b = ColorTester._to_srgb(b)
        
        # Calculate luminance
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    
    @staticmethod
    def _to_srgb(value: float) -> float:
        """Convert linear RGB to sRGB."""
        if value <= 0.03928:
            return value / 12.92
        return ((value + 0.055) / 1.055) ** 2.4
    
    @staticmethod
    def calculate_contrast_ratio(color1: str, color2: str) -> float:
        """Calculate contrast ratio between two colors."""
        l1 = ColorTester.get_relative_luminance(color1)
        l2 = ColorTester.get_relative_luminance(color2)
        
        # Ensure lighter color is l1
        if l2 > l1:
            l1, l2 = l2, l1
        
        return (l1 + 0.05) / (l2 + 0.05)
    
    @staticmethod
    def simulate_color_blindness(
        color: str,
        type_: str
    ) -> str:
        """Simulate how color appears with color blindness."""
        # Convert hex to RGB
        c = Color(color)
        rgb = np.array(c.rgb)
        
        # Transformation matrices for different types
        matrices = {
            "protanopia": np.array([
                [0.567, 0.433, 0],
                [0.558, 0.442, 0],
                [0, 0.242, 0.758]
            ]),
            "deuteranopia": np.array([
                [0.625, 0.375, 0],
                [0.7, 0.3, 0],
                [0, 0.3, 0.7]
            ]),
            "tritanopia": np.array([
                [0.95, 0.05, 0],
                [0, 0.433, 0.567],
                [0, 0.475, 0.525]
            ])
        }
        
        # Transform color
        transformed = matrices[type_] @ rgb
        
        # Convert back to hex
        return Color(rgb=tuple(transformed)).hex_l
    
    @staticmethod
    def get_color_difference(color1: str, color2: str) -> float:
        """Calculate perceptual difference between colors."""
        # Convert to Lab color space
        c1 = Color(color1)
        c2 = Color(color2)
        
        # Calculate Euclidean distance in Lab space
        lab1 = ColorTester._rgb_to_lab(c1.rgb)
        lab2 = ColorTester._rgb_to_lab(c2.rgb)
        
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(lab1, lab2)))
    
    @staticmethod
    def _rgb_to_lab(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert RGB to Lab color space."""
        # Convert to XYZ
        xyz = ColorTester._rgb_to_xyz(rgb)
        
        # Convert XYZ to Lab
        # Using D65 illuminant reference values
        xn, yn, zn = 95.047, 100.0, 108.883
        
        def f(t):
            if t > (6/29) ** 3:
                return t ** (1/3)
            return (1/3) * (29/6) ** 2 * t + 4/29
        
        l = 116 * f(xyz[1]/yn) - 16
        a = 500 * (f(xyz[0]/xn) - f(xyz[1]/yn))
        b = 200 * (f(xyz[1]/yn) - f(xyz[2]/zn))
        
        return (l, a, b)
    
    @staticmethod
    def _rgb_to_xyz(rgb: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Convert RGB to XYZ color space."""
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        return tuple(matrix @ np.array(rgb))

class TestColorScheme:
    """Test color scheme compliance."""
    
    def test_brand_colors(self, dashboard):
        """Test brand color usage."""
        html = str(dashboard.app.layout)
        soup = BeautifulSoup(html, "html.parser")
        
        # Find elements with brand colors
        primary_elements = soup.find_all(
            style=re.compile(ColorSchemeConfig.PRIMARY_COLOR)
        )
        secondary_elements = soup.find_all(
            style=re.compile(ColorSchemeConfig.SECONDARY_COLOR)
        )
        
        # Verify consistent usage
        for element in primary_elements:
            assert "header" in element.get("class", []) or \
                   "primary" in element.get("class", []), \
                   "Primary color used incorrectly"
        
        for element in secondary_elements:
            assert "secondary" in element.get("class", []) or \
                   "footer" in element.get("class", []), \
                   "Secondary color used incorrectly"
    
    def test_alert_color_contrast(self):
        """Test alert color accessibility."""
        background = "#FFFFFF"  # Assuming white background
        
        alert_colors = {
            "info": ColorSchemeConfig.INFO_COLOR,
            "warning": ColorSchemeConfig.WARNING_COLOR,
            "error": ColorSchemeConfig.ERROR_COLOR,
            "critical": ColorSchemeConfig.CRITICAL_COLOR
        }
        
        for name, color in alert_colors.items():
            contrast = ColorTester.calculate_contrast_ratio(color, background)
            assert contrast >= ColorSchemeConfig.MIN_CONTRAST_NORMAL, \
                f"{name} alert color has insufficient contrast: {contrast:.2f}"
    
    def test_chart_color_differentiation(self):
        """Test chart color distinctiveness."""
        colors = ColorSchemeConfig.CHART_COLORS
        
        # Test regular vision
        for i, color1 in enumerate(colors):
            for j, color2 in enumerate(colors[i+1:], i+1):
                diff = ColorTester.get_color_difference(color1, color2)
                assert diff >= ColorSchemeConfig.SIMILARITY_THRESHOLD, \
                    f"Colors too similar: {color1} and {color2}"
        
        # Test color blindness
        for type_ in ColorSchemeConfig.COLOR_BLIND_TYPES:
            transformed = [
                ColorTester.simulate_color_blindness(c, type_)
                for c in colors
            ]
            
            for i, color1 in enumerate(transformed):
                for j, color2 in enumerate(transformed[i+1:], i+1):
                    diff = ColorTester.get_color_difference(color1, color2)
                    assert diff >= ColorSchemeConfig.SIMILARITY_THRESHOLD, \
                        f"Colors too similar under {type_}: {color1} and {color2}"
    
    def test_sequential_color_contrast(self):
        """Test adjacent color contrast in sequences."""
        colors = ColorSchemeConfig.CHART_COLORS
        
        similar_count = 0
        for i in range(len(colors) - 1):
            diff = ColorTester.get_color_difference(
                colors[i],
                colors[i + 1]
            )
            
            if diff < ColorSchemeConfig.SIMILARITY_THRESHOLD:
                similar_count += 1
            else:
                similar_count = 0
            
            assert similar_count <= ColorSchemeConfig.MAX_SIMILAR_COLORS, \
                "Too many similar colors in sequence"
    
    def test_color_scheme_export(self, dashboard):
        """Test color scheme export and documentation."""
        scheme = {
            "brand": {
                "primary": ColorSchemeConfig.PRIMARY_COLOR,
                "secondary": ColorSchemeConfig.SECONDARY_COLOR
            },
            "alerts": {
                "info": ColorSchemeConfig.INFO_COLOR,
                "warning": ColorSchemeConfig.WARNING_COLOR,
                "error": ColorSchemeConfig.ERROR_COLOR,
                "critical": ColorSchemeConfig.CRITICAL_COLOR
            },
            "charts": {
                "palette": ColorSchemeConfig.CHART_COLORS
            },
            "accessibility": {
                "min_contrast_normal": ColorSchemeConfig.MIN_CONTRAST_NORMAL,
                "min_contrast_large": ColorSchemeConfig.MIN_CONTRAST_LARGE
            }
        }
        
        # Export color scheme
        with open("color_scheme.json", "w") as f:
            json.dump(scheme, f, indent=2)
        
        # Generate color samples
        self._generate_color_samples(scheme)
    
    def _generate_color_samples(self, scheme: Dict):
        """Generate color sample images."""
        import matplotlib.pyplot as plt
        
        # Create color swatches
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Brand colors
        brand_colors = list(scheme["brand"].items())
        axes[0].bar(
            range(len(brand_colors)),
            [1] * len(brand_colors),
            color=[c[1] for c in brand_colors]
        )
        axes[0].set_title("Brand Colors")
        axes[0].set_xticks(range(len(brand_colors)))
        axes[0].set_xticklabels([c[0] for c in brand_colors])
        
        # Alert colors
        alert_colors = list(scheme["alerts"].items())
        axes[1].bar(
            range(len(alert_colors)),
            [1] * len(alert_colors),
            color=[c[1] for c in alert_colors]
        )
        axes[1].set_title("Alert Colors")
        axes[1].set_xticks(range(len(alert_colors)))
        axes[1].set_xticklabels([c[0] for c in alert_colors])
        
        # Chart colors
        chart_colors = scheme["charts"]["palette"]
        axes[2].bar(
            range(len(chart_colors)),
            [1] * len(chart_colors),
            color=chart_colors
        )
        axes[2].set_title("Chart Colors")
        
        plt.tight_layout()
        plt.savefig("color_samples.png")
        plt.close()

def analyze_color_scheme():
    """Run color scheme analysis and generate report."""
    import pytest
    import json
    from datetime import datetime
    
    # Run tests
    results = pytest.main([__file__, "-v", "--json=color_results.json"])
    
    # Load results
    with open("color_results.json") as f:
        data = json.load(f)
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": data["summary"]["total"],
            "passed": data["summary"]["passed"],
            "failed": data["summary"]["failed"]
        },
        "color_blind_analysis": {}
    }
    
    # Add color blindness analysis
    for type_ in ColorSchemeConfig.COLOR_BLIND_TYPES:
        transformed_colors = {
            name: ColorTester.simulate_color_blindness(color, type_)
            for name, color in {
                **{"brand_" + k: v for k, v in ColorSchemeConfig.brand.items()},
                **ColorSchemeConfig.alerts
            }.items()
        }
        report["color_blind_analysis"][type_] = transformed_colors
    
    # Save report
    with open("color_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return results

if __name__ == "__main__":
    analyze_color_scheme()
