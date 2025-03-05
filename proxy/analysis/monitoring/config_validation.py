"""Validation for animation configuration."""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Type
import re
import logging
from pathlib import Path

from .animation_config import (
    AnimationConfig,
    AnimationStyle,
    PlaybackConfig,
    InteractionConfig,
    AnimationType,
    ColorScheme
)

logger = logging.getLogger(__name__)

@dataclass
class ValidationError:
    """Error in configuration validation."""
    field: str
    value: Any
    message: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    """Result of configuration validation."""
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]

class ConfigValidator:
    """Validate animation configuration."""
    
    def __init__(self):
        # Valid font families
        self.valid_fonts = {
            "Arial", "Helvetica", "Times New Roman", "Courier New",
            "Verdana", "Georgia", "Roboto", "Open Sans"
        }
        
        # Valid easing functions
        self.valid_easings = {
            "linear", "quad", "cubic", "sin", "exp", "circle", "elastic",
            "back", "bounce", "cubic-in-out"
        }
        
        # Valid color formats
        self.color_regex = re.compile(
            r"^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$|"
            r"^rgb\(\d{1,3},\s*\d{1,3},\s*\d{1,3}\)$|"
            r"^rgba\(\d{1,3},\s*\d{1,3},\s*\d{1,3},\s*[0-1](\.\d+)?\)$|"
            r"^(white|black|red|green|blue|yellow|purple|gray|orange)$"
        )
    
    def validate_style(self, style: AnimationStyle) -> List[ValidationError]:
        """Validate animation style configuration."""
        errors = []
        
        # Validate color scheme
        if not isinstance(style.color_scheme, ColorScheme):
            errors.append(ValidationError(
                "color_scheme",
                style.color_scheme,
                "Invalid color scheme",
                f"Must be one of: {', '.join(s.value for s in ColorScheme)}"
            ))
        
        # Validate background color
        if not self.color_regex.match(style.background_color):
            errors.append(ValidationError(
                "background_color",
                style.background_color,
                "Invalid color format",
                "Use hex (#RRGGBB), rgb(), rgba(), or named color"
            ))
        
        # Validate font
        if style.font_family not in self.valid_fonts:
            errors.append(ValidationError(
                "font_family",
                style.font_family,
                "Unsupported font family",
                f"Use one of: {', '.join(self.valid_fonts)}"
            ))
        
        # Validate sizes
        if not 8 <= style.font_size <= 24:
            errors.append(ValidationError(
                "font_size",
                style.font_size,
                "Font size out of range",
                "Font size should be between 8 and 24"
            ))
        
        if not 12 <= style.title_size <= 32:
            errors.append(ValidationError(
                "title_size",
                style.title_size,
                "Title size out of range",
                "Title size should be between 12 and 32"
            ))
        
        if not 5 <= style.marker_size <= 50:
            errors.append(ValidationError(
                "marker_size",
                style.marker_size,
                "Marker size out of range",
                "Marker size should be between 5 and 50"
            ))
        
        if not 1 <= style.line_width <= 10:
            errors.append(ValidationError(
                "line_width",
                style.line_width,
                "Line width out of range",
                "Line width should be between 1 and 10"
            ))
        
        # Validate opacity
        if not 0 <= style.opacity <= 1:
            errors.append(ValidationError(
                "opacity",
                style.opacity,
                "Opacity out of range",
                "Opacity should be between 0 and 1"
            ))
        
        # Validate easing
        if style.transition_easing not in self.valid_easings:
            errors.append(ValidationError(
                "transition_easing",
                style.transition_easing,
                "Invalid easing function",
                f"Use one of: {', '.join(self.valid_easings)}"
            ))
        
        return errors
    
    def validate_playback(self, playback: PlaybackConfig) -> List[ValidationError]:
        """Validate playback configuration."""
        errors = []
        
        # Validate durations
        if not 100 <= playback.frame_duration <= 5000:
            errors.append(ValidationError(
                "frame_duration",
                playback.frame_duration,
                "Frame duration out of range",
                "Duration should be between 100 and 5000 milliseconds"
            ))
        
        if not 0 <= playback.transition_duration <= playback.frame_duration:
            errors.append(ValidationError(
                "transition_duration",
                playback.transition_duration,
                "Transition duration invalid",
                "Should be between 0 and frame duration"
            ))
        
        # Validate mode
        if playback.mode not in {"immediate", "afterall"}:
            errors.append(ValidationError(
                "mode",
                playback.mode,
                "Invalid playback mode",
                "Use 'immediate' or 'afterall'"
            ))
        
        # Validate direction
        if playback.direction not in {"forward", "reverse"}:
            errors.append(ValidationError(
                "direction",
                playback.direction,
                "Invalid playback direction",
                "Use 'forward' or 'reverse'"
            ))
        
        return errors
    
    def validate_interaction(self, interaction: InteractionConfig) -> List[ValidationError]:
        """Validate interaction configuration."""
        errors = []
        
        # Validate hover mode
        if interaction.hovermode not in {"closest", "x", "y", False}:
            errors.append(ValidationError(
                "hovermode",
                interaction.hovermode,
                "Invalid hover mode",
                "Use 'closest', 'x', 'y', or False"
            ))
        
        # Validate drag mode
        if interaction.dragmode not in {"pan", "select", "lasso", False}:
            errors.append(ValidationError(
                "dragmode",
                interaction.dragmode,
                "Invalid drag mode",
                "Use 'pan', 'select', 'lasso', or False"
            ))
        
        return errors
    
    def validate_config(self, config: AnimationConfig) -> ValidationResult:
        """Validate complete animation configuration."""
        errors = []
        warnings = []
        
        # Validate style
        style_errors = self.validate_style(config.style)
        errors.extend(style_errors)
        
        # Validate playback
        playback_errors = self.validate_playback(config.playback)
        errors.extend(playback_errors)
        
        # Validate interaction
        interaction_errors = self.validate_interaction(config.interaction)
        errors.extend(interaction_errors)
        
        # Validate enabled types
        invalid_types = [
            t for t in config.enabled_types
            if not isinstance(t, AnimationType)
        ]
        if invalid_types:
            errors.append(ValidationError(
                "enabled_types",
                invalid_types,
                "Invalid animation types",
                f"Must be from: {', '.join(t.value for t in AnimationType)}"
            ))
        
        # Validate output directory
        if not isinstance(config.output_dir, Path):
            errors.append(ValidationError(
                "output_dir",
                config.output_dir,
                "Invalid output directory",
                "Must be a Path object"
            ))
        
        # Add warnings for potential issues
        if config.playback.frame_duration < 200:
            warnings.append(ValidationError(
                "frame_duration",
                config.playback.frame_duration,
                "Very short frame duration may cause performance issues",
                "Consider using at least 200ms"
            ))
        
        if not config.enabled_types:
            warnings.append(ValidationError(
                "enabled_types",
                config.enabled_types,
                "No animation types enabled",
                "Enable at least one animation type"
            ))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def format_validation_results(self, results: ValidationResult) -> str:
        """Format validation results as human-readable text."""
        lines = []
        
        if not results.valid:
            lines.append("Configuration validation failed:")
            lines.append("\nErrors:")
            for error in results.errors:
                lines.append(f"- {error.field}: {error.message}")
                if error.suggestion:
                    lines.append(f"  Suggestion: {error.suggestion}")
        
        if results.warnings:
            lines.append("\nWarnings:")
            for warning in results.warnings:
                lines.append(f"- {warning.field}: {warning.message}")
                if warning.suggestion:
                    lines.append(f"  Suggestion: {warning.suggestion}")
        
        if results.valid and not results.warnings:
            lines.append("Configuration validation successful.")
        
        return "\n".join(lines)

def validate_animation_config(config: AnimationConfig) -> ValidationResult:
    """Validate animation configuration."""
    validator = ConfigValidator()
    return validator.validate_config(config)

if __name__ == "__main__":
    # Example usage
    config = AnimationConfig()
    validator = ConfigValidator()
    results = validator.validate_config(config)
    print(validator.format_validation_results(results))
