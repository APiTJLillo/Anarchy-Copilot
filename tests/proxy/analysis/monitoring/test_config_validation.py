"""Tests for configuration validation."""

import pytest
from pathlib import Path
from datetime import timedelta
import json
import tempfile

from proxy.analysis.monitoring.config_validation import (
    ConfigValidator,
    ValidationError,
    ValidationResult,
    validate_animation_config
)
from proxy.analysis.monitoring.animation_config import (
    AnimationConfig,
    AnimationStyle,
    PlaybackConfig,
    InteractionConfig,
    AnimationType,
    ColorScheme
)

@pytest.fixture
def validator() -> ConfigValidator:
    """Create config validator."""
    return ConfigValidator()

@pytest.fixture
def valid_config() -> AnimationConfig:
    """Create valid animation config."""
    return AnimationConfig(
        style=AnimationStyle(
            color_scheme=ColorScheme.VIRIDIS,
            background_color="white",
            font_family="Arial",
            font_size=12,
            title_size=16,
            marker_size=20,
            line_width=2,
            opacity=0.8,
            transition_easing="cubic-in-out"
        ),
        playback=PlaybackConfig(
            frame_duration=500,
            transition_duration=300,
            loop=True,
            mode="immediate",
            direction="forward",
            redraw=True
        ),
        interaction=InteractionConfig(
            draggable_nodes=True,
            zoomable=True,
            hoverable=True,
            selectable=True,
            showgrid=False,
            showlegend=True,
            hovermode="closest",
            dragmode="pan"
        ),
        output_dir=Path("animations"),
        enabled_types=[AnimationType.CASCADE]
    )

class TestStyleValidation:
    """Test animation style validation."""
    
    def test_valid_style(self, validator, valid_config):
        """Test validation of valid style."""
        errors = validator.validate_style(valid_config.style)
        assert not errors
    
    def test_invalid_color_scheme(self, validator, valid_config):
        """Test invalid color scheme."""
        valid_config.style.color_scheme = "invalid"
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "color_scheme" for e in errors)
    
    def test_invalid_background_color(self, validator, valid_config):
        """Test invalid background color."""
        valid_config.style.background_color = "not-a-color"
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "background_color" for e in errors)
    
    def test_invalid_font(self, validator, valid_config):
        """Test invalid font family."""
        valid_config.style.font_family = "NonexistentFont"
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "font_family" for e in errors)
    
    def test_invalid_sizes(self, validator, valid_config):
        """Test invalid size values."""
        # Test font size
        valid_config.style.font_size = 0
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "font_size" for e in errors)
        
        # Test marker size
        valid_config.style.marker_size = 100
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "marker_size" for e in errors)
    
    def test_invalid_opacity(self, validator, valid_config):
        """Test invalid opacity."""
        valid_config.style.opacity = 1.5
        errors = validator.validate_style(valid_config.style)
        assert any(e.field == "opacity" for e in errors)

class TestPlaybackValidation:
    """Test playback configuration validation."""
    
    def test_valid_playback(self, validator, valid_config):
        """Test validation of valid playback config."""
        errors = validator.validate_playback(valid_config.playback)
        assert not errors
    
    def test_invalid_durations(self, validator, valid_config):
        """Test invalid duration values."""
        # Test frame duration
        valid_config.playback.frame_duration = 50
        errors = validator.validate_playback(valid_config.playback)
        assert any(e.field == "frame_duration" for e in errors)
        
        # Test transition duration
        valid_config.playback.transition_duration = 1000
        valid_config.playback.frame_duration = 500
        errors = validator.validate_playback(valid_config.playback)
        assert any(e.field == "transition_duration" for e in errors)
    
    def test_invalid_mode(self, validator, valid_config):
        """Test invalid playback mode."""
        valid_config.playback.mode = "invalid"
        errors = validator.validate_playback(valid_config.playback)
        assert any(e.field == "mode" for e in errors)
    
    def test_invalid_direction(self, validator, valid_config):
        """Test invalid playback direction."""
        valid_config.playback.direction = "sideways"
        errors = validator.validate_playback(valid_config.playback)
        assert any(e.field == "direction" for e in errors)

class TestInteractionValidation:
    """Test interaction configuration validation."""
    
    def test_valid_interaction(self, validator, valid_config):
        """Test validation of valid interaction config."""
        errors = validator.validate_interaction(valid_config.interaction)
        assert not errors
    
    def test_invalid_hover_mode(self, validator, valid_config):
        """Test invalid hover mode."""
        valid_config.interaction.hovermode = "invalid"
        errors = validator.validate_interaction(valid_config.interaction)
        assert any(e.field == "hovermode" for e in errors)
    
    def test_invalid_drag_mode(self, validator, valid_config):
        """Test invalid drag mode."""
        valid_config.interaction.dragmode = "invalid"
        errors = validator.validate_interaction(valid_config.interaction)
        assert any(e.field == "dragmode" for e in errors)

class TestCompleteValidation:
    """Test complete configuration validation."""
    
    def test_valid_config(self, validator, valid_config):
        """Test validation of completely valid config."""
        result = validator.validate_config(valid_config)
        assert result.valid
        assert not result.errors
    
    def test_invalid_animation_types(self, validator, valid_config):
        """Test invalid animation types."""
        valid_config.enabled_types = ["invalid"]
        result = validator.validate_config(valid_config)
        assert not result.valid
        assert any(e.field == "enabled_types" for e in result.errors)
    
    def test_invalid_output_dir(self, validator, valid_config):
        """Test invalid output directory."""
        valid_config.output_dir = "not-a-path"
        result = validator.validate_config(valid_config)
        assert not result.valid
        assert any(e.field == "output_dir" for e in result.errors)
    
    def test_performance_warnings(self, validator, valid_config):
        """Test performance warning detection."""
        valid_config.playback.frame_duration = 150
        result = validator.validate_config(valid_config)
        assert result.valid  # Warnings don't affect validity
        assert any(w.field == "frame_duration" for w in result.warnings)
    
    def test_missing_types_warning(self, validator, valid_config):
        """Test warning for no enabled types."""
        valid_config.enabled_types = []
        result = validator.validate_config(valid_config)
        assert result.valid  # Warnings don't affect validity
        assert any(w.field == "enabled_types" for w in result.warnings)

class TestResultFormatting:
    """Test validation result formatting."""
    
    def test_success_formatting(self, validator, valid_config):
        """Test formatting successful validation."""
        result = validator.validate_config(valid_config)
        formatted = validator.format_validation_results(result)
        assert "successful" in formatted.lower()
    
    def test_error_formatting(self, validator, valid_config):
        """Test formatting validation with errors."""
        valid_config.style.font_size = 0
        result = validator.validate_config(valid_config)
        formatted = validator.format_validation_results(result)
        assert "failed" in formatted.lower()
        assert "font_size" in formatted
        assert "out of range" in formatted
    
    def test_warning_formatting(self, validator, valid_config):
        """Test formatting validation with warnings."""
        valid_config.playback.frame_duration = 150
        result = validator.validate_config(valid_config)
        formatted = validator.format_validation_results(result)
        assert "warning" in formatted.lower()
        assert "performance" in formatted.lower()

@pytest.mark.integration
class TestConfigPersistence:
    """Test configuration persistence and reloading."""
    
    def test_config_roundtrip(self, valid_config):
        """Test saving and reloading configuration."""
        with tempfile.NamedTemporaryFile(suffix=".json") as f:
            # Save config
            valid_config.save(Path(f.name))
            
            # Validate saved config
            with open(f.name) as f2:
                saved_data = json.load(f2)
            
            assert saved_data["style"]["color_scheme"] == valid_config.style.color_scheme.value
            
            # Load and validate config
            loaded_config = AnimationConfig.load(Path(f.name))
            result = validate_animation_config(loaded_config)
            assert result.valid

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
