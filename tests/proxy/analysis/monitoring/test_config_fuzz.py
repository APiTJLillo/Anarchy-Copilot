"""Fuzz testing for configuration validation."""

import pytest
import random
import string
from pathlib import Path
from typing import Any, Dict, List
from datetime import timedelta
import json
import hypothesis
from hypothesis import given, strategies as st
import tempfile
import sys
from copy import deepcopy

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

# Custom Hypothesis strategies
@st.composite
def random_string_strategy(draw) -> str:
    """Generate random strings with various characteristics."""
    return draw(st.one_of(
        st.text(min_size=1),
        st.text(alphabet=string.ascii_letters),
        st.text(alphabet=string.punctuation),
        st.text(alphabet=string.whitespace),
        st.binary().map(lambda x: x.decode('utf-8', errors='ignore'))
    ))

@st.composite
def random_color_strategy(draw) -> str:
    """Generate random color-like strings."""
    return draw(st.one_of(
        st.from_regex(r"#[0-9A-Fa-f]{6}"),
        st.from_regex(r"rgb\(\d{1,3},\d{1,3},\d{1,3}\)"),
        st.from_regex(r"rgba\(\d{1,3},\d{1,3},\d{1,3},\d\.?\d*\)"),
        st.sampled_from([
            "red", "green", "blue", "yellow", "purple",
            "white", "black", "orange", "invalid-color"
        ])
    ))

@st.composite
def random_config_strategy(draw) -> Dict[str, Any]:
    """Generate random configuration dictionaries."""
    return {
        "style": {
            "color_scheme": draw(st.sampled_from([e.value for e in ColorScheme] + ["invalid"])),
            "background_color": draw(random_color_strategy()),
            "font_family": draw(random_string_strategy()),
            "font_size": draw(st.integers(min_value=-100, max_value=100)),
            "title_size": draw(st.integers(min_value=-100, max_value=100)),
            "marker_size": draw(st.integers(min_value=-100, max_value=100)),
            "line_width": draw(st.integers(min_value=-100, max_value=100)),
            "opacity": draw(st.floats(min_value=-10, max_value=10)),
            "transition_easing": draw(random_string_strategy())
        },
        "playback": {
            "frame_duration": draw(st.integers(min_value=-1000, max_value=10000)),
            "transition_duration": draw(st.integers(min_value=-1000, max_value=10000)),
            "loop": draw(st.booleans()),
            "mode": draw(st.sampled_from(["immediate", "afterall", "invalid"])),
            "direction": draw(st.sampled_from(["forward", "reverse", "invalid"])),
            "redraw": draw(st.booleans())
        },
        "interaction": {
            "draggable_nodes": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "zoomable": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "hoverable": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "selectable": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "showgrid": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "showlegend": draw(st.one_of(st.booleans(), st.just("invalid"))),
            "hovermode": draw(st.sampled_from(["closest", "x", "y", False, "invalid"])),
            "dragmode": draw(st.sampled_from(["pan", "select", "lasso", False, "invalid"]))
        },
        "enabled_types": draw(st.lists(
            st.sampled_from([e.value for e in AnimationType] + ["invalid"]),
            min_size=0,
            max_size=10
        ))
    }

class TestFuzzing:
    """Fuzz testing for configuration validation."""
    
    @given(random_config_strategy())
    def test_config_validation_fuzzing(self, config_dict):
        """Test validation with fuzzed configuration."""
        try:
            # Attempt to create and validate config
            config = self._dict_to_config(config_dict)
            result = validate_animation_config(config)
            
            # Validation should always return a result
            assert isinstance(result, ValidationResult)
            
            # Result should contain either errors or be valid
            assert (not result.valid and result.errors) or (result.valid and not result.errors)
            
            # Check error formatting
            validator = ConfigValidator()
            formatted = validator.format_validation_results(result)
            assert isinstance(formatted, str)
            assert len(formatted) > 0
            
        except Exception as e:
            pytest.fail(f"Unexpected exception during validation: {e}")
    
    @given(st.lists(random_string_strategy(), min_size=0, max_size=100))
    def test_enabled_types_fuzzing(self, types):
        """Test validation with fuzzed animation types."""
        config = AnimationConfig()
        config.enabled_types = types
        
        result = validate_animation_config(config)
        assert isinstance(result, ValidationResult)
    
    @given(random_color_strategy())
    def test_color_validation_fuzzing(self, color):
        """Test validation with fuzzed colors."""
        config = AnimationConfig()
        config.style.background_color = color
        
        result = validate_animation_config(config)
        assert isinstance(result, ValidationResult)
    
    @given(st.integers(min_value=-sys.maxsize, max_value=sys.maxsize))
    def test_duration_fuzzing(self, duration):
        """Test validation with extreme duration values."""
        config = AnimationConfig()
        config.playback.frame_duration = duration
        
        result = validate_animation_config(config)
        assert isinstance(result, ValidationResult)
    
    @hypothesis.settings(deadline=None)
    @given(st.binary(min_size=0, max_size=10000))
    def test_corrupt_file_fuzzing(self, data):
        """Test loading corrupted configuration files."""
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as f:
            f.write(data)
            path = Path(f.name)
        
        try:
            # Attempt to load corrupt config
            try:
                config = AnimationConfig.load(path)
            except Exception as e:
                # Loading may fail, but shouldn't crash
                assert isinstance(e, (json.JSONDecodeError, ValueError))
            
        finally:
            path.unlink()
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AnimationConfig:
        """Convert dictionary to AnimationConfig."""
        style = AnimationStyle(**config_dict["style"])
        playback = PlaybackConfig(**config_dict["playback"])
        interaction = InteractionConfig(**config_dict["interaction"])
        
        return AnimationConfig(
            style=style,
            playback=playback,
            interaction=interaction,
            enabled_types=config_dict["enabled_types"]
        )

class TestEdgeCases:
    """Test configuration validation edge cases."""
    
    def test_empty_config(self):
        """Test validation with minimal config."""
        config = AnimationConfig()
        result = validate_animation_config(config)
        assert isinstance(result, ValidationResult)
    
    def test_null_values(self):
        """Test validation with null values."""
        config = AnimationConfig()
        config.style.font_family = None
        config.playback.mode = None
        config.interaction.hovermode = None
        
        result = validate_animation_config(config)
        assert not result.valid
    
    def test_unicode_values(self):
        """Test validation with unicode values."""
        config = AnimationConfig()
        config.style.font_family = "测试字体"
        config.style.background_color = "红色"
        
        result = validate_animation_config(config)
        assert not result.valid
    
    def test_extreme_values(self):
        """Test validation with extreme values."""
        config = AnimationConfig()
        config.style.font_size = sys.maxsize
        config.style.marker_size = -sys.maxsize
        config.style.opacity = float('inf')
        
        result = validate_animation_config(config)
        assert not result.valid
    
    def test_recursive_config(self):
        """Test validation with recursive structures."""
        config = AnimationConfig()
        recursive_dict = {}
        recursive_dict["self"] = recursive_dict
        
        with pytest.raises(Exception):
            config.style.background_color = recursive_dict
            validate_animation_config(config)
    
    def test_memory_limits(self):
        """Test validation with large configurations."""
        config = AnimationConfig()
        large_string = "x" * (10 ** 6)  # 1MB string
        config.style.font_family = large_string
        
        result = validate_animation_config(config)
        assert not result.valid

class TestFuzzPerformance:
    """Test configuration validation performance."""
    
    @pytest.mark.benchmark
    def test_validation_performance(self, benchmark):
        """Benchmark validation performance."""
        config = AnimationConfig()
        
        def run_validation():
            return validate_animation_config(config)
        
        result = benchmark(run_validation)
        assert isinstance(result, ValidationResult)
    
    @pytest.mark.benchmark
    def test_large_config_performance(self, benchmark):
        """Benchmark validation with large configuration."""
        config = AnimationConfig()
        config.enabled_types = [AnimationType.CASCADE] * 1000
        
        def run_validation():
            return validate_animation_config(config)
        
        result = benchmark(run_validation)
        assert isinstance(result, ValidationResult)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
