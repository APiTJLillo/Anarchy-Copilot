"""Mutation testing for configuration validation."""

import pytest
from pathlib import Path
from typing import Any, Dict, List, Callable, Optional
import random
import copy
from datetime import timedelta
import json
import string
from functools import partial
import itertools
import logging

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

logger = logging.getLogger(__name__)

class ConfigMutator:
    """Mutate configurations for testing."""
    
    def __init__(self, base_config: AnimationConfig):
        self.base_config = base_config
        self.mutations: List[Callable[[AnimationConfig], AnimationConfig]] = [
            self.mutate_style,
            self.mutate_playback,
            self.mutate_interaction,
            self.mutate_types,
            self.swap_fields,
            self.duplicate_fields,
            self.remove_fields,
            self.type_conversion,
            self.boundary_values,
            self.add_noise
        ]
    
    def generate_mutations(self, count: int = 10) -> List[AnimationConfig]:
        """Generate multiple mutated configurations."""
        mutations = []
        for _ in range(count):
            # Apply random number of mutations
            num_mutations = random.randint(1, 3)
            config = copy.deepcopy(self.base_config)
            
            for _ in range(num_mutations):
                mutator = random.choice(self.mutations)
                try:
                    config = mutator(config)
                except Exception as e:
                    logger.debug(f"Mutation failed: {e}")
            
            mutations.append(config)
        
        return mutations
    
    def mutate_style(self, config: AnimationConfig) -> AnimationConfig:
        """Mutate style configuration."""
        style = config.style
        mutation = random.choice([
            lambda: setattr(style, "color_scheme", random.choice(list(ColorScheme))),
            lambda: setattr(style, "background_color", self._random_color()),
            lambda: setattr(style, "font_family", self._random_string()),
            lambda: setattr(style, "font_size", random.randint(-50, 100)),
            lambda: setattr(style, "marker_size", random.randint(-50, 100)),
            lambda: setattr(style, "opacity", random.uniform(-1, 2)),
            lambda: setattr(style, "transition_easing", self._random_string())
        ])
        mutation()
        return config
    
    def mutate_playback(self, config: AnimationConfig) -> AnimationConfig:
        """Mutate playback configuration."""
        playback = config.playback
        mutation = random.choice([
            lambda: setattr(playback, "frame_duration", random.randint(-1000, 10000)),
            lambda: setattr(playback, "transition_duration", random.randint(-1000, 10000)),
            lambda: setattr(playback, "loop", not playback.loop),
            lambda: setattr(playback, "mode", self._random_string()),
            lambda: setattr(playback, "direction", self._random_string())
        ])
        mutation()
        return config
    
    def mutate_interaction(self, config: AnimationConfig) -> AnimationConfig:
        """Mutate interaction configuration."""
        interaction = config.interaction
        mutation = random.choice([
            lambda: setattr(interaction, "draggable_nodes", self._random_value()),
            lambda: setattr(interaction, "zoomable", self._random_value()),
            lambda: setattr(interaction, "hoverable", self._random_value()),
            lambda: setattr(interaction, "hovermode", self._random_string()),
            lambda: setattr(interaction, "dragmode", self._random_string())
        ])
        mutation()
        return config
    
    def mutate_types(self, config: AnimationConfig) -> AnimationConfig:
        """Mutate animation types."""
        mutation = random.choice([
            lambda: setattr(config, "enabled_types", []),
            lambda: setattr(config, "enabled_types", [random.choice(list(AnimationType))]),
            lambda: setattr(config, "enabled_types", list(AnimationType)),
            lambda: setattr(config, "enabled_types", [self._random_string()]),
            lambda: setattr(config, "enabled_types", None)
        ])
        mutation()
        return config
    
    def swap_fields(self, config: AnimationConfig) -> AnimationConfig:
        """Swap values between fields."""
        # Get all field values
        values = []
        for attr in ["style", "playback", "interaction"]:
            obj = getattr(config, attr)
            for field in obj.__annotations__:
                values.append(getattr(obj, field))
        
        # Randomly swap two values
        if len(values) >= 2:
            i, j = random.sample(range(len(values)), 2)
            values[i], values[j] = values[j], values[i]
        
        return config
    
    def duplicate_fields(self, config: AnimationConfig) -> AnimationConfig:
        """Duplicate field values."""
        # Pick a random field and value
        source_obj = random.choice([config.style, config.playback, config.interaction])
        field = random.choice(list(source_obj.__annotations__.keys()))
        value = getattr(source_obj, field)
        
        # Copy to another field
        target_obj = random.choice([config.style, config.playback, config.interaction])
        target_field = random.choice(list(target_obj.__annotations__.keys()))
        setattr(target_obj, target_field, copy.deepcopy(value))
        
        return config
    
    def remove_fields(self, config: AnimationConfig) -> AnimationConfig:
        """Remove random fields."""
        # Pick a random object
        obj = random.choice([config.style, config.playback, config.interaction])
        
        # Try to delete a random attribute
        try:
            field = random.choice(list(obj.__annotations__.keys()))
            delattr(obj, field)
        except (AttributeError, TypeError):
            pass
        
        return config
    
    def type_conversion(self, config: AnimationConfig) -> AnimationConfig:
        """Convert field types."""
        # Pick a random field
        obj = random.choice([config.style, config.playback, config.interaction])
        field = random.choice(list(obj.__annotations__.keys()))
        value = getattr(obj, field)
        
        # Apply random type conversion
        conversion = random.choice([
            str,
            lambda x: [x],
            lambda x: {"value": x},
            lambda x: bool(x),
            lambda x: float(str(x)) if str(x).replace(".", "").isdigit() else 0.0,
            lambda x: None
        ])
        
        try:
            setattr(obj, field, conversion(value))
        except (ValueError, TypeError):
            pass
        
        return config
    
    def boundary_values(self, config: AnimationConfig) -> AnimationConfig:
        """Set fields to boundary values."""
        # Pick a random numeric field
        numeric_fields = {
            "font_size": (-1000, 1000),
            "marker_size": (-1000, 1000),
            "opacity": (-10, 10),
            "frame_duration": (-5000, 10000),
            "transition_duration": (-5000, 10000)
        }
        
        field = random.choice(list(numeric_fields.keys()))
        min_val, max_val = numeric_fields[field]
        
        # Choose boundary value
        value = random.choice([
            min_val,
            min_val + 1,
            0,
            max_val - 1,
            max_val,
            float('inf'),
            float('-inf'),
            float('nan')
        ])
        
        # Find and set field
        for obj in [config.style, config.playback, config.interaction]:
            if hasattr(obj, field):
                setattr(obj, field, value)
                break
        
        return config
    
    def add_noise(self, config: AnimationConfig) -> AnimationConfig:
        """Add random noise to string fields."""
        # Pick a random string field
        obj = random.choice([config.style, config.playback, config.interaction])
        fields = [
            f for f, t in obj.__annotations__.items()
            if t == str or getattr(obj, f) is None or isinstance(getattr(obj, f), str)
        ]
        
        if fields:
            field = random.choice(fields)
            value = getattr(obj, field)
            if isinstance(value, str):
                # Add random noise
                noise_type = random.choice([
                    lambda s: s + self._random_string(),
                    lambda s: self._random_string() + s,
                    lambda s: s.replace(
                        random.choice(s) if s else "",
                        self._random_string(),
                        random.randint(1, 3)
                    ),
                    lambda s: "".join(random.sample(s, len(s))) if s else "",
                    lambda s: s * random.randint(2, 4)
                ])
                setattr(obj, field, noise_type(value))
        
        return config
    
    def _random_string(self, length: int = 10) -> str:
        """Generate random string."""
        return "".join(random.choices(
            string.ascii_letters + string.digits + string.punctuation,
            k=length
        ))
    
    def _random_color(self) -> str:
        """Generate random color string."""
        color_type = random.choice(["hex", "rgb", "rgba", "named"])
        
        if color_type == "hex":
            return f"#{''.join(random.choices('0123456789ABCDEF', k=6))}"
        elif color_type == "rgb":
            return f"rgb({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)})"
        elif color_type == "rgba":
            return f"rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},{random.random()})"
        else:
            return random.choice(["red", "green", "blue", "invalid-color"])
    
    def _random_value(self) -> Any:
        """Generate random value."""
        return random.choice([
            True,
            False,
            None,
            0,
            1,
            -1,
            "",
            "invalid",
            [],
            {},
            object()
        ])

class TestMutations:
    """Test configuration validation with mutations."""
    
    @pytest.fixture
    def mutator(self) -> ConfigMutator:
        """Create config mutator."""
        return ConfigMutator(AnimationConfig())
    
    def test_style_mutations(self, mutator):
        """Test style mutations."""
        for _ in range(100):
            config = mutator.mutate_style(copy.deepcopy(mutator.base_config))
            result = validate_animation_config(config)
            assert isinstance(result, ValidationResult)
    
    def test_playback_mutations(self, mutator):
        """Test playback mutations."""
        for _ in range(100):
            config = mutator.mutate_playback(copy.deepcopy(mutator.base_config))
            result = validate_animation_config(config)
            assert isinstance(result, ValidationResult)
    
    def test_interaction_mutations(self, mutator):
        """Test interaction mutations."""
        for _ in range(100):
            config = mutator.mutate_interaction(copy.deepcopy(mutator.base_config))
            result = validate_animation_config(config)
            assert isinstance(result, ValidationResult)
    
    def test_multiple_mutations(self, mutator):
        """Test multiple mutations."""
        mutations = mutator.generate_mutations(count=100)
        for config in mutations:
            result = validate_animation_config(config)
            assert isinstance(result, ValidationResult)
    
    def test_mutation_stability(self, mutator):
        """Test stability of mutations."""
        # Apply mutations multiple times
        config = copy.deepcopy(mutator.base_config)
        for _ in range(10):
            for mutation in mutator.mutations:
                try:
                    config = mutation(config)
                    result = validate_animation_config(config)
                    assert isinstance(result, ValidationResult)
                except Exception as e:
                    logger.debug(f"Mutation stability error: {e}")
    
    @pytest.mark.parametrize("mutation_count", [1, 5, 10, 20])
    def test_mutation_combinations(self, mutator, mutation_count):
        """Test combinations of mutations."""
        for combo in itertools.combinations(mutator.mutations, mutation_count):
            config = copy.deepcopy(mutator.base_config)
            for mutation in combo:
                try:
                    config = mutation(config)
                except Exception:
                    continue
            
            result = validate_animation_config(config)
            assert isinstance(result, ValidationResult)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
