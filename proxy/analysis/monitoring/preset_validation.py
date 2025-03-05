"""Validation for visualization presets."""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio
from cerberus import Validator
import jsonschema
from datetime import datetime

from .visualization_presets import PresetConfig, PresetManager
from .visualization_controls import ControlConfig
from .advanced_sensitivity_viz import VisualizationConfig

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for preset validation."""
    strict_mode: bool = True
    auto_fix: bool = False
    validation_log: Optional[Path] = None
    schema_path: Optional[Path] = None
    required_fields: Set[str] = None
    value_ranges: Dict[str, Tuple[float, float]] = None
    allowed_values: Dict[str, List[Any]] = None
    
    def __post_init__(self):
        if self.required_fields is None:
            self.required_fields = {
                "template",
                "colormap",
                "resolution",
                "animation_duration"
            }
        if self.value_ranges is None:
            self.value_ranges = {
                "resolution": (20, 500),
                "animation_duration": (100, 5000),
                "width": (400, 3000),
                "height": (300, 2000)
            }
        if self.allowed_values is None:
            self.allowed_values = {
                "template": list(pio.templates),
                "colormap": [
                    "viridis", "plasma", "inferno", "magma",
                    "RdBu", "RdYlBu", "Spectral", "coolwarm"
                ]
            }

class PresetValidator:
    """Validate visualization presets."""
    
    def __init__(
        self,
        config: ValidationConfig,
        preset_manager: PresetManager
    ):
        self.config = config
        self.preset_manager = preset_manager
        self.validation_results: Dict[str, Dict[str, Any]] = {}
        self.schema = self._load_schema()
        
        self.validator = Validator(self.schema)
    
    def validate_preset(
        self,
        name: str,
        preset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate single preset."""
        try:
            # Check required fields
            missing_fields = self.config.required_fields - set(
                preset.get("settings", {}).keys()
            )
            if missing_fields:
                raise ValueError(
                    f"Missing required fields: {', '.join(missing_fields)}"
                )
            
            # Validate against schema
            settings = preset.get("settings", {})
            if not self.validator.validate(settings):
                if self.config.strict_mode:
                    raise ValueError(
                        f"Schema validation failed: {self.validator.errors}"
                    )
                logger.warning(
                    f"Schema validation issues for {name}: {self.validator.errors}"
                )
            
            # Check value ranges
            range_errors = []
            for field, (min_val, max_val) in self.config.value_ranges.items():
                if field in settings:
                    value = settings[field]
                    if not min_val <= value <= max_val:
                        error = f"{field} value {value} outside range [{min_val}, {max_val}]"
                        range_errors.append(error)
                        
                        # Auto-fix if enabled
                        if self.config.auto_fix:
                            settings[field] = max(min_val, min(value, max_val))
            
            if range_errors and self.config.strict_mode:
                raise ValueError(f"Value range errors: {', '.join(range_errors)}")
            
            # Check allowed values
            value_errors = []
            for field, allowed in self.config.allowed_values.items():
                if field in settings and settings[field] not in allowed:
                    error = f"Invalid {field} value: {settings[field]}"
                    value_errors.append(error)
                    
                    # Auto-fix if enabled
                    if self.config.auto_fix and allowed:
                        settings[field] = allowed[0]
            
            if value_errors and self.config.strict_mode:
                raise ValueError(f"Invalid values: {', '.join(value_errors)}")
            
            # Store validation result
            result = {
                "valid": not (range_errors or value_errors),
                "errors": {
                    "range_errors": range_errors,
                    "value_errors": value_errors,
                    "schema_errors": self.validator.errors
                },
                "fixed": bool(self.config.auto_fix and (range_errors or value_errors)),
                "timestamp": datetime.now().isoformat()
            }
            
            self.validation_results[name] = result
            self._log_validation(name, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for preset {name}: {e}")
            raise
    
    def validate_all_presets(self) -> Dict[str, Dict[str, Any]]:
        """Validate all presets."""
        results = {}
        
        for name, preset in self.preset_manager.presets.items():
            try:
                results[name] = self.validate_preset(name, preset)
            except Exception as e:
                results[name] = {
                    "valid": False,
                    "errors": {"validation_error": str(e)},
                    "fixed": False,
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        if not self.validation_results:
            return {}
        
        valid_count = sum(
            1 for r in self.validation_results.values()
            if r["valid"]
        )
        total_count = len(self.validation_results)
        
        return {
            "total_presets": total_count,
            "valid_presets": valid_count,
            "invalid_presets": total_count - valid_count,
            "validation_rate": valid_count / total_count if total_count else 0,
            "fixed_presets": sum(
                1 for r in self.validation_results.values()
                if r["fixed"]
            ),
            "error_types": self._summarize_errors(),
            "last_validation": max(
                r["timestamp"] for r in self.validation_results.values()
            )
        }
    
    def get_invalid_presets(self) -> List[str]:
        """Get list of invalid presets."""
        return [
            name
            for name, result in self.validation_results.items()
            if not result["valid"]
        ]
    
    def fix_preset(
        self,
        name: str,
        auto_save: bool = True
    ) -> Dict[str, Any]:
        """Fix invalid preset."""
        try:
            if name not in self.preset_manager.presets:
                raise KeyError(f"Preset not found: {name}")
            
            preset = self.preset_manager.presets[name]
            settings = preset.get("settings", {})
            
            # Fix value ranges
            for field, (min_val, max_val) in self.config.value_ranges.items():
                if field in settings:
                    value = settings[field]
                    if not min_val <= value <= max_val:
                        settings[field] = max(min_val, min(value, max_val))
            
            # Fix invalid values
            for field, allowed in self.config.allowed_values.items():
                if field in settings and settings[field] not in allowed:
                    settings[field] = allowed[0] if allowed else None
            
            # Update preset
            preset["settings"] = settings
            if auto_save:
                self.preset_manager.update_preset(
                    name,
                    settings,
                    description=f"Auto-fixed on {datetime.now().isoformat()}"
                )
            
            # Revalidate
            return self.validate_preset(name, preset)
            
        except Exception as e:
            logger.error(f"Failed to fix preset {name}: {e}")
            raise
    
    def fix_all_presets(self) -> Dict[str, Dict[str, Any]]:
        """Fix all invalid presets."""
        results = {}
        
        for name in self.get_invalid_presets():
            try:
                results[name] = self.fix_preset(name)
            except Exception as e:
                results[name] = {
                    "valid": False,
                    "errors": {"fix_error": str(e)},
                    "fixed": False,
                    "timestamp": datetime.now().isoformat()
                }
        
        return results
    
    def _load_schema(self) -> Dict[str, Any]:
        """Load validation schema."""
        if self.config.schema_path:
            try:
                with open(self.config.schema_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load schema: {e}")
        
        # Default schema
        return {
            "template": {"type": "string"},
            "colormap": {"type": "string"},
            "resolution": {"type": "integer", "min": 20, "max": 500},
            "animation_duration": {"type": "integer", "min": 100, "max": 5000},
            "width": {"type": "integer", "min": 400, "max": 3000},
            "height": {"type": "integer", "min": 300, "max": 2000}
        }
    
    def _log_validation(self, name: str, result: Dict[str, Any]):
        """Log validation result."""
        if self.config.validation_log:
            try:
                with open(self.config.validation_log, "a") as f:
                    json.dump(
                        {
                            "name": name,
                            "timestamp": datetime.now().isoformat(),
                            **result
                        },
                        f
                    )
                    f.write("\n")
            except Exception as e:
                logger.error(f"Failed to log validation: {e}")
    
    def _summarize_errors(self) -> Dict[str, int]:
        """Summarize validation errors."""
        error_counts = defaultdict(int)
        
        for result in self.validation_results.values():
            for error_type, errors in result["errors"].items():
                if errors:
                    if isinstance(errors, list):
                        error_counts[error_type] += len(errors)
                    else:
                        error_counts[error_type] += 1
        
        return dict(error_counts)

def create_preset_validator(
    preset_manager: PresetManager,
    schema_path: Optional[Path] = None,
    validation_log: Optional[Path] = None
) -> PresetValidator:
    """Create preset validator."""
    config = ValidationConfig(
        schema_path=schema_path,
        validation_log=validation_log
    )
    return PresetValidator(config, preset_manager)

if __name__ == "__main__":
    # Example usage
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(
        preset_manager,
        validation_log=Path("validation.log")
    )
    
    # Validate presets
    results = validator.validate_all_presets()
    print(json.dumps(results, indent=2))
    
    # Fix invalid presets
    fixed = validator.fix_all_presets()
    print(json.dumps(fixed, indent=2))
