"""Preset configurations for sensitivity visualizations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from pathlib import Path
import json
import logging
import yaml
from datetime import datetime
from collections import defaultdict

from .visualization_controls import ControlConfig, VisualizationControls
from .advanced_sensitivity_viz import VisualizationConfig

logger = logging.getLogger(__name__)

@dataclass
class PresetConfig:
    """Configuration for visualization presets."""
    preset_dir: Path
    auto_save: bool = True
    max_history: int = 10
    backup_dir: Optional[Path] = None
    metadata: bool = True
    categories: List[str] = None
    
    def __post_init__(self):
        if self.categories is None:
            self.categories = [
                "Default",
                "Publication",
                "Presentation",
                "Analysis",
                "Custom"
            ]

class PresetManager:
    """Manage visualization presets."""
    
    def __init__(self, config: PresetConfig):
        self.config = config
        self.presets: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []
        
        # Create directories
        self.config.preset_dir.mkdir(parents=True, exist_ok=True)
        if self.config.backup_dir:
            self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self._load_presets()
    
    def save_preset(
        self,
        name: str,
        settings: Dict[str, Any],
        category: str = "Custom",
        description: Optional[str] = None
    ):
        """Save visualization settings as preset."""
        try:
            if category not in self.config.categories:
                raise ValueError(f"Invalid category: {category}")
            
            # Add metadata
            preset = {
                "settings": settings,
                "metadata": {
                    "category": category,
                    "description": description,
                    "created": datetime.now().isoformat(),
                    "updated": datetime.now().isoformat()
                } if self.config.metadata else {}
            }
            
            # Save preset
            self.presets[name] = preset
            self._save_to_file(name, preset)
            
            # Update history
            self._update_history(name, preset)
            
            logger.info(f"Saved preset: {name}")
            
        except Exception as e:
            logger.error(f"Failed to save preset {name}: {e}")
            raise
    
    def load_preset(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Load visualization settings from preset."""
        try:
            if name not in self.presets:
                raise KeyError(f"Preset not found: {name}")
            
            preset = self.presets[name]
            self._update_history(name, preset)
            
            return preset["settings"]
            
        except Exception as e:
            logger.error(f"Failed to load preset {name}: {e}")
            raise
    
    def list_presets(
        self,
        category: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available presets."""
        presets = []
        
        for name, preset in self.presets.items():
            if (not category or 
                preset.get("metadata", {}).get("category") == category):
                presets.append({
                    "name": name,
                    **preset
                })
        
        return sorted(presets, key=lambda p: p["name"])
    
    def delete_preset(self, name: str):
        """Delete visualization preset."""
        try:
            if name not in self.presets:
                raise KeyError(f"Preset not found: {name}")
            
            # Remove from memory
            del self.presets[name]
            
            # Remove file
            preset_path = self.config.preset_dir / f"{name}.yaml"
            preset_path.unlink()
            
            # Backup if configured
            if self.config.backup_dir:
                self._backup_preset(name)
            
            logger.info(f"Deleted preset: {name}")
            
        except Exception as e:
            logger.error(f"Failed to delete preset {name}: {e}")
            raise
    
    def update_preset(
        self,
        name: str,
        settings: Dict[str, Any],
        description: Optional[str] = None
    ):
        """Update existing preset."""
        try:
            if name not in self.presets:
                raise KeyError(f"Preset not found: {name}")
            
            preset = self.presets[name]
            
            # Update settings
            preset["settings"].update(settings)
            
            # Update metadata
            if self.config.metadata:
                preset["metadata"]["updated"] = datetime.now().isoformat()
                if description:
                    preset["metadata"]["description"] = description
            
            # Save changes
            self._save_to_file(name, preset)
            self._update_history(name, preset)
            
            logger.info(f"Updated preset: {name}")
            
        except Exception as e:
            logger.error(f"Failed to update preset {name}: {e}")
            raise
    
    def get_preset_history(self) -> List[Dict[str, Any]]:
        """Get preset usage history."""
        return self.history
    
    def create_default_presets(self):
        """Create default visualization presets."""
        defaults = {
            "publication": {
                "template": "plotly_white",
                "colormap": "RdBu",
                "resolution": 150,
                "animation_duration": 1000,
                "width": 1200,
                "height": 800
            },
            "presentation": {
                "template": "plotly_dark",
                "colormap": "viridis",
                "resolution": 100,
                "animation_duration": 750,
                "width": 1600,
                "height": 900
            },
            "analysis": {
                "template": "simple_white",
                "colormap": "RdYlBu",
                "resolution": 200,
                "animation_duration": 500,
                "width": 1000,
                "height": 1000
            }
        }
        
        for name, settings in defaults.items():
            if name not in self.presets:
                self.save_preset(
                    name,
                    settings,
                    category="Default",
                    description=f"Default {name} settings"
                )
    
    def export_presets(
        self,
        output_path: Path,
        category: Optional[str] = None
    ):
        """Export presets to file."""
        try:
            presets = self.list_presets(category)
            
            with open(output_path, "w") as f:
                yaml.dump(presets, f, default_flow_style=False)
            
            logger.info(f"Exported presets to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export presets: {e}")
            raise
    
    def import_presets(
        self,
        input_path: Path,
        overwrite: bool = False
    ):
        """Import presets from file."""
        try:
            with open(input_path) as f:
                presets = yaml.safe_load(f)
            
            for preset in presets:
                name = preset.pop("name")
                if overwrite or name not in self.presets:
                    self.presets[name] = preset
                    self._save_to_file(name, preset)
            
            logger.info(f"Imported presets from {input_path}")
            
        except Exception as e:
            logger.error(f"Failed to import presets: {e}")
            raise
    
    def _load_presets(self):
        """Load presets from files."""
        try:
            for preset_file in self.config.preset_dir.glob("*.yaml"):
                with open(preset_file) as f:
                    preset = yaml.safe_load(f)
                    name = preset_file.stem
                    self.presets[name] = preset
            
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
            raise
    
    def _save_to_file(self, name: str, preset: Dict[str, Any]):
        """Save preset to file."""
        preset_path = self.config.preset_dir / f"{name}.yaml"
        with open(preset_path, "w") as f:
            yaml.dump(preset, f, default_flow_style=False)
    
    def _update_history(self, name: str, preset: Dict[str, Any]):
        """Update preset usage history."""
        entry = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "category": preset.get("metadata", {}).get("category", "Custom")
        }
        
        self.history.append(entry)
        
        # Trim history if needed
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
    
    def _backup_preset(self, name: str):
        """Create backup of deleted preset."""
        if name in self.presets and self.config.backup_dir:
            backup_path = (
                self.config.backup_dir / 
                f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
            )
            
            with open(backup_path, "w") as f:
                yaml.dump(self.presets[name], f, default_flow_style=False)

def create_preset_manager(
    preset_dir: Optional[Path] = None,
    backup_dir: Optional[Path] = None
) -> PresetManager:
    """Create preset manager."""
    if preset_dir is None:
        preset_dir = Path("visualization_presets")
    
    config = PresetConfig(
        preset_dir=preset_dir,
        backup_dir=backup_dir
    )
    
    manager = PresetManager(config)
    manager.create_default_presets()
    return manager

if __name__ == "__main__":
    # Example usage
    from .visualization_controls import create_visualization_controls
    from .advanced_sensitivity_viz import create_advanced_visualizer
    from .sensitivity_analysis import create_sensitivity_analyzer
    from .power_analysis import create_analyzer
    
    # Create components
    power_analyzer = create_analyzer()
    sensitivity = create_sensitivity_analyzer(power_analyzer)
    visualizer = create_advanced_visualizer(sensitivity)
    controls = create_visualization_controls(visualizer)
    
    # Create preset manager
    preset_manager = create_preset_manager()
    
    # Create custom preset
    preset_manager.save_preset(
        "custom_analysis",
        {
            "template": "seaborn",
            "colormap": "plasma",
            "resolution": 120,
            "animation_duration": 600
        },
        category="Analysis",
        description="Custom analysis settings"
    )
    
    # List presets
    presets = preset_manager.list_presets(category="Analysis")
    print(json.dumps(presets, indent=2))
