"""Filter presets for mutation analysis."""

import asyncio
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import json
from pathlib import Path
from datetime import datetime
import jsonschema

@dataclass
class PresetConfig:
    """Configuration for filter presets."""
    preset_dir: Path = Path("mutation_presets")
    max_presets: int = 50
    auto_sync: bool = True
    sync_interval: float = 60.0
    schema_validation: bool = True

@dataclass
class FilterPreset:
    """Preset filter configuration."""
    name: str
    description: str
    filters: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=datetime.now)
    modified: datetime = field(default_factory=datetime.now)

class PresetManager:
    """Manage mutation analysis filter presets."""
    
    PRESET_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {
                    "operators": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "error_types": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "score_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "time_range": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2
                    }
                }
            },
            "metadata": {"type": "object"},
            "created": {"type": "string", "format": "date-time"},
            "modified": {"type": "string", "format": "date-time"}
        },
        "required": ["name", "description", "filters"]
    }
    
    def __init__(
        self,
        config: PresetConfig = None
    ):
        self.config = config or PresetConfig()
        self.presets: Dict[str, FilterPreset] = {}
        self._sync_task: Optional[asyncio.Task] = None
        
        # Create preset directory
        self.config.preset_dir.mkdir(parents=True, exist_ok=True)
        
        # Start auto-sync if enabled
        if self.config.auto_sync:
            self._start_sync()
    
    def _start_sync(self):
        """Start auto-sync task."""
        async def sync_loop():
            while True:
                try:
                    await self.sync_presets()
                except Exception as e:
                    print(f"Sync error: {e}")
                await asyncio.sleep(self.config.sync_interval)
        
        self._sync_task = asyncio.create_task(sync_loop())
    
    async def sync_presets(self):
        """Sync presets with filesystem."""
        # Load presets from files
        for preset_file in self.config.preset_dir.glob("*.json"):
            try:
                with open(preset_file) as f:
                    data = json.load(f)
                
                if self.config.schema_validation:
                    jsonschema.validate(data, self.PRESET_SCHEMA)
                
                data["created"] = datetime.fromisoformat(data["created"])
                data["modified"] = datetime.fromisoformat(data["modified"])
                
                preset = FilterPreset(**data)
                self.presets[preset.name] = preset
                
            except Exception as e:
                print(f"Failed to load preset {preset_file}: {e}")
        
        # Trim if needed
        while len(self.presets) > self.config.max_presets:
            oldest = min(
                self.presets.values(),
                key=lambda p: p.modified
            )
            del self.presets[oldest.name]
    
    async def save_preset(
        self,
        name: str,
        description: str,
        filters: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilterPreset:
        """Save new filter preset."""
        # Create preset
        preset = FilterPreset(
            name=name,
            description=description,
            filters=filters,
            metadata=metadata or {}
        )
        
        # Validate
        if self.config.schema_validation:
            jsonschema.validate(
                preset.__dict__,
                self.PRESET_SCHEMA
            )
        
        # Save to memory
        self.presets[name] = preset
        
        # Save to file
        preset_file = self.config.preset_dir / f"{name}.json"
        data = {
            **preset.__dict__,
            "created": preset.created.isoformat(),
            "modified": preset.modified.isoformat()
        }
        
        with open(preset_file, "w") as f:
            json.dump(data, f, indent=2)
        
        return preset
    
    async def update_preset(
        self,
        name: str,
        filters: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FilterPreset:
        """Update existing preset."""
        if name not in self.presets:
            raise ValueError(f"Preset not found: {name}")
        
        preset = self.presets[name]
        
        if filters is not None:
            preset.filters = filters
        if description is not None:
            preset.description = description
        if metadata is not None:
            preset.metadata.update(metadata)
        
        preset.modified = datetime.now()
        
        # Save changes
        await self.save_preset(
            preset.name,
            preset.description,
            preset.filters,
            preset.metadata
        )
        
        return preset
    
    async def delete_preset(
        self,
        name: str
    ):
        """Delete preset."""
        if name not in self.presets:
            raise ValueError(f"Preset not found: {name}")
        
        # Remove from memory
        del self.presets[name]
        
        # Remove file
        preset_file = self.config.preset_dir / f"{name}.json"
        preset_file.unlink(missing_ok=True)
    
    async def get_preset(
        self,
        name: str
    ) -> FilterPreset:
        """Get preset by name."""
        if name not in self.presets:
            raise ValueError(f"Preset not found: {name}")
        
        return self.presets[name]
    
    async def list_presets(
        self,
        sort_by: str = "modified",
        reverse: bool = True
    ) -> List[FilterPreset]:
        """List all presets."""
        presets = list(self.presets.values())
        
        if sort_by == "name":
            presets.sort(key=lambda p: p.name, reverse=reverse)
        elif sort_by == "created":
            presets.sort(key=lambda p: p.created, reverse=reverse)
        elif sort_by == "modified":
            presets.sort(key=lambda p: p.modified, reverse=reverse)
        
        return presets
    
    async def search_presets(
        self,
        query: str,
        fields: Optional[List[str]] = None
    ) -> List[FilterPreset]:
        """Search presets."""
        if fields is None:
            fields = ["name", "description"]
        
        results = []
        query = query.lower()
        
        for preset in self.presets.values():
            for field in fields:
                value = str(getattr(preset, field, "")).lower()
                if query in value:
                    results.append(preset)
                    break
        
        return results
    
    async def import_presets(
        self,
        source: Path
    ) -> int:
        """Import presets from file."""
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source}")
        
        with open(source) as f:
            data = json.load(f)
        
        imported = 0
        for preset_data in data:
            try:
                await self.save_preset(
                    preset_data["name"],
                    preset_data["description"],
                    preset_data["filters"],
                    preset_data.get("metadata")
                )
                imported += 1
            except Exception as e:
                print(f"Failed to import preset: {e}")
        
        return imported
    
    async def export_presets(
        self,
        target: Path,
        presets: Optional[List[str]] = None
    ):
        """Export presets to file."""
        data = []
        
        for name, preset in self.presets.items():
            if presets is None or name in presets:
                data.append({
                    **preset.__dict__,
                    "created": preset.created.isoformat(),
                    "modified": preset.modified.isoformat()
                })
        
        with open(target, "w") as f:
            json.dump(data, f, indent=2)

def create_preset_manager(
    config: Optional[PresetConfig] = None
) -> PresetManager:
    """Create preset manager."""
    return PresetManager(config)

if __name__ == "__main__":
    async def main():
        # Create preset manager
        manager = create_preset_manager(
            PresetConfig(preset_dir=Path("test_presets"))
        )
        
        # Add some test presets
        await manager.save_preset(
            "high_priority",
            "Focus on critical mutations",
            {
                "operators": ["type_mutation", "value_mutation"],
                "error_types": ["TypeError", "ValueError"],
                "score_range": [0.8, 1.0],
                "time_range": None
            }
        )
        
        await manager.save_preset(
            "low_priority",
            "Basic validation checks",
            {
                "operators": ["null_mutation"],
                "error_types": ["AssertionError"],
                "score_range": [0.0, 0.5],
                "time_range": None
            }
        )
        
        # List presets
        presets = await manager.list_presets()
        for preset in presets:
            print(f"{preset.name}: {preset.description}")
    
    asyncio.run(main())
