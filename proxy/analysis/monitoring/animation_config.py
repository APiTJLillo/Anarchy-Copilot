"""Animation configuration and customization."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import plotly.graph_objects as go
import plotly.express as px
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class AnimationType(Enum):
    """Types of supported animations."""
    CASCADE = "cascade"
    PROPAGATION = "propagation"
    EVOLUTION = "evolution"
    TREE = "tree"
    FLOW = "flow"

class ColorScheme(Enum):
    """Available color schemes."""
    VIRIDIS = "Viridis"
    PLASMA = "Plasma"
    INFERNO = "Inferno"
    MAGMA = "Magma"
    CIVIDIS = "Cividis"
    RAINBOW = "Rainbow"

@dataclass
class AnimationStyle:
    """Style configuration for animations."""
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    background_color: str = "white"
    font_family: str = "Arial"
    font_size: int = 12
    title_size: int = 16
    marker_size: int = 20
    line_width: int = 2
    opacity: float = 0.8
    transition_easing: str = "cubic-in-out"

@dataclass
class PlaybackConfig:
    """Configuration for animation playback."""
    frame_duration: int = 500  # milliseconds
    transition_duration: int = 300  # milliseconds
    loop: bool = True
    mode: str = "immediate"  # or "afterall"
    direction: str = "forward"  # or "reverse"
    redraw: bool = True

@dataclass
class InteractionConfig:
    """Configuration for user interactions."""
    draggable_nodes: bool = True
    zoomable: bool = True
    hoverable: bool = True
    selectable: bool = True
    showgrid: bool = False
    showlegend: bool = True
    hovermode: str = "closest"
    dragmode: str = "pan"

@dataclass
class AnimationConfig:
    """Complete animation configuration."""
    style: AnimationStyle = field(default_factory=AnimationStyle)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)
    output_dir: Path = Path("animations")
    enabled_types: List[AnimationType] = field(
        default_factory=lambda: list(AnimationType)
    )
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.output_dir / "animation_config.json"
        
        config_dict = {
            "style": {
                "color_scheme": self.style.color_scheme.value,
                "background_color": self.style.background_color,
                "font_family": self.style.font_family,
                "font_size": self.style.font_size,
                "title_size": self.style.title_size,
                "marker_size": self.style.marker_size,
                "line_width": self.style.line_width,
                "opacity": self.style.opacity,
                "transition_easing": self.style.transition_easing
            },
            "playback": {
                "frame_duration": self.playback.frame_duration,
                "transition_duration": self.playback.transition_duration,
                "loop": self.playback.loop,
                "mode": self.playback.mode,
                "direction": self.playback.direction,
                "redraw": self.playback.redraw
            },
            "interaction": {
                "draggable_nodes": self.interaction.draggable_nodes,
                "zoomable": self.interaction.zoomable,
                "hoverable": self.interaction.hoverable,
                "selectable": self.interaction.selectable,
                "showgrid": self.interaction.showgrid,
                "showlegend": self.interaction.showlegend,
                "hovermode": self.interaction.hovermode,
                "dragmode": self.interaction.dragmode
            },
            "enabled_types": [t.value for t in self.enabled_types]
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "AnimationConfig":
        """Load configuration from file."""
        with open(path) as f:
            config_dict = json.load(f)
        
        style = AnimationStyle(
            color_scheme=ColorScheme(config_dict["style"]["color_scheme"]),
            background_color=config_dict["style"]["background_color"],
            font_family=config_dict["style"]["font_family"],
            font_size=config_dict["style"]["font_size"],
            title_size=config_dict["style"]["title_size"],
            marker_size=config_dict["style"]["marker_size"],
            line_width=config_dict["style"]["line_width"],
            opacity=config_dict["style"]["opacity"],
            transition_easing=config_dict["style"]["transition_easing"]
        )
        
        playback = PlaybackConfig(
            frame_duration=config_dict["playback"]["frame_duration"],
            transition_duration=config_dict["playback"]["transition_duration"],
            loop=config_dict["playback"]["loop"],
            mode=config_dict["playback"]["mode"],
            direction=config_dict["playback"]["direction"],
            redraw=config_dict["playback"]["redraw"]
        )
        
        interaction = InteractionConfig(
            draggable_nodes=config_dict["interaction"]["draggable_nodes"],
            zoomable=config_dict["interaction"]["zoomable"],
            hoverable=config_dict["interaction"]["hoverable"],
            selectable=config_dict["interaction"]["selectable"],
            showgrid=config_dict["interaction"]["showgrid"],
            showlegend=config_dict["interaction"]["showlegend"],
            hovermode=config_dict["interaction"]["hovermode"],
            dragmode=config_dict["interaction"]["dragmode"]
        )
        
        enabled_types = [
            AnimationType(t)
            for t in config_dict["enabled_types"]
        ]
        
        return cls(
            style=style,
            playback=playback,
            interaction=interaction,
            enabled_types=enabled_types
        )

class AnimationCustomizer:
    """Customize animation appearance and behavior."""
    
    def __init__(self, config: AnimationConfig):
        self.config = config
    
    def apply_style(self, fig: go.Figure) -> go.Figure:
        """Apply style configuration to figure."""
        fig.update_layout(
            font=dict(
                family=self.config.style.font_family,
                size=self.config.style.font_size
            ),
            title_font_size=self.config.style.title_size,
            plot_bgcolor=self.config.style.background_color,
            paper_bgcolor=self.config.style.background_color,
            showlegend=self.config.interaction.showlegend,
            hovermode=self.config.interaction.hovermode,
            dragmode=self.config.interaction.dragmode
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=self.config.interaction.showgrid,
            zeroline=self.config.interaction.showgrid
        )
        fig.update_yaxes(
            showgrid=self.config.interaction.showgrid,
            zeroline=self.config.interaction.showgrid
        )
        
        # Update markers and lines
        for trace in fig.data:
            if trace.type == "scatter":
                if "markers" in trace.mode:
                    trace.marker.size = self.config.style.marker_size
                    trace.marker.opacity = self.config.style.opacity
                if "lines" in trace.mode:
                    trace.line.width = self.config.style.line_width
        
        return fig
    
    def apply_animation_settings(self, fig: go.Figure) -> go.Figure:
        """Apply animation settings to figure."""
        # Update animation settings
        fig.layout.updatemenus[0].buttons[0].args[1].update(
            frame=dict(
                duration=self.config.playback.frame_duration,
                redraw=self.config.playback.redraw
            ),
            transition=dict(
                duration=self.config.playback.transition_duration,
                easing=self.config.style.transition_easing
            ),
            mode=self.config.playback.mode,
            direction=self.config.playback.direction
        )
        
        if not self.config.playback.loop:
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["redraw"] = True
        
        return fig
    
    def apply_interaction_settings(self, fig: go.Figure) -> go.Figure:
        """Apply interaction settings to figure."""
        # Update layout for interactions
        fig.update_layout(
            dragmode=self.config.interaction.dragmode if self.config.interaction.draggable_nodes else False,
            hovermode=self.config.interaction.hovermode if self.config.interaction.hoverable else False,
            showlegend=self.config.interaction.showlegend,
            modebar=dict(
                remove=[] if self.config.interaction.zoomable else ["zoom", "pan"]
            )
        )
        
        return fig
    
    def get_color_scale(self, values: List[float]) -> List[str]:
        """Get colors from configured color scheme."""
        return px.colors.sample_colorscale(
            self.config.style.color_scheme.value,
            np.linspace(0, 1, len(values))
        )
    
    def customize_figure(self, fig: go.Figure) -> go.Figure:
        """Apply all customizations to figure."""
        fig = self.apply_style(fig)
        fig = self.apply_animation_settings(fig)
        fig = self.apply_interaction_settings(fig)
        return fig

def create_default_config() -> AnimationConfig:
    """Create default animation configuration."""
    return AnimationConfig()

if __name__ == "__main__":
    # Create and save default configuration
    config = create_default_config()
    config.save()
