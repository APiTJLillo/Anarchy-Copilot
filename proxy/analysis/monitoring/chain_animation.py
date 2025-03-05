"""Animation tools for filter chain visualization."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from plotly.subplots import make_subplots
import plotly.colors as colors

from .chain_visualization import ChainVisualizer, VisualizationConfig
from .filter_chaining import FilterChain

logger = logging.getLogger(__name__)

@dataclass
class AnimationConfig:
    """Configuration for chain animation."""
    frame_duration: int = 500
    transition_duration: int = 200
    node_pulse_size: float = 1.5
    edge_fade_opacity: float = 0.3
    color_sequence: List[str] = None
    show_progress: bool = True
    show_tooltips: bool = True
    output_path: Optional[Path] = None

class ChainAnimator:
    """Animate filter chain execution and transitions."""
    
    def __init__(
        self,
        visualizer: ChainVisualizer,
        config: AnimationConfig
    ):
        self.visualizer = visualizer
        self.config = config
        
        if not config.color_sequence:
            self.config.color_sequence = colors.DEFAULT_PLOTLY_COLORS
    
    def animate_chain_execution(
        self,
        name: str,
        data: Dict[str, Any]
    ) -> go.Figure:
        """Create animated visualization of chain execution."""
        if name not in self.visualizer.chain.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.visualizer.chain.chains[name]
        pos = nx.spring_layout(graph)
        
        # Create base figure
        fig = go.Figure()
        
        # Generate frames for each step
        frames = []
        steps = []
        
        for node in nx.topological_sort(graph):
            # Create frame for this step
            frame_data = []
            
            # Add all edges (with fading for inactive)
            edge_x = []
            edge_y = []
            edge_color = []
            
            for edge in graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Color active paths
                if self._is_active_edge(edge, node):
                    edge_color.extend([1.0, 1.0, 1.0])
                else:
                    edge_color.extend([
                        self.config.edge_fade_opacity,
                        self.config.edge_fade_opacity,
                        self.config.edge_fade_opacity
                    ])
            
            frame_data.append(
                go.Scatter(
                    x=edge_x,
                    y=edge_y,
                    mode="lines",
                    line=dict(
                        width=self.visualizer.config.edge_width,
                        color="rgba(136, 136, 136, 1.0)"
                    ),
                    opacity=edge_color,
                    name="Dependencies"
                )
            )
            
            # Add nodes with highlighting for active node
            node_x = []
            node_y = []
            node_color = []
            node_size = []
            node_text = []
            
            for n, data in graph.nodes(data=True):
                x, y = pos[n]
                node_x.append(x)
                node_y.append(y)
                
                # Generate node text
                text = (
                    f"Step {n}<br>"
                    f"Filter: {data['filter']}<br>"
                    f"Parameters: {data['params']}<br>"
                    f"Optional: {data['optional']}"
                )
                node_text.append(text)
                
                # Color and size based on state
                if n == node:
                    node_color.append(self.config.color_sequence[0])
                    node_size.append(
                        self.visualizer.config.node_size * self.config.node_pulse_size
                    )
                elif n in nx.ancestors(graph, node):
                    node_color.append(self.config.color_sequence[1])
                    node_size.append(self.visualizer.config.node_size)
                else:
                    node_color.append("lightgray")
                    node_size.append(self.visualizer.config.node_size)
            
            frame_data.append(
                go.Scatter(
                    x=node_x,
                    y=node_y,
                    mode="markers+text" if self.visualizer.config.show_labels else "markers",
                    marker=dict(
                        size=node_size,
                        color=node_color,
                        line=dict(width=2)
                    ),
                    text=[f"Step {n}" for n in graph.nodes()],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=node_text,
                    name="Steps"
                )
            )
            
            # Add execution metrics if available
            if self.config.show_progress:
                metrics = self._get_step_metrics(name, node, data)
                frame_data.append(
                    self._create_metrics_trace(metrics, node)
                )
            
            # Create frame
            frames.append(go.Frame(
                data=frame_data,
                name=f"step_{node}"
            ))
            
            # Add slider step
            steps.append(dict(
                args=[
                    [f"step_{node}"],
                    dict(
                        frame=dict(duration=self.config.frame_duration),
                        mode="immediate",
                        transition=dict(duration=self.config.transition_duration)
                    )
                ],
                label=f"Step {node}",
                method="animate"
            ))
        
        # Set up animation
        fig.frames = frames
        
        # Add play button and slider
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=self.config.frame_duration),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=self.config.transition_duration
                                    )
                                )
                            ]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0),
                                    mode="immediate",
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                steps=steps
            )]
        )
        
        # Update layout
        fig.update_layout(
            title=f"Filter Chain Execution: {name}",
            showlegend=True,
            hovermode="closest",
            width=self.visualizer.config.width,
            height=self.visualizer.config.height,
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def animate_chain_transformation(
        self,
        name: str,
        data: Dict[str, Any]
    ) -> go.Figure:
        """Create animated visualization of data transformation."""
        if name not in self.visualizer.chain.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.visualizer.chain.chains[name]
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Data Distribution",
                "Feature Correlations",
                "Sample Count",
                "Value Range"
            ]
        )
        
        # Generate frames for each step
        frames = []
        steps = []
        
        # Apply chain step by step
        intermediate_data = data.copy()
        
        for node in nx.topological_sort(graph):
            frame_data = []
            
            # Apply filter
            filter_name = graph.nodes[node]["filter"]
            params = graph.nodes[node]["params"]
            
            intermediate_data = self.visualizer.chain.filters.apply_filters(
                intermediate_data,
                [filter_name],
                {filter_name: params}
            )
            
            # Data distribution
            if isinstance(intermediate_data, pd.DataFrame):
                for col in intermediate_data.columns:
                    frame_data.append(
                        go.Histogram(
                            x=intermediate_data[col],
                            name=col,
                            opacity=0.7
                        )
                    )
            
            # Feature correlations
            if isinstance(intermediate_data, pd.DataFrame):
                corr = intermediate_data.corr()
                frame_data.append(
                    go.Heatmap(
                        z=corr.values,
                        x=corr.columns,
                        y=corr.index,
                        colorscale="RdBu"
                    )
                )
            
            # Sample count
            frame_data.append(
                go.Bar(
                    x=["Samples"],
                    y=[len(intermediate_data)]
                )
            )
            
            # Value range
            if isinstance(intermediate_data, pd.DataFrame):
                ranges = intermediate_data.agg(["min", "max"])
                frame_data.append(
                    go.Bar(
                        x=ranges.columns,
                        y=ranges.max() - ranges.min(),
                        name="Range"
                    )
                )
            
            # Create frame
            frames.append(go.Frame(
                data=frame_data,
                name=f"step_{node}"
            ))
            
            # Add slider step
            steps.append(dict(
                args=[
                    [f"step_{node}"],
                    dict(
                        frame=dict(duration=self.config.frame_duration),
                        mode="immediate",
                        transition=dict(duration=self.config.transition_duration)
                    )
                ],
                label=f"Step {node}",
                method="animate"
            ))
        
        # Set up animation
        fig.frames = frames
        
        # Add play button and slider
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=self.config.frame_duration),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=self.config.transition_duration
                                    )
                                )
                            ]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0),
                                    mode="immediate",
                                    transition=dict(duration=0)
                                )
                            ]
                        )
                    ]
                )
            ],
            sliders=[dict(
                active=0,
                steps=steps
            )]
        )
        
        # Update layout
        fig.update_layout(
            title=f"Data Transformation: {name}",
            height=self.visualizer.config.height,
            width=self.visualizer.config.width,
            template="plotly_dark" if self.visualizer.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def _is_active_edge(
        self,
        edge: Tuple[int, int],
        current_node: int
    ) -> bool:
        """Check if edge is part of active path."""
        return (
            edge[1] == current_node or
            edge[0] == current_node
        )
    
    def _get_step_metrics(
        self,
        name: str,
        node: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get metrics for execution step."""
        cache_key = self.visualizer.chain._get_cache_key(name, node, data)
        
        if cache_key in self.visualizer.chain.cached_results:
            step_data = self.visualizer.chain.cached_results[cache_key]
            return {
                "data_size": self.visualizer.chain._get_data_size(step_data),
                "filter_name": self.visualizer.chain.chains[name].nodes[node]["filter"],
                "step_number": node
            }
        return {}
    
    def _create_metrics_trace(
        self,
        metrics: Dict[str, Any],
        node: int
    ) -> go.Scatter:
        """Create trace for execution metrics."""
        return go.Scatter(
            x=[node],
            y=[metrics.get("data_size", 0)],
            mode="markers+text",
            marker=dict(
                symbol="star",
                size=15,
                color=self.config.color_sequence[2]
            ),
            text=[f"Size: {metrics.get('data_size', 0)}"],
            textposition="top center",
            name="Metrics"
        )
    
    def save_animations(
        self,
        output_path: Optional[Path] = None
    ):
        """Save chain animations."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            for name in self.visualizer.chain.chains:
                # Create example data
                example_data = {
                    "values": np.random.randn(1000),
                    "features": pd.DataFrame(
                        np.random.randn(1000, 5),
                        columns=["f1", "f2", "f3", "f4", "f5"]
                    )
                }
                
                # Save execution animation
                execution = self.animate_chain_execution(name, example_data)
                execution.write_html(
                    str(path / f"{name}_execution.html")
                )
                
                # Save transformation animation
                transform = self.animate_chain_transformation(name, example_data)
                transform.write_html(
                    str(path / f"{name}_transformation.html")
                )
            
            logger.info(f"Saved chain animations to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save animations: {e}")

def create_chain_animator(
    visualizer: ChainVisualizer,
    output_path: Optional[Path] = None
) -> ChainAnimator:
    """Create chain animator."""
    config = AnimationConfig(output_path=output_path)
    return ChainAnimator(visualizer, config)

if __name__ == "__main__":
    # Example usage
    from .chain_visualization import create_chain_visualizer
    from .filter_chaining import create_filter_chain
    from .learning_filters import create_learning_filter
    from .interactive_learning import create_interactive_learning
    from .learning_visualization import create_learning_visualizer
    from .optimization_learning import create_optimization_learner
    from .composition_optimization import create_composition_optimizer
    from .composition_analysis import create_composition_analysis
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    # Create components
    easing = create_easing_functions()
    visualizer = create_easing_visualizer(easing)
    interactive = create_interactive_easing(visualizer)
    controls = create_animation_controls(interactive)
    events = create_event_manager(controls)
    scheduler = create_event_scheduler(events)
    pattern = create_scheduling_pattern(scheduler)
    composer = create_pattern_composer(pattern)
    analyzer = create_composition_analysis(composer)
    optimizer = create_composition_optimizer(analyzer)
    learner = create_optimization_learner(optimizer)
    viz = create_learning_visualizer(learner)
    interactive_learning = create_interactive_learning(viz)
    filters = create_learning_filter(interactive_learning)
    chain = create_filter_chain(filters)
    chain_viz = create_chain_visualizer(chain)
    animator = create_chain_animator(
        chain_viz,
        output_path=Path("chain_animation")
    )
    
    # Example chain
    chain.create_chain(
        "preprocessing",
        [
            {
                "filter": "time_range",
                "params": {"window": 30}
            },
            {
                "filter": "confidence",
                "params": {"threshold": 0.7},
                "requires": ["time_range"]
            },
            {
                "filter": "complexity",
                "params": {"max_complexity": 5},
                "requires": ["confidence"]
            }
        ]
    )
    
    # Example data
    data = {
        "timestamp": pd.date_range(start="2025-01-01", periods=1000, freq="H"),
        "confidence": np.random.uniform(0, 1, 1000),
        "success": np.random.choice([True, False], 1000),
        "complexity": np.random.randint(1, 20, 1000),
        "features": pd.DataFrame(
            np.random.randn(1000, 5),
            columns=["f1", "f2", "f3", "f4", "f5"]
        )
    }
    
    # Generate animations
    execution = animator.animate_chain_execution("preprocessing", data)
    transform = animator.animate_chain_transformation("preprocessing", data)
    
    # Save animations
    animator.save_animations()
