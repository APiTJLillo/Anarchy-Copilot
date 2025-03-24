"""Animation tools for comparing filter chains."""

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
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

from .chain_animation import ChainAnimator, AnimationConfig
from .chain_visualization import ChainVisualizer

logger = logging.getLogger(__name__)

@dataclass
class ComparisonConfig:
    """Configuration for chain comparison animation."""
    sync_animations: bool = True
    highlight_differences: bool = True
    show_metrics: bool = True
    diff_threshold: float = 0.1
    max_chains: int = 4
    output_path: Optional[Path] = None

class ChainComparator:
    """Compare and animate filter chain differences."""
    
    def __init__(
        self,
        animator: ChainAnimator,
        config: ComparisonConfig
    ):
        self.animator = animator
        self.config = config
    
    def animate_chain_comparison(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> go.Figure:
        """Create animated comparison of multiple chains."""
        if len(names) > self.config.max_chains:
            raise ValueError(f"Maximum number of chains for comparison is {self.config.max_chains}")
        
        for name in names:
            if name not in self.animator.visualizer.chain.chains:
                raise ValueError(f"Unknown chain: {name}")
        
        # Create subplot grid
        rows = (len(names) + 1) // 2
        cols = min(2, len(names))
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=names,
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Generate synchronized frames
        frames = []
        steps = []
        all_nodes = []
        
        # Get all nodes from all chains
        for name in names:
            graph = self.animator.visualizer.chain.chains[name]
            all_nodes.extend(nx.topological_sort(graph))
        
        unique_nodes = sorted(set(all_nodes))
        
        for node_idx, current_node in enumerate(unique_nodes):
            frame_data = []
            
            for chain_idx, name in enumerate(names):
                graph = self.animator.visualizer.chain.chains[name]
                pos = nx.spring_layout(graph)
                
                # Calculate row and column for subplot
                row = (chain_idx // 2) + 1
                col = (chain_idx % 2) + 1
                
                # Add traces for this chain
                traces = self._create_chain_traces(
                    graph,
                    pos,
                    current_node,
                    name,
                    data
                )
                
                for trace in traces:
                    # Update trace to use correct subplot
                    trace.update(
                        xaxis=f"x{chain_idx+1}",
                        yaxis=f"y{chain_idx+1}"
                    )
                    frame_data.append(trace)
            
            # Add comparison metrics if enabled
            if self.config.show_metrics:
                metrics = self._calculate_comparison_metrics(
                    names,
                    current_node,
                    data
                )
                frame_data.extend(
                    self._create_metric_traces(metrics, node_idx)
                )
            
            # Create frame
            frames.append(go.Frame(
                data=frame_data,
                name=f"step_{node_idx}"
            ))
            
            # Add slider step
            steps.append(dict(
                args=[
                    [f"step_{node_idx}"],
                    dict(
                        frame=dict(
                            duration=self.animator.config.frame_duration,
                            redraw=True
                        ),
                        mode="immediate",
                        transition=dict(
                            duration=self.animator.config.transition_duration
                        )
                    )
                ],
                label=f"Step {node_idx}",
                method="animate"
            ))
        
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
                                    frame=dict(
                                        duration=self.animator.config.frame_duration,
                                        redraw=True
                                    ),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=self.animator.config.transition_duration
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
                steps=steps,
                currentvalue=dict(
                    prefix="Step: ",
                    visible=True
                )
            )]
        )
        
        # Set frame sequence
        fig.frames = frames
        
        # Update layout
        fig.update_layout(
            title="Chain Comparison",
            showlegend=True,
            height=self.animator.config.height,
            width=self.animator.config.width,
            template="plotly_dark" if self.animator.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def animate_impact_comparison(
        self,
        names: List[str],
        data: Dict[str, Any]
    ) -> go.Figure:
        """Create animated comparison of chain impacts on data."""
        if len(names) > self.config.max_chains:
            raise ValueError(f"Maximum number of chains for comparison is {self.config.max_chains}")
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Data Size Reduction",
                "Feature Impact",
                "Execution Time",
                "Output Distribution"
            ]
        )
        
        # Generate frames
        frames = []
        steps = []
        
        # Track intermediate results
        results = {
            name: {"data": data.copy()}
            for name in names
        }
        
        # Get maximum steps
        max_steps = max(
            len(self.animator.visualizer.chain.chains[name])
            for name in names
        )
        
        for step in range(max_steps):
            frame_data = []
            
            # Apply filters and collect metrics
            for name in names:
                graph = self.animator.visualizer.chain.chains[name]
                if step < len(graph):
                    node = list(nx.topological_sort(graph))[step]
                    
                    # Apply filter
                    filter_name = graph.nodes[node]["filter"]
                    params = graph.nodes[node]["params"]
                    
                    results[name]["data"] = self.animator.visualizer.chain.filters.apply_filters(
                        results[name]["data"],
                        [filter_name],
                        {filter_name: params}
                    )
            
            # Data size reduction trace
            sizes = {
                name: self.animator.visualizer.chain._get_data_size(result["data"])
                for name, result in results.items()
            }
            frame_data.append(
                go.Bar(
                    x=list(sizes.keys()),
                    y=list(sizes.values()),
                    name="Data Size",
                    showlegend=False
                )
            )
            
            # Feature impact trace
            if all(
                isinstance(result["data"], pd.DataFrame)
                for result in results.values()
            ):
                for name, result in results.items():
                    frame_data.append(
                        go.Heatmap(
                            z=result["data"].corr().values,
                            x=result["data"].columns,
                            y=result["data"].columns,
                            colorscale="RdBu",
                            name=f"{name} Correlations",
                            showscale=False,
                            showlegend=False
                        )
                    )
            
            # Execution time trace
            times = {
                name: step * self.animator.config.frame_duration / 1000
                for name in names
            }
            frame_data.append(
                go.Scatter(
                    x=list(times.keys()),
                    y=list(times.values()),
                    mode="lines+markers",
                    name="Execution Time",
                    showlegend=False
                )
            )
            
            # Distribution comparison trace
            if all(
                isinstance(result["data"], (pd.DataFrame, pd.Series))
                for result in results.values()
            ):
                for name, result in results.items():
                    data_values = (
                        result["data"].values.flatten()
                        if isinstance(result["data"], pd.DataFrame)
                        else result["data"].values
                    )
                    frame_data.append(
                        go.Histogram(
                            x=data_values,
                            name=name,
                            opacity=0.7,
                            showlegend=True
                        )
                    )
            
            # Create frame
            frames.append(go.Frame(
                data=frame_data,
                name=f"step_{step}"
            ))
            
            # Add slider step
            steps.append(dict(
                args=[
                    [f"step_{step}"],
                    dict(
                        frame=dict(duration=self.animator.config.frame_duration),
                        mode="immediate",
                        transition=dict(duration=self.animator.config.transition_duration)
                    )
                ],
                label=f"Step {step}",
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
                                    frame=dict(duration=self.animator.config.frame_duration),
                                    fromcurrent=True,
                                    transition=dict(
                                        duration=self.animator.config.transition_duration
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
            height=self.animator.config.height,
            width=self.animator.config.width,
            template="plotly_dark" if self.animator.config.dark_mode else "plotly_white"
        )
        
        return fig
    
    def _create_chain_traces(
        self,
        graph: nx.DiGraph,
        pos: Dict[int, Tuple[float, float]],
        current_node: int,
        name: str,
        data: Dict[str, Any]
    ) -> List[go.Scatter]:
        """Create traces for chain visualization."""
        traces = []
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_color = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            if self.animator._is_active_edge(edge, current_node):
                edge_color.extend([1.0, 1.0, 1.0])
            else:
                edge_color.extend([
                    self.animator.config.edge_fade_opacity,
                    self.animator.config.edge_fade_opacity,
                    self.animator.config.edge_fade_opacity
                ])
        
        traces.append(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(
                width=self.animator.config.edge_width,
                color="rgba(136, 136, 136, 1.0)"
            ),
            opacity=edge_color,
            name=f"{name} Dependencies"
        ))
        
        # Add nodes
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        
        for n, data in graph.nodes(data=True):
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            
            text = (
                f"Step {n}<br>"
                f"Filter: {data['filter']}<br>"
                f"Parameters: {data['params']}<br>"
                f"Optional: {data['optional']}"
            )
            node_text.append(text)
            
            if n == current_node:
                node_color.append(self.animator.config.color_sequence[0])
                node_size.append(
                    self.animator.config.node_size * self.animator.config.node_pulse_size
                )
            elif n in nx.ancestors(graph, current_node):
                node_color.append(self.animator.config.color_sequence[1])
                node_size.append(self.animator.config.node_size)
            else:
                node_color.append("lightgray")
                node_size.append(self.animator.config.node_size)
        
        traces.append(go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if self.animator.config.show_labels else "markers",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2)
            ),
            text=[f"{name} Step {n}" for n in graph.nodes()],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            name=f"{name} Steps"
        ))
        
        return traces
    
    def _calculate_comparison_metrics(
        self,
        names: List[str],
        current_node: int,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate comparison metrics."""
        metrics = {
            "differences": defaultdict(list),
            "similarities": defaultdict(list)
        }
        
        # Compare results between chains
        results = {}
        
        for name in names:
            graph = self.animator.visualizer.chain.chains[name]
            if current_node in graph:
                filter_name = graph.nodes[current_node]["filter"]
                params = graph.nodes[current_node]["params"]
                
                filtered = self.animator.visualizer.chain.filters.apply_filters(
                    data.copy(),
                    [filter_name],
                    {filter_name: params}
                )
                results[name] = filtered
        
        # Calculate pairwise comparisons
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                if name1 in results and name2 in results:
                    # Compare data sizes
                    size1 = self.animator.visualizer.chain._get_data_size(results[name1])
                    size2 = self.animator.visualizer.chain._get_data_size(results[name2])
                    
                    if abs(size1 - size2) > self.config.diff_threshold * max(size1, size2):
                        metrics["differences"]["size"].append((name1, name2, abs(size1 - size2)))
                    else:
                        metrics["similarities"]["size"].append((name1, name2))
                    
                    # Compare distributions if possible
                    if all(
                        isinstance(result, (pd.DataFrame, pd.Series))
                        for result in [results[name1], results[name2]]
                    ):
                        values1 = (
                            results[name1].values.flatten()
                            if isinstance(results[name1], pd.DataFrame)
                            else results[name1].values
                        )
                        values2 = (
                            results[name2].values.flatten()
                            if isinstance(results[name2], pd.DataFrame)
                            else results[name2].values
                        )
                        
                        if len(values1) == len(values2):
                            r2 = r2_score(values1, values2)
                            mse = mean_squared_error(values1, values2)
                            
                            if r2 < 1 - self.config.diff_threshold:
                                metrics["differences"]["distribution"].append((name1, name2, mse))
                            else:
                                metrics["similarities"]["distribution"].append((name1, name2))
        
        return metrics
    
    def _create_metric_traces(
        self,
        metrics: Dict[str, Any],
        step: int
    ) -> List[Union[go.Scatter, go.Bar]]:
        """Create traces for comparison metrics."""
        traces = []
        
        # Differences bar chart
        if metrics["differences"]:
            for metric_type, differences in metrics["differences"].items():
                traces.append(go.Bar(
                    x=[f"{d[0]} vs {d[1]}" for d in differences],
                    y=[d[2] for d in differences],
                    name=f"{metric_type} Differences",
                    marker_color="red",
                    showlegend=True
                ))
        
        # Similarities scatter
        if metrics["similarities"]:
            for metric_type, similarities in metrics["similarities"].items():
                traces.append(go.Scatter(
                    x=[f"{s[0]} vs {s[1]}" for s in similarities],
                    y=[1] * len(similarities),
                    mode="markers",
                    name=f"{metric_type} Similarities",
                    marker=dict(
                        size=10,
                        color="green"
                    ),
                    showlegend=True
                ))
        
        return traces
    
    def save_comparisons(
        self,
        output_path: Optional[Path] = None
    ):
        """Save chain comparisons."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Get all possible chain combinations
            chains = list(self.animator.visualizer.chain.chains.keys())
            
            for i in range(len(chains)):
                for j in range(i + 1, len(chains)):
                    names = [chains[i], chains[j]]
                    
                    # Create example data
                    example_data = {
                        "values": np.random.randn(1000),
                        "features": pd.DataFrame(
                            np.random.randn(1000, 5),
                            columns=["f1", "f2", "f3", "f4", "f5"]
                        )
                    }
                    
                    # Save chain comparison
                    comparison = self.animate_chain_comparison(names, example_data)
                    comparison.write_html(
                        str(path / f"comparison_{names[0]}_{names[1]}.html")
                    )
                    
                    # Save impact comparison
                    impact = self.animate_impact_comparison(names, example_data)
                    impact.write_html(
                        str(path / f"impact_{names[0]}_{names[1]}.html")
                    )
            
            logger.info(f"Saved chain comparisons to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save comparisons: {e}")

def create_chain_comparator(
    animator: ChainAnimator,
    output_path: Optional[Path] = None
) -> ChainComparator:
    """Create chain comparator."""
    config = ComparisonConfig(output_path=output_path)
    return ChainComparator(animator, config)

if __name__ == "__main__":
    # Example usage
    from .chain_animation import create_chain_animator
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
    animator = create_chain_animator(chain_viz)
    comparator = create_chain_comparator(
        animator,
        output_path=Path("chain_comparison")
    )
    
    # Create example chains
    chain.create_chain(
        "preprocessing_a",
        [
            {
                "filter": "time_range",
                "params": {"window": 30}
            },
            {
                "filter": "confidence",
                "params": {"threshold": 0.7}
            }
        ]
    )
    
    chain.create_chain(
        "preprocessing_b",
        [
            {
                "filter": "time_range",
                "params": {"window": 60}
            },
            {
                "filter": "complexity",
                "params": {"max_complexity": 5}
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
    
    # Generate comparisons
    comparison = comparator.animate_chain_comparison(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    impact = comparator.animate_impact_comparison(
        ["preprocessing_a", "preprocessing_b"],
        data
    )
    
    # Save comparisons
    comparator.save_comparisons()
