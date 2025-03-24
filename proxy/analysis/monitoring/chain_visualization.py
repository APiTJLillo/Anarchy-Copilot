"""Visualization tools for filter chains."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd

from .filter_chaining import FilterChain, ChainConfig
from .learning_filters import LearningFilter

logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration for chain visualization."""
    width: int = 1200
    height: int = 800
    dark_mode: bool = False
    node_size: int = 30
    edge_width: float = 1.5
    show_labels: bool = True
    interactive: bool = True
    output_path: Optional[Path] = None

class ChainVisualizer:
    """Visualize filter chains."""
    
    def __init__(
        self,
        chain: FilterChain,
        config: VisualizationConfig
    ):
        self.chain = chain
        self.config = config
    
    def visualize_chain(
        self,
        name: str
    ) -> go.Figure:
        """Create chain structure visualization."""
        if name not in self.chain.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.chain.chains[name]
        pos = nx.spring_layout(graph)
        
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        edge_text = []
        
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"Dependency: {edge[0]} -> {edge[1]}")
        
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(
                    width=self.config.edge_width,
                    color="#888"
                ),
                hoverinfo="text",
                text=edge_text,
                name="Dependencies"
            )
        )
        
        # Add nodes
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node, data in graph.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Generate node text
            text = (
                f"Step {node}<br>"
                f"Filter: {data['filter']}<br>"
                f"Parameters: {data['params']}<br>"
                f"Optional: {data['optional']}"
            )
            node_text.append(text)
            
            # Color based on properties
            color = (
                "lightblue" if data["optional"]
                else "orange" if data["requires"]
                else "lightgreen"
            )
            node_color.append(color)
        
        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text" if self.config.show_labels else "markers",
                marker=dict(
                    size=self.config.node_size,
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
        
        # Update layout
        fig.update_layout(
            title=f"Filter Chain: {name}",
            showlegend=True,
            hovermode="closest",
            width=self.config.width,
            height=self.config.height,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def visualize_chain_stats(
        self,
        name: str
    ) -> go.Figure:
        """Create chain statistics visualization."""
        if name not in self.chain.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.chain.chains[name]
        info = self.chain.get_chain_info(name)
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Filter Distribution",
                "Dependency Structure",
                "Chain Properties",
                "Node Metrics"
            ]
        )
        
        # Filter distribution
        filter_counts = defaultdict(int)
        for _, data in graph.nodes(data=True):
            filter_counts[data["filter"]] += 1
        
        fig.add_trace(
            go.Bar(
                x=list(filter_counts.keys()),
                y=list(filter_counts.values()),
                name="Filter Types"
            ),
            row=1,
            col=1
        )
        
        # Dependency structure
        degrees = [d for _, d in graph.degree()]
        fig.add_trace(
            go.Histogram(
                x=degrees,
                nbinsx=max(degrees) + 1,
                name="Node Degrees"
            ),
            row=1,
            col=2
        )
        
        # Chain properties
        properties = {
            "Steps": info["steps"],
            "Dependencies": len(info["dependencies"]),
            "Parallel Chains": info["structure"]["parallel_chains"],
            "Path Length": info["structure"]["longest_path"]
        }
        
        fig.add_trace(
            go.Bar(
                x=list(properties.keys()),
                y=list(properties.values()),
                name="Properties"
            ),
            row=2,
            col=1
        )
        
        # Node metrics
        metrics = self._calculate_node_metrics(graph)
        fig.add_trace(
            go.Scatter(
                x=list(range(len(metrics["centrality"]))),
                y=metrics["centrality"],
                mode="lines+markers",
                name="Node Centrality"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def visualize_chain_execution(
        self,
        name: str,
        data: Dict[str, Any]
    ) -> go.Figure:
        """Visualize chain execution results."""
        if name not in self.chain.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        # Apply chain and collect metrics
        start_time = datetime.now()
        filtered_data = self.chain.apply_chain(name, data)
        end_time = datetime.now()
        
        # Calculate execution metrics
        metrics = self._calculate_execution_metrics(
            name,
            data,
            filtered_data,
            end_time - start_time
        )
        
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Data Reduction",
                "Execution Time",
                "Filter Impact",
                "Data Distribution"
            ]
        )
        
        # Data reduction
        fig.add_trace(
            go.Bar(
                x=["Original", "Filtered"],
                y=[metrics["original_size"], metrics["filtered_size"]],
                name="Data Size"
            ),
            row=1,
            col=1
        )
        
        # Execution time
        fig.add_trace(
            go.Bar(
                x=list(metrics["step_times"].keys()),
                y=list(metrics["step_times"].values()),
                name="Step Times"
            ),
            row=1,
            col=2
        )
        
        # Filter impact
        fig.add_trace(
            go.Bar(
                x=list(metrics["filter_impact"].keys()),
                y=list(metrics["filter_impact"].values()),
                name="Filter Impact"
            ),
            row=2,
            col=1
        )
        
        # Data distribution
        if isinstance(filtered_data, pd.DataFrame):
            for column in filtered_data.columns:
                fig.add_trace(
                    go.Histogram(
                        x=filtered_data[column],
                        name=column,
                        opacity=0.7
                    ),
                    row=2,
                    col=2
                )
        
        # Update layout
        fig.update_layout(
            height=self.config.height,
            width=self.config.width,
            template="plotly_dark" if self.config.dark_mode else "plotly_white",
            showlegend=True
        )
        
        return fig
    
    def _calculate_node_metrics(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, List[float]]:
        """Calculate various node metrics."""
        metrics = {
            "centrality": list(nx.betweenness_centrality(graph).values()),
            "in_degree": [d for _, d in graph.in_degree()],
            "out_degree": [d for _, d in graph.out_degree()]
        }
        
        return metrics
    
    def _calculate_execution_metrics(
        self,
        name: str,
        original_data: Dict[str, Any],
        filtered_data: Dict[str, Any],
        duration: timedelta
    ) -> Dict[str, Any]:
        """Calculate execution metrics."""
        graph = self.chain.chains[name]
        
        metrics = {
            "original_size": self.chain._get_data_size(original_data),
            "filtered_size": self.chain._get_data_size(filtered_data),
            "total_time": duration.total_seconds(),
            "step_times": {},
            "filter_impact": {}
        }
        
        # Calculate per-step metrics
        for node, data in graph.nodes(data=True):
            filter_name = data["filter"]
            cache_key = self.chain._get_cache_key(name, node, original_data)
            
            if cache_key in self.chain.cached_results:
                step_data = self.chain.cached_results[cache_key]
                step_size = self.chain._get_data_size(step_data)
                
                metrics["step_times"][f"Step {node}"] = duration.total_seconds() / len(graph)
                metrics["filter_impact"][filter_name] = 1 - (step_size / metrics["original_size"])
        
        return metrics
    
    def save_visualizations(
        self,
        output_path: Optional[Path] = None
    ):
        """Save all chain visualizations."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            for name in self.chain.chains:
                # Save chain structure
                structure = self.visualize_chain(name)
                structure.write_html(
                    str(path / f"{name}_structure.html")
                )
                
                # Save chain stats
                stats = self.visualize_chain_stats(name)
                stats.write_html(
                    str(path / f"{name}_stats.html")
                )
            
            logger.info(f"Saved chain visualizations to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save visualizations: {e}")

def create_chain_visualizer(
    chain: FilterChain,
    output_path: Optional[Path] = None
) -> ChainVisualizer:
    """Create chain visualizer."""
    config = VisualizationConfig(output_path=output_path)
    return ChainVisualizer(chain, config)

if __name__ == "__main__":
    # Example usage
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
    chain_viz = create_chain_visualizer(
        chain,
        output_path=Path("chain_visualization")
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
    
    # Generate visualizations
    structure = chain_viz.visualize_chain("preprocessing")
    stats = chain_viz.visualize_chain_stats("preprocessing")
    execution = chain_viz.visualize_chain_execution("preprocessing", data)
    
    # Save visualizations
    chain_viz.save_visualizations()
