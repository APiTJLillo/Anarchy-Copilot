"""Chain filters together for complex data transformations."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime
import networkx as nx
from collections import defaultdict

from .learning_filters import LearningFilter, FilterConfig
from .interactive_learning import InteractiveLearning

logger = logging.getLogger(__name__)

@dataclass
class ChainConfig:
    """Configuration for filter chaining."""
    max_chain_length: int = 10
    allow_cycles: bool = False
    validate_chains: bool = True
    optimize_order: bool = True
    cache_intermediate: bool = True
    output_path: Optional[Path] = None

class FilterChain:
    """Chain of filters with dependencies."""
    
    def __init__(
        self,
        filters: LearningFilter,
        config: ChainConfig
    ):
        self.filters = filters
        self.config = config
        self.chains: Dict[str, nx.DiGraph] = {}
        self.cached_results: Dict[str, Dict[str, Any]] = {}
    
    def create_chain(
        self,
        name: str,
        steps: List[Dict[str, Any]]
    ) -> nx.DiGraph:
        """Create a new filter chain."""
        if name in self.chains:
            raise ValueError(f"Chain {name} already exists")
        
        if len(steps) > self.config.max_chain_length:
            raise ValueError(f"Chain exceeds max length of {self.config.max_chain_length}")
        
        # Create dependency graph
        graph = nx.DiGraph()
        
        # Add nodes
        for i, step in enumerate(steps):
            if "filter" not in step:
                raise ValueError(f"Step {i} missing filter name")
            
            if step["filter"] not in self.filters.filters:
                raise ValueError(f"Unknown filter: {step['filter']}")
            
            graph.add_node(
                i,
                filter=step["filter"],
                params=step.get("params", {}),
                requires=step.get("requires", []),
                optional=step.get("optional", False)
            )
        
        # Add edges based on dependencies
        for i, step in enumerate(steps):
            for req in step.get("requires", []):
                if not any(s["filter"] == req for s in steps[:i]):
                    raise ValueError(f"Step {i} requires missing filter: {req}")
                
                # Find the last occurrence of required filter
                for j in range(i-1, -1, -1):
                    if steps[j]["filter"] == req:
                        graph.add_edge(j, i)
                        break
        
        # Validate graph
        if self.config.validate_chains:
            self._validate_chain(graph)
        
        # Optimize order if enabled
        if self.config.optimize_order:
            graph = self._optimize_chain_order(graph)
        
        self.chains[name] = graph
        return graph
    
    def apply_chain(
        self,
        name: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a filter chain to data."""
        if name not in self.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.chains[name]
        filtered_data = data.copy()
        
        # Get topological sort of filters
        try:
            order = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            raise ValueError("Filter chain contains cycles")
        
        # Apply filters in order
        for node in order:
            step_data = graph.nodes[node]
            filter_name = step_data["filter"]
            params = step_data["params"]
            
            try:
                # Check cache
                cache_key = self._get_cache_key(name, node, filtered_data)
                if self.config.cache_intermediate and cache_key in self.cached_results:
                    filtered_data = self.cached_results[cache_key]
                else:
                    # Apply filter
                    filtered_data = self.filters.apply_filters(
                        filtered_data,
                        [filter_name],
                        {filter_name: params}
                    )
                    
                    # Cache result
                    if self.config.cache_intermediate:
                        self.cached_results[cache_key] = filtered_data
            
            except Exception as e:
                if not step_data["optional"]:
                    raise ValueError(f"Filter {filter_name} failed: {e}")
                logger.warning(f"Optional filter {filter_name} failed: {e}")
        
        return filtered_data
    
    def merge_chains(
        self,
        name: str,
        chains: List[str]
    ) -> nx.DiGraph:
        """Merge multiple chains into one."""
        if name in self.chains:
            raise ValueError(f"Chain {name} already exists")
        
        # Combine graphs
        merged = nx.DiGraph()
        offset = 0
        mappings = {}
        
        for chain_name in chains:
            if chain_name not in self.chains:
                raise ValueError(f"Unknown chain: {chain_name}")
            
            chain = self.chains[chain_name]
            mapping = {n: n + offset for n in chain.nodes()}
            mappings[chain_name] = mapping
            
            # Add nodes
            for node in chain.nodes():
                merged.add_node(
                    mapping[node],
                    **chain.nodes[node]
                )
            
            # Add edges
            for u, v in chain.edges():
                merged.add_edge(mapping[u], mapping[v])
            
            offset += len(chain)
        
        # Validate merged chain
        if self.config.validate_chains:
            self._validate_chain(merged)
        
        # Optimize if enabled
        if self.config.optimize_order:
            merged = self._optimize_chain_order(merged)
        
        self.chains[name] = merged
        return merged
    
    def get_chain_info(
        self,
        name: str
    ) -> Dict[str, Any]:
        """Get information about a filter chain."""
        if name not in self.chains:
            raise ValueError(f"Unknown chain: {name}")
        
        graph = self.chains[name]
        
        info = {
            "name": name,
            "steps": len(graph),
            "dependencies": list(graph.edges()),
            "filters": [
                {
                    "id": node,
                    "filter": data["filter"],
                    "params": data["params"],
                    "optional": data["optional"]
                }
                for node, data in graph.nodes(data=True)
            ],
            "structure": {
                "is_dag": nx.is_directed_acyclic_graph(graph),
                "longest_path": len(nx.dag_longest_path(graph)),
                "parallel_chains": len(list(nx.weakly_connected_components(graph)))
            }
        }
        
        return info
    
    def _validate_chain(
        self,
        graph: nx.DiGraph
    ):
        """Validate a filter chain graph."""
        # Check for cycles if not allowed
        if not self.config.allow_cycles and not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Filter chain contains cycles")
        
        # Check for missing dependencies
        for node, data in graph.nodes(data=True):
            for req in data["requires"]:
                if not any(
                    graph.nodes[pred]["filter"] == req
                    for pred in nx.ancestors(graph, node)
                ):
                    raise ValueError(f"Missing required filter: {req}")
        
        # Check for conflicting filters
        seen_filters = set()
        for _, data in graph.nodes(data=True):
            filter_name = data["filter"]
            if filter_name in seen_filters and not data["optional"]:
                raise ValueError(f"Duplicate non-optional filter: {filter_name}")
            seen_filters.add(filter_name)
    
    def _optimize_chain_order(
        self,
        graph: nx.DiGraph
    ) -> nx.DiGraph:
        """Optimize the order of filters in a chain."""
        # Create new graph with optimized order
        optimized = nx.DiGraph()
        
        try:
            # Use weighted topological sort
            weights = self._calculate_filter_weights(graph)
            sorted_nodes = self._weighted_topological_sort(graph, weights)
            
            # Add nodes and edges in optimized order
            mapping = {old: new for new, old in enumerate(sorted_nodes)}
            
            for node in sorted_nodes:
                optimized.add_node(
                    mapping[node],
                    **graph.nodes[node]
                )
            
            for u, v in graph.edges():
                optimized.add_edge(mapping[u], mapping[v])
            
        except Exception as e:
            logger.warning(f"Chain optimization failed: {e}")
            return graph
        
        return optimized
    
    def _calculate_filter_weights(
        self,
        graph: nx.DiGraph
    ) -> Dict[int, float]:
        """Calculate weights for filter ordering optimization."""
        weights = {}
        
        for node in graph.nodes():
            data = graph.nodes[node]
            weight = 1.0
            
            # Adjust weight based on factors:
            # - Optional filters get lower weight
            if data["optional"]:
                weight *= 0.5
            
            # - More parameters increase weight
            weight *= (1 + 0.1 * len(data["params"]))
            
            # - More dependencies increase weight
            weight *= (1 + 0.2 * len(data["requires"]))
            
            weights[node] = weight
        
        return weights
    
    def _weighted_topological_sort(
        self,
        graph: nx.DiGraph,
        weights: Dict[int, float]
    ) -> List[int]:
        """Topologically sort nodes considering weights."""
        # Start with nodes that have no predecessors
        ready = [
            n for n in graph.nodes()
            if graph.in_degree(n) == 0
        ]
        
        # Sort by weight
        ready.sort(key=lambda x: weights[x])
        
        result = []
        visited = set()
        
        while ready:
            node = ready.pop(0)
            if node not in visited:
                result.append(node)
                visited.add(node)
                
                # Add successors whose dependencies are met
                successors = sorted(
                    [
                        n for n in graph.successors(node)
                        if all(pred in visited for pred in graph.predecessors(n))
                    ],
                    key=lambda x: weights[x]
                )
                ready.extend(successors)
        
        return result
    
    def _get_cache_key(
        self,
        chain_name: str,
        node: int,
        data: Dict[str, Any]
    ) -> str:
        """Generate cache key for intermediate results."""
        # Use chain name, node, and data hash
        data_hash = hash(str(sorted(data.items())))
        return f"{chain_name}:{node}:{data_hash}"
    
    def save_chains(
        self,
        output_path: Optional[Path] = None
    ):
        """Save filter chains to file."""
        path = output_path or self.config.output_path
        if not path:
            return
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            
            # Save each chain
            for name, graph in self.chains.items():
                chain_path = path / f"{name}.json"
                
                # Convert graph to serializable format
                chain_data = {
                    "nodes": {
                        str(n): d
                        for n, d in graph.nodes(data=True)
                    },
                    "edges": [
                        [str(u), str(v)]
                        for u, v in graph.edges()
                    ]
                }
                
                with open(chain_path, "w") as f:
                    json.dump(chain_data, f, indent=2)
            
            logger.info(f"Saved filter chains to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save filter chains: {e}")
    
    def load_chains(
        self,
        input_path: Optional[Path] = None
    ):
        """Load filter chains from file."""
        path = input_path or self.config.output_path
        if not path:
            return
        
        try:
            # Load each chain file
            for chain_file in path.glob("*.json"):
                with open(chain_file) as f:
                    chain_data = json.load(f)
                
                # Create graph from data
                graph = nx.DiGraph()
                
                # Add nodes
                for node, data in chain_data["nodes"].items():
                    graph.add_node(int(node), **data)
                
                # Add edges
                for u, v in chain_data["edges"]:
                    graph.add_edge(int(u), int(v))
                
                name = chain_file.stem
                self.chains[name] = graph
            
            logger.info(f"Loaded filter chains from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load filter chains: {e}")

def create_filter_chain(
    filters: LearningFilter,
    output_path: Optional[Path] = None
) -> FilterChain:
    """Create filter chain."""
    config = ChainConfig(output_path=output_path)
    return FilterChain(filters, config)

if __name__ == "__main__":
    # Example usage
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
    chain = create_filter_chain(
        filters,
        output_path=Path("filter_chains")
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
    
    # Create filter chain
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
    
    # Apply chain
    filtered_data = chain.apply_chain("preprocessing", data)
    
    # Get chain info
    info = chain.get_chain_info("preprocessing")
    print(json.dumps(info, indent=2))
    
    # Save chains
    chain.save_chains()
