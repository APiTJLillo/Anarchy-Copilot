"""Analysis tools for pattern compositions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd
import networkx as nx
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .pattern_composition import PatternComposer, CompositionConfig
from .event_scheduler import ScheduledEvent, AnimationEvent

logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for composition analysis."""
    min_interval: float = 0.001
    max_depth: int = 10
    time_resolution: float = 0.01
    detect_cycles: bool = True
    analyze_timing: bool = True
    generate_stats: bool = True
    output_path: Optional[Path] = None

class CompositionAnalysis:
    """Analyze pattern compositions."""
    
    def __init__(
        self,
        composer: PatternComposer,
        config: AnalysisConfig
    ):
        self.composer = composer
        self.config = config
        self.results: Dict[str, Any] = {}
    
    def analyze_composition(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze pattern composition."""
        results = {
            "timing": self.analyze_timing(composition),
            "structure": self.analyze_structure(composition),
            "statistics": self.analyze_statistics(composition),
            "dependencies": self.analyze_dependencies(composition),
            "efficiency": self.analyze_efficiency(composition),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store results
        self.results = results
        
        # Save if configured
        if self.config.output_path:
            self.save_analysis()
        
        return results
    
    def analyze_timing(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze timing characteristics."""
        if not composition:
            return {}
        
        # Get timestamps
        timestamps = [
            event.trigger_time.timestamp()
            for event in composition
        ]
        
        # Calculate intervals
        intervals = np.diff(timestamps)
        
        return {
            "duration": max(timestamps) - min(timestamps),
            "mean_interval": float(np.mean(intervals)) if len(intervals) > 0 else 0,
            "std_interval": float(np.std(intervals)) if len(intervals) > 0 else 0,
            "min_interval": float(np.min(intervals)) if len(intervals) > 0 else 0,
            "max_interval": float(np.max(intervals)) if len(intervals) > 0 else 0,
            "histogram": self._create_interval_histogram(intervals),
            "timeline": self._create_event_timeline(composition)
        }
    
    def analyze_structure(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze composition structure."""
        # Build event graph
        graph = self._build_event_graph(composition)
        
        # Analyze structure
        return {
            "event_count": len(composition),
            "depth": self._calculate_depth(graph),
            "branching": self._calculate_branching(graph),
            "cycles": self._detect_cycles(graph) if self.config.detect_cycles else [],
            "components": list(nx.connected_components(graph.to_undirected())),
            "critical_path": self._find_critical_path(graph, composition),
            "topology": self._analyze_topology(graph)
        }
    
    def analyze_statistics(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze composition statistics."""
        event_types = defaultdict(int)
        condition_count = 0
        repeat_count = 0
        data_sizes = []
        
        for event in composition:
            # Count event types
            event_name = (
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
            )
            event_types[event_name] += 1
            
            # Count conditions and repeats
            if event.condition:
                condition_count += 1
            if event.repeat:
                repeat_count += 1
            
            # Measure data size
            if isinstance(event.event, AnimationEvent) and event.event.data:
                data_sizes.append(len(json.dumps(event.event.data)))
        
        return {
            "event_types": dict(event_types),
            "condition_ratio": condition_count / len(composition) if composition else 0,
            "repeat_ratio": repeat_count / len(composition) if composition else 0,
            "avg_data_size": np.mean(data_sizes) if data_sizes else 0,
            "type_distribution": self._create_type_distribution(event_types)
        }
    
    def analyze_dependencies(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze event dependencies."""
        # Build dependency graph
        graph = self._build_dependency_graph(composition)
        
        return {
            "dependencies": [
                {
                    "source": u,
                    "target": v,
                    "weight": d["weight"]
                }
                for u, v, d in graph.edges(data=True)
            ],
            "bottlenecks": self._find_bottlenecks(graph),
            "dependency_chains": self._find_dependency_chains(graph),
            "concurrent_groups": self._find_concurrent_groups(graph, composition)
        }
    
    def analyze_efficiency(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, Any]:
        """Analyze composition efficiency."""
        # Calculate timing efficiency
        timing = self.analyze_timing(composition)
        total_duration = timing["duration"]
        active_time = sum(
            e.interval or 0
            for e in composition
            if e.interval
        )
        
        # Calculate resource utilization
        concurrent_events = self._analyze_concurrency(composition)
        max_concurrent = max(concurrent_events.values())
        
        return {
            "timing_efficiency": active_time / total_duration if total_duration > 0 else 0,
            "resource_utilization": np.mean(list(concurrent_events.values())),
            "max_concurrency": max_concurrent,
            "optimization_opportunities": self._find_optimization_opportunities(composition),
            "concurrency_timeline": self._create_concurrency_timeline(concurrent_events)
        }
    
    def _create_interval_histogram(
        self,
        intervals: np.ndarray
    ) -> Dict[str, List[float]]:
        """Create histogram of event intervals."""
        if len(intervals) == 0:
            return {"bins": [], "counts": []}
        
        hist, bins = np.histogram(
            intervals,
            bins="auto"
        )
        return {
            "bins": bins.tolist(),
            "counts": hist.tolist()
        }
    
    def _create_event_timeline(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[str, List[float]]:
        """Create timeline of events."""
        if not composition:
            return {"times": [], "events": []}
        
        start_time = min(
            event.trigger_time.timestamp()
            for event in composition
        )
        
        return {
            "times": [
                event.trigger_time.timestamp() - start_time
                for event in composition
            ],
            "events": [
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
                for event in composition
            ]
        }
    
    def _build_event_graph(
        self,
        composition: List[ScheduledEvent]
    ) -> nx.DiGraph:
        """Build graph of event relationships."""
        graph = nx.DiGraph()
        
        # Add nodes
        for i, event in enumerate(composition):
            name = (
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
            )
            graph.add_node(
                i,
                name=name,
                time=event.trigger_time.timestamp(),
                data=event.event.data if isinstance(event.event, AnimationEvent) else None
            )
        
        # Add edges based on timing and dependencies
        for i, event_i in enumerate(composition):
            for j, event_j in enumerate(composition):
                if i != j:
                    # Connect events based on timing
                    if event_i.trigger_time < event_j.trigger_time:
                        time_diff = (
                            event_j.trigger_time - event_i.trigger_time
                        ).total_seconds()
                        if time_diff < self.config.min_interval:
                            graph.add_edge(i, j, weight=time_diff)
        
        return graph
    
    def _calculate_depth(
        self,
        graph: nx.DiGraph
    ) -> int:
        """Calculate maximum depth of event graph."""
        if not graph:
            return 0
        
        try:
            return max(
                len(path)
                for path in nx.all_simple_paths(
                    graph,
                    min(graph.nodes()),
                    max(graph.nodes())
                )
            )
        except nx.NetworkXNoPath:
            return 0
    
    def _calculate_branching(
        self,
        graph: nx.DiGraph
    ) -> float:
        """Calculate average branching factor."""
        if not graph:
            return 0
            
        out_degrees = [
            d for _, d in graph.out_degree()
        ]
        return np.mean(out_degrees) if out_degrees else 0
    
    def _detect_cycles(
        self,
        graph: nx.DiGraph
    ) -> List[List[int]]:
        """Detect cycles in event graph."""
        return list(nx.simple_cycles(graph))
    
    def _find_critical_path(
        self,
        graph: nx.DiGraph,
        composition: List[ScheduledEvent]
    ) -> List[int]:
        """Find critical path through events."""
        if not graph:
            return []
            
        try:
            # Use longest path as critical path
            longest_path = nx.dag_longest_path(graph)
            return longest_path
        except nx.NetworkXUnfeasible:
            return []
    
    def _analyze_topology(
        self,
        graph: nx.DiGraph
    ) -> Dict[str, Any]:
        """Analyze graph topology."""
        return {
            "density": nx.density(graph),
            "degree_centrality": nx.degree_centrality(graph),
            "betweenness_centrality": nx.betweenness_centrality(graph),
            "strongly_connected": list(nx.strongly_connected_components(graph))
        }
    
    def _create_type_distribution(
        self,
        event_types: Dict[str, int]
    ) -> Dict[str, List[Any]]:
        """Create event type distribution data."""
        return {
            "types": list(event_types.keys()),
            "counts": list(event_types.values())
        }
    
    def _build_dependency_graph(
        self,
        composition: List[ScheduledEvent]
    ) -> nx.DiGraph:
        """Build graph of event dependencies."""
        graph = nx.DiGraph()
        
        for i, event_i in enumerate(composition):
            graph.add_node(i)
            
            for j, event_j in enumerate(composition):
                if i < j:  # Only look at future events
                    # Check timing dependency
                    time_diff = (
                        event_j.trigger_time - event_i.trigger_time
                    ).total_seconds()
                    
                    if time_diff < self.config.min_interval:
                        graph.add_edge(i, j, weight=time_diff)
        
        return graph
    
    def _find_bottlenecks(
        self,
        graph: nx.DiGraph
    ) -> List[int]:
        """Find bottleneck events."""
        if not graph:
            return []
            
        # Use betweenness centrality to identify bottlenecks
        centrality = nx.betweenness_centrality(graph)
        threshold = np.mean(list(centrality.values())) + np.std(list(centrality.values()))
        
        return [
            node
            for node, cent in centrality.items()
            if cent > threshold
        ]
    
    def _find_dependency_chains(
        self,
        graph: nx.DiGraph
    ) -> List[List[int]]:
        """Find chains of dependent events."""
        if not graph:
            return []
            
        try:
            # Find all paths longer than 2 events
            chains = []
            for source in graph.nodes():
                for target in graph.nodes():
                    if source != target:
                        paths = list(nx.all_simple_paths(graph, source, target))
                        chains.extend([p for p in paths if len(p) > 2])
            return chains
        except nx.NetworkXNoPath:
            return []
    
    def _find_concurrent_groups(
        self,
        graph: nx.DiGraph,
        composition: List[ScheduledEvent]
    ) -> List[List[int]]:
        """Find groups of concurrent events."""
        if not composition:
            return []
            
        groups = []
        current_group = []
        last_time = None
        
        for i, event in enumerate(composition):
            current_time = event.trigger_time
            
            if (
                last_time is None or
                (current_time - last_time).total_seconds() < self.config.min_interval
            ):
                current_group.append(i)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [i]
            
            last_time = current_time
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _analyze_concurrency(
        self,
        composition: List[ScheduledEvent]
    ) -> Dict[float, int]:
        """Analyze event concurrency over time."""
        if not composition:
            return {}
            
        # Create timeline of concurrent events
        timeline = defaultdict(int)
        start_time = min(e.trigger_time.timestamp() for e in composition)
        end_time = max(e.trigger_time.timestamp() for e in composition)
        
        current_time = start_time
        while current_time <= end_time:
            # Count active events at this time
            active = sum(
                1 for event in composition
                if (
                    event.trigger_time.timestamp() <= current_time and
                    (
                        not event.interval or
                        event.trigger_time.timestamp() + event.interval >= current_time
                    )
                )
            )
            timeline[current_time] = active
            current_time += self.config.time_resolution
        
        return timeline
    
    def _find_optimization_opportunities(
        self,
        composition: List[ScheduledEvent]
    ) -> List[Dict[str, Any]]:
        """Find potential optimization opportunities."""
        opportunities = []
        
        # Analyze timing gaps
        for i in range(len(composition) - 1):
            gap = (
                composition[i + 1].trigger_time -
                composition[i].trigger_time
            ).total_seconds()
            
            if gap > self.config.min_interval * 10:
                opportunities.append({
                    "type": "timing_gap",
                    "index": i,
                    "gap": gap,
                    "suggestion": "Consider reducing delay between events"
                })
        
        # Find redundant events
        event_counts = defaultdict(list)
        for i, event in enumerate(composition):
            name = (
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
            )
            event_counts[name].append(i)
        
        for name, indices in event_counts.items():
            if len(indices) > 1:
                opportunities.append({
                    "type": "redundant_events",
                    "name": name,
                    "count": len(indices),
                    "indices": indices,
                    "suggestion": "Consider combining redundant events"
                })
        
        return opportunities
    
    def _create_concurrency_timeline(
        self,
        concurrent_events: Dict[float, int]
    ) -> Dict[str, List[float]]:
        """Create timeline of concurrent events."""
        return {
            "times": list(concurrent_events.keys()),
            "counts": list(concurrent_events.values())
        }
    
    def visualize_analysis(
        self,
        analysis_results: Optional[Dict[str, Any]] = None
    ) -> go.Figure:
        """Create visualization of analysis results."""
        if analysis_results is None:
            analysis_results = self.results
        
        if not analysis_results:
            return go.Figure()
        
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Event Timeline",
                "Interval Distribution",
                "Event Type Distribution",
                "Concurrency Timeline",
                "Dependency Graph",
                "Critical Path"
            ]
        )
        
        # Add event timeline
        timeline = analysis_results["timing"]["timeline"]
        fig.add_trace(
            go.Scatter(
                x=timeline["times"],
                y=timeline["events"],
                mode="markers+lines",
                name="Events"
            ),
            row=1,
            col=1
        )
        
        # Add interval histogram
        hist = analysis_results["timing"]["histogram"]
        fig.add_trace(
            go.Bar(
                x=hist["bins"][:-1],
                y=hist["counts"],
                name="Intervals"
            ),
            row=1,
            col=2
        )
        
        # Add type distribution
        types = analysis_results["statistics"]["type_distribution"]
        fig.add_trace(
            go.Bar(
                x=types["types"],
                y=types["counts"],
                name="Event Types"
            ),
            row=2,
            col=1
        )
        
        # Add concurrency timeline
        concurrency = analysis_results["efficiency"]["concurrency_timeline"]
        fig.add_trace(
            go.Scatter(
                x=concurrency["times"],
                y=concurrency["counts"],
                mode="lines",
                name="Concurrent Events"
            ),
            row=2,
            col=2
        )
        
        # Add dependency graph visualization
        dependencies = analysis_results["dependencies"]["dependencies"]
        pos = nx.spring_layout(nx.Graph(dependencies))
        
        edge_x = []
        edge_y = []
        for edge in dependencies:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                name="Dependencies"
            ),
            row=3,
            col=1
        )
        
        # Add critical path
        critical_path = analysis_results["structure"]["critical_path"]
        if critical_path:
            path_x = []
            path_y = []
            for i in range(len(critical_path) - 1):
                x0, y0 = pos[critical_path[i]]
                x1, y1 = pos[critical_path[i + 1]]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])
                
            fig.add_trace(
                go.Scatter(
                    x=path_x,
                    y=path_y,
                    mode="lines+markers",
                    line=dict(color="red", width=2),
                    name="Critical Path"
                ),
                row=3,
                col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1000,
            showlegend=True,
            title_text="Composition Analysis"
        )
        
        return fig
    
    def save_analysis(self):
        """Save analysis results."""
        if not self.config.output_path or not self.results:
            return
        
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save results
            results_file = output_path / "composition_analysis.json"
            with open(results_file, "w") as f:
                json.dump(self.results, f, indent=2)
            
            # Save visualization
            viz_file = output_path / "composition_analysis.html"
            fig = self.visualize_analysis()
            fig.write_html(str(viz_file))
            
            logger.info(f"Saved analysis results to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_composition_analysis(
    composer: PatternComposer,
    output_path: Optional[Path] = None
) -> CompositionAnalysis:
    """Create composition analysis."""
    config = AnalysisConfig(output_path=output_path)
    return CompositionAnalysis(composer, config)

if __name__ == "__main__":
    # Example usage
    from .pattern_composition import create_pattern_composer
    from .scheduling_patterns import create_scheduling_pattern
    from .event_scheduler import create_event_scheduler
    from .animation_events import create_event_manager
    from .animation_controls import create_animation_controls
    from .interactive_easing import create_interactive_easing
    from .easing_visualization import create_easing_visualizer
    from .easing_transitions import create_easing_functions
    
    async def main():
        # Create components
        easing = create_easing_functions()
        visualizer = create_easing_visualizer(easing)
        interactive = create_interactive_easing(visualizer)
        controls = create_animation_controls(interactive)
        events = create_event_manager(controls)
        scheduler = create_event_scheduler(events)
        pattern = create_scheduling_pattern(scheduler)
        composer = create_pattern_composer(pattern)
        analyzer = create_composition_analysis(
            composer,
            output_path=Path("composition_analysis")
        )
        
        # Create example composition
        events_a = ["animation:start", "progress:update"]
        events_b = ["animation:pause", "animation:resume"]
        
        sequence_a = pattern.sequence(events_a)
        sequence_b = pattern.sequence(events_b)
        
        composition = composer.compose(
            "chain",
            [sequence_a, sequence_b],
            delay=1.0,
            gap=0.5
        )
        
        # Analyze composition
        results = analyzer.analyze_composition(composition)
        print(json.dumps(results, indent=2))
        
        # Generate visualization
        fig = analyzer.visualize_analysis()
        fig.show()
    
    asyncio.run(main())
