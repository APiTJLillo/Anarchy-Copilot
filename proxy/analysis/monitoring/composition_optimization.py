"""Optimization suggestions for pattern compositions."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import networkx as nx
from collections import defaultdict

from .composition_analysis import CompositionAnalysis, AnalysisConfig
from .event_scheduler import ScheduledEvent, AnimationEvent

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for composition optimization."""
    min_gain: float = 0.1
    max_suggestions: int = 10
    complexity_limit: int = 20
    priority_levels: int = 3
    enable_learning: bool = True
    store_history: bool = True
    output_path: Optional[Path] = None

@dataclass
class OptimizationSuggestion:
    """Suggested optimization for composition."""
    type: str
    description: str
    impact: float
    complexity: int
    priority: int
    before: Optional[List[ScheduledEvent]] = None
    after: Optional[List[ScheduledEvent]] = None
    rationale: Optional[str] = None

class CompositionOptimizer:
    """Optimize pattern compositions."""
    
    def __init__(
        self,
        analyzer: CompositionAnalysis,
        config: OptimizationConfig
    ):
        self.analyzer = analyzer
        self.config = config
        self.optimizations: Dict[str, Callable] = {}
        self.history: List[Dict[str, Any]] = []
        
        self.register_default_optimizers()
    
    def register_default_optimizers(self):
        """Register default optimization strategies."""
        self.optimizations.update({
            "merge_events": self.suggest_event_merging,
            "reduce_gaps": self.suggest_gap_reduction,
            "parallelize": self.suggest_parallelization,
            "batch_events": self.suggest_batching,
            "reorder": self.suggest_reordering,
            "simplify": self.suggest_simplification,
            "remove_redundancy": self.suggest_redundancy_removal
        })
    
    def register_optimizer(
        self,
        name: str,
        func: Callable
    ):
        """Register custom optimization strategy."""
        self.optimizations[name] = func
    
    def analyze_and_suggest(
        self,
        composition: List[ScheduledEvent]
    ) -> List[OptimizationSuggestion]:
        """Analyze composition and generate optimization suggestions."""
        # Analyze composition
        analysis = self.analyzer.analyze_composition(composition)
        
        # Generate suggestions from all optimizers
        suggestions = []
        for name, optimizer in self.optimizations.items():
            try:
                optimizer_suggestions = optimizer(composition, analysis)
                suggestions.extend(optimizer_suggestions)
            except Exception as e:
                logger.error(f"Optimizer {name} failed: {e}")
        
        # Filter and sort suggestions
        suggestions = self._filter_suggestions(suggestions)
        suggestions = self._prioritize_suggestions(suggestions)
        
        # Store in history
        if self.config.store_history:
            self._store_suggestions(suggestions, composition, analysis)
        
        return suggestions[:self.config.max_suggestions]
    
    def suggest_event_merging(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest event merging opportunities."""
        suggestions = []
        
        # Find events close in time
        concurrent_groups = analysis["dependencies"]["concurrent_groups"]
        for group in concurrent_groups:
            if len(group) > 1:
                events = [composition[i] for i in group]
                
                # Check if events can be merged
                if self._can_merge_events(events):
                    suggestions.append(
                        OptimizationSuggestion(
                            type="merge_events",
                            description=f"Merge {len(events)} events at {events[0].trigger_time}",
                            impact=0.2 * len(events),
                            complexity=1,
                            priority=2,
                            before=events,
                            after=[self._merge_events(events)],
                            rationale="Merging concurrent events reduces overhead"
                        )
                    )
        
        return suggestions
    
    def suggest_gap_reduction(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest reducing gaps between events."""
        suggestions = []
        
        timing = analysis["timing"]
        mean_interval = timing["mean_interval"]
        max_interval = timing["max_interval"]
        
        # Look for large gaps
        for i in range(len(composition) - 1):
            gap = (
                composition[i + 1].trigger_time -
                composition[i].trigger_time
            ).total_seconds()
            
            if gap > mean_interval * 2:
                reduced_gap = max(mean_interval, gap * 0.5)
                impact = (gap - reduced_gap) / max_interval
                
                suggestions.append(
                    OptimizationSuggestion(
                        type="reduce_gap",
                        description=f"Reduce gap between events {i} and {i+1}",
                        impact=impact,
                        complexity=1,
                        priority=1,
                        rationale=f"Gap of {gap:.2f}s can be reduced to {reduced_gap:.2f}s"
                    )
                )
        
        return suggestions
    
    def suggest_parallelization(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest parallelization opportunities."""
        suggestions = []
        
        # Find independent event chains
        dependencies = analysis["dependencies"]
        chains = dependencies["dependency_chains"]
        
        # Look for chains that can be parallelized
        for chain in chains:
            if len(chain) > 2:
                events = [composition[i] for i in chain]
                if self._can_parallelize_events(events):
                    suggestions.append(
                        OptimizationSuggestion(
                            type="parallelize",
                            description=f"Parallelize chain of {len(events)} events",
                            impact=0.3 * len(events),
                            complexity=2,
                            priority=3,
                            before=events,
                            rationale="Independent events can be executed in parallel"
                        )
                    )
        
        return suggestions
    
    def suggest_batching(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest event batching opportunities."""
        suggestions = []
        
        # Find similar events
        event_types = analysis["statistics"]["event_types"]
        for event_type, count in event_types.items():
            if count > 3:
                events = [
                    event for event in composition
                    if (
                        isinstance(event.event, AnimationEvent) and
                        event.event.name == event_type
                    ) or event.event == event_type
                ]
                
                suggestions.append(
                    OptimizationSuggestion(
                        type="batch_events",
                        description=f"Batch {count} {event_type} events",
                        impact=0.15 * count,
                        complexity=2,
                        priority=2,
                        before=events,
                        rationale="Similar events can be batched for efficiency"
                    )
                )
        
        return suggestions
    
    def suggest_reordering(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest event reordering opportunities."""
        suggestions = []
        
        # Find bottlenecks
        bottlenecks = analysis["dependencies"]["bottlenecks"]
        if bottlenecks:
            bottleneck_events = [composition[i] for i in bottlenecks]
            
            suggestions.append(
                OptimizationSuggestion(
                    type="reorder",
                    description=f"Reorder {len(bottlenecks)} bottleneck events",
                    impact=0.25 * len(bottlenecks),
                    complexity=3,
                    priority=2,
                    before=bottleneck_events,
                    rationale="Reordering bottleneck events can improve flow"
                )
            )
        
        return suggestions
    
    def suggest_simplification(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest composition simplification opportunities."""
        suggestions = []
        
        # Check structural complexity
        structure = analysis["structure"]
        if structure["depth"] > 5 or structure["branching"] > 3:
            suggestions.append(
                OptimizationSuggestion(
                    type="simplify",
                    description="Simplify composition structure",
                    impact=0.2,
                    complexity=3,
                    priority=1,
                    rationale=f"Structure depth {structure['depth']} and branching {structure['branching']:.1f} are high"
                )
            )
        
        # Check cycles
        cycles = structure["cycles"]
        if cycles:
            suggestions.append(
                OptimizationSuggestion(
                    type="simplify",
                    description=f"Remove {len(cycles)} event cycles",
                    impact=0.3,
                    complexity=3,
                    priority=3,
                    rationale="Event cycles can cause timing issues"
                )
            )
        
        return suggestions
    
    def suggest_redundancy_removal(
        self,
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ) -> List[OptimizationSuggestion]:
        """Suggest redundancy removal opportunities."""
        suggestions = []
        
        # Find redundant events
        seen_events: Dict[str, Set[int]] = defaultdict(set)
        for i, event in enumerate(composition):
            event_key = (
                event.event.name if isinstance(event.event, AnimationEvent)
                else event.event
            )
            seen_events[event_key].add(i)
        
        for event_key, indices in seen_events.items():
            if len(indices) > 1:
                events = [composition[i] for i in indices]
                if self._are_events_redundant(events):
                    suggestions.append(
                        OptimizationSuggestion(
                            type="remove_redundancy",
                            description=f"Remove {len(indices)-1} redundant {event_key} events",
                            impact=0.1 * (len(indices) - 1),
                            complexity=2,
                            priority=2,
                            before=events,
                            rationale="Redundant events can be combined or removed"
                        )
                    )
        
        return suggestions
    
    def _filter_suggestions(
        self,
        suggestions: List[OptimizationSuggestion]
    ) -> List[OptimizationSuggestion]:
        """Filter suggestions based on impact and complexity."""
        return [
            s for s in suggestions
            if (
                s.impact >= self.config.min_gain and
                s.complexity <= self.config.complexity_limit
            )
        ]
    
    def _prioritize_suggestions(
        self,
        suggestions: List[OptimizationSuggestion]
    ) -> List[OptimizationSuggestion]:
        """Prioritize suggestions based on impact/complexity ratio."""
        return sorted(
            suggestions,
            key=lambda s: (s.impact / s.complexity, -s.complexity),
            reverse=True
        )
    
    def _store_suggestions(
        self,
        suggestions: List[OptimizationSuggestion],
        composition: List[ScheduledEvent],
        analysis: Dict[str, Any]
    ):
        """Store suggestions in history."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "composition_size": len(composition),
            "suggestions": [
                {
                    "type": s.type,
                    "description": s.description,
                    "impact": s.impact,
                    "complexity": s.complexity,
                    "priority": s.priority,
                    "rationale": s.rationale
                }
                for s in suggestions
            ],
            "analysis_summary": {
                "timing_efficiency": analysis["efficiency"]["timing_efficiency"],
                "resource_utilization": analysis["efficiency"]["resource_utilization"],
                "event_count": analysis["structure"]["event_count"],
                "depth": analysis["structure"]["depth"]
            }
        }
        
        self.history.append(entry)
        
        # Save to file if configured
        if self.config.output_path:
            self._save_history()
    
    def _save_history(self):
        """Save optimization history to file."""
        try:
            output_path = self.config.output_path
            output_path.mkdir(parents=True, exist_ok=True)
            
            history_file = output_path / "optimization_history.json"
            with open(history_file, "w") as f:
                json.dump(self.history, f, indent=2)
            
            logger.info(f"Saved optimization history to {history_file}")
            
        except Exception as e:
            logger.error(f"Failed to save optimization history: {e}")
    
    def _can_merge_events(
        self,
        events: List[ScheduledEvent]
    ) -> bool:
        """Check if events can be merged."""
        if not events:
            return False
        
        # Check timing proximity
        times = [e.trigger_time.timestamp() for e in events]
        if max(times) - min(times) > self.analyzer.config.min_interval:
            return False
        
        # Check for conflicting conditions or intervals
        has_condition = any(e.condition for e in events)
        has_interval = any(e.interval for e in events)
        if has_condition and has_interval:
            return False
        
        return True
    
    def _merge_events(
        self,
        events: List[ScheduledEvent]
    ) -> ScheduledEvent:
        """Merge multiple events into one."""
        if not events:
            return None
        
        # Combine event data
        merged_data = {}
        for event in events:
            if isinstance(event.event, AnimationEvent):
                merged_data.update(event.event.data or {})
        
        # Create merged event
        return ScheduledEvent(
            event=events[0].event,
            trigger_time=events[0].trigger_time,
            data=merged_data,
            condition=next((e.condition for e in events if e.condition), None),
            repeat=any(e.repeat for e in events),
            interval=max(e.interval or 0 for e in events)
        )
    
    def _can_parallelize_events(
        self,
        events: List[ScheduledEvent]
    ) -> bool:
        """Check if events can be parallelized."""
        if not events:
            return False
        
        # Check for dependencies
        graph = self.analyzer._build_dependency_graph(events)
        return not nx.is_directed_acyclic_graph(graph)
    
    def _are_events_redundant(
        self,
        events: List[ScheduledEvent]
    ) -> bool:
        """Check if events are redundant."""
        if len(events) < 2:
            return False
        
        # Compare event properties
        base = events[0]
        return all(
            e.event == base.event and
            e.condition == base.condition and
            e.repeat == base.repeat and
            e.interval == base.interval
            for e in events[1:]
        )

def create_composition_optimizer(
    analyzer: CompositionAnalysis,
    output_path: Optional[Path] = None
) -> CompositionOptimizer:
    """Create composition optimizer."""
    config = OptimizationConfig(output_path=output_path)
    return CompositionOptimizer(analyzer, config)

if __name__ == "__main__":
    # Example usage
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
    optimizer = create_composition_optimizer(
        analyzer,
        output_path=Path("composition_optimization")
    )
    
    # Create example composition
    events_a = ["animation:start", "animation:start", "progress:update"]
    events_b = ["animation:pause", "animation:resume", "animation:pause"]
    
    sequence_a = pattern.sequence(events_a)
    sequence_b = pattern.sequence(events_b)
    
    composition = composer.compose(
        "chain",
        [sequence_a, sequence_b],
        delay=1.0,
        gap=0.5
    )
    
    # Get optimization suggestions
    suggestions = optimizer.analyze_and_suggest(composition)
    
    # Print suggestions
    for suggestion in suggestions:
        print(f"\nType: {suggestion.type}")
        print(f"Description: {suggestion.description}")
        print(f"Impact: {suggestion.impact:.2f}")
        print(f"Complexity: {suggestion.complexity}")
        print(f"Priority: {suggestion.priority}")
        print(f"Rationale: {suggestion.rationale}")
