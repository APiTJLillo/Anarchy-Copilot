"""Optimization techniques for validation filters."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass
import pandas as pd
import logging
from pathlib import Path
import json
from datetime import datetime
import operator
from functools import reduce
import networkx as nx
from collections import defaultdict

from .validation_filters import FilterCondition, ValidationFilter, FilterManager

logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """Configuration for filter optimization."""
    min_batch_size: int = 1000
    max_cache_size: int = 10000
    enable_indexing: bool = True
    parallel_threshold: int = 5000
    max_workers: int = -1
    reorder_conditions: bool = True
    profile_filters: bool = True
    output_path: Optional[Path] = None

class FilterOptimizer:
    """Optimize validation filter performance."""
    
    def __init__(
        self,
        filter_manager: FilterManager,
        config: OptimizationConfig
    ):
        self.filter_manager = filter_manager
        self.config = config
        self.indexes: Dict[str, Dict[Any, List[int]]] = {}
        self.condition_stats: Dict[str, Dict[str, float]] = {}
        self.filter_graphs: Dict[str, nx.DiGraph] = {}
        
        self._initialize_optimization()
    
    def optimize_filter(
        self,
        filter_name: str
    ) -> ValidationFilter:
        """Optimize existing filter."""
        if filter_name not in self.filter_manager.filters:
            raise KeyError(f"Filter not found: {filter_name}")
        
        validation_filter = self.filter_manager.filters[filter_name]
        
        # Profile conditions if enabled
        if self.config.profile_filters:
            self._profile_conditions(validation_filter)
        
        # Reorder conditions if enabled
        if self.config.reorder_conditions:
            validation_filter = self._reorder_conditions(validation_filter)
        
        # Create indexes if enabled
        if self.config.enable_indexing:
            self._create_indexes(validation_filter)
        
        # Create execution plan
        self._create_execution_plan(validation_filter)
        
        return validation_filter
    
    def apply_optimized_filter(
        self,
        filter_name: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply optimized filter to results."""
        validation_filter = self.optimize_filter(filter_name)
        
        # Use parallel processing if dataset is large
        if (
            len(results) > self.config.parallel_threshold and
            self.config.max_workers != 1
        ):
            return self._parallel_filter(validation_filter, results)
        
        # Use indexes if available
        if self.config.enable_indexing and filter_name in self.indexes:
            return self._indexed_filter(validation_filter, results)
        
        # Use batched processing for large datasets
        if len(results) > self.config.min_batch_size:
            return self._batch_filter(validation_filter, results)
        
        # Fall back to standard filtering
        return [
            result for result in results
            if validation_filter.apply(result)
        ]
    
    def _initialize_optimization(self):
        """Initialize optimization structures."""
        # Clear existing indexes
        self.indexes.clear()
        
        # Clear condition statistics
        self.condition_stats.clear()
        
        # Clear filter graphs
        self.filter_graphs.clear()
    
    def _profile_conditions(
        self,
        validation_filter: ValidationFilter
    ):
        """Profile filter conditions."""
        stats = defaultdict(lambda: {
            "selectivity": 0.0,
            "cost": 0.0,
            "cache_hits": 0,
            "total_calls": 0
        })
        
        # Sample results for profiling
        results = list(
            self.filter_manager.validator.validation_results.values()
        )[:1000]
        
        for condition in validation_filter.conditions:
            condition_key = f"{condition.field}_{condition.operator}"
            
            # Measure selectivity
            matches = sum(
                1 for result in results
                if condition.evaluate(result)
            )
            stats[condition_key]["selectivity"] = matches / len(results)
            
            # Estimate cost
            stats[condition_key]["cost"] = self._estimate_condition_cost(
                condition
            )
        
        self.condition_stats[id(validation_filter)] = dict(stats)
    
    def _estimate_condition_cost(
        self,
        condition: FilterCondition
    ) -> float:
        """Estimate computational cost of condition."""
        # Base cost for field access
        cost = 1.0
        
        # Additional cost for complex field paths
        cost += len(condition.field.split(".")) * 0.1
        
        # Operator costs
        operator_costs = {
            "eq": 1.0,
            "ne": 1.0,
            "gt": 1.2,
            "lt": 1.2,
            "ge": 1.2,
            "le": 1.2,
            "in": 2.0,
            "not_in": 2.0,
            "contains": 3.0,
            "starts_with": 2.0,
            "ends_with": 2.0,
            "matches": 5.0,
            "exists": 1.0,
            "between": 1.5
        }
        
        cost *= operator_costs.get(condition.operator, 1.0)
        
        # Additional cost for negation
        if condition.negate:
            cost *= 1.1
        
        return cost
    
    def _reorder_conditions(
        self,
        validation_filter: ValidationFilter
    ) -> ValidationFilter:
        """Reorder conditions for optimal execution."""
        if not validation_filter.conditions:
            return validation_filter
        
        # Get condition statistics
        stats = self.condition_stats.get(
            id(validation_filter),
            {}
        )
        
        if not stats:
            return validation_filter
        
        # Score conditions based on selectivity and cost
        scored_conditions = []
        for condition in validation_filter.conditions:
            condition_key = f"{condition.field}_{condition.operator}"
            if condition_key in stats:
                score = (
                    stats[condition_key]["selectivity"] /
                    stats[condition_key]["cost"]
                )
                scored_conditions.append((score, condition))
        
        # Sort by score (higher score = higher priority)
        scored_conditions.sort(key=lambda x: x[0], reverse=True)
        
        # Create new filter with reordered conditions
        return ValidationFilter(
            [cond for _, cond in scored_conditions],
            validation_filter.combine
        )
    
    def _create_indexes(
        self,
        validation_filter: ValidationFilter
    ):
        """Create indexes for filter conditions."""
        results = list(self.filter_manager.validator.validation_results.values())
        
        for condition in validation_filter.conditions:
            # Only index equality conditions on simple fields
            if (
                condition.operator == "eq" and
                "." not in condition.field and
                not condition.negate
            ):
                index = defaultdict(list)
                for i, result in enumerate(results):
                    value = result.get(condition.field)
                    if value is not None:
                        index[value].append(i)
                
                self.indexes[id(validation_filter), condition.field] = dict(index)
    
    def _create_execution_plan(
        self,
        validation_filter: ValidationFilter
    ):
        """Create optimized execution plan."""
        graph = nx.DiGraph()
        
        # Add nodes for each condition
        for i, condition in enumerate(validation_filter.conditions):
            graph.add_node(
                i,
                condition=condition,
                stats=self.condition_stats.get(
                    f"{condition.field}_{condition.operator}",
                    {}
                )
            )
        
        # Add edges based on dependencies
        for i in range(len(validation_filter.conditions) - 1):
            graph.add_edge(i, i + 1)
        
        self.filter_graphs[id(validation_filter)] = graph
    
    def _parallel_filter(
        self,
        validation_filter: ValidationFilter,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filter using parallel processing."""
        import concurrent.futures
        
        def process_batch(batch):
            return [
                result for result in batch
                if validation_filter.apply(result)
            ]
        
        # Split results into batches
        batch_size = max(
            self.config.min_batch_size,
            len(results) // (self.config.max_workers * 2)
        )
        batches = [
            results[i:i + batch_size]
            for i in range(0, len(results), batch_size)
        ]
        
        # Process batches in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            filtered = list(executor.map(process_batch, batches))
        
        # Combine results
        return [
            result for batch in filtered
            for result in batch
        ]
    
    def _indexed_filter(
        self,
        validation_filter: ValidationFilter,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filter using indexes."""
        # Find indexed conditions
        index_key = (id(validation_filter), validation_filter.conditions[0].field)
        if index_key not in self.indexes:
            return self._batch_filter(validation_filter, results)
        
        # Get matching indices from index
        index = self.indexes[index_key]
        value = validation_filter.conditions[0].value
        matching_indices = set(index.get(value, []))
        
        # Apply remaining conditions
        filtered = [
            results[i] for i in matching_indices
            if all(
                condition.evaluate(results[i])
                for condition in validation_filter.conditions[1:]
            )
        ]
        
        return filtered
    
    def _batch_filter(
        self,
        validation_filter: ValidationFilter,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply filter using batched processing."""
        filtered = []
        
        for i in range(0, len(results), self.config.min_batch_size):
            batch = results[i:i + self.config.min_batch_size]
            filtered.extend([
                result for result in batch
                if validation_filter.apply(result)
            ])
        
        return filtered
    
    def analyze_performance(
        self,
        filter_name: str
    ) -> Dict[str, Any]:
        """Analyze filter performance."""
        if filter_name not in self.filter_manager.filters:
            raise KeyError(f"Filter not found: {filter_name}")
        
        validation_filter = self.filter_manager.filters[filter_name]
        filter_id = id(validation_filter)
        
        # Get condition statistics
        condition_stats = self.condition_stats.get(filter_id, {})
        
        # Get execution graph
        graph = self.filter_graphs.get(filter_id)
        
        analysis = {
            "condition_stats": condition_stats,
            "execution_plan": self._analyze_execution_plan(graph),
            "optimization_status": {
                "indexed": bool(
                    any(
                        (filter_id, cond.field) in self.indexes
                        for cond in validation_filter.conditions
                    )
                ),
                "reordered": bool(condition_stats),
                "parallelizable": len(
                    self.filter_manager.validator.validation_results
                ) > self.config.parallel_threshold
            }
        }
        
        if self.config.output_path:
            self._save_analysis(filter_name, analysis)
        
        return analysis
    
    def _analyze_execution_plan(
        self,
        graph: Optional[nx.DiGraph]
    ) -> Dict[str, Any]:
        """Analyze filter execution plan."""
        if not graph:
            return {}
        
        return {
            "steps": len(graph.nodes),
            "critical_path": nx.dag_longest_path(graph),
            "bottlenecks": [
                node for node in graph.nodes
                if graph.nodes[node]["stats"].get("cost", 0) > 2.0
            ],
            "parallelizable_nodes": [
                node for node in graph.nodes
                if len(list(graph.predecessors(node))) == 0
            ]
        }
    
    def _save_analysis(
        self,
        filter_name: str,
        analysis: Dict[str, Any]
    ):
        """Save performance analysis."""
        if not self.config.output_path:
            return
        
        try:
            output_file = self.config.output_path / f"{filter_name}_analysis.json"
            with open(output_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Saved analysis to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")

def create_filter_optimizer(
    filter_manager: FilterManager,
    output_path: Optional[Path] = None
) -> FilterOptimizer:
    """Create filter optimizer."""
    config = OptimizationConfig(output_path=output_path)
    return FilterOptimizer(filter_manager, config)

if __name__ == "__main__":
    # Example usage
    from .validation_filters import create_filter_manager
    from .preset_validation import create_preset_validator
    from .visualization_presets import create_preset_manager
    
    # Create components
    preset_manager = create_preset_manager()
    validator = create_preset_validator(preset_manager)
    filter_manager = create_filter_manager(validator)
    
    # Create optimizer
    optimizer = create_filter_optimizer(
        filter_manager,
        output_path=Path("filter_analysis")
    )
    
    # Create and optimize filter
    filter_manager.create_error_filter(
        "critical_errors",
        ["schema", "value_range"]
    )
    optimized = optimizer.optimize_filter("critical_errors")
    
    # Analyze performance
    analysis = optimizer.analyze_performance("critical_errors")
    print(json.dumps(analysis, indent=2))
