"""Mutation testing for state migrations."""

import asyncio
import pytest
from typing import Dict, Any, List, Callable, Type
from dataclasses import replace
from copy import deepcopy
import random
import operator
from collections import defaultdict

from proxy.analysis.monitoring.state_migration import (
    StateMigrator,
    MigrationMetadata,
    MigrationResult,
    create_state_migrator
)
from proxy.analysis.monitoring.dashboard_state import DashboardState
from .test_migration_fuzz import dashboard_state

class MutationOperator:
    """Base class for mutation operators."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def mutate(self, obj: Any) -> Any:
        """Apply mutation to object."""
        raise NotImplementedError()

class NullMutation(MutationOperator):
    """Replace value with None."""
    
    def __init__(self):
        super().__init__(
            "null_mutation",
            "Replace value with None"
        )
    
    def mutate(self, obj: Any) -> Any:
        return None

class TypeMutation(MutationOperator):
    """Change value type."""
    
    def __init__(self):
        super().__init__(
            "type_mutation",
            "Change value type"
        )
    
    def mutate(self, obj: Any) -> Any:
        if isinstance(obj, bool):
            return int(obj)
        elif isinstance(obj, (int, float)):
            return str(obj)
        elif isinstance(obj, str):
            return [obj]
        elif isinstance(obj, list):
            return {str(i): v for i, v in enumerate(obj)}
        elif isinstance(obj, dict):
            return list(obj.items())
        return obj

class ValueMutation(MutationOperator):
    """Modify value while preserving type."""
    
    def __init__(self):
        super().__init__(
            "value_mutation",
            "Modify value while preserving type"
        )
    
    def mutate(self, obj: Any) -> Any:
        if isinstance(obj, bool):
            return not obj
        elif isinstance(obj, int):
            return obj + random.randint(-100, 100)
        elif isinstance(obj, float):
            return obj * random.uniform(0.5, 1.5)
        elif isinstance(obj, str):
            return obj + "_mutated"
        elif isinstance(obj, list):
            if obj:
                obj = obj.copy()
                idx = random.randrange(len(obj))
                obj[idx] = self.mutate(obj[idx])
            return obj
        elif isinstance(obj, dict):
            if obj:
                obj = obj.copy()
                key = random.choice(list(obj.keys()))
                obj[key] = self.mutate(obj[key])
            return obj
        return obj

class StructureMutation(MutationOperator):
    """Modify object structure."""
    
    def __init__(self):
        super().__init__(
            "structure_mutation",
            "Modify object structure"
        )
    
    def mutate(self, obj: Any) -> Any:
        if isinstance(obj, list):
            if obj:
                obj = obj.copy()
                if random.random() < 0.5:
                    # Remove random element
                    del obj[random.randrange(len(obj))]
                else:
                    # Add duplicated element
                    obj.append(random.choice(obj))
            return obj
        elif isinstance(obj, dict):
            if obj:
                obj = obj.copy()
                if random.random() < 0.5:
                    # Remove random key
                    key = random.choice(list(obj.keys()))
                    del obj[key]
                else:
                    # Add new key
                    key = f"mutated_{len(obj)}"
                    obj[key] = random.choice(list(obj.values()))
            return obj
        return obj

class MigrationMutator:
    """Apply mutations to test migration robustness."""
    
    def __init__(self):
        self.operators = [
            NullMutation(),
            TypeMutation(),
            ValueMutation(),
            StructureMutation()
        ]
        self.results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def mutate_state(
        self,
        state: DashboardState,
        depth: int = 3
    ) -> List[DashboardState]:
        """Generate mutated versions of state."""
        mutations = []
        
        def apply_mutations(obj: Any, current_depth: int) -> List[Any]:
            if current_depth <= 0:
                return [obj]
            
            results = []
            
            # Apply each operator
            for operator in self.operators:
                try:
                    mutated = operator.mutate(deepcopy(obj))
                    if mutated != obj:
                        results.append(mutated)
                except Exception:
                    continue
            
            # Recursively mutate nested structures
            if isinstance(obj, dict):
                for key, value in obj.items():
                    for mutated_value in apply_mutations(value, current_depth - 1):
                        mutated = deepcopy(obj)
                        mutated[key] = mutated_value
                        results.append(mutated)
                        
            elif isinstance(obj, list):
                for i, value in enumerate(obj):
                    for mutated_value in apply_mutations(value, current_depth - 1):
                        mutated = deepcopy(obj)
                        mutated[i] = mutated_value
                        results.append(mutated)
            
            return results
        
        # Mutate each state field
        for field in ["filters", "layout", "theme", "display", "metadata"]:
            value = getattr(state, field)
            for mutated_value in apply_mutations(value, depth):
                try:
                    mutated_state = replace(state, **{field: mutated_value})
                    mutations.append(mutated_state)
                except Exception:
                    continue
        
        return mutations

class MutationTestResult:
    """Results from mutation testing."""
    
    def __init__(self):
        self.total_mutations = 0
        self.killed_mutations = 0
        self.survived_mutations = 0
        self.errors = []
        self.operator_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
    
    @property
    def mutation_score(self) -> float:
        """Calculate mutation score."""
        if self.total_mutations == 0:
            return 0.0
        return self.killed_mutations / self.total_mutations
    
    def add_result(
        self,
        operator: str,
        killed: bool,
        error: Optional[str] = None
    ):
        """Add mutation test result."""
        self.total_mutations += 1
        
        if killed:
            self.killed_mutations += 1
            self.operator_stats[operator]["killed"] += 1
        else:
            self.survived_mutations += 1
            self.operator_stats[operator]["survived"] += 1
        
        if error:
            self.errors.append(error)
            self.operator_stats[operator]["errors"] += 1
    
    def __str__(self) -> str:
        return (
            f"Mutation Test Results:\n"
            f"  Total Mutations: {self.total_mutations}\n"
            f"  Killed: {self.killed_mutations}\n"
            f"  Survived: {self.survived_mutations}\n"
            f"  Score: {self.mutation_score:.2%}\n"
            f"  Errors: {len(self.errors)}\n"
            "\nOperator Stats:\n" +
            "\n".join(
                f"  {op}: {stats}"
                for op, stats in self.operator_stats.items()
            )
        )

@pytest.fixture
def mutator():
    """Create mutation tester."""
    return MigrationMutator()

@pytest.fixture
async def test_state():
    """Create test state."""
    return await dashboard_state().example()

@pytest.mark.asyncio
async def test_mutation_detection(mutator, test_state, migrator):
    """Test ability to detect mutations through testing."""
    result = MutationTestResult()
    
    # Generate mutations
    mutations = mutator.mutate_state(test_state)
    
    # Test each mutation
    for mutation in mutations:
        for operator in mutator.operators:
            if isinstance(mutation, DashboardState):
                try:
                    # Try migrating mutated state
                    migration_result = await migrator.migrate_state(mutation)
                    
                    # Check if mutation was caught
                    killed = not migration_result.success
                    result.add_result(operator.name, killed)
                    
                except Exception as e:
                    result.add_result(
                        operator.name,
                        True,
                        str(e)
                    )
    
    # Check mutation score
    assert result.mutation_score > 0.7, "Low mutation detection rate"
    print(result)

@pytest.mark.asyncio
async def test_operator_coverage(mutator, test_state):
    """Test coverage of mutation operators."""
    mutations = mutator.mutate_state(test_state)
    
    # Check each operator produced mutations
    operator_mutations = defaultdict(int)
    for mutation in mutations:
        for operator in mutator.operators:
            try:
                mutated = operator.mutate(deepcopy(test_state))
                if mutated != test_state:
                    operator_mutations[operator.name] += 1
            except Exception:
                continue
    
    # Check all operators were used
    assert all(
        count > 0
        for count in operator_mutations.values()
    ), "Some operators produced no mutations"

@pytest.mark.asyncio
async def test_mutation_stability(mutator, test_state, migrator):
    """Test stability of mutations."""
    results = []
    
    # Run multiple mutation cycles
    for _ in range(5):
        mutations = mutator.mutate_state(test_state)
        cycle_result = MutationTestResult()
        
        for mutation in mutations:
            if isinstance(mutation, DashboardState):
                try:
                    result = await migrator.migrate_state(mutation)
                    killed = not result.success
                    cycle_result.add_result("any", killed)
                except Exception as e:
                    cycle_result.add_result("any", True, str(e))
        
        results.append(cycle_result)
    
    # Check consistency
    scores = [r.mutation_score for r in results]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    
    assert variance < 0.1, "High variance in mutation scores"

if __name__ == "__main__":
    pytest.main([__file__])
