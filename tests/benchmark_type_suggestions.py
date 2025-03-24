"""Performance benchmarks for type suggestion functionality."""

import ast
import cProfile
import json
import pstats
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Tuple
import textwrap
import pytest
import numpy as np

from scripts.suggest_type_fixes import TypeSuggester, suggest_fixes, parse_source_file

BENCHMARK_SIZES = {
    'tiny': 10,
    'small': 100,
    'medium': 500,
    'large': 1000,
    'xlarge': 5000
}

def generate_test_file(size: int) -> str:
    """Generate a Python file of given size (in lines)."""
    templates = [
        # Function template
        textwrap.dedent("""
            def function_{i}(param_{i}=None):
                result = []
                for j in range(10):
                    if param_{i}:
                        result.append(j)
                return result
        """),
        
        # Class template
        textwrap.dedent("""
            class Class_{i}:
                def __init__(self, value=None):
                    self.value = value
                
                def process(self, data):
                    return [self.value] if data is None else data
                
                async def async_method(self):
                    return self.value
        """),
        
        # Variable assignments template
        textwrap.dedent("""
            var_{i}_int = 42
            var_{i}_str = "test"
            var_{i}_list = [1, 2, 3]
            var_{i}_dict = {{"key": "value"}}
            var_{i}_none = None
        """)
    ]
    
    # Generate code by cycling through templates
    code_parts = []
    for i in range(size):
        template = templates[i % len(templates)]
        code_parts.append(template.format(i=i))
    
    return "\n".join(code_parts)

def run_benchmark(code: str) -> Tuple[float, Dict[str, Any]]:
    """Run benchmark on given code and return timing and stats."""
    # Write code to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as tmp:
        tmp.write(code)
        tmp.flush()
        
        # Time the suggestion process
        start_time = time.perf_counter()
        diff, imports = suggest_fixes(Path(tmp.name))
        end_time = time.perf_counter()
        
        # Parse code for stats
        tree = ast.parse(code)
        suggester = TypeSuggester(code.splitlines(), tmp.name)
        suggester.visit(tree)
        
        stats = {
            'num_suggestions': len(suggester.suggestions),
            'imports_needed': len(imports),
            'code_size': len(code),
            'num_lines': len(code.splitlines()),
            'suggestions_per_second': len(suggester.suggestions) / (end_time - start_time)
        }
        
        return end_time - start_time, stats

def profile_suggestions(code: str) -> pstats.Stats:
    """Profile type suggestion performance."""
    profiler = cProfile.Profile()
    profiler.enable()
    
    diff, imports = suggest_fixes(Path("test.py").write_text(code))
    
    profiler.disable()
    return pstats.Stats(profiler)

@pytest.mark.benchmark
class TestTypesuggestionPerformance:
    """Performance tests for type suggestions."""
    
    @pytest.mark.parametrize("size_name,size", BENCHMARK_SIZES.items())
    def test_suggestion_scaling(self, size_name: str, size: int, benchmark) -> None:
        """Test how suggestion performance scales with file size."""
        code = generate_test_file(size)
        
        def run_test():
            return run_benchmark(code)
        
        result = benchmark(run_test)
        duration, stats = result
        
        # Save benchmark results
        self._save_benchmark_result(size_name, {
            'duration': duration,
            'stats': stats,
            'rounds': result.stats.rounds,
            'mean': result.stats.mean,
            'stddev': result.stats.stddev
        })
    
    def test_memory_usage(self, benchmark) -> None:
        """Test memory usage during type suggestions."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        code = generate_test_file(BENCHMARK_SIZES['large'])
        
        def run_test():
            initial_memory = process.memory_info().rss
            _, stats = run_benchmark(code)
            final_memory = process.memory_info().rss
            return (final_memory - initial_memory) / 1024 / 1024  # MB
        
        result = benchmark(run_test)
        self._save_benchmark_result('memory_usage', {
            'mean_memory_mb': result.stats.mean,
            'max_memory_mb': result.stats.max,
            'stddev_memory_mb': result.stats.stddev
        })
    
    @pytest.mark.parametrize("feature", [
        "functions",
        "classes",
        "variables",
        "async_functions"
    ])
    def test_feature_performance(self, feature: str, benchmark) -> None:
        """Test performance for different code features."""
        # Generate feature-specific code
        if feature == "functions":
            code = "\n".join([
                f"def func_{i}(a, b=None): return a + b if b else a"
                for i in range(100)
            ])
        elif feature == "classes":
            code = "\n".join([
                f"class Class_{i}:\n    def method(self, x): return x"
                for i in range(100)
            ])
        elif feature == "variables":
            code = "\n".join([
                f"var_{i} = {i} if i % 2 else 'test'"
                for i in range(100)
            ])
        else:  # async_functions
            code = "\n".join([
                f"async def async_func_{i}(x): return x"
                for i in range(100)
            ])
        
        result = benchmark(lambda: run_benchmark(code))
        duration, stats = result
        
        self._save_benchmark_result(f'feature_{feature}', {
            'duration': duration,
            'stats': stats,
            'rounds': result.stats.rounds,
            'mean': result.stats.mean,
            'stddev': result.stats.stddev
        })
    
    def test_import_suggestion_performance(self, benchmark) -> None:
        """Test performance of import suggestions."""
        code = textwrap.dedent("""
            def process_data(items=None):
                if items is None:
                    items = []
                data = {"count": len(items)}
                return [x for x in items if isinstance(x, (int, str))]
        """) * 100
        
        def run_test():
            tree = ast.parse(code)
            suggester = TypeSuggester(code.splitlines(), "test.py")
            suggester.visit(tree)
            return len(suggester.type_imports_needed)
        
        result = benchmark(run_test)
        self._save_benchmark_result('import_suggestions', {
            'mean': result.stats.mean,
            'stddev': result.stats.stddev,
            'rounds': result.stats.rounds
        })
    
    def test_suggestion_quality(self) -> None:
        """Test quality metrics of type suggestions."""
        code = generate_test_file(100)
        tree = ast.parse(code)
        suggester = TypeSuggester(code.splitlines(), "test.py")
        suggester.visit(tree)
        
        # Calculate quality metrics
        metrics = {
            'total_suggestions': len(suggester.suggestions),
            'any_suggestions': sum(1 for s in suggester.suggestions if "Any" in s[2]),
            'optional_suggestions': sum(1 for s in suggester.suggestions if "Optional" in s[2]),
            'specific_types': sum(1 for s in suggester.suggestions 
                                if any(t in s[2] for t in ['int', 'str', 'List', 'Dict'])),
            'imports_needed': len(suggester.type_imports_needed)
        }
        
        metrics['specificity_ratio'] = (metrics['specific_types'] / 
                                      metrics['total_suggestions'])
        
        self._save_benchmark_result('suggestion_quality', metrics)
    
    def _save_benchmark_result(self, name: str, data: Dict[str, Any]) -> None:
        """Save benchmark results to JSON file."""
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        result_file = results_dir / "type_suggestion_benchmarks.json"
        
        try:
            if result_file.exists():
                results = json.loads(result_file.read_text())
            else:
                results = {}
            
            results[name] = {
                'timestamp': time.time(),
                'data': data
            }
            
            result_file.write_text(json.dumps(results, indent=2))
            
        except Exception as e:
            print(f"Error saving benchmark results: {e}")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
