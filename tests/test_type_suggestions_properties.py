"""Property-based tests for type suggestion functionality."""

import ast
from typing import List, Dict, Any, Optional
from hypothesis import given, strategies as st
import pytest

from scripts.suggest_type_fixes import TypeSuggester

# Custom strategies for generating Python code
identifiers = st.from_regex(r'[a-zA-Z_][a-zA-Z0-9_]{0,30}')

@st.composite
def python_literals(draw) -> st.SearchStrategy[str]:
    """Generate valid Python literals."""
    return draw(st.one_of(
        st.integers().map(str),
        st.floats(allow_infinity=False, allow_nan=False).map(repr),
        st.text().map(repr),
        st.lists(st.integers()).map(repr),
        st.dictionaries(st.text(), st.integers()).map(repr),
        st.just("None"),
        st.just("True"),
        st.just("False")
    ))

@st.composite
def function_definitions(draw) -> str:
    """Generate valid Python function definitions."""
    name = draw(identifiers)
    num_params = draw(st.integers(min_value=0, max_value=5))
    params = [draw(identifiers) for _ in range(num_params)]
    
    # Maybe add default values
    params_with_defaults = []
    for param in params:
        if draw(st.booleans()):
            default_value = draw(python_literals())
            params_with_defaults.append(f"{param}={default_value}")
        else:
            params_with_defaults.append(param)
    
    return_value = draw(python_literals())
    
    return f"""def {name}({', '.join(params_with_defaults)}):
    return {return_value}
"""

@st.composite
def async_function_definitions(draw) -> str:
    """Generate valid async Python function definitions."""
    func = draw(function_definitions())
    return "async " + func

@st.composite
def variable_assignments(draw) -> str:
    """Generate valid variable assignments."""
    name = draw(identifiers)
    value = draw(python_literals())
    return f"{name} = {value}"

@st.composite
def class_definitions(draw) -> str:
    """Generate valid Python class definitions."""
    class_name = draw(identifiers)
    num_methods = draw(st.integers(min_value=1, max_value=3))
    methods = [draw(function_definitions()) for _ in range(num_methods)]
    
    return f"""class {class_name}:
    {chr(10).join('    ' + m for m in methods)}
"""

def is_valid_python(code: str) -> bool:
    """Check if string is valid Python code."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

@given(st.lists(function_definitions(), min_size=1, max_size=5))
def test_function_suggestions_properties(functions: List[str]) -> None:
    """Test type suggestions for randomly generated functions."""
    source = "\n".join(functions)
    assert is_valid_python(source), "Generated invalid Python code"
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    # Each non-annotated function should have a suggestion
    assert len(suggester.suggestions) > 0
    
    for suggestion in suggester.suggestions:
        suggestion_text = suggestion[2]
        # Verify basic properties of suggestions
        assert "def " in suggestion_text
        assert ":" in suggestion_text
        assert "Any" in suggestion_text or "Optional" in suggestion_text
        # Verify suggestion is valid Python
        assert is_valid_python(suggestion_text)

@given(st.lists(variable_assignments(), min_size=1, max_size=5))
def test_variable_suggestions_properties(assignments: List[str]) -> None:
    """Test type suggestions for randomly generated variable assignments."""
    source = "\n".join(assignments)
    assert is_valid_python(source), "Generated invalid Python code"
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    # Each assignment should have a suggestion
    assert len(suggester.suggestions) == len(assignments)
    
    for suggestion in suggester.suggestions:
        suggestion_text = suggestion[2]
        # Verify basic properties of suggestions
        assert ":" in suggestion_text
        assert "=" in suggestion_text
        # Verify suggestion is valid Python
        assert is_valid_python(suggestion_text)

@given(st.lists(async_function_definitions(), min_size=1, max_size=5))
def test_async_function_suggestions_properties(functions: List[str]) -> None:
    """Test type suggestions for randomly generated async functions."""
    source = "\n".join(functions)
    assert is_valid_python(source), "Generated invalid Python code"
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    for suggestion in suggester.suggestions:
        suggestion_text = suggestion[2]
        # Verify async function properties
        assert "async def" in suggestion_text
        assert ":" in suggestion_text
        assert is_valid_python(suggestion_text)

@given(st.lists(class_definitions(), min_size=1, max_size=3))
def test_class_suggestions_properties(classes: List[str]) -> None:
    """Test type suggestions for randomly generated classes."""
    source = "\n".join(classes)
    assert is_valid_python(source), "Generated invalid Python code"
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    for suggestion in suggester.suggestions:
        suggestion_text = suggestion[2]
        # Verify method properties
        assert "def" in suggestion_text
        assert "self" in suggestion_text
        assert is_valid_python(suggestion_text)

@given(st.lists(python_literals(), min_size=1, max_size=10))
def test_type_inference_properties(values: List[str]) -> None:
    """Test type inference properties for random Python literals."""
    # Create assignments with the literals
    assignments = [f"x{i} = {value}" for i, value in enumerate(values)]
    source = "\n".join(assignments)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    for suggestion in suggester.suggestions:
        suggestion_text = suggestion[2]
        # Verify type annotation properties
        assert ":" in suggestion_text
        assert "=" in suggestion_text
        assert is_valid_python(suggestion_text)
        
        # Check specific type properties
        if "None" in suggestion_text:
            assert "Optional" in suggestion_text
        if "[" in suggestion_text:
            assert "]" in suggestion_text
            assert "List" in suggestion_text or "Dict" in suggestion_text

def test_suggestion_idempotency() -> None:
    """Test that applying suggestions multiple times doesn't change results."""
    @given(st.lists(st.one_of(
        function_definitions(),
        variable_assignments(),
        class_definitions()
    ), min_size=1, max_size=5))
    def check_idempotency(code_items: List[str]) -> None:
        source = "\n".join(code_items)
        assert is_valid_python(source)
        
        # First pass
        tree = ast.parse(source)
        suggester1 = TypeSuggester(source.splitlines(), "test.py")
        suggester1.visit(tree)
        
        # Apply suggestions
        new_source = source
        for _, _, suggestion in suggester1.suggestions:
            new_source = new_source + "\n" + suggestion
            
        # Second pass
        tree2 = ast.parse(new_source)
        suggester2 = TypeSuggester(new_source.splitlines(), "test.py")
        suggester2.visit(tree2)
        
        # Should have no new suggestions
        assert len(suggester2.suggestions) == 0
    
    check_idempotency()

if __name__ == '__main__':
    pytest.main([__file__])
