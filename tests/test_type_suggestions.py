"""Tests for type suggestion functionality."""

import ast
import textwrap
from pathlib import Path
from typing import List, Set, Dict, Any, Optional

import pytest
from scripts.suggest_type_fixes import TypeSuggester, parse_source_file

def strip_whitespace(text: str) -> str:
    """Strip whitespace from text while preserving newlines."""
    return '\n'.join(line.strip() for line in text.splitlines())

def test_basic_function_suggestions() -> None:
    """Test basic function type suggestions."""
    source = textwrap.dedent("""
        def add_numbers(a, b):
            return a + b
            
        def greet(name="Anonymous"):
            return f"Hello, {name}!"
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    assert len(suggester.suggestions) == 2
    
    # Check first function
    assert "def add_numbers(a: Any, b: Any) -> Any:" in suggester.suggestions[0][2]
    
    # Check second function with default value
    assert "def greet(name: str = \"Anonymous\") -> str:" in suggester.suggestions[1][2]

def test_async_function_suggestions() -> None:
    """Test async function type suggestions."""
    source = textwrap.dedent("""
        async def fetch_data(url, timeout=None):
            return await some_request(url)
            
        async def process_items(items=[]):
            results = []
            for item in items:
                results.append(item)
            return results
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    assert len(suggester.suggestions) == 2
    
    # Check first function
    first_suggestion = suggester.suggestions[0][2]
    assert "async def fetch_data" in first_suggestion
    assert "Optional[Any]" in first_suggestion
    
    # Check second function
    second_suggestion = suggester.suggestions[1][2]
    assert "async def process_items" in second_suggestion
    assert "List[" in second_suggestion

def test_variable_type_suggestions() -> None:
    """Test variable type suggestions."""
    source = textwrap.dedent("""
        count = 42
        name = "test"
        items = []
        data = {"key": "value"}
        value = None
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    suggestions = {s[2] for s in suggester.suggestions}
    assert "count: int = 42" in suggestions
    assert "name: str = \"test\"" in suggestions
    assert "items: List[Any] = []" in suggestions
    assert "data: Dict[str, str] = {\"key\": \"value\"}" in suggestions
    assert "value: Optional[Any] = None" in suggestions

def test_complex_type_inference() -> None:
    """Test inference of more complex types."""
    source = textwrap.dedent("""
        def process_config(config=None):
            if config is None:
                config = {}
            return {
                "items": [1, 2, 3],
                "enabled": True,
                "name": "test"
            }
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    suggestion = suggester.suggestions[0][2]
    assert "Dict[str, Union[List[int], bool, str]]" in suggestion

def test_import_suggestions() -> None:
    """Test suggestion of required typing imports."""
    source = textwrap.dedent("""
        def process_data(items=None):
            if items is None:
                items = []
            data = {"count": len(items)}
            return data
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    required_imports = suggester.type_imports_needed
    assert "Optional" in required_imports
    assert "List" in required_imports
    assert "Dict" in required_imports

def test_existing_annotations() -> None:
    """Test handling of existing type annotations."""
    source = textwrap.dedent("""
        def process_data(items: List[str]) -> Dict[str, int]:
            return {"count": len(items)}
            
        name: str = "test"
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    # Should not suggest changes for already annotated code
    assert len(suggester.suggestions) == 0

@pytest.mark.parametrize("source,expected_type", [
    ("x = 42", "int"),
    ("x = 3.14", "float"),
    ("x = 'text'", "str"),
    ("x = [1, 2, 3]", "List[int]"),
    ("x = {'a': 1}", "Dict[str, int]"),
    ("x = None", "Optional[Any]"),
])
def test_value_type_inference(source: str, expected_type: str) -> None:
    """Test type inference for different value types."""
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    assert len(suggester.suggestions) == 1
    assert expected_type in suggester.suggestions[0][2]

def test_invalid_source() -> None:
    """Test handling of invalid source code."""
    source = "def invalid_syntax("  # Missing parenthesis
    
    with pytest.raises(SyntaxError):
        parse_source_file(Path("test.py").write_text(source))

def test_empty_file() -> None:
    """Test handling of empty files."""
    source = "\n\n"  # Empty file with newlines
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    assert len(suggester.suggestions) == 0

def test_class_method_suggestions() -> None:
    """Test type suggestions for class methods."""
    source = textwrap.dedent("""
        class TestClass:
            def __init__(self, name):
                self.name = name
                
            def process(self, data=None):
                return [self.name] if data is None else data
    """)
    
    tree = ast.parse(source)
    suggester = TypeSuggester(source.splitlines(), "test.py")
    suggester.visit(tree)
    
    assert len(suggester.suggestions) == 2
    
    # Check __init__
    init_suggestion = suggester.suggestions[0][2]
    assert "def __init__(self, name: Any) -> None:" in init_suggestion
    
    # Check process method
    process_suggestion = suggester.suggestions[1][2]
    assert "def process(self, data: Optional[Any] = None) -> List[Any]:" in process_suggestion

if __name__ == '__main__':
    pytest.main([__file__])
