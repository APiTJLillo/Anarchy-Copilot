#!/usr/bin/env python3
"""Analyze code and suggest type annotation fixes."""

import ast
import sys
import argparse
import difflib
import inspect
import importlib
import tokenize
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union

def parse_source_file(path: Path) -> Tuple[ast.Module, List[str]]:
    """Parse Python source file and return AST and lines."""
    with tokenize.open(path) as f:
        source = f.read()
        lines = source.splitlines()
        return ast.parse(source), lines

class TypeSuggester(ast.NodeVisitor):
    def __init__(self, source_lines: List[str], module_path: str):
        self.lines = source_lines
        self.suggestions: List[Tuple[int, str, str]] = []  # line, original, suggestion
        self.imported_names: Set[str] = set()
        self.module_path = module_path
        self.type_imports_needed: Set[str] = set()
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track imported names."""
        if node.module == 'typing':
            for name in node.names:
                self.imported_names.add(name.name)
        self.generic_visit(node)

    def _get_value_type(self, node: ast.AST) -> Optional[str]:
        """Infer type from value."""
        if isinstance(node, ast.Num):
            return 'int' if isinstance(node.n, int) else 'float'
        elif isinstance(node, ast.Str):
            return 'str'
        elif isinstance(node, ast.List):
            if node.elts:
                elem_type = self._get_value_type(node.elts[0])
                if elem_type:
                    if 'List' not in self.imported_names:
                        self.type_imports_needed.add('List')
                    return f'List[{elem_type}]'
            return 'list'
        elif isinstance(node, ast.Dict):
            if node.keys:
                key_type = self._get_value_type(node.keys[0])
                val_type = self._get_value_type(node.values[0])
                if key_type and val_type:
                    if 'Dict' not in self.imported_names:
                        self.type_imports_needed.add('Dict')
                    return f'Dict[{key_type}, {val_type}]'
            return 'dict'
        elif isinstance(node, ast.Name):
            if node.id == 'None':
                return 'None'
        return None

    def _suggest_function_types(self, node: ast.FunctionDef) -> None:
        """Suggest types for function arguments and return."""
        if node.returns:  # Already has return annotation
            return

        # Try to infer return type
        returns: Set[str] = set()
        class ReturnFinder(ast.NodeVisitor):
            def visit_Return(self, return_node: ast.Return) -> None:
                if return_node.value:
                    value_type = self._get_value_type(return_node.value)
                    if value_type:
                        returns.add(value_type)
                else:
                    returns.add('None')

        ReturnFinder().visit(node)
        
        # Build return type suggestion
        return_type = 'Any'
        if returns:
            if len(returns) == 1:
                return_type = next(iter(returns))
            else:
                if 'Union' not in self.imported_names:
                    self.type_imports_needed.add('Union')
                return_type = f"Union[{', '.join(sorted(returns))}]"

        # Get original line
        line_no = node.lineno - 1
        original = self.lines[line_no]
        
        # Create suggestion
        indent = len(original) - len(original.lstrip())
        suggestion = original[:original.find('def') + 3]
        suggestion += f" {node.name}("
        
        # Add argument types
        args = []
        for arg in node.args.args:
            if not arg.annotation:
                arg_type = 'Any'
                # Try to infer from default value
                if node.args.defaults:
                    idx = node.args.args.index(arg) - (len(node.args.args) - len(node.args.defaults))
                    if idx >= 0:
                        default = node.args.defaults[idx]
                        inferred = self._get_value_type(default)
                        if inferred:
                            if inferred == 'None':
                                if 'Optional' not in self.imported_names:
                                    self.type_imports_needed.add('Optional')
                                arg_type = f'Optional[Any]'
                            else:
                                arg_type = inferred
                args.append(f"{arg.arg}: {arg_type}")
            else:
                args.append(self.lines[arg.lineno-1].strip())
        
        suggestion += ", ".join(args)
        suggestion += f") -> {return_type}:"
        
        self.suggestions.append((line_no, original, suggestion))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definitions."""
        self._suggest_function_types(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Process async function definitions."""
        self._suggest_function_types(node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Process variable annotations."""
        if node.annotation:  # Already has type annotation
            return
        
        if isinstance(node.value, ast.Name) and node.value.id == 'None':
            if 'Optional' not in self.imported_names:
                self.type_imports_needed.add('Optional')
            suggested_type = 'Optional[Any]'
        else:
            suggested_type = self._get_value_type(node.value) or 'Any'
        
        line_no = node.lineno - 1
        original = self.lines[line_no]
        target = self.lines[line_no][node.target.col_offset:node.target.end_col_offset]
        suggestion = f"{target}: {suggested_type} = {original[node.value.col_offset:]}"
        
        self.suggestions.append((line_no, original, suggestion))

def generate_diff(file_path: Path, suggestions: List[Tuple[int, str, str]]) -> str:
    """Generate unified diff for suggested changes."""
    original = file_path.read_text().splitlines()
    modified = original.copy()
    
    # Apply suggestions
    offset = 0
    for line_no, _, suggestion in sorted(suggestions):
        modified[line_no + offset] = suggestion
    
    # Generate diff
    diff = difflib.unified_diff(
        original,
        modified,
        fromfile=str(file_path),
        tofile=str(file_path) + '.suggested',
        lineterm=''
    )
    
    return '\n'.join(diff)

def suggest_fixes(path: Path) -> Tuple[str, Set[str]]:
    """Analyze file and suggest type annotation fixes."""
    try:
        tree, lines = parse_source_file(path)
        suggester = TypeSuggester(lines, str(path))
        suggester.visit(tree)
        
        if suggester.type_imports_needed:
            # Add required imports
            import_line = f"from typing import {', '.join(sorted(suggester.type_imports_needed))}"
            suggester.suggestions.insert(0, (0, '', import_line))
        
        return generate_diff(path, suggester.suggestions), suggester.type_imports_needed
        
    except Exception as e:
        print(f"Error processing {path}: {e}", file=sys.stderr)
        return "", set()

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Suggest type annotation fixes for Python files"
    )
    parser.add_argument("paths", nargs="+", help="Files or directories to analyze")
    parser.add_argument("--apply", action="store_true",
                      help="Apply suggested fixes (use with caution)")
    args = parser.parse_args()

    for path_str in args.paths:
        path = Path(path_str)
        if path.is_dir():
            files = path.rglob("*.py")
        else:
            files = [path]

        for file in files:
            if file.name.startswith("__"):
                continue

            print(f"\nAnalyzing {file}...")
            diff, imports = suggest_fixes(file)
            
            if diff:
                if args.apply:
                    # Apply changes
                    file.write_text(
                        file.read_text() + '\n' +
                        '\n'.join(f"from typing import {imp}" for imp in imports) +
                        '\n' + diff
                    )
                    print("Applied suggested fixes")
                else:
                    print("\nSuggested fixes:")
                    print(diff)
                    if imports:
                        print("\nRequired imports:")
                        for imp in imports:
                            print(f"from typing import {imp}")
            else:
                print("No suggestions.")

    return 0

if __name__ == "__main__":
    sys.exit(main())
