#!/usr/bin/env python3
"""
Dependency checker for Nexor library.
Verifies import structure and detects potential circular dependencies.
"""

import os
import sys
import ast
from typing import Set, Dict, List

class DependencyChecker:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.dependencies: Dict[str, Set[str]] = {}
        self.circular_deps: List[List[str]] = []
        
    def analyze_file(self, filepath: str) -> None:
        """Analyze imports in a Python file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=filepath)
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return
            
        module_name = self._get_module_name(filepath)
        self.dependencies[module_name] = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith('texor'):
                        self.dependencies[module_name].add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('texor'):
                    self.dependencies[module_name].add(node.module)
                    
    def _get_module_name(self, filepath: str) -> str:
        """Convert filepath to module name"""
        relpath = os.path.relpath(filepath, self.root_dir)
        module_path = os.path.splitext(relpath)[0]
        return module_path.replace(os.path.sep, '.')
        
    def find_cycles(self) -> None:
        """Find circular dependencies using DFS"""
        def dfs(node: str, visited: Set[str], path: List[str]) -> None:
            if node in path:
                cycle = path[path.index(node):]
                cycle.append(node)
                self.circular_deps.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            path.append(node)
            
            for dep in self.dependencies.get(node, []):
                dfs(dep, visited, path[:])
                
            path.pop()
            
        visited: Set[str] = set()
        for module in self.dependencies:
            if module not in visited:
                dfs(module, visited, [])
                
    def check_imports(self) -> bool:
        """Check if all imported modules exist"""
        all_modules = set(self.dependencies.keys())
        missing_imports = set()
        
        for module, deps in self.dependencies.items():
            for dep in deps:
                if dep not in all_modules:
                    missing_imports.add((module, dep))
                    
        if missing_imports:
            print("\nMissing imports found:")
            for module, missing in missing_imports:
                print(f"  {module} imports {missing} but it doesn't exist")
            return False
        return True
        
    def print_report(self) -> None:
        """Print dependency analysis report"""
        print("\nDependency Analysis Report")
        print("=" * 50)
        
        # Print module dependencies
        print("\nModule Dependencies:")
        for module, deps in sorted(self.dependencies.items()):
            if deps:
                print(f"\n{module} depends on:")
                for dep in sorted(deps):
                    print(f"  - {dep}")
                    
        # Print circular dependencies
        if self.circular_deps:
            print("\nCircular Dependencies Found:")
            for cycle in self.circular_deps:
                print("  " + " -> ".join(cycle))
        else:
            print("\nNo circular dependencies found.")
            
        # Print import check results
        print("\nImport Check:", "PASS" if self.check_imports() else "FAIL")

def main():
    # Get project root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # Initialize checker
    checker = DependencyChecker(root_dir)
    
    # Find and analyze all Python files
    for root, _, files in os.walk(os.path.join(root_dir, 'texor')):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                checker.analyze_file(filepath)
                
    # Find circular dependencies
    checker.find_cycles()
    
    # Print report
    checker.print_report()
    
    # Return exit code
    return 1 if checker.circular_deps or not checker.check_imports() else 0

if __name__ == '__main__':
    sys.exit(main())