"""
CLI subpackage - Command line interface and demo runners
"""

from .main import main
from .demos import create_simple_tree_example, demonstrate_hydraulic_approaches_comparison, demonstrate_proper_hydraulic_analysis

__all__ = ['main', 'create_simple_tree_example', 'demonstrate_hydraulic_approaches_comparison', 'demonstrate_proper_hydraulic_analysis']