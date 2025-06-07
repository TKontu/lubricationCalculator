"""
CLI Module

This module contains command-line interface functions, demo runners,
and example network creation functions.
"""

from .examples import create_simple_tree_example
from .demos import (
    demonstrate_hydraulic_approaches_comparison,
    demonstrate_proper_hydraulic_analysis
)
from .main import main

__all__ = [
    'create_simple_tree_example',
    'demonstrate_hydraulic_approaches_comparison',
    'demonstrate_proper_hydraulic_analysis',
    'main'
]