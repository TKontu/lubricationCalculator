"""
Utilities Module

This module contains shared utility functions used across different solvers
to eliminate code duplication and improve maintainability.
"""

from .network_utils import (
    find_all_paths,
    compute_path_pressure,
    estimate_resistance,
    compute_node_pressures,
    validate_flow_conservation,
    calculate_path_conductances,
    distribute_flow_by_conductance,
    check_convergence
)

__all__ = [
    'find_all_paths',
    'compute_path_pressure', 
    'estimate_resistance',
    'compute_node_pressures',
    'validate_flow_conservation',
    'calculate_path_conductances',
    'distribute_flow_by_conductance',
    'check_convergence'
]