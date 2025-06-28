"""
Solver configuration and settings
"""

from dataclasses import dataclass


@dataclass
class SolverConfig:
    """Configuration for the nodal matrix solver."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    min_resistance: float = 1e-12
    dq_absolute: float = 1e-8
    relaxation_factor: float = 0.5