"""
Solver configuration and settings
"""

from dataclasses import dataclass


@dataclass
class SolverConfig:
    # Solver control
    max_iterations: int = 100
    tolerance: float = 1e-6         # Relative ΔP convergence threshold

    # Damping schedule for conductance rebalance (correct solver)
    damping_initial: float = 0.3    # aggressive early
    damping_mid: float = 0.5        # moderate
    damping_final: float = 0.7      # conservative late

    # Floors and cutoffs
    min_resistance: float = 1e-12   # avoid infinite conductance
    min_flow_fraction: float = 0.1  # 10% of pump flow to stop clipping

    # Legacy solver-specific tolerances
    legacy_min_tolerance_simple: float = 5e3      # Pa for ≤2 outlets 
    legacy_min_tolerance_medium: float = 1e6      # Pa for 3–4 outlets
    legacy_min_tolerance_complex: float = 2e6     # Pa for >4 outlets
    legacy_damping_hardy_cross: float = 0.5       # relaxation factor

    # Safety limits for warnings
    max_reasonable_dp: float = 5e6                # 5 MPa
    min_reasonable_pressure: float = -1e6         # –1 MPa