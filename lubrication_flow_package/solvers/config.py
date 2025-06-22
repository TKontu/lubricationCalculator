"""
Solver configuration and settings
"""

from dataclasses import dataclass


@dataclass
class SolverConfig:
    # Solver control
    max_iterations: int = 100
    tolerance: float = 1e-6         # Relative ΔP convergence threshold

    # Numerical derivative parameters
    dq_absolute: float = 1e-6       # Absolute step size for resistance calculation (m³/s)

    # Damping schedule for conductance rebalance (correct solver)
    damping_initial: float = 0.3    # aggressive early
    damping_mid: float = 0.5        # moderate
    damping_final: float = 0.7      # conservative late

    # Floors and cutoffs
    min_resistance: float = 1e-12   # avoid infinite conductance
    min_flow_fraction: float = 0.1  # 10% of pump flow to stop clipping

    # Safety limits for warnings
    max_reasonable_dp: float = 5e6                # 5 MPa
    min_reasonable_pressure: float = -1e6         # –1 MPa