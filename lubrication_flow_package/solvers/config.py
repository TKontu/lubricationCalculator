"""
Solver configuration and settings
"""

from dataclasses import dataclass
import yaml


@dataclass
class SolverConfig:
    """Configuration for the nodal matrix solver."""
    max_iterations: int = 100
    tolerance: float = 1e-6
    min_resistance: float = 1e-12
    dq_absolute: float = 1e-8
    relaxation_factor: float = 0.5
    
    # For RobustNonLinearSolver
    convergence: dict = None
    line_search: dict = None
    jacobian: dict = None

    def __post_init__(self):
        if self.convergence is None:
            self.convergence = {
                "residual_tolerance": 1.0e-8,
                "relative_tolerance": 1.0e-6,
                "component_tolerance": 1.0e-4,
            }
        if self.line_search is None:
            self.line_search = {
                "method": "armijo",
                "c1": 1.0e-4,
                "alpha_min": 1.0e-10,
                "max_backtracks": 20,
            }
        if self.jacobian is None:
            self.jacobian = {
                "update_method": "analytical",
                "finite_difference_step": 1.0e-8,
                "sparsity_detection": True,
            }

    @classmethod
    def from_yaml(cls, file_path: str) -> "SolverConfig":
        """Loads solver configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        solver_config_data = config_data.get('solver_config', {})
        return cls(**solver_config_data)