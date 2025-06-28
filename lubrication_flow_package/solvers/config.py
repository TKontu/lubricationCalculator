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

    @classmethod
    def from_yaml(cls, file_path: str) -> "SolverConfig":
        """Loads solver configuration from a YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        solver_config_data = config_data.get('solver_config', {})
        return cls(**solver_config_data)