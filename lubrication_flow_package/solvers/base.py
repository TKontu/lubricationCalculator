"""
Base classes for all hydraulic network solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..utils.viscosity import calculate_viscosity
from .config import SolverConfig


class SolverBase(ABC):
    """
    An abstract base class that defines the common interface for all hydraulic solvers.

    This class ensures that any solver, whether it's a linear nodal solver or a
    non-linear Newton-Raphson solver, can be used interchangeably by the
    calling application (e.g., the CLI or GUI).
    """

    def __init__(self, sim_config: SimulationConfig, solver_config: Optional[SolverConfig] = None):
        """
        Initializes the solver.

        Args:
            sim_config: The simulation configuration object, containing physical
                        parameters of the system (e.g., flow rate, fluid properties).
            solver_config: An optional configuration object for tuning the solver's
                           numerical behavior (e.g., tolerances, max iterations).
                           If None, the solver should use its default configuration.
        """
        self.sim_config = sim_config
        self.config = solver_config if solver_config else self.get_default_solver_config()
        self.fluid_properties = self._get_fluid_properties()

    @abstractmethod
    def solve(self, network: FlowNetwork) -> Dict:
        """
        Main entry point for solving the hydraulic network.

        Args:
            network: The FlowNetwork object to be solved.

        Returns:
            A dictionary containing the complete solution, including flows,
            pressures, and convergence information. The structure of this
            dictionary should be standardized across all solvers.
        """
        raise NotImplementedError("Subclasses must implement the solve method.")

    @abstractmethod
    def print_results(self, network: FlowNetwork, solution: Dict, **kwargs):
        """
        Prints the simulation results in a structured and clear format.

        Args:
            network: The solved FlowNetwork object.
            solution: The solution dictionary returned by the solve() method.
        """
        raise NotImplementedError("Subclasses must implement the print_results method.")

    def get_default_solver_config(self) -> SolverConfig:
        """
        Returns a default SolverConfig instance for the specific solver.
        This can be overridden by subclasses to provide different defaults.
        """
        return SolverConfig()

    def _get_fluid_properties(self) -> Dict:
        """
        Computes and returns the fluid properties from the simulation config.
        """
        viscosity = calculate_viscosity(
            temperature=self.sim_config.temperature,
            oil_type=self.sim_config.oil_type,
            viscosity_model=self.sim_config.viscosity_model,
            viscosity_parameters=self.sim_config.viscosity_parameters
        )
        return {
            'density': self.sim_config.oil_density,
            'viscosity': viscosity
        }
