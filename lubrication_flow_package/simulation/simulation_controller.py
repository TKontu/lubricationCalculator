"""
Simulation Controller.
"""
from typing import Dict, Optional, Callable

from ..config.network_config import NetworkConfig, NetworkConfigLoader
from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..solvers.base import SolverBase
from ..solvers.nodal_matrix_solver import NodalMatrixSolver
from ..solvers.nonlinear_tree_solver import TreeSolver
from ..utils.network_builder import NetworkBuilder


class SimulationController:
    """
    Controls the simulation workflow, decoupling the GUI from the simulation logic.
    """

    def __init__(self):
        self.network_config: Optional[NetworkConfig] = None
        self.sim_config: Optional[SimulationConfig] = None
        self.network: Optional[FlowNetwork] = None
        self.solver: Optional[SolverBase] = None
        self.results: Optional[Dict] = None
        self.progress_callback: Optional[Callable[[str], None]] = None

    def set_progress_callback(self, callback: Callable[[str], None]):
        """Sets the callback function for progress updates."""
        self.progress_callback = callback

    def _report_progress(self, message: str):
        """Reports progress using the callback if it is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def load_configuration(self, network_config: NetworkConfig, sim_config: SimulationConfig):
        """Loads the network and simulation configuration."""
        self.network_config = network_config
        self.sim_config = sim_config
        self._report_progress("Configuration loaded.")
        self._build_network()

    def _build_network(self):
        """Builds the flow network from the configuration."""
        if not self.network_config:
            raise ValueError("Network configuration not loaded.")
        
        self.network, self.sim_config = NetworkConfigLoader.build_network(self.network_config)
        self._report_progress("Network built successfully.")

    def set_solver(self, solver_name: str):
        """Sets the solver to be used for the simulation."""
        if not self.sim_config:
            raise ValueError("Simulation configuration not loaded.")

        if solver_name == "nodal":
            self.solver = NodalMatrixSolver(self.sim_config, self.progress_callback)
        elif solver_name == "tree_nonlinear":
            self.solver = TreeSolver(self.sim_config, self.progress_callback)
        # Add other solvers here as they are implemented
        # elif solver_name == "robust_newton":
        #     self.solver = RobustNewtonSolver(self.sim_config, self.progress_callback)
        else:
            raise ValueError(f"Unknown solver: {solver_name}")
        
        self._report_progress(f"Solver set to: {solver_name}")

    def run_simulation(self) -> bool:
        """Runs the simulation and returns whether it converged."""
        if not self.network or not self.solver:
            self._report_progress("Cannot run simulation: network or solver not set.")
            return False

        self._report_progress("Starting simulation...")
        try:
            self.results = self.solver.solve(self.network)
            self._report_progress("Simulation finished.")
            if self.results and self.results.get("converged"):
                self._report_progress("Solver converged.")
                return True
            else:
                self._report_progress("Solver did not converge.")
                return False
        except Exception as e:
            self._report_progress(f"An error occurred during simulation: {e}")
            return False

    def get_results(self) -> Optional[Dict]:
        """Returns the simulation results."""
        return self.results

    def get_available_solvers(self) -> list[str]:
        """Returns a list of available solver names."""
        return ["nodal", "tree_nonlinear"]