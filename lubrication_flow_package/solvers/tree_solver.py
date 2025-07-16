"""
Non-Linear Hydraulic Network Solver for Tree-Like Networks.
"""

import numpy as np
from scipy.optimize import newton
from typing import Dict, Optional
import logging

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from .config import SolverConfig
from .base import SolverBase
from .nodal_matrix_solver import NodalMatrixSolver

class NonLinearTreeSolver(SolverBase):
    """
    A non-linear hydraulic network solver for tree-like (radial) networks.
    """

    def __init__(self, sim_config: SimulationConfig, solver_config: Optional[SolverConfig] = None):
        """
        Initializes the NonLinearTreeSolver.
        """
        super().__init__(sim_config, solver_config)
        self.logger = logging.getLogger(__name__)

    def solve(self, network: FlowNetwork) -> Dict:
        """
        Main entry point for solving the hydraulic network using a robust
        nodal pressure formulation.
        """
        # 0. Comprehensive Input Validation
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network configuration: {errors}")
        if self.sim_config.total_flow_rate <= 0:
            raise ValueError("Total flow rate must be positive.")

        if not network.outlet_nodes:
            raise ValueError("Network must have at least one outlet node.")

        # 1. Identify a reference node (an outlet) and unknown nodes
        ref_node_id = network.outlet_nodes[0].id
        unknown_node_ids = [nid for nid in network.nodes if nid != ref_node_id]
        node_to_idx = {nid: i for i, nid in enumerate(unknown_node_ids)}

        # 2. Get initial guess for node pressures from the linear solver
        linear_solver = NodalMatrixSolver(self.sim_config, self.config)
        initial_solution = linear_solver.solve(network)
        
        # Validate the initial guess
        initial_pressures = initial_solution['node_pressures']
        if not all(np.isfinite(list(initial_pressures.values()))):
            self.logger.warning("Linear solver produced non-finite initial pressures. Falling back to a simple guess.")
            # Fallback to a simple pressure distribution
            inlet_pressure = self.sim_config.inlet_pressure or 200000.0
            outlet_pressure = self.sim_config.outlet_pressure or 101325.0
            for nid in initial_pressures:
                initial_pressures[nid] = (inlet_pressure + outlet_pressure) / 2.0

        pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])

        converged = False
        for i in range(self.config.max_iterations):
            self.logger.debug(f"Iteration {i}: pressures = {pressures}")
            
            # 3. Evaluate the residual F(P)
            residual = self._evaluate_residual(pressures, network, unknown_node_ids, ref_node_id)
            self.logger.debug(f"Iteration {i}: residual norm = {np.linalg.norm(residual)}")

            # 4. Check for convergence
            if np.linalg.norm(residual) < self.config.tolerance:
                self.logger.info(f"Converged after {i} iterations.")
                converged = True
                break

            # 5. Build the Jacobian matrix J(P)
            jacobian = self._build_jacobian(pressures, network, unknown_node_ids, ref_node_id)
            self.logger.debug(f"Iteration {i}: jacobian = \n{jacobian.toarray()}")

            # 6. Solve the linear system J * delta_P = -F
            from scipy.sparse.linalg import spsolve
            try:
                delta_p = spsolve(jacobian, -residual)
            except Exception as e:
                self.logger.error(f"Linear solve failed: {e}. Jacobian may be singular.")
                break # Exit loop on failure
            
            self.logger.debug(f"Iteration {i}: delta_p = {delta_p}")

            # 7. Update the solution with damping and pressure bounds
            alpha = self._line_search(pressures, delta_p, residual, network, unknown_node_ids, ref_node_id)
            self.logger.debug(f"Iteration {i}: alpha = {alpha}")
            pressures += alpha * delta_p

            # Enforce pressure bounds
            inlet_pressure = self.sim_config.inlet_pressure or 1e6 # A large default
            outlet_pressure = self.sim_config.outlet_pressure or 0.0
            pressures = np.clip(pressures, outlet_pressure, inlet_pressure)
        
        # 8. Post-process results
        final_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        final_pressures[ref_node_id] = self.sim_config.outlet_pressure or 0.0
            
        component_flows = {}
        for conn in network.connections:
            p_from = final_pressures[conn.from_node.id]
            p_to = final_pressures[conn.to_node.id]
            dp = p_from - p_to
            component_flows[conn.component.id] = conn.component.calculate_flow_rate(dp, self.fluid_properties)

        return {
            "converged": converged,
            "iterations": i + 1,
            "component_flows": component_flows,
            "node_pressures": final_pressures,
            "inlet_pressure": final_pressures[network.inlet_node.id],
            "temperature": self.sim_config.temperature,
            "viscosity": self.fluid_properties['viscosity'],
        }

    def _evaluate_residual(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list, ref_node_id: str) -> np.ndarray:
        """
        Evaluates the residual vector F(P) for the system of non-linear equations.
        The residual at each node is the net flow imbalance.
        """
        # Create a full pressure vector including the reference pressure
        full_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        full_pressures[ref_node_id] = self.sim_config.outlet_pressure or 0.0

        residuals = np.zeros(len(unknown_node_ids))

        for i, node_id in enumerate(unknown_node_ids):
            net_flow = 0
            # Sum flows from all connections to this node
            for conn in network.connections:
                if conn.from_node.id == node_id:
                    p_other = full_pressures[conn.to_node.id]
                    pressure_drop = full_pressures[node_id] - p_other
                    flow = conn.component.calculate_flow_rate(pressure_drop, self.fluid_properties)
                    net_flow -= flow
                elif conn.to_node.id == node_id:
                    p_other = full_pressures[conn.from_node.id]
                    pressure_drop = p_other - full_pressures[node_id]
                    flow = conn.component.calculate_flow_rate(pressure_drop, self.fluid_properties)
                    net_flow += flow
            
            # Add external flow constraint for the inlet node
            if node_id == network.inlet_node.id:
                net_flow -= self.sim_config.total_flow_rate

            residuals[i] = net_flow
            
        return residuals

    def _build_jacobian(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list, ref_node_id: str) -> np.ndarray:
        """
        Constructs the Jacobian matrix J(P) for the system using finite differences.
        """
        from scipy.sparse import lil_matrix

        num_unknowns = len(unknown_node_ids)
        jacobian = lil_matrix((num_unknowns, num_unknowns))
        
        # Base residual
        f0 = self._evaluate_residual(pressures, network, unknown_node_ids, ref_node_id)

        # Perturb each pressure and calculate the change in the residual
        for j in range(num_unknowns):
            p_perturbed = pressures.copy()
            # Add a small, adaptive perturbation
            pressure_mag = abs(p_perturbed[j])
            delta_p = max(1e-6 * pressure_mag, 1e-3)
            p_perturbed[j] += delta_p
            
            f1 = self._evaluate_residual(p_perturbed, network, unknown_node_ids, ref_node_id)
            
            # The j-th column of the Jacobian is (f1 - f0) / delta_p
            jacobian[:, j] = (f1 - f0) / delta_p

        # Add a small regularization term to the diagonal to improve stability
        for i in range(num_unknowns):
            jacobian[i, i] += 1e-9
            
        return jacobian.tocsr()

    def _line_search(self, pressures: np.ndarray, delta_p: np.ndarray, residual: np.ndarray, network: FlowNetwork, unknown_node_ids: list, ref_node_id: str) -> float:
        """
        Performs a line search to find an optimal step size alpha.
        """
        alpha = 1.0
        c1 = 1e-4
        residual_norm_sq = np.dot(residual, residual)

        for _ in range(10): # Max 10 backtracks
            p_new = pressures + alpha * delta_p
            new_residual = self._evaluate_residual(p_new, network, unknown_node_ids, ref_node_id)
            new_residual_norm_sq = np.dot(new_residual, new_residual)

            if new_residual_norm_sq <= (1 - alpha * c1) * residual_norm_sq:
                return alpha
            
            alpha *= 0.5
        
        return alpha
