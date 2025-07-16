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
        self.logger.info("Starting non-linear solver.")
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
        self.logger.debug(f"Reference node: {ref_node_id}")
        self.logger.debug(f"Unknown nodes: {unknown_node_ids}")

        # 2. Get initial guess for node pressures using a linear solver
        linear_solver = NodalMatrixSolver(self.sim_config)
        linear_solution = linear_solver.solve(network)
        initial_pressures = linear_solution['node_pressures']
        
        pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])
        self.logger.debug(f"Initial pressures guess from linear solver: {pressures}")

        converged = False
        for i in range(self.config.max_iterations):
            # 3. Evaluate the residual F(P)
            residual = self._evaluate_residual(pressures, network, unknown_node_ids, ref_node_id)
            residual_norm = np.linalg.norm(residual)
            
            # 4. Check for convergence
            if residual_norm < self.config.tolerance:
                self.logger.info(f"Converged after {i} iterations.")
                converged = True
                break

            # 5. Build the Jacobian matrix J(P)
            jacobian = self._build_jacobian(pressures, network, unknown_node_ids, ref_node_id)

            # 6. Solve the linear system J * delta_P = -F
            from scipy.sparse.linalg import spsolve
            try:
                delta_p = spsolve(jacobian, -residual)
            except Exception as e:
                self.logger.error(f"Linear solve failed: {e}. Jacobian may be singular.")
                break # Exit loop on failure
            
            # Limit the pressure update
            max_delta_p = 100000.0
            delta_p = np.clip(delta_p, -max_delta_p, max_delta_p)

            # 7. Update the solution with damping and pressure bounds
            alpha = self._line_search(pressures, delta_p, residual, network, unknown_node_ids, ref_node_id)
            pressures += alpha * delta_p

            # Enforce pressure bounds
            pressures = np.clip(pressures, outlet_pressure, inlet_pressure)
        
        if not converged:
            self.logger.warning(f"Solver did not converge after {self.config.max_iterations} iterations. Final residual norm: {residual_norm:.6f}")

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
                net_flow += self.sim_config.total_flow_rate

            residuals[i] = net_flow
            
        return residuals

    def _build_jacobian(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list, ref_node_id: str) -> np.ndarray:
        """
        Constructs the Jacobian matrix J(P) for the system using analytical derivatives.
        """
        from scipy.sparse import lil_matrix

        num_unknowns = len(unknown_node_ids)
        jacobian = lil_matrix((num_unknowns, num_unknowns))
        node_map = {node_id: i for i, node_id in enumerate(unknown_node_ids)}

        full_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        full_pressures[ref_node_id] = self.sim_config.outlet_pressure or 0.0

        for conn in network.connections:
            from_id, to_id = conn.from_node.id, conn.to_node.id
            
            pressure_drop = full_pressures.get(from_id, 0) - full_pressures.get(to_id, 0)
            flow = conn.component.calculate_flow_rate(pressure_drop, self.fluid_properties)
            
            try:
                g = 1.0 / conn.component.get_differential_resistance(flow, self.fluid_properties)
            except ZeroDivisionError:
                g = 1e9  # A large conductance for near-zero resistance

            if from_id in node_map:
                i = node_map[from_id]
                jacobian[i, i] += g
                if to_id in node_map:
                    j = node_map[to_id]
                    jacobian[i, j] -= g
                    jacobian[j, i] -= g

            if to_id in node_map:
                j = node_map[to_id]
                jacobian[j, j] += g
        
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
