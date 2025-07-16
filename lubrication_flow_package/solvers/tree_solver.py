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
        Main entry point for solving the hydraulic network.
        """
        # 1. Get initial guess for node pressures from the linear solver
        linear_solver = NodalMatrixSolver(self.sim_config, self.config)
        initial_solution = linear_solver.solve(network)
        
        initial_pressures_dict = initial_solution['node_pressures']
        
        # Identify unknown pressures (all nodes that are not outlets)
        unknown_node_ids = [nid for nid, node in network.nodes.items() if node not in network.outlet_nodes]
        
        pressures = np.array([initial_pressures_dict[nid] for nid in unknown_node_ids])

        converged = False
        for i in range(self.config.max_iterations):
            self.logger.debug(f"Iteration {i}: pressures = {pressures}")
            # 4. Evaluate the residual F(P)
            residual = self._evaluate_residual(pressures, network, unknown_node_ids)
            self.logger.debug(f"Iteration {i}: residual = {residual}")

            # 5. Check for convergence
            if np.linalg.norm(residual) < self.config.tolerance:
                self.logger.info(f"Converged after {i} iterations.")
                converged = True
                break

            # 6. Build the Jacobian matrix J(P)
            jacobian = self._build_jacobian(pressures, network, unknown_node_ids)
            self.logger.debug(f"Iteration {i}: jacobian = \n{jacobian.toarray()}")

            # 7. Solve the linear system J * delta_P = -F
            from scipy.sparse.linalg import spsolve
            delta_p = spsolve(jacobian, -residual)
            self.logger.debug(f"Iteration {i}: delta_p = {delta_p}")

            # 8. Update the solution
            pressures += delta_p
        
        # Post-process results
        final_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        for outlet_node in network.outlet_nodes:
            final_pressures[outlet_node.id] = self.sim_config.outlet_pressure or 0.0
            
        # Calculate final flows
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

    def _evaluate_residual(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list) -> np.ndarray:
        """
        Evaluates the residual vector F(P) for the system of non-linear equations.
        The residual at each node is the net flow imbalance.
        """
        
        # Create a full pressure vector including outlet pressures
        full_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        for outlet_node in network.outlet_nodes:
            full_pressures[outlet_node.id] = self.sim_config.outlet_pressure or 0.0

        residuals = np.zeros(len(unknown_node_ids))
        node_to_idx = {nid: i for i, nid in enumerate(unknown_node_ids)}

        for i, node_id in enumerate(unknown_node_ids):
            # Skip the inlet node, its pressure is determined by the flow constraint
            if node_id == network.inlet_node.id:
                continue

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
            
            residuals[i] = net_flow
            
        # The residual for the inlet node is the total flow
        inlet_idx = node_to_idx[network.inlet_node.id]
        inlet_flow = 0
        for conn in network.connections:
            if conn.from_node.id == network.inlet_node.id:
                p_other = full_pressures[conn.to_node.id]
                pressure_drop = full_pressures[network.inlet_node.id] - p_other
                inlet_flow += conn.component.calculate_flow_rate(pressure_drop, self.fluid_properties)
        
        residuals[inlet_idx] = self.sim_config.total_flow_rate - inlet_flow

        return residuals

    def _build_jacobian(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list) -> np.ndarray:
        """
        Constructs the Jacobian matrix J(P) for the system using finite differences.
        """
        from scipy.sparse import lil_matrix

        num_unknowns = len(unknown_node_ids)
        jacobian = lil_matrix((num_unknowns, num_unknowns))
        
        # Base residual
        f0 = self._evaluate_residual(pressures, network, unknown_node_ids)

        # Perturb each pressure and calculate the change in the residual
        for j in range(num_unknowns):
            p_perturbed = pressures.copy()
            # Add a small perturbation
            delta_p = 1e-3 # Pa
            p_perturbed[j] += delta_p
            
            f1 = self._evaluate_residual(p_perturbed, network, unknown_node_ids)
            
            # The j-th column of the Jacobian is (f1 - f0) / delta_p
            jacobian[:, j] = (f1 - f0) / delta_p
            
        return jacobian.tocsr()
