"""
Non-Linear Hydraulic Network Solver for Tree-Like Networks.
"""

import numpy as np
from scipy.optimize import newton
from typing import Dict, Optional, Callable
import logging
import logging.config
import os
from collections import deque

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from .base import SolverBase
from .nodal_matrix_solver import NodalMatrixSolver

class TreeSolver(SolverBase):
    """
    A non-linear hydraulic network solver for tree-like (radial) networks.
    """

    def __init__(self, sim_config: SimulationConfig, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Initializes the TreeSolver.
        """
        super().__init__(sim_config, progress_callback)

        # Ensure logging is configured from logging.ini
        logging_config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logging.ini')
        logging_config_path = os.path.abspath(logging_config_path)
        if os.path.exists(logging_config_path):
            logging.config.fileConfig(logging_config_path, disable_existing_loggers=False)
            
        self.logger = logging.getLogger(__name__)


    def solve(self, network: FlowNetwork) -> Dict:
        """
        Main entry point for solving the hydraulic network using a robust
        nodal pressure formulation.
        """
        log_records = []
        warnings = []
        self._report_progress("Starting non-linear solver for tree-like networks.")
        # 0. Comprehensive Input Validation
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network configuration: {errors}")
        if self.sim_config.total_flow_rate <= 0:
            raise ValueError("Total flow rate must be positive.")

        if not network.outlet_nodes:
            raise ValueError("Network must have at least one outlet node.")

        # 1. Identify boundary nodes (all outlets) and unknown nodes
        outlet_node_ids = [node.id for node in network.outlet_nodes]
        unknown_node_ids = [nid for nid in network.nodes if nid not in outlet_node_ids]
        self._report_progress(f"Outlet nodes: {outlet_node_ids}, Unknown nodes: {len(unknown_node_ids)} (includes inlet)")

        # Define pressure bounds from simulation config
        inlet_pressure = self.sim_config.inlet_pressure or 200000.0
        outlet_pressure = self.sim_config.outlet_pressure or 101325.0

        # 2. Get initial guess for node pressures using a linear solver
        self._report_progress("Getting initial pressure guess from linear solver.")
        linear_solver = NodalMatrixSolver(self.sim_config)
        linear_solution = linear_solver.solve(network)
        initial_pressures = linear_solution['node_pressures']
        
        pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])
        self.logger.debug(f"Initial pressures guess from linear solver: {pressures}")

        converged = False
        last_residual_norm = -1
        stagnation_counter = 0
        iterations_run = 0

        for i in range(self.sim_config.max_iterations):
            iterations_run = i
            # 3. Evaluate the residual F(P)
            residual = self._evaluate_residual(pressures, network, unknown_node_ids, outlet_node_ids)
            residual_norm = np.linalg.norm(residual)
            
            # 4. Check for convergence - use both absolute and relative criteria
            if residual_norm < self.sim_config.tolerance:
                self._report_progress(f"Converged after {i} iterations.")
                converged = True
                break
            
            # Additional convergence check for small relative changes
            if i > 0:
                pressure_change = np.linalg.norm(pressures - prev_pressures) if 'prev_pressures' in locals() else float('inf')
                relative_pressure_change = pressure_change / (np.linalg.norm(pressures) + 1e-12)
                
                if relative_pressure_change < 1e-8 and residual_norm < 1e-3:
                    self._report_progress(f"Converged with relative tolerance after {i} iterations.")
                    converged = True
                    break
            
            prev_pressures = pressures.copy()

            # Check for stagnation
            if abs(residual_norm - last_residual_norm) < 1e-9:
                stagnation_counter += 1
                if stagnation_counter > 5:
                    warnings.append("Solver stalled. Converged with reduced tolerance.")
                    converged = True
                    break
            else:
                stagnation_counter = 0
            last_residual_norm = residual_norm

            # 5. Build the Jacobian matrix J(P)
            jacobian = self._build_jacobian(pressures, network, unknown_node_ids, outlet_node_ids)

            # 6. Solve the linear system J * delta_P = -F
            from scipy.sparse.linalg import spsolve
            try:
                delta_p = spsolve(jacobian, -residual)
            except Exception as e:
                self._report_progress(f"Linear solve failed: {e}. Jacobian may be singular.")
                log_records.append(f"ERROR: Linear solve failed: {e}. Jacobian may be singular.")
                log_records.append(f"Jacobian matrix:\n{jacobian.toarray()}")
                warnings.append(f"Linear solve failed: {e}.")
                break # Exit loop on failure
            
            # Adaptive step size limiting based on residual norm
            max_delta_p = min(100000.0, 10.0 * residual_norm)
            delta_p = np.clip(delta_p, -max_delta_p, max_delta_p)

            # 7. Update the solution with line search
            alpha = self._line_search(pressures, delta_p, residual, network, unknown_node_ids, outlet_node_ids)
            if alpha < 1e-8:
                self._report_progress("Alpha too small, solver may be stuck. Stopping.")
                warnings.append("Line search failed: alpha too small.")
                break

            pressures += alpha * delta_p
            self._report_progress(f"Iteration {i}: Residual Norm = {residual_norm:.6e}, Alpha = {alpha:.4f}")
            
            # Enforce pressure bounds
            pressures = np.clip(pressures, outlet_pressure, inlet_pressure)
            log_records.append(f"Iteration {i}: Residual Norm = {residual_norm:.6e}, Alpha = {alpha:.4f}, Delta P Norm = {np.linalg.norm(delta_p):.6e}")

        # Process and show logs
        if log_records:
            self.logger.debug("--- Solver Iteration Log ---")
            if len(log_records) <= 10:
                for record in log_records:
                    self.logger.debug(record)
            else:
                for record in log_records[:5]:
                    self.logger.debug(record)
                self.logger.debug("...")
                for record in log_records[-5:]:
                    self.logger.debug(record)
            self.logger.debug("----------------------------")
        
        if not converged:
            self._report_progress(f"Solver did not converge after {self.sim_config.max_iterations} iterations.")
            warnings.append("Solver did not converge within the maximum number of iterations.")

        # 8. Post-process results
        self._report_progress("Packaging results.")
        final_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        outlet_pressure = self.sim_config.outlet_pressure or 101325.0
        for outlet_id in outlet_node_ids:
            final_pressures[outlet_id] = outlet_pressure
        
        # Ensure inlet pressure is included in final solution
        if network.inlet_node.id in final_pressures:
            self._report_progress(f"Inlet pressure solved: {final_pressures[network.inlet_node.id]:.0f} Pa")
        else:
            self._report_progress("WARNING: Inlet pressure not found in solution")
            
        solution = self._get_final_solution(converged, iterations_run + 1, final_pressures, network)
        solution["warnings"].extend(warnings)
        return solution

    def _evaluate_residual(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list, outlet_node_ids) -> np.ndarray:
        """
        Evaluates the residual vector F(P) for the system of non-linear equations.
        The residual at each node is the net flow imbalance.
        """
        # Create a full pressure vector including all outlet pressures
        full_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        outlet_pressure = self.sim_config.outlet_pressure or 101325.0
        
        # Handle both single outlet_id (string) and list of outlet_ids
        if isinstance(outlet_node_ids, str):
            outlet_node_ids = [outlet_node_ids]
        
        for outlet_id in outlet_node_ids:
            full_pressures[outlet_id] = outlet_pressure

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

    def _build_jacobian(self, pressures: np.ndarray, network: FlowNetwork, unknown_node_ids: list, outlet_node_ids) -> np.ndarray:
        """
        Constructs the Jacobian matrix J(P) for the system using analytical derivatives.
        """
        from scipy.sparse import lil_matrix

        num_unknowns = len(unknown_node_ids)
        jacobian = lil_matrix((num_unknowns, num_unknowns))
        node_map = {node_id: i for i, node_id in enumerate(unknown_node_ids)}

        full_pressures = {nid: p for nid, p in zip(unknown_node_ids, pressures)}
        outlet_pressure = self.sim_config.outlet_pressure or 101325.0
        
        # Handle both single outlet_id (string) and list of outlet_ids
        if isinstance(outlet_node_ids, str):
            outlet_node_ids = [outlet_node_ids]
        
        for outlet_id in outlet_node_ids:
            full_pressures[outlet_id] = outlet_pressure

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

            if to_id in node_map:
                j = node_map[to_id]
                jacobian[j, j] += g
                if from_id in node_map:
                    i = node_map[from_id]
                    jacobian[j, i] -= g
        
        # Add adaptive regularization term to improve conditioning
        # Scale regularization based on the typical diagonal magnitude
        diagonal_values = [jacobian[i, i] for i in range(num_unknowns)]
        if diagonal_values:
            avg_diagonal = np.mean([abs(val) for val in diagonal_values if val != 0])
            if avg_diagonal > 0:
                regularization = max(avg_diagonal * 1e-6, 1e-6)
            else:
                regularization = 1e-6
        else:
            regularization = 1e-6
            
        for i in range(num_unknowns):
            jacobian[i, i] += regularization
            
        return jacobian.tocsr()

    def _line_search(self, pressures: np.ndarray, delta_p: np.ndarray, residual: np.ndarray, network: FlowNetwork, unknown_node_ids: list, outlet_node_ids) -> float:
        """
        Performs a robust line search to find an optimal step size alpha.
        Uses Armijo condition with adaptive backtracking.
        """
        alpha = 1.0
        c1 = 1e-4
        residual_norm_sq = np.dot(residual, residual)
        
        # Initial directional derivative
        gradient_dot_direction = -residual_norm_sq  # Since we're solving J*delta_p = -residual
        
        # Bound pressure updates to reasonable ranges
        outlet_pressure = self.sim_config.outlet_pressure or 101325.0
        inlet_pressure = self.sim_config.inlet_pressure or 200000.0
        
        for i in range(15): # Max 15 backtracks
            p_new = pressures + alpha * delta_p
            
            # Enforce pressure bounds during line search
            p_new = np.clip(p_new, outlet_pressure, inlet_pressure)
            
            try:
                new_residual = self._evaluate_residual(p_new, network, unknown_node_ids, outlet_node_ids)
                new_residual_norm_sq = np.dot(new_residual, new_residual)
                
                # Armijo condition: sufficient decrease
                if new_residual_norm_sq <= residual_norm_sq + alpha * c1 * gradient_dot_direction:
                    return alpha
                
                # More aggressive backtracking if we're not making progress
                if i < 5:
                    alpha *= 0.5
                else:
                    alpha *= 0.1
                    
            except Exception:
                # If residual evaluation fails, try smaller step
                alpha *= 0.1
                
        # If line search fails, return very small step
        return max(alpha, 1e-8)

    def _get_final_solution(self, converged: bool, iterations: int, pressures: Dict, network: FlowNetwork) -> Dict:
        """
        Packages the final results into the standard solution dictionary format.
        """
        # Calculate component flows from the final pressures
        component_flows = {}
        for conn in network.connections:
            p_from = pressures[conn.from_node.id]
            p_to = pressures[conn.to_node.id]
            dp = p_from - p_to
            component_flows[conn.component.id] = conn.component.calculate_flow_rate(dp, self.fluid_properties)
            
        # The total flow rate is a boundary condition from the simulation config.
        total_flow_rate = self.sim_config.total_flow_rate
        
        # Assemble the solution dictionary
        solution = {
            "converged": converged,
            "iterations": iterations,
            "component_flows": component_flows,
            "node_pressures": pressures,
            "inlet_pressure": pressures.get(network.inlet_node.id, 0.0),
            "total_flow_rate": total_flow_rate,
            "temperature": self.sim_config.temperature,
            "viscosity": self.fluid_properties['viscosity'],
            "warnings": []
        }
        
        if not converged:
            solution["warnings"].append("Solver did not converge within the specified tolerance or iterations.")
            
        return solution