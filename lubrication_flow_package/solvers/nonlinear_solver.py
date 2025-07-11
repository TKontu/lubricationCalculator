"""
Robust Non-Linear Hydraulic Network Solver based on Newton-Raphson.

This module implements the robust non-linear solver as outlined in the
non-linear_solver_plan.md document. It uses a full Newton-Raphson
method with a flow-based formulation to handle complex hydraulic networks
with strong non-linearities.
"""

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, linalg, lil_matrix
from typing import Dict, List, Tuple, Optional

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from .config import SolverConfig
from .base import SolverBase


class RobustNonLinearSolver(SolverBase):
    """
    A robust, non-linear hydraulic network solver using the Newton-Raphson method.

    This solver is designed to address the limitations of simpler iterative methods
    by constructing and solving the full Jacobian matrix for the non-linear system
    of equations that describe the network's physics.
    """

    def __init__(self, sim_config: SimulationConfig, solver_config: Optional[SolverConfig] = None):
        """
        Initializes the RobustNonLinearSolver.

        Args:
            sim_config: The simulation configuration object.
            config: A SolverConfig object containing solver parameters.
        """
        super().__init__(sim_config, solver_config)
        self.convergence_config = self.config.convergence
        self.line_search_config = self.config.line_search
        self.jacobian_config = self.config.jacobian

    def solve(self, network: FlowNetwork) -> Dict:
        """
        Main entry point for solving the hydraulic network.

        Args:
            network: The FlowNetwork object to be solved.

        Returns:
            A dictionary containing the solution, including flows, pressures,
            and convergence information.
        """
        # 1. Initialize flow vector Q
        q_initial = self._initialize_flows(network)
        q_current = q_initial

        # 2. Find fundamental cycles for pressure equations
        cycles = self._find_fundamental_cycles(network)
        
        converged = False
        iterations = 0

        # 3. Start Newton-Raphson iteration
        for i in range(self.config.max_iterations):
            iterations = i + 1
            # 4. Evaluate the residual F(Q)
            residual = self._evaluate_residual(q_current, network, cycles)

            # 5. Build the Jacobian matrix J(Q)
            jacobian = self._build_jacobian(q_current, network, cycles)

            # 6. Solve the linear system J * delta_Q = -F
            delta_q = self._solve_linear_system(jacobian, -residual)

            # 7. Check for convergence
            if self._check_convergence(residual, delta_q, q_current):
                print(f"Converged after {i} iterations.")
                converged = True
                break

            # 8. Update the solution with line search
            alpha = self._line_search(q_current, delta_q, residual, network, cycles)
            q_current = q_current + alpha * delta_q
        else:
            print("Solver did not converge within the maximum number of iterations.")

        # 9. Post-process results
        results = self._package_results(q_current, network, converged, iterations)
        return results

    def _line_search(self, q_current: np.ndarray, delta_q: np.ndarray, residual: np.ndarray, network: FlowNetwork, cycles: List[List[str]]) -> float:
        """
        Performs a line search to find an optimal step size alpha.
        """
        alpha = 1.0
        c1 = self.line_search_config.get("c1", 1e-4)
        residual_norm_sq = np.dot(residual, residual)

        for _ in range(self.line_search_config.get("max_backtracks", 10)):
            q_new = q_current + alpha * delta_q
            new_residual = self._evaluate_residual(q_new, network, cycles)
            new_residual_norm_sq = np.dot(new_residual, new_residual)

            if new_residual_norm_sq <= (1 - alpha * c1) * residual_norm_sq:
                return alpha
            
            alpha *= 0.5
        
        return alpha

    def _find_fundamental_cycles(self, network: FlowNetwork) -> List[List[str]]:
        """
        Finds a set of fundamental cycles in the network graph.
        """
        graph = nx.Graph()
        for conn in network.connections:
            graph.add_edge(conn.from_node.id, conn.to_node.id, component_id=conn.component.id)
        
        return nx.cycle_basis(graph)

    def _initialize_flows(self, network: FlowNetwork) -> np.ndarray:
        """
        Provides an initial guess for the flow rates in each component.
        """
        num_components = len(network.connections)
        total_flow = self.sim_config.total_flow_rate
        initial_flow = total_flow / num_components if num_components > 0 else 0
        return np.full(num_components, initial_flow)

    def _evaluate_residual(self, q_vector: np.ndarray, network: FlowNetwork, cycles: List[List[str]]) -> np.ndarray:
        """
        Evaluates the residual vector F(Q) for the system of non-linear equations.
        """
        num_nodes = len(network.nodes)
        num_cycles = len(cycles)
        num_equations = num_nodes + num_cycles

        residual = np.zeros(num_equations)
        comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

        # 1. Mass Conservation Equations
        for i, (node_id, node) in enumerate(network.nodes.items()):
            flow_sum = 0
            for conn in network.connections:
                if conn.to_node.id == node_id:
                    flow_sum += q_vector[comp_to_idx[conn.component.id]]
                elif conn.from_node.id == node_id:
                    flow_sum -= q_vector[comp_to_idx[conn.component.id]]
            
            if node.id == network.inlet_node.id:
                flow_sum -= self.sim_config.total_flow_rate
            
            residual[i] = flow_sum

        # 2. Pressure Loop Equations
        for i, cycle in enumerate(cycles):
            pressure_drop_sum = 0
            for j in range(len(cycle)):
                u, v = cycle[j], cycle[(j + 1) % len(cycle)]
                conn = network.get_connection_by_nodes(u, v)
                if conn:
                    q = q_vector[comp_to_idx[conn.component.id]]
                    dp = conn.component.calculate_pressure_drop(q, self.fluid_properties)
                    
                    if conn.from_node.id == u:
                        pressure_drop_sum += dp
                    else:
                        pressure_drop_sum -= dp
            
            residual[num_nodes + i] = pressure_drop_sum

        return residual

    def _build_jacobian(self, q_vector: np.ndarray, network: FlowNetwork, cycles: List[List[str]]) -> csr_matrix:
        """
        Constructs the Jacobian matrix J(Q) for the system.
        """
        num_nodes = len(network.nodes)
        num_cycles = len(cycles)
        num_connections = len(network.connections)
        num_equations = num_nodes + num_cycles
        num_variables = num_connections

        lil_jacobian = lil_matrix((num_equations, num_variables))
        comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

        # 1. Mass Conservation Jacobian
        node_to_idx = {node_id: i for i, node_id in enumerate(network.nodes)}
        for j, conn in enumerate(network.connections):
            from_node_idx = node_to_idx[conn.from_node.id]
            to_node_idx = node_to_idx[conn.to_node.id]
            lil_jacobian[from_node_idx, j] = -1
            lil_jacobian[to_node_idx, j] = 1

        # 2. Pressure Loop Jacobian
        for i, cycle in enumerate(cycles):
            for j in range(len(cycle)):
                u, v = cycle[j], cycle[(j + 1) % len(cycle)]
                conn = network.get_connection_by_nodes(u, v)
                if conn:
                    comp_idx = comp_to_idx[conn.component.id]
                    q = q_vector[comp_idx]
                    resistance = self._calculate_differential_resistance(conn.component, q, self.fluid_properties)
                    
                    if conn.from_node.id == u:
                        lil_jacobian[num_nodes + i, comp_idx] = resistance
                    else:
                        lil_jacobian[num_nodes + i, comp_idx] = -resistance
        
        return lil_jacobian.tocsr()

    def _calculate_differential_resistance(self, component, q_est: float, fluid_properties: dict) -> float:
        """
        Computes R = d(ΔP)/dQ for a component by central differencing.
        """
        delta_q = max(abs(q_est) * 1e-4, 1e-9)
        dp_plus = component.calculate_pressure_drop(q_est + delta_q, fluid_properties)
        dp_minus = component.calculate_pressure_drop(q_est - delta_q, fluid_properties)
        resistance = (dp_plus - dp_minus) / (2.0 * delta_q)
        return max(resistance, 1e-3)

    def _solve_linear_system(self, jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """
        Solves the linear system J * delta_Q = -F using a least-squares solver
        that can handle non-square matrices.
        """
        try:
            # lsqr is suitable for sparse, potentially non-square matrices.
            # It solves the equation Ax = b in a least-squares sense.
            result = linalg.lsqr(jacobian, -residual, atol=1e-9, btol=1e-9)
            delta_q = result[0]
            return delta_q
        except Exception as e:
            print(f"Error solving linear system: {e}")
            return np.zeros(jacobian.shape[1])

    def _check_convergence(self, residual: np.ndarray, delta_q: np.ndarray, q_current: np.ndarray) -> bool:
        """
        Checks if the solution has converged based on multiple criteria.
        """
        residual_norm = np.linalg.norm(residual)
        if residual_norm >= self.config.convergence.get("residual_tolerance", 1e-6):
            return False

        q_norm = np.linalg.norm(q_current)
        delta_q_norm = np.linalg.norm(delta_q)
        
        if q_norm > 1e-9: # Avoid division by zero for zero flow
            relative_change = delta_q_norm / q_norm
            if relative_change >= self.config.convergence.get("relative_tolerance", 1e-6):
                return False

        return True

    def _package_results(self, q_vector: np.ndarray, network: FlowNetwork, converged: bool, iterations: int) -> Dict:
        """
        Packages the final flow vector and calculates node pressures.
        """
        component_flows = {conn.component.id: q_vector[i] for i, conn in enumerate(network.connections)}
        
        node_pressures = self._calculate_node_pressures(component_flows, network)

        inlet_pressure = node_pressures.get(network.inlet_node.id, 0.0)

        return {
            "converged": converged,
            "iterations": iterations,
            "component_flows": component_flows,
            "node_pressures": node_pressures,
            "inlet_pressure": inlet_pressure,
            "temperature": self.sim_config.temperature,
            "viscosity": self.fluid_properties['viscosity'],
            "oil_type": self.sim_config.oil_type,
            "oil_density": self.sim_config.oil_density,
            "total_flow_rate": self.sim_config.total_flow_rate,
            "outlet_pressure": self.sim_config.outlet_pressure or 0.0,
        }

    def _calculate_node_pressures(self, component_flows: Dict[str, float], network: FlowNetwork) -> Dict[str, float]:
        """
        Calculates node pressures based on a reference pressure and solved flow rates.
        """
        node_pressures = {}
        ref_node_id = network.outlet_nodes[0].id if network.outlet_nodes else network.inlet_node.id
        node_pressures[ref_node_id] = self.sim_config.outlet_pressure or 0.0

        # Use BFS to traverse the network and calculate pressures
        q = [(ref_node_id, node_pressures[ref_node_id])]
        visited = {ref_node_id}

        while q:
            curr_node_id, curr_pressure = q.pop(0)

            for conn in network.connections:
                if conn.to_node.id == curr_node_id and conn.from_node.id not in visited:
                    neighbor_id = conn.from_node.id
                    flow = component_flows[conn.component.id]
                    dp = conn.component.calculate_pressure_drop(flow, self.fluid_properties)
                    neighbor_pressure = curr_pressure + dp
                    node_pressures[neighbor_id] = neighbor_pressure
                    visited.add(neighbor_id)
                    q.append((neighbor_id, neighbor_pressure))

                elif conn.from_node.id == curr_node_id and conn.to_node.id not in visited:
                    neighbor_id = conn.to_node.id
                    flow = component_flows[conn.component.id]
                    dp = conn.component.calculate_pressure_drop(flow, self.fluid_properties)
                    neighbor_pressure = curr_pressure - dp
                    node_pressures[neighbor_id] = neighbor_pressure
                    visited.add(neighbor_id)
                    q.append((neighbor_id, neighbor_pressure))
        
        return node_pressures

    def print_results(self, network: FlowNetwork, solution: Dict, pressure_unit: str = 'kPa', flow_rate_unit: str = 'L/s'):
        """Print detailed results in a structured and clear format."""
        
        def convert_pressure(p_pa, unit):
            if unit.lower() == 'bar':
                return p_pa / 100000, 'bar'
            return p_pa / 1000, 'kPa'

        def convert_flow_rate(q_m3s, unit):
            if unit.lower() == 'l/min':
                return q_m3s * 60000, 'L/min'
            return q_m3s * 1000, 'L/s'

        p_unit_str = pressure_unit
        q_unit_str = flow_rate_unit
        
        connection_flows = solution.get("component_flows", {})
        solution_info = solution

        print(f"\n{'='*80}")
        print(f"NETWORK FLOW SIMULATION RESULTS (RobustNonLinearSolver)")
        print(f"{'='*80}")
        
        # --- General Information ---
        print(f"  Network Name:      {network.name}")
        print(f"  Temperature:       {solution_info.get('temperature', 'N/A'):.1f}°C")
        print(f"  Oil Type:          {self.sim_config.oil_type}")
        print(f"  Oil Density:       {self.sim_config.oil_density:.1f} kg/m³")
        print(f"  Dynamic Viscosity: {self.fluid_properties['viscosity']:.6f} Pa·s")
        
        # --- Simulation Summary ---
        total_flow_rate = self.sim_config.total_flow_rate
        total_flow_rate_disp, q_unit_str = convert_flow_rate(total_flow_rate, flow_rate_unit)
        print(f"\n  Total System Flow Rate: {total_flow_rate_disp:.2f} {q_unit_str}")
        
        inlet_pressure = solution_info.get('inlet_pressure', 0.0)
        inlet_pressure_disp, p_unit_str = convert_pressure(inlet_pressure, pressure_unit)
        print(f"  Inlet Pressure:         {inlet_pressure_disp:.2f} {p_unit_str}")
        
        converged = solution_info.get('converged', False)
        iterations = solution_info.get('iterations', 'N/A')
        print(f"  Solver Converged:       {'Yes' if converged else 'No'} (in {iterations} iterations)")
        
        # --- Outlet Flow Distribution ---
        print(f"\n{'='*80}")
        print("OUTLET FLOW DISTRIBUTION")
        print(f"{'='*80}")
        print(f"  {'Outlet Node':<25} {'Flow Rate (' + q_unit_str + ')':<20} {'Percentage of Total':<25}")
        print(f"  {'-'*25} {'-'*20} {'-'*25}")
        
        outlet_nodes = network.outlet_nodes
        total_outlet_flow = 0
        
        for outlet_node in outlet_nodes:
            for conn in network.connections:
                if conn.to_node.id == outlet_node.id:
                    flow = connection_flows.get(conn.component.id, 0.0)
                    total_outlet_flow += flow
                    flow_disp, _ = convert_flow_rate(flow, flow_rate_unit)
                    percentage = (flow / total_flow_rate * 100) if total_flow_rate > 0 else 0
                    print(f"  {outlet_node.name:<25} {flow_disp:<20.3f} {percentage:>24.1f}%")
        
        total_outlet_flow_disp, _ = convert_flow_rate(total_outlet_flow, flow_rate_unit)
        print(f"  {'-'*25} {'-'*20} {'-'*25}")
        print(f"  {'Total Outlet Flow':<25} {total_outlet_flow_disp:<20.3f}")

        # --- Pressure and Flow Details ---
        print(f"\n{'='*80}")
        print("PRESSURE AND FLOW DETAILS")
        print(f"{'='*80}")
        
        print(f"  {'Component':<20} {'Type':<15} {'Flow Rate (' + q_unit_str + ')':<20} {'Pressure Drop (' + p_unit_str + ')':<20}")
        print(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*20}")
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows.get(component.id, 0.0)
            pressure_drop = component.calculate_pressure_drop(flow_rate, self.fluid_properties)
            flow_rate_disp, _ = convert_flow_rate(flow_rate, flow_rate_unit)
            pressure_drop_disp, _ = convert_pressure(pressure_drop, pressure_unit)
            
            print(f"  {component.name:<20} {type(component).__name__:<15} "
                  f"{flow_rate_disp:<20.3f} {pressure_drop_disp:<20.2f}")
        
        print(f"\n  {'Node':<20} {'Pressure (' + p_unit_str + ')':<20} {'Elevation (m)':<15}")
        print(f"  {'-'*20} {'-'*20} {'-'*15}")
        
        sorted_nodes = sorted(solution_info.get('node_pressures', {}).items(), key=lambda item: item[1], reverse=True)
        
        for node_id, pressure in sorted_nodes:
            node = network.nodes.get(node_id)
            if node:
                pressure_disp, _ = convert_pressure(pressure, pressure_unit)
                print(f"  {node.name:<20} {pressure_disp:<20.2f} {node.elevation:<15.1f}")
        
        print(f"\n{'='*80}\n")
