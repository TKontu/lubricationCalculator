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
from scipy.sparse.linalg import spsolve
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
        Provides an initial guess for the flow rates by solving a simplified
        linearized version of the network. This version correctly handles
        multiple outlets by connecting them to a single virtual atmosphere node.
        """
        num_connections = len(network.connections)
        if num_connections == 0:
            return np.array([])

        # 1. Estimate a linear resistance for each *real* component
        resistances = {}
        for conn in network.connections:
            resistances[conn.component.id] = self._calculate_differential_resistance(
                conn.component, 0.0, self.fluid_properties
            )

        # 2. Build the structure for the linear solve, including a virtual node
        
        # Create a temporary, expanded node list including the virtual atmosphere node
        virtual_node_id = "VIRTUAL_ATMOSPHERE_NODE"
        node_ids = list(network.nodes.keys()) + [virtual_node_id]
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

        # The reference node is the new virtual node
        ref_idx = node_to_idx[virtual_node_id]
        active_indices = [i for i, node_id in enumerate(node_ids) if i != ref_idx]
        idx_map = {full_idx: active_idx for active_idx, full_idx in enumerate(active_indices)}
        
        n_active = len(active_indices)
        G = lil_matrix((n_active, n_active))
        b = np.zeros(n_active)

        # 3. Build the conductance matrix for the *real* connections
        for conn in network.connections:
            conductance = 1.0 / resistances[conn.component.id]
            i = node_to_idx[conn.from_node.id]
            j = node_to_idx[conn.to_node.id]

            # Both nodes are non-reference (i.e., not the virtual node)
            i_act, j_act = idx_map[i], idx_map[j]
            G[i_act, i_act] += conductance
            G[j_act, j_act] += conductance
            G[i_act, j_act] -= conductance
            G[j_act, i_act] -= conductance

        # 4. Add virtual connections from each outlet to the virtual atmosphere node
        # Use a very high conductance (very low resistance) for these virtual pipes.
        virtual_conductance = 1.0 / self.config.min_resistance
        for outlet_node in network.outlet_nodes:
            i = node_to_idx[outlet_node.id]
            i_act = idx_map[i]
            G[i_act, i_act] += virtual_conductance

        # 5. Set the total flow rate at the inlet node
        inlet_idx = node_to_idx[network.inlet_node.id]
        inlet_act = idx_map[inlet_idx]
        b[inlet_act] = self.sim_config.total_flow_rate

        # 6. Solve the linear system G*P = b for the node pressures
        try:
            pressures_active = spsolve(G.tocsr(), b)
            
            pressures = np.zeros(len(node_ids))
            for i, active_idx in enumerate(active_indices):
                pressures[active_idx] = pressures_active[i]
            
            # 7. Calculate initial flows from the solved pressures for the *real* connections
            q_initial = np.zeros(num_connections)
            for i, conn in enumerate(network.connections):
                p_from = pressures[node_to_idx[conn.from_node.id]]
                p_to = pressures[node_to_idx[conn.to_node.id]]
                resistance = resistances[conn.component.id]
                q_initial[i] = (p_from - p_to) / resistance
            
            return q_initial

        except Exception as e:
            # Fallback if the linear solve fails
            print(f"Warning: Linear initialization failed ({e}). Falling back to simple initialization.")
            return np.full(num_connections, self.sim_config.total_flow_rate / num_connections)

    def _evaluate_residual(self, q_vector: np.ndarray, network: FlowNetwork, cycles: List[List[str]]) -> np.ndarray:
        """
        Evaluates the residual vector F(Q) for the system of non-linear equations.
        The system is composed of (N-1) mass conservation equations and L pressure loop equations.
        """
        if not network.inlet_node:
            raise ValueError("An inlet node must be defined in the network for the non-linear solver.")

        num_connections = len(network.connections)
        num_mass_equations = len(network.nodes) - 1
        
        # The number of cycle equations must make the system square
        num_cycles = num_connections - num_mass_equations
        num_equations = num_mass_equations + num_cycles

        residual = np.zeros(num_equations)
        comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

        # 1. Mass Conservation Equations (excluding the reference node, e.g., inlet)
        reference_node_id = network.inlet_node.id
        node_list = [node for node_id, node in network.nodes.items() if node_id != reference_node_id]

        for i, node in enumerate(node_list):
            flow_sum = 0
            for conn in network.connections:
                if conn.to_node.id == node.id:
                    flow_sum += q_vector[comp_to_idx[conn.component.id]]
                elif conn.from_node.id == node.id:
                    flow_sum -= q_vector[comp_to_idx[conn.component.id]]
            
            residual[i] = flow_sum

        # 2. Pressure Loop Equations
        for i in range(num_cycles):
            cycle = cycles[i]
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
            
            residual[num_mass_equations + i] = pressure_drop_sum

        return residual

    def _build_jacobian(self, q_vector: np.ndarray, network: FlowNetwork, cycles: List[List[str]]) -> csr_matrix:
        """
        Constructs the square Jacobian matrix J(Q) for the system.
        """
        if not network.inlet_node:
            raise ValueError("An inlet node must be defined in the network for the non-linear solver.")

        num_connections = len(network.connections)
        num_variables = num_connections
        
        num_mass_equations = len(network.nodes) - 1
        
        # The number of cycle equations must make the system square
        num_cycles = num_connections - num_mass_equations
        num_equations = num_mass_equations + num_cycles

        if num_equations != num_variables:
            # This check is now more of a safeguard; the logic should always produce a square system.
            raise RuntimeError(
                f"The system must be square. Equations: {num_equations}, Variables: {num_variables}"
            )

        lil_jacobian = lil_matrix((num_equations, num_variables))
        comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

        # 1. Mass Conservation Jacobian (excluding the reference node)
        reference_node_id = network.inlet_node.id
        node_list = [node for node_id, node in network.nodes.items() if node_id != reference_node_id]
        node_to_row_idx = {node.id: i for i, node in enumerate(node_list)}

        for col_idx, conn in enumerate(network.connections):
            if conn.from_node.id in node_to_row_idx:
                row_idx = node_to_row_idx[conn.from_node.id]
                lil_jacobian[row_idx, col_idx] = -1
            
            if conn.to_node.id in node_to_row_idx:
                row_idx = node_to_row_idx[conn.to_node.id]
                lil_jacobian[row_idx, col_idx] = 1

        # 2. Pressure Loop Jacobian
        for i in range(num_cycles):
            cycle = cycles[i]
            row_idx = num_mass_equations + i
            for j in range(len(cycle)):
                u, v = cycle[j], cycle[(j + 1) % len(cycle)]
                conn = network.get_connection_by_nodes(u, v)
                if conn:
                    col_idx = comp_to_idx[conn.component.id]
                    q = q_vector[col_idx]
                    resistance = self._calculate_differential_resistance(conn.component, q, self.fluid_properties)
                    
                    if conn.from_node.id == u:
                        lil_jacobian[row_idx, col_idx] = resistance
                    else:
                        lil_jacobian[row_idx, col_idx] = -resistance
        
        return lil_jacobian.tocsr()

    def _calculate_differential_resistance(self, component, q_est: float, fluid_properties: dict) -> float:
        """
        Computes R = d(ΔP)/dQ for a component by central differencing.
        """
        delta_q = max(abs(q_est) * 1e-6, 1e-9) # Use a smaller relative step for better accuracy
        dp_plus = component.calculate_pressure_drop(q_est + delta_q, fluid_properties)
        dp_minus = component.calculate_pressure_drop(q_est - delta_q, fluid_properties)
        resistance = (dp_plus - dp_minus) / (2.0 * delta_q)
        return max(resistance, self.config.min_resistance)

    def _solve_linear_system(self, jacobian: csr_matrix, residual: np.ndarray) -> np.ndarray:
        """
        Solves the linear system J * delta_Q = -residual using a direct sparse solver.
        """
        try:
            # Use spsolve for sparse, square linear systems.
            delta_q = linalg.spsolve(jacobian, -residual)
            return delta_q
        except linalg.LinAlgError as e:
            # This can happen if the Jacobian is singular.
            print(f"Warning: Linear system may be singular. Solver failed with: {e}")
            # Return a zero vector as a fallback to prevent crashing.
            return np.zeros(jacobian.shape[1])
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
        Calculates node pressures by solving a linear system after flows are known.
        This method is more robust for networks with loops than a simple traversal.
        """
        node_ids = list(network.nodes.keys())
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        n_nodes = len(node_ids)

        # At least one reference pressure is needed. Use the first outlet node.
        if not network.outlet_nodes:
            raise ValueError("At least one outlet node must be defined to set a reference pressure.")
        
        ref_node_id = network.outlet_nodes[0].id
        ref_idx = node_to_idx[ref_node_id]
        ref_pressure = self.sim_config.outlet_pressure or 0.0

        # A is the conductance matrix, b is the flow vector
        A = lil_matrix((n_nodes, n_nodes))
        b = np.zeros(n_nodes)

        for conn in network.connections:
            q = component_flows[conn.component.id]
            dp = conn.component.calculate_pressure_drop(q, self.fluid_properties)
            
            i = node_to_idx[conn.from_node.id]
            j = node_to_idx[conn.to_node.id]

            # Build a system based on ΔP = P_i - P_j
            # We can use a pseudo-conductance of 1 since we are solving for P directly
            A[i, i] += 1
            A[i, j] -= 1
            b[i] += dp

            A[j, j] += 1
            A[j, i] -= 1
            b[j] -= dp

        # Set the reference pressure constraint
        A[ref_idx, :] = 0
        A[ref_idx, ref_idx] = 1
        b[ref_idx] = ref_pressure

        # Solve the linear system A*P = b for the node pressures
        try:
            pressures = spsolve(A.tocsr(), b)
            return {node_id: pressures[node_to_idx[node_id]] for node_id in node_ids}
        except Exception as e:
            print(f"Warning: Pressure calculation failed: {e}. Falling back to BFS.")
            # Fallback to the old BFS method if the linear solve fails
            return self._calculate_node_pressures_bfs(component_flows, network)

    def _calculate_node_pressures_bfs(self, component_flows: Dict[str, float], network: FlowNetwork) -> Dict[str, float]:
        """
        Calculates node pressures based on a reference pressure and solved flow rates using BFS.
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
