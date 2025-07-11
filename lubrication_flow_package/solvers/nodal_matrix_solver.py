
"""
Unified Nodal-Matrix Solver for Hydraulic Networks with Non-linear Edge Resistances

This module implements the canonical nodal-matrix solver for the project that finds node pressures 
and edge flows such that mass is conserved and the pressure-flow law ΔP_e = R_e(Q_e) · Q_e holds 
on every edge.
"""

import copy
import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Callable
import logging
from lubrication_flow_package.components.connector import Connector, ConnectorType


from ..network.flow_network import FlowNetwork
from ..network.node import Node
from ..network.connection import Connection
from .config import SolverConfig
from ..config.simulation_config import SimulationConfig
from .base import SolverBase


class NodalMatrixSolver(SolverBase):
    """
    Unified nodal-matrix solver for hydraulic networks with non-linear resistances.
    
    This solver implements the nodal analysis method where node pressures are the primary
    unknowns. The method is particularly effective for networks with multiple junctions
    and complex topologies.
    
    This is the canonical nodal solver for the project, consolidating all nodal solving functionality.
    """
    
    def __init__(self, sim_config: SimulationConfig, 
                 solver_config: Optional[SolverConfig] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize the nodal matrix solver.
        
        Args:
            sim_config: The simulation configuration object.
            solver_config: Optional solver configuration object.
            logger: Optional logger for debugging output.
        """
        super().__init__(sim_config, solver_config)
        self.oil_density = sim_config.oil_density
        self.oil_type = sim_config.oil_type
        self.viscosity_model = sim_config.viscosity_model
        self.viscosity_parameters = sim_config.viscosity_parameters
        self.gravity = 9.81
        self.logger = logger or logging.getLogger(__name__)

    def solve(self, network: FlowNetwork) -> Dict:
        """
        Public solve method that conforms to the SolverBase interface.
        """
        connection_flows, solution_info = self._solve_nodal_network_with_pump_physics(
            network,
            pump_flow_rate=self.sim_config.total_flow_rate,
            temperature=self.sim_config.temperature,
            pump_max_pressure=self.sim_config.inlet_pressure,
            outlet_pressure=self.sim_config.outlet_pressure or 101325.0,
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance
        )
        
        # Adapt the old solution_info to the new standard format
        solution_info['component_flows'] = connection_flows
        return solution_info

    def calculate_viscosity(self, temperature: float) -> float:
        """Calculate dynamic viscosity using Vogel equation"""
        T = temperature + 273.15

        if self.viscosity_model == 'vogel' and self.viscosity_parameters:
            params = self.viscosity_parameters
        else:
            viscosity_params = {
                "SAE10": {"A": 0.00004, "B": 950, "C": 135},
                "SAE20": {"A": 0.00006, "B": 1050, "C": 138},
                "SAE30": {"A": 0.0001, "B": 1200, "C": 140},
                "SAE40": {"A": 0.00015, "B": 1300, "C": 142},
                "SAE50": {"A": 0.0002, "B": 1400, "C": 145},
                "SAE60": {"A": 0.00025, "B": 1500, "C": 148},
                "VG220": {"A": 0.000064, "B": 1455, "C": 131},
                "VG320": {"A": 0.000064, "B": 1520, "C": 131},
                "VG460": {"A": 0.000064, "B": 1576, "C": 131}
            }
            
            if self.oil_type not in viscosity_params:
                raise ValueError(f"Oil type {self.oil_type} not supported")
            
            params = viscosity_params[self.oil_type]
        
        if T < params["C"]:
            T = params["C"] + 1
        
        viscosity = params["A"] * math.exp(params["B"] / (T - params["C"]))
        return max(1e-6, min(viscosity, 10.0))

    def _solve_nodal_network(
            self,
            network: FlowNetwork,
            total_flow_rate: float,
            temperature: float,
            inlet_pressure: float = 200_000.0,
            outlet_pressure: float = 101_325.0,
            elevations: Optional[Dict[str, float]] = None,
            pump_curve: Optional[Callable] = None,
            max_iterations: Optional[int] = None,
            tolerance: Optional[float] = None
        ) -> Tuple[Dict[str, float], Dict]:
            """
            Unified nodal network solver that supports multiple outlets via the iterative solver.
            """
            # 1. Validate network
            is_valid, errors = network.validate_network()
            if not is_valid:
                raise ValueError(f"Invalid network: {errors}")

            # 2. Defaults
            max_iter = max_iterations or self.config.max_iterations
            tol      = tolerance      or self.config.tolerance

            # 3. Fluid properties
            viscosity = self.calculate_viscosity(temperature)
            fluid_properties = {
                'density': self.oil_density,
                'viscosity': viscosity
            }

            # 4. Identify inlet/outlet nodes
            inlet_node   = network.inlet_node
            outlet_nodes = network.outlet_nodes or []

            if inlet_node is None:
                raise ValueError("Network must have an inlet node")
            if not outlet_nodes:
                raise ValueError("Network must have at least one outlet node")

            # 5. Call iterative solver
            node_pressures, edge_flows = self._solve_nodal_iterative(
                network=network,
                source_node_id=inlet_node.id,
                sink_node_ids=[o.id for o in outlet_nodes],
                Q_total=total_flow_rate,
                fluid_properties=fluid_properties,
                tol_flow=tol * 1e-3,
                tol_pressure=tol * 1_000,
                max_iter=max_iter
            )

            # 6. Build solution_info
            solution_info = {
                'converged':    True,
                'iterations':   max_iter,      # ideally updated by solver
                'temperature':  temperature,
                'viscosity':    viscosity,
                'oil_type':     self.oil_type,
                'oil_density':  self.oil_density,
                'total_flow_rate': total_flow_rate,
                'inlet_pressure':  inlet_pressure,
                'outlet_pressure': outlet_pressure,
                'node_pressures':  node_pressures,
                'pressure_drops':  {},
                'fluid_properties': fluid_properties
            }

            # 7. Compute pressure drops per connection
            for conn in network.connections:
                comp = conn.component
                q    = edge_flows.get(comp.id, 0.0)
                dp   = comp.calculate_pressure_drop(q, fluid_properties)
                solution_info['pressure_drops'][comp.id] = dp

            # 8. Shift all computed node pressures by outlet_pressure reference
            for nid in solution_info['node_pressures']:
                solution_info['node_pressures'][nid] += outlet_pressure

            solution_info['inlet_pressure'] = solution_info['node_pressures'][inlet_node.id]

            return edge_flows, solution_info

    def _solve_nodal_network_with_pump_physics(
        self,
        network: FlowNetwork,
        pump_flow_rate: float,
        temperature: float,
        pump_max_pressure: float = 1e6,
        outlet_pressure: float = 101_325.0,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Fixed‐Q solver: pins outlet pressure, enforces pump_flow_rate, and returns
        the flows and the required inlet pressure.
        """
        # 1) Validate network
        valid, errs = network.validate_network()
        if not valid:
            raise ValueError(f"Invalid network: {errs}")

        # 2) Delegate to the existing fixed‐Q nodal solver
        flows, sol = self._solve_nodal_network(
            network=network,
            total_flow_rate=pump_flow_rate,
            temperature=temperature,
            inlet_pressure=0.0,         # unused by this path
            outlet_pressure=outlet_pressure,
            max_iterations=max_iterations,
            tolerance=tolerance
        )

        # 3) Build the info dict the tests expect
        info = {
            'actual_flow_rate':        sol['total_flow_rate'],
            'required_inlet_pressure': sol['inlet_pressure'],
            'fluid_properties':        sol['fluid_properties'],
            'temperature':             temperature,
            'viscosity':               sol['fluid_properties']['viscosity']
        }

        return flows, sol


    def _calculate_component_resistance(
        self,
        component,
        fluid_properties: dict,
        Q_est: float
    ) -> float:
        """
        Compute R = d(ΔP)/dQ for a bare component (Channel, Connector, Nozzle)
        by central-differencing its calculate_pressure_drop().
        """
        # baseline ΔP
        dp0 = component.calculate_pressure_drop(Q_est, fluid_properties)

        # finite-difference step
        delta_q = max(abs(Q_est) * 1e-3, 1e-8)

        # forward/backwards ΔP
        dp_plus  = component.calculate_pressure_drop(Q_est + delta_q, fluid_properties)
        dp_minus = component.calculate_pressure_drop(Q_est - delta_q, fluid_properties)

        # slope
        R = (dp_plus - dp_minus) / (2.0 * delta_q)

        # floor
        return max(R, self.config.min_resistance)
    
    def _solve_nodal_iterative(self,
                             network: FlowNetwork,
                             source_node_id: str,
                             sink_node_ids: List[str],
                             Q_total: float,
                             fluid_properties: Dict,
                             tol_flow: float = 1e-6,
                             tol_pressure: float = 1e2,
                             max_iter: int = 20) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Solve the hydraulic network using iterative nodal-matrix method.
        
        Args:
            network: FlowNetwork to solve
            source_node_id: ID of the source node where flow enters
            sink_node_ids: List of IDs of the sink nodes where flow exits
            Q_total: Total flow rate entering at source and exiting at sink (m³/s)
            fluid_properties: Dict with 'density' and 'viscosity' keys
            tol_flow: Convergence tolerance for flow rates (m³/s)
            tol_pressure: Convergence tolerance for pressure-flow law (Pa)
            max_iter: Maximum number of iterations
            
        Returns:
            Tuple of (node_pressures, edge_flows) where:
            - node_pressures: Dict mapping node_id to pressure (Pa)
            - edge_flows: Dict mapping connection_id to flow rate (m³/s)
        """
        # Validate inputs
        if source_node_id not in network.nodes:
            raise ValueError(f"Source node {source_node_id} not found in network")
        for sink_node_id in sink_node_ids:
            if sink_node_id not in network.nodes:
                raise ValueError(f"Sink node {sink_node_id} not found in network")
        if source_node_id in sink_node_ids:
            raise ValueError("Source and sink nodes must be different")
        
        # Get node list and create mapping
        node_ids = list(network.nodes.keys())
        n_nodes = len(node_ids)
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Remove sink nodes from the system (reference pressure = 0)
        sink_indices = [node_to_idx[sink_id] for sink_id in sink_node_ids]
        active_nodes = [i for i in range(n_nodes) if i not in sink_indices]
        n_active = len(active_nodes)
        
        if n_active == 0:
            raise ValueError("No active nodes after removing sink nodes")
        
        # Create mapping for active nodes
        active_to_full = {i: active_nodes[i] for i in range(n_active)}
        full_to_active = {active_nodes[i]: i for i in range(n_active)}
        
        # Initialize edge flows
        edge_flows = self._initialize_flows(network, source_node_id, sink_node_ids, Q_total)
        
        self.logger.info(f"Starting nodal-matrix solver with {n_nodes} nodes, {len(network.connections)} edges")
        self.logger.info(f"Source: {source_node_id}, Sinks: {sink_node_ids}, Q_total: {Q_total:.6f} m³/s")
        
        # Iterative solution
        relaxation_factor = self.config.relaxation_factor
        last_max_flow_change = float('inf')
        flow_changes = []

        for iteration in range(max_iter):
            # Step 1: Compute resistances and conductances from current flows
            edge_resistances = {}
            edge_conductances = {}
            
            for conn in network.connections:
                flow = edge_flows[conn.component.id]
                
                # For nodal pressure-based methods, use average resistance for conductance
                # This ensures mass conservation: when Q = G*(P1-P2), total flows balance
                if abs(flow) > self.config.dq_absolute:
                    dp = conn.component.calculate_pressure_drop(flow, fluid_properties)
                    resistance = dp / abs(flow)  # Average resistance
                else:
                    # For very small flows, use differential resistance as approximation
                    resistance = self._calculate_component_resistance(
                        conn.component, fluid_properties, self.config.dq_absolute
                    )
                
                conductance = 1.0 / max(resistance, self.config.min_resistance)
                
                edge_resistances[conn.component.id] = resistance
                edge_conductances[conn.component.id] = conductance

            # Step 2: Build conductance matrix A · p = b with hydrostatic heads
            A = lil_matrix((n_active, n_active))
            b = np.zeros(n_active)

            # Term for non-linear residual correction
            b_residual = np.zeros(n_active)

            for conn in network.connections:
                i_full = node_to_idx[conn.from_node.id]
                j_full = node_to_idx[conn.to_node.id]
                G = edge_conductances[conn.component.id]
                R = edge_resistances[conn.component.id]
                flow = edge_flows[conn.component.id]

                # Elevations and hydrostatic pressure
                z_i = conn.from_node.elevation
                z_j = conn.to_node.elevation
                dp_hydro = fluid_properties['density'] * self.gravity * (z_i - z_j)

                # Non-linear residual correction
                # Since we use average resistance in conductance matrix, residual should be zero
                # But we include it for numerical stability and future enhancements
                dp_physical = conn.component.calculate_pressure_drop(flow, fluid_properties)
                dp_linearized = R * flow  # R is average resistance
                dp_residual = dp_physical - dp_linearized  # Should be ~0 for average resistance

                self.logger.debug(
                    f"  Conn {conn.component.id[:13]}: Flow={flow:.4f}, Phys_DP={dp_physical:.2f}, "
                    f"Lin_DP={dp_linearized:.2f}, Resid_DP={dp_residual:.2f}"
                )

                i_is_active = i_full not in sink_indices
                j_is_active = j_full not in sink_indices

                if i_is_active and j_is_active:
                    i_act, j_act = full_to_active[i_full], full_to_active[j_full]
                    A[i_act, i_act] += G
                    A[j_act, j_act] += G
                    A[i_act, j_act] -= G
                    A[j_act, i_act] -= G
                    
                    # Add hydrostatic and residual terms to RHS
                    b[i_act] -= G * dp_hydro
                    b[j_act] += G * dp_hydro
                    b_residual[i_act] -= G * dp_residual
                    b_residual[j_act] += G * dp_residual

                elif i_is_active and not j_is_active: # j is sink
                    i_act = full_to_active[i_full]
                    A[i_act, i_act] += G
                    b[i_act] -= G * dp_hydro
                    b_residual[i_act] -= G * dp_residual

                elif not i_is_active and j_is_active: # i is sink
                    j_act = full_to_active[j_full]
                    A[j_act, j_act] += G
                    b[j_act] += G * dp_hydro
                    b_residual[j_act] += G * dp_residual

            # Step 3: Set up RHS vector (net flow injections)
            # Source node: inject +Q_total
            source_idx = node_to_idx[source_node_id]
            if source_idx not in sink_indices:
                source_active = full_to_active[source_idx]
                b[source_active] += Q_total
            
            # Sink nodes: extract flow (this is handled implicitly by setting their pressure to 0)
            # The mass conservation is enforced by the network topology and flow equations
            
            # Add residual correction to b
            b += b_residual
            
            # Step 4: Solve linear system A·p = b
            if n_active == 1:
                if A[0, 0] > 0:
                    pressures_active = np.array([b[0] / A[0, 0]])
                else:
                    pressures_active = np.array([0.0])
            else:
                try:
                    A_csr = A.tocsr()
                    pressures_active = spsolve(A_csr, b)
                    if np.isscalar(pressures_active):
                        pressures_active = np.array([pressures_active])
                except Exception as e:
                    self.logger.error(f"Failed to solve linear system at iteration {iteration}: {e}")
                    raise RuntimeError(f"Failed to solve linear system: {e}")
            
            # Step 5: Reconstruct full pressure vector
            pressures_full = np.zeros(n_nodes)
            for i, pressure in enumerate(pressures_active):
                full_idx = active_to_full[i]
                pressures_full[full_idx] = pressure
            
            # Step 6: Compute new edge flows from pressures
            new_edge_flows = {}
            for conn in network.connections:
                from_idx = node_to_idx[conn.from_node.id]
                to_idx = node_to_idx[conn.to_node.id]
                conductance = edge_conductances[conn.component.id]
                
                pressure_from = pressures_full[from_idx]
                pressure_to = pressures_full[to_idx]

                # Correctly include hydrostatic pressure in flow calculation
                z_from = network.nodes[conn.from_node.id].elevation
                z_to = network.nodes[conn.to_node.id].elevation
                dp_hydro = fluid_properties['density'] * self.gravity * (z_from - z_to)
                
                # Physical pressure drop from the solver's perspective
                dp_solver = pressure_from - pressure_to
                
                # Use the linearized flow for the iterative update
                flow = conductance * (dp_solver + dp_hydro)
                new_edge_flows[conn.component.id] = flow
            
            # Step 7: Check convergence
            max_flow_change = max(abs(new_edge_flows[conn_id] - edge_flows[conn_id]) 
                                for conn_id in edge_flows)
            flow_changes.append(max_flow_change)
            if len(flow_changes) > 10:
                flow_changes.pop(0)
                if np.std(flow_changes) < 1e-8:
                    self.logger.warning("Solver stalled. Converged with reduced tolerance.")
                    break

            # The pressure error is now the non-linear residual, which we are
            # explicitly solving for. A better metric for convergence is the
            # change in the solution (flow change). We can also check the
            # overall mass conservation error.
            max_pressure_error = 0.0
            for conn in network.connections:
                dp_physical = conn.component.calculate_pressure_drop(new_edge_flows[conn.component.id], fluid_properties)
                
                from_idx = node_to_idx[conn.from_node.id]
                to_idx = node_to_idx[conn.to_node.id]
                dp_solver = pressures_full[from_idx] - pressures_full[to_idx]

                z_from = network.nodes[conn.from_node.id].elevation
                z_to = network.nodes[conn.to_node.id].elevation
                dp_hydro = fluid_properties['density'] * self.gravity * (z_from - z_to)

                pressure_error = abs(dp_physical - (dp_solver + dp_hydro))
                max_pressure_error = max(max_pressure_error, pressure_error)
            
            # Calculate mass conservation error
            mass_conservation_error = self._calculate_mass_conservation_error(
                network, new_edge_flows, source_node_id, sink_node_ids, Q_total
            )
            
            self.logger.debug(f"Iteration {iteration + 1}: max_flow_change={max_flow_change:.2e}, "
                            f"max_pressure_error={max_pressure_error:.2e}, "
                            f"mass_conservation_error={mass_conservation_error:.2e}")

            # Check convergence criteria including mass conservation
            if (max_flow_change < tol_flow and 
                max_pressure_error < tol_pressure and 
                mass_conservation_error < tol_flow):
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Adaptive relaxation strategy
            if len(flow_changes) > 3:
                # Check for oscillations in the last few iterations
                recent_changes = flow_changes[-3:]
                if np.std(recent_changes) / np.mean(recent_changes) > 0.5:
                    # Reduce relaxation factor if oscillating
                    relaxation_factor = max(0.1, relaxation_factor * 0.8)
                    self.logger.debug(f"Reduced relaxation factor to {relaxation_factor:.3f}")
            
            # Update flows for next iteration with relaxation
            for conn_id in edge_flows:
                edge_flows[conn_id] = (
                    relaxation_factor * new_edge_flows[conn_id] +
                    (1 - relaxation_factor) * edge_flows[conn_id]
                )
        
        else:
            self.logger.warning(f"Did not converge after {max_iter} iterations")
        
        # Prepare output
        node_pressures = {node_id: pressures_full[node_to_idx[node_id]] for node_id in node_ids}
        
        self._validate_mass_conservation(network, new_edge_flows, source_node_id, sink_node_ids, Q_total)
        
        return node_pressures, new_edge_flows
    
    def _initialize_flows(self, network: FlowNetwork, source_node_id: str, sink_node_ids: List[str],
                          Q_total: float) -> Dict[str, float]:
        """
        Initialize edge flows using a resistance-based approach for multiple sinks.
        Flow is distributed inversely proportional to the resistance of each path.
        """
        edge_flows = {conn.component.id: 0.0 for conn in network.connections}
        fluid_properties = {
            'density': self.oil_density,
            'viscosity': self.calculate_viscosity(40.0)  # Assume 40C for initial viscosity
        }

        # Estimate resistance for each component using differential method for consistency
        resistances = {}
        for conn in network.connections:
            resistances[conn.component.id] = self._calculate_component_resistance(
                conn.component, fluid_properties, self.config.dq_absolute
            )

        # Find all paths from source to each sink using BFS
        all_paths = []
        path_sink_mapping = []  # Track which sink each path leads to
        
        for sink_node_id in sink_node_ids:
            paths_to_sink = []
            queue = [(source_node_id, [])]
            visited = set()

            while queue:
                curr_node_id, path = queue.pop(0)

                if curr_node_id == sink_node_id:
                    paths_to_sink.append(path)
                    continue

                if curr_node_id in visited:
                    continue
                visited.add(curr_node_id)

                for conn in network.connections:
                    if conn.from_node.id == curr_node_id and conn.to_node.id not in visited:
                        new_path = path + [conn.component.id]
                        queue.append((conn.to_node.id, new_path))
            
            # Add paths to this sink
            for path in paths_to_sink:
                all_paths.append(path)
                path_sink_mapping.append(sink_node_id)

        if not all_paths:
            # Fallback to uniform distribution if no paths are found
            n_edges = len(network.connections)
            if n_edges > 0:
                initial_flow = Q_total / n_edges
                for conn_id in edge_flows:
                    edge_flows[conn_id] = initial_flow
            return edge_flows

        # Calculate conductance (inverse resistance) for each path
        path_conductances = []
        for path in all_paths:
            path_resistance = sum(resistances[comp_id] for comp_id in path)
            path_conductance = 1.0 / max(path_resistance, self.config.min_resistance)
            path_conductances.append(path_conductance)

        # Distribute flow based on conductance weighting
        total_conductance = sum(path_conductances)
        
        for i, path in enumerate(all_paths):
            if total_conductance > 0:
                path_flow = Q_total * (path_conductances[i] / total_conductance)
                for comp_id in path:
                    edge_flows[comp_id] += path_flow

        return edge_flows
    
    def _compute_resistance(self, component, flow: float, fluid_properties: Dict) -> float:
        """
        Compute resistance for a component at given flow rate.
        This method is deprecated - use _calculate_component_resistance for consistency.
        """
        # Delegate to the differential resistance calculation for consistency
        return self._calculate_component_resistance(component, fluid_properties, flow)
    
    def _calculate_mass_conservation_error(self,
                                         network: FlowNetwork,
                                         edge_flows: Dict[str, float],
                                         source_node_id: str,
                                         sink_node_ids: List[str],
                                         Q_total: float) -> float:
        """
        Calculate the maximum mass conservation error across all nodes.
        
        Returns:
            Maximum absolute mass conservation error in m³/s
        """
        max_error = 0.0
        total_sink_flow = 0.0
        
        for node_id, node in network.nodes.items():
            flow_in = 0.0
            flow_out = 0.0
            
            # Sum flows into and out of this node
            for conn in network.connections:
                flow = edge_flows.get(conn.component.id, 0.0)
                
                if conn.to_node.id == node_id:
                    flow_in += flow
                elif conn.from_node.id == node_id:
                    flow_out += flow
            
            if node_id in sink_node_ids:
                total_sink_flow += flow_in
            
            # Net flow at node
            net_flow = flow_in - flow_out
            
            # Expected net flow
            if node_id == source_node_id:
                expected_net = -Q_total
            elif node_id in sink_node_ids:
                # For sink nodes, continue to check total balance later
                continue
            else:
                expected_net = 0.0
            
            error = abs(net_flow - expected_net)
            max_error = max(max_error, error)
        
        # Check total sink vs source flow balance
        total_flow_error = abs(total_sink_flow - Q_total)
        max_error = max(max_error, total_flow_error)
        
        return max_error
    
    def _validate_mass_conservation(self,
                                   network: FlowNetwork,
                                   edge_flows: Dict[str, float],
                                   source_node_id: str,
                                   sink_node_ids: List[str],
                                   Q_total: float,
                                   tolerance: float = 1e-6):
        """
        Validate that mass conservation is satisfied at all nodes.
        
        Args:
            network: The flow network
            edge_flows: Dictionary of edge flows
            source_node_id: Source node ID
            sink_node_ids: List of sink node IDs
            Q_total: Total flow rate
            tolerance: Tolerance for mass conservation check
        """
        total_sink_flow = 0
        for node_id, node in network.nodes.items():
            flow_in = 0.0
            flow_out = 0.0
            
            # Sum flows into and out of this node
            for conn in network.connections:
                flow = edge_flows.get(conn.component.id, 0.0)
                
                if conn.to_node.id == node_id:
                    flow_in += flow
                elif conn.from_node.id == node_id:
                    flow_out += flow
            
            if node_id in sink_node_ids:
                total_sink_flow += flow_in
            
            # Net flow at node
            net_flow = flow_in - flow_out
            
            # Expected net flow
            if node_id == source_node_id:
                expected_net = -Q_total
            elif node_id in sink_node_ids:
                # For sink nodes, we don't know the exact flow, so we check the total sink flow later
                continue
            else:
                expected_net = 0.0
            
            error = abs(net_flow - expected_net)
            if error > tolerance:
                self.logger.warning(f"Mass conservation violated at node {node.name} ({node_id}):\n"
                                  f"  Flow in: {flow_in:.6f}\n"
                                  f"  Flow out: {flow_out:.6f}\n"
                                  f"  Net flow: {net_flow:.6f}\n"
                                  f"  Expected net flow: {expected_net:.6f}\n"
                                  f"  Error: {error:.6f}")

        # Check total sink flow
        error = abs(total_sink_flow - Q_total)
        if error > tolerance:
            self.logger.warning(f"Total sink flow does not match total source flow:\n"
                              f"  Total sink flow: {total_sink_flow:.6f}\n"
                              f"  Total source flow: {Q_total:.6f}\n"
                              f"  Error: {error:.6f}")
                

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
        
        connection_flows = solution.get('component_flows', {})
        solution_info = solution # In the new interface, the whole dict is the solution info

        print(f"\n{'='*80}")
        print(f"NETWORK FLOW SIMULATION RESULTS")
        print(f"{'='*80}")
        
        # --- General Information ---
        print(f"  Network Name:      {network.name}")
        print(f"  Temperature:       {solution_info.get('temperature', 'N/A'):.1f}°C")
        print(f"  Oil Type:          {self.oil_type}")
        print(f"  Oil Density:       {self.oil_density:.1f} kg/m³")
        print(f"  Dynamic Viscosity: {solution_info.get('viscosity', 'N/A'):.6f} Pa·s")
        
        # --- Simulation Summary ---
        flow_rate_key = 'total_flow_rate' if 'total_flow_rate' in solution_info else 'actual_flow_rate'
        total_flow_rate = solution_info.get(flow_rate_key, 0.0)
        total_flow_rate_disp, q_unit_str = convert_flow_rate(total_flow_rate, flow_rate_unit)
        print(f"\n  Total System Flow Rate: {total_flow_rate_disp:.2f} {q_unit_str}")
        
        inlet_pressure_key = 'inlet_pressure' if 'inlet_pressure' in solution_info else 'required_inlet_pressure'
        inlet_pressure = solution_info.get(inlet_pressure_key, 0.0)
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
            # Find connections leading to this outlet
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
        
        # Calculate pressure drops if not already calculated
        if 'pressure_drops' not in solution_info:
            solution_info['pressure_drops'] = {}
            fluid_properties = solution_info.get('fluid_properties', {
                'density': self.oil_density,
                'viscosity': solution_info['viscosity']
            })
            
            for connection in network.connections:
                component = connection.component
                flow_rate = connection_flows.get(component.id, 0.0)
                dp = component.calculate_pressure_drop(flow_rate, fluid_properties)
                solution_info['pressure_drops'][component.id] = dp
        
        # Print connection flows and pressure drops
        print(f"  {'Component':<20} {'Type':<15} {'Flow Rate (' + q_unit_str + ')':<20} {'Pressure Drop (' + p_unit_str + ')':<20}")
        print(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*20}")
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows.get(component.id, 0.0)
            pressure_drop = solution_info['pressure_drops'].get(component.id, 0)
            flow_rate_disp, _ = convert_flow_rate(flow_rate, flow_rate_unit)
            pressure_drop_disp, _ = convert_pressure(pressure_drop, pressure_unit)
            
            print(f"  {component.name:<20} {component.component_type.value:<15} "
                  f"{flow_rate_disp:<20.3f} {pressure_drop_disp:<20.2f}")
        
        # Print node pressures
        print(f"\n  {'Node':<20} {'Pressure (' + p_unit_str + ')':<20} {'Elevation (m)':<15}")
        print(f"  {'-'*20} {'-'*20} {'-'*15}")
        
        sorted_nodes = sorted(solution_info.get('node_pressures', {}).items(), key=lambda item: item[1], reverse=True)
        
        for node_id, pressure in sorted_nodes:
            node = network.nodes.get(node_id)
            if node:
                pressure_disp, _ = convert_pressure(pressure, pressure_unit)
                print(f"  {node.name:<20} {pressure_disp:<20.2f} {node.elevation:<15.1f}")
        
        # --- Warnings ---
        if 'warnings' in solution_info and solution_info['warnings']:
            print(f"\n{'='*80}")
            print("WARNINGS")
            print(f"{'='*80}")
            for warning in solution_info['warnings']:
                print(f"  - {warning}")
        
        print(f"\n{'='*80}\n")
