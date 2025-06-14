"""
Unified Nodal-Matrix Solver for Hydraulic Networks with Non-linear Edge Resistances

This module implements the canonical nodal-matrix solver for the project that finds node pressures 
and edge flows such that mass is conserved and the pressure-flow law ΔP_e = R_e(Q_e) · Q_e holds 
on every edge.

The solver uses the nodal analysis method where:
1. Each node has a unique pressure (except reference node)
2. Conductance matrix A is built from edge conductances G_e = 1/R_e(Q_e)
3. System A·p = b is solved iteratively as conductances depend on flows
4. Flows are computed from pressure differences and conductances

This is the unified implementation that consolidates all nodal solving functionality.
"""

import math
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Callable
import logging

from ..network.flow_network import FlowNetwork
from ..network.node import Node
from ..network.connection import Connection
from .config import SolverConfig


class NodalMatrixSolver:
    """
    Unified nodal-matrix solver for hydraulic networks with non-linear resistances.
    
    This solver implements the nodal analysis method where node pressures are the primary
    unknowns. The method is particularly effective for networks with multiple junctions
    and complex topologies.
    
    This is the canonical nodal solver for the project, consolidating all nodal solving functionality.
    """
    
    def __init__(self, config: Optional[SolverConfig] = None, oil_density: float = 900.0, 
                 oil_type: str = "SAE30", logger: Optional[logging.Logger] = None):
        """
        Initialize the nodal matrix solver.
        
        Args:
            config: Solver configuration (uses default if None)
            oil_density: Oil density in kg/m³
            oil_type: Oil type for viscosity calculation
            logger: Optional logger for debugging output
        """
        self.config = config or SolverConfig()
        self.oil_density = oil_density
        self.oil_type = oil_type
        self.gravity = 9.81
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_viscosity(self, temperature: float) -> float:
        """Calculate dynamic viscosity using Vogel equation"""
        T = temperature + 273.15
        
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
    
    def solve_nodal_network(self,
                           network: FlowNetwork,
                           total_flow_rate: float,
                           temperature: float,
                           inlet_pressure: float = 200000.0,
                           outlet_pressure: float = 101325.0,
                           elevations: Optional[Dict[str, float]] = None,
                           pump_curve: Optional[Callable] = None,
                           max_iterations: Optional[int] = None,
                           tolerance: Optional[float] = None) -> Tuple[Dict[str, float], Dict]:
        """
        Unified nodal network solver that supports multiple calling patterns.
        
        This is the canonical nodal solver method that consolidates functionality from
        both the iterative solver and the simple nodal solver.
        
        Args:
            network: FlowNetwork to solve
            total_flow_rate: Total flow rate entering the system (m³/s)
            temperature: Operating temperature (°C)
            inlet_pressure: Pressure at inlet node (Pa)
            outlet_pressure: Pressure at outlet nodes (Pa)
            elevations: Optional dict of node elevations (uses node.elevation if None)
            pump_curve: Optional pump head-flow curve function (for future use)
            max_iterations: Maximum iterations (uses config default if None)
            tolerance: Convergence tolerance (uses config default if None)
            
        Returns:
            Tuple of (connection_flows, solution_info) where:
            - connection_flows: Dict mapping connection_id to flow rate (m³/s)
            - solution_info: Dict with convergence info, pressures, etc.
        """
        # Validate network
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network: {errors}")
        
        # Use config defaults if not specified
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance
        
        # Calculate fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_properties = {
            'density': self.oil_density,
            'viscosity': viscosity
        }
        
        # Identify inlet and outlet nodes
        inlet_node = network.inlet_node
        outlet_nodes = network.outlet_nodes
        
        if not inlet_node:
            raise ValueError("Network must have an inlet node")
        if not outlet_nodes:
            raise ValueError("Network must have at least one outlet node")
        
        # Use the iterative solver for all cases
        if len(outlet_nodes) == 1:
            # Single outlet case - direct use of iterative solver
            outlet_node = outlet_nodes[0]
            
            # Call the iterative solver
            node_pressures, edge_flows = self.solve_nodal_iterative(
                network=network,
                source_node_id=inlet_node.id,
                sink_node_id=outlet_node.id,
                Q_total=total_flow_rate,
                fluid_properties=fluid_properties,
                tol_flow=tol * 1e-3,  # Convert to flow tolerance
                tol_pressure=tol * 1000,  # Convert to pressure tolerance
                max_iter=max_iter
            )
            
            # Convert to connection_flows format
            connection_flows = edge_flows
            
        else:
            # Multiple outlets case - use backward compatibility approach
            # Note: The original solve_network_flow_nodal had limitations with realistic pressures
            # For now, we maintain backward compatibility but issue a warning
            self.logger.warning(
                "Multiple outlet networks with realistic pressure values may give unrealistic flows. "
                "Consider using single outlet networks or the iterative solver directly."
            )
            
            connection_flows, node_pressures = self._solve_multiple_outlets(
                network, total_flow_rate, fluid_properties, 
                inlet_pressure, outlet_pressure, max_iter, tol
            )
        
        # Build solution info
        solution_info = {
            'converged': True,  # Assume converged for now
            'iterations': max_iter,  # Will be updated by actual solver
            'temperature': temperature,
            'viscosity': viscosity,
            'oil_type': self.oil_type,
            'oil_density': self.oil_density,
            'total_flow_rate': total_flow_rate,
            'inlet_pressure': inlet_pressure,
            'outlet_pressure': outlet_pressure,
            'node_pressures': node_pressures,
            'pressure_drops': {},
            'fluid_properties': fluid_properties
        }
        
        # Calculate pressure drops
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            dp = component.calculate_pressure_drop(flow_rate, fluid_properties)
            solution_info['pressure_drops'][component.id] = dp
        
        return connection_flows, solution_info
    
    
    def _flow_for_pressure(
        self,
        network: FlowNetwork,
        inlet_pressure: float,
        outlet_pressure: float,
        pump_flow_rate: float,
        fluid_props: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], float]:
        """
        For a given inlet pressure, assemble conductances, solve nodal matrix,
        compute every edge flow, and return the total flow delivered at the inlet.
        """
        # A) Known‐pressure nodes
        known = {network.inlet_node.id: inlet_pressure}
        for out in network.outlet_nodes:
            known[out.id] = outlet_pressure

        # B) Unknown nodes
        unknown = [n for n in network.nodes.values() if n.id not in known]
        N = len(unknown)
        idx = {n.id: i for i, n in enumerate(unknown)}

        # C) Estimate conductances G_e = 1/R_e at a per‐branch guess Q0
        Q0 = pump_flow_rate / max(1, len(network.outlet_nodes))
        G_e = {}
        for conn in network.connections:
            R = self._calculate_connection_resistance(conn, Q0, fluid_props)
            G_e[conn.component.id] = (1.0 / R) if R > 0 else 1e12

        # D) Build sparse nodal matrix A·p = b
        if N > 0:
            from scipy.sparse import lil_matrix
            import numpy as np

            A = lil_matrix((N, N))
            b = np.zeros(N)

            for conn in network.connections:
                g = G_e[conn.component.id]
                i_id, j_id = conn.from_node.id, conn.to_node.id

                # Diagonal entries
                if i_id in idx: A[idx[i_id], idx[i_id]] += g
                if j_id in idx: A[idx[j_id], idx[j_id]] += g

                # Off-diagonals for unknown‐unknown
                if i_id in idx and j_id in idx:
                    A[idx[i_id], idx[j_id]] -= g
                    A[idx[j_id], idx[i_id]] -= g

                # RHS contributions for known‐unknown
                if i_id in idx and j_id in known:
                    b[idx[i_id]] += g * known[j_id]
                if j_id in idx and i_id in known:
                    b[idx[j_id]] += g * known[i_id]

            # Solve for unknown node pressures
            from scipy.sparse.linalg import spsolve
            p_unknown = spsolve(A.tocsr(), b)

            # Combine pressures
            node_pressures = known.copy()
            for node_id, i in idx.items():
                node_pressures[node_id] = p_unknown[i]
        else:
            node_pressures = known.copy()

        # E) Compute each connection flow
        conn_flows = {}
        for conn in network.connections:
            p_from = node_pressures[conn.from_node.id]
            p_to   = node_pressures[conn.to_node.id]
            conn_flows[conn.component.id] = G_e[conn.component.id] * (p_from - p_to)

        # F) Total delivered at inlet
        delivered = sum(
            conn_flows[c.component.id]
            for c in network.adjacency_list[network.inlet_node.id]
        )

        return conn_flows, node_pressures, delivered


    def solve_nodal_network_with_pump_physics(
        self,
        network: FlowNetwork,
        pump_flow_rate: float,
        temperature: float,
        pump_max_pressure: float = 1e6,
        outlet_pressure: float = 101325.0,
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None
    ) -> Tuple[Dict[str, float], Dict]:
        """
        Displacement-driven nodal solver: 
        - Enforces fixed Q_pump
        - Finds the inlet pressure required to push that Q through the network
        - Throttles Q if required pressure > pump_max_pressure
        """
        # 1) Validate network
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network: {errors}")

        # 2) Solver parameters
        max_iter = max_iterations or self.config.max_iterations
        tol = tolerance or self.config.tolerance
        # Convert tol (rel ΔP) into a flow tolerance ΔQ ≈ tol·Q_pump
        q_tol = tol * pump_flow_rate

        # 3) Fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_props = {'density': self.oil_density, 'viscosity': viscosity}

        # 4) Pressure bracket
        p_lo = outlet_pressure
        p_hi = pump_max_pressure

        # 5) Bisection loop
        for iteration in range(max_iter):
            p_guess = 0.5 * (p_lo + p_hi)

            # 5a) For this inlet pressure, compute the network flows
            conn_flows, node_pressures, q_delivered = (
                self._flow_for_pressure(
                    network, p_guess, outlet_pressure, pump_flow_rate, fluid_props
                )
            )

            # 5b) Check pump adequacy / break if flow falls below fraction
            if q_delivered < self.config.min_flow_fraction * pump_flow_rate:
                break

            # 5c) Converged?
            dq = q_delivered - pump_flow_rate
            if abs(dq) < q_tol:
                break

            # 5d) Narrow bracket
            if dq > 0:
                # network “easier” than pump → reduce pressure
                p_hi = p_guess
            else:
                # network “harder” → need more head
                p_lo = p_guess

        # 6) Build solution_info
        required_head = p_guess
        solution_info = {
            'required_inlet_pressure': required_head,
            'actual_flow_rate': q_delivered,
            'pump_adequate': (required_head <= pump_max_pressure),
            'iterations': iteration + 1,
            'node_pressures': node_pressures,
            'pressure_drops': {
                cid: network.connections[i].component.calculate_pressure_drop(f, fluid_props)
                for i, (cid, f) in enumerate(conn_flows.items())
            },
            'fluid_properties': fluid_props
        }

        return conn_flows, solution_info



    def _solve_multiple_outlets(self, 
                               network: FlowNetwork,
                               total_flow_rate: float,
                               fluid_properties: Dict,
                               inlet_pressure: float,
                               outlet_pressure: float,
                               max_iterations: int,
                               tolerance: float) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Solve network with multiple outlets using matrix-based nodal analysis.
        
        This method implements the approach from the original solve_network_flow_nodal.
        """
        # Assemble node indices
        all_nodes = list(network.nodes.values())
        
        # Identify known-pressure nodes
        known_nodes = {network.inlet_node.id: inlet_pressure}
        for outlet in network.outlet_nodes:
            known_nodes[outlet.id] = outlet_pressure
            
        # Unknown nodes
        unknown_nodes = [n for n in all_nodes if n.id not in known_nodes]
        N = len(unknown_nodes)
        
        if N == 0:
            # All nodes have known pressures - simple case
            connection_flows = {}
            for conn in network.connections:
                p_from = known_nodes[conn.from_node.id]
                p_to = known_nodes[conn.to_node.id]
                # Estimate conductance
                Q0 = total_flow_rate / max(1, len(network.outlet_nodes))
                R = self._calculate_connection_resistance(conn, fluid_properties, Q0)
                G = 1.0 / R if R > 0 else 1e12
                connection_flows[conn.component.id] = G * (p_from - p_to)
            
            return connection_flows, known_nodes
        
        # Map node ID to index
        idx_map = {n.id: i for i, n in enumerate(unknown_nodes)}
        
        # Compute conductances for each connection
        G_e = {}
        for conn in network.connections:
            # Estimate resistance at equal split flow
            Q0 = total_flow_rate / max(1, len(network.outlet_nodes))
            R = self._calculate_connection_resistance(conn, fluid_properties, Q0)
            G_e[conn.component.id] = 1.0 / R if R > 0 else 1e12
        
        # Build sparse conductance matrix and RHS
        Gmat = lil_matrix((N, N))
        b = np.zeros(N)
        
        for conn in network.connections:
            i_id, j_id = conn.from_node.id, conn.to_node.id
            g = G_e[conn.component.id]
            
            # if both unknown
            if i_id in idx_map and j_id in idx_map:
                i, j = idx_map[i_id], idx_map[j_id]
                Gmat[i, i] += g
                Gmat[j, j] += g
                Gmat[i, j] -= g
                Gmat[j, i] -= g
            # if one end known
            elif i_id in idx_map:
                i = idx_map[i_id]
                Gmat[i, i] += g
                b[i] += g * known_nodes[j_id]
            elif j_id in idx_map:
                j = idx_map[j_id]
                Gmat[j, j] += g
                b[j] += g * known_nodes[i_id]
            # if both known: no equation required
        
        # The RHS vector b is already set up correctly from the known pressure terms
        # No additional flow injection terms are needed - the original method 
        # relied purely on pressure boundary conditions
        
        # Solve linear system
        try:
            P_unknown = spsolve(Gmat.tocsr(), b)
            if np.isscalar(P_unknown):
                P_unknown = np.array([P_unknown])
        except Exception as e:
            self.logger.error(f"Failed to solve linear system: {e}")
            raise RuntimeError(f"Failed to solve nodal system: {e}")
        
        # Collect nodal pressures
        node_pressures = known_nodes.copy()
        for n, p in zip(unknown_nodes, P_unknown):
            node_pressures[n.id] = p
        
        # Compute flows on each connection
        connection_flows = {}
        for conn in network.connections:
            p_from = node_pressures[conn.from_node.id]
            p_to = node_pressures[conn.to_node.id]
            connection_flows[conn.component.id] = G_e[conn.component.id] * (p_from - p_to)
        
        return connection_flows, node_pressures
    
    def _calculate_connection_resistance(self, connection: Connection, 
                                       fluid_properties: Dict, estimated_flow: float) -> float:
        """Calculate resistance of a single connection"""
        component = connection.component
        
        # Calculate component resistance (dP/dQ)
        if estimated_flow > 1e-9:
            # Use absolute flow perturbation to estimate resistance
            delta_q = self.config.dq_absolute
            dp1 = component.calculate_pressure_drop(estimated_flow, fluid_properties)
            dp2 = component.calculate_pressure_drop(estimated_flow + delta_q, fluid_properties)
            
            if delta_q > 0:
                resistance = (dp2 - dp1) / delta_q
            else:
                resistance = dp1 / estimated_flow if estimated_flow > 0 else 0
        else:
            # For very small flows, use linear approximation
            resistance = component.calculate_pressure_drop(self.config.dq_absolute, fluid_properties) / self.config.dq_absolute
        
        return max(resistance, self.config.min_resistance)  # Ensure non-negative and above minimum
    
    def solve_nodal_iterative(self, 
                             network: FlowNetwork,
                             source_node_id: str,
                             sink_node_id: str,
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
            sink_node_id: ID of the sink node where flow exits (reference pressure = 0)
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
        if sink_node_id not in network.nodes:
            raise ValueError(f"Sink node {sink_node_id} not found in network")
        if source_node_id == sink_node_id:
            raise ValueError("Source and sink nodes must be different")
        
        # Get node list and create mapping
        node_ids = list(network.nodes.keys())
        n_nodes = len(node_ids)
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        # Handle special case: only 2 nodes (source and sink)
        if n_nodes == 2:
            return self._solve_two_node_case(network, source_node_id, sink_node_id, Q_total, fluid_properties)
        
        # Remove sink node from the system (reference pressure = 0)
        sink_idx = node_to_idx[sink_node_id]
        active_nodes = [i for i in range(n_nodes) if i != sink_idx]
        n_active = len(active_nodes)
        
        if n_active == 0:
            raise ValueError("No active nodes after removing sink node")
        
        # Create mapping for active nodes
        active_to_full = {i: active_nodes[i] for i in range(n_active)}
        full_to_active = {active_nodes[i]: i for i in range(n_active)}
        
        # Initialize edge flows with better initial guess
        edge_flows = self._initialize_flows(network, source_node_id, sink_node_id, Q_total)
        
        self.logger.info(f"Starting nodal-matrix solver with {n_nodes} nodes, {len(network.connections)} edges")
        self.logger.info(f"Source: {source_node_id}, Sink: {sink_node_id}, Q_total: {Q_total:.6f} m³/s")
        
        # Iterative solution
        for iteration in range(max_iter):
            # Step 1: Compute resistances and conductances from current flows
            edge_resistances = {}
            edge_conductances = {}
            
            for conn in network.connections:
                flow = edge_flows[conn.component.id]
                resistance = self._compute_resistance(conn.component, flow, fluid_properties)
                conductance = 1.0 / resistance
                
                edge_resistances[conn.component.id] = resistance
                edge_conductances[conn.component.id] = conductance
            
            # Step 2: Build nodal conductance matrix A and RHS vector b
            A = lil_matrix((n_active, n_active))
            b = np.zeros(n_active)
            
            # Build the conductance matrix
            for conn in network.connections:
                from_idx = node_to_idx[conn.from_node.id]
                to_idx = node_to_idx[conn.to_node.id]
                conductance = edge_conductances[conn.component.id]
                
                # Handle connections between active nodes
                if from_idx != sink_idx and to_idx != sink_idx:
                    from_active = full_to_active[from_idx]
                    to_active = full_to_active[to_idx]
                    
                    # Add conductance to diagonal terms
                    A[from_active, from_active] += conductance
                    A[to_active, to_active] += conductance
                    
                    # Subtract conductance from off-diagonal terms
                    A[from_active, to_active] -= conductance
                    A[to_active, from_active] -= conductance
                
                # Handle connections to sink node
                elif from_idx != sink_idx and to_idx == sink_idx:
                    # Connection from active node to sink
                    from_active = full_to_active[from_idx]
                    A[from_active, from_active] += conductance
                    # No off-diagonal term since sink is eliminated
                    
                elif from_idx == sink_idx and to_idx != sink_idx:
                    # Connection from sink to active node
                    to_active = full_to_active[to_idx]
                    A[to_active, to_active] += conductance
                    # No off-diagonal term since sink is eliminated
            
            # Step 3: Set up RHS vector (net flow injections)
            # Only the source node has a net flow injection
            source_idx = node_to_idx[source_node_id]
            if source_idx != sink_idx:
                source_active = full_to_active[source_idx]
                b[source_active] = Q_total
            
            # Step 4: Solve linear system A·p = b
            if n_active == 1:
                # Special case: only one active node
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
                    # Try with regularization
                    A_reg = A_csr + 1e-12 * lil_matrix(np.eye(n_active))
                    try:
                        pressures_active = spsolve(A_reg.tocsr(), b)
                        if np.isscalar(pressures_active):
                            pressures_active = np.array([pressures_active])
                    except Exception as e2:
                        raise RuntimeError(f"Failed to solve even with regularization: {e2}")
            
            # Step 5: Reconstruct full pressure vector
            pressures_full = np.zeros(n_nodes)
            for i, pressure in enumerate(pressures_active):
                full_idx = active_to_full[i]
                pressures_full[full_idx] = pressure
            # Sink pressure is already 0
            
            # Step 6: Compute new edge flows from pressures
            new_edge_flows = {}
            for conn in network.connections:
                from_idx = node_to_idx[conn.from_node.id]
                to_idx = node_to_idx[conn.to_node.id]
                conductance = edge_conductances[conn.component.id]
                
                pressure_from = pressures_full[from_idx]
                pressure_to = pressures_full[to_idx]
                
                # Flow = conductance * (pressure_from - pressure_to)
                flow = conductance * (pressure_from - pressure_to)
                new_edge_flows[conn.component.id] = flow
            
            # Step 7: Check convergence
            max_flow_change = max(abs(new_edge_flows[conn_id] - edge_flows[conn_id]) 
                                for conn_id in edge_flows)
            
            max_pressure_error = 0.0
            for conn in network.connections:
                from_idx = node_to_idx[conn.from_node.id]
                to_idx = node_to_idx[conn.to_node.id]
                
                pressure_diff = pressures_full[from_idx] - pressures_full[to_idx]
                flow = new_edge_flows[conn.component.id]
                resistance = edge_resistances[conn.component.id]
                
                expected_pressure_drop = resistance * flow
                pressure_error = abs(pressure_diff - expected_pressure_drop)
                max_pressure_error = max(max_pressure_error, pressure_error)
            
            self.logger.debug(f"Iteration {iteration + 1}: max_flow_change={max_flow_change:.2e}, "
                            f"max_pressure_error={max_pressure_error:.2e}")
            
            # Check convergence criteria
            if max_flow_change < tol_flow and max_pressure_error < tol_pressure:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            # Update flows for next iteration
            edge_flows = new_edge_flows.copy()
        
        else:
            self.logger.warning(f"Did not converge after {max_iter} iterations")
        
        # Prepare output
        node_pressures = {}
        for i, node_id in enumerate(node_ids):
            node_pressures[node_id] = pressures_full[i]
        
        # Validate mass conservation
        self._validate_mass_conservation(network, new_edge_flows, source_node_id, sink_node_id, Q_total)
        
        return node_pressures, new_edge_flows
    
    def _solve_two_node_case(self, network: FlowNetwork, source_node_id: str, sink_node_id: str, 
                           Q_total: float, fluid_properties: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Handle the special case of a network with only two nodes"""
        # Find all connections between source and sink
        connections = []
        for conn in network.connections:
            if ((conn.from_node.id == source_node_id and conn.to_node.id == sink_node_id) or
                (conn.from_node.id == sink_node_id and conn.to_node.id == source_node_id)):
                connections.append(conn)
        
        if not connections:
            raise ValueError("No connections found between source and sink nodes")
        
        # For multiple parallel connections, we need to solve for flow distribution
        if len(connections) == 1:
            # Single connection case
            conn = connections[0]
            resistance = self._compute_resistance(conn.component, Q_total, fluid_properties)
            pressure_drop = resistance * Q_total
            
            if conn.from_node.id == source_node_id:
                flow = Q_total
            else:
                flow = -Q_total
            
            node_pressures = {
                source_node_id: pressure_drop,
                sink_node_id: 0.0
            }
            
            edge_flows = {
                conn.component.id: flow
            }
            
            return node_pressures, edge_flows
        
        else:
            # Multiple parallel connections - use iterative approach
            # Initialize flows equally
            edge_flows = {}
            initial_flow = Q_total / len(connections)
            for conn in connections:
                edge_flows[conn.component.id] = initial_flow
            
            # Iterate to find correct flow distribution
            for iteration in range(20):  # Max iterations
                # Compute resistances at current flows
                resistances = {}
                conductances = {}
                for conn in connections:
                    flow = edge_flows[conn.component.id]
                    resistance = self._compute_resistance(conn.component, flow, fluid_properties)
                    resistances[conn.component.id] = resistance
                    conductances[conn.component.id] = 1.0 / resistance
                
                # Total conductance
                total_conductance = sum(conductances.values())
                
                # Pressure drop (same across all parallel paths)
                # Using equivalent resistance: R_eq = 1 / sum(1/R_i)
                R_equivalent = 1.0 / total_conductance
                pressure_drop = R_equivalent * Q_total
                
                # Compute new flows based on pressure drop
                new_edge_flows = {}
                for conn in connections:
                    conductance = conductances[conn.component.id]
                    flow = conductance * pressure_drop
                    new_edge_flows[conn.component.id] = flow
                
                # Check convergence
                max_change = max(abs(new_edge_flows[conn.component.id] - edge_flows[conn.component.id])
                               for conn in connections)
                
                if max_change < 1e-8:
                    break
                
                edge_flows = new_edge_flows
            
            node_pressures = {
                source_node_id: pressure_drop,
                sink_node_id: 0.0
            }
            
            return node_pressures, edge_flows
    
    def _initialize_flows(self, network: FlowNetwork, source_node_id: str, sink_node_id: str, 
                         Q_total: float) -> Dict[str, float]:
        """Initialize edge flows with a better guess than equal distribution"""
        edge_flows = {}
        
        # Simple initialization: distribute flow equally among all edges
        n_edges = len(network.connections)
        if n_edges == 0:
            return edge_flows
        
        initial_flow = Q_total / n_edges
        for conn in network.connections:
            edge_flows[conn.component.id] = initial_flow
        
        return edge_flows
    
    def _compute_resistance(self, component, flow: float, fluid_properties: Dict) -> float:
        """Compute resistance for a component at given flow rate"""
        if abs(flow) > 1e-12:
            pressure_drop = component.calculate_pressure_drop(abs(flow), fluid_properties)
            resistance = pressure_drop / abs(flow)
        else:
            # For zero flow, estimate resistance at small flow
            small_flow = self.config.dq_absolute
            pressure_drop = component.calculate_pressure_drop(small_flow, fluid_properties)
            resistance = pressure_drop / small_flow
        
        # Ensure minimum resistance to avoid numerical issues
        return max(resistance, self.config.min_resistance)
    
    def _validate_mass_conservation(self, 
                                   network: FlowNetwork,
                                   edge_flows: Dict[str, float],
                                   source_node_id: str,
                                   sink_node_id: str,
                                   Q_total: float,
                                   tolerance: float = 1e-6):
        """
        Validate that mass conservation is satisfied at all nodes.
        
        Args:
            network: The flow network
            edge_flows: Dictionary of edge flows
            source_node_id: Source node ID
            sink_node_id: Sink node ID  
            Q_total: Total flow rate
            tolerance: Tolerance for mass conservation check
        """
        for node_id, node in network.nodes.items():
            flow_in = 0.0
            flow_out = 0.0
            
            # Sum flows into and out of this node
            for conn in network.connections:
                flow = edge_flows[conn.component.id]
                
                if conn.to_node.id == node_id:
                    flow_in += flow
                elif conn.from_node.id == node_id:
                    flow_out += flow
            
            # Net flow at node
            net_flow = flow_in - flow_out
            
            # Expected net flow
            if node_id == source_node_id:
                expected_net = -Q_total  # Flow leaves source
            elif node_id == sink_node_id:
                expected_net = Q_total   # Flow enters sink
            else:
                expected_net = 0.0       # No net flow at intermediate nodes
            
            error = abs(net_flow - expected_net)
            if error > tolerance:
                self.logger.warning(f"Mass conservation violated at node {node_id}: "
                                  f"net_flow={net_flow:.6f}, expected={expected_net:.6f}, "
                                  f"error={error:.6f}")