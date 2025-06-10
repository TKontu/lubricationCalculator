"""
Iterative Nodal-Matrix Solver for Hydraulic Networks with Non-linear Edge Resistances

This module implements an iterative nodal-matrix solver that finds node pressures and edge flows
such that mass is conserved and the pressure-flow law ΔP_e = R_e(Q_e) · Q_e holds on every edge.

The solver uses the nodal analysis method where:
1. Each node has a unique pressure (except reference node)
2. Conductance matrix A is built from edge conductances G_e = 1/R_e(Q_e)
3. System A·p = b is solved iteratively as conductances depend on flows
4. Flows are computed from pressure differences and conductances
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, List, Tuple, Optional, Callable
import logging

from ..network.flow_network import FlowNetwork
from ..network.node import Node
from ..network.connection import Connection


class NodalMatrixSolver:
    """
    Iterative nodal-matrix solver for hydraulic networks with non-linear resistances.
    
    This solver implements the nodal analysis method where node pressures are the primary
    unknowns. The method is particularly effective for networks with multiple junctions
    and complex topologies.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the nodal matrix solver.
        
        Args:
            logger: Optional logger for debugging output
        """
        self.logger = logger or logging.getLogger(__name__)
    
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
            small_flow = 1e-9
            pressure_drop = component.calculate_pressure_drop(small_flow, fluid_properties)
            resistance = pressure_drop / small_flow
        
        # Ensure minimum resistance to avoid numerical issues
        return max(resistance, 1e-6)
    
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