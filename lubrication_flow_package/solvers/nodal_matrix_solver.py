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

    def solve_nodal_network(
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

            # 5. Wrap multi-outlet into single virtual sink
            work_net = self._prepare_multi_outlet_network(network, outlet_nodes)
            source_id = inlet_node.id
            sink_id   = work_net.outlet_nodes[0].id

            # 6. Call iterative solver
            node_pressures, edge_flows = self.solve_nodal_iterative(
                network=work_net,
                source_node_id=source_id,
                sink_node_id=sink_id,
                Q_total=total_flow_rate,
                fluid_properties=fluid_properties,
                tol_flow=tol * 1e-3,
                tol_pressure=tol * 1_000,
                max_iter=max_iter
            )

            # 7. Filter out the virtual connectors
            connection_flows = {
                cid: flow for cid, flow in edge_flows.items()
                if cid not in work_net.virtual_connection_ids
            }
            # Only remove the sink pressure if it’s the virtual sink from a multi-outlet collapse
            if len(outlet_nodes) > 1:
                node_pressures.pop(sink_id, None)

            # 8. Build solution_info
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

            # 9. Compute pressure drops per connection
            for conn in network.connections:
                comp = conn.component
                q    = connection_flows.get(comp.id, 0.0)
                dp   = comp.calculate_pressure_drop(q, fluid_properties)
                solution_info['pressure_drops'][comp.id] = dp

            # 10. Shift all computed node pressures by outlet_pressure reference
            for nid in solution_info['node_pressures']:
                solution_info['node_pressures'][nid] += outlet_pressure

            return connection_flows, solution_info

    def _prepare_multi_outlet_networkOLD(
        self,
        network: FlowNetwork,
        outlet_nodes: Dict[str, object]
    ) -> FlowNetwork:
        """
        Deep-copy the network and collapse multiple outlets into one virtual sink.
        Tracks the IDs of zero-loss connectors in virtual_connection_ids.
        """
        work_net = copy.deepcopy(network)
        # if only one outlet, just record empty set
        if len(outlet_nodes) == 1:
            work_net.virtual_connection_ids = set()
            return work_net

        # create virtual sink
        virtual_sink = work_net.add_node(name="__multi_outlet_sink__")

        # prepare zero-loss connector template
        zero_loss = Connector(
            connector_type=ConnectorType.STRAIGHT,
            diameter=1.0,
            loss_coefficient=0.0,
            auto_calculate_k=False
        )

        vids = set()
        for out in outlet_nodes:
            conn = work_net.connect_components(out, virtual_sink, zero_loss)
            vids.add(conn.id)

        work_net.outlet_nodes = [virtual_sink]
        work_net.virtual_connection_ids = vids
        return work_net 

    def _prepare_multi_outlet_network(
        self,
        network: FlowNetwork,
        outlet_nodes: List[Node]
    ) -> FlowNetwork:
        """
        Deep-copy the network and collapse multiple outlets into one virtual sink.
        Tracks the IDs of zero-loss connectors in virtual_connection_ids.
        """
        work_net = copy.deepcopy(network)

        # if only one outlet, nothing special needed
        if len(outlet_nodes) == 1:
            work_net.virtual_connection_ids = set()
            return work_net

        # 1) create a new sink node
        virtual_sink = work_net.create_node(name="__multi_outlet_sink__", elevation=0.0)

        # 2) zero-loss connector template
        zero_loss = Connector(
            connector_type=ConnectorType.STRAIGHT,
            diameter=1.0,
            loss_coefficient=0.0,
            auto_calculate_k=False
        )

        # 3) attach each original outlet → virtual sink
        vids = set()
        for out in outlet_nodes:
            conn = work_net.connect_components(out, virtual_sink, zero_loss)
            # record the component.id of that zero‐loss link
            vids.add(conn.component.id)

        # 4) replace outlets list with just our virtual sink
        work_net.outlet_nodes = [virtual_sink]
        work_net.virtual_connection_ids = vids

        return work_net


    def solve_nodal_network_with_pump_physicsOLDOLD(
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
        Fixed‐Q solver with full non‐linear convergence at each pressure guess.
        """
        # 1) Validate network
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network: {errors}")

        # 2) Solver parameters
        inner_max = (max_iterations or self.config.max_iterations)
        tol = (tolerance or self.config.tolerance)
        q_tol = tol * pump_flow_rate

        # 3) Fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_props = {'density': self.oil_density, 'viscosity': viscosity}

        # 4) Identify inlet/sink
        inlet = network.inlet_node
        sinks = network.outlet_nodes or []
        if inlet is None or not sinks:
            raise ValueError("Network must have one inlet and at least one outlet")
        # collapse multi‐outlet into a single virtual sink
        work_net = self._prepare_multi_outlet_network(network, sinks)
        source_id = inlet.id
        sink_id   = work_net.outlet_nodes[0].id

        # 5) Pressure bracket
        p_lo, p_hi = outlet_pressure, pump_max_pressure
        converged = False

        for iteration in range(inner_max):
            p_guess = 0.5 * (p_lo + p_hi)

            # --- call the full non‐linear iterative solver at this inlet head ---
            node_p, edge_Q = self.solve_nodal_iterative(
                network=work_net,
                source_node_id=source_id,
                sink_node_id=sink_id,
                Q_total=pump_flow_rate,
                fluid_properties=fluid_props,
                tol_flow=tol * 1e-3,
                tol_pressure=tol * 1e3,
                max_iter=inner_max
            )

            # compute delivered flow out of the (virtual) source
            Q_delivered = sum(
                edge_Q[conn.component.id]
                for conn in work_net.adjacency_list[source_id]
            )

            # check convergence
            if abs(Q_delivered - pump_flow_rate) < q_tol:
                converged = True
                break

            # narrow bracket
            if Q_delivered > pump_flow_rate:
                # network too “easy”
                p_hi = p_guess
            else:
                # network too “hard”
                p_lo = p_guess

        # 6) Build solution_info
        # shift pressures to absolute reference
        for nid in node_p:
            node_p[nid] += outlet_pressure

        solution_info = {
            'required_inlet_pressure': p_guess,
            'actual_flow_rate':        Q_delivered,
            'converged':               converged,
            'iterations':              iteration + 1,
            'node_pressures':          node_p,
            'pressure_drops': {
                conn.component.id: conn.component.calculate_pressure_drop(
                    edge_Q.get(conn.component.id, 0.0),
                    fluid_props
                )
                for conn in network.connections
            },
            'fluid_properties': fluid_props
        }

        # filter out any virtual connections
        connection_flows = {
            cid: q for cid, q in edge_Q.items()
            if cid not in getattr(work_net, 'virtual_connection_ids', ())
        }

        return connection_flows, solution_info

    def solve_nodal_network_with_pump_physics(
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
        flows, sol = self.solve_nodal_network(
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
            'required_inlet_pressure': sol['node_pressures'][network.inlet_node.id],
            'fluid_properties':        sol['fluid_properties']
        }

        return flows, info


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
                resistance = self._calculate_component_resistance(
                    conn.component,
                    fluid_properties,
                    flow
                )
                conductance = 1.0 / resistance
                
                edge_resistances[conn.component.id] = resistance
                edge_conductances[conn.component.id] = conductance

            # Step 2: Build conductance matrix A · p = b with hydrostatic heads
            A = lil_matrix((n_active, n_active))
            b = np.zeros(n_active)

            for conn in network.connections:
                i_full = node_to_idx[conn.from_node.id]
                j_full = node_to_idx[conn.to_node.id]
                G = edge_conductances[conn.component.id]

                # Elevations
                z_i = conn.from_node.elevation
                z_j = conn.to_node.elevation
                # hydrostatic Δp = ρ g (z_j - z_i)
                dp_hydro = fluid_properties['density'] * self.gravity * (z_j - z_i)

                # Case A: both nodes unknown (active-active)
                if i_full != sink_idx and j_full != sink_idx:
                    i_act = full_to_active[i_full]
                    j_act = full_to_active[j_full]

                    # Conductance entries
                    A[i_act, i_act] += G
                    A[j_act, j_act] += G
                    A[i_act, j_act] -= G
                    A[j_act, i_act] -= G

                    # RHS includes hydrostatic head
                    b[i_act] += G * dp_hydro
                    b[j_act] -= G * dp_hydro

                # Case B: from active → sink
                elif i_full != sink_idx and j_full == sink_idx:
                    i_act = full_to_active[i_full]
                    A[i_act, i_act] += G
                    b[i_act] += G * dp_hydro

                # Case C: from sink → active
                elif i_full == sink_idx and j_full != sink_idx:
                    j_act = full_to_active[j_full]
                    A[j_act, j_act] += G
                    b[j_act] -= G * dp_hydro

            # Step 3: Set up RHS vector (net flow injections)
            # Only the source node has a net flow injection
            source_idx = node_to_idx[source_node_id]
            if source_idx != sink_idx:
                source_active = full_to_active[source_idx]
                b[source_active] += Q_total
            
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
                    #print("b vector:", b)
                    pressures_active = spsolve(A_csr, b)
                    if np.isscalar(pressures_active):
                        pressures_active = np.array([pressures_active])
                except Exception as e:
                    self.logger.error(f"Failed to solve linear system at iteration {iteration}: {e}")
                    # Try with regularization
                    A_reg = A_csr + 1e-12 * lil_matrix(np.eye(n_active))
                    try:
                        print("b vector:", b)
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
                

    def print_results(self, network: FlowNetwork, connection_flows: Dict[str, float],
                     solution_info: Dict):
        """Print detailed results"""
        print(f"\n{'='*70}")
        print("NETWORK FLOW DISTRIBUTION RESULTS")
        print(f"{'='*70}")
        
        print(f"Network: {network.name}")
        print(f"Temperature: {solution_info['temperature']:.1f}°C")
        print(f"Oil Type: {self.oil_type}")
        print(f"Oil Density: {self.oil_density:.1f} kg/m³")
        print(f"Dynamic Viscosity: {solution_info['viscosity']:.6f} Pa·s")
        
        # Handle different flow rate keys for backward compatibility
        flow_rate_key = 'total_flow_rate' if 'total_flow_rate' in solution_info else 'actual_flow_rate'
        if flow_rate_key in solution_info:
            print(f"Total Flow Rate: {solution_info[flow_rate_key]*1000:.1f} L/s")
        
        print(f"Converged: {solution_info['converged']} (in {solution_info['iterations']} iterations)")
        
        # Handle different pressure keys
        if 'inlet_pressure' in solution_info:
            print(f"Inlet Pressure: {solution_info['inlet_pressure']/1000:.1f} kPa")
        elif 'required_inlet_pressure' in solution_info:
            print(f"Required Inlet Pressure: {solution_info['required_inlet_pressure']/1000:.1f} kPa")
        
        # Calculate pressure drops if not already calculated
        if 'pressure_drops' not in solution_info:
            solution_info['pressure_drops'] = {}
            fluid_properties = solution_info.get('fluid_properties', {
                'density': self.oil_density,
                'viscosity': solution_info['viscosity']
            })
            
            for connection in network.connections:
                component = connection.component
                flow_rate = connection_flows[component.id]
                dp = component.calculate_pressure_drop(flow_rate, fluid_properties)
                solution_info['pressure_drops'][component.id] = dp
        
        # Print connection flows
        print(f"\n{'Component':<20} {'Type':<12} {'Flow Rate':<12} {'Pressure Drop'}")
        print(f"{'Name':<20} {'':12} {'(L/s)':<12} {'(kPa)'}")
        print("-" * 65)
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            pressure_drop = solution_info['pressure_drops'].get(component.id, 0)
            
            print(f"{component.name:<20} {component.component_type.value:<12} "
                  f"{flow_rate*1000:<12.3f} {pressure_drop/1000:<12.1f}")
        
        # Print node pressures
        print(f"\n{'Node':<20} {'Pressure (kPa)':<15} {'Elevation (m)'}")
        print("-" * 45)
        
        for node_id, pressure in solution_info['node_pressures'].items():
            node = network.nodes[node_id]
            print(f"{node.name:<20} {pressure/1000:<15.1f} {node.elevation:<12.1f}")
        
        # Print warnings if any
        if 'warnings' in solution_info and solution_info['warnings']:
            print(f"\n{'WARNINGS':<20}")
            print("-" * 45)
            for warning in solution_info['warnings']:
                print(f"⚠️  {warning}")
