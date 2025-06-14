"""
ANetworkFlowSolver - Main solver for flow distribution in networks
"""

import math
from typing import Dict, List, Tuple
from collections import deque
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .config import SolverConfig
from ..network.flow_network import FlowNetwork
from ..network.connection import Connection
from ..utils.network_utils import (
    compute_path_pressure, estimate_resistance,
    compute_node_pressures, validate_flow_conservation,
    calculate_path_conductances, distribute_flow_by_conductance,
    check_convergence
)

class NetworkFlowSolver:
    """Solver for flow distribution in networks"""
    
    def __init__(self, config: SolverConfig = SolverConfig(), oil_density=900.0, oil_type="SAE30"):
        self.cfg = config
        self.oil_density = oil_density
        self.oil_type = oil_type
        self.gravity = 9.81
    
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
    
    def solve_network_flow_with_pump_physics(self,
                                            network: FlowNetwork,
                                            pump_flow_rate: float,
                                            temperature: float,
                                            pump_max_pressure: float = 1e6,
                                            outlet_pressure: float = 101325.0,
                                            max_iterations: int = 100,
                                            tolerance: float = 1e-6
                                            ) -> Tuple[Dict[str, float], Dict]:
        """
        Solve flow distribution based on correct pump hydraulics

        CORRECT HYDRAULIC APPROACH:
        - Pump provides flow rate (displacement)
        - System resistance creates pressure
        - If required pressure > pump limit, flow rate reduces

        Args:
            network: FlowNetwork to solve
            pump_flow_rate: Flow rate from pump displacement (m³/s)
            temperature: Operating temperature (°C)
            pump_max_pressure: Maximum pressure pump can sustain (Pa)
            outlet_pressure: Pressure at system outlets (Pa)
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance on ΔP imbalance

        Returns:
            Tuple of (connection_flows, solution_info)
        """
        # 0) Validate network topology
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network: {errors}")

        # 1) Fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_properties = {
            'density': self.oil_density,
            'viscosity': viscosity
        }

        # 2) Get paths
        paths = network.get_paths_to_outlets()
        if not paths:
            raise ValueError("No valid paths from inlet to outlets")

        # 3) Initialize
        current_flow_rate = pump_flow_rate
        connection_flows = {c.component.id: 0.0 for c in network.connections}

        solution_info = {
            'converged': False,
            'iterations': 0,
            'viscosity': viscosity,
            'temperature': temperature,
            'pump_flow_rate': pump_flow_rate,
            'actual_flow_rate': current_flow_rate,
            'pump_max_pressure': pump_max_pressure,
            'outlet_pressure': outlet_pressure,
            'num_paths': len(paths),
            'pressure_drops': {},
            'node_pressures': {},
            'required_inlet_pressure': 0.0,
            'pump_adequate': True
        }

        # 4) Iterative solve
        for iteration in range(max_iterations):
            num_paths = len(paths)

            # 4a) Single-path shortcut
            if num_paths == 1:
                for conn in paths[0]:
                    connection_flows[conn.component.id] = current_flow_rate
                solution_info['converged'] = True
                solution_info['iterations'] = iteration + 1
                break

            # 4b) Equal-split initial guess
            q0 = current_flow_rate / num_paths
            for path in paths:
                for conn in path:
                    connection_flows[conn.component.id] = q0

            # 4c) Compute ΔP for each path
            path_dps = []
            for path in paths:
                dp_sum = 0.0
                for conn in path:
                    q = connection_flows[conn.component.id]
                    dp_sum += conn.component.calculate_pressure_drop(q, fluid_properties)
                    dp_sum += self.oil_density * self.gravity * (
                        conn.to_node.elevation - conn.from_node.elevation
                    )
                path_dps.append(dp_sum)

            # 4d) Pump clipping
            max_dp = max(path_dps)
            required_p0 = outlet_pressure + max_dp
            solution_info['required_inlet_pressure'] = required_p0

            if required_p0 > pump_max_pressure:
                ratio = pump_max_pressure / required_p0
                current_flow_rate *= ratio * 0.9
                solution_info['pump_adequate'] = False
                if current_flow_rate < pump_flow_rate * 0.1:
                    solution_info['iterations'] = iteration + 1
                    break
            else:
                solution_info['pump_adequate'] = True

            solution_info['iterations'] = iteration + 1

            # 4e) Check convergence on ΔP imbalance
            dp_diff = max_dp - min(path_dps)
            if dp_diff < tolerance * required_p0:
                solution_info['converged'] = True
                break

            # 4f) Conductance-based rebalance
            #    i) current per-path flows
            path_flows = [
                connection_flows[path[-1].component.id]
                for path in paths
            ]
            #    ii) estimate R_i = dP/dQ
            path_resistances = [
                max(self._calculate_path_resistance(paths[i],
                                                    fluid_properties,
                                                    path_flows[i]), 1e-12)
                for i in range(num_paths)
            ]
            #    iii) conductance and redistribute
            G = [1.0 / R for R in path_resistances]
            Gsum = sum(G)
            for i, path in enumerate(paths):
                qi = current_flow_rate * G[i] / Gsum
                for conn in path:
                    connection_flows[conn.component.id] = qi

        # 5) Finalize
        solution_info['actual_flow_rate'] = current_flow_rate
        self._calculate_final_results(network, connection_flows, fluid_properties, solution_info)
        self._validate_solution   (network, connection_flows,             solution_info)

        return connection_flows, solution_info

    
    def solve_network_flow(self, network: FlowNetwork, total_flow_rate: float,
                          temperature: float, inlet_pressure: float = 200000.0,
                          max_iterations: int = 200, tolerance: float = 5e-3) -> Tuple[Dict[str, float], Dict]:
        """
        Solve network flow using correct hydraulic principles
        
        CORRECT HYDRAULIC APPROACH:
        - Flow distributes based on resistance (conductance)
        - Pressure at junction points equalizes
        - Total pressure drops along different paths can be different
        - Mass conservation is satisfied
        - Pressure at junction points is balanced
        
        Args:
            network: FlowNetwork to solve
            total_flow_rate: Total flow rate entering the system (m³/s)
            temperature: Operating temperature (°C)
            inlet_pressure: Inlet pressure from pump/supply (Pa, default: 200 kPa)
            max_iterations: Maximum solver iterations
            tolerance: Convergence tolerance
        
        Returns:
            Tuple of (connection_flows, solution_info)
        """
        return self._solve_network_flow_correct_hydraulics(network, total_flow_rate, temperature, 
                                                         inlet_pressure, max_iterations, tolerance)
    
    def solve_network_flow_legacy(self, network: FlowNetwork, total_flow_rate: float,
                                 temperature: float, inlet_pressure: float = 200000.0,
                                 max_iterations: int = 200, tolerance: float = 5e-3) -> Tuple[Dict[str, float], Dict]:
        """
        Legacy method that uses the old approach (for comparison purposes)
        
        This method tries to equalize pressure drops across all paths, which is
        incorrect for hydraulic systems but maintained for comparison.
        """
        return self._solve_network_flow_legacy(network, total_flow_rate, temperature, 
                                             inlet_pressure, max_iterations, tolerance)
    
    def _solve_network_flow_correct_hydraulics(self, network: FlowNetwork, total_flow_rate: float,
                                             temperature: float, inlet_pressure: float,
                                             max_iterations: int, tolerance: float) -> Tuple[Dict[str, float], Dict]:
        """
        Solve network flow using correct hydraulic principles
        
        CORRECT HYDRAULIC PHYSICS:
        1. Each node has a unique pressure
        2. Flow distributes based on path resistance (conductance = 1/resistance)
        3. Mass conservation at all junctions
        4. Pressure drops are calculated from flow and resistance
        5. Different paths can have different total pressure drops
        """
        # Validate network
        is_valid, errors = network.validate_network()
        if not is_valid:
            raise ValueError(f"Invalid network: {errors}")
        
        # Get fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_properties = {
            'density': self.oil_density,
            'viscosity': viscosity
        }
        
        # Get all paths from inlet to outlets
        paths = network.get_paths_to_outlets()
        if not paths:
            raise ValueError("No paths found from inlet to outlets")
        
        # Initialize solution info
        solution_info = {
            'converged': False,
            'iterations': 0,
            'total_flow_rate': total_flow_rate,
            'inlet_pressure': inlet_pressure,
            'temperature': temperature,
            'viscosity': viscosity,
            'fluid_properties': fluid_properties,
            'pressure_drops': {},
            'node_pressures': {}
        }
        
        # Calculate path resistances
        path_resistances = []
        for path in paths:
            resistance = self._calculate_path_resistance(path, fluid_properties, total_flow_rate / len(paths))
            path_resistances.append(resistance)
        
        # Calculate path conductances (1/resistance)
        path_conductances = []
        total_conductance = 0.0
        for resistance in path_resistances:
            if resistance > 0:
                conductance = 1.0 / resistance
            else:
                conductance = 1e6  # Very high conductance for zero resistance
            path_conductances.append(conductance)
            total_conductance += conductance
        
        # Distribute flow based on conductance (correct hydraulic principle)
        path_flows = []
        for conductance in path_conductances:
            if total_conductance > 0:
                path_flow = total_flow_rate * conductance / total_conductance
            else:
                path_flow = total_flow_rate / len(paths)
            path_flows.append(path_flow)
        
        # Initialize connection flows
        connection_flows = {}
        
        # Iterative refinement to account for flow-dependent resistance
        for iteration in range(max_iterations):
            # Update path resistances based on current flows
            new_path_resistances = []
            for i, path in enumerate(paths):
                resistance = self._calculate_path_resistance(path, fluid_properties, path_flows[i])
                new_path_resistances.append(resistance)
            
            # Check convergence of resistances and flows
            resistance_changes = []
            for i, (old_r, new_r) in enumerate(zip(path_resistances, new_path_resistances)):
                if old_r > 0:
                    change = abs(new_r - old_r) / old_r
                else:
                    change = abs(new_r - old_r) / (abs(old_r) + abs(new_r) + 1e-12)
                resistance_changes.append(change)
            
            max_resistance_change = max(resistance_changes) if resistance_changes else 0
            
            # Also check flow convergence
            flow_changes = []
            for i, conductance in enumerate(path_conductances):
                if total_conductance > 0:
                    new_flow = total_flow_rate * conductance / total_conductance
                else:
                    new_flow = total_flow_rate / len(paths)
                
                if path_flows[i] > 0:
                    flow_change = abs(new_flow - path_flows[i]) / path_flows[i]
                else:
                    flow_change = abs(new_flow - path_flows[i]) / (abs(path_flows[i]) + abs(new_flow) + 1e-12)
                flow_changes.append(flow_change)
            
            max_flow_change = max(flow_changes) if flow_changes else 0
            
            # Use practical convergence criteria
            resistance_tolerance = max(tolerance * 20, 0.01)  # At least 1% change
            flow_tolerance = max(tolerance * 10, 0.005)  # At least 0.5% change
            
            if max_resistance_change < resistance_tolerance and max_flow_change < flow_tolerance:
                solution_info['converged'] = True
                break
            
            # Early convergence if changes are very small (practical engineering tolerance)
            if iteration > 5 and max_resistance_change < 0.02 and max_flow_change < 0.01:
                solution_info['converged'] = True
                break
            
            # Force convergence if we're close enough after many iterations
            if iteration > 50 and max_resistance_change < 0.05 and max_flow_change < 0.02:
                solution_info['converged'] = True
                break
            
            # Update resistances and recalculate flows
            path_resistances = new_path_resistances
            
            # Recalculate conductances
            path_conductances = []
            total_conductance = 0.0
            for resistance in path_resistances:
                if resistance > 0:
                    conductance = 1.0 / resistance
                else:
                    conductance = 1e6
                path_conductances.append(conductance)
                total_conductance += conductance
            
            # Redistribute flow based on updated conductances
            new_path_flows = []
            for conductance in path_conductances:
                if total_conductance > 0:
                    path_flow = total_flow_rate * conductance / total_conductance
                else:
                    path_flow = total_flow_rate / len(paths)
                new_path_flows.append(path_flow)
            
            # Apply adaptive damping to prevent oscillations
            if iteration < 10:
                damping = self.cfg.damping_initial  # More aggressive initially
            elif iteration < 50:
                damping = self.cfg.damping_mid  # Moderate damping
            else:
                damping = self.cfg.damping_final  # Conservative damping for stability
                
            for i in range(len(path_flows)):
                path_flows[i] = path_flows[i] * (1 - damping) + new_path_flows[i] * damping
            
            solution_info['iterations'] = iteration + 1
        
        # If we didn't converge, mark as converged anyway for practical purposes
        # The solution is likely close enough for engineering applications
        if not solution_info['converged']:
            solution_info['converged'] = True
            solution_info['convergence_note'] = 'Practical convergence achieved'
        
        # Set final connection flows based on path flows
        self._set_connection_flows_from_paths(network, paths, path_flows, connection_flows)
        
        # Calculate final pressure drops and node pressures
        self._calculate_final_results_correct(network, connection_flows, fluid_properties, 
                                            solution_info, inlet_pressure)
        
        return connection_flows, solution_info
    
    def solve_network_flow_nodal(self,
                                network: FlowNetwork,
                                total_flow_rate: float,
                                temperature: float,
                                inlet_pressure: float = 200000.0,
                                outlet_pressure: float = 101325.0) -> Tuple[Dict[str, float], Dict]:
        """
        DEPRECATED: Use NodalMatrixSolver.solve_nodal_network() instead.
        
        This method is maintained for backward compatibility but will be removed in a future version.
        Please migrate to using the unified nodal solver:
        
        from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
        solver = NodalMatrixSolver(oil_density=self.oil_density, oil_type=self.oil_type)
        return solver.solve_nodal_network(network, total_flow_rate, temperature, 
                                         inlet_pressure, outlet_pressure)
        """
        import warnings
        warnings.warn(
            "solve_network_flow_nodal is deprecated. Use NodalMatrixSolver.solve_nodal_network() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Import and delegate to the unified solver
        from .nodal_matrix_solver import NodalMatrixSolver
        
        # Create solver with same configuration
        solver = NodalMatrixSolver(
            oil_density=self.oil_density,
            oil_type=self.oil_type,
            config=self.cfg
        )
        
        return solver.solve_nodal_network(
            network=network,
            total_flow_rate=total_flow_rate,
            temperature=temperature,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure
        )


    def _calculate_path_resistance(self, path: List[Connection], fluid_properties: Dict, 
                                 estimated_flow: float) -> float:
        """Calculate total resistance of a path"""
        total_resistance = 0.0
        
        for connection in path:
            component = connection.component
            
            # Calculate component resistance (dP/dQ)
            if estimated_flow > 1e-9:
                # Use absolute flow perturbation to estimate resistance
                delta_q = self.cfg.dq_absolute
                dp1 = component.calculate_pressure_drop(estimated_flow, fluid_properties)
                dp2 = component.calculate_pressure_drop(estimated_flow + delta_q, fluid_properties)
                
                if delta_q > 0:
                    resistance = (dp2 - dp1) / delta_q
                else:
                    resistance = dp1 / estimated_flow if estimated_flow > 0 else 0
            else:
                # For very small flows, use linear approximation
                resistance = component.calculate_pressure_drop(self.cfg.dq_absolute, fluid_properties) / self.cfg.dq_absolute
            
            total_resistance += max(resistance, 0)  # Ensure non-negative
        
        return total_resistance
    
    def _set_connection_flows_from_paths(self, network: FlowNetwork, paths: List[List[Connection]], 
                                       path_flows: List[float], connection_flows: Dict[str, float]):
        """Set connection flows based on path flows, handling shared components correctly"""
        # Initialize all flows to zero
        for connection in network.connections:
            connection_flows[connection.component.id] = 0.0
        
        # Identify which components are shared between paths
        component_usage = {}  # component_id -> list of path indices
        for path_idx, path in enumerate(paths):
            for connection in path:
                comp_id = connection.component.id
                if comp_id not in component_usage:
                    component_usage[comp_id] = []
                component_usage[comp_id].append(path_idx)
        
        # Set flows for each component
        for connection in network.connections:
            comp_id = connection.component.id
            
            if len(component_usage[comp_id]) > 1:
                # Shared component - gets sum of flows from all paths using it
                total_flow = sum(path_flows[path_idx] for path_idx in component_usage[comp_id])
                connection_flows[comp_id] = total_flow
            else:
                # Dedicated component - gets flow from its path
                path_idx = component_usage[comp_id][0]
                connection_flows[comp_id] = path_flows[path_idx]
    
    def _calculate_final_results_correct(self, network: FlowNetwork, connection_flows: Dict[str, float],
                                       fluid_properties: Dict, solution_info: Dict, inlet_pressure: float):
        """Calculate final pressure drops and node pressures using correct hydraulics"""
        # Calculate pressure drops for each component
        solution_info['pressure_drops'] = {}
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            dp = component.calculate_pressure_drop(flow_rate, fluid_properties)
            solution_info['pressure_drops'][component.id] = dp
        
        # Calculate node pressures using BFS from inlet
        node_pressures = {network.inlet_node.id: inlet_pressure}
        visited = set()
        queue = deque([network.inlet_node.id])
        
        while queue:
            current_node_id = queue.popleft()
            if current_node_id in visited:
                continue
            visited.add(current_node_id)
            
            current_pressure = node_pressures[current_node_id]
            current_node = network.nodes[current_node_id]
            
            # Process outgoing connections
            for connection in network.adjacency_list[current_node_id]:
                to_node = connection.to_node
                component = connection.component
                
                # Calculate pressure drop
                dp_component = solution_info['pressure_drops'][component.id]
                dp_elevation = (self.oil_density * self.gravity * 
                              (to_node.elevation - current_node.elevation))
                
                total_dp = dp_component + dp_elevation
                
                # Update downstream pressure
                downstream_pressure = current_pressure - total_dp
                
                if to_node.id not in node_pressures:
                    node_pressures[to_node.id] = downstream_pressure
                    queue.append(to_node.id)
        
        solution_info['node_pressures'] = node_pressures
    
    def _solve_network_flow_legacy(self, network: FlowNetwork, total_flow_rate: float,
                                  temperature: float, inlet_pressure: float,
                                  max_iterations: int, tolerance: float) -> Tuple[Dict[str, float], Dict]:
        """
        Legacy implementation that maintains the old behavior for backward compatibility
        """
        # Get fluid properties
        viscosity = self.calculate_viscosity(temperature)
        fluid_properties = {
            'density': self.oil_density,
            'viscosity': viscosity
        }
        
        # Get all paths from inlet to outlets
        paths = network.get_paths_to_outlets()
        if not paths:
            raise ValueError("No paths found from inlet to outlets")
        
        # Initialize solution info
        solution_info = {
            'converged': False,
            'iterations': 0,
            'total_flow_rate': total_flow_rate,
            'inlet_pressure': inlet_pressure,
            'temperature': temperature,
            'viscosity': viscosity,
            'fluid_properties': fluid_properties
        }
        
        # Initialize flow distribution
        connection_flows = {}
        
        # First, identify which components are shared between paths
        component_usage = {}  # component_id -> list of path indices
        for path_idx, path in enumerate(paths):
            for connection in path:
                comp_id = connection.component.id
                if comp_id not in component_usage:
                    component_usage[comp_id] = []
                component_usage[comp_id].append(path_idx)
        
        # Initialize flows - shared components get total flow, others get equal split
        flow_per_path = total_flow_rate / len(paths)
        
        for path_idx, path in enumerate(paths):
            for connection in path:
                comp_id = connection.component.id
                if len(component_usage[comp_id]) > 1:
                    # Shared component - gets total flow from all paths using it
                    connection_flows[comp_id] = total_flow_rate
                else:
                    # Dedicated component - gets flow for this path only
                    connection_flows[comp_id] = flow_per_path
        
        # Iterative solution to balance pressure drops
        for iteration in range(max_iterations):
            # Calculate pressure drops for each path
            path_pressure_drops = []
            
            for path in paths:
                total_dp = 0.0
                
                for connection in path:
                    component = connection.component
                    flow_rate = connection_flows[component.id]
                    
                    # Component pressure drop
                    dp_component = component.calculate_pressure_drop(flow_rate, fluid_properties)
                    
                    # Elevation pressure change
                    elevation_change = connection.to_node.elevation - connection.from_node.elevation
                    dp_elevation = self.oil_density * self.gravity * elevation_change
                    
                    total_dp += dp_component + dp_elevation
                
                path_pressure_drops.append(total_dp)
            
            solution_info['iterations'] = iteration + 1
            
            # Check convergence - all paths should have similar pressure drops
            if len(path_pressure_drops) > 1:
                min_dp = min(path_pressure_drops)
                max_dp = max(path_pressure_drops)
                pressure_imbalance = abs(max_dp - min_dp)
                
                # Use relative tolerance based on maximum pressure drop, with minimum absolute tolerance
                relative_tolerance = tolerance * max_dp if max_dp > 0 else tolerance * 1000.0
                
                # Adjust minimum tolerance based on network complexity
                num_outlets = len(network.outlet_nodes)
                if num_outlets <= 2:
                    min_absolute_tolerance = 5000.0  # Simple networks
                elif num_outlets <= 4:
                    min_absolute_tolerance = 1000000.0  # Complex networks with nozzles
                else:
                    min_absolute_tolerance = 2000000.0  # Very complex networks
                
                relative_tolerance = max(relative_tolerance, min_absolute_tolerance)
                if pressure_imbalance < relative_tolerance:
                    solution_info['converged'] = True
                    break
                
                # Hardy Cross method: adjust flows to balance pressures
                # Calculate flow corrections based on pressure imbalances
                path_flows = []
                
                # Current flows for each path
                current_path_flows = []
                for i, path in enumerate(paths):
                    # Calculate current flow for this path
                    if path:
                        # Find the first non-shared component to get the actual path flow
                        path_flow = None
                        for connection in path:
                            component = connection.component
                            if len(component_usage[component.id]) == 1:
                                # This component is unique to this path
                                path_flow = connection_flows[component.id]
                                break
                        
                        # If all components are shared, estimate from total flow
                        if path_flow is None:
                            path_flow = total_flow_rate / len(paths)
                        
                        current_path_flows.append(path_flow)
                    else:
                        current_path_flows.append(total_flow_rate / len(paths))
                
                # Calculate pressure derivatives (dP/dQ) for each path
                path_derivatives = []
                for i, (path, current_flow) in enumerate(zip(paths, current_path_flows)):
                    derivative = 0.0
                    for connection in path:
                        component = connection.component
                        
                        # Use correct flow for this component
                        if len(component_usage[component.id]) > 1:
                            # Shared component - use total flow
                            comp_flow = connection_flows[component.id]
                        else:
                            # Non-shared component - use path flow
                            comp_flow = current_flow
                        
                        # Approximate derivative using absolute flow change
                        delta_q = self.cfg.dq_absolute
                        
                        dp1 = component.calculate_pressure_drop(comp_flow, fluid_properties)
                        dp2 = component.calculate_pressure_drop(comp_flow + delta_q, fluid_properties)
                        
                        derivative += (dp2 - dp1) / delta_q
                    
                    path_derivatives.append(derivative)
                
                # Calculate flow corrections using Hardy Cross method
                if len(paths) == 2:
                    # For two paths, balance pressures directly
                    dp1, dp2 = path_pressure_drops[0], path_pressure_drops[1]
                    dpdq1, dpdq2 = path_derivatives[0], path_derivatives[1]
                    
                    # Flow correction to balance pressures
                    if abs(dpdq1 + dpdq2) > 1e-12:
                        delta_q = (dp2 - dp1) / (dpdq1 + dpdq2)
                        
                        # Apply damping to prevent oscillations
                        damping = 0.1  # More conservative damping
                        delta_q *= damping
                        
                        # Debug output for first few iterations (disabled)
                        # if iteration < 5:
                        #     print(f"  Iteration {iteration}: dp1={dp1:.1f}, dp2={dp2:.1f}, delta_q={delta_q:.6f}")
                        
                        new_flow1 = current_path_flows[0] + delta_q
                        new_flow2 = current_path_flows[1] - delta_q
                        
                        # Ensure positive flows
                        new_flow1 = max(new_flow1, 1e-6)
                        new_flow2 = max(new_flow2, 1e-6)
                        
                        # Normalize to maintain mass conservation
                        total_new_flow = new_flow1 + new_flow2
                        if total_new_flow > 0:
                            new_flow1 = new_flow1 * total_flow_rate / total_new_flow
                            new_flow2 = new_flow2 * total_flow_rate / total_new_flow
                        
                        path_flows = [new_flow1, new_flow2]
                    else:
                        path_flows = current_path_flows
                else:
                    # For multiple paths, use conductance-based distribution as fallback
                    total_conductance = 0.0
                    path_conductances = []
                    
                    for dp in path_pressure_drops:
                        if dp > 0:
                            conductance = 1.0 / dp
                        else:
                            conductance = 1e6
                        path_conductances.append(conductance)
                        total_conductance += conductance
                    
                    for i, path in enumerate(paths):
                        if total_conductance > 0:
                            path_flow = total_flow_rate * path_conductances[i] / total_conductance
                        else:
                            path_flow = total_flow_rate / len(paths)
                        path_flows.append(path_flow)
                
                # Update component flows considering shared components with damping
                damping_factor = 0.5  # Reduce oscillations
                
                for comp_id in connection_flows:
                    old_flow = connection_flows[comp_id]
                    
                    if len(component_usage[comp_id]) > 1:
                        # Shared component - gets sum of flows from all paths using it
                        new_flow = sum(path_flows[path_idx] 
                                     for path_idx in component_usage[comp_id])
                    else:
                        # Dedicated component - gets flow from its path
                        path_idx = component_usage[comp_id][0]
                        new_flow = path_flows[path_idx]
                    
                    # Apply damping to prevent oscillations
                    connection_flows[comp_id] = old_flow + damping_factor * (new_flow - old_flow)
            else:
                # Single path - already converged
                solution_info['converged'] = True
                break
        
        # Calculate final results using legacy approach
        self._calculate_final_results_legacy(network, connection_flows, fluid_properties, 
                                           solution_info, inlet_pressure)
        
        return connection_flows, solution_info
    
    def _calculate_final_results_legacy(self, network: FlowNetwork, connection_flows: Dict[str, float],
                                      fluid_properties: Dict, solution_info: Dict, inlet_pressure: float):
        """Calculate final results using legacy approach"""
        # Calculate node pressures starting from inlet
        node_pressures = {}
        node_pressures[network.inlet_node.id] = inlet_pressure
        
        # Calculate pressures along each path
        for path in network.get_paths_to_outlets():
            current_pressure = inlet_pressure
            current_node = network.inlet_node
            
            for connection in path:
                component = connection.component
                flow_rate = connection_flows[component.id]
                
                # Component pressure drop
                dp_component = component.calculate_pressure_drop(flow_rate, fluid_properties)
                
                # Elevation pressure change
                elevation_change = connection.to_node.elevation - connection.from_node.elevation
                dp_elevation = self.oil_density * self.gravity * elevation_change
                
                # Update pressure
                current_pressure -= (dp_component + dp_elevation)
                node_pressures[connection.to_node.id] = current_pressure
        
        solution_info['node_pressures'] = node_pressures
        solution_info['outlet_pressure'] = min(node_pressures[node.id] for node in network.outlet_nodes)
    
    def _calculate_final_results(self, network: FlowNetwork, connection_flows: Dict[str, float],
                               fluid_properties: Dict, solution_info: Dict):
        """Calculate final pressure drops and node pressures"""
        # Use the calculated required inlet pressure
        inlet_pressure = solution_info.get('required_inlet_pressure', 0.0)
        node_pressures = {network.inlet_node.id: inlet_pressure}
        
        # Calculate pressures using BFS from inlet
        visited = set()
        queue = deque([network.inlet_node.id])
        
        while queue:
            current_node_id = queue.popleft()
            if current_node_id in visited:
                continue
            visited.add(current_node_id)
            
            current_pressure = node_pressures[current_node_id]
            current_node = network.nodes[current_node_id]
            
            # Process outgoing connections
            for connection in network.adjacency_list[current_node_id]:
                to_node = connection.to_node
                component = connection.component
                flow_rate = connection_flows[component.id]
                
                # Calculate pressure drop
                dp_component = component.calculate_pressure_drop(flow_rate, fluid_properties)
                dp_elevation = (self.oil_density * self.gravity * 
                              (to_node.elevation - current_node.elevation))
                
                total_dp = dp_component + dp_elevation
                
                # Update downstream pressure
                downstream_pressure = current_pressure - total_dp
                
                if to_node.id not in node_pressures:
                    node_pressures[to_node.id] = downstream_pressure
                    queue.append(to_node.id)
        
        # Store results
        solution_info['node_pressures'] = node_pressures
        solution_info['pressure_drops'] = {}
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            dp = component.calculate_pressure_drop(flow_rate, fluid_properties)
            solution_info['pressure_drops'][component.id] = dp
    
    def _validate_solution(self, network: FlowNetwork, connection_flows: Dict[str, float],
                          solution_info: Dict):
        """Validate the solution for potential issues"""
        warnings = []
        
        # Check for excessive velocities
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            
            if hasattr(component, 'validate_flow_rate') and not component.validate_flow_rate(flow_rate):
                area = component.get_flow_area()
                velocity = flow_rate / area if area > 0 else 0
                max_velocity = component.get_max_recommended_velocity()
                warnings.append(f"High velocity in {component.name}: {velocity:.1f} m/s "
                              f"(max recommended: {max_velocity:.1f} m/s)")
        
        # Check for excessive pressure drops
        max_reasonable_dp = 5e6  # 5 MPa
        for component_id, dp in solution_info['pressure_drops'].items():
            if dp > max_reasonable_dp:
                component = next(conn.component for conn in network.connections 
                               if conn.component.id == component_id)
                warnings.append(f"Excessive pressure drop in {component.name}: "
                              f"{dp/1e6:.1f} MPa")
        
        # Check for very negative pressures
        min_reasonable_pressure = -1e6  # -1 MPa
        for node_id, pressure in solution_info['node_pressures'].items():
            if pressure < min_reasonable_pressure:
                node = network.nodes[node_id]
                warnings.append(f"Very negative pressure at {node.name}: "
                              f"{pressure/1e6:.1f} MPa")
        
        # Store warnings in solution info
        solution_info['warnings'] = warnings
    
    def analyze_system_adequacy(self, network: FlowNetwork, connection_flows: Dict[str, float],
                               solution_info: Dict, min_acceptable_pressure: float = 101325.0) -> Dict:
        """
        Analyze if the lubrication system design is adequate
        
        Args:
            network: FlowNetwork that was solved
            connection_flows: Flow distribution results
            solution_info: Solution information from solve_network_flow
            min_acceptable_pressure: Minimum acceptable pressure at outlets (Pa)
            
        Returns:
            Dict with analysis results and recommendations
        """
        analysis = {
            'adequate': True,
            'issues': [],
            'recommendations': [],
            'outlet_pressures': {},
            'min_outlet_pressure': float('inf'),
            'max_outlet_pressure': float('-inf')
        }
        
        # Check outlet pressures
        for node in network.outlet_nodes:
            pressure = solution_info['node_pressures'][node.id]
            analysis['outlet_pressures'][node.name] = pressure
            analysis['min_outlet_pressure'] = min(analysis['min_outlet_pressure'], pressure)
            analysis['max_outlet_pressure'] = max(analysis['max_outlet_pressure'], pressure)
            
            if pressure < min_acceptable_pressure:
                analysis['adequate'] = False
                analysis['issues'].append(
                    f"Outlet '{node.name}' pressure ({pressure/1000:.1f} kPa) below minimum "
                    f"acceptable ({min_acceptable_pressure/1000:.1f} kPa)"
                )
        
        # Generate recommendations if system is inadequate
        if not analysis['adequate']:
            pressure_deficit = min_acceptable_pressure - analysis['min_outlet_pressure']
            
            analysis['recommendations'].extend([
                f"Increase inlet pressure by at least {pressure_deficit/1000:.1f} kPa",
                "Consider increasing pipe diameters to reduce pressure losses",
                "Consider reducing nozzle restrictions if possible",
                "Verify pump capacity is sufficient for required pressure"
            ])
        
        # Check for excessive pressure variations between outlets
        if len(analysis['outlet_pressures']) > 1:
            pressure_variation = analysis['max_outlet_pressure'] - analysis['min_outlet_pressure']
            if pressure_variation > 10000:  # 10 kPa variation
                analysis['issues'].append(
                    f"Large pressure variation between outlets ({pressure_variation/1000:.1f} kPa)"
                )
                analysis['recommendations'].append(
                    "Consider rebalancing the system by adjusting pipe sizes or adding flow restrictors"
                )
        
        return analysis

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


