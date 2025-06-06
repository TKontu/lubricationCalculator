#!/usr/bin/env python3
"""
Network-Based Lubrication Flow Distribution Calculator

This enhanced version supports tree-like branching structures with component-based
system building including channels, connectors, and nozzles.

Key features:
- Tree-like network topology support
- Component-based architecture (channels, connectors, nozzles)
- Intuitive system building with connections
- Advanced network flow analysis
- Mass conservation at all junctions
- Pressure drop calculations through component sequences
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from collections import defaultdict, deque


class ComponentType(Enum):
    """Types of flow components"""
    CHANNEL = "channel"
    CONNECTOR = "connector"
    NOZZLE = "nozzle"


class ConnectorType(Enum):
    """Types of connectors"""
    T_JUNCTION = "t_junction"
    X_JUNCTION = "x_junction"
    ELBOW_90 = "elbow_90"
    REDUCER = "reducer"
    STRAIGHT = "straight"


class NozzleType(Enum):
    """Types of nozzles"""
    SHARP_EDGED = "sharp_edged"
    ROUNDED = "rounded"
    VENTURI = "venturi"
    FLOW_NOZZLE = "flow_nozzle"


@dataclass
class Node:
    """Represents a connection point in the network"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pressure: float = 0.0  # Pa
    elevation: float = 0.0  # m
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Node_{self.id}"


@dataclass
class Connection:
    """Represents a connection between two nodes through a component"""
    from_node: Node
    to_node: Node
    component: 'FlowComponent'
    flow_rate: float = 0.0  # m³/s (positive = from_node to to_node)


class FlowComponent:
    """Base class for all flow components"""
    
    def __init__(self, component_id: str = None, name: str = ""):
        self.id = component_id or str(uuid.uuid4())[:8]
        self.name = name or f"{self.__class__.__name__}_{self.id}"
        self.component_type = ComponentType.CHANNEL  # Override in subclasses
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop through this component"""
        raise NotImplementedError("Subclasses must implement calculate_pressure_drop")
    
    def get_flow_area(self) -> float:
        """Get the flow area of this component"""
        raise NotImplementedError("Subclasses must implement get_flow_area")
    
    def validate_flow_rate(self, flow_rate: float) -> bool:
        """Validate if the flow rate is acceptable for this component"""
        return flow_rate >= 0
    
    def get_max_recommended_velocity(self) -> float:
        """Get maximum recommended velocity for this component type"""
        # Default conservative velocity limit
        return 10.0  # m/s


class Channel(FlowComponent):
    """Represents a pipe or drilling channel"""
    
    def __init__(self, diameter: float, length: float, roughness: float = 0.00015,
                 component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.CHANNEL
        self.diameter = diameter  # m
        self.length = length      # m
        self.roughness = roughness  # m
        
        # Validation
        if diameter <= 0:
            raise ValueError("Channel diameter must be positive")
        if length <= 0:
            raise ValueError("Channel length must be positive")
        if roughness < 0:
            raise ValueError("Channel roughness cannot be negative")
    
    def get_flow_area(self) -> float:
        """Get the flow area"""
        return math.pi * (self.diameter / 2) ** 2
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using Darcy-Weisbach equation"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        viscosity = fluid_properties['viscosity']
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        reynolds = (density * velocity * self.diameter) / viscosity
        
        # Calculate friction factor
        relative_roughness = self.roughness / self.diameter
        friction_factor = self._calculate_friction_factor(reynolds, relative_roughness)
        
        # Darcy-Weisbach equation
        pressure_drop = friction_factor * (self.length / self.diameter) * \
                       (density * velocity ** 2) / 2
        
        return pressure_drop
    
    def _calculate_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate friction factor using appropriate correlations"""
        if reynolds <= 0:
            return 0
            
        if reynolds < 2300:  # Laminar flow
            return 64 / reynolds
        elif reynolds < 4000:  # Transition region
            f_lam = 64 / 2300
            f_turb = self._turbulent_friction_factor(4000, relative_roughness)
            # Smooth transition
            x = (reynolds - 2300) / (4000 - 2300)
            return f_lam * (1 - x)**3 + f_turb * x**3
        else:  # Turbulent flow
            return self._turbulent_friction_factor(reynolds, relative_roughness)
    
    def _turbulent_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate turbulent friction factor using Swamee-Jain approximation"""
        if relative_roughness <= 0:  # Smooth pipe
            if reynolds < 100000:
                return 0.316 / (reynolds ** 0.25)
            else:
                return 0.0032 + 0.221 / (reynolds ** 0.237)
        else:
            # Swamee-Jain approximation
            term1 = relative_roughness / 3.7
            term2 = 5.74 / (reynolds ** 0.9)
            log_arg = max(term1 + term2, 1e-10)
            denominator = math.log10(log_arg)
            
            if abs(denominator) < 1e-10:
                return 0.02
                
            return 0.25 / (denominator ** 2)


class Connector(FlowComponent):
    """Represents a connector (junction, elbow, reducer, etc.)"""
    
    def __init__(self, connector_type: ConnectorType, diameter: float,
                 diameter_out: float = None, loss_coefficient: float = None,
                 component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.CONNECTOR
        self.connector_type = connector_type
        self.diameter = diameter  # m (inlet diameter)
        self.diameter_out = diameter_out or diameter  # m (outlet diameter)
        
        # Set default loss coefficients if not provided
        if loss_coefficient is None:
            self.loss_coefficient = self._get_default_loss_coefficient()
        else:
            self.loss_coefficient = loss_coefficient
        
        # Validation
        if diameter <= 0:
            raise ValueError("Connector diameter must be positive")
        if self.diameter_out <= 0:
            raise ValueError("Connector outlet diameter must be positive")
    
    def _get_default_loss_coefficient(self) -> float:
        """Get default loss coefficient based on connector type"""
        defaults = {
            ConnectorType.T_JUNCTION: 1.5,    # Branch tee
            ConnectorType.X_JUNCTION: 2.0,    # Cross junction
            ConnectorType.ELBOW_90: 0.9,      # 90-degree elbow
            ConnectorType.REDUCER: 0.5,       # Gradual reducer
            ConnectorType.STRAIGHT: 0.0       # Straight connector
        }
        return defaults.get(self.connector_type, 1.0)
    
    def get_flow_area(self) -> float:
        """Get the flow area (use smaller diameter)"""
        return math.pi * (min(self.diameter, self.diameter_out) / 2) ** 2
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using loss coefficient method"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        
        # Use inlet diameter for velocity calculation
        area = math.pi * (self.diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Minor loss equation: ΔP = K * ρ * v² / 2
        pressure_drop = self.loss_coefficient * density * velocity ** 2 / 2
        
        # Add expansion/contraction losses for reducers
        if self.connector_type == ConnectorType.REDUCER and self.diameter != self.diameter_out:
            area_ratio = (self.diameter_out / self.diameter) ** 2
            if area_ratio < 1:  # Contraction
                contraction_loss = 0.5 * (1 - area_ratio) * density * velocity ** 2 / 2
                pressure_drop += contraction_loss
            else:  # Expansion
                expansion_loss = (1 - 1/area_ratio) ** 2 * density * velocity ** 2 / 2
                pressure_drop += expansion_loss
        
        return pressure_drop


class Nozzle(FlowComponent):
    """Represents a flow nozzle or orifice"""
    
    def __init__(self, diameter: float, nozzle_type: NozzleType = NozzleType.SHARP_EDGED,
                 discharge_coeff: float = None, component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.NOZZLE
        self.diameter = diameter  # m
        self.nozzle_type = nozzle_type
        
        # Set default discharge coefficient if not provided
        if discharge_coeff is None:
            self.discharge_coeff = self._get_default_discharge_coeff()
        else:
            self.discharge_coeff = discharge_coeff
        
        # Validation
        if diameter <= 0:
            raise ValueError("Nozzle diameter must be positive")
        if not (0 < self.discharge_coeff <= 1):
            raise ValueError("Discharge coefficient must be between 0 and 1")
    
    def _get_default_discharge_coeff(self) -> float:
        """Get default discharge coefficient based on nozzle type"""
        defaults = {
            NozzleType.SHARP_EDGED: 0.6,
            NozzleType.ROUNDED: 0.8,
            NozzleType.VENTURI: 0.95,
            NozzleType.FLOW_NOZZLE: 0.98
        }
        return defaults[self.nozzle_type]
    
    def get_flow_area(self) -> float:
        """Get the flow area"""
        return math.pi * (self.diameter / 2) ** 2
    
    def get_max_recommended_velocity(self) -> float:
        """Get maximum recommended velocity for nozzles"""
        # Nozzles can handle higher velocities than pipes
        velocity_limits = {
            NozzleType.SHARP_EDGED: 15.0,    # m/s
            NozzleType.ROUNDED: 20.0,        # m/s  
            NozzleType.VENTURI: 30.0,        # m/s
            NozzleType.FLOW_NOZZLE: 25.0     # m/s
        }
        return velocity_limits.get(self.nozzle_type, 15.0)
    
    def validate_flow_rate(self, flow_rate: float) -> bool:
        """Validate if the flow rate is acceptable for this nozzle"""
        if flow_rate <= 0:
            return True  # Zero flow is always acceptable
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        max_velocity = self.get_max_recommended_velocity()
        
        return velocity <= max_velocity
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using orifice flow equation"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        
        # Orifice pressure drop calculation
        if self.nozzle_type == NozzleType.VENTURI:
            # Venturi has lower permanent pressure loss due to diffuser recovery
            K = ((1 / self.discharge_coeff ** 2) - 1) * 0.1  # 10% permanent loss
        else:
            # Standard orifice equation
            K = (1 / self.discharge_coeff ** 2) - 1
        
        pressure_drop = K * density * velocity ** 2 / 2
        
        return pressure_drop


class FlowNetwork:
    """Represents a complete flow network with nodes and connections"""
    
    def __init__(self, name: str = "Flow Network"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self.inlet_node: Optional[Node] = None
        self.outlet_nodes: List[Node] = []
        
        # Network topology
        self.adjacency_list: Dict[str, List[Connection]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[Connection]] = defaultdict(list)
    
    def add_node(self, node: Node) -> Node:
        """Add a node to the network"""
        self.nodes[node.id] = node
        return node
    
    def create_node(self, name: str = "", elevation: float = 0.0) -> Node:
        """Create and add a new node to the network"""
        node = Node(name=name, elevation=elevation)
        return self.add_node(node)
    
    def connect_components(self, from_node: Node, to_node: Node, 
                          component: FlowComponent) -> Connection:
        """Connect two nodes through a component"""
        connection = Connection(from_node, to_node, component)
        self.connections.append(connection)
        
        # Update adjacency lists
        self.adjacency_list[from_node.id].append(connection)
        self.reverse_adjacency[to_node.id].append(connection)
        
        return connection
    
    def set_inlet(self, node: Node):
        """Set the inlet node for the network"""
        self.inlet_node = node
    
    def add_outlet(self, node: Node):
        """Add an outlet node to the network"""
        if node not in self.outlet_nodes:
            self.outlet_nodes.append(node)
    
    def get_paths_to_outlets(self) -> List[List[Connection]]:
        """Get all paths from inlet to outlets"""
        if not self.inlet_node:
            raise ValueError("No inlet node defined")
        
        paths = []
        
        def dfs_paths(current_node_id: str, current_path: List[Connection], 
                     visited: Set[str]):
            if current_node_id in visited:
                return  # Avoid cycles
            
            visited.add(current_node_id)
            
            # Check if this is an outlet node
            current_node = self.nodes[current_node_id]
            if current_node in self.outlet_nodes:
                paths.append(current_path.copy())
            
            # Continue to connected nodes
            for connection in self.adjacency_list[current_node_id]:
                new_path = current_path + [connection]
                dfs_paths(connection.to_node.id, new_path, visited.copy())
        
        dfs_paths(self.inlet_node.id, [], set())
        return paths
    
    def validate_network(self) -> Tuple[bool, List[str]]:
        """Validate the network topology"""
        errors = []
        
        # Check if inlet is defined
        if not self.inlet_node:
            errors.append("No inlet node defined")
        
        # Check if outlets are defined
        if not self.outlet_nodes:
            errors.append("No outlet nodes defined")
        
        # Check connectivity
        if self.inlet_node:
            paths = self.get_paths_to_outlets()
            if not paths:
                errors.append("No paths from inlet to outlets")
            
            # Check if all outlets are reachable
            reachable_outlets = set()
            for path in paths:
                if path:
                    reachable_outlets.add(path[-1].to_node.id)
            
            unreachable = [node for node in self.outlet_nodes 
                          if node.id not in reachable_outlets]
            if unreachable:
                errors.append(f"Unreachable outlets: {[n.name for n in unreachable]}")
        
        # Check for isolated nodes
        connected_nodes = set()
        for connection in self.connections:
            connected_nodes.add(connection.from_node.id)
            connected_nodes.add(connection.to_node.id)
        
        isolated = [node for node_id, node in self.nodes.items() 
                   if node_id not in connected_nodes]
        if isolated:
            errors.append(f"Isolated nodes: {[n.name for n in isolated]}")
        
        return len(errors) == 0, errors
    
    def get_junction_nodes(self) -> List[Node]:
        """Get nodes that are junctions (more than 2 connections)"""
        junction_nodes = []
        
        for node_id, node in self.nodes.items():
            # Count total connections (incoming + outgoing)
            incoming = len(self.reverse_adjacency[node_id])
            outgoing = len(self.adjacency_list[node_id])
            total_connections = incoming + outgoing
            
            if total_connections > 2:
                junction_nodes.append(node)
        
        return junction_nodes
    
    def print_network_info(self):
        """Print network information"""
        print(f"\nNetwork: {self.name}")
        print(f"Nodes: {len(self.nodes)}")
        print(f"Connections: {len(self.connections)}")
        print(f"Inlet: {self.inlet_node.name if self.inlet_node else 'None'}")
        print(f"Outlets: {[n.name for n in self.outlet_nodes]}")
        
        # Print paths
        try:
            paths = self.get_paths_to_outlets()
            print(f"Paths to outlets: {len(paths)}")
            for i, path in enumerate(paths):
                path_str = " -> ".join([path[0].from_node.name] + 
                                     [conn.to_node.name for conn in path])
                print(f"  Path {i+1}: {path_str}")
        except Exception as e:
            print(f"Error getting paths: {e}")


class NetworkFlowSolver:
    """Solver for flow distribution in networks"""
    
    def __init__(self, oil_density: float = 900.0, oil_type: str = "SAE30"):
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
            "SAE60": {"A": 0.00025, "B": 1500, "C": 148}
        }
        
        if self.oil_type not in viscosity_params:
            raise ValueError(f"Oil type {self.oil_type} not supported")
        
        params = viscosity_params[self.oil_type]
        
        if T < params["C"]:
            T = params["C"] + 1
        
        viscosity = params["A"] * math.exp(params["B"] / (T - params["C"]))
        return max(1e-6, min(viscosity, 10.0))
    
    def solve_network_flow(self, network: FlowNetwork, total_flow_rate: float,
                          temperature: float, max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[Dict[str, float], Dict]:
        """
        Solve flow distribution in the network
        
        Returns:
            Tuple of (connection_flows, solution_info)
            connection_flows: Dict mapping connection component IDs to flow rates
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
            raise ValueError("No valid paths from inlet to outlets")
        
        # Initialize flow rates for each connection
        connection_flows = {conn.component.id: 0.0 for conn in network.connections}
        
        # Initial guess: distribute flow equally among outlets
        initial_flow_per_outlet = total_flow_rate / len(network.outlet_nodes)
        
        # For each path, set the flow rate to the outlet flow rate
        for i, path in enumerate(paths):
            path_flow = initial_flow_per_outlet
            for connection in path:
                # Each connection in the path carries the full path flow
                connection_flows[connection.component.id] = path_flow
        
        solution_info = {
            'converged': False,
            'iterations': 0,
            'viscosity': viscosity,
            'temperature': temperature,
            'total_flow_rate': total_flow_rate,
            'num_paths': len(paths),
            'pressure_drops': {},
            'node_pressures': {}
        }
        
        # Iterative solution
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
            
            # For proper hydraulic balancing, outlet pressures should be equal
            # Calculate current outlet pressures for each path
            outlet_pressures = []
            for i, path in enumerate(paths):
                # Outlet pressure = Inlet pressure - Total pressure drop
                outlet_pressure = 0.0 - path_pressure_drops[i]  # Inlet is at 0 Pa reference
                outlet_pressures.append(outlet_pressure)
            
            # Target outlet pressure (average of current outlet pressures)
            if len(outlet_pressures) > 1:
                # Use weighted average based on flow rates
                path_flows = [connection_flows[path[-1].component.id] for path in paths]
                total_flow = sum(path_flows)
                
                if total_flow > 0:
                    weights = [flow / total_flow for flow in path_flows]
                    target_outlet_pressure = sum(p * weight for p, weight in zip(outlet_pressures, weights))
                else:
                    target_outlet_pressure = np.mean(outlet_pressures)
            else:
                target_outlet_pressure = outlet_pressures[0]
            
            # Convert target outlet pressure back to target pressure drops
            target_pressure_drops = []
            for i, path in enumerate(paths):
                target_dp = 0.0 - target_outlet_pressure  # Target total pressure drop
                target_pressure_drops.append(target_dp)
            
            # Adjust flows to equalize pressure drops
            new_path_flows = []
            total_new_flow = 0.0
            
            for i, (path, current_dp, target_dp) in enumerate(zip(paths, path_pressure_drops, target_pressure_drops)):
                if current_dp > 0:
                    # Get current flow rate for this path (all components in series should have same flow)
                    current_path_flow = connection_flows[path[-1].component.id]  # Use outlet component flow
                    
                    if current_path_flow > 0:
                        # For hydraulic systems: ΔP ∝ Q^n where n ≈ 2 for turbulent flow
                        # But for nozzles, the relationship is more complex
                        flow_ratio = (target_dp / current_dp) ** 0.5  # Square root for turbulent flow
                        new_flow = current_path_flow * flow_ratio
                        
                        # Apply adaptive damping to prevent oscillations
                        # Use stronger damping for larger changes
                        change_ratio = abs(new_flow - current_path_flow) / current_path_flow
                        damping = 0.9 if change_ratio > 0.5 else 0.7
                        new_flow = damping * new_flow + (1 - damping) * current_path_flow
                    else:
                        new_flow = total_flow_rate / len(paths)
                else:
                    new_flow = total_flow_rate / len(paths)
                
                new_path_flows.append(new_flow)
                total_new_flow += new_flow
            
            # Normalize to maintain total flow
            if total_new_flow > 0:
                normalization_factor = total_flow_rate / total_new_flow
                new_path_flows = [flow * normalization_factor for flow in new_path_flows]
            
            # Update connection flows using a simpler approach
            new_connection_flows = {conn.component.id: 0.0 for conn in network.connections}
            
            # Calculate flows for each connection based on which paths use it
            for connection in network.connections:
                total_flow_through_connection = 0.0
                
                # Find all paths that use this connection
                for i, path in enumerate(paths):
                    if connection in path:
                        total_flow_through_connection += new_path_flows[i]
                
                new_connection_flows[connection.component.id] = total_flow_through_connection
            
            # Check convergence
            max_change = 0.0
            for conn_id in connection_flows:
                change = abs(new_connection_flows[conn_id] - connection_flows[conn_id])
                max_change = max(max_change, change)
            
            if max_change < tolerance:
                solution_info['converged'] = True
                solution_info['iterations'] = iteration + 1
                break
            
            # Update flows with damping for stability
            damping_factor = 0.7
            for conn_id in connection_flows:
                connection_flows[conn_id] = (damping_factor * new_connection_flows[conn_id] + 
                                           (1 - damping_factor) * connection_flows[conn_id])
        
        # Calculate final pressure drops and node pressures
        self._calculate_final_results(network, connection_flows, fluid_properties, solution_info)
        
        # Validate results
        self._validate_solution(network, connection_flows, solution_info)
        
        return connection_flows, solution_info
    
    def _calculate_final_results(self, network: FlowNetwork, connection_flows: Dict[str, float],
                               fluid_properties: Dict, solution_info: Dict):
        """Calculate final pressure drops and node pressures"""
        # Set inlet pressure as reference (0 Pa gauge)
        node_pressures = {network.inlet_node.id: 0.0}
        
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
        print(f"Total Flow Rate: {solution_info['total_flow_rate']*1000:.1f} L/s")
        print(f"Converged: {solution_info['converged']} (in {solution_info['iterations']} iterations)")
        
        # Print connection flows
        print(f"\n{'Component':<20} {'Type':<12} {'Flow Rate':<12} {'Pressure Drop'}")
        print(f"{'Name':<20} {'':12} {'(L/s)':<12} {'(Pa)'}")
        print("-" * 65)
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows[component.id]
            pressure_drop = solution_info['pressure_drops'].get(component.id, 0)
            
            print(f"{component.name:<20} {component.component_type.value:<12} "
                  f"{flow_rate*1000:<12.3f} {pressure_drop:<12.1f}")
        
        # Print node pressures
        print(f"\n{'Node':<20} {'Pressure (Pa)':<15} {'Elevation (m)'}")
        print("-" * 45)
        
        for node_id, pressure in solution_info['node_pressures'].items():
            node = network.nodes[node_id]
            print(f"{node.name:<20} {pressure:<15.1f} {node.elevation:<12.1f}")
        
        # Print warnings if any
        if 'warnings' in solution_info and solution_info['warnings']:
            print(f"\n{'WARNINGS':<20}")
            print("-" * 45)
            for warning in solution_info['warnings']:
                print(f"⚠️  {warning}")


def create_simple_tree_example() -> Tuple[FlowNetwork, float, float]:
    """Create a simple tree network example"""
    network = FlowNetwork("Simple Tree Example")
    
    # Create nodes
    inlet = network.create_node("Inlet", elevation=0.0)
    junction1 = network.create_node("Junction1", elevation=1.0)
    branch1_end = network.create_node("Branch1_End", elevation=2.0)
    branch2_end = network.create_node("Branch2_End", elevation=1.5)
    outlet1 = network.create_node("Outlet1", elevation=2.0)
    outlet2 = network.create_node("Outlet2", elevation=1.5)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create components with better nozzle sizing
    main_channel = Channel(diameter=0.08, length=10.0, name="Main Channel")
    branch1_channel = Channel(diameter=0.05, length=8.0, name="Branch1 Channel")
    branch2_channel = Channel(diameter=0.04, length=6.0, name="Branch2 Channel")
    # Further increase nozzle sizes to reduce pressure drops and velocities
    nozzle1 = Nozzle(diameter=0.025, nozzle_type=NozzleType.VENTURI, name="Nozzle1")
    nozzle2 = Nozzle(diameter=0.020, nozzle_type=NozzleType.SHARP_EDGED, name="Nozzle2")
    
    # Connect components to form tree structure
    # Main line: Inlet -> Junction1
    network.connect_components(inlet, junction1, main_channel)
    
    # Branch 1: Junction1 -> Branch1_End -> Outlet1 (through nozzle)
    network.connect_components(junction1, branch1_end, branch1_channel)
    network.connect_components(branch1_end, outlet1, nozzle1)
    
    # Branch 2: Junction1 -> Branch2_End -> Outlet2 (through nozzle)
    network.connect_components(junction1, branch2_end, branch2_channel)
    network.connect_components(branch2_end, outlet2, nozzle2)
    
    total_flow_rate = 0.015  # 15 L/s
    temperature = 40  # °C
    
    return network, total_flow_rate, temperature


def main():
    """Demonstrate the network-based flow calculator"""
    print("NETWORK-BASED LUBRICATION FLOW DISTRIBUTION CALCULATOR")
    print("="*60)
    
    # Create solver
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Create example network
    network, total_flow_rate, temperature = create_simple_tree_example()
    
    # Print network info
    network.print_network_info()
    
    # Validate network
    is_valid, errors = network.validate_network()
    print(f"\nNetwork validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")
    
    if is_valid:
        # Solve flow distribution
        connection_flows, solution_info = solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Print results
        solver.print_results(network, connection_flows, solution_info)


if __name__ == "__main__":
    main()
