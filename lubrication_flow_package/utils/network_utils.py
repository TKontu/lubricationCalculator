"""
Network Utility Functions

Shared functions for path enumeration, pressure calculations, and resistance estimation
used across different solver implementations.
"""

from typing import List, Dict, Set, Tuple
from collections import deque

# Import types from parent modules
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..network.connection import Connection
    from ..network.flow_network import FlowNetwork


def find_all_paths(network: 'FlowNetwork') -> List[List['Connection']]:
    """
    Find all paths from inlet to outlets in the network.
    
    Args:
        network: The flow network to analyze
        
    Returns:
        List of paths, where each path is a list of connections
        
    Raises:
        ValueError: If no inlet node is defined or no paths found
    """
    if not network.inlet_node:
        raise ValueError("No inlet node defined")
    
    paths = []
    
    def dfs_paths(current_node_id: str, current_path: List['Connection'], 
                 visited: Set[str]):
        """Depth-first search to find all paths"""
        if current_node_id in visited:
            return  # Avoid cycles
        
        visited.add(current_node_id)
        
        # Check if this is an outlet node
        current_node = network.nodes[current_node_id]
        if current_node in network.outlet_nodes:
            paths.append(current_path.copy())
        
        # Continue to connected nodes
        for connection in network.adjacency_list[current_node_id]:
            new_path = current_path + [connection]
            dfs_paths(connection.to_node.id, new_path, visited.copy())
    
    dfs_paths(network.inlet_node.id, [], set())
    
    if not paths:
        raise ValueError("No paths found from inlet to outlets")
    
    return paths


def compute_path_pressure(path: List['Connection'], flow_rates: Dict[str, float], 
                         fluid_props: Dict, gravity: float = 9.81, 
                         oil_density: float = 900.0) -> float:
    """
    Compute total pressure drop along a path.
    
    Args:
        path: List of connections forming the path
        flow_rates: Dictionary mapping component IDs to flow rates
        fluid_props: Fluid properties dictionary (density, viscosity)
        gravity: Gravitational acceleration (m/s²)
        oil_density: Oil density (kg/m³)
        
    Returns:
        Total pressure drop along the path (Pa)
    """
    total_pressure_drop = 0.0
    
    for connection in path:
        component = connection.component
        flow_rate = flow_rates.get(component.id, 0.0)
        
        # Component pressure drop
        dp_component = component.calculate_pressure_drop(flow_rate, fluid_props)
        
        # Elevation pressure change (positive for upward flow)
        elevation_change = connection.to_node.elevation - connection.from_node.elevation
        dp_elevation = oil_density * gravity * elevation_change
        
        # Total pressure drop for this connection
        total_pressure_drop += dp_component + dp_elevation
    
    return total_pressure_drop


def estimate_resistance(path: List['Connection'], Q: float, fluid_props: Dict) -> float:
    """
    Estimate the hydraulic resistance (dP/dQ) of a path.
    
    Args:
        path: List of connections forming the path
        Q: Estimated flow rate through the path (m³/s)
        fluid_props: Fluid properties dictionary (density, viscosity)
        
    Returns:
        Total hydraulic resistance of the path (Pa·s/m³)
    """
    total_resistance = 0.0
    
    for connection in path:
        component = connection.component
        
        # Calculate component resistance using numerical differentiation
        if Q > 1e-9:
            # Use small flow perturbation to estimate resistance
            delta_q = max(Q * 0.01, 1e-8)  # At least 1e-8 for numerical stability
            
            dp1 = component.calculate_pressure_drop(Q, fluid_props)
            dp2 = component.calculate_pressure_drop(Q + delta_q, fluid_props)
            
            resistance = (dp2 - dp1) / delta_q
        else:
            # For very small flows, use linear approximation
            test_flow = 1e-6
            dp = component.calculate_pressure_drop(test_flow, fluid_props)
            resistance = dp / test_flow
        
        # Ensure non-negative resistance
        total_resistance += max(resistance, 0.0)
    
    return total_resistance


def compute_node_pressures(network: 'FlowNetwork', flow_rates: Dict[str, float],
                          fluid_props: Dict, inlet_pressure: float,
                          gravity: float = 9.81, oil_density: float = 900.0) -> Dict[str, float]:
    """
    Compute pressure at all nodes using breadth-first traversal from inlet.
    
    Args:
        network: The flow network
        flow_rates: Dictionary mapping component IDs to flow rates
        fluid_props: Fluid properties dictionary
        inlet_pressure: Pressure at the inlet node (Pa)
        gravity: Gravitational acceleration (m/s²)
        oil_density: Oil density (kg/m³)
        
    Returns:
        Dictionary mapping node IDs to pressures (Pa)
    """
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
            flow_rate = flow_rates.get(component.id, 0.0)
            
            # Calculate pressure drop
            dp_component = component.calculate_pressure_drop(flow_rate, fluid_props)
            dp_elevation = (oil_density * gravity * 
                          (to_node.elevation - current_node.elevation))
            
            total_dp = dp_component + dp_elevation
            
            # Update downstream pressure
            downstream_pressure = current_pressure - total_dp
            
            if to_node.id not in node_pressures:
                node_pressures[to_node.id] = downstream_pressure
                queue.append(to_node.id)
    
    return node_pressures


def validate_flow_conservation(network: 'FlowNetwork', flow_rates: Dict[str, float],
                              tolerance: float = 1e-6) -> Tuple[bool, List[str]]:
    """
    Validate that flow conservation is satisfied at all junction nodes.
    
    Args:
        network: The flow network
        flow_rates: Dictionary mapping component IDs to flow rates
        tolerance: Tolerance for flow conservation check
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check flow conservation at each node (except inlet and outlets)
    for node_id, node in network.nodes.items():
        if node == network.inlet_node or node in network.outlet_nodes:
            continue  # Skip inlet and outlet nodes
        
        # Calculate net flow into the node
        inflow = 0.0
        outflow = 0.0
        
        # Check all connections
        for connection in network.connections:
            flow_rate = flow_rates.get(connection.component.id, 0.0)
            
            if connection.to_node.id == node_id:
                # Flow into this node
                inflow += flow_rate
            elif connection.from_node.id == node_id:
                # Flow out of this node
                outflow += flow_rate
        
        # Check conservation
        net_flow = abs(inflow - outflow)
        if net_flow > tolerance:
            errors.append(f"Flow conservation violated at node {node.name}: "
                         f"inflow={inflow:.6f}, outflow={outflow:.6f}, "
                         f"imbalance={net_flow:.6f}")
    
    return len(errors) == 0, errors


def calculate_path_conductances(paths: List[List['Connection']], fluid_props: Dict,
                               estimated_flows: List[float]) -> List[float]:
    """
    Calculate hydraulic conductances (1/resistance) for a list of paths.
    
    Args:
        paths: List of paths (each path is a list of connections)
        fluid_props: Fluid properties dictionary
        estimated_flows: Estimated flow rates for each path
        
    Returns:
        List of conductances for each path
    """
    conductances = []
    
    for i, path in enumerate(paths):
        resistance = estimate_resistance(path, estimated_flows[i], fluid_props)
        
        if resistance > 0:
            conductance = 1.0 / resistance
        else:
            conductance = 1e6  # Very high conductance for zero resistance
        
        conductances.append(conductance)
    
    return conductances


def distribute_flow_by_conductance(total_flow: float, conductances: List[float]) -> List[float]:
    """
    Distribute total flow among paths based on their conductances.
    
    Args:
        total_flow: Total flow rate to distribute
        conductances: List of path conductances
        
    Returns:
        List of flow rates for each path
    """
    total_conductance = sum(conductances)
    
    if total_conductance <= 0:
        # Equal distribution if no valid conductances
        return [total_flow / len(conductances)] * len(conductances)
    
    # Distribute proportionally to conductance
    path_flows = []
    for conductance in conductances:
        flow = total_flow * conductance / total_conductance
        path_flows.append(flow)
    
    return path_flows


def check_convergence(old_values: List[float], new_values: List[float], 
                     tolerance: float) -> Tuple[bool, float]:
    """
    Check convergence between old and new values.
    
    Args:
        old_values: Previous iteration values
        new_values: Current iteration values
        tolerance: Convergence tolerance
        
    Returns:
        Tuple of (converged, max_relative_change)
    """
    if len(old_values) != len(new_values):
        return False, float('inf')
    
    max_change = 0.0
    
    for old_val, new_val in zip(old_values, new_values):
        if old_val != 0:
            relative_change = abs(new_val - old_val) / abs(old_val)
        else:
            relative_change = abs(new_val) if new_val != 0 else 0.0
        
        max_change = max(max_change, relative_change)
    
    return max_change < tolerance, max_change


def apply_damping(old_values: List[float], new_values: List[float], 
                 damping_factor: float) -> List[float]:
    """
    Apply damping to new values to improve convergence stability.
    
    Args:
        old_values: Previous iteration values
        new_values: Current iteration values
        damping_factor: Damping factor (0 = no change, 1 = full update)
        
    Returns:
        Damped values
    """
    if len(old_values) != len(new_values):
        return new_values
    
    damped_values = []
    for old_val, new_val in zip(old_values, new_values):
        damped_val = old_val + damping_factor * (new_val - old_val)
        damped_values.append(damped_val)
    
    return damped_values


def get_adaptive_damping(iteration: int, max_iterations: int, 
                        initial_damping: float = 0.3,
                        final_damping: float = 0.7) -> float:
    """
    Get adaptive damping factor that increases with iterations.
    
    Args:
        iteration: Current iteration number (0-based)
        max_iterations: Maximum number of iterations
        initial_damping: Starting damping factor
        final_damping: Final damping factor
        
    Returns:
        Adaptive damping factor
    """
    if max_iterations <= 1:
        return final_damping
    
    # Linear interpolation between initial and final damping
    progress = min(iteration / (max_iterations - 1), 1.0)
    damping = initial_damping + progress * (final_damping - initial_damping)
    
    return damping