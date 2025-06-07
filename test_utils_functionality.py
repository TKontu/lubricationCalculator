#!/usr/bin/env python3
"""
Test script to verify utils module functionality
"""

from lubrication_flow_package import FlowNetwork, Node, Connection, Channel
from lubrication_flow_package.utils import (
    find_all_paths, estimate_resistance, calculate_path_conductances,
    distribute_flow_by_conductance, check_convergence, compute_node_pressures
)

def test_utils_functionality():
    """Test all utility functions with a simple network"""
    print("TESTING UTILS MODULE FUNCTIONALITY")
    print("=" * 50)
    
    # Create a simple test network
    network = FlowNetwork()
    
    # Add nodes
    inlet = Node("inlet", 0, 0, 0)
    junction = Node("junction", 1, 0, 0)
    outlet1 = Node("outlet1", 2, 0, 0)
    outlet2 = Node("outlet2", 2, 1, 0)
    
    network.add_node(inlet)
    network.add_node(junction)
    network.add_node(outlet1)
    network.add_node(outlet2)
    
    # Add connections
    main_channel = Channel(0.01, 1.0, 0.001, "main")  # 10mm diameter, 1m length, 1mm roughness
    branch1_channel = Channel(0.008, 0.5, 0.001, "branch1")  # 8mm diameter, 0.5m length
    branch2_channel = Channel(0.008, 0.5, 0.001, "branch2")  # 8mm diameter, 0.5m length
    
    main_conn = network.connect_components(inlet, junction, main_channel)
    branch1_conn = network.connect_components(junction, outlet1, branch1_channel)
    branch2_conn = network.connect_components(junction, outlet2, branch2_channel)
    
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Test fluid properties
    fluid_props = {'density': 900.0, 'viscosity': 0.1}
    
    print("1. Testing find_all_paths...")
    paths = find_all_paths(network)
    print(f"   Found {len(paths)} paths:")
    for i, path in enumerate(paths):
        path_str = " -> ".join([conn.component.id for conn in path])
        print(f"   Path {i+1}: {path_str}")
    
    print("\n2. Testing estimate_resistance...")
    test_flow = 0.001  # 1 L/s
    for i, path in enumerate(paths):
        resistance = estimate_resistance(path, test_flow, fluid_props)
        print(f"   Path {i+1} resistance: {resistance:.2e} Pa·s/m³")
    
    print("\n3. Testing calculate_path_conductances...")
    flows = [0.0005, 0.0005]  # 0.5 L/s each
    conductances = calculate_path_conductances(paths, fluid_props, flows)
    print(f"   Path conductances: {[f'{c:.2e}' for c in conductances]}")
    
    print("\n4. Testing distribute_flow_by_conductance...")
    total_flow = 0.001  # 1 L/s
    distributed_flows = distribute_flow_by_conductance(total_flow, conductances)
    print(f"   Distributed flows: {[f'{f*1000:.3f}' for f in distributed_flows]} L/s")
    print(f"   Total: {sum(distributed_flows)*1000:.3f} L/s")
    
    print("\n5. Testing check_convergence...")
    old_values = [1.0, 2.0, 3.0]
    new_values = [1.01, 2.02, 3.03]
    converged, max_change = check_convergence(old_values, new_values, 0.05)
    print(f"   Converged: {converged}, Max change: {max_change:.4f}")
    
    print("\n6. Testing compute_node_pressures...")
    # Create flow rates dict
    flow_rates = {
        "main": 0.001,
        "branch1": 0.0005,
        "branch2": 0.0005
    }
    inlet_pressure = 200000.0  # 2 bar
    node_pressures = compute_node_pressures(network, flow_rates, fluid_props, inlet_pressure)
    print(f"   Node pressures:")
    for node_id, pressure in node_pressures.items():
        print(f"     {node_id}: {pressure/1000:.1f} kPa")
    
    print("\n" + "=" * 50)
    print("✓ All utils functions tested successfully!")
    return True

if __name__ == "__main__":
    test_utils_functionality()