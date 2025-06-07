#!/usr/bin/env python3
"""
Integration test to verify utils module functionality across all solver methods
"""

from lubrication_flow_package import FlowNetwork, Node, Connection, Channel
from lubrication_flow_package.solvers import NetworkFlowSolver

def convert_flows_to_names(flows_by_id, network):
    """Convert flow dictionary from component IDs to component names"""
    flows_by_name = {}
    for connection in network.connections:
        component_id = connection.component.id
        component_name = connection.component.name
        if component_id in flows_by_id:
            flows_by_name[component_name] = flows_by_id[component_id]
    return flows_by_name

def test_utils_integration():
    """Test that all solver methods work correctly with utils functions"""
    print("TESTING UTILS MODULE INTEGRATION")
    print("=" * 50)
    
    # Create a test network
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
    main_channel = Channel(0.01, 1.0, 0.001, name="main")  # 10mm diameter, 1m length
    branch1_channel = Channel(0.008, 0.5, 0.001, name="branch1")  # 8mm diameter, 0.5m length
    branch2_channel = Channel(0.008, 0.5, 0.001, name="branch2")  # 8mm diameter, 0.5m length
    
    network.connect_components(inlet, junction, main_channel)
    network.connect_components(junction, outlet1, branch1_channel)
    network.connect_components(junction, outlet2, branch2_channel)
    
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create solver
    solver = NetworkFlowSolver()
    
    # Test parameters
    total_flow_rate = 0.001  # 1 L/s
    temperature = 40.0  # °C
    inlet_pressure = 200000.0  # 2 bar
    outlet_pressure = 101325.0  # 1 bar
    
    print("1. Testing iterative path-based solver (with utils)...")
    try:
        flows1, info1 = solver.solve_network_flow(
            network, total_flow_rate, temperature, inlet_pressure
        )
        flows1_by_name = convert_flows_to_names(flows1, network)
        print(f"   ✓ Converged: {info1['converged']}")
        print(f"   ✓ Iterations: {info1['iterations']}")
        print(f"   ✓ Flow rates: {[f'{flows1_by_name[c]*1000:.3f}' for c in ['main', 'branch1', 'branch2']]} L/s")
        print(f"   ✓ Node pressures calculated: {len(info1['node_pressures'])} nodes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n2. Testing pump physics solver (with utils)...")
    try:
        flows2, info2 = solver.solve_network_flow_with_pump_physics(
            network, total_flow_rate, temperature, 
            pump_max_pressure=500000.0
        )
        flows2_by_name = convert_flows_to_names(flows2, network)
        print(f"   ✓ Converged: {info2['converged']}")
        print(f"   ✓ Iterations: {info2['iterations']}")
        print(f"   ✓ Pump adequate: {info2['pump_adequate']}")
        print(f"   ✓ Flow rates: {[f'{flows2_by_name[c]*1000:.3f}' for c in ['main', 'branch1', 'branch2']]} L/s")
        print(f"   ✓ Node pressures calculated: {len(info2['node_pressures'])} nodes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n3. Testing nodal solver (matrix-based)...")
    try:
        flows3, info3 = solver.solve_network_flow_nodal(
            network, total_flow_rate, temperature, inlet_pressure
        )
        flows3_by_name = convert_flows_to_names(flows3, network)
        print(f"   ✓ Converged: {info3['converged']}")
        print(f"   ✓ Flow rates: {[f'{flows3_by_name[c]*1000:.3f}' for c in ['main', 'branch1', 'branch2']]} L/s")
        print(f"   ✓ Node pressures calculated: {len(info3['node_pressures'])} nodes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n4. Testing legacy solver...")
    try:
        flows4, info4 = solver.solve_network_flow_legacy(
            network, total_flow_rate, temperature, inlet_pressure
        )
        flows4_by_name = convert_flows_to_names(flows4, network)
        print(f"   ✓ Converged: {info4['converged']}")
        print(f"   ✓ Iterations: {info4['iterations']}")
        print(f"   ✓ Flow rates: {[f'{flows4_by_name[c]*1000:.3f}' for c in ['main', 'branch1', 'branch2']]} L/s")
        print(f"   ✓ Node pressures calculated: {len(info4['node_pressures'])} nodes")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False
    
    print("\n5. Comparing solver results...")
    # Check that all solvers give reasonable results
    solvers_data = [
        ("Iterative", flows1_by_name, info1),
        ("Pump Physics", flows2_by_name, info2),
        ("Nodal", flows3_by_name, info3),
        ("Legacy", flows4_by_name, info4)
    ]
    
    # Check flow conservation
    for name, flows, info in solvers_data:
        main_flow = flows['main'] * 1000  # Convert to L/s
        branch_sum = (flows['branch1'] + flows['branch2']) * 1000
        conservation_error = abs(main_flow - branch_sum)
        print(f"   {name}: Main={main_flow:.3f} L/s, Branches={branch_sum:.3f} L/s, Error={conservation_error:.6f} L/s")
        
        if conservation_error > 0.001:  # 1 mL/s tolerance
            print(f"   ✗ Flow conservation violated for {name}")
            return False
    
    print("\n" + "=" * 50)
    print("✓ All solver methods work correctly with utils module!")
    print("✓ Flow conservation maintained across all solvers")
    print("✓ Node pressure calculations successful")
    return True

if __name__ == "__main__":
    success = test_utils_integration()
    if success:
        print("\n🎉 Utils integration test PASSED!")
    else:
        print("\n❌ Utils integration test FAILED!")