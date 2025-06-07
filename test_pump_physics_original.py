#!/usr/bin/env python3
"""
Test the pump physics solver before and after refactoring to see if flow conservation
was an existing issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lubrication_flow_package'))

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver

def create_test_network():
    """Create a simple branching network for testing"""
    network = FlowNetwork("Test Network")
    
    # Create nodes
    inlet = network.create_node("Inlet", elevation=0.0)
    junction = network.create_node("Junction", elevation=0.5)
    outlet1 = network.create_node("Outlet1", elevation=1.0)
    outlet2 = network.create_node("Outlet2", elevation=1.0)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create components
    main_channel = Channel(diameter=0.05, length=5.0, name="main")
    branch1 = Channel(diameter=0.03, length=3.0, name="branch1")
    branch2 = Channel(diameter=0.03, length=3.0, name="branch2")
    
    # Connect components
    network.connect_components(inlet, junction, main_channel)
    network.connect_components(junction, outlet1, branch1)
    network.connect_components(junction, outlet2, branch2)
    
    return network

def main():
    print("TESTING PUMP PHYSICS SOLVER FLOW CONSERVATION")
    print("=" * 60)
    
    # Create test network
    network = create_test_network()
    solver = NetworkFlowSolver()
    
    # Test parameters
    total_flow_rate = 0.001  # 1 L/s
    temperature = 40.0  # °C
    inlet_pressure = 200000.0  # 2 bar
    
    print("Testing pump physics solver...")
    print(f"   Input: flow_rate={total_flow_rate*1000:.3f} L/s, temp={temperature}°C")
    try:
        flows, info = solver.solve_network_flow_with_pump_physics(
            network, total_flow_rate, temperature, 
            pump_max_pressure=500000.0, outlet_pressure=101325.0
        )
        
        print(f"   ✓ Converged: {info['converged']}")
        print(f"   ✓ Iterations: {info['iterations']}")
        print(f"   ✓ Pump adequate: {info.get('pump_adequate', 'N/A')}")
        
        # Check flow conservation
        print(f"   Available flow keys: {list(flows.keys())}")
        # Get flows by component ID (since that's what the solver returns)
        flow_values = list(flows.values())
        if len(flow_values) >= 3:
            main_flow = flow_values[0]  # First component (main channel)
            branch1_flow = flow_values[1]  # Second component (branch1)
            branch2_flow = flow_values[2]  # Third component (branch2)
        else:
            main_flow = branch1_flow = branch2_flow = 0.0
        
        total_branch_flow = branch1_flow + branch2_flow
        flow_error = abs(main_flow - total_branch_flow)
        
        print(f"   Flow rates:")
        print(f"     Main: {main_flow*1000:.3f} L/s")
        print(f"     Branch1: {branch1_flow*1000:.3f} L/s")
        print(f"     Branch2: {branch2_flow*1000:.3f} L/s")
        print(f"     Total branches: {total_branch_flow*1000:.3f} L/s")
        print(f"     Flow error: {flow_error*1000:.6f} L/s")
        
        if flow_error < 1e-6:
            print("   ✓ Flow conservation satisfied")
        else:
            print("   ✗ Flow conservation violated")
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()