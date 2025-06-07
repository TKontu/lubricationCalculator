#!/usr/bin/env python3
"""
Debug the solver error
"""

from lubrication_flow_package import FlowNetwork, Node, Connection, Channel
from lubrication_flow_package.solvers import NetworkFlowSolver
import traceback

def debug_solver():
    """Debug the solver error"""
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
    main_channel = Channel(0.01, 1.0, 0.001, "main")
    branch1_channel = Channel(0.008, 0.5, 0.001, "branch1")
    branch2_channel = Channel(0.008, 0.5, 0.001, "branch2")
    
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
    
    try:
        flows, info = solver.solve_network_flow(
            network, total_flow_rate, temperature, inlet_pressure
        )
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_solver()