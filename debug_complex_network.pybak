#!/usr/bin/env python3
"""
Debug script to analyze mass conservation issues in the complex network from the test.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver
from tests.test_solver_convergence import TestSolverConvergence

def debug_complex_network_mass_conservation():
    """Debug mass conservation in the complex network from the test."""
    
    # Create test instance to get fixtures
    test_instance = TestSolverConvergence()
    
    # Get fixtures
    basic_config = SimulationConfig(
        total_flow_rate=0.015,
        temperature=55.0,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0,
        oil_density=850.0,
        tolerance=1e-6,
        max_iterations=100
    )
    
    # Create complex network manually (based on conftest.py)
    from lubrication_flow_package.network.flow_network import FlowNetwork
    from lubrication_flow_package.network.node import Node
    from lubrication_flow_package.components.channel import Channel
    from lubrication_flow_package.components.connector import Connector
    from lubrication_flow_package.components.nozzle import Nozzle
    
    network = FlowNetwork("Complex Network")
    
    from lubrication_flow_package.network.node import NodeType
    
    # Create nodes
    inlet = Node(id="inlet", name="Inlet", node_type=NodeType.INLET)
    node_1 = Node(id="node_1", name="Node 1", node_type=NodeType.JUNCTION)
    node_2 = Node(id="node_2", name="Node 2", node_type=NodeType.JUNCTION)
    node_3 = Node(id="node_3", name="Node 3", node_type=NodeType.JUNCTION)
    node_4 = Node(id="node_4", name="Node 4", node_type=NodeType.JUNCTION)
    node_5 = Node(id="node_5", name="Node 5", node_type=NodeType.JUNCTION)
    node_6 = Node(id="node_6", name="Node 6", node_type=NodeType.JUNCTION)
    node_7 = Node(id="node_7", name="Node 7", node_type=NodeType.JUNCTION)
    node_8 = Node(id="node_8", name="Node 8", node_type=NodeType.JUNCTION)
    branch = Node(id="branch", name="Branch", node_type=NodeType.JUNCTION)
    outlet_1 = Node(id="outlet_1", name="Outlet 1", node_type=NodeType.OUTLET)
    outlet_2 = Node(id="outlet_2", name="Outlet 2", node_type=NodeType.OUTLET)
    outlet_3 = Node(id="outlet_3", name="Outlet 3", node_type=NodeType.OUTLET)
    
    # Add nodes
    for node in [inlet, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, branch, outlet_1, outlet_2, outlet_3]:
        network.add_node(node)
    
    # Create components and connections (simplified to channels only)
    network.connect_components(inlet, node_1, Channel(diameter=0.02, length=0.5, roughness=1e-5, component_id="ch1"))
    network.connect_components(node_1, node_2, Channel(diameter=0.015, length=0.3, roughness=1e-5, component_id="ch2"))
    network.connect_components(node_2, node_3, Channel(diameter=0.01, length=0.1, roughness=1e-5, component_id="ch3"))
    network.connect_components(node_3, node_4, Channel(diameter=0.012, length=0.4, roughness=1e-5, component_id="ch4"))
    network.connect_components(node_4, node_5, Channel(diameter=0.01, length=0.1, roughness=1e-5, component_id="ch5"))
    network.connect_components(node_5, node_6, Channel(diameter=0.008, length=0.2, roughness=1e-5, component_id="ch6"))
    network.connect_components(node_6, node_7, Channel(diameter=0.006, length=0.15, roughness=1e-5, component_id="ch7"))
    network.connect_components(node_7, node_8, Channel(diameter=0.005, length=0.1, roughness=1e-5, component_id="ch8"))
    network.connect_components(node_8, branch, Channel(diameter=0.01, length=0.1, roughness=1e-5, component_id="ch9"))
    network.connect_components(branch, outlet_1, Channel(diameter=0.008, length=0.05, roughness=1e-5, component_id="ch10"))
    network.connect_components(branch, outlet_2, Channel(diameter=0.006, length=0.08, roughness=1e-5, component_id="ch11"))
    network.connect_components(node_4, outlet_3, Channel(diameter=0.004, length=0.12, roughness=1e-5, component_id="ch12"))
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet_1)
    network.add_outlet(outlet_2)
    network.add_outlet(outlet_3)
    
    print("=== Complex Network Mass Conservation Debug ===")
    print(f"Total flow rate: {basic_config.total_flow_rate} m³/s")
    print(f"Network has {len(network.nodes)} nodes and {len(network.connections)} connections")
    
    # Solve
    solver = TreeSolver(basic_config)
    result = solver.solve(network)
    
    print(f"\nSolver converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    
    # Check mass conservation manually
    print("\n=== Mass Conservation Check ===")
    component_flows = result['component_flows']
    node_pressures = result['node_pressures']
    
    print("\nComponent flows:")
    for comp_id, flow in component_flows.items():
        print(f"  {comp_id}: {flow:.8f} m³/s")
    
    print("\nNode pressures:")
    for node_id, pressure in node_pressures.items():
        print(f"  {node_id}: {pressure:.1f} Pa")
    
    # Manual mass conservation check for intermediate nodes only
    print("\n=== Mass Conservation Check (Intermediate Nodes Only) ===")
    mass_conservation_errors = []
    
    for node_id, node in network.nodes.items():
        # Skip inlet and outlet nodes - they are boundary conditions
        if (node_id == network.inlet_node.id or 
            any(node_id == outlet.id for outlet in network.outlet_nodes)):
            continue
            
        net_flow = 0.0
        
        print(f"\nNode {node_id}:")
        
        for connection in network.connections:
            flow = component_flows[connection.component.id]
            
            if connection.from_node.id == node_id:
                net_flow -= flow  # Outgoing flow
                print(f"  Outgoing to {connection.to_node.id} via {connection.component.id}: -{flow:.8f}")
            elif connection.to_node.id == node_id:
                net_flow += flow  # Incoming flow
                print(f"  Incoming from {connection.from_node.id} via {connection.component.id}: +{flow:.8f}")
        
        mass_conservation_errors.append(abs(net_flow))
        
        print(f"  Net flow: {net_flow:.8f} m³/s")
        
        if abs(net_flow) > basic_config.tolerance:
            print(f"  *** MASS CONSERVATION VIOLATION at {node_id}: {net_flow:.8f} m³/s ***")
        else:
            print(f"  ✓ Mass conservation OK at {node_id}")
    
    max_mass_error = max(mass_conservation_errors)
    relative_error = max_mass_error / basic_config.total_flow_rate
    print(f"\nMaximum mass conservation error: {max_mass_error:.8f}")
    print(f"Relative error: {relative_error * 100:.4f}%")
    
    if relative_error >= 0.01:
        print(f"ISSUE: Mass conservation relative error too large: {relative_error * 100:.4f}%")
    else:
        print("✓ Mass conservation within acceptable limits")

if __name__ == "__main__":
    debug_complex_network_mass_conservation()