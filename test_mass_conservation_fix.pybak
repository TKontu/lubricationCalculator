#!/usr/bin/env python3
"""
Test script to verify mass conservation fix.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node, NodeType
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver


def create_test_network():
    """Create the exact network from the failing test."""
    network = FlowNetwork()
    
    # Create nodes (fixing the constructor)
    nodes = {}
    for i in range(10):
        node = Node(id=f"node_{i}", name=f"Node {i}", elevation=i * 0.1, node_type=NodeType.JUNCTION)
        nodes[f"node_{i}"] = node
        network.add_node(node)
    
    # Create connections with varying resistances
    connections = []
    for i in range(9):
        from_node = nodes[f"node_{i}"]
        to_node = nodes[f"node_{i+1}"]
        
        # Alternate between channels and nozzles
        if i % 2 == 0:
            component = Channel(diameter=0.005 + i * 0.001, length=0.1, roughness=0.0001, component_id=f"channel_{i}")
        else:
            component = Nozzle(diameter=0.003 + i * 0.0005, component_id=f"nozzle_{i}")
        
        connection = network.connect_components(from_node, to_node, component)
        connections.append(connection)
    
    # Add branching
    branch_node = Node(id="branch", name="Branch", elevation=0.1, node_type=NodeType.JUNCTION)
    network.add_node(branch_node)
    
    branch_outlet = Node(id="branch_outlet", name="Branch Outlet", elevation=0.2, node_type=NodeType.OUTLET)
    network.add_node(branch_outlet)
    
    # Connect branch
    branch_channel = Channel(diameter=0.008, length=0.1, roughness=0.0001, component_id="branch_channel")
    branch_connection = network.connect_components(nodes["node_5"], branch_node, branch_channel)
    
    branch_nozzle = Nozzle(diameter=0.004, component_id="branch_nozzle")
    branch_outlet_connection = network.connect_components(branch_node, branch_outlet, branch_nozzle)
    
    # Set inlet and outlets
    network.set_inlet(nodes["node_0"])
    network.add_outlet(nodes["node_9"])
    network.add_outlet(branch_outlet)
    
    return network


def test_mass_conservation_fix():
    """Test the mass conservation fix."""
    
    # Create configuration
    config = SimulationConfig(
        total_flow_rate=0.015,
        temperature=55.0,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0,
        oil_density=850.0,
        tolerance=1e-6,
        max_iterations=100
    )
    
    # Create network
    network = create_test_network()
    
    print("=== Testing Mass Conservation Fix ===")
    print(f"Total flow rate: {config.total_flow_rate} m³/s")
    print(f"Network has {len(network.nodes)} nodes and {len(network.connections)} connections")
    
    # Solve
    solver = TreeSolver(config)
    result = solver.solve(network)
    
    print(f"\nSolver converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    
    # Check mass conservation
    component_flows = result['component_flows']
    node_pressures = result['node_pressures']
    
    print("\n=== Mass Conservation Check (Intermediate Nodes Only) ===")
    mass_conservation_errors = []
    
    for node_id, node in network.nodes.items():
        # Skip inlet and outlet nodes - they are boundary conditions
        if (node_id == network.inlet_node.id or 
            any(node_id == outlet.id for outlet in network.outlet_nodes)):
            continue
        
        net_flow = 0.0
        
        for connection in network.connections:
            flow = component_flows[connection.component.id]
            
            if connection.from_node.id == node_id:
                net_flow -= flow  # Outgoing flow
            elif connection.to_node.id == node_id:
                net_flow += flow  # Incoming flow
        
        mass_conservation_errors.append(abs(net_flow))
        
        if abs(net_flow) > config.tolerance:
            print(f"Mass conservation violation at node {node_id}: {net_flow:.8f}")
    
    if mass_conservation_errors:
        max_mass_error = max(mass_conservation_errors)
        relative_error = max_mass_error / config.total_flow_rate
        print(f"\nMaximum mass conservation error: {max_mass_error:.8f}")
        print(f"Relative error: {relative_error * 100:.4f}%")
        
        if relative_error >= 0.01:
            print(f"ISSUE: Mass conservation relative error too large: {relative_error * 100:.4f}%")
            return False
        else:
            print("✓ Mass conservation within acceptable limits")
            return True
    else:
        print("✓ No intermediate nodes to check")
        return True


if __name__ == "__main__":
    success = test_mass_conservation_fix()
    if success:
        print("\n🎉 Mass conservation test PASSED!")
    else:
        print("\n❌ Mass conservation test FAILED!")
        sys.exit(1)