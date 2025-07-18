#!/usr/bin/env python3
"""
Debug the solver residual calculation to understand the mass conservation issue.
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


def debug_solver_residual():
    """Debug the solver residual calculation."""
    
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
    
    print("=== Debugging Solver Residual ===")
    print(f"Total flow rate: {config.total_flow_rate} m³/s")
    print(f"Network has {len(network.nodes)} nodes and {len(network.connections)} connections")
    
    # Create solver
    solver = TreeSolver(config)
    
    # Get outlet nodes
    outlet_node_ids = [node.id for node in network.outlet_nodes]
    unknown_node_ids = [nid for nid in network.nodes if nid not in outlet_node_ids]
    
    print(f"\nOutlet nodes: {outlet_node_ids}")
    print(f"Unknown nodes: {unknown_node_ids}")
    
    # Solve the system
    result = solver.solve(network)
    
    print(f"\nSolver converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")
    
    # Get final solution
    final_pressures = result['node_pressures']
    component_flows = result['component_flows']
    
    print(f"\nFinal node pressures:")
    for node_id, pressure in final_pressures.items():
        print(f"  {node_id}: {pressure:.1f} Pa")
    
    print(f"\nComponent flows:")
    for comp_id, flow in component_flows.items():
        print(f"  {comp_id}: {flow:.8f} m³/s")
    
    # Extract final pressure array for unknown nodes
    final_pressure_array = np.array([final_pressures[nid] for nid in unknown_node_ids])
    
    # Calculate residual at final solution
    final_residual = solver._evaluate_residual(final_pressure_array, network, unknown_node_ids, outlet_node_ids)
    
    print(f"\nFinal residual vector: {final_residual}")
    print(f"Final residual norm: {np.linalg.norm(final_residual)}")
    
    # Now check mass conservation manually
    print(f"\n=== Manual Mass Conservation Check ===")
    
    for node_id in unknown_node_ids:
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
        
        # Add external flow constraint for inlet
        if node_id == network.inlet_node.id:
            net_flow += config.total_flow_rate
            print(f"  Inlet flow: +{config.total_flow_rate:.8f}")
        
        print(f"  Net flow (manual): {net_flow:.8f}")
        print(f"  Solver residual: {final_residual[unknown_node_ids.index(node_id)]:.8f}")
        print(f"  Match: {abs(net_flow - final_residual[unknown_node_ids.index(node_id)]) < 1e-10}")
    
    print(f"\n=== Problem Analysis ===")
    print(f"The solver finds a solution where the residual norm is {np.linalg.norm(final_residual):.2e}")
    print(f"This is within the solver tolerance of {config.tolerance:.2e}")
    print(f"However, the actual mass conservation violations are:")
    
    mass_conservation_errors = []
    for node_id in network.nodes:
        if (node_id == network.inlet_node.id or 
            any(node_id == outlet.id for outlet in network.outlet_nodes)):
            continue
        
        net_flow = 0.0
        for connection in network.connections:
            flow = component_flows[connection.component.id]
            if connection.from_node.id == node_id:
                net_flow -= flow
            elif connection.to_node.id == node_id:
                net_flow += flow
        
        if abs(net_flow) > config.tolerance:
            mass_conservation_errors.append(abs(net_flow))
            print(f"  Node {node_id}: {net_flow:.8f} m³/s")
    
    if mass_conservation_errors:
        max_error = max(mass_conservation_errors)
        relative_error = max_error / config.total_flow_rate
        print(f"\nMaximum mass conservation error: {max_error:.8f}")
        print(f"Relative error: {relative_error * 100:.4f}%")


if __name__ == "__main__":
    debug_solver_residual()