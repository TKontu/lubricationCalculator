#!/usr/bin/env python3
"""
Debug script to analyze mass conservation issues in the TreeSolver.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.network.connection import Connection

def create_debug_network():
    """Create a simple debug network to test mass conservation."""
    # Create nodes
    inlet = Node(id="inlet", name="Inlet", node_type="inlet")
    branch = Node(id="branch", name="Branch", node_type="junction")
    outlet1 = Node(id="outlet1", name="Outlet1", node_type="outlet")
    outlet2 = Node(id="outlet2", name="Outlet2", node_type="outlet")
    
    # Create components
    channel1 = Channel(diameter=0.01, length=0.1, roughness=1e-6, component_id="ch1")
    channel2 = Channel(diameter=0.008, length=0.05, roughness=1e-6, component_id="ch2")
    channel3 = Channel(diameter=0.008, length=0.05, roughness=1e-6, component_id="ch3")
    
    # Create network
    network = FlowNetwork()
    network.add_node(inlet)
    network.add_node(branch)
    network.add_node(outlet1)
    network.add_node(outlet2)
    
    network.connect_components(inlet, branch, channel1)
    network.connect_components(branch, outlet1, channel2)
    network.connect_components(branch, outlet2, channel3)
    
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    return network

def debug_mass_conservation():
    """Debug the mass conservation issue."""
    # Create configuration
    config = SimulationConfig(
        total_flow_rate=0.001,  # 1 liter/minute
        temperature=55.0,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0,
        tolerance=1e-6,
        max_iterations=100
    )
    
    # Create network
    network = create_debug_network()
    
    # Create solver
    solver = TreeSolver(config)
    
    print("=== Debug Network Mass Conservation ===")
    print(f"Total flow rate: {config.total_flow_rate} m³/s")
    print(f"Network has {len(network.nodes)} nodes and {len(network.connections)} connections")
    
    # Solve
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
    
    # Manual mass conservation check
    print("\n=== Manual Mass Conservation Check ===")
    for node_id in network.nodes:
        net_flow = 0.0
        
        for conn in network.connections:
            flow = component_flows[conn.component.id]
            
            if conn.from_node.id == node_id:
                net_flow -= flow  # Outgoing flow
                print(f"  Node {node_id}: Outgoing flow to {conn.to_node.id}: -{flow:.8f}")
            elif conn.to_node.id == node_id:
                net_flow += flow  # Incoming flow
                print(f"  Node {node_id}: Incoming flow from {conn.from_node.id}: +{flow:.8f}")
        
        # Add inlet flow
        if node_id == network.inlet_node.id:
            net_flow += config.total_flow_rate
            print(f"  Node {node_id}: Inlet flow: +{config.total_flow_rate:.8f}")
        
        print(f"  Node {node_id}: Net flow = {net_flow:.8f} m³/s")
        
        if abs(net_flow) > config.tolerance:
            print(f"  *** MASS CONSERVATION VIOLATION at {node_id}: {net_flow:.8f} m³/s ***")
        else:
            print(f"  ✓ Mass conservation OK at {node_id}")
        print()

def debug_residual_evaluation():
    """Debug the residual evaluation specifically."""
    config = SimulationConfig(
        total_flow_rate=0.001,
        temperature=55.0,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0,
        tolerance=1e-6,
        max_iterations=100
    )
    
    network = create_debug_network()
    solver = TreeSolver(config)
    
    print("=== Debug Residual Evaluation ===")
    
    # Get outlet nodes
    outlet_node_ids = [node.id for node in network.outlet_nodes]
    unknown_node_ids = [nid for nid in network.nodes if nid not in outlet_node_ids]
    
    print(f"Outlet nodes: {outlet_node_ids}")
    print(f"Unknown nodes: {unknown_node_ids}")
    
    # Test with a simple pressure distribution
    test_pressures = np.array([180000.0, 160000.0])  # inlet and branch pressures
    
    print(f"\nTest pressures: {test_pressures}")
    
    # Evaluate residual
    residual = solver._evaluate_residual(test_pressures, network, unknown_node_ids, outlet_node_ids)
    
    print(f"Residual vector: {residual}")
    print(f"Residual norm: {np.linalg.norm(residual)}")
    
    # Manual calculation
    print("\n=== Manual Residual Calculation ===")
    full_pressures = {nid: p for nid, p in zip(unknown_node_ids, test_pressures)}
    full_pressures.update({outlet_id: config.outlet_pressure for outlet_id in outlet_node_ids})
    
    print(f"Full pressure map: {full_pressures}")
    
    for i, node_id in enumerate(unknown_node_ids):
        net_flow = 0
        print(f"\nNode {node_id}:")
        
        for conn in network.connections:
            if conn.from_node.id == node_id:
                p_other = full_pressures[conn.to_node.id]
                pressure_drop = full_pressures[node_id] - p_other
                flow = conn.component.calculate_flow_rate(pressure_drop, solver.fluid_properties)
                net_flow -= flow
                print(f"  Outgoing to {conn.to_node.id}: dp={pressure_drop:.1f}, flow={flow:.8f}")
            elif conn.to_node.id == node_id:
                p_other = full_pressures[conn.from_node.id]
                pressure_drop = p_other - full_pressures[node_id]
                flow = conn.component.calculate_flow_rate(pressure_drop, solver.fluid_properties)
                net_flow += flow
                print(f"  Incoming from {conn.from_node.id}: dp={pressure_drop:.1f}, flow={flow:.8f}")
        
        # Add external flow constraint for inlet
        if node_id == network.inlet_node.id:
            net_flow += config.total_flow_rate
            print(f"  Inlet flow: +{config.total_flow_rate:.8f}")
        
        print(f"  Net flow (residual): {net_flow:.8f}")
        print(f"  Solver residual[{i}]: {residual[i]:.8f}")
        print(f"  Match: {abs(net_flow - residual[i]) < 1e-10}")

if __name__ == "__main__":
    debug_mass_conservation()
    print("\n" + "="*50 + "\n")
    debug_residual_evaluation()