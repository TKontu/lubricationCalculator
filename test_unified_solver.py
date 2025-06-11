#!/usr/bin/env python3
"""
Test script for the unified nodal solver
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver
from typing import Dict


class LinearResistanceComponent(FlowComponent):
    """Simple component with linear resistance for testing"""
    
    def __init__(self, resistance: float, component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.resistance = resistance  # Pa·s/m³
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Linear pressure drop: ΔP = R * Q"""
        return self.resistance * abs(flow_rate)
    
    def get_flow_area(self) -> float:
        return 1e-4  # 1 cm²


def test_unified_solver():
    """Test the unified nodal solver"""
    print("Testing Unified Nodal Solver")
    print("=" * 50)
    
    # Create a simple network: Source -> R1 -> Junction -> R2 -> Sink
    #                                           |
    #                                           R3 -> Outlet2
    
    source = Node("source", "Source", elevation=0.0)
    junction = Node("junction", "Junction", elevation=0.0)
    sink = Node("sink", "Sink", elevation=0.0)
    outlet2 = Node("outlet2", "Outlet2", elevation=0.0)
    
    # Create components
    R1 = 1000.0  # Pa·s/m³
    R2 = 2000.0
    R3 = 3000.0
    
    comp1 = LinearResistanceComponent(R1, "R1", "Resistor 1")
    comp2 = LinearResistanceComponent(R2, "R2", "Resistor 2")
    comp3 = LinearResistanceComponent(R3, "R3", "Resistor 3")
    
    # Build network
    network = FlowNetwork("test_network")
    network.add_node(source)
    network.add_node(junction)
    network.add_node(sink)
    network.add_node(outlet2)
    
    network.connect_components(source, junction, comp1)
    network.connect_components(junction, sink, comp2)
    network.connect_components(junction, outlet2, comp3)
    
    network.set_inlet(source)
    network.add_outlet(sink)
    network.add_outlet(outlet2)
    
    # Test parameters
    total_flow_rate = 0.001  # 1 L/s
    temperature = 80.0  # °C
    inlet_pressure = 200000.0  # 200 kPa
    outlet_pressure = 101325.0  # 101.325 kPa
    
    print(f"Network: {network.name}")
    print(f"Total flow rate: {total_flow_rate * 1000:.1f} L/s")
    print(f"Temperature: {temperature}°C")
    print(f"Inlet pressure: {inlet_pressure / 1000:.1f} kPa")
    print(f"Outlet pressure: {outlet_pressure / 1000:.1f} kPa")
    print()
    
    # Test 1: New unified solver
    print("1. Testing Unified NodalMatrixSolver.solve_nodal_network()")
    print("-" * 50)
    
    solver = NodalMatrixSolver()
    connection_flows, solution_info = solver.solve_nodal_network(
        network=network,
        total_flow_rate=total_flow_rate,
        temperature=temperature,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure
    )
    
    print("Results:")
    print(f"  Converged: {solution_info['converged']}")
    print(f"  Iterations: {solution_info['iterations']}")
    print(f"  Viscosity: {solution_info['viscosity']:.6f} Pa·s")
    print()
    print("  Node Pressures:")
    for node_id, pressure in solution_info['node_pressures'].items():
        node_name = network.nodes[node_id].name
        print(f"    {node_name}: {pressure / 1000:.3f} kPa")
    print()
    print("  Connection Flows:")
    for conn_id, flow in connection_flows.items():
        print(f"    {conn_id}: {flow * 1000:.6f} L/s")
    print()
    print("  Pressure Drops:")
    for conn_id, dp in solution_info['pressure_drops'].items():
        print(f"    {conn_id}: {dp / 1000:.3f} kPa")
    print()
    
    # Test 2: Deprecated method (should show warning)
    print("2. Testing Deprecated NetworkFlowSolver.solve_network_flow_nodal()")
    print("-" * 50)
    
    old_solver = NetworkFlowSolver()
    try:
        old_connection_flows, old_solution_info = old_solver.solve_network_flow_nodal(
            network=network,
            total_flow_rate=total_flow_rate,
            temperature=temperature,
            inlet_pressure=inlet_pressure,
            outlet_pressure=outlet_pressure
        )
        
        print("Results from deprecated method:")
        print(f"  Converged: {old_solution_info['converged']}")
        print(f"  Node Pressures match: {solution_info['node_pressures'] == old_solution_info['node_pressures']}")
        print(f"  Connection Flows match: {connection_flows == old_connection_flows}")
        print()
        
    except Exception as e:
        print(f"Error with deprecated method: {e}")
        print()
    
    # Test 3: Single outlet case (should use iterative solver)
    print("3. Testing Single Outlet Case (uses iterative solver)")
    print("-" * 50)
    
    # Create simple two-node network
    simple_network = FlowNetwork("simple_network")
    source_simple = Node("source", "Source", elevation=0.0)
    sink_simple = Node("sink", "Sink", elevation=0.0)
    
    simple_network.add_node(source_simple)
    simple_network.add_node(sink_simple)
    
    comp_simple = LinearResistanceComponent(1000.0, "R_simple", "Simple Resistor")
    simple_network.connect_components(source_simple, sink_simple, comp_simple)
    simple_network.set_inlet(source_simple)
    simple_network.add_outlet(sink_simple)
    
    simple_flows, simple_info = solver.solve_nodal_network(
        network=simple_network,
        total_flow_rate=total_flow_rate,
        temperature=temperature,
        inlet_pressure=inlet_pressure,
        outlet_pressure=outlet_pressure
    )
    
    print("Simple network results:")
    print(f"  Flow: {simple_flows['R_simple'] * 1000:.6f} L/s")
    print(f"  Source pressure: {simple_info['node_pressures']['source'] / 1000:.3f} kPa")
    print(f"  Sink pressure: {simple_info['node_pressures']['sink'] / 1000:.3f} kPa")
    print(f"  Pressure drop: {simple_info['pressure_drops']['R_simple'] / 1000:.3f} kPa")
    print()
    
    print("✅ All tests completed successfully!")


if __name__ == "__main__":
    test_unified_solver()