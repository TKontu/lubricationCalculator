#!/usr/bin/env python3
"""
Debug conductance calculation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
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


def debug_conductance():
    """Debug conductance calculation"""
    
    # Create simple network: Source -> R1 -> Junction -> R2 -> Sink
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
    
    solver = NodalMatrixSolver()
    viscosity = solver.calculate_viscosity(temperature)
    fluid_properties = {
        'density': solver.oil_density,
        'viscosity': viscosity
    }
    
    print("Debug Conductance Calculation")
    print("=" * 50)
    print(f"Total flow rate: {total_flow_rate * 1000:.1f} L/s")
    print(f"Temperature: {temperature}°C")
    print(f"Viscosity: {viscosity:.6f} Pa·s")
    print(f"Number of outlets: {len(network.outlet_nodes)}")
    print()
    
    # Calculate conductances manually
    print("Manual conductance calculation:")
    for conn in network.connections:
        Q0 = total_flow_rate / max(1, len(network.outlet_nodes))
        print(f"  Connection {conn.component.id}: Q0 = {Q0 * 1000:.6f} L/s")
        
        # Calculate resistance using the same method as the solver
        R = solver._calculate_connection_resistance(conn, fluid_properties, Q0)
        G = 1.0 / R if R > 0 else 1e12
        
        print(f"    Resistance: {R:.3f} Pa·s/m³")
        print(f"    Conductance: {G:.6e} m³/(Pa·s)")
        
        # Check pressure drop at this flow
        dp = conn.component.calculate_pressure_drop(Q0, fluid_properties)
        print(f"    Pressure drop at Q0: {dp:.6f} Pa")
        print(f"    Expected resistance: {dp/Q0:.3f} Pa·s/m³")
        print()
    
    # Test the actual solver
    print("Solver results:")
    connection_flows, solution_info = solver.solve_nodal_network(
        network=network,
        total_flow_rate=total_flow_rate,
        temperature=temperature,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0
    )
    
    print("  Connection Flows:")
    for conn_id, flow in connection_flows.items():
        print(f"    {conn_id}: {flow * 1000:.6f} L/s")
    
    print("  Node Pressures:")
    for node_id, pressure in solution_info['node_pressures'].items():
        node_name = network.nodes[node_id].name
        print(f"    {node_name}: {pressure / 1000:.3f} kPa")


if __name__ == "__main__":
    debug_conductance()