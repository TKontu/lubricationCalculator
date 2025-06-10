"""
Debug script to understand the nodal matrix solver behavior
"""

import numpy as np
import logging
from lubrication_flow_package.network.flow_network import FlowNetwork
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

def debug_parallel_network():
    """Debug the parallel network case"""
    print("=" * 60)
    print("DEBUGGING PARALLEL NETWORK")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    solver = NodalMatrixSolver()
    
    # Create network: Source splits into two parallel paths to Sink
    network = FlowNetwork("Parallel Network Debug")
    
    source = network.create_node("Source")
    sink = network.create_node("Sink")
    
    R1 = 1000.0
    R2 = 2000.0
    
    comp1 = LinearResistanceComponent(R1, "comp1", "R1")
    comp2 = LinearResistanceComponent(R2, "comp2", "R2")
    
    network.connect_components(source, sink, comp1)
    network.connect_components(source, sink, comp2)
    
    network.set_inlet(source)
    network.add_outlet(sink)
    
    # Debug: Check network connections
    print(f"\nNetwork connections:")
    for i, conn in enumerate(network.connections):
        print(f"  Connection {i}: {conn.from_node.id} -> {conn.to_node.id} via {conn.component.id} ({conn.component.name})")
    
    Q_total = 0.001  # 1 L/s
    
    fluid_properties = {
        'density': 900.0,
        'viscosity': 0.01
    }
    
    print(f"Network setup:")
    print(f"  Source: {source.id}")
    print(f"  Sink: {sink.id}")
    print(f"  R1 = {R1} Pa·s/m³")
    print(f"  R2 = {R2} Pa·s/m³")
    print(f"  Q_total = {Q_total} m³/s")
    
    # Analytical solution
    R_parallel = 1.0 / (1.0/R1 + 1.0/R2)  # = 666.67
    P_source_expected = Q_total * R_parallel
    Q1_expected = P_source_expected / R1
    Q2_expected = P_source_expected / R2
    
    print(f"\nAnalytical solution:")
    print(f"  R_parallel = {R_parallel:.2f} Pa·s/m³")
    print(f"  P_source_expected = {P_source_expected:.6f} Pa")
    print(f"  Q1_expected = {Q1_expected:.6f} m³/s")
    print(f"  Q2_expected = {Q2_expected:.6f} m³/s")
    print(f"  Q1 + Q2 = {Q1_expected + Q2_expected:.6f} m³/s")
    
    # Solve using nodal matrix solver
    pressures, flows = solver.solve_nodal_iterative(
        network=network,
        source_node_id=source.id,
        sink_node_id=sink.id,
        Q_total=Q_total,
        fluid_properties=fluid_properties
    )
    
    print(f"\nSolver results:")
    print(f"  P_source = {pressures[source.id]:.6f} Pa")
    print(f"  P_sink = {pressures[sink.id]:.6f} Pa")
    print(f"  Available flows: {list(flows.keys())}")
    if 'comp1' in flows:
        print(f"  Q1 = {flows['comp1']:.6f} m³/s")
    if 'comp2' in flows:
        print(f"  Q2 = {flows['comp2']:.6f} m³/s")
        print(f"  Q1 + Q2 = {flows['comp1'] + flows['comp2']:.6f} m³/s")
    else:
        print(f"  ERROR: comp2 not found in flows!")
    
    print(f"\nErrors:")
    print(f"  P_source error = {abs(pressures[source.id] - P_source_expected):.6f} Pa")
    print(f"  Q1 error = {abs(flows['comp1'] - Q1_expected):.6f} m³/s")
    print(f"  Q2 error = {abs(flows['comp2'] - Q2_expected):.6f} m³/s")

def debug_y_network():
    """Debug the Y-network case"""
    print("\n" + "=" * 60)
    print("DEBUGGING Y-NETWORK")
    print("=" * 60)
    
    solver = NodalMatrixSolver()
    
    # Create network
    network = FlowNetwork("Y-Network Debug")
    
    # Create nodes
    source = network.create_node("Source")
    junction = network.create_node("Junction") 
    sink1 = network.create_node("Sink1")
    sink2 = network.create_node("Sink2")
    
    # Define resistances (Pa·s/m³)
    R1 = 1000.0
    R2 = 2000.0
    R3 = 3000.0
    
    # Create components with linear resistances
    comp1 = LinearResistanceComponent(R1, "comp1", "R1")
    comp2 = LinearResistanceComponent(R2, "comp2", "R2") 
    comp3 = LinearResistanceComponent(R3, "comp3", "R3")
    
    # Connect components
    network.connect_components(source, junction, comp1)
    network.connect_components(junction, sink1, comp2)
    network.connect_components(junction, sink2, comp3)
    
    # Set inlet and outlets
    network.set_inlet(source)
    network.add_outlet(sink1)
    network.add_outlet(sink2)
    
    Q_total = 0.001  # 1 L/s
    
    fluid_properties = {
        'density': 900.0,
        'viscosity': 0.01
    }
    
    print(f"Network setup:")
    print(f"  Source: {source.id}")
    print(f"  Junction: {junction.id}")
    print(f"  Sink1: {sink1.id}")
    print(f"  Sink2: {sink2.id}")
    print(f"  R1 = {R1} Pa·s/m³")
    print(f"  R2 = {R2} Pa·s/m³")
    print(f"  R3 = {R3} Pa·s/m³")
    print(f"  Q_total = {Q_total} m³/s")
    
    # Analytical solution
    R_parallel = 1.0 / (1.0/R2 + 1.0/R3)  # = 1200
    R_total = R1 + R_parallel  # = 2200
    
    Q1_expected = Q_total
    P_junction_expected = Q1_expected * R_parallel
    P_source_expected = Q1_expected * R_total
    Q2_expected = P_junction_expected / R2
    Q3_expected = P_junction_expected / R3
    
    print(f"\nAnalytical solution:")
    print(f"  R_parallel = {R_parallel:.2f} Pa·s/m³")
    print(f"  R_total = {R_total:.2f} Pa·s/m³")
    print(f"  P_source_expected = {P_source_expected:.6f} Pa")
    print(f"  P_junction_expected = {P_junction_expected:.6f} Pa")
    print(f"  Q1_expected = {Q1_expected:.6f} m³/s")
    print(f"  Q2_expected = {Q2_expected:.6f} m³/s")
    print(f"  Q3_expected = {Q3_expected:.6f} m³/s")
    print(f"  Q2 + Q3 = {Q2_expected + Q3_expected:.6f} m³/s")
    
    # Solve using nodal matrix solver
    pressures, flows = solver.solve_nodal_iterative(
        network=network,
        source_node_id=source.id,
        sink_node_id=sink1.id,  # Use sink1 as reference
        Q_total=Q_total,
        fluid_properties=fluid_properties
    )
    
    print(f"\nSolver results:")
    print(f"  P_source = {pressures[source.id]:.6f} Pa")
    print(f"  P_junction = {pressures[junction.id]:.6f} Pa")
    print(f"  P_sink1 = {pressures[sink1.id]:.6f} Pa")
    print(f"  P_sink2 = {pressures[sink2.id]:.6f} Pa")
    print(f"  Q1 = {flows['comp1']:.6f} m³/s")
    print(f"  Q2 = {flows['comp2']:.6f} m³/s")
    print(f"  Q3 = {flows['comp3']:.6f} m³/s")
    print(f"  Q2 + Q3 = {flows['comp2'] + flows['comp3']:.6f} m³/s")
    
    print(f"\nErrors:")
    print(f"  P_source error = {abs(pressures[source.id] - P_source_expected):.6f} Pa")
    print(f"  P_junction error = {abs(pressures[junction.id] - P_junction_expected):.6f} Pa")
    print(f"  Q1 error = {abs(flows['comp1'] - Q1_expected):.6f} m³/s")
    print(f"  Q2 error = {abs(flows['comp2'] - Q2_expected):.6f} m³/s")
    print(f"  Q3 error = {abs(flows['comp3'] - Q3_expected):.6f} m³/s")

if __name__ == "__main__":
    debug_parallel_network()
    debug_y_network()