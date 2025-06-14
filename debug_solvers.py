#!/usr/bin/env python3
"""
Debug script to understand solver interfaces and outputs
"""

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

def debug_single_pipe():
    """Debug a simple single pipe network"""
    print("=== Debugging Single Pipe Network ===")
    
    # Create simple network
    network = FlowNetwork("Debug Single Pipe")
    inlet = network.create_node("Inlet")
    outlet = network.create_node("Outlet")
    
    pipe = Channel(diameter=0.020, length=10.0, roughness=0.00015)  # 20mm, 10m
    network.connect_components(inlet, outlet, pipe)
    
    network.set_inlet(inlet)
    network.add_outlet(outlet)
    
    # Test parameters
    flow_lpm = 100.0
    flow_m3s = flow_lpm / 60000.0
    temperature = 40.0
    
    print(f"Test conditions: {flow_lpm} L/min, {pipe.diameter*1000:.0f}mm diameter, {pipe.length}m length")
    print(f"Flow rate: {flow_m3s:.6f} m³/s")
    
    # Test NetworkFlowSolver
    print("\n--- NetworkFlowSolver ---")
    network_solver = NetworkFlowSolver()
    try:
        flows, info = network_solver.solve_network_flow(
            network=network,
            total_flow_rate=flow_m3s,
            temperature=temperature,
            inlet_pressure=200000.0
        )
        print("SUCCESS")
        print(f"Flows: {flows}")
        print(f"Info keys: {list(info.keys())}")
        if 'node_pressures' in info:
            print(f"Node pressures: {info['node_pressures']}")
        if 'pressure_drops' in info:
            print(f"Pressure drops: {info['pressure_drops']}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test NodalMatrixSolver
    print("\n--- NodalMatrixSolver ---")
    nodal_solver = NodalMatrixSolver()
    try:
        pressures, flows = nodal_solver.solve_nodal_iterative(
            network=network,
            source_node_id=inlet.id,
            sink_node_id=outlet.id,
            Q_total=flow_m3s,
            fluid_properties={'density': 850.0, 'viscosity': 0.032}
        )
        print("SUCCESS")
        print(f"Pressures: {pressures}")
        print(f"Flows: {flows}")
    except Exception as e:
        print(f"FAILED: {e}")
    
    # Test analytical calculation
    print("\n--- Analytical Calculation ---")
    fluid_props = {'density': 850.0, 'viscosity': 0.032}
    analytical_dp = pipe.calculate_pressure_drop(flow_m3s, fluid_props)
    print(f"Analytical pressure drop: {analytical_dp:.2f} Pa ({analytical_dp/100000:.3f} bar)")

if __name__ == "__main__":
    debug_single_pipe()