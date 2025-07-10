#!/usr/bin/env python3
"""
Debug script for simple two-node network
"""

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

# Constants
DENSITY = 900.0
VISCOSITY = 1e-3
Q_TOTAL = 1e-4

def main():
    # Create simple network
    net = FlowNetwork("debug_simple") 
    n0 = net.create_node(name="inlet", elevation=0.0)
    n1 = net.create_node(name="outlet", elevation=0.0)
    net.set_inlet(n0)
    net.add_outlet(n1)

    # Simple channel
    ch = Channel(diameter=0.02, length=1.0, component_id="ch0")
    net.connect_components(n0, n1, ch)

    # Solver
    solver = NodalMatrixSolver(oil_density=DENSITY, oil_type="VG220")
    solver.calculate_viscosity = lambda T: VISCOSITY
    
    print(f"Debugging simple network: Q_total = {Q_TOTAL}")
    print(f"Channel: D={ch.diameter}m, L={ch.length}m")
    
    # Test pressure drop calculation
    fluid_props = {'density': DENSITY, 'viscosity': VISCOSITY}
    dp_test = ch.calculate_pressure_drop(Q_TOTAL, fluid_props)
    print(f"Direct pressure drop for Q={Q_TOTAL}: {dp_test:.2f} Pa")
    
    # Test resistance calculation methods
    R_diff = solver._calculate_component_resistance(ch, fluid_props, Q_TOTAL)
    R_avg = dp_test / Q_TOTAL  # True average resistance
    print(f"Differential resistance: {R_diff:.2e}")
    print(f"Average resistance: {R_avg:.2e}")
    print(f"Ratio (diff/avg): {R_diff/R_avg:.3f}")
    
    # Solve
    try:
        pressures, flows = solver.solve_nodal_iterative(
            network=net,
            source_node_id=n0.id,
            sink_node_ids=[n1.id],
            Q_total=Q_TOTAL,
            fluid_properties=fluid_props
        )
        
        print(f"Solved flows: {flows}")
        print(f"Solved pressures: {pressures}")
        print(f"Flow error: {abs(flows[ch.id] - Q_TOTAL)}")
        
    except Exception as e:
        print(f"Solver failed: {e}")

if __name__ == "__main__":
    main()