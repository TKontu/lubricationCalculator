#!/usr/bin/env python3
"""
Network-Based Lubrication Flow Distribution Calculator (Refactored)

This is the refactored version using the multi-module package structure.
It maintains the same functionality as the original single script.
"""

# Import everything from the refactored package to maintain backward compatibility
from lubrication_flow_package import *

if __name__ == "__main__":
    from lubrication_flow_package.cli.main import main as cli_main
    from lubrication_flow_package.cli.demos import create_simple_tree_example
    
    # Existing setup from main()
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    network, total_flow_rate, temperature = create_simple_tree_example()

    print("\n" + "="*70)
    print("APPROACH 3: MATRIX-BASED NODAL SOLVER (SCALABLE)")
    print("="*70)
    print("✓ Single linear solve for nodal pressures")
    print("✓ Direct flow via conductance" )

    # Solve using unified nodal method
    from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
    nodal_solver = NodalMatrixSolver(oil_density=solver.oil_density, oil_type=solver.oil_type)
    conn_flows_nodal, sol_info_nodal = nodal_solver.solve_nodal_network(
        network,
        total_flow_rate,
        temperature,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0
    )

    # Print results
    solver.print_results(network, conn_flows_nodal, sol_info_nodal)

    # Optional: path-level comparison summary
    paths = network.get_paths_to_outlets()
    print(f"\nPATH PRESSURE DROPS (NODAL):")
    for i, path in enumerate(paths, 1):
        total_dp = sum(sol_info_nodal['pressure_drops'][c.component.id] for c in path)
        print(f"  Path {i}: Total ΔP = {total_dp/1000:.1f} kPa")