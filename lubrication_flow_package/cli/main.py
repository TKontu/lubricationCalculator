"""
Main CLI entry point
"""

from ..solvers import NetworkFlowSolver
from .examples import create_simple_tree_example
from .demos import demonstrate_hydraulic_approaches_comparison, demonstrate_proper_hydraulic_analysis


def main():
    """Demonstrate the network-based flow calculator"""
    print("NETWORK-BASED LUBRICATION FLOW DISTRIBUTION CALCULATOR")
    print("="*60)
    
    # Run the comparison demonstration
    demonstrate_hydraulic_approaches_comparison()
    
    print("\n\n")
    
    # Run the original demonstration
    demonstrate_proper_hydraulic_analysis()


if __name__ == "__main__":
    # Existing setup from main()
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    network, total_flow_rate, temperature = create_simple_tree_example()

    print("\n" + "="*70)
    print("APPROACH 3: MATRIX-BASED NODAL SOLVER (SCALABLE)")
    print("="*70)
    print("✓ Single linear solve for nodal pressures")
    print("✓ Direct flow via conductance" )

    # Solve using new nodal method
    conn_flows_nodal, sol_info_nodal = solver.solve_network_flow_nodal(
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
    
    #main()