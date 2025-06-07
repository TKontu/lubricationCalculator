"""
Main CLI entry point
"""

import argparse
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
    parser = argparse.ArgumentParser(description='Lubrication Flow Calculator')
    parser.add_argument('--solver', choices=['iterative', 'pump', 'nodal', 'legacy'], 
                       default='nodal', help='Solver method to use')
    parser.add_argument('--demo', action='store_true', help='Run full demo comparison')
    args = parser.parse_args()
    
    if args.demo:
        main()
    else:
        # Run single solver
        solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
        network, total_flow_rate, temperature = create_simple_tree_example()

        solver_methods = {
            'iterative': solver.solve_network_flow,
            'pump': solver.solve_network_flow_with_pump_physics,
            'nodal': solver.solve_network_flow_nodal,
            'legacy': solver.solve_network_flow_legacy
        }
        
        solver_names = {
            'iterative': "ITERATIVE PATH-BASED SOLVER",
            'pump': "PUMP PHYSICS SOLVER", 
            'nodal': "MATRIX-BASED NODAL SOLVER",
            'legacy': "LEGACY SOLVER"
        }

        print("\n" + "="*70)
        print(f"{solver_names[args.solver]}")
        print("="*70)

        # Solve using selected method
        if args.solver == 'nodal':
            conn_flows, sol_info = solver_methods[args.solver](
                network, total_flow_rate, temperature,
                inlet_pressure=200000.0, outlet_pressure=101325.0
            )
        else:
            conn_flows, sol_info = solver_methods[args.solver](
                network, total_flow_rate, temperature
            )

        # Print results
        solver.print_results(network, conn_flows, sol_info)

        # Optional: path-level comparison summary
        paths = network.get_paths_to_outlets()
        print(f"\nPATH PRESSURE DROPS ({args.solver.upper()}):")
        for i, path in enumerate(paths, 1):
            total_dp = sum(sol_info['pressure_drops'][c.component.id] for c in path)
            print(f"  Path {i}: Total ΔP = {total_dp/1000:.1f} kPa")