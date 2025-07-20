

import argparse
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path to allow importing package modules
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from lubrication_flow_package.config.network_config import NetworkConfigLoader
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver
from lubrication_flow_package.solvers.nonlinear_loop_solver import RobustNonLinearSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.connection import Connection

def get_solver(solver_name: str, sim_config):
    """Factory function to get a solver instance."""
    solver_map = {
        'nodal': NodalMatrixSolver,
        'tree_nonlinear': TreeSolver,
        'robust_newton': RobustNonLinearSolver
    }
    solver_class = solver_map.get(solver_name)
    if not solver_class:
        raise ValueError(f"Unknown solver: {solver_name}")
    return solver_class(sim_config)

def verify_mass_conservation(network: FlowNetwork, solution: Dict, tolerance=1e-9) -> List[str]:
    """
    Checks if mass is conserved at each non-boundary node.
    For each node, sum of incoming flows must equal sum of outgoing flows.
    """
    errors = []
    node_flows: Dict[str, float] = {node_id: 0.0 for node_id in network.nodes}
    component_flows = solution.get('component_flows', {})

    # Calculate net flow at each node based on component flows
    for conn in network.connections:
        flow = component_flows.get(conn.component.id, 0.0)
        node_flows[conn.from_node.id] -= flow
        node_flows[conn.to_node.id] += flow

    # Add total source flow
    inlet_node_id = network.inlet_node.id
    total_flow = solution.get('total_flow_rate', 0.0)
    node_flows[inlet_node_id] += total_flow

    # Check balance at each internal node
    internal_nodes = [nid for nid in network.nodes if nid != inlet_node_id and nid not in [n.id for n in network.outlet_nodes]]
    for node_id in internal_nodes:
        net_flow = node_flows[node_id]
        if abs(net_flow) > tolerance:
            errors.append(f"Mass conservation failed at node '{node_id}'. Net flow: {net_flow:.2e}")

    return errors

def verify_pressure_drops(network: FlowNetwork, solution: Dict, tolerance=1e-6) -> List[str]:
    """
    Checks if the pressure drop across each component matches the flow through it.
    """
    errors = []
    node_pressures = solution.get('node_pressures', {})
    component_flows = solution.get('component_flows', {})
    
    # Need fluid properties to calculate pressure drops
    solver = NodalMatrixSolver(network.sim_config) # Dummy solver to get properties
    fluid_properties = solver.fluid_properties

    for conn in network.connections:
        p_from = node_pressures.get(conn.from_node.id)
        p_to = node_pressures.get(conn.to_node.id)
        if p_from is None or p_to is None:
            errors.append(f"Missing pressure for nodes in connection {conn.component.id}")
            continue

        solver_dp = p_from - p_to
        flow = component_flows.get(conn.component.id, 0.0)
        
        # Calculate what the pressure drop *should* be for that flow
        calculated_dp = conn.component.calculate_pressure_drop(flow, fluid_properties)

        if abs(solver_dp - calculated_dp) > tolerance:
            errors.append(
                f"Pressure drop mismatch for component '{conn.component.id}'. "
                f"Solver dP: {solver_dp:.2f}, Calculated dP: {calculated_dp:.2f}, Flow: {flow:.2e}"
            )
    return errors

def verify_path_pressures(network: FlowNetwork, solution: Dict, tolerance=1e-6) -> List[str]:
    """
    Checks if the sum of pressure drops along each path from inlet to an outlet
    equals the total pressure difference.
    """
    errors = []
    node_pressures = solution.get('node_pressures', {})
    component_flows = solution.get('component_flows', {})
    
    inlet_pressure = node_pressures.get(network.inlet_node.id)
    if inlet_pressure is None:
        return ["Inlet pressure not found in solution."]

    paths = network.get_paths_to_outlets()

    for i, path in enumerate(paths):
        path_sum_dp = 0
        
        # Get the outlet node for this path
        outlet_node_id = path[-1].to_node.id
        outlet_pressure = node_pressures.get(outlet_node_id)
        if outlet_pressure is None:
            errors.append(f"Missing pressure for outlet node '{outlet_node_id}' in path {i+1}")
            continue
            
        total_path_dp = inlet_pressure - outlet_pressure

        for conn in path:
            p_from = node_pressures.get(conn.from_node.id)
            p_to = node_pressures.get(conn.to_node.id)
            if p_from is None or p_to is None:
                errors.append(f"Missing pressure for nodes in connection {conn.component.id} in path {i+1}")
                continue
            path_sum_dp += (p_from - p_to)

        if abs(path_sum_dp - total_path_dp) > tolerance:
            errors.append(
                f"Path {i+1} (to outlet '{outlet_node_id}') failed pressure summation. "
                f"Sum of component dPs: {path_sum_dp:.2f}, Total Path dP: {total_path_dp:.2f}"
            )
    return errors


def main():
    parser = argparse.ArgumentParser(description="Verify a hydraulic network simulation solution.")
    parser.add_argument('config_file', type=str, help='Path to the network configuration JSON file.')
    parser.add_argument('--solver', type=str, default='nodal', choices=['nodal', 'tree_nonlinear', 'robust_newton'],
                        help='The solver to use for the simulation.')
    args = parser.parse_args()

    # --- 1. Load Network and Config ---
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Configuration file not found at '{config_path}'")
        sys.exit(1)

    print(f"Loading network from '{config_path}'...")
    config = NetworkConfigLoader.load_json(str(config_path))
    network, sim_config = NetworkConfigLoader.build_network(config)
    network.sim_config = sim_config # Attach sim_config to network for easy access

    # --- 2. Run Simulation ---
    print(f"Running simulation with '{args.solver}' solver...")
    try:
        solver = get_solver(args.solver, sim_config)
        solution = solver.solve(network)
    except Exception as e:
        print(f"Error during simulation: {e}")
        sys.exit(1)

    if not solution.get('converged', False):
        print("Warning: Solver did not converge. Verification may not be meaningful.")
    
    final_residual = solution.get('final_residual_norm')
    if final_residual is not None:
        print(f"Final Residual Norm: {final_residual:.4e}")

    print("\n--- Verification Results ---")

    # --- 3. Perform Verification ---
    conservation_errors = verify_mass_conservation(network, solution)
    pressure_errors = verify_pressure_drops(network, solution)
    path_errors = verify_path_pressures(network, solution)

    # --- 4. Report Results ---
    if not any([conservation_errors, pressure_errors, path_errors]):
        print("[+] Solution is physically consistent. All checks passed.")
    else:
        print("[-] Solution failed verification checks.")
        if conservation_errors:
            print("\n--- Mass Conservation Errors ---")
            for error in conservation_errors:
                print(f"  - {error}")
        if pressure_errors:
            print("\n--- Pressure Drop Consistency Errors ---")
            for error in pressure_errors:
                print(f"  - {error}")
        if path_errors:
            print("\n--- Path Pressure Summation Errors ---")
            for error in path_errors:
                print(f"  - {error}")

if __name__ == "__main__":
    main()

