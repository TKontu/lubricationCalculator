
"""
Diagnostic script to analyze solver convergence issues with the specific failing network.
"""

import pytest
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from lubrication_flow_package.config.network_config import NetworkConfigLoader
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver
from lubrication_flow_package.solvers.nonlinear_loop_solver import RobustNonLinearSolver


class SolverDiagnostics:
    """Diagnostic tools for analyzing solver convergence issues."""

    def __init__(self, config_path: str):
        """Initialize with config file path."""
        self.config_path = Path(config_path)
        network_config = NetworkConfigLoader.load_json(self.config_path)
        self.network, self.simulation_config = NetworkConfigLoader.build_network(network_config)

    def analyze_network_properties(self):
        """Analyze basic network properties."""
        print("=== NETWORK ANALYSIS ===")
        print(f"Nodes: {len(self.network.nodes)}")
        print(f"Connections: {len(self.network.connections)}")
        print(f"Outlets: {len(self.network.outlet_nodes)}")

        # Check network topology
        in_degrees = {}
        out_degrees = {}

        for node_id in self.network.nodes:
            in_degrees[node_id] = 0
            out_degrees[node_id] = 0

        for conn in self.network.connections:
            out_degrees[conn.from_node.id] += 1
            in_degrees[conn.to_node.id] += 1

        print(f"Max in-degree: {max(in_degrees.values())}")
        print(f"Max out-degree: {max(out_degrees.values())}")

        # Check for cycles
        import networkx as nx
        G = nx.Graph()
        for conn in self.network.connections:
            G.add_edge(conn.from_node.id, conn.to_node.id)

        cycles = nx.cycle_basis(G)
        print(f"Number of cycles: {len(cycles)}")

        # Analyze component resistances
        resistances = []
        for conn in self.network.connections:
            # Use a nominal flow rate for resistance calculation
            nominal_flow = 1.0  # L/min
            fluid_props = {'viscosity': 0.142736, 'density': 900.0}

            try:
                resistance = conn.component.get_differential_resistance(nominal_flow, fluid_props)
                resistances.append(resistance)
            except:
                resistances.append(np.inf)

        valid_resistances = [r for r in resistances if np.isfinite(r)]
        if valid_resistances:
            print(f"Resistance range: {min(valid_resistances):.2e} to {max(valid_resistances):.2e}")
            print(f"Resistance ratio: {max(valid_resistances)/min(valid_resistances):.2e}")

        return {
            'nodes': len(self.network.nodes),
            'connections': len(self.network.connections),
            'cycles': len(cycles),
            'resistances': resistances
        }

    def analyze_linear_initial_guess(self):
        """Analyze the linear solver initial guess."""
        print("\n=== LINEAR INITIAL GUESS ANALYSIS ===")

        from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

        linear_solver = NodalMatrixSolver(self.simulation_config)
        result = linear_solver.solve(self.network)

        pressures = result['node_pressures']
        flows = result['component_flows']

        print(f"Pressure range: {min(pressures.values()):.2e} to {max(pressures.values()):.2e}")
        print(f"Flow range: {min(flows.values()):.2e} to {max(flows.values()):.2e}")

        # Check for negative flows
        negative_flows = [f for f in flows.values() if f < 0]
        print(f"Negative flows: {len(negative_flows)}")

        # Check pressure monotonicity
        inlet_pressure = pressures[self.network.inlet_node.id]
        outlet_pressures = [pressures[node.id] for node in self.network.outlet_nodes]

        print(f"Inlet pressure: {inlet_pressure:.2e}")
        print(f"Outlet pressure range: {min(outlet_pressures):.2e} to {max(outlet_pressures):.2e}")

        return result

    def analyze_jacobian_properties(self, solver_type='tree'):
        """Analyze Jacobian matrix properties."""
        print(f"\n=== JACOBIAN ANALYSIS ({solver_type.upper()}) ===")

        if solver_type == 'tree':
            solver = TreeSolver(self.simulation_config)
        else:
            solver = RobustNonLinearSolver(self.simulation_config)

        # Get initial guess
        linear_result = self.analyze_linear_initial_guess()

        if solver_type == 'tree':
            ref_node_id = self.network.outlet_nodes[0].id
            unknown_node_ids = [nid for nid in self.network.nodes if nid != ref_node_id]

            initial_pressures = linear_result['node_pressures']
            pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])

            jacobian = solver._build_jacobian(pressures, self.network, unknown_node_ids, ref_node_id)
        else:
            # For nonlinear solver, we need to get the flow vector
            q_initial = solver._initialize_flows(self.network)
            cycles = solver._find_fundamental_cycles(self.network)
            jacobian = solver._build_jacobian(q_initial, self.network, cycles)

        # Convert to dense for analysis
        jacobian_dense = jacobian.toarray()

        print(f"Jacobian shape: {jacobian_dense.shape}")

        # Condition number
        try:
            cond_num = np.linalg.cond(jacobian_dense)
            print(f"Condition number: {cond_num:.2e}")
        except np.linalg.LinAlgError:
            print("Jacobian is singular!")
            cond_num = np.inf

        # Eigenvalues
        try:
            eigenvals = np.linalg.eigvals(jacobian_dense)
            real_eigenvals = eigenvals.real
            print(f"Eigenvalue range: {min(real_eigenvals):.2e} to {max(real_eigenvals):.2e}")

            # Check for negative eigenvalues
            negative_eigenvals = sum(1 for ev in real_eigenvals if ev < 0)
            print(f"Negative eigenvalues: {negative_eigenvals}")

            # Check for zero eigenvalues
            zero_eigenvals = sum(1 for ev in real_eigenvals if abs(ev) < 1e-12)
            print(f"Near-zero eigenvalues: {zero_eigenvals}")

        except np.linalg.LinAlgError:
            print("Could not compute eigenvalues")

        # Matrix norm
        frobenius_norm = np.linalg.norm(jacobian_dense, 'fro')
        print(f"Frobenius norm: {frobenius_norm:.2e}")

        # Check diagonal dominance
        diagonal_elements = np.diag(jacobian_dense)
        off_diagonal_sums = np.sum(np.abs(jacobian_dense), axis=1) - np.abs(diagonal_elements)

        diagonal_dominant = np.all(np.abs(diagonal_elements) >= off_diagonal_sums)
        print(f"Diagonally dominant: {diagonal_dominant}")

        return {
            'condition_number': cond_num,
            'eigenvalues': eigenvals if 'eigenvals' in locals() else None,
            'frobenius_norm': frobenius_norm,
            'diagonal_dominant': diagonal_dominant
        }

    def run_convergence_iteration_analysis(self, solver_type='tree', max_iterations=10):
        """Run solver and analyze convergence behavior."""
        print(f"\n=== CONVERGENCE ITERATION ANALYSIS ({solver_type.upper()}) ===")

        if solver_type == 'tree':
            solver = TreeSolver(self.simulation_config)
        else:
            solver = RobustNonLinearSolver(self.simulation_config)

        # Patch solver to record iteration data
        iteration_data = {
            'residuals': [],
            'step_sizes': [],
            'jacobian_cond_numbers': [],
            'pressures': [] if solver_type == 'tree' else [],
            'flows': [] if solver_type == 'nonlinear' else []
        }

        if solver_type == 'tree':
            original_solve = solver.solve

            def patched_solve(network):
                # Get initial setup
                ref_node_id = network.outlet_nodes[0].id
                unknown_node_ids = [nid for nid in network.nodes if nid != ref_node_id]

                # Get initial guess
                from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
                linear_solver = NodalMatrixSolver(self.simulation_config)
                linear_solution = linear_solver.solve(network)
                initial_pressures = linear_solution['node_pressures']

                pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])

                for i in range(max_iterations):
                    # Evaluate residual
                    residual = solver._evaluate_residual(pressures, network, unknown_node_ids, ref_node_id)
                    residual_norm = np.linalg.norm(residual)
                    iteration_data['residuals'].append(residual_norm)
                    iteration_data['pressures'].append(pressures.copy())

                    # Build Jacobian
                    jacobian = solver._build_jacobian(pressures, network, unknown_node_ids, ref_node_id)

                    # Condition number
                    try:
                        cond_num = np.linalg.cond(jacobian.toarray())
                        iteration_data['jacobian_cond_numbers'].append(cond_num)
                    except:
                        iteration_data['jacobian_cond_numbers'].append(np.inf)

                    # Check convergence
                    if residual_norm < self.simulation_config.tolerance:
                        break

                    # Solve for step
                    try:
                        from scipy.sparse.linalg import spsolve
                        delta_p = spsolve(jacobian, -residual)

                        # Line search
                        alpha = solver._line_search(pressures, delta_p, residual, network, unknown_node_ids, ref_node_id)
                        iteration_data['step_sizes'].append(alpha)

                        # Update
                        pressures += alpha * delta_p

                    except Exception as e:
                        print(f"Iteration {i} failed: {e}")
                        break

                return original_solve(network)

            solver.solve = patched_solve

        # Run solver
        result = solver.solve(self.network)

        # Analyze iteration data
        print(f"Total iterations: {len(iteration_data['residuals'])}")

        if iteration_data['residuals']:
            print(f"Initial residual: {iteration_data['residuals'][0]:.6e}")
            print(f"Final residual: {iteration_data['residuals'][-1]:.6e}")

            # Check convergence rate
            if len(iteration_data['residuals']) > 1:
                convergence_rates = []
                for i in range(1, len(iteration_data['residuals'])):
                    if iteration_data['residuals'][i-1] > 0:
                        rate = iteration_data['residuals'][i] / iteration_data['residuals'][i-1]
                        convergence_rates.append(rate)

                if convergence_rates:
                    avg_rate = np.mean(convergence_rates)
                    print(f"Average convergence rate: {avg_rate:.6f}")

                    if avg_rate >= 1.0:
                        print("WARNING: Convergence rate >= 1.0 (diverging or stalled)")

        if iteration_data['step_sizes']:
            print(f"Step size range: {min(iteration_data['step_sizes']):.6f} to {max(iteration_data['step_sizes']):.6f}")

            # Check for consistently small step sizes
            small_steps = sum(1 for alpha in iteration_data['step_sizes'] if alpha < 0.1)
            print(f"Small step sizes (<0.1): {small_steps}/{len(iteration_data['step_sizes'])}")

        if iteration_data['jacobian_cond_numbers']:
            finite_cond_nums = [c for c in iteration_data['jacobian_cond_numbers'] if np.isfinite(c)]
            if finite_cond_nums:
                print(f"Condition number range: {min(finite_cond_nums):.2e} to {max(finite_cond_nums):.2e}")

        return iteration_data, result

    def generate_diagnostic_report(self):
        """Generate a comprehensive diagnostic report."""
        print("=" * 60)
        print("SOLVER CONVERGENCE DIAGNOSTIC REPORT")
        print("=" * 60)

        # Network analysis
        network_props = self.analyze_network_properties()

        # Linear analysis
        linear_result = self.analyze_linear_initial_guess()

        # Jacobian analysis for both solvers
        tree_jacobian = self.analyze_jacobian_properties('tree')
        nonlinear_jacobian = self.analyze_jacobian_properties('nonlinear')

        # Convergence analysis
        tree_convergence, tree_result = self.run_convergence_iteration_analysis('tree')
        nonlinear_convergence, nonlinear_result = self.run_convergence_iteration_analysis('nonlinear')

        # Summary
        print("\n" + "=" * 60)
        print("DIAGNOSTIC SUMMARY")
        print("=" * 60)

        print(f"Network complexity: {network_props['nodes']} nodes, {network_props['connections']} connections")
        print(f"Network cycles: {network_props['cycles']}")

        print(f"\nTree solver:")
        print(f"  Converged: {tree_result['converged']}")
        print(f"  Iterations: {tree_result['iterations']}")
        print(f"  Jacobian condition: {tree_jacobian['condition_number']:.2e}")

        print(f"\nNonlinear solver:")
        print(f"  Converged: {nonlinear_result['converged']}")
        print(f"  Iterations: {nonlinear_result['iterations']}")
        print(f"  Jacobian condition: {nonlinear_jacobian['condition_number']:.2e}")

        # Identify likely issues
        print(f"\nLikely issues:")

        if tree_jacobian['condition_number'] > 1e12:
            print("  - Tree solver Jacobian is ill-conditioned")

        if nonlinear_jacobian['condition_number'] > 1e12:
            print("  - Nonlinear solver Jacobian is ill-conditioned")

        if network_props['cycles'] > 0:
            print("  - Network has cycles (may cause issues for tree solver)")

        if len(tree_convergence['residuals']) > 1:
            if tree_convergence['residuals'][-1] >= tree_convergence['residuals'][0]:
                print("  - Tree solver residual not decreasing")

        if len(nonlinear_convergence['residuals']) > 1:
            if nonlinear_convergence['residuals'][-1] >= nonlinear_convergence['residuals'][0]:
                print("  - Nonlinear solver residual not decreasing")


def test_complex_network_diagnostics():
    """Test diagnostics on the complex network that's failing."""
    config_path = "examples/complex_network.json"

    diagnostics = SolverDiagnostics(config_path)
    diagnostics.generate_diagnostic_report()


if __name__ == "__main__":
    test_complex_network_diagnostics()
