#!/usr/bin/env python3
"""
Quick analysis script to identify the solver convergence issue
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from lubrication_flow_package.config.network_config import NetworkConfigLoader
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.solvers.tree_solver import TreeSolver
from lubrication_flow_package.solvers.nonlinear_solver import RobustNonLinearSolver
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

def analyze_network_convergence():
    """Analyze convergence issues with the complex network."""
    
    # Load the failing network
    config_path = "examples/complex_network_modified.json"
    config = NetworkConfigLoader.load_json(config_path)
    
    # Create simulation config
    sim_config = SimulationConfig(**config.simulation)
    
    # Build network
    from lubrication_flow_package.utils.network_builder import NetworkBuilder
    builder = NetworkBuilder(sim_config)
    network = builder.build_from_config(config)
    
    print("=== NETWORK ANALYSIS ===")
    print(f"Nodes: {len(network.nodes)}")
    print(f"Connections: {len(network.connections)}")
    print(f"Outlets: {len(network.outlet_nodes)}")
    
    # Check for cycles
    import networkx as nx
    G = nx.Graph()
    for conn in network.connections:
        G.add_edge(conn.from_node.id, conn.to_node.id)
    
    cycles = nx.cycle_basis(G)
    print(f"Cycles detected: {len(cycles)}")
    
    # Get linear solution for initial guess
    print("\n=== LINEAR SOLVER ANALYSIS ===")
    linear_solver = NodalMatrixSolver(sim_config)
    try:
        linear_result = linear_solver.solve(network)
        print(f"Linear solver converged: {linear_result.get('converged', 'Unknown')}")
        
        pressures = linear_result['node_pressures']
        flows = linear_result['component_flows']
        
        print(f"Pressure range: {min(pressures.values()):.2e} to {max(pressures.values()):.2e}")
        print(f"Flow range: {min(flows.values()):.2e} to {max(flows.values()):.2e}")
        
        # Check for unrealistic values
        huge_pressures = [p for p in pressures.values() if abs(p) > 1e10]
        huge_flows = [f for f in flows.values() if abs(f) > 1e6]
        
        print(f"Unrealistically large pressures: {len(huge_pressures)}")
        print(f"Unrealistically large flows: {len(huge_flows)}")
        
    except Exception as e:
        print(f"Linear solver failed: {e}")
        return
    
    # Analyze tree solver
    print("\n=== TREE SOLVER ANALYSIS ===")
    tree_solver = TreeSolver(sim_config)
    
    # Get initial setup
    ref_node_id = network.outlet_nodes[0].id
    unknown_node_ids = [nid for nid in network.nodes if nid != ref_node_id]
    
    initial_pressures = linear_result['node_pressures']
    pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])
    
    print(f"Initial pressure guess range: {pressures.min():.2e} to {pressures.max():.2e}")
    
    # Build and analyze Jacobian
    try:
        jacobian = tree_solver._build_jacobian(pressures, network, unknown_node_ids, ref_node_id)
        jacobian_dense = jacobian.toarray()
        
        print(f"Jacobian shape: {jacobian_dense.shape}")
        
        # Condition number
        cond_num = np.linalg.cond(jacobian_dense)
        print(f"Jacobian condition number: {cond_num:.2e}")
        
        if cond_num > 1e12:
            print("WARNING: Jacobian is ill-conditioned!")
            
        # Check eigenvalues
        eigenvals = np.linalg.eigvals(jacobian_dense)
        real_eigenvals = eigenvals.real
        
        print(f"Eigenvalue range: {real_eigenvals.min():.2e} to {real_eigenvals.max():.2e}")
        
        negative_eigenvals = sum(1 for ev in real_eigenvals if ev < 0)
        zero_eigenvals = sum(1 for ev in real_eigenvals if abs(ev) < 1e-12)
        
        print(f"Negative eigenvalues: {negative_eigenvals}")
        print(f"Near-zero eigenvalues: {zero_eigenvals}")
        
        # Check residual
        residual = tree_solver._evaluate_residual(pressures, network, unknown_node_ids, ref_node_id)
        residual_norm = np.linalg.norm(residual)
        
        print(f"Initial residual norm: {residual_norm:.2e}")
        print(f"Residual components: min={residual.min():.2e}, max={residual.max():.2e}")
        
        # Check scaling
        if residual_norm > 0:
            non_zero_residuals = residual[residual != 0]
            if len(non_zero_residuals) > 0:
                scaling_ratio = np.max(np.abs(non_zero_residuals)) / np.min(np.abs(non_zero_residuals))
                print(f"Residual scaling ratio: {scaling_ratio:.2e}")
                
                if scaling_ratio > 1e6:
                    print("WARNING: Poor residual scaling detected!")
        
    except Exception as e:
        print(f"Jacobian analysis failed: {e}")
        return
    
    # Run actual solver
    print("\n=== SOLVER EXECUTION ===")
    try:
        result = tree_solver.solve(network)
        print(f"Tree solver converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        
        if not result['converged']:
            print("Tree solver failed to converge")
    except Exception as e:
        print(f"Tree solver failed: {e}")
    
    # Try nonlinear solver
    print("\n=== NONLINEAR SOLVER ANALYSIS ===")
    try:
        nonlinear_solver = RobustNonLinearSolver(config.simulation)
        
        # Get initial flow guess
        q_initial = nonlinear_solver._initialize_flows(network)
        cycles = nonlinear_solver._find_fundamental_cycles(network)
        
        print(f"Initial flow guess range: {q_initial.min():.2e} to {q_initial.max():.2e}")
        print(f"Number of cycles: {len(cycles)}")
        
        # Build Jacobian
        jacobian = nonlinear_solver._build_jacobian(q_initial, network, cycles)
        jacobian_dense = jacobian.toarray()
        
        print(f"Nonlinear Jacobian shape: {jacobian_dense.shape}")
        
        cond_num = np.linalg.cond(jacobian_dense)
        print(f"Nonlinear Jacobian condition number: {cond_num:.2e}")
        
        # Check residual
        residual = nonlinear_solver._evaluate_residual(q_initial, network, cycles)
        residual_norm = np.linalg.norm(residual)
        
        print(f"Initial nonlinear residual norm: {residual_norm:.2e}")
        
        # Run solver
        result = nonlinear_solver.solve(network)
        print(f"Nonlinear solver converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        
    except Exception as e:
        print(f"Nonlinear solver analysis failed: {e}")
    
    # Summary and recommendations
    print("\n=== SUMMARY AND RECOMMENDATIONS ===")
    
    if len(huge_pressures) > 0 or len(huge_flows) > 0:
        print("ISSUE: Unrealistically large pressure/flow values from linear solver")
        print("RECOMMENDATION: Check component parameters and scaling")
    
    if cond_num > 1e12:
        print("ISSUE: Ill-conditioned Jacobian matrix")
        print("RECOMMENDATION: Improve matrix conditioning or use regularization")
    
    if len(cycles) > 0:
        print("ISSUE: Network has cycles - may not be suitable for tree solver")
        print("RECOMMENDATION: Use nonlinear solver for networks with cycles")
    
    print("\nPossible root causes:")
    print("1. Extreme component parameters leading to numerical issues")
    print("2. Poor scaling between different components")
    print("3. Network topology issues (cycles for tree solver)")
    print("4. Step size too large causing oscillations")

if __name__ == "__main__":
    analyze_network_convergence()