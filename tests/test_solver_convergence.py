"""
Comprehensive pytest suite for testing solver convergence issues.
Identifies root causes of convergence failures in both tree and nonlinear solvers.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock, patch
import json

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.solvers.tree_solver import TreeSolver
from lubrication_flow_package.solvers.nonlinear_solver import RobustNonLinearSolver
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.network.connection import Connection


class TestSolverConvergence:
    """Test suite for analyzing solver convergence issues."""

    @pytest.fixture
    def basic_config(self):
        """Basic simulation configuration for testing."""
        return SimulationConfig(
            temperature=55.0,
            oil_type="VG320",
            oil_density=900.0,
            total_flow_rate=60.0,
            inlet_pressure=200000.0,
            outlet_pressure=101325.0,
            max_iterations=100,
            tolerance=1e-6,
            min_resistance=1e-12
        )

    @pytest.fixture
    def simple_network(self):
        """Create a simple 2-node network for testing."""
        network = FlowNetwork()
        
        # Add nodes
        inlet = Node("inlet", 0.0, 0.0, 0.0)
        outlet = Node("outlet", 1.0, 0.0, 0.0)
        
        network.add_node(inlet)
        network.add_node(outlet)
        
        # Add a simple channel connection
        channel = Channel(0.01, 1.0, 0.0001, "channel_1")  # 10mm diameter, 1m length
        connection = network.connect_components(inlet, outlet, channel)
        
        network.set_inlet(inlet)
        network.add_outlet(outlet)
        
        return network

    @pytest.fixture
    def complex_network(self):
        """Create a more complex network similar to the failing case."""
        network = FlowNetwork()
        
        # Create nodes
        nodes = {}
        for i in range(10):
            node = Node(f"node_{i}", i * 0.1, 0.0, 0.0)
            nodes[f"node_{i}"] = node
            network.add_node(node)
        
        # Create connections with varying resistances
        connections = []
        for i in range(9):
            from_node = nodes[f"node_{i}"]
            to_node = nodes[f"node_{i+1}"]
            
            # Alternate between channels and nozzles
            if i % 2 == 0:
                component = Channel(0.005 + i * 0.001, 0.1, 0.0001, f"channel_{i}")
            else:
                component = Nozzle(0.003 + i * 0.0005, component_id=f"nozzle_{i}")
            
            connection = network.connect_components(from_node, to_node, component)
            connections.append(connection)
        
        # Add branching
        branch_node = Node("branch", 0.5, 0.1, 0.0)
        network.add_node(branch_node)
        
        branch_outlet = Node("branch_outlet", 0.5, 0.2, 0.0)
        network.add_node(branch_outlet)
        
        # Connect branch
        branch_channel = Channel(0.008, 0.1, 0.0001, "branch_channel")
        branch_connection = network.connect_components(nodes["node_5"], branch_node, branch_channel)
        
        branch_nozzle = Nozzle(0.004, component_id="branch_nozzle")
        branch_outlet_connection = network.connect_components(branch_node, branch_outlet, branch_nozzle)
        
        # Set inlet and outlets
        network.set_inlet(nodes["node_0"])
        network.add_outlet(nodes["node_9"])
        network.add_outlet(branch_outlet)
        
        return network

    def test_simple_network_convergence(self, simple_network, basic_config):
        """Test convergence on a simple 2-node network."""
        solver = TreeSolver(basic_config)
        result = solver.solve(simple_network)
        
        assert result["converged"] == True
        assert result["iterations"] < 10
        assert abs(result["component_flows"]["channel_1"]) > 0
        
    def test_jacobian_condition_number(self, complex_network, basic_config):
        """Test the condition number of the Jacobian matrix."""
        solver = TreeSolver(basic_config)
        
        # Get initial guess
        from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
        linear_solver = NodalMatrixSolver(basic_config)
        linear_solution = linear_solver.solve(complex_network)
        
        # Extract unknown nodes
        ref_node_id = complex_network.outlet_nodes[0].id
        unknown_node_ids = [nid for nid in complex_network.nodes if nid != ref_node_id]
        
        initial_pressures = linear_solution['node_pressures']
        pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])
        
        # Build Jacobian
        jacobian = solver._build_jacobian(pressures, complex_network, unknown_node_ids, ref_node_id)
        
        # Calculate condition number
        jacobian_dense = jacobian.toarray()
        try:
            cond_num = np.linalg.cond(jacobian_dense)
            print(f"Jacobian condition number: {cond_num}")
            
            # Check if matrix is ill-conditioned
            assert cond_num < 1e12, f"Jacobian is ill-conditioned with condition number {cond_num}"
            
        except np.linalg.LinAlgError:
            pytest.fail("Jacobian matrix is singular")

    def test_residual_scaling(self, complex_network, basic_config):
        """Test if residual components have vastly different scales."""
        solver = TreeSolver(basic_config)
        
        # Get initial guess
        from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
        linear_solver = NodalMatrixSolver(basic_config)
        linear_solution = linear_solver.solve(complex_network)
        
        ref_node_id = complex_network.outlet_nodes[0].id
        unknown_node_ids = [nid for nid in complex_network.nodes if nid != ref_node_id]
        
        initial_pressures = linear_solution['node_pressures']
        pressures = np.array([initial_pressures[nid] for nid in unknown_node_ids])
        
        # Evaluate residual
        residual = solver._evaluate_residual(pressures, complex_network, unknown_node_ids, ref_node_id)
        
        # Check scaling
        max_residual = np.max(np.abs(residual))
        min_residual = np.min(np.abs(residual[residual != 0]))
        
        if min_residual > 0:
            scaling_ratio = max_residual / min_residual
            print(f"Residual scaling ratio: {scaling_ratio}")
            print(f"Residual values: {residual}")
            
            # Large scaling ratios indicate poor conditioning
            if scaling_ratio > 1e6:
                print("WARNING: Poor residual scaling detected")

    def test_step_size_behavior(self, complex_network, basic_config):
        """Test how step sizes behave during iteration."""
        solver = TreeSolver(basic_config)
        
        # Mock the line search to record alpha values
        alpha_values = []
        original_line_search = solver._line_search
        
        def mock_line_search(pressures, delta_p, residual, network, unknown_node_ids, ref_node_id):
            alpha = original_line_search(pressures, delta_p, residual, network, unknown_node_ids, ref_node_id)
            alpha_values.append(alpha)
            return alpha
        
        solver._line_search = mock_line_search
        
        # Run solver
        result = solver.solve(complex_network)
        
        print(f"Alpha values during iteration: {alpha_values}")
        
        # Check for consistently small step sizes (indicates convergence issues)
        if len(alpha_values) > 5:
            recent_alphas = alpha_values[-5:]
            if all(alpha < 0.1 for alpha in recent_alphas):
                print("WARNING: Consistently small step sizes detected")

    def test_differential_resistance_accuracy(self, complex_network, basic_config):
        """Test the accuracy of differential resistance calculations."""
        solver = TreeSolver(basic_config)
        
        # Test differential resistance calculation for various components
        for connection in complex_network.connections:
            component = connection.component
            
            # Test at different flow rates
            test_flows = [0.1, 1.0, 10.0, 50.0]
            
            for flow in test_flows:
                # Calculate analytical differential resistance
                try:
                    diff_resistance = component.get_differential_resistance(flow, solver.fluid_properties)
                    
                    # Calculate numerical differential resistance
                    delta_q = max(abs(flow) * 1e-6, 1e-9)
                    dp_plus = component.calculate_pressure_drop(flow + delta_q, solver.fluid_properties)
                    dp_minus = component.calculate_pressure_drop(flow - delta_q, solver.fluid_properties)
                    numerical_diff_resistance = (dp_plus - dp_minus) / (2.0 * delta_q)
                    
                    # Compare
                    if abs(diff_resistance) > 1e-12:
                        relative_error = abs(diff_resistance - numerical_diff_resistance) / abs(diff_resistance)
                        
                        print(f"Component {component.id}, Flow {flow}: "
                              f"Analytical={diff_resistance:.6e}, "
                              f"Numerical={numerical_diff_resistance:.6e}, "
                              f"Relative Error={relative_error:.6e}")
                        
                        # Large errors indicate issues with differential resistance
                        if relative_error > 0.1:
                            print(f"WARNING: Large differential resistance error for {component.id}")
                
                except Exception as e:
                    print(f"Error calculating differential resistance for {component.id}: {e}")

    def test_mass_conservation_accuracy(self, complex_network, basic_config):
        """Test mass conservation accuracy in the solution."""
        solver = TreeSolver(basic_config)
        result = solver.solve(complex_network)
        
        # Check mass conservation at each node
        component_flows = result["component_flows"]
        node_pressures = result["node_pressures"]
        
        mass_conservation_errors = []
        
        for node_id, node in complex_network.nodes.items():
            if node_id == complex_network.inlet_node.id:
                continue  # Skip inlet node
                
            net_flow = 0.0
            
            for connection in complex_network.connections:
                flow = component_flows[connection.component.id]
                
                if connection.from_node.id == node_id:
                    net_flow -= flow  # Outgoing flow
                elif connection.to_node.id == node_id:
                    net_flow += flow  # Incoming flow
            
            # Add inlet flow constraint
            if node_id == complex_network.inlet_node.id:
                net_flow += basic_config.total_flow_rate
            
            mass_conservation_errors.append(abs(net_flow))
            
            if abs(net_flow) > basic_config.tolerance:
                print(f"Mass conservation violation at node {node_id}: {net_flow}")
        
        max_mass_error = max(mass_conservation_errors)
        print(f"Maximum mass conservation error: {max_mass_error}")
        
        # This should be small for a converged solution
        assert max_mass_error < 1e-3, f"Mass conservation error too large: {max_mass_error}"

    def test_pressure_monotonicity(self, simple_network, basic_config):
        """Test that pressure decreases monotonically in simple networks."""
        solver = TreeSolver(basic_config)
        result = solver.solve(simple_network)
        
        node_pressures = result["node_pressures"]
        
        inlet_pressure = node_pressures[simple_network.inlet_node.id]
        outlet_pressure = node_pressures[simple_network.outlet_nodes[0].id]
        
        print(f"Inlet pressure: {inlet_pressure}, Outlet pressure: {outlet_pressure}")
        
        # Inlet pressure should be higher than outlet pressure
        assert inlet_pressure > outlet_pressure, "Pressure should decrease from inlet to outlet"

    def test_solver_with_extreme_resistances(self, basic_config):
        """Test solver behavior with extreme resistance values."""
        network = FlowNetwork()
        
        # Create nodes
        inlet = Node("inlet", 0.0, 0.0, 0.0)
        mid = Node("mid", 0.5, 0.0, 0.0)
        outlet = Node("outlet", 1.0, 0.0, 0.0)
        
        network.add_node(inlet)
        network.add_node(mid)
        network.add_node(outlet)
        
        # Very high resistance component
        high_resistance = Channel(0.001, 1.0, 0.0001, "high_res")  # Very small diameter
        conn1 = network.connect_components(inlet, mid, high_resistance)
        
        # Very low resistance component
        low_resistance = Channel(0.05, 0.1, 0.0001, "low_res")  # Large diameter
        conn2 = network.connect_components(mid, outlet, low_resistance)
        
        network.set_inlet(inlet)
        network.add_outlet(outlet)
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        print(f"Extreme resistance test - Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        
        # The solver should handle extreme resistances gracefully
        assert result["converged"] == True or result["iterations"] < basic_config.max_iterations

    def test_network_with_cycles(self, basic_config):
        """Test solver behavior with cyclic networks (should fail gracefully)."""
        network = FlowNetwork()
        
        # Create a simple cycle
        nodes = {}
        for i in range(4):
            node = Node(f"node_{i}", i * 0.25, 0.0, 0.0)
            nodes[f"node_{i}"] = node
            network.add_node(node)
        
        # Create cycle connections
        for i in range(4):
            from_node = nodes[f"node_{i}"]
            to_node = nodes[f"node_{(i+1)%4}"]
            
            channel = Channel(0.01, 0.25, 0.0001, f"channel_{i}")
            connection = network.connect_components(from_node, to_node, channel)
        
        # Add inlet and outlet
        network.set_inlet(nodes["node_0"])
        network.add_outlet(nodes["node_2"])
        
        solver = TreeSolver(basic_config)
        
        # This should either solve or fail gracefully
        try:
            result = solver.solve(network)
            print(f"Cyclic network test - Converged: {result['converged']}")
        except Exception as e:
            print(f"Cyclic network handling: {e}")
            # This is expected for tree solvers with cyclic networks

    def test_nonlinear_solver_convergence(self, complex_network, basic_config):
        """Test the nonlinear solver convergence on complex networks."""
        solver = RobustNonLinearSolver(basic_config)
        result = solver.solve(complex_network)
        
        print(f"Nonlinear solver - Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        
        # Test mass conservation
        component_flows = result["component_flows"]
        
        # Check total flow rate
        total_outlet_flow = sum(
            component_flows[conn.component.id] 
            for conn in complex_network.connections 
            if conn.to_node.id in [node.id for node in complex_network.outlet_nodes]
        )
        
        print(f"Total outlet flow: {total_outlet_flow}")
        print(f"Expected flow: {basic_config.total_flow_rate}")
        
        # Flow should be conserved
        flow_error = abs(total_outlet_flow - basic_config.total_flow_rate)
        assert flow_error < 1e-2, f"Flow conservation error: {flow_error}"

    @pytest.mark.parametrize("solver_class", [TreeSolver, RobustNonLinearSolver])
    def test_solver_comparison(self, simple_network, basic_config, solver_class):
        """Compare different solvers on the same network."""
        solver = solver_class(basic_config)
        result = solver.solve(simple_network)
        
        print(f"{solver_class.__name__} results:")
        print(f"  Converged: {result['converged']}")
        print(f"  Iterations: {result['iterations']}")
        print(f"  Component flows: {result['component_flows']}")
        print(f"  Node pressures: {result['node_pressures']}")
        
        # Basic checks
        assert "converged" in result
        assert "iterations" in result
        assert "component_flows" in result
        assert "node_pressures" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])