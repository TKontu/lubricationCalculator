"""
Tests for the TreeSolver.
"""

import pytest
import numpy as np
import logging

#logging.basicConfig(level=logging.DEBUG)

from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.components.connector import Connector, ConnectorType
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.solvers.nonlinear_tree_solver import TreeSolver

from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.config.simulation_config import SimulationConfig


def test_component_flow_calculation():
    """
    Test the inverse flow calculation for components.
    This test must pass before the solver implementation begins.
    """
    fluid_properties = {'density': 900, 'viscosity': 0.015}
    
    # Test Channel
    channel = Channel(diameter=0.05, length=10)
    pressure_drop = 50000  # Pa
    
    # Add a placeholder for the new method
    if hasattr(channel, 'calculate_flow_rate'):
        flow_rate = channel.calculate_flow_rate(pressure_drop, fluid_properties)
        # Verify that the calculated flow rate produces the original pressure drop
        calculated_dp = channel.calculate_pressure_drop(flow_rate, fluid_properties)
        assert calculated_dp == pytest.approx(pressure_drop, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Channel")

    # Test Nozzle
    nozzle = Nozzle(diameter=0.01, nozzle_type=NozzleType.ROUNDED)
    pressure_drop_nozzle = 20000 # Pa

    if hasattr(nozzle, 'calculate_flow_rate'):
        flow_rate_nozzle = nozzle.calculate_flow_rate(pressure_drop_nozzle, fluid_properties)
        calculated_dp_nozzle = nozzle.calculate_pressure_drop(flow_rate_nozzle, fluid_properties)
        assert calculated_dp_nozzle == pytest.approx(pressure_drop_nozzle, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Nozzle")

    # Test Connector
    connector = Connector(diameter=0.05, connector_type=ConnectorType.ELBOW_90)
    pressure_drop_connector = 10000 # Pa

    if hasattr(connector, 'calculate_flow_rate'):
        flow_rate_connector = connector.calculate_flow_rate(pressure_drop_connector, fluid_properties)
        calculated_dp_connector = connector.calculate_pressure_drop(flow_rate_connector, fluid_properties)
        assert calculated_dp_connector == pytest.approx(pressure_drop_connector, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Connector")


def test_solver_on_simple_tree():
    """
    Test the solver on a simple tree network.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.01,
        temperature=40.0,
        inlet_pressure=200000.0,
        oil_type="SAE30",
        outlet_pressure=101325.0
    )

    builder = NetworkBuilder(sim_config)
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "j1", length=10, diameter=0.05)
        .add_pipe("j1", "out1", length=5, diameter=0.03)
        .add_nozzle("out1", "nozzle1", diameter=0.01)
        .add_outlet("nozzle1")
        .build()
    )

    solver = TreeSolver(sim_config)
    solution = solver.solve(network)

    assert solution['converged']
    assert solution['node_pressures'][network.get_node('inlet').id] > solution['node_pressures'][network.get_node('j1').id]
    assert solution['node_pressures'][network.get_node('j1').id] > solution['node_pressures'][network.get_node('out1').id]
    assert solution['node_pressures'][network.get_node('out1').id] > solution['node_pressures'][network.get_node('nozzle1').id]
    assert solution['node_pressures'][network.get_node('nozzle1').id] > 0


def test_solver_against_linear_solver():
    """
    Compare the new solver's results against the existing NodalMatrixSolver.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.0001, # Low flow rate to minimize non-linear effects
        temperature=40.0,
        inlet_pressure=200000.0,
        oil_type="SAE30",
        outlet_pressure=101325.0
    )

    builder = NetworkBuilder(sim_config)
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "j1", length=10, diameter=0.05)
        .add_pipe("j1", "out1", length=5, diameter=0.03)
        .add_nozzle("out1", "nozzle1", diameter=0.01)
        .add_outlet("nozzle1")
        .build()
    )

    # Solve with the new non-linear solver
    nonlinear_solver = TreeSolver(sim_config)
    nonlinear_solution = nonlinear_solver.solve(network)

    # Solve with the linear solver
    linear_solver = NodalMatrixSolver(sim_config)
    linear_solution = linear_solver.solve(network)

    # Compare the results
    assert nonlinear_solution['converged']
    for node_id in linear_solution['node_pressures']:
        assert nonlinear_solution['node_pressures'][node_id] == pytest.approx(
            linear_solution['node_pressures'][node_id], rel=1e-2
        )
    for comp_id in linear_solution['component_flows']:
        assert nonlinear_solution['component_flows'][comp_id] == pytest.approx(
            linear_solution['component_flows'][comp_id], rel=1e-2
        )


class TestTreeSolverFundamentals:
    """Comprehensive tests for TreeSolver fundamental functionality."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic simulation configuration for testing."""
        return SimulationConfig(
            temperature=55.0,
            oil_type="VG320",
            oil_density=900.0,
            total_flow_rate=0.015,  # 15 ml/s - realistic for lubrication systems
            inlet_pressure=200000.0,  # Not used as boundary condition - just for reference
            outlet_pressure=101325.0,  # Atmospheric pressure - fixed boundary condition
            max_iterations=100,
            tolerance=1e-6,
            min_resistance=1e-12
        )
    
    def test_mass_conservation_simple_network(self, basic_config):
        """Test mass conservation in a simple 2-node network."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "outlet", length=0.1, diameter=0.005)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Solver should converge for simple network"
        
        # Check that inlet flow equals outlet flow
        inlet_flow = basic_config.total_flow_rate
        outlet_flow = 0.0
        
        for connection in network.connections:
            if connection.to_node.id == network.outlet_nodes[0].id:
                outlet_flow = result['component_flows'][connection.component.id]
                break
        
        flow_error = abs(inlet_flow - outlet_flow)
        relative_error = flow_error / inlet_flow
        
        assert relative_error < 0.001, f"Mass conservation error too large: {relative_error * 100:.3f}%"
    
    def test_mass_conservation_branched_network(self, basic_config):
        """Test mass conservation in a branched network."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.05, diameter=0.006)
            .add_pipe("junction", "outlet1", length=0.03, diameter=0.004)
            .add_pipe("junction", "outlet2", length=0.04, diameter=0.003)
            .add_outlet("outlet1")
            .add_outlet("outlet2")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Solver should converge for branched network"
        
        # Check that inlet flow equals sum of outlet flows
        inlet_flow = basic_config.total_flow_rate
        total_outlet_flow = 0.0
        
        for connection in network.connections:
            for outlet in network.outlet_nodes:
                if connection.to_node.id == outlet.id:
                    total_outlet_flow += result['component_flows'][connection.component.id]
        
        flow_error = abs(inlet_flow - total_outlet_flow)
        relative_error = flow_error / inlet_flow
        
        assert relative_error < 0.01, f"Mass conservation error too large: {relative_error * 100:.3f}%"
    
    def test_nodal_mass_conservation(self, basic_config):
        """Test mass conservation at internal nodes."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "node1", length=0.02, diameter=0.006)
            .add_pipe("node1", "node2", length=0.03, diameter=0.005)
            .add_pipe("node2", "junction", length=0.04, diameter=0.004)
            .add_pipe("junction", "outlet1", length=0.02, diameter=0.003)
            .add_pipe("junction", "outlet2", length=0.03, diameter=0.003)
            .add_outlet("outlet1")
            .add_outlet("outlet2")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Solver should converge"
        
        # Check mass conservation at each internal node
        component_flows = result['component_flows']
        
        for node_id, node in network.nodes.items():
            # Skip inlet and outlet nodes
            if (node_id == network.inlet_node.id or 
                any(node_id == outlet.id for outlet in network.outlet_nodes)):
                continue
            
            net_flow = 0.0
            
            for connection in network.connections:
                flow = component_flows[connection.component.id]
                
                if connection.from_node.id == node_id:
                    net_flow -= flow  # Outgoing flow
                elif connection.to_node.id == node_id:
                    net_flow += flow  # Incoming flow
            
            relative_error = abs(net_flow) / basic_config.total_flow_rate
            assert relative_error < 0.01, f"Mass conservation at node {node_id}: {relative_error * 100:.3f}%"
    
    def test_flow_balance_verification(self, basic_config):
        """Test that flow balance is maintained across different network topologies."""
        # Test 1: Linear network
        builder = NetworkBuilder(basic_config)
        linear_network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "mid1", length=0.05, diameter=0.006)
            .add_pipe("mid1", "mid2", length=0.04, diameter=0.005)
            .add_pipe("mid2", "outlet", length=0.03, diameter=0.004)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(linear_network)
        
        assert result['converged'], "Linear network should converge"
        
        # All components should have the same flow rate
        flows = list(result['component_flows'].values())
        for flow in flows:
            relative_error = abs(flow - basic_config.total_flow_rate) / basic_config.total_flow_rate
            assert relative_error < 0.001, f"Flow deviation in linear network: {relative_error * 100:.3f}%"
        
        # Test 2: Branched network with different branch resistances
        builder = NetworkBuilder(basic_config)
        branched_network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.02, diameter=0.008)
            .add_pipe("junction", "branch1", length=0.05, diameter=0.004)  # Higher resistance
            .add_pipe("junction", "branch2", length=0.02, diameter=0.006)  # Lower resistance
            .add_pipe("junction", "branch3", length=0.03, diameter=0.005)  # Medium resistance
            .add_outlet("branch1")
            .add_outlet("branch2")
            .add_outlet("branch3")
            .build()
        )
        
        result = solver.solve(branched_network)
        assert result['converged'], "Branched network should converge"
        
        # Sum of branch flows should equal total flow
        branch_flows = []
        for connection in branched_network.connections:
            if connection.from_node.id == branched_network.get_node("junction").id:
                branch_flows.append(result['component_flows'][connection.component.id])
        
        total_branch_flow = sum(branch_flows)
        flow_error = abs(total_branch_flow - basic_config.total_flow_rate)
        relative_error = flow_error / basic_config.total_flow_rate
        
        assert relative_error < 0.01, f"Branch flow balance error: {relative_error * 100:.3f}%"
        
        # Higher resistance branch should have lower flow
        assert branch_flows[0] < branch_flows[1], "Higher resistance branch should have lower flow"
        assert branch_flows[2] < branch_flows[1], "Medium resistance branch should have lower flow than low resistance"
        assert branch_flows[0] < branch_flows[2], "High resistance branch should have lowest flow"
    
    def test_flow_distribution_physics(self, basic_config):
        """Test that flow distribution follows physical laws."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.05, diameter=0.006)
            .add_pipe("junction", "branch1", length=0.10, diameter=0.002)  # High resistance
            .add_pipe("junction", "branch2", length=0.05, diameter=0.004)  # Medium resistance
            .add_pipe("junction", "branch3", length=0.02, diameter=0.006)  # Low resistance
            .add_outlet("branch1")
            .add_outlet("branch2")
            .add_outlet("branch3")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Network should converge"
        
        # Get flow rates for each branch
        branch_flows = {}
        for connection in network.connections:
            if connection.from_node.id == network.get_node("junction").id:
                branch_flows[connection.to_node.name] = result['component_flows'][connection.component.id]
        
        # Low resistance branch should have highest flow
        assert branch_flows["branch3"] > branch_flows["branch2"], "Low resistance branch should have higher flow"
        assert branch_flows["branch2"] > branch_flows["branch1"], "Medium resistance branch should have higher flow than high resistance"
        
        # Total flow should be conserved
        total_flow = sum(branch_flows.values())
        flow_error = abs(total_flow - basic_config.total_flow_rate)
        relative_error = flow_error / basic_config.total_flow_rate
        
        assert relative_error < 0.01, f"Flow conservation error: {relative_error * 100:.3f}%"
    
    def test_pressure_monotonicity(self, basic_config):
        """Test that pressure decreases monotonically along flow paths."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "node1", length=0.05, diameter=0.006)
            .add_pipe("node1", "node2", length=0.04, diameter=0.005)
            .add_pipe("node2", "node3", length=0.03, diameter=0.004)
            .add_pipe("node3", "outlet", length=0.02, diameter=0.003)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Network should converge"
        
        # Check pressure decreases along the flow path
        pressures = result['node_pressures']
        
        assert pressures[network.get_node("inlet").id] > pressures[network.get_node("node1").id], "Pressure should decrease from inlet to node1"
        assert pressures[network.get_node("node1").id] > pressures[network.get_node("node2").id], "Pressure should decrease from node1 to node2"
        assert pressures[network.get_node("node2").id] > pressures[network.get_node("node3").id], "Pressure should decrease from node2 to node3"
        assert pressures[network.get_node("node3").id] > pressures[network.get_node("outlet").id], "Pressure should decrease from node3 to outlet"
        
        # Check that pressure at outlet matches boundary condition
        expected_outlet_pressure = basic_config.outlet_pressure
        actual_outlet_pressure = pressures[network.get_node("outlet").id]
        
        pressure_error = abs(actual_outlet_pressure - expected_outlet_pressure)
        relative_error = pressure_error / expected_outlet_pressure
        
        assert relative_error < 0.01, f"Outlet pressure error: {relative_error * 100:.3f}%"
    
    def test_pressure_distribution_branched(self, basic_config):
        """Test pressure distribution in branched networks."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.05, diameter=0.006)
            .add_pipe("junction", "branch1", length=0.03, diameter=0.004)
            .add_pipe("junction", "branch2", length=0.04, diameter=0.005)
            .add_outlet("branch1")
            .add_outlet("branch2")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Branched network should converge"
        
        pressures = result['node_pressures']
        
        assert pressures[network.get_node("inlet").id] > pressures[network.get_node("junction").id], "Pressure should decrease from inlet to junction"
        
        # Pressure should decrease from junction to both outlets
        assert pressures[network.get_node("junction").id] > pressures[network.get_node("branch1").id], "Pressure should decrease from junction to branch1"
        assert pressures[network.get_node("junction").id] > pressures[network.get_node("branch2").id], "Pressure should decrease from junction to branch2"
        
        # Both outlets should have the same pressure (boundary condition)
        pressure_diff = abs(pressures[network.get_node("branch1").id] - pressures[network.get_node("branch2").id])
        relative_diff = pressure_diff / basic_config.outlet_pressure
        
        assert relative_diff < 0.01, f"Outlet pressure difference: {relative_diff * 100:.3f}%"
    
    def test_pressure_boundary_conditions(self, basic_config):
        """Test that boundary conditions are properly enforced in flow-driven system."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "mid", length=0.05, diameter=0.005)
            .add_pipe("mid", "outlet", length=0.04, diameter=0.004)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Network should converge"
        
        pressures = result['node_pressures']
        
        # Check outlet pressure boundary condition (fixed at atmospheric)
        outlet_pressure_error = abs(pressures[network.get_node("outlet").id] - basic_config.outlet_pressure)
        outlet_relative_error = outlet_pressure_error / basic_config.outlet_pressure
        
        assert outlet_relative_error < 0.01, f"Outlet pressure boundary condition error: {outlet_relative_error * 100:.3f}%"
        
        # Check inlet pressure is reasonable (should be higher than outlet for flow)
        inlet_pressure = pressures[network.get_node("inlet").id]
        assert inlet_pressure > basic_config.outlet_pressure, f"Inlet pressure ({inlet_pressure:.0f} Pa) should be higher than outlet ({basic_config.outlet_pressure:.0f} Pa)"
        
        # Check pressure drop is reasonable for given flow rate
        pressure_drop = inlet_pressure - basic_config.outlet_pressure
        assert pressure_drop > 0, f"Pressure drop should be positive, got {pressure_drop:.0f} Pa"
    
    def test_convergence_behavior(self, basic_config):
        """Test solver convergence behavior under different conditions."""
        # Test 1: Simple network should converge quickly
        builder = NetworkBuilder(basic_config)
        simple_network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "outlet", length=0.1, diameter=0.005)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(simple_network)
        
        assert result['converged'], "Simple network should converge"
        assert result['iterations'] < 10, f"Simple network should converge quickly, took {result['iterations']} iterations"
        
        # Test 2: Complex network should still converge
        complex_network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "j1", length=0.05, diameter=0.006)
            .add_pipe("j1", "j2", length=0.04, diameter=0.005)
            .add_pipe("j2", "j3", length=0.03, diameter=0.004)
            .add_pipe("j3", "b1", length=0.02, diameter=0.003)
            .add_pipe("j3", "b2", length=0.025, diameter=0.0035)
            .add_pipe("j3", "b3", length=0.03, diameter=0.004)
            .add_outlet("b1")
            .add_outlet("b2")
            .add_outlet("b3")
            .build()
        )
        
        result = solver.solve(complex_network)
        
        assert result['converged'], "Complex network should converge"
        assert result['iterations'] < basic_config.max_iterations, f"Complex network should converge within max iterations"
        
        # Test 3: Convergence with different tolerances
        tight_config = SimulationConfig(
            temperature=basic_config.temperature,
            oil_type=basic_config.oil_type,
            oil_density=basic_config.oil_density,
            total_flow_rate=basic_config.total_flow_rate,
            inlet_pressure=basic_config.inlet_pressure,
            outlet_pressure=basic_config.outlet_pressure,
            max_iterations=basic_config.max_iterations,
            tolerance=1e-8,  # Tighter tolerance
            min_resistance=basic_config.min_resistance
        )
        
        tight_solver = TreeSolver(tight_config)
        tight_result = tight_solver.solve(simple_network)
        
        assert tight_result['converged'], "Solver should converge with tight tolerance"
        assert tight_result['iterations'] >= result['iterations'], "Tighter tolerance should require more iterations"
    
    def test_robustness_extreme_conditions(self, basic_config):
        """Test solver robustness under extreme conditions."""
        # Test 1: Very small flow rate
        small_flow_config = SimulationConfig(
            temperature=basic_config.temperature,
            oil_type=basic_config.oil_type,
            oil_density=basic_config.oil_density,
            total_flow_rate=1e-6,  # Very small flow
            inlet_pressure=basic_config.inlet_pressure,
            outlet_pressure=basic_config.outlet_pressure,
            max_iterations=basic_config.max_iterations,
            tolerance=basic_config.tolerance,
            min_resistance=basic_config.min_resistance
        )
        
        builder = NetworkBuilder(small_flow_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "mid", length=0.05, diameter=0.005)
            .add_pipe("mid", "outlet", length=0.04, diameter=0.004)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(small_flow_config)
        result = solver.solve(network)
        
        assert result['converged'], "Solver should handle very small flow rates"
        
        # Test 2: High pressure difference
        high_pressure_config = SimulationConfig(
            temperature=basic_config.temperature,
            oil_type=basic_config.oil_type,
            oil_density=basic_config.oil_density,
            total_flow_rate=basic_config.total_flow_rate,
            inlet_pressure=1000000.0,  # 10 bar
            outlet_pressure=basic_config.outlet_pressure,
            max_iterations=basic_config.max_iterations,
            tolerance=basic_config.tolerance,
            min_resistance=basic_config.min_resistance
        )
        
        solver = TreeSolver(high_pressure_config)
        result = solver.solve(network)
        
        assert result['converged'], "Solver should handle high pressure differences"
        
        # Flow should be higher with higher pressure difference
        flow_values = list(result['component_flows'].values())
        assert all(flow > 0 for flow in flow_values), "All flows should be positive"
    
    def test_solver_consistency(self, basic_config):
        """Test that solver produces consistent results."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.05, diameter=0.006)
            .add_pipe("junction", "branch1", length=0.03, diameter=0.004)
            .add_pipe("junction", "branch2", length=0.04, diameter=0.005)
            .add_outlet("branch1")
            .add_outlet("branch2")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        
        # Solve the same network multiple times
        results = []
        for i in range(5):
            result = solver.solve(network)
            assert result['converged'], f"Solution {i+1} should converge"
            results.append(result)
        
        # All results should be identical
        base_result = results[0]
        for i, result in enumerate(results[1:], 1):
            # Check node pressures
            for node_id in base_result['node_pressures']:
                pressure_diff = abs(result['node_pressures'][node_id] - base_result['node_pressures'][node_id])
                relative_diff = pressure_diff / base_result['node_pressures'][node_id]
                assert relative_diff < 1e-10, f"Pressure inconsistency at node {node_id} in solution {i+1}"
            
            # Check component flows
            for comp_id in base_result['component_flows']:
                flow_diff = abs(result['component_flows'][comp_id] - base_result['component_flows'][comp_id])
                relative_diff = flow_diff / abs(base_result['component_flows'][comp_id])
                assert relative_diff < 1e-10, f"Flow inconsistency at component {comp_id} in solution {i+1}"
    
    def test_boundary_condition_enforcement(self, basic_config):
        """Test that boundary conditions are properly enforced in flow-driven system."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "node1", length=0.05, diameter=0.006)
            .add_pipe("node1", "node2", length=0.04, diameter=0.005)
            .add_pipe("node2", "outlet", length=0.03, diameter=0.004)
            .add_outlet("outlet")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Network should converge"
        
        pressures = result['node_pressures']
        
        # Test outlet pressure boundary condition (fixed at atmospheric)
        outlet_pressure_error = abs(pressures[network.get_node("outlet").id] - basic_config.outlet_pressure)
        outlet_relative_error = outlet_pressure_error / basic_config.outlet_pressure
        
        assert outlet_relative_error < 1e-6, f"Outlet pressure boundary condition not enforced: {outlet_relative_error * 100:.6f}%"
        
        # Test flow rate boundary condition (total flow should match specified)
        total_flow = result['total_flow_rate']
        flow_error = abs(total_flow - basic_config.total_flow_rate)
        flow_relative_error = flow_error / basic_config.total_flow_rate
        
        assert flow_relative_error < 1e-6, f"Flow rate boundary condition not enforced: {flow_relative_error * 100:.6f}%"
        
        # Test pressure monotonicity (inlet > intermediate > outlet)
        inlet_pressure = pressures[network.get_node("inlet").id]
        node1_pressure = pressures[network.get_node("node1").id]
        node2_pressure = pressures[network.get_node("node2").id]
        outlet_pressure = pressures[network.get_node("outlet").id]
        
        assert inlet_pressure > node1_pressure, f"Pressure should decrease from inlet to node1"
        assert node1_pressure > node2_pressure, f"Pressure should decrease from node1 to node2"
        assert node2_pressure > outlet_pressure, f"Pressure should decrease from node2 to outlet"
    
    def test_multiple_outlet_boundary_conditions(self, basic_config):
        """Test boundary condition enforcement with multiple outlets."""
        builder = NetworkBuilder(basic_config)
        network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "junction", length=0.05, diameter=0.006)
            .add_pipe("junction", "outlet1", length=0.03, diameter=0.004)
            .add_pipe("junction", "outlet2", length=0.04, diameter=0.005)
            .add_pipe("junction", "outlet3", length=0.035, diameter=0.0045)
            .add_outlet("outlet1")
            .add_outlet("outlet2")
            .add_outlet("outlet3")
            .build()
        )
        
        solver = TreeSolver(basic_config)
        result = solver.solve(network)
        
        assert result['converged'], "Multi-outlet network should converge"
        
        pressures = result['node_pressures']
        
        # All outlets should have the same pressure (boundary condition)
        outlet_pressures = [pressures[network.get_node("outlet1").id], pressures[network.get_node("outlet2").id], pressures[network.get_node("outlet3").id]]
        
        for i, pressure in enumerate(outlet_pressures):
            pressure_error = abs(pressure - basic_config.outlet_pressure)
            relative_error = pressure_error / basic_config.outlet_pressure
            
            assert relative_error < 1e-6, f"Outlet {i+1} pressure boundary condition not enforced: {relative_error * 100:.6f}%"
        
        # Sum of outlet flows should equal total flow
        total_outlet_flow = 0.0
        outlet_node_ids = [network.get_node("outlet1").id, network.get_node("outlet2").id, network.get_node("outlet3").id]
        for connection in network.connections:
            if connection.to_node.id in outlet_node_ids:
                total_outlet_flow += result['component_flows'][connection.component.id]
        
        flow_error = abs(total_outlet_flow - basic_config.total_flow_rate)
        flow_relative_error = flow_error / basic_config.total_flow_rate
        
        assert flow_relative_error < 1e-3, f"Total flow boundary condition not enforced: {flow_relative_error * 100:.3f}%"
    
    def test_solver_error_handling(self, basic_config):
        """Test solver error handling and graceful failures."""
        builder = NetworkBuilder(basic_config)
        
        # Test 1: Network with no connections should fail gracefully
        with pytest.raises(ValueError, match="Constructed network is invalid"):
            (builder
                .set_inlet("inlet")
                .add_outlet("outlet")
                .build()
            )
        
        # Test 2: Unrealistic parameters should be handled
        unrealistic_config = SimulationConfig(
            temperature=basic_config.temperature,
            oil_type=basic_config.oil_type,
            oil_density=basic_config.oil_density,
            total_flow_rate=1000.0,  # Unrealistically high flow
            inlet_pressure=basic_config.inlet_pressure,
            outlet_pressure=basic_config.outlet_pressure,
            max_iterations=5,  # Very few iterations
            tolerance=basic_config.tolerance,
            min_resistance=basic_config.min_resistance
        )
        
        simple_network = (builder
            .set_inlet("inlet")
            .add_pipe("inlet", "outlet", length=0.1, diameter=0.005)
            .add_outlet("outlet")
            .build()
        )
        
        unrealistic_solver = TreeSolver(unrealistic_config)
        result = unrealistic_solver.solve(simple_network)
        
        # Should either converge or fail gracefully
        if 'converged' in result:
            assert isinstance(result['converged'], bool), "Converged flag should be boolean"
            assert isinstance(result['iterations'], int), "Iterations should be integer"
            assert result['iterations'] <= unrealistic_config.max_iterations, "Should not exceed max iterations"
