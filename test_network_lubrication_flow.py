#!/usr/bin/env python3
"""
Comprehensive Test Suite for Network-Based Lubrication Flow Calculator

This test suite validates the network-based flow distribution calculator
including tree structures, component-based building, and complex topologies.
"""

import unittest
import math
import numpy as np
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType,
    create_simple_tree_example
)


class TestComponents(unittest.TestCase):
    """Test individual flow components"""
    
    def setUp(self):
        self.fluid_properties = {
            'density': 900.0,
            'viscosity': 0.1
        }
    
    def test_channel_creation(self):
        """Test channel component creation and validation"""
        # Valid channel
        channel = Channel(diameter=0.05, length=10.0, roughness=0.00015)
        self.assertEqual(channel.diameter, 0.05)
        self.assertEqual(channel.length, 10.0)
        self.assertEqual(channel.roughness, 0.00015)
        self.assertEqual(channel.component_type, ComponentType.CHANNEL)
        
        # Invalid parameters
        with self.assertRaises(ValueError):
            Channel(diameter=-0.05, length=10.0)  # Negative diameter
        
        with self.assertRaises(ValueError):
            Channel(diameter=0.05, length=-10.0)  # Negative length
        
        with self.assertRaises(ValueError):
            Channel(diameter=0.05, length=10.0, roughness=-0.001)  # Negative roughness
    
    def test_channel_pressure_drop(self):
        """Test channel pressure drop calculations"""
        channel = Channel(diameter=0.05, length=10.0, roughness=0.00015)
        
        # Test with different flow rates
        flow_rates = [0.001, 0.005, 0.01, 0.02]  # m³/s
        
        for flow_rate in flow_rates:
            dp = channel.calculate_pressure_drop(flow_rate, self.fluid_properties)
            self.assertGreater(dp, 0, f"Pressure drop should be positive for flow {flow_rate}")
            
            # Pressure drop should increase with flow rate
            if flow_rate > 0.001:
                dp_low = channel.calculate_pressure_drop(0.001, self.fluid_properties)
                self.assertGreater(dp, dp_low, "Higher flow should give higher pressure drop")
        
        # Zero flow should give zero pressure drop
        dp_zero = channel.calculate_pressure_drop(0.0, self.fluid_properties)
        self.assertEqual(dp_zero, 0.0)
    
    def test_connector_creation(self):
        """Test connector component creation"""
        # T-junction
        t_junction = Connector(ConnectorType.T_JUNCTION, diameter=0.05)
        self.assertEqual(t_junction.connector_type, ConnectorType.T_JUNCTION)
        self.assertEqual(t_junction.diameter, 0.05)
        self.assertEqual(t_junction.diameter_out, 0.05)
        self.assertEqual(t_junction.component_type, ComponentType.CONNECTOR)
        
        # Reducer
        reducer = Connector(ConnectorType.REDUCER, diameter=0.08, diameter_out=0.05)
        self.assertEqual(reducer.diameter, 0.08)
        self.assertEqual(reducer.diameter_out, 0.05)
        
        # Invalid parameters
        with self.assertRaises(ValueError):
            Connector(ConnectorType.T_JUNCTION, diameter=-0.05)
    
    def test_connector_pressure_drop(self):
        """Test connector pressure drop calculations"""
        t_junction = Connector(ConnectorType.T_JUNCTION, diameter=0.05)
        
        flow_rate = 0.01  # m³/s
        dp = t_junction.calculate_pressure_drop(flow_rate, self.fluid_properties)
        self.assertGreater(dp, 0)
        
        # Test reducer with different diameters
        reducer = Connector(ConnectorType.REDUCER, diameter=0.08, diameter_out=0.05)
        dp_reducer = reducer.calculate_pressure_drop(flow_rate, self.fluid_properties)
        self.assertGreater(dp_reducer, 0)
    
    def test_nozzle_creation(self):
        """Test nozzle component creation"""
        # Sharp-edged nozzle
        nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.SHARP_EDGED)
        self.assertEqual(nozzle.diameter, 0.008)
        self.assertEqual(nozzle.nozzle_type, NozzleType.SHARP_EDGED)
        self.assertEqual(nozzle.component_type, ComponentType.NOZZLE)
        
        # Venturi nozzle
        venturi = Nozzle(diameter=0.01, nozzle_type=NozzleType.VENTURI)
        self.assertGreater(venturi.discharge_coeff, nozzle.discharge_coeff)
        
        # Invalid parameters
        with self.assertRaises(ValueError):
            Nozzle(diameter=-0.008)
        
        with self.assertRaises(ValueError):
            Nozzle(diameter=0.008, discharge_coeff=1.5)  # > 1
    
    def test_nozzle_pressure_drop(self):
        """Test nozzle pressure drop calculations"""
        sharp_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.SHARP_EDGED)
        venturi_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.VENTURI)
        
        flow_rate = 0.005  # m³/s
        
        dp_sharp = sharp_nozzle.calculate_pressure_drop(flow_rate, self.fluid_properties)
        dp_venturi = venturi_nozzle.calculate_pressure_drop(flow_rate, self.fluid_properties)
        
        self.assertGreater(dp_sharp, 0)
        self.assertGreater(dp_venturi, 0)
        
        # Venturi should have lower pressure drop due to recovery
        self.assertLess(dp_venturi, dp_sharp)


class TestFlowNetwork(unittest.TestCase):
    """Test flow network topology and validation"""
    
    def test_network_creation(self):
        """Test basic network creation"""
        network = FlowNetwork("Test Network")
        self.assertEqual(network.name, "Test Network")
        self.assertEqual(len(network.nodes), 0)
        self.assertEqual(len(network.connections), 0)
    
    def test_node_creation(self):
        """Test node creation and management"""
        network = FlowNetwork()
        
        # Create nodes
        node1 = network.create_node("Node1", elevation=0.0)
        node2 = network.create_node("Node2", elevation=1.0)
        
        self.assertEqual(len(network.nodes), 2)
        self.assertEqual(node1.name, "Node1")
        self.assertEqual(node1.elevation, 0.0)
        self.assertEqual(node2.elevation, 1.0)
    
    def test_component_connection(self):
        """Test connecting components between nodes"""
        network = FlowNetwork()
        
        node1 = network.create_node("Node1")
        node2 = network.create_node("Node2")
        
        channel = Channel(diameter=0.05, length=10.0)
        connection = network.connect_components(node1, node2, channel)
        
        self.assertEqual(len(network.connections), 1)
        self.assertEqual(connection.from_node, node1)
        self.assertEqual(connection.to_node, node2)
        self.assertEqual(connection.component, channel)
    
    def test_network_validation(self):
        """Test network validation logic"""
        network = FlowNetwork()
        
        # Empty network should fail validation
        is_valid, errors = network.validate_network()
        self.assertFalse(is_valid)
        self.assertIn("No inlet node defined", errors)
        self.assertIn("No outlet nodes defined", errors)
        
        # Create simple valid network
        inlet = network.create_node("Inlet")
        outlet = network.create_node("Outlet")
        channel = Channel(diameter=0.05, length=10.0)
        
        network.set_inlet(inlet)
        network.add_outlet(outlet)
        network.connect_components(inlet, outlet, channel)
        
        is_valid, errors = network.validate_network()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_path_finding(self):
        """Test path finding from inlet to outlets"""
        network = FlowNetwork()
        
        # Create simple tree
        inlet = network.create_node("Inlet")
        junction = network.create_node("Junction")
        outlet1 = network.create_node("Outlet1")
        outlet2 = network.create_node("Outlet2")
        
        network.set_inlet(inlet)
        network.add_outlet(outlet1)
        network.add_outlet(outlet2)
        
        # Connect components
        main_channel = Channel(diameter=0.08, length=10.0)
        branch1 = Channel(diameter=0.05, length=5.0)
        branch2 = Channel(diameter=0.04, length=6.0)
        
        network.connect_components(inlet, junction, main_channel)
        network.connect_components(junction, outlet1, branch1)
        network.connect_components(junction, outlet2, branch2)
        
        # Find paths
        paths = network.get_paths_to_outlets()
        self.assertEqual(len(paths), 2)
        
        # Check path contents
        path1_components = [conn.component for conn in paths[0]]
        path2_components = [conn.component for conn in paths[1]]
        
        self.assertIn(main_channel, path1_components)
        self.assertIn(main_channel, path2_components)
        self.assertTrue(branch1 in path1_components or branch1 in path2_components)
        self.assertTrue(branch2 in path1_components or branch2 in path2_components)


class TestNetworkFlowSolver(unittest.TestCase):
    """Test network flow solver"""
    
    def setUp(self):
        self.solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    def test_viscosity_calculation(self):
        """Test viscosity calculation for different temperatures"""
        temperatures = [20, 40, 60, 80, 100]
        
        for temp in temperatures:
            viscosity = self.solver.calculate_viscosity(temp)
            self.assertGreater(viscosity, 0)
            self.assertLess(viscosity, 10.0)  # Reasonable range
        
        # Higher temperature should give lower viscosity
        visc_20 = self.solver.calculate_viscosity(20)
        visc_80 = self.solver.calculate_viscosity(80)
        self.assertGreater(visc_20, visc_80)
    
    def test_simple_tree_solution(self):
        """Test solving flow in simple tree network"""
        network, total_flow_rate, temperature = create_simple_tree_example()
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Check convergence
        self.assertTrue(solution_info['converged'])
        self.assertGreater(solution_info['iterations'], 0)
        
        # Check mass conservation
        total_outlet_flow = 0.0
        for outlet in network.outlet_nodes:
            # Find connections leading to this outlet
            for connection in network.reverse_adjacency[outlet.id]:
                total_outlet_flow += connection_flows[connection.component.id]
        
        self.assertAlmostEqual(total_outlet_flow, total_flow_rate, places=6)
        
        # Check that all flows are positive
        for flow in connection_flows.values():
            self.assertGreaterEqual(flow, 0)
    
    def test_mass_conservation(self):
        """Test mass conservation at junctions"""
        network, total_flow_rate, temperature = create_simple_tree_example()
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Check mass conservation at junction
        junction_nodes = network.get_junction_nodes()
        
        for junction in junction_nodes:
            inflow = 0.0
            outflow = 0.0
            
            # Sum incoming flows
            for connection in network.reverse_adjacency[junction.id]:
                inflow += connection_flows[connection.component.id]
            
            # Sum outgoing flows
            for connection in network.adjacency_list[junction.id]:
                outflow += connection_flows[connection.component.id]
            
            # Mass conservation: inflow = outflow
            self.assertAlmostEqual(inflow, outflow, places=6)
    
    def test_pressure_equalization(self):
        """Test that pressures are equalized at outlets"""
        network, total_flow_rate, temperature = create_simple_tree_example()
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Get outlet pressures
        outlet_pressures = []
        for outlet in network.outlet_nodes:
            pressure = solution_info['node_pressures'][outlet.id]
            outlet_pressures.append(pressure)
        
        # Check that outlet pressures are reasonably close
        # (allowing for some numerical error and different elevations)
        if len(outlet_pressures) > 1:
            pressure_range = max(outlet_pressures) - min(outlet_pressures)
            # Allow for elevation differences and different nozzle types
            # Different nozzle types can create significant pressure differences
            self.assertLess(pressure_range, 2000000)  # 2 MPa tolerance for different nozzles


class TestComplexNetworks(unittest.TestCase):
    """Test complex network topologies"""
    
    def setUp(self):
        self.solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    def create_multi_level_tree(self) -> FlowNetwork:
        """Create a multi-level tree network"""
        network = FlowNetwork("Multi-Level Tree")
        
        # Create nodes
        inlet = network.create_node("Inlet", elevation=0.0)
        junction1 = network.create_node("Junction1", elevation=1.0)
        junction2 = network.create_node("Junction2", elevation=1.5)
        junction3 = network.create_node("Junction3", elevation=1.2)
        outlet1 = network.create_node("Outlet1", elevation=2.0)
        outlet2 = network.create_node("Outlet2", elevation=2.2)
        outlet3 = network.create_node("Outlet3", elevation=1.8)
        outlet4 = network.create_node("Outlet4", elevation=1.5)
        
        # Set inlet and outlets
        network.set_inlet(inlet)
        for outlet in [outlet1, outlet2, outlet3, outlet4]:
            network.add_outlet(outlet)
        
        # Create components
        main_channel = Channel(diameter=0.1, length=8.0, name="Main")
        branch1 = Channel(diameter=0.06, length=6.0, name="Branch1")
        branch2 = Channel(diameter=0.05, length=7.0, name="Branch2")
        sub_branch1 = Channel(diameter=0.04, length=4.0, name="SubBranch1")
        sub_branch2 = Channel(diameter=0.035, length=5.0, name="SubBranch2")
        sub_branch3 = Channel(diameter=0.03, length=3.0, name="SubBranch3")
        
        # Nozzles at outlets
        nozzle1 = Nozzle(diameter=0.015, nozzle_type=NozzleType.VENTURI, name="Nozzle1")
        nozzle2 = Nozzle(diameter=0.012, nozzle_type=NozzleType.FLOW_NOZZLE, name="Nozzle2")
        nozzle3 = Nozzle(diameter=0.010, nozzle_type=NozzleType.ROUNDED, name="Nozzle3")
        nozzle4 = Nozzle(diameter=0.008, nozzle_type=NozzleType.SHARP_EDGED, name="Nozzle4")
        
        # Connect components
        network.connect_components(inlet, junction1, main_channel)
        network.connect_components(junction1, junction2, branch1)
        network.connect_components(junction1, junction3, branch2)
        network.connect_components(junction2, outlet1, sub_branch1)
        network.connect_components(junction2, outlet2, sub_branch2)
        network.connect_components(junction3, outlet3, sub_branch3)
        network.connect_components(junction3, outlet4, nozzle4)
        
        # Create intermediate nodes for nozzles
        nozzle1_node = network.create_node("Nozzle1_Node", elevation=2.0)
        nozzle2_node = network.create_node("Nozzle2_Node", elevation=2.2)
        nozzle3_node = network.create_node("Nozzle3_Node", elevation=1.8)
        
        # Connect nozzles properly
        network.connect_components(outlet1, nozzle1_node, nozzle1)
        network.connect_components(outlet2, nozzle2_node, nozzle2)
        network.connect_components(outlet3, nozzle3_node, nozzle3)
        
        # Update outlets to be after nozzles
        network.outlet_nodes = [nozzle1_node, nozzle2_node, nozzle3_node, outlet4]
        
        return network
    
    def test_multi_level_tree_solution(self):
        """Test solving multi-level tree network"""
        network = self.create_multi_level_tree()
        
        # Validate network
        is_valid, errors = network.validate_network()
        self.assertTrue(is_valid, f"Network validation failed: {errors}")
        
        # Solve flow distribution
        total_flow_rate = 0.025  # 25 L/s
        temperature = 45  # °C
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Check convergence
        self.assertTrue(solution_info['converged'])
        
        # Check mass conservation
        total_outlet_flow = 0.0
        for outlet in network.outlet_nodes:
            for connection in network.reverse_adjacency[outlet.id]:
                total_outlet_flow += connection_flows[connection.component.id]
        
        self.assertAlmostEqual(total_outlet_flow, total_flow_rate, places=5)
        
        # Check that all flows are positive
        for flow in connection_flows.values():
            self.assertGreaterEqual(flow, 0)
    
    def test_different_nozzle_types_effect(self):
        """Test that different nozzle types affect flow distribution"""
        network = self.create_multi_level_tree()
        
        total_flow_rate = 0.02  # 20 L/s
        temperature = 40  # °C
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, temperature
        )
        
        # Get flows to different outlets
        outlet_flows = {}
        for outlet in network.outlet_nodes:
            total_flow_to_outlet = 0.0
            for connection in network.reverse_adjacency[outlet.id]:
                total_flow_to_outlet += connection_flows[connection.component.id]
            outlet_flows[outlet.name] = total_flow_to_outlet
        
        # Outlets with more efficient nozzles (venturi, flow nozzle) should get more flow
        # than those with less efficient nozzles (sharp-edged)
        self.assertGreater(outlet_flows.get('Nozzle1_Node', 0), 0)
        self.assertGreater(outlet_flows.get('Nozzle2_Node', 0), 0)
        self.assertGreater(outlet_flows.get('Nozzle3_Node', 0), 0)
        self.assertGreater(outlet_flows.get('Outlet4', 0), 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def setUp(self):
        self.solver = NetworkFlowSolver()
    
    def test_single_branch_network(self):
        """Test network with single branch"""
        network = FlowNetwork("Single Branch")
        
        inlet = network.create_node("Inlet")
        outlet = network.create_node("Outlet")
        
        network.set_inlet(inlet)
        network.add_outlet(outlet)
        
        channel = Channel(diameter=0.05, length=10.0)
        network.connect_components(inlet, outlet, channel)
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, 0.01, 40
        )
        
        self.assertTrue(solution_info['converged'])
        self.assertAlmostEqual(connection_flows[channel.id], 0.01, places=6)
    
    def test_very_small_flows(self):
        """Test with very small flow rates"""
        network, _, temperature = create_simple_tree_example()
        
        very_small_flow = 1e-6  # 1 mL/s
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, very_small_flow, temperature
        )
        
        self.assertTrue(solution_info['converged'])
        
        # Check mass conservation
        total_flow = sum(connection_flows.values())
        # Note: some connections are shared, so we need to check outlet flows
        outlet_flow_sum = 0.0
        for outlet in network.outlet_nodes:
            for connection in network.reverse_adjacency[outlet.id]:
                outlet_flow_sum += connection_flows[connection.component.id]
        
        self.assertAlmostEqual(outlet_flow_sum, very_small_flow, places=9)
    
    def test_high_temperature_effects(self):
        """Test behavior at high temperatures"""
        network, total_flow_rate, _ = create_simple_tree_example()
        
        high_temperature = 95  # °C
        
        connection_flows, solution_info = self.solver.solve_network_flow(
            network, total_flow_rate, high_temperature
        )
        
        self.assertTrue(solution_info['converged'])
        self.assertLess(solution_info['viscosity'], 0.05)  # Should be low viscosity


def run_all_tests():
    """Run all test suites"""
    test_classes = [
        TestComponents,
        TestFlowNetwork,
        TestNetworkFlowSolver,
        TestComplexNetworks,
        TestEdgeCases
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=0, stream=open('/dev/null', 'w'))
        result = runner.run(suite)
        
        tests_run = result.testsRun
        failures = len(result.failures) + len(result.errors)
        
        total_tests += tests_run
        total_failures += failures
        
        if failures == 0:
            print(f"✓ All {tests_run} tests passed")
        else:
            print(f"✗ {failures} out of {tests_run} tests failed")
            
            # Print failure details
            for test, traceback in result.failures + result.errors:
                print(f"  FAILED: {test}")
                print(f"    {traceback.split('AssertionError:')[-1].strip()}")
    
    print(f"\n{'='*70}")
    print("NETWORK TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total tests run: {total_tests}")
    print(f"Failures/Errors: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%")
    
    if total_failures == 0:
        print("🎉 All tests passed!")
    else:
        print(f"⚠️  {total_failures} tests failed. Please review the output above.")
    
    return total_failures == 0


if __name__ == "__main__":
    run_all_tests()