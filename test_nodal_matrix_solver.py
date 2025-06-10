"""
Unit tests for the Iterative Nodal-Matrix Solver

This module contains comprehensive tests for the nodal matrix solver, including:
1. Simple Y-network test with linear resistances
2. Verification against analytical solutions
3. Non-linear resistance tests
4. Convergence tests
"""

import unittest
import numpy as np
import logging
from typing import Dict

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver


class LinearResistanceComponent(FlowComponent):
    """Simple component with linear resistance for testing"""
    
    def __init__(self, resistance: float, component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.resistance = resistance  # Pa·s/m³
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Linear pressure drop: ΔP = R * Q"""
        return self.resistance * abs(flow_rate)
    
    def get_flow_area(self) -> float:
        return 1e-4  # 1 cm²


class NonLinearResistanceComponent(FlowComponent):
    """Component with quadratic resistance for testing"""
    
    def __init__(self, linear_coeff: float, quadratic_coeff: float, component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.linear_coeff = linear_coeff      # Pa·s/m³
        self.quadratic_coeff = quadratic_coeff  # Pa·s²/m⁶
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Non-linear pressure drop: ΔP = a*Q + b*Q²"""
        Q = abs(flow_rate)
        return self.linear_coeff * Q + self.quadratic_coeff * Q * Q
    
    def get_flow_area(self) -> float:
        return 1e-4  # 1 cm²


class TestNodalMatrixSolver(unittest.TestCase):
    """Test cases for the nodal matrix solver"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Configure logging for debugging
        logging.basicConfig(level=logging.INFO)
        self.solver = NodalMatrixSolver()
        
        # Standard fluid properties for testing
        self.fluid_properties = {
            'density': 900.0,    # kg/m³
            'viscosity': 0.01    # Pa·s
        }
    
    def test_simple_y_network_linear(self):
        """
        Test a simple Y-network with linear resistances.
        
        Network topology:
        Source (A) --R1--> Junction (B) --R2--> Sink1 (C)
                                    |
                                   R3
                                    |
                                    v
                                 Sink2 (D)
        
        Modified to use a common sink approach:
        Source (A) --R1--> Junction (B) --R2--> Sink (C)
                                    |
                                   R3
                                    |
                                    v
                                 Sink (C)
        
        For linear resistances, we can solve analytically:
        - Total resistance from B to sink: R_parallel = 1/(1/R2 + 1/R3)
        - Total resistance: R_total = R1 + R_parallel
        - Current through R1: I = V_total / R_total
        - Voltage at B: V_B = V_total - I * R1
        - Currents through R2, R3: I2 = V_B/R2, I3 = V_B/R3
        """
        # Create network
        network = FlowNetwork("Y-Network Test")
        
        # Create nodes - use single sink for both outlets
        source = network.create_node("Source")
        junction = network.create_node("Junction") 
        sink = network.create_node("Sink")  # Common sink for both paths
        
        # Define resistances (Pa·s/m³)
        R1 = 1000.0
        R2 = 2000.0
        R3 = 3000.0
        
        # Create components with linear resistances
        comp1 = LinearResistanceComponent(R1, "comp1", "R1")
        comp2 = LinearResistanceComponent(R2, "comp2", "R2") 
        comp3 = LinearResistanceComponent(R3, "comp3", "R3")
        
        # Connect components - both R2 and R3 go to the same sink
        network.connect_components(source, junction, comp1)
        network.connect_components(junction, sink, comp2)
        network.connect_components(junction, sink, comp3)
        
        # Set inlet and outlet
        network.set_inlet(source)
        network.add_outlet(sink)
        
        # Test parameters
        Q_total = 0.001  # 1 L/s
        
        # Solve using nodal matrix solver
        pressures, flows = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties,
            tol_flow=1e-8,
            tol_pressure=1e-3
        )
        
        # Analytical solution
        # Parallel resistance of R2 and R3
        R_parallel = 1.0 / (1.0/R2 + 1.0/R3)  # = R2*R3/(R2+R3) = 1200
        R_total = R1 + R_parallel  # = 2200
        
        # Flow through R1 (total flow)
        Q1_expected = Q_total
        
        # Pressure at junction (relative to sink)
        # P_junction = Q1 * R_parallel = 0.001 * 1200 = 1.2 Pa
        P_junction_expected = Q1_expected * R_parallel
        
        # Pressure at source (relative to sink)  
        # P_source = Q1 * R_total = 0.001 * 2200 = 2.2 Pa
        P_source_expected = Q1_expected * R_total
        
        # Flows through R2 and R3
        Q2_expected = P_junction_expected / R2  # = 1.2 / 2000 = 0.0006
        Q3_expected = P_junction_expected / R3  # = 1.2 / 3000 = 0.0004
        
        # Verify pressures
        self.assertAlmostEqual(pressures[sink.id], 0.0, places=6, 
                              msg="Sink pressure should be 0 (reference)")
        self.assertAlmostEqual(pressures[junction.id], P_junction_expected, places=3,
                              msg=f"Junction pressure mismatch: expected {P_junction_expected}, got {pressures[junction.id]}")
        self.assertAlmostEqual(pressures[source.id], P_source_expected, places=3,
                              msg=f"Source pressure mismatch: expected {P_source_expected}, got {pressures[source.id]}")
        
        # Verify flows
        self.assertAlmostEqual(flows["comp1"], Q1_expected, places=6,
                              msg=f"Flow through R1 mismatch: expected {Q1_expected}, got {flows['comp1']}")
        self.assertAlmostEqual(flows["comp2"], Q2_expected, places=6,
                              msg=f"Flow through R2 mismatch: expected {Q2_expected}, got {flows['comp2']}")
        self.assertAlmostEqual(flows["comp3"], Q3_expected, places=6,
                              msg=f"Flow through R3 mismatch: expected {Q3_expected}, got {flows['comp3']}")
        
        # Verify mass conservation
        self.assertAlmostEqual(Q2_expected + Q3_expected, Q1_expected, places=6,
                              msg="Mass conservation violated")
        
        print(f"✓ Y-Network Test Passed")
        print(f"  Pressures: Source={pressures[source.id]:.3f} Pa, Junction={pressures[junction.id]:.3f} Pa")
        print(f"  Flows: Q1={flows['comp1']:.6f}, Q2={flows['comp2']:.6f}, Q3={flows['comp3']:.6f} m³/s")
    
    def test_series_network_linear(self):
        """Test a simple series network with linear resistances"""
        # Create network: Source --R1--> Node1 --R2--> Sink
        network = FlowNetwork("Series Network Test")
        
        source = network.create_node("Source")
        node1 = network.create_node("Node1")
        sink = network.create_node("Sink")
        
        R1 = 1000.0
        R2 = 2000.0
        
        comp1 = LinearResistanceComponent(R1, "comp1", "R1")
        comp2 = LinearResistanceComponent(R2, "comp2", "R2")
        
        network.connect_components(source, node1, comp1)
        network.connect_components(node1, sink, comp2)
        
        network.set_inlet(source)
        network.add_outlet(sink)
        
        Q_total = 0.001  # 1 L/s
        
        pressures, flows = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties
        )
        
        # Analytical solution for series circuit
        R_total = R1 + R2  # = 3000
        Q_expected = Q_total  # Same flow through both resistors
        P_source_expected = Q_total * R_total  # = 3.0 Pa
        P_node1_expected = Q_total * R2  # = 2.0 Pa (pressure drop across R2 to sink)
        
        self.assertAlmostEqual(pressures[sink.id], 0.0, places=6)
        self.assertAlmostEqual(pressures[node1.id], P_node1_expected, places=3)
        self.assertAlmostEqual(pressures[source.id], P_source_expected, places=3)
        self.assertAlmostEqual(flows["comp1"], Q_expected, places=6)
        self.assertAlmostEqual(flows["comp2"], Q_expected, places=6)
        
        print(f"✓ Series Network Test Passed")
    
    def test_parallel_network_linear(self):
        """Test a parallel network with linear resistances"""
        # Create network: Source splits into two parallel paths to Sink
        network = FlowNetwork("Parallel Network Test")
        
        source = network.create_node("Source")
        sink = network.create_node("Sink")
        
        R1 = 1000.0
        R2 = 2000.0
        
        comp1 = LinearResistanceComponent(R1, "comp1", "R1")
        comp2 = LinearResistanceComponent(R2, "comp2", "R2")
        
        network.connect_components(source, sink, comp1)
        network.connect_components(source, sink, comp2)
        
        network.set_inlet(source)
        network.add_outlet(sink)
        
        Q_total = 0.001  # 1 L/s
        
        pressures, flows = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties
        )
        
        # Analytical solution for parallel circuit
        R_parallel = 1.0 / (1.0/R1 + 1.0/R2)  # = 666.67
        P_source_expected = Q_total * R_parallel
        Q1_expected = P_source_expected / R1
        Q2_expected = P_source_expected / R2
        
        self.assertAlmostEqual(pressures[sink.id], 0.0, places=6)
        self.assertAlmostEqual(pressures[source.id], P_source_expected, places=3)
        self.assertAlmostEqual(flows["comp1"], Q1_expected, places=6)
        self.assertAlmostEqual(flows["comp2"], Q2_expected, places=6)
        self.assertAlmostEqual(Q1_expected + Q2_expected, Q_total, places=6)
        
        print(f"✓ Parallel Network Test Passed")
    
    def test_nonlinear_resistance(self):
        """Test solver with non-linear resistances"""
        # Simple series network with quadratic resistance
        network = FlowNetwork("Non-linear Test")
        
        source = network.create_node("Source")
        sink = network.create_node("Sink")
        
        # Component with ΔP = 1000*Q + 500000*Q²
        comp = NonLinearResistanceComponent(1000.0, 500000.0, "comp1", "NonLinear")
        network.connect_components(source, sink, comp)
        
        network.set_inlet(source)
        network.add_outlet(sink)
        
        Q_total = 0.001  # 1 L/s
        
        pressures, flows = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties,
            max_iter=50
        )
        
        # Verify flow equals input
        self.assertAlmostEqual(flows["comp1"], Q_total, places=6)
        
        # Verify pressure drop matches non-linear law
        expected_pressure_drop = 1000.0 * Q_total + 500000.0 * Q_total * Q_total
        actual_pressure_drop = pressures[source.id] - pressures[sink.id]
        self.assertAlmostEqual(actual_pressure_drop, expected_pressure_drop, places=3)
        
        print(f"✓ Non-linear Resistance Test Passed")
        print(f"  Expected ΔP: {expected_pressure_drop:.3f} Pa, Actual: {actual_pressure_drop:.3f} Pa")
    
    def test_convergence_tolerance(self):
        """Test that solver respects convergence tolerances"""
        network = FlowNetwork("Convergence Test")
        
        source = network.create_node("Source")
        sink = network.create_node("Sink")
        
        comp = LinearResistanceComponent(1000.0, "comp1", "Linear")
        network.connect_components(source, sink, comp)
        
        network.set_inlet(source)
        network.add_outlet(sink)
        
        Q_total = 0.001
        
        # Test with tight tolerance
        pressures1, flows1 = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties,
            tol_flow=1e-10,
            tol_pressure=1e-6
        )
        
        # Test with loose tolerance
        pressures2, flows2 = self.solver.solve_nodal_iterative(
            network=network,
            source_node_id=source.id,
            sink_node_id=sink.id,
            Q_total=Q_total,
            fluid_properties=self.fluid_properties,
            tol_flow=1e-3,
            tol_pressure=1e3
        )
        
        # Both should give similar results for linear case
        self.assertAlmostEqual(flows1["comp1"], flows2["comp1"], places=3)
        
        print(f"✓ Convergence Tolerance Test Passed")


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("=" * 60)
    print("NODAL MATRIX SOLVER - COMPREHENSIVE TESTS")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNodalMatrixSolver)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_comprehensive_tests()