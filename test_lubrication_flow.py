#!/usr/bin/env python3
"""
AComprehensive Test Suite for Lubrication Flow Distribution Calculator

This test suite validates the calculation engine with:
- Unit tests for individual components
- Integration tests for complete systems
- Analytical validation cases
- Edge cases and stress tests
- Performance tests for large systems
- Numerical accuracy verification
"""

import unittest
import numpy as np
import math
import time
from typing import List, Tuple

from advanced_lubrication_flow_tool import (
    AdvancedLubricationFlowCalculator,
    PipeSegment,
    Nozzle,
    Branch,
    NozzleType,
    FlowRegime
)


class TestPipeSegment(unittest.TestCase):
    """Test PipeSegment class"""
    
    def test_valid_pipe_creation(self):
        """Test creation of valid pipe segments"""
        pipe = PipeSegment(diameter=0.05, length=10.0, roughness=0.00015)
        self.assertEqual(pipe.diameter, 0.05)
        self.assertEqual(pipe.length, 10.0)
        self.assertEqual(pipe.roughness, 0.00015)
    
    def test_invalid_pipe_parameters(self):
        """Test validation of pipe parameters"""
        with self.assertRaises(ValueError):
            PipeSegment(diameter=-0.05, length=10.0)  # Negative diameter
        
        with self.assertRaises(ValueError):
            PipeSegment(diameter=0.05, length=-10.0)  # Negative length
        
        with self.assertRaises(ValueError):
            PipeSegment(diameter=0.05, length=10.0, roughness=-0.001)  # Negative roughness


class TestNozzle(unittest.TestCase):
    """Test Nozzle class"""
    
    def test_nozzle_creation_with_defaults(self):
        """Test nozzle creation with default parameters"""
        nozzle = Nozzle(diameter=0.01)
        self.assertEqual(nozzle.diameter, 0.01)
        self.assertEqual(nozzle.nozzle_type, NozzleType.SHARP_EDGED)
        self.assertEqual(nozzle.discharge_coeff, 0.6)
    
    def test_different_nozzle_types(self):
        """Test different nozzle types and their default coefficients"""
        test_cases = [
            (NozzleType.SHARP_EDGED, 0.6),
            (NozzleType.ROUNDED, 0.8),
            (NozzleType.VENTURI, 0.95),
            (NozzleType.FLOW_NOZZLE, 0.98)
        ]
        
        for nozzle_type, expected_cd in test_cases:
            nozzle = Nozzle(diameter=0.01, nozzle_type=nozzle_type)
            self.assertEqual(nozzle.discharge_coeff, expected_cd)
    
    def test_custom_discharge_coefficient(self):
        """Test custom discharge coefficient override"""
        nozzle = Nozzle(diameter=0.01, discharge_coeff=0.75)
        self.assertEqual(nozzle.discharge_coeff, 0.75)
    
    def test_invalid_nozzle_diameter(self):
        """Test validation of nozzle diameter"""
        with self.assertRaises(ValueError):
            Nozzle(diameter=-0.01)


class TestAdvancedLubricationFlowCalculator(unittest.TestCase):
    """Test the main calculator class"""
    
    def setUp(self):
        """Set up test calculator"""
        self.calculator = AdvancedLubricationFlowCalculator(
            oil_density=900.0, 
            oil_type="SAE30"
        )
    
    def test_calculator_initialization(self):
        """Test calculator initialization"""
        self.assertEqual(self.calculator.oil_density, 900.0)
        self.assertEqual(self.calculator.oil_type, "SAE30")
        self.assertEqual(self.calculator.gravity, 9.81)
    
    def test_invalid_calculator_parameters(self):
        """Test validation of calculator parameters"""
        with self.assertRaises(ValueError):
            AdvancedLubricationFlowCalculator(oil_density=-900.0)  # Negative density
        
        with self.assertRaises(ValueError):
            AdvancedLubricationFlowCalculator(compressibility=-1e-9)  # Negative compressibility
    
    def test_viscosity_calculation(self):
        """Test viscosity calculation for different temperatures and oil types"""
        # Test SAE30 at different temperatures
        temp_viscosity_pairs = [
            (20, 0.2),   # Approximate values
            (40, 0.1),
            (60, 0.05),
            (80, 0.03)
        ]
        
        for temp, expected_order in temp_viscosity_pairs:
            viscosity = self.calculator.calculate_viscosity(temp)
            self.assertGreater(viscosity, 0)
            self.assertLess(viscosity, 10.0)  # Reasonable upper bound
            # Viscosity should decrease with temperature
            if temp > 20:
                prev_viscosity = self.calculator.calculate_viscosity(temp - 20)
                self.assertLess(viscosity, prev_viscosity)
    
    def test_different_oil_types(self):
        """Test viscosity calculation for different oil types"""
        oil_types = ["SAE10", "SAE20", "SAE30", "SAE40", "SAE50", "SAE60"]
        temperature = 40
        
        viscosities = []
        for oil_type in oil_types:
            calc = AdvancedLubricationFlowCalculator(oil_type=oil_type)
            viscosity = calc.calculate_viscosity(temperature)
            viscosities.append(viscosity)
            self.assertGreater(viscosity, 0)
        
        # Higher SAE numbers should generally have higher viscosity
        for i in range(len(viscosities) - 1):
            self.assertLessEqual(viscosities[i], viscosities[i + 1])
    
    def test_reynolds_number_calculation(self):
        """Test Reynolds number calculation"""
        velocity = 2.0  # m/s
        diameter = 0.05  # m
        viscosity = 0.1  # Pa·s
        
        reynolds = self.calculator.calculate_reynolds_number(velocity, diameter, viscosity)
        expected_reynolds = (900.0 * velocity * diameter) / viscosity
        self.assertAlmostEqual(reynolds, expected_reynolds, places=6)
        
        # Test edge cases
        self.assertEqual(self.calculator.calculate_reynolds_number(0, diameter, viscosity), 0)
        self.assertEqual(self.calculator.calculate_reynolds_number(velocity, 0, viscosity), 0)
        self.assertEqual(self.calculator.calculate_reynolds_number(velocity, diameter, 0), 0)
    
    def test_friction_factor_calculation(self):
        """Test friction factor calculation for different flow regimes"""
        relative_roughness = 0.003
        
        # Laminar flow
        reynolds_lam = 1000
        f_lam = self.calculator.calculate_friction_factor(reynolds_lam, relative_roughness)
        expected_f_lam = 64 / reynolds_lam
        self.assertAlmostEqual(f_lam, expected_f_lam, places=6)
        
        # Turbulent flow
        reynolds_turb = 10000
        f_turb = self.calculator.calculate_friction_factor(reynolds_turb, relative_roughness)
        self.assertGreater(f_turb, 0)
        self.assertLess(f_turb, 0.1)  # Reasonable upper bound
        
        # Transition flow
        reynolds_trans = 3000
        f_trans = self.calculator.calculate_friction_factor(reynolds_trans, relative_roughness)
        self.assertGreater(f_trans, f_turb)
        self.assertLess(f_trans, f_lam)
    
    def test_pipe_pressure_drop_calculation(self):
        """Test pipe pressure drop calculation"""
        flow_rate = 0.001  # m³/s
        pipe = PipeSegment(diameter=0.05, length=10.0, roughness=0.00015)
        viscosity = 0.1  # Pa·s
        
        pressure_drop = self.calculator.calculate_pipe_pressure_drop(flow_rate, pipe, viscosity)
        
        # Verify positive pressure drop
        self.assertGreater(pressure_drop, 0)
        
        # Test with zero flow
        zero_pressure_drop = self.calculator.calculate_pipe_pressure_drop(0, pipe, viscosity)
        self.assertEqual(zero_pressure_drop, 0)
        
        # Test elevation effect
        elevation_change = 5.0  # m
        pressure_drop_with_elevation = self.calculator.calculate_pipe_pressure_drop(
            flow_rate, pipe, viscosity, elevation_change
        )
        expected_elevation_effect = self.calculator.oil_density * self.calculator.gravity * elevation_change
        self.assertAlmostEqual(
            pressure_drop_with_elevation - pressure_drop, 
            expected_elevation_effect, 
            places=1
        )
    
    def test_nozzle_pressure_drop_calculation(self):
        """Test nozzle pressure drop calculation"""
        flow_rate = 0.001  # m³/s
        
        # Test different nozzle types
        nozzle_types = [
            NozzleType.SHARP_EDGED,
            NozzleType.ROUNDED,
            NozzleType.VENTURI,
            NozzleType.FLOW_NOZZLE
        ]
        
        pressure_drops = []
        for nozzle_type in nozzle_types:
            nozzle = Nozzle(diameter=0.01, nozzle_type=nozzle_type)
            pressure_drop = self.calculator.calculate_nozzle_pressure_drop(flow_rate, nozzle)
            pressure_drops.append(pressure_drop)
            self.assertGreater(pressure_drop, 0)
        
        # Venturi and flow nozzles should have lower pressure drops
        sharp_edged_dp = pressure_drops[0]
        venturi_dp = pressure_drops[2]
        flow_nozzle_dp = pressure_drops[3]
        
        self.assertLess(venturi_dp, sharp_edged_dp)
        self.assertLess(flow_nozzle_dp, sharp_edged_dp)


class TestFlowDistributionSolver(unittest.TestCase):
    """Test flow distribution solver"""
    
    def setUp(self):
        """Set up test calculator and simple system"""
        self.calculator = AdvancedLubricationFlowCalculator(
            oil_density=900.0, 
            oil_type="SAE30"
        )
        
        # Simple two-branch system
        self.branches = [
            Branch(
                pipe=PipeSegment(diameter=0.05, length=5.0),
                nozzle=Nozzle(diameter=0.008),
                name="Branch 1"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.04, length=6.0),
                nozzle=Nozzle(diameter=0.006),
                name="Branch 2"
            )
        ]
        
        self.total_flow_rate = 0.01  # m³/s
        self.temperature = 40  # °C
    
    def test_single_branch_system(self):
        """Test system with single branch"""
        single_branch = [self.branches[0]]
        
        flows, info = self.calculator.solve_flow_distribution(
            self.total_flow_rate, single_branch, self.temperature
        )
        
        self.assertEqual(len(flows), 1)
        self.assertAlmostEqual(flows[0], self.total_flow_rate, places=6)
        self.assertTrue(info['converged'])
        self.assertEqual(info['iterations'], 1)
    
    def test_mass_conservation(self):
        """Test that mass is conserved in solution"""
        flows, info = self.calculator.solve_flow_distribution(
            self.total_flow_rate, self.branches, self.temperature
        )
        
        total_calculated_flow = sum(flows)
        self.assertAlmostEqual(total_calculated_flow, self.total_flow_rate, places=8)
        self.assertTrue(info['converged'])
    
    def test_pressure_equalization(self):
        """Test that pressure drops are approximately equal"""
        flows, info = self.calculator.solve_flow_distribution(
            self.total_flow_rate, self.branches, self.temperature
        )
        
        pressure_drops = info['pressure_drops']
        
        # Check that pressure drops are reasonably close
        max_dp = max(pressure_drops)
        min_dp = min(pressure_drops)
        relative_difference = abs(max_dp - min_dp) / max_dp if max_dp > 0 else 0
        
        self.assertLess(relative_difference, 0.01)  # Within 1%
    
    def test_newton_vs_iterative_methods(self):
        """Compare Newton-Raphson and iterative methods"""
        flows_newton, info_newton = self.calculator.solve_flow_distribution(
            self.total_flow_rate, self.branches, self.temperature, method="newton"
        )
        
        flows_iter, info_iter = self.calculator.solve_flow_distribution(
            self.total_flow_rate, self.branches, self.temperature, method="iterative"
        )
        
        # Both should converge
        self.assertTrue(info_newton['converged'])
        self.assertTrue(info_iter['converged'])
        
        # Results should be similar
        for f_newton, f_iter in zip(flows_newton, flows_iter):
            relative_diff = abs(f_newton - f_iter) / f_newton if f_newton > 0 else 0
            self.assertLess(relative_diff, 0.05)  # Within 5%
        
        # Newton method should converge faster
        self.assertLessEqual(info_newton['iterations'], info_iter['iterations'])
    
    def test_flow_regime_classification(self):
        """Test flow regime classification"""
        flows, info = self.calculator.solve_flow_distribution(
            self.total_flow_rate, self.branches, self.temperature
        )
        
        reynolds_numbers = info['reynolds_numbers']
        
        for reynolds in reynolds_numbers:
            regime = self.calculator.get_flow_regime(reynolds)
            
            if reynolds < 2300:
                self.assertEqual(regime, FlowRegime.LAMINAR)
            elif reynolds < 4000:
                self.assertEqual(regime, FlowRegime.TRANSITION)
            else:
                self.assertEqual(regime, FlowRegime.TURBULENT)


class TestAnalyticalValidation(unittest.TestCase):
    """Test against known analytical solutions"""
    
    def setUp(self):
        """Set up calculator"""
        self.calculator = AdvancedLubricationFlowCalculator(
            oil_density=1000.0,  # Use water properties for easier validation
            oil_type="SAE30"
        )
    
    def test_equal_branches_equal_flow(self):
        """Test that identical branches get equal flow"""
        # Create two identical branches
        branches = [
            Branch(
                pipe=PipeSegment(diameter=0.05, length=10.0),
                nozzle=Nozzle(diameter=0.01),
                name="Branch 1"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.05, length=10.0),
                nozzle=Nozzle(diameter=0.01),
                name="Branch 2"
            )
        ]
        
        total_flow = 0.01
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        # Flows should be equal within numerical tolerance
        self.assertAlmostEqual(flows[0], flows[1], places=6)
        self.assertAlmostEqual(flows[0], total_flow / 2, places=6)
    
    def test_no_nozzle_vs_with_nozzle(self):
        """Test that branch without nozzle gets more flow"""
        branches = [
            Branch(
                pipe=PipeSegment(diameter=0.05, length=10.0),
                nozzle=Nozzle(diameter=0.01),
                name="With Nozzle"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.05, length=10.0),
                nozzle=None,
                name="No Nozzle"
            )
        ]
        
        total_flow = 0.01
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        # Branch without nozzle should get more flow
        self.assertGreater(flows[1], flows[0])
    
    def test_diameter_effect(self):
        """Test effect of pipe diameter on flow distribution"""
        branches = [
            Branch(
                pipe=PipeSegment(diameter=0.03, length=10.0),  # Smaller diameter
                name="Small Pipe"
            ),
            Branch(
                pipe=PipeSegment(diameter=0.06, length=10.0),  # Larger diameter
                name="Large Pipe"
            )
        ]
        
        total_flow = 0.01
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        # Larger diameter pipe should get more flow
        self.assertGreater(flows[1], flows[0])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and extreme conditions"""
    
    def setUp(self):
        """Set up calculator"""
        self.calculator = AdvancedLubricationFlowCalculator()
    
    def test_very_small_flow_rates(self):
        """Test with very small flow rates"""
        branches = [
            Branch(pipe=PipeSegment(diameter=0.001, length=1.0), name="Micro 1"),
            Branch(pipe=PipeSegment(diameter=0.001, length=1.0), name="Micro 2")
        ]
        
        total_flow = 1e-8  # Very small flow
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        self.assertTrue(info['converged'])
        self.assertAlmostEqual(sum(flows), total_flow, places=10)
    
    def test_very_large_flow_rates(self):
        """Test with very large flow rates"""
        branches = [
            Branch(pipe=PipeSegment(diameter=0.5, length=10.0), name="Large 1"),
            Branch(pipe=PipeSegment(diameter=0.5, length=10.0), name="Large 2")
        ]
        
        total_flow = 1.0  # Large flow
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        self.assertTrue(info['converged'])
        self.assertAlmostEqual(sum(flows), total_flow, places=6)
    
    def test_extreme_temperatures(self):
        """Test with extreme temperatures"""
        branches = [
            Branch(pipe=PipeSegment(diameter=0.05, length=10.0), name="Branch 1"),
            Branch(pipe=PipeSegment(diameter=0.05, length=10.0), name="Branch 2")
        ]
        
        total_flow = 0.01
        
        # Test low temperature
        flows_low, info_low = self.calculator.solve_flow_distribution(
            total_flow, branches, 5  # 5°C
        )
        
        # Test high temperature
        flows_high, info_high = self.calculator.solve_flow_distribution(
            total_flow, branches, 100  # 100°C
        )
        
        self.assertTrue(info_low['converged'])
        self.assertTrue(info_high['converged'])
        
        # Viscosity should be higher at low temperature
        self.assertGreater(info_low['viscosity'], info_high['viscosity'])
    
    def test_many_branches(self):
        """Test system with many branches"""
        num_branches = 20
        branches = []
        
        for i in range(num_branches):
            # Vary diameters and lengths slightly
            diameter = 0.03 + 0.01 * (i % 5)
            length = 5.0 + 2.0 * (i % 3)
            
            branches.append(Branch(
                pipe=PipeSegment(diameter=diameter, length=length),
                name=f"Branch {i+1}"
            ))
        
        total_flow = 0.05
        temperature = 40
        
        flows, info = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature
        )
        
        self.assertTrue(info['converged'])
        self.assertEqual(len(flows), num_branches)
        self.assertAlmostEqual(sum(flows), total_flow, places=6)


class TestPerformance(unittest.TestCase):
    """Test performance for large systems"""
    
    def setUp(self):
        """Set up calculator"""
        self.calculator = AdvancedLubricationFlowCalculator()
    
    def test_convergence_speed(self):
        """Test convergence speed for different methods"""
        # Create moderately complex system
        branches = []
        for i in range(10):
            branches.append(Branch(
                pipe=PipeSegment(diameter=0.02 + 0.01*i, length=5.0 + i),
                nozzle=Nozzle(diameter=0.005 + 0.001*i) if i % 2 == 0 else None,
                name=f"Branch {i+1}"
            ))
        
        total_flow = 0.02
        temperature = 40
        
        # Test Newton method
        start_time = time.time()
        flows_newton, info_newton = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature, method="newton"
        )
        newton_time = time.time() - start_time
        
        # Test iterative method
        start_time = time.time()
        flows_iter, info_iter = self.calculator.solve_flow_distribution(
            total_flow, branches, temperature, method="iterative"
        )
        iterative_time = time.time() - start_time
        
        print(f"\nPerformance comparison:")
        print(f"Newton method: {info_newton['iterations']} iterations, {newton_time:.4f}s")
        print(f"Iterative method: {info_iter['iterations']} iterations, {iterative_time:.4f}s")
        
        # Both should converge
        self.assertTrue(info_newton['converged'])
        self.assertTrue(info_iter['converged'])
        
        # Newton should generally converge in fewer iterations
        self.assertLessEqual(info_newton['iterations'], info_iter['iterations'] * 2)
    
    def test_large_system_scalability(self):
        """Test scalability with large number of branches"""
        branch_counts = [5, 10, 20, 50]
        
        for num_branches in branch_counts:
            branches = []
            for i in range(num_branches):
                branches.append(Branch(
                    pipe=PipeSegment(diameter=0.02, length=10.0),
                    name=f"Branch {i+1}"
                ))
            
            total_flow = 0.01 * num_branches
            temperature = 40
            
            start_time = time.time()
            flows, info = self.calculator.solve_flow_distribution(
                total_flow, branches, temperature, method="newton"
            )
            solve_time = time.time() - start_time
            
            print(f"Branches: {num_branches}, Time: {solve_time:.4f}s, "
                  f"Iterations: {info['iterations']}, Converged: {info['converged']}")
            
            self.assertTrue(info['converged'])
            self.assertAlmostEqual(sum(flows), total_flow, places=6)


def run_comprehensive_tests():
    """Run all tests and provide summary"""
    print("="*70)
    print("COMPREHENSIVE LUBRICATION FLOW CALCULATOR TEST SUITE")
    print("="*70)
    
    # Create test suite
    test_classes = [
        TestPipeSegment,
        TestNozzle,
        TestAdvancedLubricationFlowCalculator,
        TestFlowDistributionSolver,
        TestAnalyticalValidation,
        TestEdgeCases,
        TestPerformance
    ]
    
    total_tests = 0
    total_failures = 0
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=1)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
    
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"Total tests run: {total_tests}")
    print(f"Failures/Errors: {total_failures}")
    print(f"Success rate: {(total_tests - total_failures) / total_tests * 100:.1f}%")
    
    if total_failures == 0:
        print("🎉 ALL TESTS PASSED! 🎉")
    else:
        print(f"⚠️  {total_failures} tests failed. Please review the output above.")


if __name__ == "__main__":
    run_comprehensive_tests()