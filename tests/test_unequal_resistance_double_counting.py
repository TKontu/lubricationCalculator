#!/usr/bin/env python3
"""
Test case with unequal resistances to demonstrate where path-based 
flow initialization fails due to double-counting.
"""

import pytest
import numpy as np
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder

class TestUnequalResistanceDoubleCounting:
    
    def test_unequal_resistance_double_counting(self):
        """
        Test with unequal branch resistances to show double-counting problems.
        
        Network:  Source → [Shared] → Junction → [Easy Branch] → Sink1 (low resistance)
                                             └→ [Hard Branch] → Sink2 (high resistance)
        
        Expected: Most flow goes through Easy Branch, little through Hard Branch
        Path-based algorithm: Will still double-count shared component incorrectly
        """
        
        builder = NetworkBuilder()
        net = (builder
            .set_inlet("source")
            .add_node("junction")
            .add_outlet("sink1") # Easy path
            .add_outlet("sink2") # Hard path
            .add_pipe("source", "junction", length=1.0, diameter=0.020, name="shared")
            .add_pipe("junction", "sink1", length=1.0, diameter=0.020, name="easy") # Low resistance
            .add_pipe("junction", "sink2", length=10.0, diameter=0.005, name="hard") # High resistance
            .build()
        )
        net.name = "unequal_resistance_test"
        
        source = net.get_node("source")
        sink1 = net.get_node("sink1")
        sink2 = net.get_node("sink2")

        Q_total = 0.001  # 1 L/s
        
        # Create components with very different resistances
        shared = net.get_component_by_name("shared")
        easy_branch = net.get_component_by_name("easy")
        hard_branch = net.get_component_by_name("hard")
        
        # Calculate actual resistances for analysis
        fluid_props = {'density': 900.0, 'viscosity': 1e-3}
        
        sim_config = SimulationConfig(total_flow_rate=Q_total, oil_density=900.0, oil_type="SAE30", temperature=40, inlet_pressure=101325)
        solver = NodalMatrixSolver(sim_config)
        solver.calculate_viscosity = lambda T: 1e-3
        
        R_shared = solver._calculate_component_resistance(shared, fluid_props, 1e-6)
        R_easy = solver._calculate_component_resistance(easy_branch, fluid_props, 1e-6)
        R_hard = solver._calculate_component_resistance(hard_branch, fluid_props, 1e-6)
        
        print("=== Component Resistances ===")
        print(f"Shared:     {R_shared:.2e} Pa⋅s/m³")
        print(f"Easy branch: {R_easy:.2e} Pa⋅s/m³")
        print(f"Hard branch: {R_hard:.2e} Pa⋅s/m³")
        print(f"Resistance ratio (hard/easy): {R_hard/R_easy:.1f}")
        
        # Calculate analytical solution
        print("\n=== Analytical Solution ===")
        R_path1 = R_shared + R_easy   # Total resistance path 1
        R_path2 = R_shared + R_hard   # Total resistance path 2
        
        # For parallel paths from junction: flow splits by inverse resistance
        R_parallel = 1.0 / (1.0/R_easy + 1.0/R_hard)  # Equivalent resistance of parallel branches
        R_total = R_shared + R_parallel  # Total network resistance
        
        # Pressure drop across parallel section
        P_junction = Q_total * R_parallel
        
        # Flow splits
        Q_easy_analytical = P_junction / R_easy
        Q_hard_analytical = P_junction / R_hard
        Q_shared_analytical = Q_total  # All flow goes through shared
        
        print(f"Shared pipe flow: {Q_shared_analytical:.6f} m³/s")
        print(f"Easy branch flow: {Q_easy_analytical:.6f} m³/s")
        print(f"Hard branch flow: {Q_hard_analytical:.6f} m³/s")
        print(f"Flow ratio (easy/hard): {Q_easy_analytical/Q_hard_analytical:.1f}")
        
        # Test the actual solver
        print("\n=== Actual Solver Test ===")
        actual_flows = solver._initialize_flows(net, source.id, [sink1.id, sink2.id], Q_total)
        
        print("Actual solver results:")
        for comp, flow in actual_flows.items():
            print(f"{comp}: {flow:.6f} m³/s")
        
        # Check mass conservation
        print("\n=== Mass Conservation Check ===")
        shared_flow = actual_flows[shared.id]
        branch_sum = actual_flows[easy_branch.id] + actual_flows[hard_branch.id]
        mass_error = abs(shared_flow - branch_sum)
        
        print(f"Flow into junction: {shared_flow:.6f}")
        print(f"Flow out of junction: {branch_sum:.6f}")
        print(f"Mass conservation error: {mass_error:.6e}")
        
        # Assert mass conservation
        assert mass_error < 1e-6, f"Mass conservation violated: error = {mass_error:.6e}"
        
        # Assert shared flow equals total flow
        assert abs(shared_flow - Q_total) < 1e-6, f"Shared flow should equal total: {shared_flow:.6f} vs {Q_total:.6f}"
        
        print("✅ All assertions passed")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])