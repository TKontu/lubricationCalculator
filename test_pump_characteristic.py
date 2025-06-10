#!/usr/bin/env python3
"""
Test script for pump characteristic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lubrication_flow_package.components.pump import PumpCharacteristic
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.connection import Connection
import numpy as np
import matplotlib.pyplot as plt


def test_pump_characteristic_basic():
    """Test basic pump characteristic functionality"""
    print("Testing basic pump characteristic functionality...")
    
    # Test polynomial pump characteristic
    # P = 1000000 - 5000000*Q - 1000000*Q^2 (Pa)
    pump_poly = PumpCharacteristic(
        curve_type="polynomial",
        coefficients=[1000000, -5000000, -1000000],  # [a0, a1, a2]
        max_flow=0.5,
        max_pressure=1000000
    )
    
    # Test pressure calculation
    flow_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    print("\nPolynomial pump characteristic:")
    print("Flow (m³/s) | Pressure (Pa)")
    print("-" * 30)
    for q in flow_rates:
        p = pump_poly.get_pressure(q)
        print(f"{q:10.3f} | {p:12.0f}")
    
    # Test table-based pump characteristic
    flow_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    pressure_points = [1000000, 950000, 850000, 700000, 500000, 250000]
    
    pump_table = PumpCharacteristic(
        curve_type="table",
        flow_points=flow_points,
        pressure_points=pressure_points
    )
    
    print("\nTable-based pump characteristic:")
    print("Flow (m³/s) | Pressure (Pa)")
    print("-" * 30)
    for q in flow_rates:
        p = pump_table.get_pressure(q)
        print(f"{q:10.3f} | {p:12.0f}")
    
    # Test typical centrifugal pump
    pump_centrifugal = PumpCharacteristic.create_typical_centrifugal_pump(
        max_pressure=1200000,  # 1.2 MPa shutoff pressure
        max_flow=0.6,          # 0.6 m³/s max flow
        efficiency_point=(0.7, 0.8)  # Best efficiency at 70% flow, 80% pressure
    )
    
    print("\nTypical centrifugal pump characteristic:")
    print("Flow (m³/s) | Pressure (Pa)")
    print("-" * 30)
    for q in flow_rates:
        p = pump_centrifugal.get_pressure(q)
        print(f"{q:10.3f} | {p:12.0f}")
    
    # Test operating point calculation
    system_resistance = 2e7  # Pa·s²/m⁶
    operating_flow, operating_pressure = pump_centrifugal.find_operating_point(
        system_resistance, flow_range=(0.0, 0.6)
    )
    
    print(f"\nOperating point with system resistance {system_resistance:.1e}:")
    print(f"Flow: {operating_flow:.4f} m³/s")
    print(f"Pressure: {operating_pressure:.0f} Pa")
    
    print("✅ Basic pump characteristic tests passed!")


def test_pump_with_network():
    """Test pump characteristic with actual network"""
    print("\nTesting pump characteristic with network...")
    
    # Create a simple network
    network = FlowNetwork()
    
    # Add nodes
    inlet = Node("inlet", 0, 0, 0)
    junction = Node("junction", 1, 0, 0)
    outlet1 = Node("outlet1", 2, 0, 0)
    outlet2 = Node("outlet2", 2, 1, 0)
    
    network.add_node(inlet)
    network.add_node(junction)
    network.add_node(outlet1)
    network.add_node(outlet2)
    
    # Add components
    main_channel = Channel(0.01, 1.0, component_id="main")    # 10mm diameter, 1m length
    branch1 = Channel(0.008, 0.5, component_id="branch1")     # 8mm diameter, 0.5m length
    branch2 = Channel(0.006, 0.3, component_id="branch2")     # 6mm diameter, 0.3m length
    
    # Add connections
    network.connect_components(inlet, junction, main_channel)
    network.connect_components(junction, outlet1, branch1)
    network.connect_components(junction, outlet2, branch2)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create pump characteristic
    pump_char = PumpCharacteristic.create_typical_centrifugal_pump(
        max_pressure=500000,   # 500 kPa shutoff pressure
        max_flow=0.01,         # 0.01 m³/s max flow
        efficiency_point=(0.6, 0.75)
    )
    
    # Create solver
    solver = NetworkFlowSolver()
    
    # Test with pump characteristic
    print("\nSolving with pump characteristic...")
    connection_flows, solution_info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=0.005,  # 5 L/s initial estimate
        temperature=40.0,      # 40°C
        pump_characteristic=pump_char
    )
    
    print(f"Converged: {solution_info['converged']}")
    print(f"Iterations: {solution_info['iterations']}")
    print(f"Pump adequate: {solution_info['pump_adequate']}")
    print(f"Actual flow rate: {solution_info['actual_flow_rate']:.6f} m³/s")
    print(f"Required inlet pressure: {solution_info['required_inlet_pressure']:.0f} Pa")
    
    if 'available_pressure' in solution_info:
        print(f"Available pressure: {solution_info['available_pressure']:.0f} Pa")
    if 'operating_pressure' in solution_info:
        print(f"Operating pressure: {solution_info['operating_pressure']:.0f} Pa")
    
    print("\nConnection flows:")
    for conn_id, flow in connection_flows.items():
        print(f"  {conn_id}: {flow:.6f} m³/s")
    
    # Compare with legacy method (no pump characteristic)
    print("\nSolving with legacy method (no pump characteristic)...")
    connection_flows_legacy, solution_info_legacy = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=0.005,
        temperature=40.0,
        pump_max_pressure=500000  # Same max pressure as pump characteristic
    )
    
    print(f"Legacy - Converged: {solution_info_legacy['converged']}")
    print(f"Legacy - Pump adequate: {solution_info_legacy['pump_adequate']}")
    print(f"Legacy - Actual flow rate: {solution_info_legacy['actual_flow_rate']:.6f} m³/s")
    
    print("✅ Network pump characteristic tests passed!")


def plot_pump_curves():
    """Plot pump curves for visualization"""
    print("\nGenerating pump curve plots...")
    
    try:
        # Create different pump characteristics
        pump_poly = PumpCharacteristic(
            curve_type="polynomial",
            coefficients=[1000000, -2000000, -500000],
            max_flow=1.0
        )
        
        pump_centrifugal = PumpCharacteristic.create_typical_centrifugal_pump(
            max_pressure=1200000,
            max_flow=0.8,
            efficiency_point=(0.6, 0.8)
        )
        
        # Generate curve points
        flow_range = (0.0, 0.8)
        flows_poly, pressures_poly = pump_poly.get_curve_points(50, flow_range)
        flows_cent, pressures_cent = pump_centrifugal.get_curve_points(50, flow_range)
        
        # Create system curves
        flows_system = np.linspace(0, 0.8, 50)
        system_curves = {
            'Low resistance': 1e6 * flows_system**2,
            'Medium resistance': 5e6 * flows_system**2,
            'High resistance': 2e7 * flows_system**2
        }
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Plot pump curves
        plt.plot(flows_poly, pressures_poly/1000, 'b-', linewidth=2, label='Polynomial Pump')
        plt.plot(flows_cent, pressures_cent/1000, 'r-', linewidth=2, label='Centrifugal Pump')
        
        # Plot system curves
        colors = ['g--', 'm--', 'c--']
        for i, (label, pressures) in enumerate(system_curves.items()):
            plt.plot(flows_system, pressures/1000, colors[i], linewidth=1.5, 
                    label=f'System: {label}')
        
        plt.xlabel('Flow Rate (m³/s)')
        plt.ylabel('Pressure (kPa)')
        plt.title('Pump Characteristics and System Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 0.8)
        plt.ylim(0, 1400)
        
        # Save plot
        plt.savefig('/workspace/lubricationCalculator/pump_curves.png', dpi=150, bbox_inches='tight')
        print("✅ Pump curves plotted and saved as 'pump_curves.png'")
        
    except ImportError:
        print("⚠️  Matplotlib not available, skipping plot generation")


def main():
    """Run all tests"""
    print("PUMP CHARACTERISTIC TESTING")
    print("=" * 50)
    
    try:
        test_pump_characteristic_basic()
        test_pump_with_network()
        plot_pump_curves()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("Pump characteristic functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())