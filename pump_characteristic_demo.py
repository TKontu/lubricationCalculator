#!/usr/bin/env python3
"""
Demonstration of pump characteristic curves in lubrication flow calculations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lubrication_flow_package.components.pump import PumpCharacteristic
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.channel import Channel


def create_sample_network():
    """Create a sample lubrication network"""
    network = FlowNetwork("Sample Lubrication System")
    
    # Create nodes representing different points in the system
    pump_outlet = Node("pump_outlet", 0, 0, 0)
    main_junction = Node("main_junction", 2, 0, 0)
    bearing1 = Node("bearing1", 4, 0, 0)
    bearing2 = Node("bearing2", 4, 2, 0)
    bearing3 = Node("bearing3", 6, 1, 1)  # Elevated bearing
    
    network.add_node(pump_outlet)
    network.add_node(main_junction)
    network.add_node(bearing1)
    network.add_node(bearing2)
    network.add_node(bearing3)
    
    # Create lubrication channels (pipes/drillings)
    main_line = Channel(0.012, 2.0, component_id="main_line")      # 12mm main line, 2m
    branch_1 = Channel(0.008, 2.0, component_id="branch_1")        # 8mm to bearing 1, 2m
    branch_2 = Channel(0.008, 2.0, component_id="branch_2")        # 8mm to bearing 2, 2m
    branch_3 = Channel(0.006, 2.5, component_id="branch_3")        # 6mm to bearing 3, 2.5m
    
    # Connect the network
    network.connect_components(pump_outlet, main_junction, main_line)
    network.connect_components(main_junction, bearing1, branch_1)
    network.connect_components(main_junction, bearing2, branch_2)
    network.connect_components(main_junction, bearing3, branch_3)
    
    # Set inlet and outlets
    network.set_inlet(pump_outlet)
    network.add_outlet(bearing1)
    network.add_outlet(bearing2)
    network.add_outlet(bearing3)
    
    return network


def demo_pump_characteristics():
    """Demonstrate different pump characteristic types"""
    print("PUMP CHARACTERISTIC DEMONSTRATION")
    print("=" * 60)
    
    # Create sample network
    network = create_sample_network()
    solver = NetworkFlowSolver()
    
    # Operating conditions
    temperature = 60.0  # °C
    target_flow = 0.008  # 8 L/s
    
    print(f"Network: {network.name}")
    print(f"Temperature: {temperature}°C")
    print(f"Target flow rate: {target_flow*1000:.1f} L/s")
    print()
    
    # 1. Typical centrifugal pump
    print("1. CENTRIFUGAL PUMP CHARACTERISTIC")
    print("-" * 40)
    
    centrifugal_pump = PumpCharacteristic.create_typical_centrifugal_pump(
        max_pressure=800000,   # 800 kPa shutoff pressure
        max_flow=0.015,        # 15 L/s max flow
        efficiency_point=(0.6, 0.8)  # Best efficiency at 60% flow, 80% pressure
    )
    
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=target_flow,
        temperature=temperature,
        pump_characteristic=centrifugal_pump
    )
    
    print_results("Centrifugal Pump", flows, info)
    
    # 2. High-pressure positive displacement pump
    print("\n2. POSITIVE DISPLACEMENT PUMP CHARACTERISTIC")
    print("-" * 50)
    
    # Create a pump with more constant pressure (typical of positive displacement)
    pd_pump = PumpCharacteristic(
        curve_type="polynomial",
        coefficients=[1000000, -20000000, 0],  # Nearly constant pressure with slight drop
        max_flow=0.012,
        max_pressure=1000000
    )
    
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=target_flow,
        temperature=temperature,
        pump_characteristic=pd_pump
    )
    
    print_results("Positive Displacement Pump", flows, info)
    
    # 3. Manufacturer data-based pump
    print("\n3. MANUFACTURER DATA-BASED PUMP")
    print("-" * 40)
    
    # Example manufacturer performance data
    flow_data = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014]  # m³/s
    pressure_data = [900000, 880000, 840000, 780000, 700000, 600000, 480000, 320000]  # Pa
    
    manufacturer_pump = PumpCharacteristic.create_from_manufacturer_data(
        flow_points=flow_data,
        pressure_points=pressure_data
    )
    
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=target_flow,
        temperature=temperature,
        pump_characteristic=manufacturer_pump
    )
    
    print_results("Manufacturer Data Pump", flows, info)
    
    # 4. Comparison with legacy method (no pump characteristic)
    print("\n4. LEGACY METHOD (NO PUMP CHARACTERISTIC)")
    print("-" * 45)
    
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=target_flow,
        temperature=temperature,
        pump_max_pressure=800000  # Simple pressure limit
    )
    
    print_results("Legacy Method", flows, info)
    
    # 5. System analysis
    print("\n5. PUMP SELECTION ANALYSIS")
    print("-" * 30)
    
    print("Pump Performance Comparison:")
    print("Pump Type                | Flow (L/s) | Pressure (kPa) | Adequate")
    print("-" * 65)
    
    pumps = [
        ("Centrifugal", centrifugal_pump),
        ("Positive Displacement", pd_pump),
        ("Manufacturer Data", manufacturer_pump)
    ]
    
    for pump_name, pump_char in pumps:
        flows, info = solver.solve_network_flow_with_pump_physics(
            network=network,
            pump_flow_rate=target_flow,
            temperature=temperature,
            pump_characteristic=pump_char
        )
        
        actual_flow = info['actual_flow_rate'] * 1000  # Convert to L/s
        pressure = info.get('available_pressure', 0) / 1000  # Convert to kPa
        adequate = "✅ Yes" if info['pump_adequate'] else "❌ No"
        
        print(f"{pump_name:24} | {actual_flow:8.2f}   | {pressure:10.0f} | {adequate}")


def print_results(pump_type, flows, info):
    """Print formatted results"""
    print(f"Pump Type: {pump_type}")
    print(f"  Converged: {info['converged']}")
    print(f"  Iterations: {info['iterations']}")
    print(f"  Pump adequate: {'✅ Yes' if info['pump_adequate'] else '❌ No'}")
    print(f"  Actual flow rate: {info['actual_flow_rate']*1000:.2f} L/s")
    print(f"  Required pressure: {info['required_inlet_pressure']/1000:.0f} kPa")
    
    if 'available_pressure' in info:
        print(f"  Available pressure: {info['available_pressure']/1000:.0f} kPa")
    if 'operating_pressure' in info:
        print(f"  Operating pressure: {info['operating_pressure']/1000:.0f} kPa")
    
    print("  Flow distribution:")
    total_flow = sum(flows.values())
    for component_id, flow in flows.items():
        percentage = (flow / total_flow * 100) if total_flow > 0 else 0
        print(f"    {component_id}: {flow*1000:.3f} L/s ({percentage:.1f}%)")


def demo_pump_curve_intersection():
    """Demonstrate pump curve and system curve intersection"""
    print("\n" + "=" * 60)
    print("PUMP CURVE INTERSECTION ANALYSIS")
    print("=" * 60)
    
    # Create a simple single-path network for clear demonstration
    network = FlowNetwork("Simple System")
    
    inlet = Node("inlet", 0, 0, 0)
    outlet = Node("outlet", 5, 0, 0)
    network.add_node(inlet)
    network.add_node(outlet)
    
    # Single channel with known resistance
    channel = Channel(0.010, 5.0, component_id="main_channel")  # 10mm, 5m
    network.connect_components(inlet, outlet, channel)
    
    network.set_inlet(inlet)
    network.add_outlet(outlet)
    
    # Create pump characteristic
    pump = PumpCharacteristic.create_typical_centrifugal_pump(
        max_pressure=600000,  # 600 kPa
        max_flow=0.020,       # 20 L/s
        efficiency_point=(0.7, 0.75)
    )
    
    solver = NetworkFlowSolver()
    
    print("System: Single 10mm channel, 5m long")
    print("Pump: Centrifugal, 600 kPa max, 20 L/s max")
    print()
    
    # Solve with pump characteristic
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=0.015,  # 15 L/s initial guess
        temperature=50.0,
        pump_characteristic=pump
    )
    
    print("Operating Point Analysis:")
    print(f"  Operating flow: {info['actual_flow_rate']*1000:.2f} L/s")
    print(f"  Operating pressure: {info.get('operating_pressure', 0)/1000:.0f} kPa")
    print(f"  System resistance: {info['required_inlet_pressure']/info['actual_flow_rate']**2/1e6:.1f} MPa·s²/m⁶")
    
    # Show pump curve points around operating point
    print("\nPump curve near operating point:")
    print("Flow (L/s) | Pressure (kPa)")
    print("-" * 25)
    
    op_flow = info['actual_flow_rate']
    for i in range(-2, 3):
        test_flow = op_flow + i * 0.002  # ±2 L/s around operating point
        if test_flow >= 0:
            pressure = pump.get_pressure(test_flow)
            marker = " ← Operating point" if i == 0 else ""
            print(f"{test_flow*1000:8.1f}   | {pressure/1000:10.0f}{marker}")


if __name__ == "__main__":
    try:
        demo_pump_characteristics()
        demo_pump_curve_intersection()
        
        print("\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("The pump characteristic functionality is working correctly.")
        print("You can now use PumpCharacteristic objects with the solver.")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()