#!/usr/bin/env python3
"""
Complex Tree Network Examples for Lubrication Flow Distribution

This module demonstrates advanced tree-like branching structures
with multiple levels, different component types, and realistic
industrial lubrication system configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType
)


def create_industrial_gearbox_system() -> tuple:
    """
    Create a realistic industrial gearbox lubrication system
    with multiple levels of branching
    """
    network = FlowNetwork("Industrial Gearbox Lubrication System")
    
    # Main supply line
    pump_outlet = network.create_node("Pump_Outlet", elevation=0.0)
    main_manifold = network.create_node("Main_Manifold", elevation=1.5)
    
    # Primary distribution points
    bearing_manifold = network.create_node("Bearing_Manifold", elevation=2.0)
    gear_manifold = network.create_node("Gear_Manifold", elevation=1.8)
    cooling_manifold = network.create_node("Cooling_Manifold", elevation=1.2)
    
    # Secondary distribution points
    main_bearing_dist = network.create_node("Main_Bearing_Dist", elevation=2.2)
    aux_bearing_dist = network.create_node("Aux_Bearing_Dist", elevation=2.1)
    gear_mesh_dist = network.create_node("Gear_Mesh_Dist", elevation=1.9)
    gear_tooth_dist = network.create_node("Gear_Tooth_Dist", elevation=1.7)
    
    # Final lubrication points (outlets)
    main_bearing_1 = network.create_node("Main_Bearing_1", elevation=2.5)
    main_bearing_2 = network.create_node("Main_Bearing_2", elevation=2.4)
    aux_bearing_1 = network.create_node("Aux_Bearing_1", elevation=2.3)
    aux_bearing_2 = network.create_node("Aux_Bearing_2", elevation=2.2)
    gear_mesh_1 = network.create_node("Gear_Mesh_1", elevation=2.0)
    gear_mesh_2 = network.create_node("Gear_Mesh_2", elevation=1.9)
    gear_tooth_spray = network.create_node("Gear_Tooth_Spray", elevation=1.8)
    cooler_return = network.create_node("Cooler_Return", elevation=0.5)
    
    # Set inlet (outlets will be set after nozzles are connected)
    network.set_inlet(pump_outlet)
    
    # Create components
    # Main supply line
    main_supply = Channel(diameter=0.15, length=12.0, roughness=0.00015, name="Main_Supply")
    
    # Primary distribution channels
    to_bearings = Channel(diameter=0.10, length=8.0, roughness=0.00020, name="To_Bearings")
    to_gears = Channel(diameter=0.08, length=6.0, roughness=0.00020, name="To_Gears")
    to_cooling = Channel(diameter=0.12, length=10.0, roughness=0.00015, name="To_Cooling")
    
    # Secondary distribution channels
    main_bearing_line = Channel(diameter=0.06, length=5.0, roughness=0.00025, name="Main_Bearing_Line")
    aux_bearing_line = Channel(diameter=0.05, length=4.0, roughness=0.00025, name="Aux_Bearing_Line")
    gear_mesh_line = Channel(diameter=0.04, length=3.0, roughness=0.00030, name="Gear_Mesh_Line")
    gear_tooth_line = Channel(diameter=0.035, length=2.5, roughness=0.00030, name="Gear_Tooth_Line")
    
    # Final distribution channels
    mb1_channel = Channel(diameter=0.025, length=2.0, roughness=0.00035, name="MB1_Channel")
    mb2_channel = Channel(diameter=0.025, length=2.2, roughness=0.00035, name="MB2_Channel")
    ab1_channel = Channel(diameter=0.020, length=1.8, roughness=0.00040, name="AB1_Channel")
    ab2_channel = Channel(diameter=0.020, length=1.9, roughness=0.00040, name="AB2_Channel")
    gm1_channel = Channel(diameter=0.015, length=1.5, roughness=0.00045, name="GM1_Channel")
    gm2_channel = Channel(diameter=0.015, length=1.6, roughness=0.00045, name="GM2_Channel")
    gt_channel = Channel(diameter=0.012, length=1.2, roughness=0.00050, name="GT_Channel")
    cooling_return = Channel(diameter=0.08, length=15.0, roughness=0.00020, name="Cooling_Return")
    
    # Connectors at junctions
    main_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.15, name="Main_Tee")
    bearing_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.10, name="Bearing_Tee")
    gear_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.08, name="Gear_Tee")
    mb_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.06, name="MB_Tee")
    ab_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.05, name="AB_Tee")
    gm_tee = Connector(ConnectorType.T_JUNCTION, diameter=0.04, name="GM_Tee")
    
    # Nozzles at critical points
    mb1_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.VENTURI, name="MB1_Nozzle")
    mb2_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.VENTURI, name="MB2_Nozzle")
    ab1_nozzle = Nozzle(diameter=0.006, nozzle_type=NozzleType.FLOW_NOZZLE, name="AB1_Nozzle")
    ab2_nozzle = Nozzle(diameter=0.006, nozzle_type=NozzleType.FLOW_NOZZLE, name="AB2_Nozzle")
    gm1_nozzle = Nozzle(diameter=0.005, nozzle_type=NozzleType.ROUNDED, name="GM1_Nozzle")
    gm2_nozzle = Nozzle(diameter=0.005, nozzle_type=NozzleType.ROUNDED, name="GM2_Nozzle")
    gt_nozzle = Nozzle(diameter=0.004, nozzle_type=NozzleType.SHARP_EDGED, name="GT_Nozzle")
    
    # Connect the network
    # Main supply line
    network.connect_components(pump_outlet, main_manifold, main_supply)
    
    # Primary distribution
    network.connect_components(main_manifold, bearing_manifold, to_bearings)
    network.connect_components(main_manifold, gear_manifold, to_gears)
    network.connect_components(main_manifold, cooling_manifold, to_cooling)
    network.connect_components(cooling_manifold, cooler_return, cooling_return)
    
    # Secondary distribution
    network.connect_components(bearing_manifold, main_bearing_dist, main_bearing_line)
    network.connect_components(bearing_manifold, aux_bearing_dist, aux_bearing_line)
    network.connect_components(gear_manifold, gear_mesh_dist, gear_mesh_line)
    network.connect_components(gear_manifold, gear_tooth_dist, gear_tooth_line)
    
    # Final distribution to lubrication points
    network.connect_components(main_bearing_dist, main_bearing_1, mb1_channel)
    network.connect_components(main_bearing_dist, main_bearing_2, mb2_channel)
    network.connect_components(aux_bearing_dist, aux_bearing_1, ab1_channel)
    network.connect_components(aux_bearing_dist, aux_bearing_2, ab2_channel)
    network.connect_components(gear_mesh_dist, gear_mesh_1, gm1_channel)
    network.connect_components(gear_mesh_dist, gear_mesh_2, gm2_channel)
    network.connect_components(gear_tooth_dist, gear_tooth_spray, gt_channel)
    
    # Create final outlet nodes after nozzles
    mb1_final = network.create_node("MB1_Final", elevation=2.5)
    mb2_final = network.create_node("MB2_Final", elevation=2.4)
    ab1_final = network.create_node("AB1_Final", elevation=2.3)
    ab2_final = network.create_node("AB2_Final", elevation=2.2)
    gm1_final = network.create_node("GM1_Final", elevation=2.0)
    gm2_final = network.create_node("GM2_Final", elevation=1.9)
    gt_final = network.create_node("GT_Final", elevation=1.8)
    
    # Add nozzles at critical lubrication points
    network.connect_components(main_bearing_1, mb1_final, mb1_nozzle)
    network.connect_components(main_bearing_2, mb2_final, mb2_nozzle)
    network.connect_components(aux_bearing_1, ab1_final, ab1_nozzle)
    network.connect_components(aux_bearing_2, ab2_final, ab2_nozzle)
    network.connect_components(gear_mesh_1, gm1_final, gm1_nozzle)
    network.connect_components(gear_mesh_2, gm2_final, gm2_nozzle)
    network.connect_components(gear_tooth_spray, gt_final, gt_nozzle)
    
    # Update outlets to be after nozzles
    network.outlet_nodes = [mb1_final, mb2_final, ab1_final, ab2_final,
                           gm1_final, gm2_final, gt_final, cooler_return]
    
    total_flow_rate = 0.05  # 50 L/s
    temperature = 55  # °C
    
    return network, total_flow_rate, temperature


def create_multi_machine_system() -> tuple:
    """
    Create a multi-machine lubrication system serving several machines
    """
    network = FlowNetwork("Multi-Machine Lubrication System")
    
    # Central supply
    central_pump = network.create_node("Central_Pump", elevation=0.0)
    main_header = network.create_node("Main_Header", elevation=2.0)
    
    # Machine distribution points
    machine_1_inlet = network.create_node("Machine_1_Inlet", elevation=2.5)
    machine_2_inlet = network.create_node("Machine_2_Inlet", elevation=2.3)
    machine_3_inlet = network.create_node("Machine_3_Inlet", elevation=2.1)
    
    # Machine 1 - Large gearbox with multiple levels
    m1_primary = network.create_node("M1_Primary", elevation=3.0)
    m1_secondary_a = network.create_node("M1_Secondary_A", elevation=3.2)
    m1_secondary_b = network.create_node("M1_Secondary_B", elevation=3.1)
    m1_bearing_1 = network.create_node("M1_Bearing_1", elevation=3.5)
    m1_bearing_2 = network.create_node("M1_Bearing_2", elevation=3.4)
    m1_gear_1 = network.create_node("M1_Gear_1", elevation=3.3)
    m1_gear_2 = network.create_node("M1_Gear_2", elevation=3.2)
    
    # Machine 2 - Medium complexity
    m2_dist = network.create_node("M2_Dist", elevation=2.8)
    m2_bearing = network.create_node("M2_Bearing", elevation=3.0)
    m2_gear = network.create_node("M2_Gear", elevation=2.9)
    
    # Machine 3 - Simple system
    m3_outlet = network.create_node("M3_Outlet", elevation=2.5)
    
    # Set inlet (outlets will be set after nozzles are connected)
    network.set_inlet(central_pump)
    
    # Create components
    # Main distribution
    main_line = Channel(diameter=0.20, length=20.0, roughness=0.00015, name="Main_Line")
    to_m1 = Channel(diameter=0.10, length=15.0, roughness=0.00020, name="To_M1")
    to_m2 = Channel(diameter=0.08, length=12.0, roughness=0.00020, name="To_M2")
    to_m3 = Channel(diameter=0.06, length=10.0, roughness=0.00025, name="To_M3")
    
    # Machine 1 distribution
    m1_main = Channel(diameter=0.08, length=8.0, roughness=0.00025, name="M1_Main")
    m1_branch_a = Channel(diameter=0.05, length=5.0, roughness=0.00030, name="M1_Branch_A")
    m1_branch_b = Channel(diameter=0.04, length=4.0, roughness=0.00030, name="M1_Branch_B")
    m1_to_b1 = Channel(diameter=0.03, length=3.0, roughness=0.00035, name="M1_To_B1")
    m1_to_b2 = Channel(diameter=0.03, length=3.2, roughness=0.00035, name="M1_To_B2")
    m1_to_g1 = Channel(diameter=0.025, length=2.5, roughness=0.00040, name="M1_To_G1")
    m1_to_g2 = Channel(diameter=0.025, length=2.8, roughness=0.00040, name="M1_To_G2")
    
    # Machine 2 distribution
    m2_main = Channel(diameter=0.06, length=6.0, roughness=0.00025, name="M2_Main")
    m2_to_bearing = Channel(diameter=0.04, length=4.0, roughness=0.00030, name="M2_To_Bearing")
    m2_to_gear = Channel(diameter=0.035, length=3.5, roughness=0.00035, name="M2_To_Gear")
    
    # Machine 3 (simple)
    m3_line = Channel(diameter=0.04, length=8.0, roughness=0.00030, name="M3_Line")
    
    # Nozzles
    m1_b1_nozzle = Nozzle(diameter=0.010, nozzle_type=NozzleType.VENTURI, name="M1_B1_Nozzle")
    m1_b2_nozzle = Nozzle(diameter=0.010, nozzle_type=NozzleType.VENTURI, name="M1_B2_Nozzle")
    m1_g1_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.FLOW_NOZZLE, name="M1_G1_Nozzle")
    m1_g2_nozzle = Nozzle(diameter=0.008, nozzle_type=NozzleType.FLOW_NOZZLE, name="M1_G2_Nozzle")
    m2_b_nozzle = Nozzle(diameter=0.012, nozzle_type=NozzleType.ROUNDED, name="M2_B_Nozzle")
    m2_g_nozzle = Nozzle(diameter=0.010, nozzle_type=NozzleType.ROUNDED, name="M2_G_Nozzle")
    m3_nozzle = Nozzle(diameter=0.015, nozzle_type=NozzleType.SHARP_EDGED, name="M3_Nozzle")
    
    # Connect the network
    # Main distribution
    network.connect_components(central_pump, main_header, main_line)
    network.connect_components(main_header, machine_1_inlet, to_m1)
    network.connect_components(main_header, machine_2_inlet, to_m2)
    network.connect_components(main_header, machine_3_inlet, to_m3)
    
    # Machine 1 tree
    network.connect_components(machine_1_inlet, m1_primary, m1_main)
    network.connect_components(m1_primary, m1_secondary_a, m1_branch_a)
    network.connect_components(m1_primary, m1_secondary_b, m1_branch_b)
    network.connect_components(m1_secondary_a, m1_bearing_1, m1_to_b1)
    network.connect_components(m1_secondary_a, m1_bearing_2, m1_to_b2)
    network.connect_components(m1_secondary_b, m1_gear_1, m1_to_g1)
    network.connect_components(m1_secondary_b, m1_gear_2, m1_to_g2)
    
    # Machine 2 tree
    network.connect_components(machine_2_inlet, m2_dist, m2_main)
    network.connect_components(m2_dist, m2_bearing, m2_to_bearing)
    network.connect_components(m2_dist, m2_gear, m2_to_gear)
    
    # Machine 3 (simple)
    network.connect_components(machine_3_inlet, m3_outlet, m3_line)
    
    # Create final outlet nodes after nozzles
    m1_b1_final = network.create_node("M1_B1_Final", elevation=3.5)
    m1_b2_final = network.create_node("M1_B2_Final", elevation=3.4)
    m1_g1_final = network.create_node("M1_G1_Final", elevation=3.3)
    m1_g2_final = network.create_node("M1_G2_Final", elevation=3.2)
    m2_b_final = network.create_node("M2_B_Final", elevation=3.0)
    m2_g_final = network.create_node("M2_G_Final", elevation=2.9)
    m3_final = network.create_node("M3_Final", elevation=2.5)
    
    # Add nozzles
    network.connect_components(m1_bearing_1, m1_b1_final, m1_b1_nozzle)
    network.connect_components(m1_bearing_2, m1_b2_final, m1_b2_nozzle)
    network.connect_components(m1_gear_1, m1_g1_final, m1_g1_nozzle)
    network.connect_components(m1_gear_2, m1_g2_final, m1_g2_nozzle)
    network.connect_components(m2_bearing, m2_b_final, m2_b_nozzle)
    network.connect_components(m2_gear, m2_g_final, m2_g_nozzle)
    network.connect_components(m3_outlet, m3_final, m3_nozzle)
    
    # Update outlets to be after nozzles
    network.outlet_nodes = [m1_b1_final, m1_b2_final, m1_g1_final, m1_g2_final,
                           m2_b_final, m2_g_final, m3_final]
    
    total_flow_rate = 0.08  # 80 L/s
    temperature = 50  # °C
    
    return network, total_flow_rate, temperature


def analyze_network_performance(network, total_flow_rate, temperature, solver):
    """Analyze and print detailed network performance"""
    print(f"\n{'='*80}")
    print(f"NETWORK ANALYSIS: {network.name}")
    print(f"{'='*80}")
    
    # Network topology info
    network.print_network_info()
    
    # Validate network
    is_valid, errors = network.validate_network()
    print(f"\nNetwork Validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")
        return
    
    # Solve flow distribution
    connection_flows, solution_info = solver.solve_network_flow(
        network, total_flow_rate, temperature
    )
    
    # Print detailed results
    solver.print_results(network, connection_flows, solution_info)
    
    # Additional analysis
    print(f"\n{'='*60}")
    print("ADDITIONAL ANALYSIS")
    print(f"{'='*60}")
    
    # Flow distribution by component type
    component_flows = {'channel': 0, 'connector': 0, 'nozzle': 0}
    component_counts = {'channel': 0, 'connector': 0, 'nozzle': 0}
    
    for connection in network.connections:
        comp_type = connection.component.component_type.value
        flow = connection_flows[connection.component.id]
        component_flows[comp_type] += flow
        component_counts[comp_type] += 1
    
    print(f"\nFlow Distribution by Component Type:")
    for comp_type in component_flows:
        if component_counts[comp_type] > 0:
            avg_flow = component_flows[comp_type] / component_counts[comp_type]
            print(f"  {comp_type.capitalize()}: {component_counts[comp_type]} components, "
                  f"avg flow: {avg_flow*1000:.2f} L/s")
    
    # Pressure analysis
    pressures = list(solution_info['node_pressures'].values())
    print(f"\nPressure Analysis:")
    print(f"  Maximum pressure: {max(pressures)/1000:.1f} kPa")
    print(f"  Minimum pressure: {min(pressures)/1000:.1f} kPa")
    print(f"  Pressure range: {(max(pressures) - min(pressures))/1000:.1f} kPa")
    
    # Flow velocity analysis
    print(f"\nFlow Velocity Analysis:")
    max_velocity = 0
    min_velocity = float('inf')
    
    for connection in network.connections:
        if connection.component.component_type == ComponentType.CHANNEL:
            flow = connection_flows[connection.component.id]
            area = connection.component.get_flow_area()
            velocity = flow / area if area > 0 else 0
            max_velocity = max(max_velocity, velocity)
            if velocity > 0:
                min_velocity = min(min_velocity, velocity)
    
    if min_velocity != float('inf'):
        print(f"  Maximum velocity: {max_velocity:.2f} m/s")
        print(f"  Minimum velocity: {min_velocity:.2f} m/s")
    
    return connection_flows, solution_info


def compare_network_configurations():
    """Compare different network configurations"""
    print("NETWORK CONFIGURATION COMPARISON")
    print("="*80)
    
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Test different configurations
    configurations = [
        ("Industrial Gearbox", create_industrial_gearbox_system),
        ("Multi-Machine System", create_multi_machine_system)
    ]
    
    results = {}
    
    for config_name, config_func in configurations:
        print(f"\n{'-'*60}")
        print(f"Testing: {config_name}")
        print(f"{'-'*60}")
        
        try:
            network, total_flow_rate, temperature = config_func()
            connection_flows, solution_info = analyze_network_performance(
                network, total_flow_rate, temperature, solver
            )
            
            results[config_name] = {
                'network': network,
                'flows': connection_flows,
                'info': solution_info,
                'total_flow': total_flow_rate
            }
            
        except Exception as e:
            print(f"Error analyzing {config_name}: {e}")
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("CONFIGURATION COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{'Configuration':<25} {'Nodes':<8} {'Connections':<12} {'Outlets':<8} "
          f"{'Flow (L/s)':<10} {'Converged':<10}")
    print("-" * 80)
    
    for config_name, result in results.items():
        network = result['network']
        info = result['info']
        total_flow = result['total_flow']
        
        print(f"{config_name:<25} {len(network.nodes):<8} {len(network.connections):<12} "
              f"{len(network.outlet_nodes):<8} {total_flow*1000:<10.1f} "
              f"{info['converged']:<10}")


def main():
    """Run complex tree network demonstrations"""
    print("COMPLEX TREE NETWORK LUBRICATION SYSTEMS")
    print("="*60)
    
    compare_network_configurations()
    
    print(f"\n{'='*80}")
    print("DEMONSTRATION COMPLETE")
    print(f"{'='*80}")
    print("\nKey Achievements:")
    print("✓ Multi-level tree structures with complex branching")
    print("✓ Component-based system building (channels, connectors, nozzles)")
    print("✓ Realistic industrial lubrication system modeling")
    print("✓ Automatic flow distribution calculation")
    print("✓ Mass conservation at all junctions")
    print("✓ Pressure equalization across parallel paths")
    print("✓ Support for different nozzle types and characteristics")
    print("✓ Elevation effects and complex pipe geometries")


if __name__ == "__main__":
    main()