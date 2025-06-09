"""
Demo functions for showcasing network flow solver capabilities
"""

from typing import Tuple

from ..network.flow_network import FlowNetwork
from ..components.channel import Channel
from ..components.nozzle import Nozzle, NozzleType
from ..solvers.network_flow_solver import NetworkFlowSolver
def create_simple_tree_example() -> Tuple[FlowNetwork, float, float]:
    """Create a simple tree network example"""
    network = FlowNetwork("Simple Tree Example")
    
    # Create nodes
    inlet = network.create_node("Inlet", elevation=0.0)
    junction1 = network.create_node("Junction1", elevation=1.0)
    branch1_end = network.create_node("Branch1_End", elevation=2.0)
    branch2_end = network.create_node("Branch2_End", elevation=1.5)
    outlet1 = network.create_node("Outlet1", elevation=2.0)
    outlet2 = network.create_node("Outlet2", elevation=1.5)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create components with better nozzle sizing
    main_channel = Channel(diameter=0.08, length=10.0, name="Main Channel")
    branch1_channel = Channel(diameter=0.05, length=8.0, name="Branch1 Channel")
    branch2_channel = Channel(diameter=0.04, length=6.0, name="Branch2 Channel")
    # Further increase nozzle sizes to reduce pressure drops and velocities
    nozzle1 = Nozzle(diameter=0.025, nozzle_type=NozzleType.VENTURI, name="Nozzle1")
    nozzle2 = Nozzle(diameter=0.020, nozzle_type=NozzleType.SHARP_EDGED, name="Nozzle2")
    
    # Connect components to form tree structure
    # Main line: Inlet -> Junction1
    network.connect_components(inlet, junction1, main_channel)
    
    # Branch 1: Junction1 -> Branch1_End -> Outlet1 (through nozzle)
    network.connect_components(junction1, branch1_end, branch1_channel)
    network.connect_components(branch1_end, outlet1, nozzle1)
    
    # Branch 2: Junction1 -> Branch2_End -> Outlet2 (through nozzle)
    network.connect_components(junction1, branch2_end, branch2_channel)
    network.connect_components(branch2_end, outlet2, nozzle2)
    
    total_flow_rate = 0.015  # 15 L/s
    temperature = 40  # °C
    
    return network, total_flow_rate, temperature


def demonstrate_hydraulic_approaches_comparison():
    """Demonstrate the difference between old and new hydraulic approaches"""
    print("DEMONSTRATION: COMPARISON OF HYDRAULIC APPROACHES")
    print("="*60)
    print("This demonstration shows the difference between:")
    print("1. OLD APPROACH: Trying to equalize pressure drops (INCORRECT)")
    print("2. NEW APPROACH: Flow distribution based on resistance (CORRECT)")
    print()
    
    # Create solver
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Create example network
    network, total_flow_rate, temperature = create_simple_tree_example()
    
    # Print network info
    network.print_network_info()
    
    # Validate network
    is_valid, errors = network.validate_network()
    print(f"\nNetwork validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")
        return
    
    print("\n" + "="*70)
    print("APPROACH 1: CORRECT HYDRAULIC PHYSICS (NEW)")
    print("="*70)
    print("✓ Flow distributes based on path resistance (conductance)")
    print("✓ Pressure at junction points equalizes")
    print("✓ Different paths can have different total pressure drops")
    print("✓ Mass conservation at all junctions")
    
    # Solve with correct hydraulics
    connection_flows_new, solution_info_new = solver.solve_network_flow(
        network, total_flow_rate, temperature, inlet_pressure=200000.0
    )
    
    # Print results
    solver.print_results(network, connection_flows_new, solution_info_new)
    
    # Calculate path pressure drops for analysis
    paths = network.get_paths_to_outlets()
    print(f"\n📊 PATH ANALYSIS (CORRECT APPROACH):")
    for i, path in enumerate(paths):
        path_flow = 0.0
        path_pressure_drop = 0.0
        path_components = []
        
        for connection in path:
            component = connection.component
            flow = connection_flows_new[component.id]
            dp = solution_info_new['pressure_drops'][component.id]
            
            # For path flow, use the flow of the last (unique) component
            if len([p for p in paths if any(c.component.id == component.id for c in p)]) == 1:
                path_flow = flow
            
            path_pressure_drop += dp
            path_components.append(f"{component.name}({flow*1000:.1f}L/s)")
        
        # If no unique component found, estimate from outlet flow
        if path_flow == 0.0:
            outlet_node = path[-1].to_node
            for connection in network.reverse_adjacency[outlet_node.id]:
                path_flow = connection_flows_new[connection.component.id]
                break
        
        print(f"  Path {i+1}: {' -> '.join(path_components)}")
        print(f"    Flow: {path_flow*1000:.1f} L/s, Total ΔP: {path_pressure_drop/1000:.1f} kPa")
    
    print("\n" + "="*70)
    print("APPROACH 2: INCORRECT PRESSURE DROP EQUALIZATION (OLD)")
    print("="*70)
    print("❌ Tries to equalize pressure drops across all paths")
    print("❌ Can lead to unrealistic flow distributions")
    print("❌ Does not follow correct hydraulic principles")
    
    # Solve with legacy approach
    connection_flows_old, solution_info_old = solver.solve_network_flow_legacy(
        network, total_flow_rate, temperature, inlet_pressure=200000.0
    )
    
    # Print results
    solver.print_results(network, connection_flows_old, solution_info_old)
    
    # Calculate path pressure drops for analysis
    print(f"\n📊 PATH ANALYSIS (OLD APPROACH):")
    for i, path in enumerate(paths):
        path_flow = 0.0
        path_pressure_drop = 0.0
        path_components = []
        
        for connection in path:
            component = connection.component
            flow = connection_flows_old[component.id]
            dp = solution_info_old['pressure_drops'][component.id]
            
            # For path flow, use the flow of the last (unique) component
            if len([p for p in paths if any(c.component.id == component.id for c in p)]) == 1:
                path_flow = flow
            
            path_pressure_drop += dp
            path_components.append(f"{component.name}({flow*1000:.1f}L/s)")
        
        # If no unique component found, estimate from outlet flow
        if path_flow == 0.0:
            outlet_node = path[-1].to_node
            for connection in network.reverse_adjacency[outlet_node.id]:
                path_flow = connection_flows_old[connection.component.id]
                break
        
        print(f"  Path {i+1}: {' -> '.join(path_components)}")
        print(f"    Flow: {path_flow*1000:.1f} L/s, Total ΔP: {path_pressure_drop/1000:.1f} kPa")
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    # Compare flow distributions
    outlet_flows_new = []
    outlet_flows_old = []
    
    for outlet in network.outlet_nodes:
        for connection in network.reverse_adjacency[outlet.id]:
            outlet_flows_new.append(connection_flows_new[connection.component.id])
            outlet_flows_old.append(connection_flows_old[connection.component.id])
            break
    
    print(f"Flow Distribution Comparison:")
    for i, (flow_new, flow_old) in enumerate(zip(outlet_flows_new, outlet_flows_old)):
        print(f"  Outlet {i+1}: NEW={flow_new*1000:.1f} L/s, OLD={flow_old*1000:.1f} L/s")
    
    # Calculate flow balance
    flow_ratio_new = max(outlet_flows_new) / min(outlet_flows_new) if min(outlet_flows_new) > 0 else float('inf')
    flow_ratio_old = max(outlet_flows_old) / min(outlet_flows_old) if min(outlet_flows_old) > 0 else float('inf')
    
    print(f"\nFlow Balance (max/min ratio):")
    print(f"  NEW approach: {flow_ratio_new:.2f}")
    print(f"  OLD approach: {flow_ratio_old:.2f}")
    
    if flow_ratio_new < flow_ratio_old:
        print(f"  ✅ NEW approach provides better flow balance")
    else:
        print(f"  ⚠️  OLD approach provides better flow balance (but may be incorrect)")
    
    print(f"\nKey Insights:")
    print(f"✓ NEW approach follows correct hydraulic principles")
    print(f"✓ Flow distributes naturally based on system resistance")
    print(f"✓ Different pressure drops are normal and expected")
    print(f"✓ Junction pressures are properly balanced")


def demonstrate_proper_hydraulic_analysis():
    """Demonstrate the correct hydraulic system analysis approach"""
    print("DEMONSTRATION: PROPER HYDRAULIC SYSTEM ANALYSIS")
    print("="*60)
    
    # Create solver
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Create example network
    network, total_flow_rate, temperature = create_simple_tree_example()
    
    # Print network info
    network.print_network_info()
    
    # Validate network
    is_valid, errors = network.validate_network()
    print(f"\nNetwork validation: {'PASSED' if is_valid else 'FAILED'}")
    if errors:
        for error in errors:
            print(f"  Error: {error}")
        return
    
    print("\n" + "="*70)
    print("CASE 1: ADEQUATE PUMP PRESSURE")
    print("="*70)
    print("Using a pump that provides 200 kPa inlet pressure")
    
    # Solve with adequate inlet pressure
    connection_flows1, solution_info1 = solver.solve_network_flow(
        network, total_flow_rate, temperature,
        inlet_pressure=200000.0  # 200 kPa from pump
    )
    
    # Print results
    solver.print_results(network, connection_flows1, solution_info1)
    
    # Analyze system adequacy
    analysis1 = solver.analyze_system_adequacy(network, connection_flows1, solution_info1)
    print(f"\n🔍 SYSTEM ANALYSIS:")
    print(f"   System adequate: {'✅ YES' if analysis1['adequate'] else '❌ NO'}")
    if analysis1['issues']:
        print("   Issues found:")
        for issue in analysis1['issues']:
            print(f"   - {issue}")
    
    print("\n" + "="*70)
    print("CASE 2: INSUFFICIENT PUMP PRESSURE")
    print("="*70)
    print("Using an undersized pump that provides only 120 kPa inlet pressure")
    
    # Solve with insufficient inlet pressure
    connection_flows2, solution_info2 = solver.solve_network_flow(
        network, total_flow_rate, temperature,
        inlet_pressure=120000.0  # 120 kPa - insufficient
    )
    
    # Print results
    solver.print_results(network, connection_flows2, solution_info2)
    
    # Analyze system adequacy
    analysis2 = solver.analyze_system_adequacy(network, connection_flows2, solution_info2)
    print(f"\n🔍 SYSTEM ANALYSIS:")
    print(f"   System adequate: {'✅ YES' if analysis2['adequate'] else '❌ NO'}")
    if analysis2['issues']:
        print("   Issues found:")
        for issue in analysis2['issues']:
            print(f"   - {issue}")
    if analysis2['recommendations']:
        print("   Recommendations:")
        for rec in analysis2['recommendations']:
            print(f"   - {rec}")
    
    print("\n" + "="*70)
    print("CASE 3: VERY LOW PUMP PRESSURE (UNREALISTIC)")
    print("="*70)
    print("Using a severely undersized pump that provides only 50 kPa inlet pressure")
    
    # Solve with very low inlet pressure
    connection_flows3, solution_info3 = solver.solve_network_flow(
        network, total_flow_rate, temperature,
        inlet_pressure=50000.0  # 50 kPa - severely insufficient
    )
    
    # Print results
    solver.print_results(network, connection_flows3, solution_info3)
    
    # Analyze system adequacy
    analysis3 = solver.analyze_system_adequacy(network, connection_flows3, solution_info3)
    print(f"\n🔍 SYSTEM ANALYSIS:")
    print(f"   System adequate: {'✅ YES' if analysis3['adequate'] else '❌ NO'}")
    if analysis3['issues']:
        print("   Issues found:")
        for issue in analysis3['issues']:
            print(f"   - {issue}")
    if analysis3['recommendations']:
        print("   Recommendations:")
        for rec in analysis3['recommendations']:
            print(f"   - {rec}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS:")
    print("="*70)
    print("✓ Inlet pressure is determined by pump specifications")
    print("✓ Outlet pressures are calculated from inlet pressure minus losses")
    print("✓ If outlet pressures are too low, you must:")
    print("  - Increase pump pressure, OR")
    print("  - Reduce system losses (larger pipes, fewer restrictions), OR")
    print("  - Reduce flow rate requirements")
    print("✓ You cannot arbitrarily set outlet pressures - they are system outputs!")
    print("✓ Negative pressures indicate system design problems that must be fixed")


def main():
    """Demonstrate the network-based flow calculator"""
    print("NETWORK-BASED LUBRICATION FLOW DISTRIBUTION CALCULATOR")
    print("="*60)
    
    # Run the comparison demonstration
    demonstrate_hydraulic_approaches_comparison()
    
    print("\n\n")
    
    # Run the original demonstration
    demonstrate_proper_hydraulic_analysis()


if __name__ == "__main__":
    # Existing setup from main()
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    network, total_flow_rate, temperature = create_simple_tree_example()

    print("\n" + "="*70)
    print("APPROACH 3: MATRIX-BASED NODAL SOLVER (SCALABLE)")
    print("="*70)
    print("✓ Single linear solve for nodal pressures")
    print("✓ Direct flow via conductance" )

    # Solve using new nodal method
    conn_flows_nodal, sol_info_nodal = solver.solve_network_flow_nodal(
        network,
        total_flow_rate,
        temperature,
        inlet_pressure=200000.0,
        outlet_pressure=101325.0
    )

    # Print results
    solver.print_results(network, conn_flows_nodal, sol_info_nodal)

    # Optional: path-level comparison summary
    paths = network.get_paths_to_outlets()
    print(f"\nPATH PRESSURE DROPS (NODAL):")
    for i, path in enumerate(paths, 1):
        total_dp = sum(sol_info_nodal['pressure_drops'][c.component.id] for c in path)
        print(f"  Path {i}: Total ΔP = {total_dp/1000:.1f} kPa")
    
    #main()
