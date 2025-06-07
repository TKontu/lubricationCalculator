#!/usr/bin/env python3
"""
Test compatibility between original and refactored implementations
"""

import sys
import os

# Test that we can import from both the original and refactored versions
def test_imports():
    """Test that all classes can be imported from both versions"""
    print("Testing imports...")
    
    # Import from original
    from network_lubrication_flow_tool import (
        ComponentType, ConnectorType, NozzleType,
        FlowComponent, Channel, Connector, Nozzle,
        Node, Connection, FlowNetwork,
        SolverConfig, NetworkFlowSolver
    )
    print("✓ Original imports successful")
    
    # Import from refactored package
    from lubrication_flow_package import (
        ComponentType as ComponentType2, ConnectorType as ConnectorType2, NozzleType as NozzleType2,
        FlowComponent as FlowComponent2, Channel as Channel2, Connector as Connector2, Nozzle as Nozzle2,
        Node as Node2, Connection as Connection2, FlowNetwork as FlowNetwork2,
        SolverConfig as SolverConfig2, NetworkFlowSolver as NetworkFlowSolver2
    )
    print("✓ Refactored package imports successful")
    
    return True

def test_functionality_equivalence():
    """Test that both versions produce the same results"""
    print("\nTesting functionality equivalence...")
    
    # Import both versions
    import network_lubrication_flow_tool as original
    import lubrication_flow_package as refactored
    
    # Create identical networks using both versions
    def create_test_network_original():
        network = original.FlowNetwork("Test Network")
        inlet = network.create_node("Inlet", elevation=0.0)
        outlet1 = network.create_node("Outlet1", elevation=1.0)
        outlet2 = network.create_node("Outlet2", elevation=1.0)
        
        network.set_inlet(inlet)
        network.add_outlet(outlet1)
        network.add_outlet(outlet2)
        
        # Create a simple branching network
        junction = network.create_node("Junction", elevation=0.5)
        
        main_channel = original.Channel(diameter=0.05, length=5.0, name="Main")
        branch1 = original.Channel(diameter=0.03, length=3.0, name="Branch1")
        branch2 = original.Channel(diameter=0.03, length=3.0, name="Branch2")
        
        network.connect_components(inlet, junction, main_channel)
        network.connect_components(junction, outlet1, branch1)
        network.connect_components(junction, outlet2, branch2)
        
        return network
    
    def create_test_network_refactored():
        network = refactored.FlowNetwork("Test Network")
        inlet = network.create_node("Inlet", elevation=0.0)
        outlet1 = network.create_node("Outlet1", elevation=1.0)
        outlet2 = network.create_node("Outlet2", elevation=1.0)
        
        network.set_inlet(inlet)
        network.add_outlet(outlet1)
        network.add_outlet(outlet2)
        
        # Create a simple branching network
        junction = network.create_node("Junction", elevation=0.5)
        
        main_channel = refactored.Channel(diameter=0.05, length=5.0, name="Main")
        branch1 = refactored.Channel(diameter=0.03, length=3.0, name="Branch1")
        branch2 = refactored.Channel(diameter=0.03, length=3.0, name="Branch2")
        
        network.connect_components(inlet, junction, main_channel)
        network.connect_components(junction, outlet1, branch1)
        network.connect_components(junction, outlet2, branch2)
        
        return network
    
    # Create networks
    network_orig = create_test_network_original()
    network_refact = create_test_network_refactored()
    
    # Create solvers
    solver_orig = original.NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    solver_refact = refactored.NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Solve with both
    total_flow = 0.01  # 10 L/s
    temperature = 40.0  # °C
    inlet_pressure = 200000.0  # 200 kPa
    
    flows_orig, info_orig = solver_orig.solve_network_flow(
        network_orig, total_flow, temperature, inlet_pressure
    )
    
    flows_refact, info_refact = solver_refact.solve_network_flow(
        network_refact, total_flow, temperature, inlet_pressure
    )
    
    # Compare results
    print(f"Original converged: {info_orig['converged']}")
    print(f"Refactored converged: {info_refact['converged']}")
    
    # Compare flow distributions (should be very similar)
    flow_diff_threshold = 1e-6  # Very small tolerance
    max_flow_diff = 0.0
    
    # Get component IDs from both networks
    orig_components = {conn.component.name: conn.component.id for conn in network_orig.connections}
    refact_components = {conn.component.name: conn.component.id for conn in network_refact.connections}
    
    for comp_name in orig_components:
        if comp_name in refact_components:
            orig_flow = flows_orig[orig_components[comp_name]]
            refact_flow = flows_refact[refact_components[comp_name]]
            flow_diff = abs(orig_flow - refact_flow)
            max_flow_diff = max(max_flow_diff, flow_diff)
            
            print(f"  {comp_name}: Original={orig_flow*1000:.3f} L/s, "
                  f"Refactored={refact_flow*1000:.3f} L/s, "
                  f"Diff={flow_diff*1000:.6f} L/s")
    
    if max_flow_diff < flow_diff_threshold:
        print("✓ Flow distributions match within tolerance")
        return True
    else:
        print(f"❌ Flow distributions differ by {max_flow_diff*1000:.6f} L/s")
        return False

def test_class_compatibility():
    """Test that classes have the same interface"""
    print("\nTesting class compatibility...")
    
    import network_lubrication_flow_tool as original
    import lubrication_flow_package as refactored
    
    # Test Channel class
    orig_channel = original.Channel(diameter=0.05, length=10.0)
    refact_channel = refactored.Channel(diameter=0.05, length=10.0)
    
    # Test that they have the same methods
    orig_methods = set(dir(orig_channel))
    refact_methods = set(dir(refact_channel))
    
    # Check for essential methods
    essential_methods = {
        'calculate_pressure_drop', 'get_flow_area', 'validate_flow_rate'
    }
    
    missing_orig = essential_methods - orig_methods
    missing_refact = essential_methods - refact_methods
    
    if missing_orig:
        print(f"❌ Original missing methods: {missing_orig}")
        return False
    
    if missing_refact:
        print(f"❌ Refactored missing methods: {missing_refact}")
        return False
    
    # Test that pressure drop calculations are the same
    fluid_props = {'density': 900.0, 'viscosity': 0.1}
    flow_rate = 0.005  # 5 L/s
    
    dp_orig = orig_channel.calculate_pressure_drop(flow_rate, fluid_props)
    dp_refact = refact_channel.calculate_pressure_drop(flow_rate, fluid_props)
    
    dp_diff = abs(dp_orig - dp_refact)
    if dp_diff < 1e-6:
        print("✓ Channel pressure drop calculations match")
    else:
        print(f"❌ Channel pressure drop differs by {dp_diff:.6f} Pa")
        return False
    
    print("✓ Class compatibility verified")
    return True

def main():
    """Run all compatibility tests"""
    print("TESTING REFACTORED PACKAGE COMPATIBILITY")
    print("="*50)
    
    tests = [
        test_imports,
        test_functionality_equivalence,
        test_class_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print(f"COMPATIBILITY TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All compatibility tests passed!")
        print("✓ Refactored package maintains full compatibility")
        return True
    else:
        print("❌ Some compatibility tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)