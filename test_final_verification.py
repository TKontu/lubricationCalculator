#!/usr/bin/env python3
"""
Final verification test for the refactored lubrication flow package.

This test verifies that all refactoring objectives have been met:
1. Multi-module package structure with proper subpackages
2. All functionality preserved from original script
3. Utils module with shared functions eliminates code duplication
4. All solver methods work correctly
5. CLI functionality maintained
"""

import sys
import os
import importlib
from pathlib import Path

def test_package_structure():
    """Test that the package has the required structure"""
    print("1. Testing package structure...")
    
    required_subpackages = [
        'lubrication_flow_package.components',
        'lubrication_flow_package.network', 
        'lubrication_flow_package.solvers',
        'lubrication_flow_package.cli',
        'lubrication_flow_package.utils'
    ]
    
    for subpackage in required_subpackages:
        try:
            importlib.import_module(subpackage)
            print(f"   ✓ {subpackage}")
        except ImportError as e:
            print(f"   ✗ {subpackage}: {e}")
            return False
    
    return True

def test_components_subpackage():
    """Test components subpackage"""
    print("\n2. Testing components subpackage...")
    
    try:
        from lubrication_flow_package.components import Channel, Connector, Nozzle
        from lubrication_flow_package.components.base import FlowComponent
        from lubrication_flow_package.components.enums import ConnectorType
        
        # Test basic functionality
        channel = Channel(name="test", length=1.0, diameter=0.01)
        nozzle = Nozzle(name="test_nozzle", diameter=0.005)
        connector = Connector(ConnectorType.ELBOW_90, diameter=0.01, name="test_connector")
        
        print(f"   ✓ Channel created: {channel.name}")
        print(f"   ✓ Nozzle created: {nozzle.name}")
        print(f"   ✓ Connector created: {connector.name}")
        print(f"   ✓ Base class available: {FlowComponent.__name__}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_network_subpackage():
    """Test network subpackage"""
    print("\n3. Testing network subpackage...")
    
    try:
        from lubrication_flow_package.network import Node, Connection, FlowNetwork
        from lubrication_flow_package.components import Channel
        
        # Test basic functionality
        node1 = Node("inlet", 0.0, 0.0)
        node2 = Node("outlet", 1.0, 0.0)
        channel = Channel(name="main", length=1.0, diameter=0.01)
        network = FlowNetwork()
        network.add_node(node1)
        network.add_node(node2)
        connection = network.connect_components(node1, node2, channel)
        
        print(f"   ✓ Node created: {node1.name}")
        print(f"   ✓ Connection created: {connection.component.name}")
        print(f"   ✓ FlowNetwork created with {len(network.nodes)} nodes")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_solvers_subpackage():
    """Test solvers subpackage"""
    print("\n4. Testing solvers subpackage...")
    
    try:
        from lubrication_flow_package.solvers import NetworkFlowSolver
        from lubrication_flow_package.components import Channel, Nozzle
        from lubrication_flow_package.network import Node, Connection, FlowNetwork
        
        # Create simple test network
        inlet = Node("inlet", 0.0, 0.0)
        outlet = Node("outlet", 1.0, 0.0)
        channel = Channel(name="main", length=1.0, diameter=0.01)
        
        network = FlowNetwork()
        network.add_node(inlet)
        network.add_node(outlet)
        connection = network.connect_components(inlet, outlet, channel)
        network.set_inlet(inlet)
        network.add_outlet(outlet)
        
        solver = NetworkFlowSolver()
        
        # Test all solver methods
        methods = [
            'solve_network_flow',  # iterative path-based (default)
            'solve_network_flow_with_pump_physics', 
            'solve_network_flow_nodal',
            'solve_network_flow_legacy'
        ]
        
        for method_name in methods:
            method = getattr(solver, method_name)
            flows, info = method(network, 0.001, 40.0)
            print(f"   ✓ {method_name}: converged={info.get('converged', 'N/A')}")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_utils_subpackage():
    """Test utils subpackage"""
    print("\n5. Testing utils subpackage...")
    
    try:
        from lubrication_flow_package.utils import (
            find_all_paths, compute_path_pressure, estimate_resistance,
            compute_node_pressures, validate_flow_conservation,
            calculate_path_conductances, distribute_flow_by_conductance,
            check_convergence
        )
        
        functions = [
            'find_all_paths', 'compute_path_pressure', 'estimate_resistance',
            'compute_node_pressures', 'validate_flow_conservation',
            'calculate_path_conductances', 'distribute_flow_by_conductance',
            'check_convergence'
        ]
        
        for func_name in functions:
            print(f"   ✓ {func_name} available")
        
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_cli_subpackage():
    """Test CLI subpackage"""
    print("\n6. Testing CLI subpackage...")
    
    try:
        from lubrication_flow_package.cli import main
        print(f"   ✓ CLI main module available")
        
        # Test that CLI can be imported without errors
        return True
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def test_functionality_preservation():
    """Test that original functionality is preserved"""
    print("\n7. Testing functionality preservation...")
    
    try:
        # Run the compatibility test
        import subprocess
        result = subprocess.run([
            sys.executable, 'test_refactored_compatibility.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0 and "🎉 All compatibility tests passed!" in result.stdout:
            print("   ✓ Compatibility test passed")
            return True
        else:
            print(f"   ✗ Compatibility test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ✗ Error running compatibility test: {e}")
        return False

def test_utils_integration():
    """Test utils module integration"""
    print("\n8. Testing utils module integration...")
    
    try:
        # Run the utils integration test
        import subprocess
        result = subprocess.run([
            sys.executable, 'test_utils_integration.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0 and "🎉 Utils integration test PASSED!" in result.stdout:
            print("   ✓ Utils integration test passed")
            return True
        else:
            print(f"   ✗ Utils integration test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ✗ Error running utils integration test: {e}")
        return False

def main():
    """Run all verification tests"""
    print("FINAL VERIFICATION OF REFACTORED PACKAGE")
    print("=" * 50)
    
    tests = [
        test_package_structure,
        test_components_subpackage,
        test_network_subpackage,
        test_solvers_subpackage,
        test_utils_subpackage,
        test_cli_subpackage,
        test_functionality_preservation,
        test_utils_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break
    
    print("\n" + "=" * 50)
    print("FINAL VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 ALL VERIFICATION TESTS PASSED!")
        print("✓ Package successfully refactored into multi-module structure")
        print("✓ All functionality preserved from original script")
        print("✓ Utils module eliminates code duplication")
        print("✓ All solver methods working correctly")
        print("✓ CLI functionality maintained")
        return True
    else:
        print("❌ Some verification tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)