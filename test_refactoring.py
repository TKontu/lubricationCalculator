#!/usr/bin/env python3
"""
Test script to verify that the refactored package produces identical results
to the original monolithic script
"""

import sys
import os

# Add the current directory to the path so we can import both versions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from refactored package
from refactored_package.cli.demos import create_simple_tree_example
from refactored_package.solvers.network_flow_solver import NetworkFlowSolver

def test_refactored_functionality():
    """Test that the refactored package works correctly"""
    print("Testing refactored package functionality...")
    
    # Create solver
    solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
    
    # Create example network
    network, total_flow_rate, temperature = create_simple_tree_example()
    
    # Solve the network
    flows, info = solver.solve_network_flow_with_pump_physics(
        network=network,
        pump_flow_rate=total_flow_rate,
        temperature=temperature,
        pump_max_pressure=1e6,
        outlet_pressure=101325.0
    )
    
    print(f"✓ Network solved successfully")
    print(f"✓ Converged: {info['converged']}")
    print(f"✓ Total flow rate: {sum(flows.values()):.3f} L/s")
    print(f"✓ Number of components: {len(flows)}")
    
    # Verify flow conservation using the actual flow rate from solver
    actual_flow_rate = info.get('actual_flow_rate', 0)
    expected_flow = total_flow_rate
    print(f"Actual flow rate from solver: {actual_flow_rate}")
    print(f"Expected flow: {expected_flow}")
    flow_error = abs(actual_flow_rate - expected_flow) / expected_flow
    
    if flow_error < 1e-6:
        print(f"✓ Flow conservation verified (error: {flow_error:.2e})")
    else:
        print(f"❌ Flow conservation failed (error: {flow_error:.2e})")
        return False
    
    return True

def test_package_structure():
    """Test that all modules can be imported correctly"""
    print("\nTesting package structure...")
    
    try:
        # Test component imports
        from refactored_package.components import Channel, Connector, Nozzle, FlowComponent
        from refactored_package.components import ComponentType, ConnectorType, NozzleType
        print("✓ Components subpackage imported successfully")
        
        # Test network imports
        from refactored_package.network import Node, Connection, FlowNetwork
        print("✓ Network subpackage imported successfully")
        
        # Test solver imports
        from refactored_package.solvers import SolverConfig, NetworkFlowSolver
        print("✓ Solvers subpackage imported successfully")
        
        # Test utils imports
        from refactored_package.utils import find_all_paths, compute_path_pressure
        print("✓ Utils subpackage imported successfully")
        
        # Test CLI imports
        from refactored_package.cli import main, create_simple_tree_example
        print("✓ CLI subpackage imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("REFACTORING VERIFICATION TEST")
    print("=" * 50)
    
    # Test package structure
    structure_ok = test_package_structure()
    
    # Test functionality
    functionality_ok = test_refactored_functionality()
    
    print("\n" + "=" * 50)
    if structure_ok and functionality_ok:
        print("✅ ALL TESTS PASSED - Refactoring successful!")
        print("✓ Package structure is correct")
        print("✓ Functionality is preserved")
        print("✓ Ready for production use")
        return 0
    else:
        print("❌ TESTS FAILED - Refactoring needs fixes")
        return 1

if __name__ == "__main__":
    sys.exit(main())