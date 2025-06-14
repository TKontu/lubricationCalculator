#!/usr/bin/env python3
"""
Comprehensive test runner for solver integration tests.

This script runs all solver comparison tests and provides detailed output
about solver performance and agreement.
"""

import subprocess
import sys
from pathlib import Path

def run_solver_tests():
    """Run all solver integration tests with detailed output"""
    
    print("=" * 80)
    print("SOLVER INTEGRATION TEST SUITE")
    print("=" * 80)
    print()
    print("Testing NetworkFlowSolver vs NodalMatrixSolver on various network topologies:")
    print("- Single-pipe cases: one inlet → one pipe → outlet")
    print("- Parallel pipes: two identical pipes in parallel")
    print("- Asymmetric parallel: two different-diameter pipes")
    print("- T-junction loop: branching network with merge")
    print("- Robustness tests: small and high flow rates")
    print()
    print("Tolerance requirements:")
    print("- Flow rate tolerance: ±1 L/min")
    print("- Pressure tolerance: ±0.2 bar")
    print()
    
    # Run tests with verbose output
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_solvers.py", 
        "-v", "-s", "--tb=short"
    ]
    
    print("Running tests...")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("-" * 40)
        
        if result.returncode == 0:
            print("✓ ALL SOLVER INTEGRATION TESTS PASSED")
            print()
            print("Key findings:")
            print("- Both solvers handle single-pipe cases identically")
            print("- Both solvers handle identical parallel pipes correctly")
            print("- Solvers may disagree on complex networks (asymmetric parallel, T-junction)")
            print("- Both solvers are robust to small and high flow rates")
            print("- All mass conservation laws are satisfied")
            
        else:
            print("✗ SOME TESTS FAILED")
            print(f"Exit code: {result.returncode}")
            
        return result.returncode
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

def run_specific_test_category(category):
    """Run a specific category of tests"""
    
    categories = {
        "single": "TestSinglePipeCase",
        "parallel": "TestParallelPipes", 
        "asymmetric": "TestAsymmetricParallel",
        "tjunction": "TestTJunctionLoop",
        "robustness": "TestSolverRobustness"
    }
    
    if category not in categories:
        print(f"Unknown category: {category}")
        print(f"Available categories: {list(categories.keys())}")
        return 1
    
    test_class = categories[category]
    
    print(f"Running {category} tests ({test_class})...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        f"tests/test_solvers.py::{test_class}", 
        "-v", "-s"
    ]
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode

if __name__ == "__main__":
    if len(sys.argv) > 1:
        category = sys.argv[1]
        exit_code = run_specific_test_category(category)
    else:
        exit_code = run_solver_tests()
    
    sys.exit(exit_code)