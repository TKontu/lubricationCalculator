#!/usr/bin/env python3
"""
Test runner for hydraulics unit tests

This script provides an easy way to run different categories of hydraulics tests
with various output options.
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle the output"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run hydraulics unit tests")
    parser.add_argument(
        '--test-type', 
        choices=['all', 'darcy', 'orifice', 'minor', 'integration'],
        default='all',
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet output'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Run verification script to show calculation details'
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    base_cmd = ['python', '-m', 'pytest', 'tests/test_hydraulics.py']
    
    # Add verbosity flags
    if args.verbose:
        base_cmd.append('-v')
    elif args.quiet:
        base_cmd.append('-q')
    else:
        base_cmd.extend(['-v', '--tb=short'])
    
    # Test type selection
    test_commands = {
        'all': (base_cmd, "All hydraulics tests"),
        'darcy': (base_cmd + ['-k', 'TestChannelDarcyWeisbach'], "Darcy-Weisbach pipe flow tests"),
        'orifice': (base_cmd + ['-k', 'TestNozzleOrifice'], "Orifice/Nozzle flow tests"),
        'minor': (base_cmd + ['-k', 'TestConnectorMinorLosses'], "Minor losses tests"),
        'integration': (base_cmd + ['-k', 'TestIntegrationAndEdgeCases'], "Integration and edge case tests")
    }
    
    success = True
    
    # Run verification script if requested
    if args.verify:
        verify_cmd = ['python', 'test_verification.py']
        success &= run_command(verify_cmd, "Verification script")
    
    # Run selected tests
    if args.test_type in test_commands:
        cmd, description = test_commands[args.test_type]
        success &= run_command(cmd, description)
    else:
        print(f"Unknown test type: {args.test_type}")
        success = False
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("✓ All requested tests completed successfully!")
    else:
        print("✗ Some tests failed. Check output above for details.")
    print('='*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())