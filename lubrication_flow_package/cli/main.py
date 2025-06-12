"""
Main CLI entry point for the lubrication flow calculator
"""

import argparse
import sys
from .demos import main as run_demos
from .network_cli import main as run_network_cli


def main():
    """Main CLI entry point with subcommands"""
    parser = argparse.ArgumentParser(
        description="Lubrication Flow Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  demo     - Run demonstration examples
  network  - Network configuration and simulation tools
  
Examples:
  # Run demonstrations
  python -m lubrication_flow_package.cli demo
  
  # Create network template
  python -m lubrication_flow_package.cli network template -o example.json
  
  # Simulate network
  python -m lubrication_flow_package.cli network simulate example.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run demonstration examples')
    
    # Network command
    network_parser = subparsers.add_parser('network', help='Network configuration and simulation')
    
    # Parse only the first argument to determine which subcommand to use
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            run_demos()
        elif sys.argv[1] == 'network':
            # Remove 'network' from argv and call network CLI
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            run_network_cli()
        else:
            parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()