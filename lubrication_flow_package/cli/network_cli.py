"""
CLI commands for network configuration and simulation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..config.network_config import NetworkConfigLoader, NetworkConfigSaver
from ..config.simulation_config import SimulationConfig
from ..solvers.nodal_matrix_solver import NodalMatrixSolver
from ..solvers.nonlinear_solver import RobustNonLinearSolver
from ...create_example_networks import create_simple_tree_network, create_complex_network_with_tee



def create_network_template(output_file: str, format_type: str = 'json'):
    """Create a template network configuration file"""
    
    template_config = {
        "network_name": "Example Network",
        "description": "A simple example network with two outlets",
        "nodes": [
            {
                "id": "inlet",
                "name": "Inlet",
                "elevation": 0.0,
                "type": "inlet",
                "x": 0.0,
                "y": 0.0
            },
            {
                "id": "junction1",
                "name": "Junction 1",
                "elevation": 1.0,
                "type": "junction",
                "x": 10.0,
                "y": 0.0
            },
            {
                "id": "outlet1",
                "name": "Outlet 1",
                "elevation": 2.0,
                "type": "outlet",
                "x": 20.0,
                "y": 5.0
            },
            {
                "id": "outlet2",
                "name": "Outlet 2",
                "elevation": 1.5,
                "type": "outlet",
                "x": 20.0,
                "y": -5.0
            }
        ],
        "components": [
            {
                "id": "main_channel",
                "name": "Main Channel",
                "type": "channel",
                "diameter": 0.08,
                "length": 10.0
            },
            {
                "id": "branch1_channel",
                "name": "Branch 1 Channel",
                "type": "channel",
                "diameter": 0.05,
                "length": 8.0
            },
            {
                "id": "branch2_channel",
                "name": "Branch 2 Channel",
                "type": "channel",
                "diameter": 0.04,
                "length": 6.0
            },
            {
                "id": "nozzle1",
                "name": "Nozzle 1",
                "type": "nozzle",
                "diameter": 0.025,
                "nozzle_type": "venturi"
            },
            {
                "id": "nozzle2",
                "name": "Nozzle 2",
                "type": "nozzle",
                "diameter": 0.020,
                "nozzle_type": "sharp_edged"
            }
        ],
        "connections": [
            {
                "from_node": "inlet",
                "to_node": "junction1",
                "component": "main_channel"
            },
            {
                "from_node": "junction1",
                "to_node": "outlet1",
                "component": "branch1_channel"
            },
            {
                "from_node": "junction1",
                "to_node": "outlet2",
                "component": "branch2_channel"
            }
        ],
        "simulation": {
            "flow_parameters": {
                "total_flow_rate": 0.015,
                "temperature": 40.0,
                "inlet_pressure": 200000.0,
                "outlet_pressure": None
            },
            "fluid_properties": {
                "oil_density": 900.0,
                "oil_type": "SAE30"
            },
            "solver_settings": {
                "max_iterations": 100,
                "tolerance": 1e-6,
                "relaxation_factor": 0.8
            },
            "output_settings": {
                "output_units": "metric",
                "detailed_output": True,
                "save_results": False,
                "results_file": None
            }
        },
        "metadata": {
            "created_by": "network_cli",
            "version": "1.0",
            "notes": "Template network for demonstration"
        }
    }
    
    from ..config.network_config import NetworkConfig
    config = NetworkConfig(
        network_name=template_config["network_name"],
        description=template_config["description"],
        nodes=template_config["nodes"],
        components=template_config["components"],
        connections=template_config["connections"],
        simulation=template_config["simulation"],
        metadata=template_config["metadata"]
    )
    
    if format_type.lower() == 'json':
        NetworkConfigSaver.save_json(config, output_file)
        print(f"JSON template created: {output_file}")
    elif format_type.lower() == 'xml':
        NetworkConfigSaver.save_xml(config, output_file)
        print(f"XML template created: {output_file}")
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def simulate_network(config_file: str, output_file: Optional[str] = None, solver_type: str = 'nodal', solver_config_file: Optional[str] = None, verbose: bool = False):
    """Simulate a network from configuration file"""
    
    # Determine file format
    file_path = Path(config_file)
    if not file_path.exists():
        print(f"Configuration file not found: {config_file}")
        return False
    
    # Load configuration
    try:
        if file_path.suffix.lower() == '.json':
            config = NetworkConfigLoader.load_json(config_file)
        elif file_path.suffix.lower() == '.xml':
            config = NetworkConfigLoader.load_xml(config_file)
        else:
            print(f"Unsupported file format: {file_path.suffix}")
            return False
        
        if verbose:
            print(f"Loaded configuration: {config.network_name}")
            print(f"Description: {config.description}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return False
    
    # Build network and simulation config
    try:
        network, sim_config = NetworkConfigLoader.build_network(config)
        if verbose:
            print(f"Built network with {len(network.nodes)} nodes and {len(network.connections)} connections")
        
    except Exception as e:
        print(f"Error building network: {e}")
        return False
    
    # Validate network
    is_valid, errors = network.validate_network()
    if not is_valid:
        print("Network validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if verbose:
        print("Network validation passed")
    
    # Print network info
    if verbose:
        network.print_network_info()
    
    # Create and run solver
    try:
        # Solver Factory
        solver_map = {
            'nodal': NodalMatrixSolver,
            'robust_newton': RobustNonLinearSolver
        }
        solver_class = solver_map.get(solver_type)
        if not solver_class:
            print(f"Unknown solver type: {solver_type}")
            return False

        # Load optional solver config
        solver_config = None
        if solver_config_file:
            from ..solvers.config import SolverConfig
            solver_config = SolverConfig.from_yaml(solver_config_file)

        # Instantiate and run the solver using the unified interface
        solver = solver_class(sim_config, solver_config)
        solution = solver.solve(network)
        
        print(f"Simulation completed with {solver_type} solver.")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        return False
    
    # Print results using the solver's own print method
    solver.print_results(network, solution,
                         pressure_unit=sim_config.output_pressure_unit,
                         flow_rate_unit=sim_config.output_flow_rate_unit)
    
    # Analyze system adequacy (if the method exists)
    if hasattr(solver, 'analyze_system_adequacy'):
        analysis = solver.analyze_system_adequacy(network, solution.get('component_flows', {}), solution)
        print(f"\n🔍 SYSTEM ANALYSIS:")
        print(f"   System adequate: {'YES' if analysis['adequate'] else 'NO'}")
        if analysis['issues']:
            print("   Issues found:")
            for issue in analysis['issues']:
                print(f"   - {issue}")
        if analysis.get('recommendations'):
            print("   Recommendations:")
            for rec in analysis['recommendations']:
                print(f"   - {rec}")
    
    # Save results if requested
    if output_file:
        try:
            import json
            results = {
                'network_name': network.name,
                'simulation_parameters': sim_config.to_dict(),
                'solution': solution,
                'analysis': analysis if 'analysis' in locals() else None
            }
            
            # Remove non-serializable items from solution if they exist
            if 'fluid_properties' in results['solution']:
                del results['solution']['fluid_properties']

            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results saved to: {output_file}")
            
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
    
    return True


def validate_network_config(config_file: str):
    """Validate a network configuration file"""
    
    file_path = Path(config_file)
    if not file_path.exists():
        print(f"Configuration file not found: {config_file}")
        return False
    
    try:
        # Load configuration
        if file_path.suffix.lower() == '.json':
            config = NetworkConfigLoader.load_json(config_file)
        elif file_path.suffix.lower() == '.xml':
            config = NetworkConfigLoader.load_xml(config_file)
        else:
            print(f"Unsupported file format: {file_path.suffix}")
            return False
        
        print(f"📁 Validating configuration: {config.network_name}")
        
        # Build network
        network, sim_config = NetworkConfigLoader.build_network(config)
        
        # Validate network topology
        is_valid, errors = network.validate_network()
        
        if is_valid:
            print("Network configuration is valid")
            print(f"   - {len(network.nodes)} nodes")
            print(f"   - {len(network.connections)} connections")
            print(f"   - {len(network.outlet_nodes)} outlets")
            
            # Check for potential issues
            paths = network.get_paths_to_outlets()
            print(f"   - {len(paths)} paths to outlets")
            
            return True
        else:
            print("Network validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
            
    except Exception as e:
        print(f"Error validating configuration: {e}")
        return False


def main():
    """Main CLI entry point for network operations"""
    parser = argparse.ArgumentParser(
        description="Lubrication Network Flow Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a template network configuration
  python -m lubrication_flow_package.cli.network_cli template -o example.json
  
  # Simulate a network
  python -m lubrication_flow_package.cli.network_cli simulate example.json
  
  # Simulate with nodal solver and save results
  python -m lubrication_flow_package.cli.network_cli simulate example.json --solver nodal --output results.json
  
  # Validate a configuration file
  python -m lubrication_flow_package.cli.network_cli validate example.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create a template network configuration')
    template_parser.add_argument('-o', '--output', required=True, help='Output file path')
    template_parser.add_argument('-f', '--format', choices=['json', 'xml'], default='json', 
                                help='Output format (default: json)')
    
    # Simulate command
    simulate_parser = subparsers.add_parser('simulate', help='Simulate a network from configuration file')
    simulate_parser.add_argument('config_file', help='Network configuration file')
    simulate_parser.add_argument('--solver', default='nodal', choices=['nodal', 'robust_newton'],
                                help='Solver type to use (default: nodal)')
    simulate_parser.add_argument('--output', help='Save results to file')
    simulate_parser.add_argument('--solver-config', help='Path to solver configuration file')
    simulate_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a network configuration file')
    validate_parser.add_argument('config_file', help='Network configuration file to validate')

    # Create and Simulate command
    create_and_simulate_parser = subparsers.add_parser('create-and-simulate', help='Create and simulate a network on the fly')
    create_and_simulate_parser.add_argument('network_type', choices=['simple', 'complex'], help='Type of network to create')
    create_and_simulate_parser.add_argument('--solver', default='nodal', choices=['nodal', 'robust_newton'], help='Solver type to use')
    
    args = parser.parse_args()
    
    if args.command == 'template':
        create_network_template(args.output, args.format)
    elif args.command == 'simulate':
        success = simulate_network(args.config_file, args.output, args.solver, args.solver_config, args.verbose)
        sys.exit(0 if success else 1)
    elif args.command == 'create-and-simulate':
        if args.network_type == 'simple':
            network = create_simple_tree_network()
        else:
            network = create_complex_network_with_tee()
        
        sim_config = SimulationConfig(
            total_flow_rate=0.02,
            temperature=50.0,
            inlet_pressure=250000.0
        )
        
        solver_map = {
            'nodal': NodalMatrixSolver,
            'robust_newton': RobustNonLinearSolver
        }
        solver_class = solver_map.get(args.solver)
        solver = solver_class(sim_config)
        solution = solver.solve(network)
        solver.print_results(network, solution)

    elif args.command == 'validate':
        success = validate_network_config(args.config_file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()