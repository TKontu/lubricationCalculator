"""
CLI commands for network configuration and simulation
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from ..config.network_config import NetworkConfigLoader, NetworkConfigSaver
from ..config.simulation_config import SimulationConfig
from ..solvers.network_flow_solver import NetworkFlowSolver
from ..solvers.nodal_matrix_solver import NodalMatrixSolver


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
        print(f"✅ JSON template created: {output_file}")
    elif format_type.lower() == 'xml':
        NetworkConfigSaver.save_xml(config, output_file)
        print(f"✅ XML template created: {output_file}")
    else:
        raise ValueError(f"Unsupported format: {format_type}")


def simulate_network(config_file: str, solver_type: str = 'network', output_file: Optional[str] = None):
    """Simulate a network from configuration file"""
    
    # Determine file format
    file_path = Path(config_file)
    if not file_path.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    # Load configuration
    try:
        if file_path.suffix.lower() == '.json':
            config = NetworkConfigLoader.load_json(config_file)
        elif file_path.suffix.lower() == '.xml':
            config = NetworkConfigLoader.load_xml(config_file)
        else:
            print(f"❌ Unsupported file format: {file_path.suffix}")
            return False
        
        print(f"📁 Loaded configuration: {config.network_name}")
        print(f"📝 Description: {config.description}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # Build network and simulation config
    try:
        network, sim_config = NetworkConfigLoader.build_network(config)
        print(f"🔧 Built network with {len(network.nodes)} nodes and {len(network.connections)} connections")
        
    except Exception as e:
        print(f"❌ Error building network: {e}")
        return False
    
    # Validate network
    is_valid, errors = network.validate_network()
    if not is_valid:
        print("❌ Network validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    print("✅ Network validation passed")
    
    # Print network info
    network.print_network_info()
    
    # Create solver
    try:
        if solver_type.lower() == 'network':
            solver = NetworkFlowSolver(oil_density=sim_config.oil_density, oil_type=sim_config.oil_type)
            connection_flows, solution_info = solver.solve_network_flow(
                network, 
                sim_config.total_flow_rate, 
                sim_config.temperature,
                inlet_pressure=sim_config.inlet_pressure
            )
        elif solver_type.lower() == 'nodal':
            solver = NodalMatrixSolver(
                oil_density=sim_config.oil_density,
                oil_type=sim_config.oil_type
            )
            connection_flows, solution_info = (
                solver.solve_nodal_network_with_pump_physics(
                    network,
                    pump_flow_rate=sim_config.total_flow_rate,
                    temperature=sim_config.temperature,
                    pump_max_pressure=sim_config.inlet_pressure,
                    outlet_pressure=sim_config.outlet_pressure or 101325.0,
                    max_iterations=sim_config.solver_settings.max_iterations,
                    tolerance=sim_config.solver_settings.tolerance
                )
            )
        else:
            print(f"❌ Unknown solver type: {solver_type}")
            return False
        
        print(f"🔬 Simulation completed using {solver_type} solver")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        return False
    
    # Print results (use NetworkFlowSolver for printing regardless of solver type)
    if solver_type.lower() == 'nodal':
        # Create a NetworkFlowSolver instance just for printing
        print_solver = NetworkFlowSolver(oil_density=sim_config.oil_density, oil_type=sim_config.oil_type)
        print_solver.print_results(network, connection_flows, solution_info)
    else:
        solver.print_results(network, connection_flows, solution_info)
    
    # Analyze system adequacy
    if hasattr(solver, 'analyze_system_adequacy'):
        analysis = solver.analyze_system_adequacy(network, connection_flows, solution_info)
        print(f"\n🔍 SYSTEM ANALYSIS:")
        print(f"   System adequate: {'✅ YES' if analysis['adequate'] else '❌ NO'}")
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
                'connection_flows': {comp_id: flow for comp_id, flow in connection_flows.items()},
                'solution_info': solution_info,
                'analysis': analysis if 'analysis' in locals() else None
            }
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save results: {e}")
    
    return True


def validate_network_config(config_file: str):
    """Validate a network configuration file"""
    
    file_path = Path(config_file)
    if not file_path.exists():
        print(f"❌ Configuration file not found: {config_file}")
        return False
    
    try:
        # Load configuration
        if file_path.suffix.lower() == '.json':
            config = NetworkConfigLoader.load_json(config_file)
        elif file_path.suffix.lower() == '.xml':
            config = NetworkConfigLoader.load_xml(config_file)
        else:
            print(f"❌ Unsupported file format: {file_path.suffix}")
            return False
        
        print(f"📁 Validating configuration: {config.network_name}")
        
        # Build network
        network, sim_config = NetworkConfigLoader.build_network(config)
        
        # Validate network topology
        is_valid, errors = network.validate_network()
        
        if is_valid:
            print("✅ Network configuration is valid")
            print(f"   - {len(network.nodes)} nodes")
            print(f"   - {len(network.connections)} connections")
            print(f"   - {len(network.outlet_nodes)} outlets")
            
            # Check for potential issues
            paths = network.get_paths_to_outlets()
            print(f"   - {len(paths)} paths to outlets")
            
            return True
        else:
            print("❌ Network validation failed:")
            for error in errors:
                print(f"   - {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error validating configuration: {e}")
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
    simulate_parser.add_argument('--solver', choices=['network', 'nodal'], default='network',
                                help='Solver type to use (default: network)')
    simulate_parser.add_argument('--output', help='Save results to file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a network configuration file')
    validate_parser.add_argument('config_file', help='Network configuration file to validate')
    
    args = parser.parse_args()
    
    if args.command == 'template':
        create_network_template(args.output, args.format)
    elif args.command == 'simulate':
        success = simulate_network(args.config_file, args.solver, args.output)
        sys.exit(0 if success else 1)
    elif args.command == 'validate':
        success = validate_network_config(args.config_file)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()