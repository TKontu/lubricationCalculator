# Lubrication Flow Calculator

A comprehensive tool for analyzing lubrication flow distribution in complex hydraulic networks. This calculator supports both traditional iterative solving and modern nodal matrix methods for accurate flow and pressure analysis.

## Features

### 🔧 Network Configuration
- **JSON/XML Configuration Files**: Define networks in human-readable formats
- **Template Generation**: Quick start with predefined network templates
- **Programmatic Builder**: Create networks using a fluent API
- **Validation**: Comprehensive network topology validation

### 🧮 Advanced Solvers
- **Network Flow Solver**: Iterative solver with hydraulic physics
- **Nodal Matrix Solver**: Scalable linear algebra approach
- **Multiple Oil Types**: Support for SAE10, SAE20, SAE30, SAE40, SAE50, SAE60
- **Temperature Effects**: Viscosity calculations based on temperature

### 📊 Comprehensive Analysis
- **Flow Distribution**: Natural flow distribution based on resistance
- **Pressure Analysis**: Detailed pressure calculations at all nodes
- **System Adequacy**: Automatic assessment of system performance
- **Recommendations**: Intelligent suggestions for system improvements

### 🖥️ Multiple Interfaces
- **Command Line Interface**: Full-featured CLI for automation
- **Demonstration Mode**: Educational examples and comparisons
- **Configuration Management**: Load, save, and validate network configurations

## Quick Start

### Installation
```bash
git clone <repository-url>
cd lubricationCalculator
pip install -r requirements.txt
```

### Create Your First Network
```bash
# Generate a template network
python main.py network template -o my_network.json

# Simulate the network
python main.py network simulate my_network.json

# View detailed help
python main.py network --help
```

### Example Network Creation
```python
from lubrication_flow_package.utils.network_builder import NetworkBuilder

# Create a simple tree network
network = (NetworkBuilder("My Lubrication System")
           .add_inlet("pump", "Main Pump", 0.0)
           .add_junction("manifold", "Distribution Manifold", 1.0)
           .add_outlet("bearing1", "Bearing 1", 2.0)
           .add_outlet("bearing2", "Bearing 2", 1.5)
           .add_channel("main_line", 0.08, 10.0, "Main Supply Line")
           .add_channel("branch1", 0.05, 8.0, "Branch 1")
           .add_channel("branch2", 0.04, 6.0, "Branch 2")
           .connect("pump", "manifold", "main_line")
           .connect("manifold", "bearing1", "branch1")
           .connect("manifold", "bearing2", "branch2")
           .set_simulation_params(0.015, 40.0, 200000.0))

network.save_json("my_system.json")
```

## Command Line Usage

### Network Operations
```bash
# Create templates
python main.py network template -o network.json          # JSON format
python main.py network template -o network.xml -f xml    # XML format

# Simulate networks
python main.py network simulate network.json             # Basic simulation
python main.py network simulate network.json --solver nodal  # Use nodal solver
python main.py network simulate network.json --output results.json  # Save results

# Validate configurations
python main.py network validate network.json
```

### Demonstration Mode
```bash
# Run educational demonstrations
python main.py demo
```

## Configuration File Format

### JSON Example
```json
{
  "network_name": "Industrial Lubrication System",
  "description": "Multi-zone industrial lubrication network",
  "nodes": [
    {
      "id": "inlet",
      "name": "Main Inlet",
      "elevation": 0.0,
      "type": "inlet"
    }
  ],
  "components": [
    {
      "id": "main_channel",
      "name": "Main Channel",
      "type": "channel",
      "diameter": 0.08,
      "length": 10.0
    }
  ],
  "connections": [
    {
      "from_node": "inlet",
      "to_node": "junction",
      "component": "main_channel"
    }
  ],
  "simulation": {
    "flow_parameters": {
      "total_flow_rate": 0.015,
      "temperature": 40.0,
      "inlet_pressure": 200000.0
    },
    "fluid_properties": {
      "oil_density": 900.0,
      "oil_type": "SAE30"
    }
  }
}
```

## Supported Components

### Channels
- **Diameter**: Pipe internal diameter (m)
- **Length**: Pipe length (m)
- **Pressure Drop**: Calculated using Darcy-Weisbach equation

### Nozzles
- **Types**: sharp_edged, rounded, venturi, flow_nozzle
- **Diameter**: Nozzle diameter (m)
- **Pressure Drop**: Based on nozzle type and flow coefficient

### Connectors
- **Types**: t_junction, x_junction, elbow_90, gate_valve, ball_valve, etc.
- **Diameter**: Connection diameter (m)
- **Loss Coefficient**: Based on connector type

## Example Networks

The `examples/` directory contains several ready-to-use networks:

- **simple_tree.json**: Basic two-outlet tree network
- **complex_network.json**: Multi-level branching system
- **industrial_system.json**: High-flow industrial application
- **precision_system.json**: High-pressure precision machining
- **simple_network.xml**: XML format example

## Analysis Output

The calculator provides comprehensive analysis including:

### Flow Distribution
- Flow rates through each component
- Natural distribution based on hydraulic resistance
- Mass conservation verification

### Pressure Analysis
- Pressure at each network node
- Pressure drops across components
- Elevation effects included

### System Adequacy Assessment
- Outlet pressure validation
- Pump capacity requirements
- System balance analysis
- Improvement recommendations

### Example Output
```
======================================================================
NETWORK FLOW DISTRIBUTION RESULTS
======================================================================
Network: Industrial Lubrication System
Temperature: 40.0°C
Oil Type: SAE30
Total Flow Rate: 15.0 L/s
Inlet Pressure: 200.0 kPa

Component            Type         Flow Rate    Pressure Drop
Name                              (L/s)        (kPa)
-----------------------------------------------------------------
Main Channel         channel      15.000       15.4
Branch 1             channel      9.061        48.4
Branch 2             channel      5.939        58.0

Node                 Pressure (kPa)  Elevation (m)
---------------------------------------------
Inlet                200.0           0.0
Junction             175.8           1.0
Outlet 1             118.5           2.0
Outlet 2             113.4           1.5

🔍 SYSTEM ANALYSIS:
   System adequate: ✅ YES
```

## Technical Details

### Solver Methods
1. **Network Flow Solver**: Iterative method with proper hydraulic physics
2. **Nodal Matrix Solver**: Linear algebra approach for large networks

### Fluid Properties
- **Viscosity**: Temperature-dependent using Vogel equation
- **Density**: User-configurable oil density
- **Oil Types**: Standard SAE viscosity grades

### Validation
- **Network Topology**: Connectivity and reachability checks
- **Component Parameters**: Physical parameter validation
- **Flow Conservation**: Mass balance verification

## Documentation

- **[Network Configuration Guide](NETWORK_CONFIG_GUIDE.md)**: Detailed configuration documentation
- **[Implementation Summary](NETWORK_IMPLEMENTATION_SUMMARY.md)**: Technical implementation details
- **[Nodal Solver README](NODAL_MATRIX_SOLVER_README.md)**: Nodal matrix solver documentation

## Requirements

- Python 3.8+
- NumPy >= 1.20.0
- SciPy >= 1.7.0

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

For questions, issues, or feature requests, please [create an issue](link-to-issues) or contact the development team.