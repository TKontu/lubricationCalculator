# Refactored Network Lubrication Flow Calculator

## Overview

This package has been refactored from a monolithic script (`network_lubrication_flow_tool.py`) into a well-organized multi-module package structure for better maintainability, reusability, and code organization.

## Package Structure

```
refactored_package/
├── __init__.py                 # Main package initialization
├── components/                 # Flow component classes
│   ├── __init__.py
│   ├── base.py                # Base classes and enums
│   ├── channel.py             # Channel component (pipes, tubes)
│   ├── connector.py           # Connector component (junctions, elbows)
│   └── nozzle.py              # Nozzle component (orifices, restrictions)
├── network/                   # Network topology classes
│   ├── __init__.py
│   ├── node.py                # Node class (connection points)
│   ├── connection.py          # Connection class (node-to-node links)
│   └── flow_network.py        # FlowNetwork class (topology management)
├── solvers/                   # Flow distribution solvers
│   ├── __init__.py
│   ├── config.py              # Solver configuration
│   └── network_flow_solver.py # Main network flow solver
├── utils/                     # Utility functions
│   ├── __init__.py
│   └── network_utils.py       # Network analysis utilities
└── cli/                       # Command line interface
    ├── __init__.py
    ├── demos.py               # Demo functions
    └── main.py                # CLI entry point
```

## Key Components

### Components Subpackage (`components/`)

- **FlowComponent**: Base class for all flow components
- **Channel**: Represents pipes, tubes with Darcy-Weisbach pressure drop calculations
- **Connector**: Represents junctions, elbows, reducers with loss coefficients
- **Nozzle**: Represents orifices, restrictions with discharge coefficient calculations
- **Enums**: ComponentType, ConnectorType, NozzleType for type safety

### Network Subpackage (`network/`)

- **Node**: Represents connection points in the network
- **Connection**: Represents links between nodes through components
- **FlowNetwork**: Manages network topology, path finding, and validation

### Solvers Subpackage (`solvers/`)

- **SolverConfig**: Configuration dataclass for solver parameters
- **NetworkFlowSolver**: Main solver for flow distribution with multiple solution approaches

### Utils Subpackage (`utils/`)

- **Path Analysis**: Find all paths, compute path pressures and resistances
- **Flow Distribution**: Calculate conductances, distribute flow by resistance
- **Convergence**: Check convergence, apply damping for stability
- **Network Validation**: Validate flow conservation and network topology

### CLI Subpackage (`cli/`)

- **Demo Functions**: Create example networks and demonstrate solver capabilities
- **Main Entry Point**: Command line interface for running demonstrations

## Usage

### Basic Usage

```python
from refactored_package import *

# Create a solver
solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")

# Create a network
network = FlowNetwork("My Network")

# Add nodes
inlet = network.create_node("Inlet", elevation=0.0)
outlet = network.create_node("Outlet", elevation=1.0)

# Set inlet and outlet
network.set_inlet(inlet)
network.add_outlet(outlet)

# Add components
channel = Channel(diameter=0.05, length=10.0, name="Main Channel")
network.connect_components(inlet, outlet, channel)

# Solve the network
flows, info = solver.solve_network_flow_with_pump_physics(
    network=network,
    pump_flow_rate=0.01,  # 10 L/s
    temperature=40.0,     # 40°C
    pump_max_pressure=1e6,
    outlet_pressure=101325.0
)
```

### Running Demonstrations

```bash
# Run the main demonstration
python main.py

# Or run directly
python -m refactored_package.cli.main
```

### Testing

```bash
# Run the refactoring verification test
python test_refactoring.py
```

## Features

### Hydraulic Modeling

- **Correct Physics**: Flow distribution based on resistance (conductance)
- **Pressure Calculations**: Proper pressure drop calculations for all component types
- **Viscosity Models**: Temperature-dependent viscosity using Vogel equation
- **Multiple Oil Types**: Support for SAE10, SAE20, SAE30, SAE40, SAE50, SAE60

### Network Analysis

- **Path Finding**: Automatic detection of all flow paths from inlet to outlets
- **Topology Validation**: Comprehensive network validation and error reporting
- **Flow Conservation**: Automatic verification of mass conservation at all nodes
- **Pressure Balance**: Proper pressure equalization at junction points

### Solver Approaches

1. **Conductance-Based Distribution**: Flow distributes based on path resistance
2. **Matrix-Based Nodal Solver**: Scalable linear algebra approach
3. **Iterative Refinement**: Convergence checking with adaptive damping

### Component Types

- **Channels**: Pipes, tubes with friction losses (Darcy-Weisbach equation)
- **Connectors**: Junctions, elbows, reducers with minor losses
- **Nozzles**: Orifices, restrictions with discharge coefficients

## Migration from Original Script

The refactored package maintains 100% compatibility with the original functionality:

- All classes and methods have the same interfaces
- All calculation results are identical
- All demonstration functions are preserved
- Original script remains functional alongside the refactored package

## Benefits of Refactoring

1. **Modularity**: Clear separation of concerns across subpackages
2. **Maintainability**: Easier to modify and extend individual components
3. **Reusability**: Components can be imported and used independently
4. **Testability**: Each module can be tested in isolation
5. **Documentation**: Better code organization and documentation
6. **Scalability**: Easier to add new component types and solver approaches

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing (linear algebra)
- `math`: Mathematical functions
- `typing`: Type hints
- `dataclasses`: Configuration classes
- `enum`: Enumeration types
- `uuid`: Unique identifiers
- `collections`: Data structures

## Version Information

- **Version**: 2.0.0
- **Original Script**: `network_lubrication_flow_tool.py` (preserved)
- **Refactored Package**: `refactored_package/`
- **Entry Point**: `main.py`

## Verification

The refactoring has been thoroughly tested:

- ✅ Package structure verification
- ✅ Import functionality testing  
- ✅ Numerical results validation
- ✅ Flow conservation verification
- ✅ Demonstration compatibility

All tests pass, confirming that the refactored package produces identical results to the original monolithic script.