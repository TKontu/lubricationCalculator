# Network Configuration Guide

This guide explains how to create and simulate lubrication networks using configuration files.

## Overview

The lubrication calculator now supports network definition through configuration files in JSON or XML format. This allows you to:

- Define complex network topologies easily
- Store and version control network designs
- Simulate networks with different parameters
- Share network configurations between team members

## Quick Start

### 1. Create a Template Network

```bash
# Create a JSON template
python main.py network template -o my_network.json

# Create an XML template
python main.py network template -o my_network.xml -f xml
```

### 2. Simulate a Network

```bash
# Simulate using the network solver
python main.py network simulate my_network.json

# Simulate using the nodal matrix solver
python main.py network simulate my_network.json --solver nodal

# Save results to a file
python main.py network simulate my_network.json --output results.json
```

### 3. Validate a Configuration

```bash
python main.py network validate my_network.json
```

## Configuration File Format

### JSON Format

```json
{
  "network_name": "My Network",
  "description": "Description of the network",
  "nodes": [
    {
      "id": "inlet",
      "name": "Main Inlet",
      "elevation": 0.0,
      "type": "inlet",
      "x": 0.0,
      "y": 0.0
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
      "to_node": "junction1",
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

### XML Format

```xml
<?xml version='1.0' encoding='utf-8'?>
<network name="My Network" description="Description of the network">
  <nodes>
    <node id="inlet" name="Main Inlet" elevation="0.0" type="inlet" x="0.0" y="0.0" />
  </nodes>
  <components>
    <component id="main_channel" name="Main Channel" type="channel">
      <property name="diameter">0.08</property>
      <property name="length">10.0</property>
    </component>
  </components>
  <connections>
    <connection from_node="inlet" to_node="junction1" component="main_channel" />
  </connections>
  <simulation>
    <parameter name="total_flow_rate">0.015</parameter>
    <parameter name="temperature">40.0</parameter>
    <parameter name="inlet_pressure">200000.0</parameter>
  </simulation>
</network>
```

## Configuration Elements

### Nodes

Nodes represent connection points in the network:

- **id**: Unique identifier for the node
- **name**: Human-readable name
- **elevation**: Height in meters (affects pressure calculations)
- **type**: One of "inlet", "outlet", or "junction"
- **x, y**: Position coordinates (for visualization, optional)

### Components

Components represent physical elements that carry or control flow:

#### Channels
```json
{
  "id": "channel1",
  "name": "Main Channel",
  "type": "channel",
  "diameter": 0.08,
  "length": 10.0
}
```

#### Nozzles
```json
{
  "id": "nozzle1",
  "name": "Outlet Nozzle",
  "type": "nozzle",
  "diameter": 0.025,
  "nozzle_type": "venturi"
}
```

Nozzle types: `sharp_edged`, `rounded`, `venturi`, `flow_nozzle`

#### Connectors
```json
{
  "id": "connector1",
  "name": "T-Junction",
  "type": "connector",
  "diameter": 0.06,
  "connector_type": "t_junction"
}
```

Connector types: `t_junction`, `x_junction`, `elbow_90`, `gate_valve`, etc.

### Connections

Connections link nodes through components:

```json
{
  "from_node": "node1_id",
  "to_node": "node2_id",
  "component": "component_id"
}
```

### Simulation Parameters

#### Flow Parameters
- **total_flow_rate**: Total flow rate in m³/s
- **temperature**: Fluid temperature in °C
- **inlet_pressure**: Inlet pressure in Pa
- **outlet_pressure**: Outlet pressure in Pa (optional)

#### Fluid Properties
- **oil_density**: Oil density in kg/m³
- **oil_type**: Oil viscosity grade (SAE10, SAE20, SAE30, SAE40, SAE50, SAE60)

#### Solver Settings
- **max_iterations**: Maximum solver iterations
- **tolerance**: Convergence tolerance
- **relaxation_factor**: Solver relaxation factor

## Examples

The `examples/` directory contains several example networks:

- `simple_tree.json`: Basic tree network with two outlets
- `complex_network.json`: Multi-level branching network
- `simple_network.xml`: Simple network in XML format

## Command Line Interface

### Main Commands

```bash
# Show help
python main.py network --help

# Create template
python main.py network template --help

# Simulate network
python main.py network simulate --help

# Validate configuration
python main.py network validate --help
```

### Template Creation

```bash
# Create JSON template
python main.py network template -o template.json

# Create XML template
python main.py network template -o template.xml -f xml
```

### Network Simulation

```bash
# Basic simulation
python main.py network simulate network.json

# Use nodal solver
python main.py network simulate network.json --solver nodal

# Save results
python main.py network simulate network.json --output results.json
```

### Configuration Validation

```bash
# Validate a configuration file
python main.py network validate network.json
```

## Tips for Creating Networks

1. **Start Simple**: Begin with a template and modify it incrementally
2. **Validate Early**: Use the validate command to catch errors early
3. **Use Meaningful Names**: Give nodes and components descriptive names
4. **Check Units**: All dimensions are in meters, pressures in Pa, flow rates in m³/s
5. **Consider Elevations**: Elevation differences affect pressure calculations
6. **Test Different Solvers**: Try both network and nodal solvers to compare results

## Troubleshooting

### Common Issues

1. **Network Validation Fails**
   - Check that all nodes referenced in connections exist
   - Ensure there's exactly one inlet and at least one outlet
   - Verify all outlets are reachable from the inlet

2. **Simulation Fails**
   - Check that all component parameters are positive
   - Ensure flow rates and pressures are reasonable
   - Try adjusting solver tolerance or iterations

3. **Unrealistic Results**
   - Check component dimensions (very small diameters cause high pressure drops)
   - Verify fluid properties match your application
   - Consider if inlet pressure is sufficient for the network

### Getting Help

- Use `--help` with any command for detailed usage information
- Check the examples directory for working configurations
- Validate your configuration before simulating
- Start with the template and modify incrementally