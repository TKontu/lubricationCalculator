# Pump Characteristic Curves Guide

This guide explains how to use the new pump characteristic curve functionality in the lubrication flow calculator.

## Overview

The pump characteristic functionality allows you to model realistic pump behavior using P-Q (Pressure-Flow) curves instead of simple pressure limits. This provides more accurate flow predictions by finding the actual operating point where the pump curve intersects with the system resistance curve.

## Key Features

- **Multiple curve types**: Polynomial, lookup table, and linear curves
- **Manufacturer data support**: Import real pump performance data
- **Automatic operating point calculation**: Find intersection of pump and system curves
- **Backward compatibility**: Falls back to legacy pressure limiting if no curve provided

## Basic Usage

### 1. Creating Pump Characteristics

#### Polynomial Pump Characteristic
```python
from lubrication_flow_package.components.pump import PumpCharacteristic

# Define pump curve as P = a0 + a1*Q + a2*Q^2 + ...
pump = PumpCharacteristic(
    curve_type="polynomial",
    coefficients=[1000000, -5000000, -1000000],  # [a0, a1, a2]
    max_flow=0.5,
    max_pressure=1000000
)
```

#### Table-Based Pump Characteristic
```python
# Use manufacturer performance data
flow_points = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010]  # m³/s
pressure_points = [900000, 880000, 840000, 780000, 700000, 600000]  # Pa

pump = PumpCharacteristic(
    curve_type="table",
    flow_points=flow_points,
    pressure_points=pressure_points
)
```

#### Typical Centrifugal Pump
```python
# Create a typical centrifugal pump curve
pump = PumpCharacteristic.create_typical_centrifugal_pump(
    max_pressure=800000,   # 800 kPa shutoff pressure
    max_flow=0.015,        # 15 L/s max flow
    efficiency_point=(0.6, 0.8)  # Best efficiency at 60% flow, 80% pressure
)
```

### 2. Using with Network Solver

```python
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver

solver = NetworkFlowSolver()

# Solve with pump characteristic
flows, info = solver.solve_network_flow_with_pump_physics(
    network=your_network,
    pump_flow_rate=0.008,  # Initial flow estimate (m³/s)
    temperature=60.0,      # Operating temperature (°C)
    pump_characteristic=pump,  # Your pump characteristic
    outlet_pressure=101325.0   # Outlet pressure (Pa)
)
```

### 3. Analyzing Results

The solution info dictionary now includes additional pump-related information:

```python
print(f"Pump adequate: {info['pump_adequate']}")
print(f"Available pressure: {info['available_pressure']} Pa")
print(f"Operating pressure: {info['operating_pressure']} Pa")
print(f"Actual flow rate: {info['actual_flow_rate']} m³/s")
```

## Advanced Features

### Operating Point Calculation

Find where pump curve intersects with system resistance:

```python
# Estimate system resistance coefficient (Pa·s²/m⁶)
system_resistance = 2e7

# Find operating point
operating_flow, operating_pressure = pump.find_operating_point(
    system_resistance, 
    flow_range=(0.0, 0.02)
)
```

### Curve Visualization

Generate points for plotting pump curves:

```python
import matplotlib.pyplot as plt

# Get curve points
flows, pressures = pump.get_curve_points(num_points=50, flow_range=(0.0, 0.02))

# Plot
plt.plot(flows, pressures/1000, label='Pump Curve')
plt.xlabel('Flow Rate (m³/s)')
plt.ylabel('Pressure (kPa)')
plt.legend()
plt.show()
```

## Pump Types and Applications

### Centrifugal Pumps
- **Characteristics**: Pressure decreases with increasing flow
- **Applications**: General lubrication systems, cooling circuits
- **Curve shape**: Typically quadratic, steep drop at high flows

```python
centrifugal = PumpCharacteristic.create_typical_centrifugal_pump(
    max_pressure=600000,
    max_flow=0.020,
    efficiency_point=(0.7, 0.75)
)
```

### Positive Displacement Pumps
- **Characteristics**: Nearly constant pressure across flow range
- **Applications**: High-pressure lubrication, precision systems
- **Curve shape**: Flat with slight pressure drop

```python
positive_displacement = PumpCharacteristic(
    curve_type="polynomial",
    coefficients=[1000000, -20000000, 0],  # Nearly constant pressure
    max_flow=0.012,
    max_pressure=1000000
)
```

### Variable Displacement Pumps
- **Characteristics**: Adjustable flow at constant pressure
- **Applications**: Load-sensing systems, variable speed drives
- **Implementation**: Use multiple curves for different displacement settings

## Migration from Legacy Code

### Before (Legacy Method)
```python
flows, info = solver.solve_network_flow_with_pump_physics(
    network=network,
    pump_flow_rate=0.008,
    temperature=60.0,
    pump_max_pressure=800000  # Simple pressure limit
)
```

### After (With Pump Characteristic)
```python
pump_char = PumpCharacteristic.create_typical_centrifugal_pump(
    max_pressure=800000,
    max_flow=0.015
)

flows, info = solver.solve_network_flow_with_pump_physics(
    network=network,
    pump_flow_rate=0.008,
    temperature=60.0,
    pump_characteristic=pump_char  # Real pump curve
)
```

## Best Practices

### 1. Pump Selection
- Use manufacturer data when available
- Consider operating range vs. system requirements
- Account for temperature effects on pump performance

### 2. Curve Definition
- Ensure curves are monotonic (pressure decreases with flow)
- Include sufficient data points in the operating range
- Validate curves against manufacturer specifications

### 3. System Design
- Check that pump curve intersects system curve in efficient region
- Avoid operation near pump limits (cavitation, overheating)
- Consider safety margins for varying conditions

### 4. Troubleshooting
- If convergence fails, check pump curve validity
- Verify flow range includes realistic operating points
- Use legacy method for comparison

## Example: Complete Workflow

```python
from lubrication_flow_package.components.pump import PumpCharacteristic
from lubrication_flow_package.solvers.network_flow_solver import NetworkFlowSolver
from lubrication_flow_package.network.flow_network import FlowNetwork

# 1. Create network (your existing code)
network = create_your_network()

# 2. Define pump characteristic from manufacturer data
flow_data = [0.000, 0.002, 0.004, 0.006, 0.008, 0.010]  # m³/s
pressure_data = [900000, 880000, 840000, 780000, 700000, 600000]  # Pa

pump = PumpCharacteristic.create_from_manufacturer_data(
    flow_points=flow_data,
    pressure_points=pressure_data
)

# 3. Solve system
solver = NetworkFlowSolver()
flows, info = solver.solve_network_flow_with_pump_physics(
    network=network,
    pump_flow_rate=0.008,
    temperature=60.0,
    pump_characteristic=pump
)

# 4. Analyze results
if info['pump_adequate']:
    print(f"✅ Pump suitable: {info['actual_flow_rate']*1000:.1f} L/s")
    print(f"Operating pressure: {info['operating_pressure']/1000:.0f} kPa")
else:
    print("❌ Pump inadequate - consider larger pump or system redesign")
```

## API Reference

### PumpCharacteristic Class

#### Constructor
```python
PumpCharacteristic(
    curve_type: str = "polynomial",
    coefficients: List[float] = None,
    flow_points: List[float] = None,
    pressure_points: List[float] = None,
    max_flow: float = None,
    max_pressure: float = None
)
```

#### Methods
- `get_pressure(flow_rate: float) -> float`: Get pressure for given flow
- `find_operating_point(system_resistance: float) -> Tuple[float, float]`: Find intersection with system curve
- `get_curve_points(num_points: int) -> Tuple[np.ndarray, np.ndarray]`: Generate curve points for plotting

#### Class Methods
- `create_typical_centrifugal_pump(max_pressure, max_flow, efficiency_point)`: Create typical centrifugal pump
- `create_from_manufacturer_data(flow_points, pressure_points)`: Create from performance data

### NetworkFlowSolver Updates

The `solve_network_flow_with_pump_physics` method now accepts an optional `pump_characteristic` parameter:

```python
solve_network_flow_with_pump_physics(
    network: FlowNetwork,
    pump_flow_rate: float,
    temperature: float,
    pump_characteristic: Optional[PumpCharacteristic] = None,  # NEW
    pump_max_pressure: float = 1e6,  # Fallback for legacy mode
    outlet_pressure: float = 101325.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> Tuple[Dict[str, float], Dict]
```

## Conclusion

The pump characteristic functionality provides a more realistic and accurate way to model pump behavior in lubrication systems. By using actual P-Q curves, you can:

- Predict real operating points
- Optimize pump selection
- Identify system design issues
- Improve flow distribution accuracy

The implementation maintains backward compatibility while adding powerful new capabilities for advanced hydraulic analysis.