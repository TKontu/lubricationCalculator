# Migration Guide: Enhanced Connector Features

## Overview

This guide helps you migrate from the basic connector implementation to the enhanced version with geometry-based and Reynolds number-dependent loss coefficients.

## Backward Compatibility

**Good news**: All existing code will continue to work without changes! The enhanced connector maintains full backward compatibility.

## Migration Steps

### Step 1: Enable Auto-Calculation (Recommended)

**Before (basic usage):**
```python
connector = Connector(ConnectorType.ELBOW_90, diameter=0.05)
```

**After (enhanced usage):**
```python
connector = Connector(
    ConnectorType.ELBOW_90, 
    diameter=0.05,
    auto_calculate_k=True  # Enable enhanced calculations
)
```

### Step 2: Add Fluid Viscosity

**Before:**
```python
fluid_props = {'density': 900.0}
```

**After:**
```python
fluid_props = {
    'density': 900.0,
    'viscosity': 0.01  # Pa·s - Required for Reynolds number calculations
}
```

### Step 3: Use Specific Connector Types

**Before:**
```python
# Generic reducer
reducer = Connector(ConnectorType.REDUCER, diameter=0.06, diameter_out=0.04)
```

**After:**
```python
# Specific reducer type with geometry
reducer = Connector(
    ConnectorType.REDUCER_GRADUAL,  # More specific type
    diameter=0.06,
    diameter_out=0.04,
    taper_angle=30.0,  # Geometric parameter
    auto_calculate_k=True
)
```

### Step 4: Leverage Geometric Parameters

**Before:**
```python
elbow = Connector(ConnectorType.ELBOW_90, diameter=0.05)
```

**After:**
```python
# Long radius elbow for better performance
elbow = Connector(
    ConnectorType.ELBOW_90,
    diameter=0.05,
    bend_radius_ratio=3.0,  # R/D = 3 (long radius)
    auto_calculate_k=True
)
```

## Common Migration Patterns

### Pattern 1: Valve Control

**Before:**
```python
valve = Connector(ConnectorType.GATE_VALVE, diameter=0.05, loss_coefficient=0.15)
```

**After:**
```python
valve = Connector(
    ConnectorType.GATE_VALVE,
    diameter=0.05,
    valve_opening=1.0,  # Fully open
    auto_calculate_k=True
)

# Later, to partially close the valve:
valve.valve_opening = 0.75  # 75% open
```

### Pattern 2: Junction Optimization

**Before:**
```python
tee = Connector(ConnectorType.T_JUNCTION, diameter=0.05)
```

**After:**
```python
# Use WYE junction for better flow characteristics
wye = Connector(
    ConnectorType.WYE_JUNCTION,
    diameter=0.05,
    flow_split_ratio=0.5,  # Equal split
    auto_calculate_k=True
)
```

### Pattern 3: Reducer Design

**Before:**
```python
reducer = Connector(ConnectorType.REDUCER, diameter=0.08, diameter_out=0.05)
```

**After:**
```python
# Gradual reducer with optimized taper angle
reducer = Connector(
    ConnectorType.REDUCER_GRADUAL,
    diameter=0.08,
    diameter_out=0.05,
    taper_angle=20.0,  # Optimized for low loss
    auto_calculate_k=True
)
```

## New Connector Types Available

### Elbows
- `ELBOW_30` - 30-degree elbow
- `ELBOW_45` - 45-degree elbow  
- `LONG_RADIUS_ELBOW` - R/D ≥ 1.5
- `SHORT_RADIUS_ELBOW` - R/D ≈ 1.0
- `ELBOW_SMOOTH` - Smooth bend
- `ELBOW_MITERED` - Sharp mitered elbow

### Reducers/Expanders
- `REDUCER_GRADUAL` - Gradual contraction
- `REDUCER_SUDDEN` - Sudden contraction
- `EXPANDER_GRADUAL` - Gradual expansion
- `EXPANDER_SUDDEN` - Sudden expansion

### Valves
- `GATE_VALVE` - Gate valve
- `BALL_VALVE` - Ball valve
- `GLOBE_VALVE` - Globe valve
- `CHECK_VALVE` - Check valve
- `BUTTERFLY_VALVE` - Butterfly valve

### Junctions
- `WYE_JUNCTION` - Y-junction (lower loss than T)
- `LATERAL_TEE` - Lateral tee

### Fittings
- `UNION` - Pipe union
- `COUPLING` - Pipe coupling
- `ADAPTER` - Pipe adapter
- `RETURN_BEND` - 180-degree return bend

## Performance Optimization Tips

### 1. Choose the Right Connector Type
```python
# Instead of generic T_JUNCTION
tee = Connector(ConnectorType.T_JUNCTION, diameter=0.05)

# Use WYE_JUNCTION for 20-30% lower pressure drop
wye = Connector(ConnectorType.WYE_JUNCTION, diameter=0.05, auto_calculate_k=True)
```

### 2. Optimize Elbow Geometry
```python
# Standard elbow
elbow_std = Connector(ConnectorType.ELBOW_90, diameter=0.05)

# Long radius elbow (50-60% lower pressure drop)
elbow_lr = Connector(
    ConnectorType.LONG_RADIUS_ELBOW,
    diameter=0.05,
    bend_radius_ratio=3.0,
    auto_calculate_k=True
)
```

### 3. Optimize Reducer Taper Angles
```python
# Sudden reducer (high loss)
reducer_sudden = Connector(
    ConnectorType.REDUCER_SUDDEN,
    diameter=0.08, diameter_out=0.05
)

# Gradual reducer with optimized angle (much lower loss)
reducer_gradual = Connector(
    ConnectorType.REDUCER_GRADUAL,
    diameter=0.08, diameter_out=0.05,
    taper_angle=15.0,  # Optimal for most applications
    auto_calculate_k=True
)
```

## Troubleshooting

### Issue: Higher pressure drops than expected
**Solution**: Check if `auto_calculate_k=True` and ensure viscosity is provided in fluid properties.

### Issue: Unrealistic loss coefficients
**Solution**: Verify geometric parameters are reasonable (e.g., bend_radius_ratio ≥ 1.0).

### Issue: Valve not responding to opening changes
**Solution**: Ensure `auto_calculate_k=True` for dynamic valve calculations.

## Testing Your Migration

Use this simple test to verify your migration:

```python
# Test script
from lubrication_flow_package.components.connector import Connector
from lubrication_flow_package.components.base import ConnectorType

# Create enhanced connector
connector = Connector(
    ConnectorType.ELBOW_90,
    diameter=0.05,
    bend_radius_ratio=2.0,
    auto_calculate_k=True
)

# Test with proper fluid properties
fluid_props = {'density': 900.0, 'viscosity': 0.01}
flow_rate = 0.001

# Calculate pressure drop
dp = connector.calculate_pressure_drop(flow_rate, fluid_props)
print(f"Enhanced pressure drop: {dp:.2f} Pa")

# Get connector info
info = connector.get_connector_info()
print(f"Loss coefficient: {info['current_loss_coefficient']:.3f}")
```

## Questions?

If you encounter any issues during migration, check:
1. Fluid properties include both density and viscosity
2. Geometric parameters are within reasonable ranges
3. `auto_calculate_k=True` is set for enhanced calculations
4. Connector types are appropriate for your application