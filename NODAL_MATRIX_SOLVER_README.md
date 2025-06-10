# Iterative Nodal-Matrix Solver for Hydraulic Networks

## Overview

This implementation provides an **iterative nodal-matrix solver** for hydraulic networks with non-linear edge resistances. The solver finds node pressures and edge flows such that:

1. **Mass conservation** is satisfied at all nodes
2. **Pressure-flow law** holds on every edge: `ΔP_e = R_e(Q_e) · Q_e`

The solver uses the **nodal analysis method** where node pressures are the primary unknowns, making it particularly effective for networks with multiple junctions and complex topologies.

## Key Features

- ✅ **Iterative solution** for non-linear resistances `R_e(Q)`
- ✅ **Sparse matrix implementation** using SciPy for performance
- ✅ **Handles complex topologies**: series, parallel, Y-networks, multi-junction networks
- ✅ **Automatic convergence checking** with configurable tolerances
- ✅ **Mass conservation validation** at all nodes
- ✅ **Comprehensive unit tests** with analytical verification

## Algorithm Description

### Mathematical Foundation

The solver implements the **nodal analysis method**:

1. **Conductance Matrix**: Build matrix `A` where `A[i,j]` represents conductance between nodes
2. **RHS Vector**: Build vector `b` representing net flow injections at each node
3. **Linear System**: Solve `A · p = b` for node pressures `p`
4. **Flow Computation**: Calculate edge flows from `Q_e = G_e · (p_i - p_j)`
5. **Iteration**: Update conductances `G_e = 1/R_e(Q_e)` and repeat until convergence

### Implementation Steps

```python
def solve_nodal_iterative(network, source_node_id, sink_node_id, Q_total, fluid_properties):
    # 1. Initialize edge flows
    Q_e = Q_total / number_of_edges
    
    for iteration in range(max_iter):
        # 2. Compute resistances and conductances
        R_e = component.calculate_pressure_drop(Q_e) / Q_e
        G_e = 1.0 / R_e
        
        # 3. Build conductance matrix A and RHS vector b
        for each connection (i,j):
            A[i,i] += G_e
            A[j,j] += G_e
            A[i,j] -= G_e
            A[j,i] -= G_e
        
        b[source] = Q_total  # Flow injection
        
        # 4. Solve linear system A·p = b
        pressures = spsolve(A, b)
        
        # 5. Compute new flows
        Q_e_new = G_e * (p_i - p_j)
        
        # 6. Check convergence
        if max(|Q_e_new - Q_e|) < tol_flow and max(|ΔP - R_e*Q_e|) < tol_pressure:
            break
        
        Q_e = Q_e_new
```

## Usage Examples

### Basic Usage

```python
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.network.flow_network import FlowNetwork

# Create solver
solver = NodalMatrixSolver()

# Solve network
pressures, flows = solver.solve_nodal_iterative(
    network=my_network,
    source_node_id="source_id",
    sink_node_id="sink_id", 
    Q_total=0.001,  # m³/s
    fluid_properties={'density': 900.0, 'viscosity': 0.01},
    tol_flow=1e-6,
    tol_pressure=1e2,
    max_iter=20
)
```

### Creating Custom Components

```python
class LinearResistanceComponent(FlowComponent):
    def __init__(self, resistance):
        super().__init__()
        self.resistance = resistance  # Pa·s/m³
    
    def calculate_pressure_drop(self, flow_rate, fluid_properties):
        return self.resistance * abs(flow_rate)

class QuadraticResistanceComponent(FlowComponent):
    def __init__(self, linear_coeff, quadratic_coeff):
        super().__init__()
        self.a = linear_coeff      # Pa·s/m³
        self.b = quadratic_coeff   # Pa·s²/m⁶
    
    def calculate_pressure_drop(self, flow_rate, fluid_properties):
        Q = abs(flow_rate)
        return self.a * Q + self.b * Q * Q
```

## Network Topologies Supported

### 1. Series Networks
```
Source --R1--> Node1 --R2--> Sink
```

### 2. Parallel Networks  
```
Source --R1--> Sink
       --R2--> Sink
```

### 3. Y-Networks (Series-Parallel)
```
Source --R1--> Junction --R2--> Sink
                        --R3--> Sink
```

### 4. Complex Multi-Junction Networks
```
Source --R1--> J1 --R2--> J2 --R4--> Sink
               |          |
              R3         R5
               |          |
               +----------+
```

## Performance Characteristics

### Convergence Properties

- **Linear resistances**: Converges in 1-2 iterations
- **Non-linear resistances**: Typically converges in 3-10 iterations
- **Complex networks**: Convergence depends on network conditioning

### Computational Complexity

- **Matrix size**: `(N-1) × (N-1)` where N = number of nodes
- **Sparse matrix**: Only non-zero entries stored and computed
- **Time complexity**: `O(E + N²)` per iteration where E = number of edges

### Memory Usage

- **Sparse storage**: Efficient for large networks with sparse connectivity
- **Memory scaling**: Linear with number of edges for sparse networks

## Validation and Testing

### Unit Tests

The implementation includes comprehensive unit tests:

```bash
cd /workspace/lubricationCalculator
python test_nodal_matrix_solver.py
```

Test cases cover:
- ✅ Simple Y-network with analytical verification
- ✅ Series networks
- ✅ Parallel networks  
- ✅ Non-linear resistance components
- ✅ Convergence tolerance verification

### Analytical Verification

For linear resistances, results are verified against analytical solutions:

**Parallel Resistance**: `R_parallel = 1/(1/R1 + 1/R2 + ...)`

**Series Resistance**: `R_series = R1 + R2 + ...`

**Flow Distribution**: `Q_i = Q_total * G_i / G_total` where `G_i = 1/R_i`

## Demonstration

Run the comprehensive demonstration:

```bash
python nodal_solver_demo.py
```

This demonstrates:
- Series, parallel, and Y-networks
- Non-linear resistance handling
- Complex multi-junction networks
- Flow distribution analysis

## API Reference

### NodalMatrixSolver Class

```python
class NodalMatrixSolver:
    def __init__(self, logger=None):
        """Initialize the nodal matrix solver"""
    
    def solve_nodal_iterative(self, 
                             network: FlowNetwork,
                             source_node_id: str,
                             sink_node_id: str, 
                             Q_total: float,
                             fluid_properties: Dict,
                             tol_flow: float = 1e-6,
                             tol_pressure: float = 1e2,
                             max_iter: int = 20) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Solve hydraulic network using iterative nodal-matrix method.
        
        Args:
            network: FlowNetwork to solve
            source_node_id: ID of source node (flow injection)
            sink_node_id: ID of sink node (reference pressure = 0)
            Q_total: Total flow rate (m³/s)
            fluid_properties: Dict with 'density' and 'viscosity'
            tol_flow: Flow convergence tolerance (m³/s)
            tol_pressure: Pressure-flow law tolerance (Pa)
            max_iter: Maximum iterations
            
        Returns:
            Tuple of (node_pressures, edge_flows)
        """
```

### Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tol_flow` | float | Flow convergence tolerance (m³/s) | 1e-6 |
| `tol_pressure` | float | Pressure-flow law tolerance (Pa) | 1e2 |
| `max_iter` | int | Maximum iterations | 20 |

### Return Values

- **`node_pressures`**: Dict mapping node_id → pressure (Pa)
- **`edge_flows`**: Dict mapping component_id → flow rate (m³/s)

## Advantages vs. Path-Based Methods

| Aspect | Nodal Matrix | Path-Based |
|--------|--------------|------------|
| **Scalability** | O(N²) matrix | O(P) paths |
| **Complex topologies** | Excellent | Can struggle |
| **Multiple junctions** | Natural | Requires path enumeration |
| **Sparse networks** | Efficient | Less efficient |
| **Convergence** | Robust | Path-dependent |

## Limitations and Considerations

### Current Limitations

1. **Single source/sink**: Currently supports one source and one sink node
2. **Positive flows**: Assumes all flows are positive (no reverse flow)
3. **Connected networks**: Requires fully connected network topology

### Future Enhancements

- [ ] Multiple source/sink support
- [ ] Reverse flow handling
- [ ] Network partitioning for disconnected components
- [ ] Adaptive time stepping for transient analysis
- [ ] GPU acceleration for large networks

## Integration with Existing Codebase

The nodal matrix solver integrates seamlessly with the existing lubrication flow package:

```python
# Import alongside existing solvers
from lubrication_flow_package.solvers import NetworkFlowSolver, NodalMatrixSolver

# Use with existing network and component classes
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
```

## References

1. **Nodal Analysis**: Classical circuit analysis method adapted for hydraulic networks
2. **Sparse Matrix Methods**: SciPy sparse matrix implementation for computational efficiency
3. **Iterative Methods**: Fixed-point iteration for non-linear resistance convergence

---

**Author**: OpenHands AI Assistant  
**Date**: 2025-06-10  
**Version**: 1.0  
**License**: Same as parent project