# Refactoring Plan: NetworkBuilder Implementation

This section outlines the plan to refactor the project by introducing a `NetworkBuilder` to centralize and simplify network creation.

**Status: Completed**

## Phase 1: Create the `NetworkBuilder` Foundation

- [x] **Create `network_builder.py`:** Create a new file in `lubrication_flow_package/utils/`.
- [x] **Define `NetworkBuilder` Class:**
  - [x] Initialize with an optional `SimulationConfig`.
  - [x] Hold a private `FlowNetwork` instance.
  - [x] Maintain an internal dictionary to track nodes by name.
- [x] **Implement Core Methods:**
  - [x] `_get_or_create_node(name: str)`: Private helper to manage node creation and prevent duplicates.
  - [x] `set_inlet(node_name: str)`: Method to define the network inlet.
  - [x] `add_outlet(node_name: str)`: Method to define network outlets.
  - [x] `build() -> FlowNetwork`: Finalize and return the `FlowNetwork` object.

## Phase 2: Implement High-Level Component-Adding Methods

- [x] **`add_pipe(...)`:** Add a method to create a `Channel` between two nodes.
- [x] **`add_nozzle(...)`:** Add a method to create a `Nozzle`.
- [x] **`add_fitting(...)`:** Add a generic method for `Connector` components (e.g., elbows, valves).
- [x] **`add_tee_junction(...)`:** Implement a physically-aware method for T-junctions.
  - [x] Model the tee as a central node.
  - [x] Use three `Connector` instances with asymmetric, realistic loss coefficients (K-factors) to accurately model pressure drops for the run and branch paths.

## Phase 3: Integrate the `NetworkBuilder` Across the Project

- [x] **Refactor `create_example_networks.py`:** Rewrite the script to use the `NetworkBuilder`.
- [x] **Update `main.py` and `network_cli.py`:** Modify the main script and CLI to use the `NetworkBuilder` for creating template files and loading networks from config.
- [x] **Refactor Unit Tests:** Update unit tests to use the `NetworkBuilder` for network creation, where appropriate.
  - _Note: Specialized tests, such as `test_nonlinear_solver.py`, will continue to use mock objects and direct instantiation to effectively test internal logic._

## Phase 4: Cleanup and Finalization

- [ ] **Review `FlowNetwork` API:** Mark old methods as private to encourage builder usage.
- [ ] **Update `readme.md`:** Add documentation for the new `NetworkBuilder` API.

---

# Robust NonLinear Tree Solver Implementation Plan

## Overview

This plan addresses the critical numerical stability issues in the `NonLinearTreeSolver` that are causing NaN propagation, singular Jacobian matrices, and convergence failures in the test suite.

## High Priority Tasks

### 1. Implement Robust Initial Guess Generation

- **Problem**: Linear solver provides poor/NaN initial guesses that propagate through iterations
- **Solution**:
  - Validate linear solver output for NaN/infinity values
  - Implement fallback strategies: physical bounds, previous solution, or simple pressure distribution
  - Add pressure bounds checking (0 < P < inlet_pressure)
  - Use outlet pressure as lower bound, inlet pressure as upper bound

### 2. Add Comprehensive Input Validation

- **Problem**: Invalid inputs cause cascading numerical failures
- **Solution**:
  - Validate all pressure values are finite and positive
  - Check flow rate bounds and physical constraints
  - Validate component parameters (diameter > 0, length > 0, etc.)
  - Add early termination for invalid network configurations

### 3. Implement Adaptive Finite Difference Step Size

- **Problem**: Fixed 1.0 Pa perturbation causes numerical instability
- **Solution**:
  - Calculate step size as fraction of pressure magnitude: `delta_p = max(1e-6 * abs(pressure), 1e-3)`
  - Use different step sizes for different pressure ranges
  - Implement Richardson extrapolation for higher accuracy
  - Add step size adaptation based on residual sensitivity

### 4. Add NaN/Infinity Detection and Handling

- **Problem**: NaN values propagate through calculations unchecked
- **Solution**:
  - Check for NaN/inf in all residual evaluations
  - Implement graceful fallback when component calculations fail
  - Add logging for debugging NaN origins
  - Use robust numerical methods that handle edge cases

### 5. Implement Robust Bracketing Algorithm

- **Problem**: Component flow rate calculations fail when bracketing fails
- **Solution**:
  - Improve initial bracket estimation using physical bounds
  - Add systematic bracket expansion with sign checking
  - Implement multiple fallback methods (Brent, secant, etc.)
  - Handle zero/negative pressure drops gracefully

## Medium Priority Tasks

### 6. Add Jacobian Matrix Conditioning and Singularity Detection

- **Problem**: Singular Jacobian matrices cause linear solver failures
- **Solution**:
  - Calculate matrix condition number and determinant
  - Add regularization for near-singular matrices
  - Implement pseudo-inverse for rank-deficient systems
  - Add warning/error messages for ill-conditioned systems

### 7. Implement Adaptive Damping Factor

- **Problem**: Fixed damping factor causes oscillations or slow convergence
- **Solution**:
  - Implement line search to find optimal step size
  - Use adaptive damping based on residual reduction
  - Add backtracking when solution diverges
  - Monitor convergence history for oscillation detection

### 8. Fix Flow Direction Sign Consistency

- **Problem**: Inconsistent sign conventions in flow calculations
- **Solution**:
  - Standardize pressure drop calculation: always `from_node - to_node`
  - Ensure consistent flow direction interpretation
  - Add clear documentation for sign conventions
  - Validate flow conservation at each node

### 9. Add Pressure Bounds Checking

- **Problem**: Unphysical negative pressures cause numerical issues
- **Solution**:
  - Implement pressure bounds: `outlet_pressure ≤ P ≤ inlet_pressure`
  - Add penalty methods for constraint violations
  - Project solutions back to feasible region
  - Use logarithmic pressure variables for positivity

### 10. Implement Fallback Solver Strategies

- **Problem**: Newton-Raphson can fail on difficult problems
- **Solution**:
  - Add quasi-Newton methods (BFGS, DFP) as fallbacks
  - Implement hybrid methods (Newton + bisection)
  - Add continuation methods for difficult cases
  - Use linear solver as final fallback

## Low Priority Tasks

### 11. Add Comprehensive Error Handling and Logging

- **Problem**: Debugging failures is difficult without proper logging
- **Solution**:
  - Add detailed logging at each iteration
  - Include residual norms, pressure values, and convergence metrics
  - Log warning for near-singular matrices
  - Add debug mode with extensive diagnostics

### 12. Implement Convergence Monitoring and Oscillation Detection

- **Problem**: Solver may oscillate without detecting it
- **Solution**:
  - Monitor residual history for oscillation patterns
  - Add stall detection (no progress for N iterations)
  - Implement solution averaging for oscillating solutions
  - Add adaptive tolerance based on problem difficulty

## Implementation Strategy

### Phase 1: Core Stability (High Priority)

1. Start with robust initial guess generation
2. Add comprehensive input validation
3. Implement adaptive finite differences
4. Add NaN/infinity detection

### Phase 2: Numerical Robustness (Medium Priority)

1. Fix flow direction consistency
2. Add Jacobian conditioning
3. Implement adaptive damping
4. Add pressure bounds checking

### Phase 3: Advanced Features (Low Priority)

1. Add fallback strategies
2. Implement comprehensive logging
3. Add convergence monitoring

## Testing Strategy

- Run existing test suite after each major change
- Add unit tests for each new validation function
- Test with pathological cases (near-zero flows, extreme pressure ratios)
- Benchmark against linear solver for low-flow cases
- Test convergence on complex network topologies

## Expected Outcomes

- Elimination of NaN-related failures
- Improved convergence reliability
- Better handling of edge cases
- More informative error messages
- Increased numerical stability across all test cases

---

# Previous Implementation History

## Non-Linear Solver for Tree-Like Networks (Completed - with issues)

- **Phase 1: Test-Driven Development (TDD) Setup**

  - [x] **Create Test File:** Create `tests/test_tree_solver.py`.
  - [x] **Test Inverse Flow Calculation:** Write a test `test_component_flow_calculation` to verify the new `calculate_flow_rate(dP)` method on components. This test must pass before the solver implementation begins.
  - [x] **Test Solver Convergence:** Write a test `test_solver_on_simple_tree` that builds a simple tree network and asserts that the new solver converges with physically plausible results (no negative pressures, flow conservation).
  - [x] **Test Against Linear Solver:** Write a test `test_solver_against_linear_solver` to compare the new solver's results against the existing `NodalMatrixSolver` on a simple network where non-linear effects are minimal. The results should be nearly identical.

- **Phase 2: Component Modification**

  - [x] **Add Abstract Method:** Add `calculate_flow_rate(self, pressure_drop, fluid_properties)` to the `FlowComponent` base class.
  - [x] **Implement on Components:** Implement the `calculate_flow_rate` method on `Channel`, `Nozzle`, and `Connector`. This will require using a numerical root-finder (e.g., `scipy.optimize.newton`) to solve for the flow that produces the given pressure drop.

- **Phase 3: Solver Implementation**

  - [x] **Create Solver File:** Create `lubrication_flow_package/solvers/tree_solver.py` and the `NonLinearTreeSolver` class.
  - [x] **Identify Unknowns:** The solver's primary variables will be the pressures at all non-outlet nodes.
  - [x] **Generate Initial Guess:** Use the existing `NodalMatrixSolver` to generate a high-quality initial guess for the node pressures.
  - [x] **Implement Residual Function:** Write the `_evaluate_residual` method, where the residual is the net flow imbalance at each node. This will use the new `calculate_flow_rate` component method.
  - [x] **Implement Jacobian Build:** Write the `_build_jacobian` method. The Jacobian entries will be the derivatives of the nodal flow imbalances with respect to the nodal pressures (`dq/dP`), which can be derived from the existing differential resistance calculation.
  - [x] **Implement Newton-Raphson Loop:** Write the main solver loop to iterate until convergence.

- **Phase 4: Integration and Validation**
  - [ ] **Integrate with CLI:** Add the `tree_nonlinear` option to the solver factory in `network_cli.py`.
  - [ ] **Pass All Tests:** Ensure all tests in `test_tree_solver.py` pass. **CURRENT ISSUE - 2/3 tests failing**
  - [ ] **Validate with Complex Network:** Run the new solver on the `complex_network_modified.json` example and confirm that it converges successfully and produces correct, physically valid results.
  - [ ] **Code Cleanup:** Remove any temporary debugging statements and finalize the code.

---

# Next Steps / Backlog

## High Priority:

- [ ] **Component Library Expansion:**
  - [ ] Implement `Pump` component with PQ-curve modeling.
  - [ ] Implement `ThermalExchanger` and `VariableValve` components.

## Low Priority / Future Enhancements:

- [ ] **Advanced Solver Features:**
  - [ ] Implement Trust Region methods as an alternative to line search.
  - [ ] Implement Broyden's method for cheaper Jacobian updates.
- [ ] **GUI Improvements:**
  - [ ] Refactor the GUI to use the unified solver interface.
  - [ ] Stabilize the GUI data model and fix interaction bugs.
- [ ] **Advanced Physics Models:**
  - [ ] Add support for transient simulations.
  - [ ] Add support for compressible flow.
  - [ ] Re-evaluate temperature-dependent viscosity during solver iterations.
