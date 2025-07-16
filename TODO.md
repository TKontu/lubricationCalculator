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

# Unified Hydraulic Network Solver Architecture Plan

## Overview
This plan addresses the architectural challenges of having multiple specialized solvers and proposes a unified approach that automatically selects the optimal solver based on network topology while maintaining numerical robustness.

## Current Solver Assessment

### **Existing Solvers Analysis:**
1. **NodalMatrixSolver**: Iterative conductance matrix method, handles both tree and looped networks but poor convergence for strong non-linearities
2. **RobustNonLinearSolver**: Flow-based Newton-Raphson for looped networks, mathematically handles trees but computationally inefficient (O(n³) vs O(n))
3. **NonLinearTreeSolver**: Nodal pressure formulation for tree networks, appropriate algorithm but currently has numerical stability issues

### **Key Insight:**
- **Tree Networks**: Unique flow distribution, optimal with pressure-based formulation (O(n) structure, O(n²) Newton-Raphson)
- **Looped Networks**: Multiple flow paths, requires flow-based formulation with cycle constraints
- **Hybrid Networks**: Can be solved by RobustNonLinearSolver but benefit from decomposition strategies

## High Priority: Unified Solver Architecture

### 1. Implement Automatic Topology Detection
- **Problem**: Users must manually select solvers, no automatic optimization
- **Solution**:
  - Create `NetworkTopologyAnalyzer` class
  - Implement `classify_network(network) -> NetworkTopology`
  - Detect: TREE, LOOPED, HYBRID topologies using NetworkX
  - Add network characteristics analysis (strong non-linearities, outlet count, etc.)

### 2. Create Unified Solver Interface
- **Problem**: Multiple solver interfaces confuse users and duplicate code
- **Solution**:
  ```python
  class UnifiedNetworkSolver(SolverBase):
      def solve(self, network: FlowNetwork) -> Dict:
          topology = self.topology_analyzer.classify_network(network)
          solver = self.solver_factory.create_solver(topology, network)
          return solver.solve(network)
  ```

### 3. Fix NonLinearTreeSolver Numerical Issues (Critical)
- **Problem**: Tree solver has NaN propagation and convergence failures
- **Solution**:
  - [x] Implement robust initial guess generation (completed)
  - [x] Add comprehensive input validation (completed)
  - [x] Implement adaptive finite difference step size (completed)
  - [x] Add NaN/infinity detection and handling (completed)
  - [x] Implement robust bracketing algorithm (completed)
  - [ ] **Still needed**: Jacobian conditioning, adaptive damping, pressure bounds

### 4. Implement Solver Factory with Fallback Chain
- **Problem**: No fallback when primary solver fails
- **Solution**:
  ```python
  class SolverFactory:
      def create_solver_chain(self, topology: NetworkTopology, network: FlowNetwork) -> List[SolverBase]:
          if topology == NetworkTopology.TREE:
              return [NonLinearTreeSolver(self.config), 
                      NodalMatrixSolver(self.config)]
          elif topology == NetworkTopology.LOOPED:
              return [RobustNonLinearSolver(self.config), 
                      NodalMatrixSolver(self.config)]
          else:  # HYBRID
              return [RobustNonLinearSolver(self.config), 
                      NodalMatrixSolver(self.config)]
  ```

### 5. Add Solver Performance Monitoring
- **Problem**: No visibility into solver performance and selection decisions
- **Solution**:
  - Add solver timing and convergence metrics
  - Log solver selection rationale
  - Monitor fallback usage patterns
  - Add performance comparison between solvers

## Medium Priority: Enhanced Solver Capabilities

### 6. Implement Hybrid Network Decomposition
- **Problem**: Hybrid networks inefficiently solved as monolithic systems
- **Solution**:
  - Decompose hybrid networks into tree and looped regions
  - Solve each region with appropriate solver
  - Handle interface coupling between regions
  - Use decomposition for large networks to improve scalability

### 7. Complete Tree Solver Robustness
- **Remaining Issues**:
  - Add Jacobian matrix conditioning and singularity detection
  - Implement adaptive damping factor with line search
  - Add pressure bounds checking and constraint handling
  - Fix flow direction sign consistency throughout the solver

### 8. Optimize RobustNonLinearSolver for Tree Networks
- **Problem**: Flow-based solver inefficient for tree networks
- **Solution**:
  - Add tree network detection within RobustNonLinearSolver
  - Implement simplified algorithm path for tree networks
  - Skip cycle detection and pressure loop equations for trees
  - Use BFS pressure calculation instead of matrix solve

### 9. Add Adaptive Algorithm Selection
- **Problem**: Static topology classification may miss optimal solver selection
- **Solution**:
  - Monitor convergence behavior during solving
  - Switch solvers if convergence stalls or diverges
  - Use machine learning to predict optimal solver based on network characteristics
  - Implement solver recommendation system

## Low Priority: Advanced Features

### 10. Implement Continuation Methods
- **Problem**: Difficult networks may require parameter continuation
- **Solution**:
  - Add parameter continuation for difficult convergence cases
  - Implement pseudo-arc-length continuation for bifurcation problems
  - Add automatic restart with different initial conditions

### 11. Add Parallel Processing Support
- **Problem**: Large networks could benefit from parallel processing
- **Solution**:
  - Implement parallel Jacobian construction
  - Add parallel residual evaluation
  - Use parallel linear solvers for large sparse systems
  - Implement parallel region solving for decomposed networks

### 12. Enhanced Error Handling and Diagnostics
- **Problem**: Debugging solver failures is difficult
- **Solution**:
  - Add comprehensive logging throughout solver chain
  - Implement solver failure analysis and reporting
  - Add network diagnostic tools (conditioning, singularity detection)
  - Create solver performance profiling tools

## Implementation Strategy

### Phase 1: Core Architecture (Immediate)
1. Fix remaining NonLinearTreeSolver issues to pass all tests
2. Implement topology detection and unified solver interface
3. Create solver factory with fallback mechanisms
4. Add comprehensive testing for all topology types

### Phase 2: Performance Optimization (Medium Term)
1. Optimize RobustNonLinearSolver for tree networks
2. Implement hybrid network decomposition
3. Add adaptive algorithm selection
4. Performance benchmarking and optimization

### Phase 3: Advanced Features (Long Term)
1. Add continuation methods and parallel processing
2. Implement machine learning-based solver selection
3. Create comprehensive diagnostics and monitoring
4. Add support for specialized network types (thermal, transient, etc.)

## Testing Strategy
- **Unit Tests**: Each solver component tested independently
- **Integration Tests**: Unified solver interface with all topology types
- **Performance Tests**: Benchmark solver selection and execution times
- **Regression Tests**: Ensure no degradation in existing functionality
- **Stress Tests**: Large networks, pathological cases, extreme parameter values

## Expected Outcomes
- **User Experience**: Single, simple interface for all network types
- **Performance**: Optimal solver selection based on network characteristics
- **Reliability**: Robust fallback mechanisms prevent solver failures
- **Maintainability**: Unified architecture reduces code duplication
- **Scalability**: Efficient algorithms for both small and large networks

## Migration Strategy
- **Backward Compatibility**: Existing solver interfaces remain available
- **Gradual Migration**: CLI and GUI can migrate to unified interface over time
- **Documentation**: Clear guidelines for when to use unified vs. specialized solvers
- **Testing**: Extensive validation that unified solver produces identical results

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
