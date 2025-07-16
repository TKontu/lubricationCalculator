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

# Next Steps / Backlog

## High Priority:

- [ ] **Implement Non-Linear Solver for Tree-Like Networks**
  - **Goal:** Create a new non-linear solver that is mathematically appropriate for tree-like (radial) networks, as the current `robust_newton` solver is designed for looped networks and fails to converge. The new solver will use a nodal pressure formulation.

  - **Phase 1: Test-Driven Development (TDD) Setup**
    - [ ] **Create Test File:** Create `tests/test_tree_solver.py`.
    - [ ] **Test Inverse Flow Calculation:** Write a test `test_component_flow_calculation` to verify the new `calculate_flow_rate(dP)` method on components. This test must pass before the solver implementation begins.
    - [ ] **Test Solver Convergence:** Write a test `test_solver_on_simple_tree` that builds a simple tree network and asserts that the new solver converges with physically plausible results (no negative pressures, flow conservation).
    - [ ] **Test Against Linear Solver:** Write a test `test_solver_against_linear_solver` to compare the new solver's results against the existing `NodalMatrixSolver` on a simple network where non-linear effects are minimal. The results should be nearly identical.

  - **Phase 2: Component Modification**
    - [ ] **Add Abstract Method:** Add `calculate_flow_rate(self, pressure_drop, fluid_properties)` to the `FlowComponent` base class.
    - [ ] **Implement on Components:** Implement the `calculate_flow_rate` method on `Channel`, `Nozzle`, and `Connector`. This will require using a numerical root-finder (e.g., `scipy.optimize.newton`) to solve for the flow that produces the given pressure drop.

  - **Phase 3: Solver Implementation**
    - [ ] **Create Solver File:** Create `lubrication_flow_package/solvers/tree_solver.py` and the `NonLinearTreeSolver` class.
    - [ ] **Identify Unknowns:** The solver's primary variables will be the pressures at all non-outlet nodes.
    - [ ] **Generate Initial Guess:** Use the existing `NodalMatrixSolver` to generate a high-quality initial guess for the node pressures.
    - [ ] **Implement Residual Function:** Write the `_evaluate_residual` method, where the residual is the net flow imbalance at each node. This will use the new `calculate_flow_rate` component method.
    - [ ] **Implement Jacobian Build:** Write the `_build_jacobian` method. The Jacobian entries will be the derivatives of the nodal flow imbalances with respect to the nodal pressures (`dq/dP`), which can be derived from the existing differential resistance calculation.
    - [ ] **Implement Newton-Raphson Loop:** Write the main solver loop to iterate until convergence.

  - **Phase 4: Integration and Validation**
    - [ ] **Integrate with CLI:** Add the `tree_nonlinear` option to the solver factory in `network_cli.py`.
    - [ ] **Pass All Tests:** Ensure all tests in `test_tree_solver.py` pass.
    - [ ] **Validate with Complex Network:** Run the new solver on the `complex_network_modified.json` example and confirm that it converges successfully and produces correct, physically valid results.
    - [ ] **Code Cleanup:** Remove any temporary debugging statements and finalize the code.

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