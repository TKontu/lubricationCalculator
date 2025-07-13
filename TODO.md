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
