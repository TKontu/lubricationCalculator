# Refactoring Plan: NetworkBuilder Implementation

This section outlines the plan to refactor the project by introducing a `NetworkBuilder` to centralize and simplify network creation.

## Phase 1: Create the `NetworkBuilder` Foundation
- [ ] **Create `network_builder.py`:** Create a new file in `lubrication_flow_package/utils/`.
- [ ] **Define `NetworkBuilder` Class:**
    - [ ] Initialize with an optional `SimulationConfig`.
    - [ ] Hold a private `FlowNetwork` instance.
    - [ ] Maintain an internal dictionary to track nodes by name.
- [ ] **Implement Core Methods:**
    - [ ] `_get_or_create_node(name: str)`: Private helper to manage node creation and prevent duplicates.
    - [ ] `set_inlet(node_name: str)`: Method to define the network inlet.
    - [ ] `add_outlet(node_name: str)`: Method to define network outlets.
    - [ ] `build() -> FlowNetwork`: Finalize and return the `FlowNetwork` object.

## Phase 2: Implement High-Level Component-Adding Methods
- [ ] **`add_pipe(...)`:** Add a method to create a `Channel` between two nodes.
- [ ] **`add_nozzle(...)`:** Add a method to create a `Nozzle`.
- [ ] **`add_fitting(...)`:** Add a generic method for `Connector` components (e.g., elbows, valves).
- [ ] **`add_tee_junction(...)`:** Implement a physically-aware method for T-junctions.
    - [ ] Model the tee as a central node.
    - [ ] Use three `Connector` instances with asymmetric, realistic loss coefficients (K-factors) to accurately model pressure drops for the run and branch paths.

## Phase 3: Integrate the `NetworkBuilder` Across the Project
- [ ] **Refactor `create_example_networks.py`:** Rewrite the script to use the `NetworkBuilder`.
- [ ] **Update `main.py`:** Modify the main script to use the `NetworkBuilder`.
- [ ] **Refactor Unit Tests:** Update all tests that create networks to use the `NetworkBuilder`.

## Phase 4: Cleanup and Finalization
- [ ] **Review `FlowNetwork` API:** Mark old methods as private to encourage builder usage.
- [ ] **Update `readme.md`:** Add documentation for the new `NetworkBuilder` API.

---

# Original Task List / Backlog

## High Priority:

- [ ] **Component Library Expansion:**
  - [ ] Implement components so that nodes are automatically created with implemented components.
    - [ ] T-junction = junction node and node at each end of each tee
    - [ ] Pipe = nodes at both ends
    - [ ] bend = nodes at both ends
    - [ ] etc.
    - [ ] Ensure that when components are connected to each other, the connected nodes merge to become a single node
  - [ ] Implement `Pump` component with PQ-curve modeling.
  - [ ] Implement `ThermalExchanger` and `VariableValve` components.

## Low Priority / Future Enhancements

- [ ] **Component Library Expansion:**

  - [ ] Implement `Pump` component with PQ-curve modeling.
  - [ ] Implement `ThermalExchanger` and `VariableValve` components.

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