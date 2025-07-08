# Task List / Backlog

## High Priority / Next Steps

- **Solver Accuracy and Robustness:**
  - [ ] **Fix Non-Linear Solver:** The iterative solver in `nodal_matrix_solver.py` incorrectly uses total resistance (`R = ΔP / Q`) instead of the physically correct differential resistance (`R = d(ΔP)/dQ`). This leads to slow convergence and potential instability.
    - [ ] **Action:** Modify the `solve_nodal_iterative` loop to call the `_calculate_component_resistance` function, which correctly computes the differential resistance using a finite-difference method. The existing `_compute_resistance` function should be deprecated or removed.
- **GUI Data Model Refactoring (Robustness):**
  - [ ] **Phase 1: Core Logic and Data Structure**
    - [ ] 1.1. In `app.py`, define a constant for the `SINK_NODE` and update the `add_nozzle` method to create a directed edge to this sink node instead of setting a node property.
    - [ ] 1.2. In `app.py`, modify `create_flow_network` to identify outlet nodes by finding nodes with edges pointing to the `SINK_NODE`.
    - [ ] 1.3. In `app.py`, update the logic in `create_flow_network` to correctly instantiate `Nozzle` components when an edge connects to the `SINK_NODE`.
  - [ ] **Phase 2: Visualization**
    - [ ] 2.1. In `canvas.py`, modify `draw_graph` to exclude the `SINK_NODE` from being rendered on the canvas.
    - [ ] 2.2. In `canvas.py`, enhance `draw_graph` to render a distinct visual representation for nozzle edges (e.g., a different color or line style) to clearly mark exit points.
  - [ ] **Phase 3: UI and Cleanup**
    - [ ] 3.1. In `app.py`, remove the now-redundant "Set as Outlet" option from the node context menu.
    - [ ] 3.2. In `dialogs.py`, remove the "outlet" option from the "type" combobox in the `PropertiesEditor`.
    - [ ] 3.3. Thoroughly test the new implementation to ensure creating, connecting, simulating, and visualizing networks with nozzles is robust.

- **Graphical User Interface (GUI) - Polish and Refine:**
  - [ ] **Phase 1: Canvas and Usability**
    - [ ] 1.1. Implement dynamic canvas zooming (e.g., zoom to fit, mouse wheel zoom) and panning to replace the fixed-size canvas.
    - [ ] 1.2. Add a "Save Network" and "Load Network" feature to persist and retrieve the graph layout and properties.
    - [ ] 1.3. Improve visual feedback during operations (e.g., highlight nodes/edges on hover, show a "connecting" line when creating a channel).
  - [ ] **Phase 2: Code Refactoring and Robustness**
    - [ ] 2.1. Refactor the monolithic `gui/main.py` into smaller, more manageable modules (e.g., `app.py`, `canvas.py`, `sidebar.py`, `dialogs.py`).
    - [ ] 2.2. Enhance error handling with more specific dialogs for different simulation or file I/O errors.
    - [ ] 2.3. Add undo/redo functionality for network modifications.
  - [ ] **Phase 3: Advanced Features**
    - [ ] 3.1. Implement a toolbar for common actions (e.g., save, load, zoom).
    - [ ] 3.2. Allow direct editing of component properties on the canvas (e.g., double-clicking a channel to open its properties).
    - [ ] 3.3. Add support for visualizing multiple result quantities simultaneously (e.g., pressure and flow).

- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django.

## Completed

- **Graphical User Interface (GUI) - Initial Implementation:**
  - [x] **Phase 1: A Stable and Usable Foundation**
    - [x] 1.1. Fix the canvas auto-zoom issue to ensure a consistent view.
    - [x] 1.2. Implement "Nodes" and "Components" listboxes to serve as an element browser.
    - [x] 1.3. Link the element browser to the "Properties" editor.
  - [x] **Phase 2: Intuitive Network Building**
    - [x] 2.1. Implement a right-click context menu on canvas nodes.
    - [x] 2.2. Add a "type" property to the "Properties" editor for nodes (`inlet`, `outlet`, `junction`) and update the node's color accordingly.
    - [x] 2.3. Implement a visual workflow for connecting two nodes with a "Channel".
  - [x] **Phase 3: Full Simulation and Visualization**
    - [x] 3.1. Implement adding "Nozzles" to outlet nodes via the context menu.
    - [x] 3.2. Wire up the "Simulation Settings" panel.
    - [x] 3.3. Implement the "Run Simulation" button with full network creation and error handling.
    - [x] 3.4. Implement color-coding of the graph to visualize simulation results.

## Medium Priority

- **Improve Connector Physics:**
  - [ ] The pressure drop calculation in `connector.py` for reducers and expanders may be inaccurate. It currently uses the inlet diameter for velocity calculation in all cases.
    - [ ] **Action:** Review and correct the `calculate_pressure_drop` method in `connector.py` to use the appropriate diameter (inlet or outlet) for velocity calculation based on the specific connector type (e.g., expansion vs. contraction).
- **Enhance Flow Initialization:**
  - [ ] The `_initialize_flows` method in the solver only considers a single sink node when distributing the initial flow. This can lead to a poor initial guess in networks with multiple outlets.
    - [ ] **Action:** Update the flow initialization logic to properly handle multiple sink nodes, distributing the total flow among all outlets for a more balanced and effective initial state.

## Low Priority / Future Enhancements

- **Validation & Calibration:**
  - [ ] Develop a strategy for validating solver accuracy against real-world data or established benchmarks (e.g., for the specified pressure/flow/temperature ranges).
  - [ ] Implement automated tests to verify accuracy against known analytical solutions or experimental data for various network configurations and parameter ranges.
- **Component Library Expansion:**
  - [ ] Implement a `Valve` component with adjustable opening/closing settings.
  - [ ] Implement an `Accumulator` component.