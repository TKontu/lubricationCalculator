# Task List / Backlog

## High Priority / Next Steps

- **Graphical User Interface (GUI) - Incremental Refactor:**
  - [ ] **Phase 1: A Stable and Usable Foundation**
    - [ ] 1.1. Fix the canvas auto-zoom issue to ensure a consistent view.
    - [ ] 1.2. Implement "Nodes" and "Components" listboxes to serve as an element browser.
    - [ ] 1.3. Link the element browser to the "Properties" editor.
  - [ ] **Phase 2: Intuitive Network Building**
    - [ ] 2.1. Implement a right-click context menu on canvas nodes.
    - [ ] 2.2. Add a "type" property to the "Properties" editor for nodes (`inlet`, `outlet`, `junction`) and update the node's color accordingly.
    - [ ] 2.3. Implement a visual workflow for connecting two nodes with a "Channel".
  - [ ] **Phase 3: Full Simulation and Visualization**
    - [ ] 3.1. Implement adding "Nozzles" to outlet nodes via the context menu.
    - [ ] 3.2. Wire up the "Simulation Settings" panel.
    - [ ] 3.3. Implement the "Run Simulation" button with full network creation and error handling.
    - [ ] 3.4. Implement color-coding of the graph to visualize simulation results.
- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django.

## Medium Priority

## Low Priority / Future Enhancements

- **Validation & Calibration:**
  - [ ] Develop a strategy for validating solver accuracy against real-world data or established benchmarks (e.g., for the specified pressure/flow/temperature ranges).
  - [ ] Implement automated tests to verify accuracy against known analytical solutions or experimental data for various network configurations and parameter ranges.
- **Component Library Expansion:**
  - [ ] Implement a `Valve` component with adjustable opening/closing settings.
  - [ ] Implement an `Accumulator` component.
