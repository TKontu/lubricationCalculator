# Task List / Backlog

## High Priority / Next Steps

- **Graphical User Interface (GUI) - Overhaul:**
  - [ ] **Phase 1: Core Interaction Model**
    - [ ] Implement a right-click context menu for nodes.
    - [ ] Add options to the context menu: `Set as Inlet`, `Set as Outlet`, `Start Connection`, `Add Nozzle`, `Delete Node`.
    - [ ] The `Add Nozzle` option should only be available for outlet nodes.
  - [ ] **Phase 2: Intuitive Connection Workflow**
    - [ ] Implement a "drawing mode" for creating connections (channels) between nodes.
    - [ ] When connecting two nodes, a dialog should appear to define the channel's properties.
  - [ ] **Phase 3: Component Placement**
    - [ ] Implement adding nozzles to outlet nodes via the context menu.
    - [ ] A dialog should appear to define the nozzle's properties.
    - [ ] The visual representation of the node should change to indicate it has a nozzle.
  - [ ] **Phase 4: Properties and Simulation**
    - [ ] Ensure the properties editor correctly displays and saves properties for all elements.
    - [ ] Ensure the simulation runs correctly with the new network building logic.
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
