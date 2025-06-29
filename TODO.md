# Task List / Backlog

## High Priority / Next Steps

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

## Low Priority / Future Enhancements

- **Validation & Calibration:**
  - [ ] Develop a strategy for validating solver accuracy against real-world data or established benchmarks (e.g., for the specified pressure/flow/temperature ranges).
  - [ ] Implement automated tests to verify accuracy against known analytical solutions or experimental data for various network configurations and parameter ranges.
- **Component Library Expansion:**
  - [ ] Implement a `Valve` component with adjustable opening/closing settings.
  - [ ] Implement an `Accumulator` component.