# TODO.md

## GUI Development

### Core Functionality
- [ ] **Interactive Network Building:**
    - [ ] Implement features to build a flow network from scratch within the GUI.
    - [ ] Add/delete nodes and components directly on the canvas.
    - [ ] Connect components to build the network graph visually.
- [ ] **Full Element Configuration:**
    - [ ] Allow editing of all properties for nodes, elements, and components (e.g., name, connections, pipe size, length, elevation, x/y position).
    - [ ] Implement dialogs or property editors for each element type.
- [ ] **Configurable Units:**
    - [ ] Add options in the GUI to select input and output units (e.g., pressure in Pa/psi, flow in m³/s/gpm).
    - [ ] Ensure all displayed values and inputs respect the selected units.
- [ ] **Complete Simulation Settings:**
    - [ ] Make all simulation settings from `SimulationConfig` configurable in the GUI.
- [ ] **Display Results:**
    - [ ] Show numerical results in a text area.
    - [ ] Visualize results (pressures/flows) on the network graph.

### UI/UX Improvements
- [ ] **Improved Edge Rendering:**
    - [ ] Update plot drawing so that edges do not overlap unless they are crossing.
- [ ] **Refactor UI Components:**
    - [ ] `app.py`: Main window with menu and layout orchestration.
    - [ ] `canvas.py`: Responsible for rendering the network graph.
    - [ ] `sidebar.py`: Controls for simulation, solver selection, and results.
    - [ ] `dialogs.py`: Property editing for network components.
- [ ] **Dynamic Content:**
    - [ ] Populate simulation settings from the loaded `SimulationConfig`.
    - [ ] Update the network view when a new configuration is loaded.

## TreeSolver Refactoring Plan

### Phase 1: Solver Robustness Improvements (High Priority)

- [ ] **Adaptive Stagnation Detection**

  - Fixed threshold (1e-9) not suitable for all scales
  - Action: Scale convergence thresholds based on problem magnitude

- [ ] **Improved Pressure Bounds Handling**

  - Pressure clamping interferes with convergence
  - Location: `nonlinear_tree_solver.py:143,280`
  - Action: Use penalty methods or barrier functions

- [ ] **Network Topology Validation**
  - No check for tree-like structure
  - Action: Add cycle detection and validate topology

### Phase 2: Convergence Enhancements (Medium Priority)

- [ ] **Jacobian Conditioning and Monitoring**

  - Monitor Jacobian condition number
  - Add adaptive regularization and pivoting

- [ ] **Advanced Line Search Methods**

  - Add trust-region fallback
  - Improve step-size selection for difficult problems

- [ ] **Enhanced Convergence Criteria**
  - Implement adaptive tolerances and early termination

### Phase 3: Code Quality and Maintainability (Low Priority)

- [ ] **Code Clarity and Refactoring**

  - Split large methods, improve naming and documentation
  - Add error handling and logging

- [ ] **Comprehensive Testing**
  - Edge cases, pathological networks, performance benchmarks
  - Regression and stress tests