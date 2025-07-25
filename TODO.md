# TODO.md

## GUI Refactoring Plan

### Core Functionality

- [ ] **Integrate Configuration Management:**
  - [ ] Add menu bar with "File" -> "Open", "Save", "Save As...".
  - [ ] Use `NetworkConfigLoader` to load JSON/XML files.
  - [ ] Use `NetworkConfigSaver` to save network and simulation state.
- [ ] **Refactor Simulation Workflow:**
  - [ ] Create `SimulationController` to decouple logic from UI.
  - [ ] Add a solver selection dropdown to the sidebar.
  - [ ] Use the unified `solver.solve(network)` method.
- [ ] **Display Results:**
  - [ ] Show numerical results in a text area.
  - [ ] Visualize results (pressures/flows) on the network graph.

### UI/UX Improvements

- [ ] **Refactor UI Components:**
  - [ ] `app.py`: Main window with menu and layout orchestration.
  - [ ] `canvas.py`: Responsible for rendering the network graph.
  - [ ] `sidebar.py`: Controls for simulation, solver selection, and results.
  - [ ] `dialogs.py`: Property editing for network components.
- [ ] **Dynamic Content:**
  - [ ] Populate simulation settings from the loaded `SimulationConfig`.
  - [ ] Update the network view when a new configuration is loaded.

### Advanced Features (Future)

- [ ] **Interactive Network Editing:**
  - [ ] Add/delete nodes and components directly on the canvas.
  - [ ] Edit component properties through dialogs.
  - [ ] Ensure changes are reflected in the underlying `FlowNetwork` object.

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
