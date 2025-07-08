# Task List / Backlog

## High Priority / Next Steps

- **Solver Accuracy and Robustness:**

  - [ ] **Fix Critical Physics Errors in Solver:** Address fundamental flaws in nodal solver implementation:
    - [x] **Hydrostatic Flow Calculation:** Correct flow computation to include hydrostatic effects: `Q = G * [(P_i - P_j) - ρgΔz]` instead of `Q = G * (P_i - P_j)`
    - [ ] **Resistance Linearization:** Replace average resistance (ΔP/Q) with differential resistance (dΔP/dQ) using central differencing in all solver iterations
    - [ ] **Pressure-Flow Validation:** Update convergence check to include hydrostatic component in pressure-drop validation
    - [ ] **Initialization Physics:** Replace BFS pathfinding with linearized system solve for flow initialization
  - [ ] **Refactor Resistance Calculation:**
    - [ ] Remove `_compute_resistance()` and exclusively use `_calculate_component_resistance()`
    - [ ] Add temperature-dependent viscosity re-evaluation during iterations

- **GUI Data Model Refactoring (Robustness):**

  - [ ] **Phase 1: Core Logic and Data Structure**
  - [ ] **Phase 2: Visualization**
  - [ ] **Phase 3: UI and Cleanup**

- **Graphical User Interface (GUI) - Polish and Refine:**

  - [ ] **Phase 1: Canvas and Usability**
  - [ ] **Phase 2: Code Refactoring and Robustness**
  - [ ] **Phase 3: Advanced Features**

- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django.

## Medium Priority

- **Improve Connector Physics:**
  - [ ] The pressure drop calculation in `connector.py` for reducers and expanders may be inaccurate. It currently uses the inlet diameter for velocity calculation in all cases.
    - [ ] **Action:** Review and correct the `calculate_pressure_drop` method in `connector.py` to use the appropriate diameter (inlet or outlet) for velocity calculation based on the specific connector type (e.g., expansion vs. contraction).
- **Enhance Solver Diagnostics:**
  - [ ] Add detailed convergence metrics logging for each iteration
  - [ ] Implement component-level physics validation hooks
  - [ ] Create network connectivity checks for disconnected components

## Low Priority / Future Enhancements

- **Validation & Calibration:**
  - [ ] Develop validation suite against analytical solutions for:
    - [ ] Hydrostatic-dominated networks
    - [ ] Non-linear turbulent flow regimes
    - [ ] Mixed elevation networks
  - [ ] Create calibration framework for experimental data
- **Advanced Physics Models:**
  - [ ] Implement temperature gradient support across components
  - [ ] Add transient simulation capabilities
- **Component Library Expansion:**
  - [ ] Implement `Pump` component with PQ-curve modeling
  - [ ] Implement `ThermalExchanger` component
  - [ ] Create `VariableValve` with adjustable K-values
