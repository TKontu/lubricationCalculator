# Task List / Backlog

## High Priority / Next Steps

- **Solver Accuracy and Robustness:**

  - [ ] **Fix Critical Physics Errors in Solver:** Address fundamental flaws in nodal solver implementation:
    - [x] **Hydrostatic Flow Calculation:** Correct flow computation to include hydrostatic effects: `Q = G * [(P_i - P_j) - ρgΔz]` instead of `Q = G * (P_i - P_j)`
    - [ ] **Resistance Linearization:** Replace average resistance (ΔP/Q) with differential resistance (dΔP/dQ) using central differencing in all solver iterations
      - [x] Main solver loop uses differential resistance  
      - [ ] CRITICAL: Flow initialization still uses average resistance (_compute_resistance)
    - [ ] **Pressure-Flow Validation:** Update convergence check to include hydrostatic component in pressure-drop validation
    - [ ] **Initialization Physics:** Replace BFS pathfinding with linearized system solve for flow initialization
  - [ ] **Refactor Resistance Calculation:**
    - [ ] Remove `_compute_resistance()` and exclusively use `_calculate_component_resistance()`
    - [ ] Add temperature-dependent viscosity re-evaluation during iterations

- **Solver Performance and Convergence Improvements:**

  - [ ] **Adaptive Relaxation Factor:** Implement dynamic relaxation based on convergence behavior
    - [ ] Add oscillation detection to reduce relaxation when flow changes oscillate
    - [ ] Increase relaxation factor when convergence is steady
    - [ ] Implement adaptive relaxation bounds (0.1 to 0.9)
  - [ ] **Enhanced Initial Flow Estimation:** Improve flow initialization using electrical analogy
    - [ ] Build resistance network using NetworkX
    - [ ] Solve equivalent resistor network for better initial guess
    - [ ] Use path conductance weighting instead of simple distribution
  - [ ] **Multi-Level Convergence Criteria:** Replace single tolerance with adaptive criteria
    - [ ] Implement primary (strict) and secondary (relaxed) convergence thresholds
    - [ ] Add stagnation detection for oscillating solutions
    - [ ] Include iteration-dependent tolerance relaxation
  - [ ] **Differential Resistance Caching:** Cache expensive resistance calculations
    - [ ] Implement resistance cache with flow-rate/viscosity keys
    - [ ] Add cache size limits and cleanup
    - [ ] Use interpolation for nearby flow rates
  - [ ] **Newton-Raphson Hybrid Approach:** Add N-R acceleration for difficult networks
    - [ ] Implement Jacobian matrix construction for non-linear system
    - [ ] Add second derivative calculation for components
    - [ ] Use N-R steps with fallback to current method

- **Critical Component Physics Fixes:**

  - [ ] **Fix Connector Physics:** URGENT - Current reducer/expander calculations are incorrect
    - [ ] Fix velocity calculation to use appropriate diameter (inlet vs outlet) based on connector type
    - [ ] Implement proper area ratio calculations for expansions vs contractions
    - [ ] Add Reynolds number dependency for loss coefficients
    - [ ] Validate against known hydraulic handbook values

- **Validation and Testing Framework:**

  - [ ] **Comprehensive Physics Validation:** Build robust test suite before adding features
    - [ ] Create analytical solution benchmarks for simple networks
    - [ ] Implement component-level unit tests against published data
    - [ ] Add regression tests for solver convergence behavior
    - [ ] Validate pressure-flow relationships across Reynolds number ranges

## Medium Priority

- **GUI Stabilization and Improvement:**

  - [ ] **GUI Data Model Refactoring:** Improve robustness before adding features
    - [ ] **Phase 1: Core Logic and Data Structure** - Fix data model inconsistencies
    - [ ] **Phase 2: Visualization** - Improve canvas rendering and performance
    - [ ] **Phase 3: UI and Cleanup** - Polish user experience

  - [ ] **Graphical User Interface Enhancements:**
    - [ ] **Phase 1: Canvas and Usability** - Fix interaction bugs and improve workflow
    - [ ] **Phase 2: Code Refactoring and Robustness** - Clean up codebase architecture
    - [ ] **Phase 3: Advanced Features** - Add advanced simulation features

- **Enhanced Solver Diagnostics:**
  - [ ] Add detailed convergence metrics logging for each iteration
  - [ ] Implement component-level physics validation hooks
  - [ ] Create network connectivity checks for disconnected components
  - [ ] Add performance profiling for solver bottlenecks
  - [ ] Implement convergence history visualization

## Low Priority / Future Enhancements

- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django
  - [ ] Only pursue after GUI is stable and core solver is robust

- **Advanced Physics Models:**
  - [ ] Implement temperature gradient support across components
  - [ ] Add transient simulation capabilities
  - [ ] Support for compressible flow in high-pressure systems

- **Component Library Expansion:**
  - [ ] Implement `Pump` component with PQ-curve modeling
  - [ ] Implement `ThermalExchanger` component
  - [ ] Create `VariableValve` with adjustable K-values
  - [ ] Add `FlowMeter` components for monitoring

- **Advanced Validation & Calibration:**
  - [ ] Create calibration framework for experimental data
  - [ ] Develop uncertainty quantification methods
  - [ ] Implement parameter sensitivity analysis

## Implementation Priority Order

**Phase 1: Critical Fixes (Immediate - 1-2 weeks)**
1. Fix Connector Physics - URGENT: Incorrect reducer/expander calculations affecting accuracy
2. Complete Resistance Linearization - CRITICAL: Flow initialization still uses wrong resistance formula
3. Adaptive Relaxation Factor - Quick win, significant stability improvement  
4. Enhanced Convergence Criteria - Prevents premature termination/oscillation
5. Pressure-Flow Validation - Completes hydrostatic implementation

**Phase 2: Validation & Robustness (Short-term - 2-4 weeks)**
5. Comprehensive Physics Validation - Test suite against analytical solutions
6. Component-level unit tests - Validate against published hydraulic data
7. Refactor Resistance Calculation - Clean up solver architecture
8. Solver Diagnostics - Better debugging and monitoring

**Phase 3: Performance Optimizations (Medium-term - 1-2 months)**
9. Differential Resistance Caching - Performance boost for large networks
10. Enhanced Initial Flow Estimation - Faster convergence startup
11. Newton-Raphson Hybrid - Advanced acceleration for difficult cases
12. Initialization Physics - Replace BFS with proper linear solve

**Phase 4: User Interface (Long-term - 2-3 months)**
13. GUI Stabilization - Fix data model and interaction bugs
14. GUI Enhancements - Improve usability and features
15. Web Interface - Only after core solver and GUI are stable
