# TreeSolver Refactoring Plan

## Phase 1: Critical Boundary Condition Fixes (Immediate Priority)

- [ ] **Fix Boundary Condition Setup**

  - Current: Inlet pressure treated inconsistently (sometimes fixed, sometimes unknown)
  - Problem: Tests expect inlet pressure to be solved for, not fixed
  - Location: `nonlinear_tree_solver.py:57-58, 166-167`
  - Action: Ensure inlet node is included in unknowns, solve for inlet pressure

- [ ] **Fix Reference Pressure Inconsistency**

  - Reference pressure is `0.0` in residual, but `outlet_pressure` used elsewhere
  - Location: `nonlinear_tree_solver.py:167,180`
  - Action: Use consistent reference pressure throughout

- [ ] **Fix Flow Rate Constraint Enforcement**

  - Current: Flow constraint added at inlet but may not be properly enforced
  - Problem: Mass conservation violations (98.8% error)
  - Location: `nonlinear_tree_solver.py:200-201`
  - Action: Verify flow constraint is correctly applied in residual calculation

- [ ] **Fix Final Solution Assembly**
  - Current: Inlet pressure may not be included in final solution
  - Problem: Boundary condition tests fail with large errors
  - Location: `nonlinear_tree_solver.py:164-171`
  - Action: Ensure all solved pressures (including inlet) are included in results

## Phase 2: Solver Robustness Improvements (High Priority)

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

## Phase 3: Convergence Enhancements (Medium Priority)

- [ ] **Jacobian Conditioning and Monitoring**

  - Monitor Jacobian condition number
  - Add adaptive regularization and pivoting

- [ ] **Advanced Line Search Methods**

  - Add trust-region fallback
  - Improve step-size selection for difficult problems

- [ ] **Enhanced Convergence Criteria**
  - Use multiple criteria (residual, update norm, gradient)
  - Implement adaptive tolerances and early termination

## Phase 4: Code Quality and Maintainability (Low Priority)

- [ ] **Code Clarity and Refactoring**

  - Split large methods, improve naming and documentation
  - Add error handling and logging

- [ ] **Comprehensive Testing**
  - Edge cases, pathological networks, performance benchmarks
  - Regression and stress tests

## Immediate Next Steps

- [ ] Fix boundary condition inconsistency
- [ ] Fix Jacobian singularity handling
- [ ] Make logging configuration robust
- [ ] Implement stagnation detection and pressure bounds handling
- [ ] Add topology validation for tree networks
