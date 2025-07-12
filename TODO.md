# Task List / Backlog

This list has been updated to prioritize critical solver fixes, code quality improvements, and future enhancements.

## High Priority: Critical Solver & Physics Fixes

- [x] **Fix `RobustNonLinearSolver` Jacobian Formulation:**

  - **Issue:** The current implementation creates a non-square Jacobian matrix, which is mathematically incorrect for the Newton-Raphson method.
  - **Fix:** Modify `_evaluate_residual` and `_build_jacobian` to use `N-1` mass conservation equations, ensuring a square and solvable system. Replace the `lsqr` solver with the more appropriate `spsolve`.

- [ ] **Centralize and Correct Viscosity Calculation:**

  - **Issue:** `SolverBase` uses a hardcoded placeholder viscosity, making `RobustNonLinearSolver` results incorrect. `NodalMatrixSolver` uses a separate, hardcoded internal table. This violates DRY and leads to incorrect physics.
  - **Fix:** Create a single, robust viscosity calculation function in a central utility module. All solvers **must** call this function to get fluid properties. This ensures consistent and correct physics across the application.

- [ ] **Improve `RobustNonLinearSolver` Pressure Calculation:**
  - **Issue:** The current BFS-based pressure calculation is susceptible to inaccuracies in networks with loops.
  - **Fix:** After solving for flows, formulate and solve a separate linear system for all node pressures simultaneously to ensure global consistency.

## Medium Priority: Code Quality and Refactoring

- [ ] **Unify `print_results` Method in `SolverBase`:**

  - **Issue:** `print_results` is duplicated across both solvers, making maintenance difficult. The method in `SolverBase` is abstract.
  - **Fix:** Implement the `print_results` method fully in the `SolverBase` class. Remove the duplicate implementations from the subclasses.

- [ ] **Externalize Fluid Data:**

  - **Issue:** Viscosity parameters are hardcoded.
  - **Fix:** Move all fluid property data to an external configuration file (e.g., `fluids.yaml`) and have the centralized viscosity function load it at runtime.

- [ ] **Improve Flow Initialization:**

  - **Issue:** Both solvers use naive or overly complex initial flow guesses.
  - **Fix:** Implement a consistent, topology-aware flow initialization method. A good approach is to use a single linear solve with estimated resistances.

- [ ] **Refine `NodalMatrixSolver` Resistance Calculation:**

  - **Issue:** The finite-difference step (`delta_q`) is too large, potentially leading to inaccurate resistance values and slower convergence.
  - **Fix:** Reduce the relative step size in `_calculate_component_resistance` from `1e-3` to `1e-6`.

- [ ] **Standardize Configuration:**
  - **Issue:** Solver parameters are passed inconsistently.
  - **Fix:** Enforce that all solver settings are managed exclusively through the `SolverConfig` object.

## Low Priority / Future Enhancements

- [ ] **Component Library Expansion:**

  - [ ] Implement `Pump` component with PQ-curve modeling.
  - [ ] Implement `ThermalExchanger` and `VariableValve` components.

- [ ] **Advanced Physics Models:**

  - [ ] Add support for transient simulations.
  - [ ] Add support for compressible flow.
  - [ ] Re-evaluate temperature-dependent viscosity during solver iterations.

- [ ] **Advanced Solver Features:**

  - [ ] Implement Trust Region methods as an alternative to line search.
  - [ ] Implement Broyden's method for cheaper Jacobian updates.

- [ ] **GUI Improvements:**
  - [ ] Refactor the GUI to use the unified solver interface.
  - [ ] Stabilize the GUI data model and fix interaction bugs.
