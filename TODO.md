# Task List / Backlog

This list has been refactored to prioritize the unification of the solver APIs and streamline future development.

## High Priority: Solver API Refactoring

The immediate goal is to refactor all solvers to conform to a single, unified interface. This will eliminate technical debt, simplify the CLI, and make the system more maintainable and extensible.

- [ ] **1. Define a `SolverBase` Abstract Class:**
  - [ ] Create a new file `lubrication_flow_package/solvers/base.py`.
  - [ ] Define an abstract base class `SolverBase` with the following methods:
    - `__init__(self, sim_config: SimulationConfig, solver_config: Optional[SolverConfig] = None)`
    - `solve(self, network: FlowNetwork) -> Dict`
    - `print_results(self, ...)`
    - `get_default_solver_config(self) -> SolverConfig`

- [ ] **2. Refactor `NodalMatrixSolver` to Conform to `SolverBase`:**
  - [ ] Modify `NodalMatrixSolver` to inherit from `SolverBase`.
  - [ ] Update its `__init__` method to accept `(sim_config, solver_config)`.
  - [ ] Create a public `solve()` method that matches the base class interface.
  - [ ] Move the existing solve logic into a private method (e.g., `_solve_nodal_network(...)`).
  - [ ] Adapt the return value of the private method to the standardized results dictionary format.

- [ ] **3. Refactor `RobustNonLinearSolver` to Conform to `SolverBase`:**
  - [ ] Modify `RobustNonLinearSolver` to inherit from `SolverBase`.
  - [ ] Update its `__init__` method to accept `(sim_config, solver_config)`.
  - [ ] Ensure its `solve()` method and return value already match the interface.

- [ ] **4. Simplify the CLI (`network_cli.py`):**
  - [ ] Remove the complex `if/else` block for solver selection.
  - [ ] Implement a simple factory pattern to choose the correct solver class (`RobustNonLinearSolver` or `NodalMatrixSolver`).
  - [ ] Instantiate and call the chosen solver using the single, unified interface.

- [ ] **5. Standardize Configuration Handling:**
  - [ ] Ensure both solvers can be initialized with a default `SolverConfig` or a user-provided one.
  - [ ] Update the CLI to properly load and pass the `--solver-config` file to the chosen solver.

## Medium Priority: Post-Refactoring Improvements

Once the solver API is unified, we can focus on improving the underlying physics and adding features in a solver-agnostic way.

- [ ] **Enhanced Physics & Validation:**
  - [ ] **Fix Connector Physics:** URGENT - Current reducer/expander calculations are incorrect. Validate against hydraulic handbooks.
  - [ ] **Temperature-Dependent Viscosity:** Re-evaluate viscosity during solver iterations for better accuracy in systems with significant temperature changes.
  - [ ] **Comprehensive Validation Suite:** Expand the test suite to include more complex networks and analytical benchmarks to validate the physics of both solvers.

- [ ] **Advanced Solver Features (for `RobustNonLinearSolver`):**
  - [ ] Implement Trust Region methods as an alternative to line search for improved global convergence.
  - [ ] Implement Broyden's method for cheaper Jacobian updates in large networks.

- [ ] **GUI Improvements:**
  - [ ] Refactor the GUI to use the new unified solver interface.
  - [ ] Stabilize the GUI data model and fix interaction bugs.

## Low Priority / Future Enhancements

- [ ] **Component Library Expansion:**
  - [ ] Implement `Pump` component with PQ-curve modeling.
  - [ ] Implement `ThermalExchanger` and `VariableValve` components.

- [ ] **Advanced Physics Models:**
  - [ ] Add support for transient simulations and compressible flow.