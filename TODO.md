### Task List / Backlog

#### High Priority / Next Steps

- **Configuration Management:**
  - [ ] Move `SolverConfig` parameters to a configuration file (e.g., `config.yaml`).
  - [ ] Load solver configuration from the file instead of using hardcoded defaults.
- **Improve CLI:**
  - [ ] Add command-line arguments to specify the network file to solve.
  - [ ] Add options to control output verbosity.
  - [ ] Add a command to validate a network file without running a full simulation.

#### Medium Priority

- **Component Library Expansion:**
  - [ ] Implement a `Pump` component with a user-definable pump curve.
- **Solver Enhancements:**
  - [ ] Add support for user-defined fluid properties in the configuration file.
  - [ ] Implement a more sophisticated convergence check to handle stalling.

#### Low Priority / Future Enhancements

- **Graphical User Interface (GUI):**
  - [ ] Develop a simple GUI for building and visualizing networks.
  - [ ] Add plotting capabilities to visualize pressure and flow distribution.
- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django.
