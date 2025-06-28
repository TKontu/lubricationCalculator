# Task List / Backlog

## High Priority / Next Steps

- **Improve CLI:**
  - [ ] Add command-line arguments to specify the network file to solve.
  - [ ] Add options to control output verbosity.
  - [ ] Add a command to validate a network file without running a full simulation.
  - [ ] Enhance CLI output to clearly present output flow distribution at end points and pressure losses across components.
- **Solver Enhancements:**
  - [ ] Add support for user-defined fluid properties in the configuration file.
  - [ ] Implement a more sophisticated convergence check to handle stalling.

## Medium Priority

- **Graphical User Interface (GUI):**
  - [ ] Develop a simple GUI for building and visualizing networks.
  - [ ] Add plotting capabilities to visualize pressure and flow distribution.
- **Web-Based Interface:**
  - [ ] Create a web-based version of the tool using a framework like Flask or Django.

## Low Priority / Future Enhancements

- **Validation & Calibration:**
  - [ ] Develop a strategy for validating solver accuracy against real-world data or established benchmarks (e.g., for the specified pressure/flow/temperature ranges).
  - [ ] Implement automated tests to verify accuracy against known analytical solutions or experimental data for various network configurations and parameter ranges.
- **Component Library Expansion:**
  - [ ] Implement a `Valve` component with adjustable opening/closing settings.
  - [ ] Implement an `Accumulator` component.
