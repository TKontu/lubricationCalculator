# Lubrication Flow Network Calculator

This project is a Python-based tool for simulating and analyzing fluid flow in lubrication networks. It allows users to build complex hydraulic circuits, define component properties, and solve for steady-state pressures and flow rates using a robust nodal-matrix solver.

Primarily it is a tool in which a specific network is built and simulated.
The input is always the network structure and geometry and a fixed input flow volume.

Output of the simulation should be pressure at each node: input and all other nodes of the system.
Pressure loss at different parts of the system is of high interest:
Most important result is the output flow distribution at different end points of the tree-like branching network with nozzles at ends of each branch path.
Target is to achieve relatively good accuracy in the pressure and flow distribution approximations for a given network.

Parameters:
Pressure range 0...24 bar. Typically < 4 bar
Flow rate 0...400 l/min
Temperature 60 degC
Oil VG220 / VG320 typically
Typically the network should operate in laminar regime
Nozzles of course turbulentic.

Target precision +/- 1 l/min for end points, +/- 0,3 bar compared to measurement of a real physical system.

## Key Features

- **Network Modeling:** Construct complex lubrication networks with components like pipes, nozzles, and junctions.
- **Nodal-Matrix Solver:** A powerful iterative solver that calculates node pressures and edge flows while conserving mass.
- **Non-Linear Components:** Accurately models non-linear pressure-flow relationships in hydraulic components.
- **Fluid Properties:** Calculates fluid viscosity based on oil type (e.g., SAE30, VG460) and temperature.
- **Command-Line Interface:** Provides a CLI for running simulations and analyzing network behavior.
- **Extensible:** The modular design allows for the addition of new component types and solver configurations.

## Project Structure

- `lubrication_flow_package/`: The core Python package.
  - `components/`: Defines hydraulic components (e.g., `Channel`, `Nozzle`).
  - `network/`: Contains the `FlowNetwork` class for building circuits.
  - `solvers/`: Implements the `NodalMatrixSolver`.
  - `cli/`: The command-line interface.
- `examples/`: JSON files defining example networks.
- `tests/`: Unit tests for the solver and components.
- `main.py`: The main entry point for running the application.

## Getting Started

### Installation

1.  Clone the repository.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running a Simulation

You can run a simulation using the main script, which will solve the example networks:

```bash
python main.py
```

### Running Tests

To run the unit tests, use pytest:

```bash
pytest
```
