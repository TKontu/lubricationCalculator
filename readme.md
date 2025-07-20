# Lubrication Flow Network Calculator

This project is a Python-based engineering tool for simulating and analyzing steady-state fluid flow in hydraulic lubrication networks. It allows users to define complex circuits, model component pressure losses, and solve for pressures and flow rates using several numerical methods.

The primary input is a network definition (from a JSON or XML file) and a set of simulation parameters, such as total system flow rate and fluid properties. The tool calculates the pressure at each node and the flow rate through each component, with a key focus on the flow distribution to the various outlet points of the network.

## Key Features

- **Flexible Network Modeling:** Construct complex hydraulic networks with components like pipes, nozzles, and fittings.
- **Multiple Solvers:**
    - **`nodal`:** A linear solver for rapid estimation.
    - **`tree_nonlinear`:** A robust and accurate non-linear solver for tree-like (radial) networks using the Newton-Raphson method.
    - **`robust_newton`:** A non-linear solver capable of handling networks with loops.
- **Configurable Physics:**
    - Accurately models non-linear pressure-flow relationships.
    - Calculates temperature-dependent fluid viscosity for various oil types (e.g., SAE30, VG460).
- **Data-Driven Configuration:** Define networks and simulation parameters using simple JSON or XML files.
- **Command-Line Interface:** A full-featured CLI for creating, validating, and simulating networks.
- **Extensible Design:** The modular architecture (Builder pattern, Strategy pattern for solvers) makes it easy to add new components or solution algorithms.

## Project Architecture

- `lubrication_flow_package/`: The core Python package.
  - `cli/`: Command-line interface for user interaction.
  - `config/`: Handles loading and saving of network and simulation configurations.
  - `components/`: Defines hydraulic components (`Channel`, `Nozzle`, `Connector`).
  - `network/`: Core data structures for the flow network (`FlowNetwork`, `Node`).
  - `solvers/`: Implements the various numerical solvers.
  - `utils/`: Provides utilities like the `NetworkBuilder` and fluid property calculators.
- `examples/`: Example network definition files.
- `tests/`: Unit and integration tests.
- `main.py`: The main entry point for the CLI application.

## Workflow Overview

The simulation process follows a clear, three-step architectural pattern:

1.  **Configuration Loading:** When a simulation is initiated, the `NetworkConfigLoader` class in the `config` module reads the specified JSON or XML file. It parses the data into a structured `NetworkConfig` object, which serves as a standardized, in-memory representation of the entire network definition.

2.  **Network Building:** The `NetworkConfig` object is then passed to the `NetworkBuilder` utility. Following the **Builder design pattern**, this class provides a clean API to construct the final `FlowNetwork` object. It translates the lists of nodes, components, and connections from the configuration into a graph of instantiated `Node` and `Component` objects, ensuring the network is assembled correctly.

3.  **Solving:** The constructed `FlowNetwork` is passed to the selected solver (e.g., `TreeSolver`). The solvers use numerical methods (specifically, the **Newton-Raphson method** for non-linear systems) to iteratively solve for the pressures at each node and the flow through each component. The iteration continues until the net flow at each internal node is balanced (conserving mass) to within a defined tolerance.

## Getting Started

### Installation

1.  Clone the repository.
2.  It is recommended to create a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The application is controlled via the command-line interface.

### 1. Create a Network Template

To get started, generate a template network configuration file.

```bash
python main.py network template -o my_network.json
```
This will create a `my_network.json` file that you can customize.

### 2. Simulate a Network

Run a simulation using your configuration file and a chosen solver.

```bash
# Simulate with the default non-linear tree solver
python main.py network simulate examples/simple_branch.json --solver tree_nonlinear

# Simulate a more complex network with the robust Newton solver
python main.py network simulate examples/complex_network.json --solver robust_newton
```

### 3. Validate a Network

You can check a configuration file for structural integrity and completeness without running a full simulation.

```bash
python main.py network validate examples/simple_branch.json
```

### Running Tests

To run the suite of unit tests, use pytest:

```bash
pytest
```