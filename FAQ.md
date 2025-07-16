# Answers

- **How do nodes and components relate to each other?**

  - Nodes are the connection points in the network, defined by the `Node` class. They have properties like pressure and elevation.
  - Components (e.g., `Channel`, `Nozzle`, `Connector`) are the physical elements that connect two nodes. They are responsible for calculating pressure drops based on flow.
  - A `Connection` object represents the link between a `from_node`, a `to_node`, and the `component` that connects them.
  - The `NetworkBuilder` class is used to assemble the network by adding nodes and then connecting them with components like pipes or fittings.

- **What nodes exist for a T-fitting?**

  - A T-fitting is modeled using four nodes: a central junction node (`tee_node_name`) and three connecting nodes (`main_in`, `main_out`, and `branch_out`). The `add_tee_junction` method in the `NetworkBuilder` handles the creation of the necessary internal connections between these nodes.

- **What attributes can be used in network.json files, and can the id also be self-defined?**

  - The network structure is typically loaded from a JSON file (or a dictionary). The main keys are `nodes`, `inlet`, `outlets`, and `components`.
  - **`nodes`**: A list of objects, each with a `name` and optional `elevation`.
  - **`components`**: A list of objects, each with a `type` (`pipe`, `nozzle`, `fitting`, `tee`) and other properties specific to that type (e.g., `length` and `diameter` for a `pipe`).
  - The `id` of nodes and components is not meant to be set directly from the JSON file. Instead, you use the `name` attribute to identify nodes and components. The system will automatically assign a unique `id` internally. You define the network using human-readable `name` attributes, and the builder handles the connections.

- **How are the configs defined?**
  - The configuration system uses a combination of Python classes and YAML files.
  - **Python Classes**: Pydantic `BaseModel` classes (e.g., `SimulationConfig`, `NetworkConfig`) define the structure and data types of the configuration, providing validation.
  - **YAML Files**: Files like `config.yaml` provide the actual values for the configuration. The structure of the YAML file (e.g., `simulation:`, `network:`) maps directly to the Python configuration classes. This allows you to easily switch between different simulation setups by using different YAML files without changing the code.
