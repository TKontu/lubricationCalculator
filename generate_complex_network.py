"""
This script generates a complex lubrication network configuration file
(complex_network.json) that can be used with the project's CLI.
"""

from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.config.network_config import NetworkConfigSaver
from lubrication_flow_package.components.base import NozzleType, ConnectorType

def generate_complex_network_config():
    """
    Builds the complex network and returns it as a NetworkConfig object.
    """
    # --- 1. Unit Conversions and Constants ---
    MAIN_TRUNK_DIAMETER_M = 42 / 1000  # 42mm to m
    TOTAL_FLOW_RATE_M3S = 200 / 60000  # 200 L/min to m³/s

    builder = NetworkBuilder()

    # --- 2. Define the Main Trunk and Junctions ---
    builder.add_node("inlet", elevation=5.0)
    builder.add_pipe("inlet", "J1_node", length=5.0, diameter=MAIN_TRUNK_DIAMETER_M, to_node_elevation=4.9)

    # Create 5 cross-junctions
    for i in range(1, 6):
        junction_node = f"J{i}_node"
        next_junction_node = f"J{i+1}_node"
        
        # Connect the main trunk to the next junction
        builder.add_pipe(
            from_node_name=junction_node,
            to_node_name=next_junction_node,
            length=5.0,
            diameter=MAIN_TRUNK_DIAMETER_M,
            from_node_elevation=4.9 - ((i-1) * 0.1),
            to_node_elevation=4.9 - (i * 0.1)
        )

        # --- 3. Create Two Unique Diverging Paths at Each Junction ---
        
        # Path A
        path_a_prefix = f"path_{i}a"
        branch_a_dia_m = (30 - i * 2) / 1000
        nozzle_a_dia_m = (2 + i * 0.4) / 1000

        builder.add_pipe(junction_node, f"{path_a_prefix}_start", length=1.0, diameter=branch_a_dia_m, to_node_elevation=4.0)
        builder.add_fitting(f"{path_a_prefix}_start", f"{path_a_prefix}_bend_out", ConnectorType.ELBOW_45, diameter=branch_a_dia_m, name=f"{path_a_prefix}_bend")
        builder.add_pipe(f"{path_a_prefix}_bend_out", f"{path_a_prefix}_nozzle_in", length=2.0, diameter=branch_a_dia_m, name=f"{path_a_prefix}_drilling")
        builder.add_nozzle(f"{path_a_prefix}_nozzle_in", f"{path_a_prefix}_out", diameter=nozzle_a_dia_m, nozzle_type=NozzleType.ROUNDED)
        builder.add_outlet(f"{path_a_prefix}_out", elevation=2.0)

        # Path B
        path_b_prefix = f"path_{i}b"
        branch_b_dia_m = (28 - i * 2) / 1000
        nozzle_b_dia_m = (4 + i * 0.4) / 1000

        builder.add_pipe(junction_node, f"{path_b_prefix}_start", length=1.2, diameter=branch_b_dia_m, to_node_elevation=3.8)
        builder.add_fitting(f"{path_b_prefix}_start", f"{path_b_prefix}_bend_out", ConnectorType.ELBOW_90, diameter=branch_b_dia_m, name=f"{path_b_prefix}_bend")
        builder.add_pipe(f"{path_b_prefix}_bend_out", f"{path_b_prefix}_nozzle_in", length=2.5, diameter=branch_b_dia_m, name=f"{path_b_prefix}_drilling")
        builder.add_nozzle(f"{path_b_prefix}_nozzle_in", f"{path_b_prefix}_out", diameter=nozzle_b_dia_m, nozzle_type=NozzleType.SHARP_EDGED)
        builder.add_outlet(f"{path_b_prefix}_out", elevation=1.8)

    # --- 4. Finalize the Network ---
    builder.add_outlet("J6_node")
    builder.set_inlet("inlet")

    network = builder.build()
    
    # --- 5. Create Simulation and Network Configuration ---
    sim_config = SimulationConfig(
        total_flow_rate=TOTAL_FLOW_RATE_M3S,
        temperature=50.0,
        inlet_pressure=500000.0,
        oil_type="ISO_VG_46",
        output_pressure_unit="bar",
        output_flow_rate_unit="l/min"
    )
    
    network_config = NetworkConfigSaver.from_network(network, sim_config)
    network_config.description = "Complex network with 5 cross-junctions and 10 unique diverging paths."
    
    return network_config

def main():
    """
    Generates the network configuration and saves it to a JSON file.
    """
    print("Generating complex network configuration...")
    config = generate_complex_network_config()
    output_file = "complex_network.json"
    
    NetworkConfigSaver.save_json(config, output_file)
    print(f"Network configuration saved to: {output_file}")
    print("\nYou can now simulate this network using the CLI.")

if __name__ == "__main__":
    main()
