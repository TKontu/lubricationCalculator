
"""
Builds and solves a complex lubrication network with multiple cross junctions
and unique diverging paths to demonstrate the capabilities of the NetworkBuilder
and the nonlinear solver.
"""

from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.components.base import NozzleType, ConnectorType
from lubrication_flow_package.solvers.nonlinear_solver import RobustNonLinearSolver

def create_complex_lubrication_network():
    """
    Builds and returns a complex lubrication network based on specified
    engineering requirements.
    """
    # --- 1. Unit Conversions and Constants ---
    # Convert all inputs to the required SI units (meters and m³/s)
    MAIN_TRUNK_DIAMETER_M = 42 / 1000  # 42mm to m
    TOTAL_FLOW_RATE_M3S = 200 / 60000  # 200 L/min to m³/s

    builder = NetworkBuilder()

    # --- 2. Define the Main Trunk and Junctions ---
    # The main path starts at a high elevation and drops.
    builder.add_node("inlet", elevation=5.0)
    builder.add_pipe("inlet", "J1_node", length=5.0, diameter=MAIN_TRUNK_DIAMETER_M, to_node_elevation=4.9)

    # Loop to create 5 cross-junctions
    for i in range(1, 6):
        junction_node = f"J{i}_node"
        next_junction_node = f"J{i+1}_node"
        
        # Connect the main trunk from the current junction to the next one
        builder.add_pipe(
            from_node_name=junction_node,
            to_node_name=next_junction_node,
            length=5.0,
            diameter=MAIN_TRUNK_DIAMETER_M,
            from_node_elevation=4.9 - ((i-1) * 0.1),
            to_node_elevation=4.9 - (i * 0.1)
        )

        # --- 3. Create Two Unique Diverging Paths at Each Junction ---
        
        # --- Path A ---
        path_a_prefix = f"path_{i}a"
        branch_a_start = f"{path_a_prefix}_start"
        branch_a_dia_m = (30 - i * 2) / 1000  # Unique diameter, smaller than main
        nozzle_a_dia_m = (2 + i * 0.4) / 1000  # Unique nozzle size between 2-4mm

        builder.add_pipe(junction_node, branch_a_start, length=1.0, diameter=branch_a_dia_m, to_node_elevation=4.0)
        builder.add_fitting(branch_a_start, f"{path_a_prefix}_bend_out", ConnectorType.ELBOW_45, diameter=branch_a_dia_m, name=f"{path_a_prefix}_bend")
        builder.add_pipe(f"{path_a_prefix}_bend_out", f"{path_a_prefix}_nozzle_in", length=2.0, diameter=branch_a_dia_m, name=f"{path_a_prefix}_drilling")
        builder.add_nozzle(f"{path_a_prefix}_nozzle_in", f"{path_a_prefix}_out", diameter=nozzle_a_dia_m, nozzle_type=NozzleType.ROUNDED)
        builder.add_outlet(f"{path_a_prefix}_out", elevation=2.0)

        # --- Path B ---
        path_b_prefix = f"path_{i}b"
        branch_b_start = f"{path_b_prefix}_start"
        branch_b_dia_m = (28 - i * 2) / 1000  # Different unique diameter
        nozzle_b_dia_m = (4 + i * 0.4) / 1000  # Unique nozzle size between 4-6mm

        builder.add_pipe(junction_node, branch_b_start, length=1.2, diameter=branch_b_dia_m, to_node_elevation=3.8)
        builder.add_fitting(branch_b_start, f"{path_b_prefix}_bend_out", ConnectorType.ELBOW_90, diameter=branch_b_dia_m, name=f"{path_b_prefix}_bend")
        builder.add_pipe(f"{path_b_prefix}_bend_out", f"{path_b_prefix}_nozzle_in", length=2.5, diameter=branch_b_dia_m, name=f"{path_b_prefix}_drilling")
        builder.add_nozzle(f"{path_b_prefix}_nozzle_in", f"{path_b_prefix}_out", diameter=nozzle_b_dia_m, nozzle_type=NozzleType.SHARP_EDGED)
        builder.add_outlet(f"{path_b_prefix}_out", elevation=1.8)

    # --- 4. Finalize the Network ---
    # The last node of the main trunk is also an outlet.
    builder.add_outlet("J6_node")
    
    # Set the main inlet for the entire system
    builder.set_inlet("inlet")

    print("✅ Complex network definition created.")
    return builder.build(), TOTAL_FLOW_RATE_M3S


def main():
    """
    Builds the network, runs the solver, and prints the results.
    """
    print("🔧 Building complex lubrication network...")
    network, flow_rate = create_complex_lubrication_network()
    
    print(f"\n🔬 Configuring simulation with a total flow rate of {flow_rate:.4f} m³/s (200 L/min)...")
    sim_config = SimulationConfig(
        total_flow_rate=flow_rate,
        temperature=50.0,
        inlet_pressure=500000.0,  # 5 bar initial guess
        oil_type="ISO_VG_46"
    )

    print("\n⚙️  Initializing solver...")
    # The RobustNonLinearSolver is required for this complex, looped network
    solver = RobustNonLinearSolver(sim_config)

    print("\n🚀 Running simulation...")
    try:
        solution = solver.solve(network)
        print("\n🎉 Simulation completed successfully!")
        
        print("\n--- Simulation Results ---")
        solver.print_results(
            network, 
            solution,
            pressure_unit='bar',
            flow_rate_unit='l/min'
        )

    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        print("This type of complex network often requires a robust solver and a good initial guess for the inlet pressure.")


if __name__ == "__main__":
    main()
