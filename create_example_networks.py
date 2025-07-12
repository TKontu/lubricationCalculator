
Script to create example network configurations using the new NetworkBuilder.
"""

from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.components.base import NozzleType

def create_simple_tree_network():
    """
    Creates a simple tree-like network with one inlet and two outlets.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.02,
        temperature=50.0,
        inlet_pressure=250000.0
    )
    
    builder = NetworkBuilder(sim_config)
    
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "j1", length=5, diameter=0.1)
        .add_pipe("j1", "out1", length=10, diameter=0.08)
        .add_nozzle("out1", "nozzle_out1", diameter=0.05, nozzle_type=NozzleType.ROUNDED)
        .add_pipe("j1", "out2", length=10, diameter=0.08)
        .add_nozzle("out2", "nozzle_out2", diameter=0.05, nozzle_type=NozzleType.ROUNDED)
        .add_outlet("nozzle_out1")
        .add_outlet("nozzle_out2")
        .build()
    )
    
    # The FlowNetwork object does not have a save_json method.
    # This functionality would need to be added to the FlowNetwork class.
    # For now, we just return the network object.
    print("   ✅ Created: Simple Tree Network")
    return network

def create_complex_network_with_tee():
    """
    Creates a more complex network featuring a T-junction.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.03,
        temperature=60.0,
        inlet_pressure=300000.0
    )
    
    builder = NetworkBuilder(sim_config)
    
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "tee_in", length=5, diameter=0.1)
        .add_tee_junction("tee_in", "main_out", "branch_out", "T1", diameter=0.1)
        .add_pipe("main_out", "out1", length=10, diameter=0.1)
        .add_nozzle("out1", "nozzle_out1", diameter=0.06)
        .add_pipe("branch_out", "out2", length=15, diameter=0.08)
        .add_nozzle("out2", "nozzle_out2", diameter=0.04)
        .add_outlet("nozzle_out1")
        .add_outlet("nozzle_out2")
        .build()
    )
    
    print("   ✅ Created: Complex Network with Tee")
    return network


def main():
    """Create and demonstrate the example networks."""
    print("Creating example network configurations using the new NetworkBuilder...")
    
    simple_network = create_simple_tree_network()
    complex_network = create_complex_network_with_tee()
    
    print("\n🎉 All example networks created successfully!")
    print("\nThese networks can now be used with the solvers.")

if __name__ == "__main__":
    main()
