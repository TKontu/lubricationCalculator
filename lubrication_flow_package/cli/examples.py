"""
Example network creation functions
"""

from typing import Tuple
from ..network import FlowNetwork
from ..components import Channel, Nozzle, NozzleType


def create_simple_tree_example() -> Tuple[FlowNetwork, float, float]:
    """Create a simple tree network example"""
    network = FlowNetwork("Simple Tree Example")
    
    # Create nodes
    inlet = network.create_node("Inlet", elevation=0.0)
    junction1 = network.create_node("Junction1", elevation=1.0)
    branch1_end = network.create_node("Branch1_End", elevation=2.0)
    branch2_end = network.create_node("Branch2_End", elevation=1.5)
    outlet1 = network.create_node("Outlet1", elevation=2.0)
    outlet2 = network.create_node("Outlet2", elevation=1.5)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    
    # Create components with better nozzle sizing
    main_channel = Channel(diameter=0.08, length=10.0, name="Main Channel")
    branch1_channel = Channel(diameter=0.05, length=8.0, name="Branch1 Channel")
    branch2_channel = Channel(diameter=0.04, length=6.0, name="Branch2 Channel")
    # Further increase nozzle sizes to reduce pressure drops and velocities
    nozzle1 = Nozzle(diameter=0.025, nozzle_type=NozzleType.VENTURI, name="Nozzle1")
    nozzle2 = Nozzle(diameter=0.020, nozzle_type=NozzleType.SHARP_EDGED, name="Nozzle2")
    
    # Connect components to form tree structure
    # Main line: Inlet -> Junction1
    network.connect_components(inlet, junction1, main_channel)
    
    # Branch 1: Junction1 -> Branch1_End -> Outlet1 (through nozzle)
    network.connect_components(junction1, branch1_end, branch1_channel)
    network.connect_components(branch1_end, outlet1, nozzle1)
    
    # Branch 2: Junction1 -> Branch2_End -> Outlet2 (through nozzle)
    network.connect_components(junction1, branch2_end, branch2_channel)
    network.connect_components(branch2_end, outlet2, nozzle2)
    
    total_flow_rate = 0.015  # 15 L/s
    temperature = 40  # °C
    
    return network, total_flow_rate, temperature