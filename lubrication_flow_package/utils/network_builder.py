"""
Network builder utility for programmatically creating networks
"""

from typing import Dict, List, Optional, Tuple
from ..network.flow_network import FlowNetwork
from ..components.channel import Channel
from ..components.nozzle import Nozzle, NozzleType
from ..components.connector import Connector, ConnectorType
from ..config.network_config import NetworkConfig, NetworkConfigSaver
from ..config.simulation_config import SimulationConfig


class NetworkBuilder:
    """Helper class for building networks programmatically"""
    
    def __init__(self, name: str = "Network", description: str = ""):
        self.name = name
        self.description = description
        self.nodes = []
        self.components = []
        self.connections = []
        self.simulation_params = {}
        
    def add_node(self, node_id: str, name: str = "", elevation: float = 0.0, 
                 node_type: str = "junction", x: float = 0.0, y: float = 0.0) -> 'NetworkBuilder':
        """Add a node to the network"""
        self.nodes.append({
            'id': node_id,
            'name': name or node_id,
            'elevation': elevation,
            'type': node_type,
            'x': x,
            'y': y
        })
        return self
    
    def add_inlet(self, node_id: str, name: str = "", elevation: float = 0.0,
                  x: float = 0.0, y: float = 0.0) -> 'NetworkBuilder':
        """Add an inlet node"""
        return self.add_node(node_id, name, elevation, 'inlet', x, y)
    
    def add_outlet(self, node_id: str, name: str = "", elevation: float = 0.0,
                   x: float = 0.0, y: float = 0.0) -> 'NetworkBuilder':
        """Add an outlet node"""
        return self.add_node(node_id, name, elevation, 'outlet', x, y)
    
    def add_junction(self, node_id: str, name: str = "", elevation: float = 0.0,
                     x: float = 0.0, y: float = 0.0) -> 'NetworkBuilder':
        """Add a junction node"""
        return self.add_node(node_id, name, elevation, 'junction', x, y)
    
    def add_channel(self, component_id: str, diameter: float, length: float,
                    name: str = "") -> 'NetworkBuilder':
        """Add a channel component"""
        self.components.append({
            'id': component_id,
            'name': name or component_id,
            'type': 'channel',
            'diameter': diameter,
            'length': length
        })
        return self
    
    def add_nozzle(self, component_id: str, diameter: float, 
                   nozzle_type: str = 'sharp_edged', name: str = "") -> 'NetworkBuilder':
        """Add a nozzle component"""
        self.components.append({
            'id': component_id,
            'name': name or component_id,
            'type': 'nozzle',
            'diameter': diameter,
            'nozzle_type': nozzle_type
        })
        return self
    
    def add_connector(self, component_id: str, diameter: float,
                      connector_type: str = 't_junction', name: str = "") -> 'NetworkBuilder':
        """Add a connector component"""
        self.components.append({
            'id': component_id,
            'name': name or component_id,
            'type': 'connector',
            'diameter': diameter,
            'connector_type': connector_type
        })
        return self
    
    def connect(self, from_node: str, to_node: str, component: str) -> 'NetworkBuilder':
        """Connect two nodes through a component"""
        self.connections.append({
            'from_node': from_node,
            'to_node': to_node,
            'component': component
        })
        return self
    
    def set_simulation_params(self, total_flow_rate: float = 0.015, 
                             temperature: float = 40.0, inlet_pressure: float = 200000.0,
                             oil_density: float = 900.0, oil_type: str = "SAE30",
                             **kwargs) -> 'NetworkBuilder':
        """Set simulation parameters"""
        self.simulation_params = {
            'flow_parameters': {
                'total_flow_rate': total_flow_rate,
                'temperature': temperature,
                'inlet_pressure': inlet_pressure,
                'outlet_pressure': kwargs.get('outlet_pressure')
            },
            'fluid_properties': {
                'oil_density': oil_density,
                'oil_type': oil_type
            },
            'solver_settings': {
                'max_iterations': kwargs.get('max_iterations', 100),
                'tolerance': kwargs.get('tolerance', 1e-6),
                'relaxation_factor': kwargs.get('relaxation_factor', 0.8)
            },
            'output_settings': {
                'output_units': kwargs.get('output_units', 'metric'),
                'detailed_output': kwargs.get('detailed_output', True),
                'save_results': kwargs.get('save_results', False),
                'results_file': kwargs.get('results_file')
            }
        }
        return self
    
    def build_config(self) -> NetworkConfig:
        """Build a NetworkConfig object"""
        return NetworkConfig(
            network_name=self.name,
            description=self.description,
            nodes=self.nodes,
            components=self.components,
            connections=self.connections,
            simulation=self.simulation_params,
            metadata={'created_by': 'NetworkBuilder', 'version': '1.0'}
        )
    
    def save_json(self, filename: str):
        """Save the network configuration to a JSON file"""
        config = self.build_config()
        NetworkConfigSaver.save_json(config, filename)
    
    def save_xml(self, filename: str):
        """Save the network configuration to an XML file"""
        config = self.build_config()
        NetworkConfigSaver.save_xml(config, filename)


def create_simple_tree(name: str = "Simple Tree", flow_rate: float = 0.015,
                      inlet_pressure: float = 200000.0) -> NetworkBuilder:
    """Create a simple tree network configuration"""
    return (NetworkBuilder(name, "A simple tree network with two outlets")
            .add_inlet("inlet", "Main Inlet", 0.0, 0.0, 0.0)
            .add_junction("junction", "Main Junction", 1.0, 10.0, 0.0)
            .add_outlet("outlet1", "Outlet 1", 2.0, 20.0, 5.0)
            .add_outlet("outlet2", "Outlet 2", 1.5, 20.0, -5.0)
            .add_channel("main_channel", 0.08, 10.0, "Main Channel")
            .add_channel("branch1", 0.05, 8.0, "Branch 1")
            .add_channel("branch2", 0.04, 6.0, "Branch 2")
            .connect("inlet", "junction", "main_channel")
            .connect("junction", "outlet1", "branch1")
            .connect("junction", "outlet2", "branch2")
            .set_simulation_params(flow_rate, 40.0, inlet_pressure))


def create_complex_network(name: str = "Complex Network", flow_rate: float = 0.025,
                          inlet_pressure: float = 250000.0) -> NetworkBuilder:
    """Create a more complex network configuration"""
    return (NetworkBuilder(name, "A complex network with multiple junctions")
            .add_inlet("inlet", "Main Inlet", 0.0, 0.0, 0.0)
            .add_junction("junction1", "Primary Junction", 1.0, 10.0, 0.0)
            .add_junction("junction2", "Secondary Junction A", 2.0, 20.0, 5.0)
            .add_junction("junction3", "Secondary Junction B", 1.5, 20.0, -5.0)
            .add_outlet("outlet1", "Outlet 1A", 2.5, 30.0, 8.0)
            .add_outlet("outlet2", "Outlet 1B", 2.0, 30.0, 2.0)
            .add_outlet("outlet3", "Outlet 2A", 1.8, 30.0, -2.0)
            .add_outlet("outlet4", "Outlet 2B", 1.2, 30.0, -8.0)
            .add_channel("main_channel", 0.10, 10.0, "Main Supply")
            .add_channel("branch_a", 0.06, 12.0, "Branch A")
            .add_channel("branch_b", 0.06, 12.0, "Branch B")
            .add_channel("sub_a1", 0.04, 8.0, "Sub-branch A1")
            .add_channel("sub_a2", 0.04, 6.0, "Sub-branch A2")
            .add_channel("sub_b1", 0.04, 7.0, "Sub-branch B1")
            .add_channel("sub_b2", 0.04, 9.0, "Sub-branch B2")
            .connect("inlet", "junction1", "main_channel")
            .connect("junction1", "junction2", "branch_a")
            .connect("junction1", "junction3", "branch_b")
            .connect("junction2", "outlet1", "sub_a1")
            .connect("junction2", "outlet2", "sub_a2")
            .connect("junction3", "outlet3", "sub_b1")
            .connect("junction3", "outlet4", "sub_b2")
            .set_simulation_params(flow_rate, 45.0, inlet_pressure, oil_type="SAE40"))


# Example usage functions
def create_example_networks():
    """Create example network files"""
    
    # Simple tree network
    simple = create_simple_tree("Simple Tree Example", 0.015, 200000.0)
    simple.save_json("simple_tree_example.json")
    simple.save_xml("simple_tree_example.xml")
    
    # Complex network
    complex_net = create_complex_network("Complex Distribution Network", 0.025, 250000.0)
    complex_net.save_json("complex_network_example.json")
    
    # Custom network with nozzles
    custom = (NetworkBuilder("Custom Network", "Network with nozzles and connectors")
              .add_inlet("inlet", "Pump Inlet", 0.0, 0.0, 0.0)
              .add_junction("manifold", "Distribution Manifold", 1.0, 10.0, 0.0)
              .add_junction("branch1_end", "Branch 1 End", 2.0, 18.0, 5.0)
              .add_junction("branch2_end", "Branch 2 End", 1.5, 18.0, -5.0)
              .add_outlet("outlet1", "Lubrication Point 1", 2.0, 20.0, 5.0)
              .add_outlet("outlet2", "Lubrication Point 2", 1.5, 20.0, -5.0)
              .add_channel("supply_line", 0.08, 10.0, "Main Supply Line")
              .add_channel("branch1_line", 0.05, 8.0, "Branch 1 Line")
              .add_channel("branch2_line", 0.04, 8.0, "Branch 2 Line")
              .add_nozzle("nozzle1", 0.025, "venturi", "Precision Nozzle 1")
              .add_nozzle("nozzle2", 0.020, "sharp_edged", "Precision Nozzle 2")
              .connect("inlet", "manifold", "supply_line")
              .connect("manifold", "branch1_end", "branch1_line")
              .connect("manifold", "branch2_end", "branch2_line")
              .connect("branch1_end", "outlet1", "nozzle1")
              .connect("branch2_end", "outlet2", "nozzle2")
              .set_simulation_params(0.012, 35.0, 180000.0))
    
    custom.save_json("custom_network_example.json")
    
    print("✅ Example networks created:")
    print("   - simple_tree_example.json")
    print("   - simple_tree_example.xml")
    print("   - complex_network_example.json")
    print("   - custom_network_example.json")


if __name__ == "__main__":
    create_example_networks()