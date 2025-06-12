"""
Network configuration for loading and saving network definitions
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from ..network.flow_network import FlowNetwork
from ..network.node import Node
from ..components.channel import Channel
from ..components.nozzle import Nozzle, NozzleType
from ..components.connector import Connector, ConnectorType
from .simulation_config import SimulationConfig


@dataclass
class NetworkConfig:
    """Complete network configuration including topology and simulation parameters"""
    
    network_name: str
    description: str
    nodes: List[Dict[str, Any]]
    components: List[Dict[str, Any]]
    connections: List[Dict[str, Any]]
    simulation: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'network_name': self.network_name,
            'description': self.description,
            'nodes': self.nodes,
            'components': self.components,
            'connections': self.connections,
            'simulation': self.simulation,
            'metadata': self.metadata or {}
        }


class NetworkConfigLoader:
    """Load network configurations from various formats"""
    
    @staticmethod
    def load_json(file_path: Union[str, Path]) -> NetworkConfig:
        """Load network configuration from JSON file"""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return NetworkConfig(
            network_name=data.get('network_name', 'Unnamed Network'),
            description=data.get('description', ''),
            nodes=data.get('nodes', []),
            components=data.get('components', []),
            connections=data.get('connections', []),
            simulation=data.get('simulation', {}),
            metadata=data.get('metadata', {})
        )
    
    @staticmethod
    def load_xml(file_path: Union[str, Path]) -> NetworkConfig:
        """Load network configuration from XML file"""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        # Parse nodes
        nodes = []
        nodes_elem = root.find('nodes')
        if nodes_elem is not None:
            for node_elem in nodes_elem.findall('node'):
                nodes.append({
                    'id': node_elem.get('id'),
                    'name': node_elem.get('name', ''),
                    'elevation': float(node_elem.get('elevation', 0.0)),
                    'type': node_elem.get('type', 'junction'),
                    'x': float(node_elem.get('x', 0.0)),
                    'y': float(node_elem.get('y', 0.0))
                })
        
        # Parse components
        components = []
        components_elem = root.find('components')
        if components_elem is not None:
            for comp_elem in components_elem.findall('component'):
                comp_data = {
                    'id': comp_elem.get('id'),
                    'name': comp_elem.get('name', ''),
                    'type': comp_elem.get('type')
                }
                
                # Add type-specific properties
                for prop_elem in comp_elem.findall('property'):
                    prop_name = prop_elem.get('name')
                    prop_value = prop_elem.text
                    
                    # Try to convert to appropriate type
                    try:
                        if '.' in prop_value:
                            comp_data[prop_name] = float(prop_value)
                        else:
                            comp_data[prop_name] = int(prop_value)
                    except ValueError:
                        comp_data[prop_name] = prop_value
                
                components.append(comp_data)
        
        # Parse connections
        connections = []
        connections_elem = root.find('connections')
        if connections_elem is not None:
            for conn_elem in connections_elem.findall('connection'):
                connections.append({
                    'from_node': conn_elem.get('from_node'),
                    'to_node': conn_elem.get('to_node'),
                    'component': conn_elem.get('component')
                })
        
        # Parse simulation parameters
        simulation = {}
        sim_elem = root.find('simulation')
        if sim_elem is not None:
            for param_elem in sim_elem.findall('parameter'):
                param_name = param_elem.get('name')
                param_value = param_elem.text
                
                # Try to convert to appropriate type
                try:
                    if '.' in param_value:
                        simulation[param_name] = float(param_value)
                    else:
                        simulation[param_name] = int(param_value)
                except ValueError:
                    simulation[param_name] = param_value
        
        return NetworkConfig(
            network_name=root.get('name', 'Unnamed Network'),
            description=root.get('description', ''),
            nodes=nodes,
            components=components,
            connections=connections,
            simulation=simulation,
            metadata={'format': 'xml'}
        )
    
    @staticmethod
    def build_network(config: NetworkConfig) -> tuple[FlowNetwork, SimulationConfig]:
        """Build FlowNetwork and SimulationConfig from NetworkConfig"""
        network = FlowNetwork(config.network_name)
        
        # Create nodes
        node_map = {}
        for node_data in config.nodes:
            node = network.create_node(
                name=node_data.get('name', node_data['id']),
                elevation=node_data.get('elevation', 0.0)
            )
            node_map[node_data['id']] = node
            
            # Set inlet/outlet based on type
            node_type = node_data.get('type', 'junction')
            if node_type == 'inlet':
                network.set_inlet(node)
            elif node_type == 'outlet':
                network.add_outlet(node)
        
        # Create components
        component_map = {}
        for comp_data in config.components:
            comp_type = comp_data['type']
            comp_id = comp_data['id']
            comp_name = comp_data.get('name', comp_id)
            
            if comp_type == 'channel':
                component = Channel(
                    diameter=comp_data['diameter'],
                    length=comp_data['length'],
                    name=comp_name
                )
            elif comp_type == 'nozzle':
                nozzle_type = NozzleType(comp_data.get('nozzle_type', 'sharp_edged'))
                component = Nozzle(
                    diameter=comp_data['diameter'],
                    nozzle_type=nozzle_type,
                    name=comp_name
                )
            elif comp_type == 'connector':
                connector_type = ConnectorType(comp_data.get('connector_type', 't_junction'))
                component = Connector(
                    diameter=comp_data['diameter'],
                    connector_type=connector_type,
                    name=comp_name
                )
            else:
                raise ValueError(f"Unknown component type: {comp_type}")
            
            component.id = comp_id  # Override generated ID
            component_map[comp_id] = component
        
        # Create connections
        for conn_data in config.connections:
            from_node = node_map[conn_data['from_node']]
            to_node = node_map[conn_data['to_node']]
            component = component_map[conn_data['component']]
            
            network.connect_components(from_node, to_node, component)
        
        # Create simulation config
        sim_config = SimulationConfig.from_dict(config.simulation)
        
        return network, sim_config


class NetworkConfigSaver:
    """Save network configurations to various formats"""
    
    @staticmethod
    def save_json(config: NetworkConfig, file_path: Union[str, Path]):
        """Save network configuration to JSON file"""
        with open(file_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
    
    @staticmethod
    def save_xml(config: NetworkConfig, file_path: Union[str, Path]):
        """Save network configuration to XML file"""
        root = ET.Element('network')
        root.set('name', config.network_name)
        root.set('description', config.description)
        
        # Add nodes
        nodes_elem = ET.SubElement(root, 'nodes')
        for node_data in config.nodes:
            node_elem = ET.SubElement(nodes_elem, 'node')
            for key, value in node_data.items():
                node_elem.set(key, str(value))
        
        # Add components
        components_elem = ET.SubElement(root, 'components')
        for comp_data in config.components:
            comp_elem = ET.SubElement(components_elem, 'component')
            comp_elem.set('id', comp_data['id'])
            comp_elem.set('name', comp_data.get('name', ''))
            comp_elem.set('type', comp_data['type'])
            
            # Add properties as sub-elements
            for key, value in comp_data.items():
                if key not in ['id', 'name', 'type']:
                    prop_elem = ET.SubElement(comp_elem, 'property')
                    prop_elem.set('name', key)
                    prop_elem.text = str(value)
        
        # Add connections
        connections_elem = ET.SubElement(root, 'connections')
        for conn_data in config.connections:
            conn_elem = ET.SubElement(connections_elem, 'connection')
            for key, value in conn_data.items():
                conn_elem.set(key, str(value))
        
        # Add simulation parameters
        sim_elem = ET.SubElement(root, 'simulation')
        for key, value in config.simulation.items():
            param_elem = ET.SubElement(sim_elem, 'parameter')
            param_elem.set('name', key)
            param_elem.text = str(value)
        
        # Write to file
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        tree.write(file_path, encoding='utf-8', xml_declaration=True)
    
    @staticmethod
    def from_network(network: FlowNetwork, sim_config: SimulationConfig) -> NetworkConfig:
        """Create NetworkConfig from FlowNetwork and SimulationConfig"""
        
        # Extract nodes
        nodes = []
        for node_id, node in network.nodes.items():
            node_type = 'junction'
            if network.inlet_node and node.id == network.inlet_node.id:
                node_type = 'inlet'
            elif node in network.outlet_nodes:
                node_type = 'outlet'
            
            nodes.append({
                'id': node.id,
                'name': node.name,
                'elevation': node.elevation,
                'type': node_type,
                'x': 0.0,  # Default position
                'y': 0.0   # Default position
            })
        
        # Extract components
        components = []
        for connection in network.connections:
            component = connection.component
            comp_data = {
                'id': component.id,
                'name': component.name,
                'type': component.component_type.value
            }
            
            # Add type-specific properties
            if hasattr(component, 'diameter'):
                comp_data['diameter'] = component.diameter
            if hasattr(component, 'length'):
                comp_data['length'] = component.length
            if hasattr(component, 'nozzle_type'):
                comp_data['nozzle_type'] = component.nozzle_type.value
            if hasattr(component, 'connector_type'):
                comp_data['connector_type'] = component.connector_type.value
            
            components.append(comp_data)
        
        # Extract connections
        connections = []
        for connection in network.connections:
            connections.append({
                'from_node': connection.from_node.id,
                'to_node': connection.to_node.id,
                'component': connection.component.id
            })
        
        return NetworkConfig(
            network_name=network.name,
            description=f"Network with {len(network.nodes)} nodes and {len(network.connections)} connections",
            nodes=nodes,
            components=components,
            connections=connections,
            simulation=sim_config.to_dict(),
            metadata={'created_from': 'network_object'}
        )