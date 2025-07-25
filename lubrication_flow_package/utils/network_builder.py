"""
NetworkBuilder for constructing hydraulic networks.

This module provides a high-level API for building FlowNetwork objects
in a robust and user-friendly way.
"""

from typing import Dict, Optional

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..network.node import Node, NodeType
from ..network.connection import Connection
from ..components.base import FlowComponent, ConnectorType
from ..components.channel import Channel
from ..components.connector import Connector
from ..components.nozzle import Nozzle, NozzleType


class NetworkBuilder:
    """
    A builder class for assembling hydraulic networks.

    This class provides a user-friendly interface to create complex
    FlowNetwork objects by abstracting away the manual creation of
    nodes and connections.
    """

    def __init__(self, sim_config: Optional[SimulationConfig] = None):
        """
        Initializes the NetworkBuilder.

        Args:
            sim_config: Optional simulation configuration.
        """
        self._network = FlowNetwork()
        self._sim_config = sim_config
        self._nodes: Dict[str, Node] = {}

    def _get_or_create_node(self, name: str, **kwargs) -> Node:
        """
        Retrieves an existing node by name or creates a new one.
        This method ensures that nodes are unique and safely updates them.
        """
        if name in self._nodes:
            # Update existing node only with non-None properties
            if kwargs:
                for key, value in kwargs.items():
                    if value is not None:
                        setattr(self._nodes[name], key, value)
            return self._nodes[name]
        
        # Filter out None values so Node constructor defaults are used
        creation_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        node = Node(name=name, **creation_kwargs)
        self._nodes[name] = node
        self._network._add_node(node)
        return node

    def add_node(self, name: str, elevation: float = 0.0, **kwargs) -> 'NetworkBuilder':
        """
        Adds or updates a node with specific properties.

        Args:
            name: The unique name of the node.
            elevation: The elevation of the node (in meters).
            **kwargs: Other node properties.

        Returns:
            The builder instance for method chaining.
        """
        self._get_or_create_node(name, elevation=elevation, **kwargs)
        return self

    def set_inlet(self, node_name: str, elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Sets the network's inlet node.

        Args:
            node_name: The name of the inlet node.
            elevation: Optional elevation for the node.
        """
        node = self._get_or_create_node(node_name, elevation=elevation)
        self._network._set_inlet(node)
        return self

    def add_outlet(self, node_name: str, pressure: float = 101325.0, elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Adds an outlet node to the network.

        Args:
            node_name: The name of the outlet node.
            pressure: The pressure at the outlet node.
            elevation: Optional elevation for the node.
        """
        node = self._get_or_create_node(node_name, pressure=pressure, elevation=elevation)
        self._network._add_outlet(node)
        return self

    def add_pipe(self, from_node_name: str, to_node_name: str, length: float, diameter: float, 
                 roughness: float = 0.00015, name: str = "",
                 from_node_elevation: Optional[float] = None, 
                 to_node_elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Adds a pipe (Channel) between two nodes.

        Args:
            from_node_elevation: Optional elevation for the 'from' node.
            to_node_elevation: Optional elevation for the 'to' node.
        """
        from_node = self._get_or_create_node(from_node_name, elevation=from_node_elevation)
        to_node = self._get_or_create_node(to_node_name, elevation=to_node_elevation)
        
        pipe = Channel(length=length, diameter=diameter, roughness=roughness, name=name)
        
        self._network._connect_components(from_node=from_node, to_node=to_node, component=pipe)
        return self

    def add_nozzle(self, from_node_name: str, to_node_name: str, diameter: float, 
                   nozzle_type: NozzleType = NozzleType.STANDARD_ANGLE, name: str = "",
                   from_node_elevation: Optional[float] = None,
                   to_node_elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Adds a nozzle between two nodes.

        Args:
            from_node_elevation: Optional elevation for the 'from' node.
            to_node_elevation: Optional elevation for the 'to' node.
        """
        from_node = self._get_or_create_node(from_node_name, elevation=from_node_elevation)
        to_node = self._get_or_create_node(to_node_name, elevation=to_node_elevation)
        
        nozzle = Nozzle(nozzle_type=nozzle_type, diameter=diameter, name=name)
        
        self._network._connect_components(from_node=from_node, to_node=to_node, component=nozzle)
        return self

    def add_fitting(self, from_node_name: str, to_node_name: str, connector_type: ConnectorType, 
                    diameter: float, name: str = "",
                    from_node_elevation: Optional[float] = None,
                    to_node_elevation: Optional[float] = None, **kwargs) -> 'NetworkBuilder':
        """
        Adds a fitting (Connector) between two nodes.

        Args:
            from_node_elevation: Optional elevation for the 'from' node.
            to_node_elevation: Optional elevation for the 'to' node.
        """
        from_node = self._get_or_create_node(from_node_name, elevation=from_node_elevation)
        to_node = self._get_or_create_node(to_node_name, elevation=to_node_elevation)
        
        fitting = Connector(connector_type=connector_type, diameter=diameter, name=name, **kwargs)
        
        self._network._connect_components(from_node=from_node, to_node=to_node, component=fitting)
        return self

    def add_component(self, from_node_name: str, to_node_name: str, component: FlowComponent,
                      from_node_elevation: Optional[float] = None,
                      to_node_elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Adds a pre-created component between two nodes.
        """
        from_node = self._get_or_create_node(from_node_name, elevation=from_node_elevation)
        to_node = self._get_or_create_node(to_node_name, elevation=to_node_elevation)
        
        self._network._connect_components(from_node=from_node, to_node=to_node, component=component)
        return self

    def add_tee_junction(self, main_in: str, main_out: str, branch_out: str, 
                         tee_node_name: str, diameter: float,
                         main_in_elevation: Optional[float] = None,
                         main_out_elevation: Optional[float] = None,
                         branch_out_elevation: Optional[float] = None,
                         tee_node_elevation: Optional[float] = None) -> 'NetworkBuilder':
        """
        Adds a physically-modeled T-junction for dividing flow.
        """
        self._get_or_create_node(tee_node_name, elevation=tee_node_elevation)
        self._get_or_create_node(main_in, elevation=main_in_elevation)
        self._get_or_create_node(main_out, elevation=main_out_elevation)
        self._get_or_create_node(branch_out, elevation=branch_out_elevation)

        self.add_fitting(from_node_name=main_in, to_node_name=tee_node_name,
                         connector_type=ConnectorType.STRAIGHT,
                         diameter=diameter,
                         loss_coefficient=0.05,
                         name=f"tee_{tee_node_name}_inlet")

        self.add_fitting(from_node_name=tee_node_name, to_node_name=main_out,
                         connector_type=ConnectorType.STRAIGHT,
                         diameter=diameter,
                         loss_coefficient=0.2,
                         name=f"tee_{tee_node_name}_run")

        self.add_fitting(from_node_name=tee_node_name, to_node_name=branch_out,
                         connector_type=ConnectorType.STRAIGHT,
                         diameter=diameter,
                         loss_coefficient=1.0,
                         name=f"tee_{tee_node_name}_branch")
        return self

    def build(self) -> FlowNetwork:
        """
        Validates and returns the constructed FlowNetwork.
        """
        is_valid, errors = self._network.validate_network()
        if not is_valid:
            raise ValueError(f"Constructed network is invalid: {errors}")
        return self._network