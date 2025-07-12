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
        This method ensures that nodes are unique.
        """
        if name in self._nodes:
            # Update existing node with new properties if provided
            if kwargs:
                for key, value in kwargs.items():
                    setattr(self._nodes[name], key, value)
            return self._nodes[name]
        
        node = Node(name=name, **kwargs)
        self._nodes[name] = node
        self._network.add_node(node)
        return node

    def set_inlet(self, node_name: str):
        """
        Sets the network's inlet node.
        """
        node = self._get_or_create_node(node_name)
        self._network.set_inlet(node)
        return self

    def add_outlet(self, node_name: str, pressure: float = 101325.0):
        """
        Adds an outlet node to the network.
        """
        node = self._get_or_create_node(node_name, pressure=pressure)
        self._network.add_outlet(node)
        return self

    def add_pipe(self, from_node_name: str, to_node_name: str, length: float, diameter: float, roughness: float = 0.00015, name: str = ""):
        """
        Adds a pipe (Channel) between two nodes.
        """
        from_node = self._get_or_create_node(from_node_name)
        to_node = self._get_or_create_node(to_node_name)
        
        pipe = Channel(length=length, diameter=diameter, roughness=roughness, name=name)
        
        self._network.connect_components(from_node=from_node, to_node=to_node, component=pipe)
        return self

    def add_nozzle(self, from_node_name: str, to_node_name: str, diameter: float, nozzle_type: NozzleType = NozzleType.STANDARD_ANGLE, name: str = ""):
        """
        Adds a nozzle between two nodes.
        """
        from_node = self._get_or_create_node(from_node_name)
        to_node = self._get_or_create_node(to_node_name)
        
        nozzle = Nozzle(nozzle_type=nozzle_type, diameter=diameter, name=name)
        
        self._network.connect_components(from_node=from_node, to_node=to_node, component=nozzle)
        return self

    def add_fitting(self, from_node_name: str, to_node_name: str, connector_type: ConnectorType, diameter: float, name: str = "", **kwargs):
        """
        Adds a fitting (Connector) between two nodes.
        """
        from_node = self._get_or_create_node(from_node_name)
        to_node = self._get_or_create_node(to_node_name)
        
        fitting = Connector(connector_type=connector_type, diameter=diameter, name=name, **kwargs)
        
        self._network.connect_components(from_node=from_node, to_node=to_node, component=fitting)
        return self

    def add_tee_junction(self, main_in: str, main_out: str, branch_out: str, 
                         tee_node_name: str, diameter: float):
        """
        Adds a physically-modeled T-junction for dividing flow.
        """
        self._get_or_create_node(tee_node_name)

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