"""
Network Module

This module contains classes for representing flow networks including
nodes, connections, and the complete flow network topology.
"""

from .node import Node
from .connection import Connection
from .flow_network import FlowNetwork

__all__ = ['Node', 'Connection', 'FlowNetwork']