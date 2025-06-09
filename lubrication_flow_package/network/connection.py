"""
Connection class - Represents connections between nodes through components
"""

from dataclasses import dataclass
from .node import Node


@dataclass
class Connection:
    """Represents a connection between two nodes through a component"""
    from_node: Node
    to_node: Node
    component: 'FlowComponent'
    flow_rate: float = 0.0  # m³/s (positive = from_node to to_node)