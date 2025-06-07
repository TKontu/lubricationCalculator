"""
Network connection representation
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .node import Node
    from ..components.base import FlowComponent


@dataclass
class Connection:
    """Represents a connection between two nodes through a component"""
    from_node: 'Node'
    to_node: 'Node'
    component: 'FlowComponent'
    flow_rate: float = 0.0  # m³/s (positive = from_node to to_node)