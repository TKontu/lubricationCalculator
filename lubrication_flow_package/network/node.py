"""
Node class - Represents connection points in the network
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto


class NodeType(Enum):
    """Type of node in the network"""
    JUNCTION = auto()
    INLET = auto()
    OUTLET = auto()


@dataclass
class Node:
    """Represents a connection point in the network"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pressure: float = 0.0  # Pa
    elevation: float = 0.0  # m
    name: str = ""
    node_type: NodeType = NodeType.JUNCTION
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Node_{self.id}"
