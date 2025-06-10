"""
FlowNetwork class - Represents a complete flow network with nodes and connections
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from .node import Node
from .connection import Connection


class FlowNetwork:
    """Represents a complete flow network with nodes and connections"""
    
    def __init__(self, name: str = "Flow Network"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self.inlet_node: Optional[Node] = None
        self.outlet_nodes: List[Node] = []
        
        # Network topology
        self.adjacency_list: Dict[str, List[Connection]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[Connection]] = defaultdict(list)
    
    def add_node(self, node: Node) -> Node:
        """Add a node to the network"""
        self.nodes[node.id] = node
        return node
    
    def create_node(self, name: str = "", elevation: float = 0.0) -> Node:
        """Create and add a new node to the network"""
        node = Node(name=name, elevation=elevation)
        return self.add_node(node)
    
    def connect_components(self, from_node: Node, to_node: Node, 
                          component: 'FlowComponent') -> Connection:
        """Connect two nodes through a component"""
        connection = Connection(from_node, to_node, component)
        self.connections.append(connection)
        
        # Update adjacency lists
        self.adjacency_list[from_node.id].append(connection)
        self.reverse_adjacency[to_node.id].append(connection)
        
        return connection
    
    def set_inlet(self, node: Node):
        """Set the inlet node for the network"""
        self.inlet_node = node
    
    def add_outlet(self, node: Node):
        """Add an outlet node to the network"""
        if node not in self.outlet_nodes:
            self.outlet_nodes.append(node)
    
    def get_paths_to_outlets(self) -> List[List[Connection]]:
        """Get all paths from inlet to outlets using the unified DFS implementation"""
        # Import here to avoid circular imports
        from ..utils.network_utils import find_all_paths
        return find_all_paths(self, raise_on_no_paths=False)
    
    def validate_network(self) -> Tuple[bool, List[str]]:
        """Validate the network topology"""
        errors = []
        
        # Check if inlet is defined
        if not self.inlet_node:
            errors.append("No inlet node defined")
        
        # Check if outlets are defined
        if not self.outlet_nodes:
            errors.append("No outlet nodes defined")
        
        # Check connectivity
        if self.inlet_node:
            paths = self.get_paths_to_outlets()
            if not paths:
                errors.append("No paths from inlet to outlets")
            
            # Check if all outlets are reachable
            reachable_outlets = set()
            for path in paths:
                if path:
                    reachable_outlets.add(path[-1].to_node.id)
            
            unreachable = [node for node in self.outlet_nodes 
                          if node.id not in reachable_outlets]
            if unreachable:
                errors.append(f"Unreachable outlets: {[n.name for n in unreachable]}")
        
        # Check for isolated nodes
        connected_nodes = set()
        for connection in self.connections:
            connected_nodes.add(connection.from_node.id)
            connected_nodes.add(connection.to_node.id)
        
        isolated = [node for node_id, node in self.nodes.items() 
                   if node_id not in connected_nodes]
        if isolated:
            errors.append(f"Isolated nodes: {[n.name for n in isolated]}")
        
        return len(errors) == 0, errors
    
    def get_junction_nodes(self) -> List[Node]:
        """Get nodes that are junctions (more than 2 connections)"""
        junction_nodes = []
        
        for node_id, node in self.nodes.items():
            # Count total connections (incoming + outgoing)
            incoming = len(self.reverse_adjacency[node_id])
            outgoing = len(self.adjacency_list[node_id])
            total_connections = incoming + outgoing
            
            if total_connections > 2:
                junction_nodes.append(node)
        
        return junction_nodes
    
    def print_network_info(self):
        """Print network information"""
        print(f"\nNetwork: {self.name}")
        print(f"Nodes: {len(self.nodes)}")
        print(f"Connections: {len(self.connections)}")
        print(f"Inlet: {self.inlet_node.name if self.inlet_node else 'None'}")
        print(f"Outlets: {[n.name for n in self.outlet_nodes]}")
        
        # Print paths
        try:
            paths = self.get_paths_to_outlets()
            print(f"Paths to outlets: {len(paths)}")
            for i, path in enumerate(paths):
                path_str = " -> ".join([path[0].from_node.name] + 
                                     [conn.to_node.name for conn in path])
                print(f"  Path {i+1}: {path_str}")
        except Exception as e:
            print(f"Error getting paths: {e}")