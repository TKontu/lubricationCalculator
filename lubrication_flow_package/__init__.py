"""
Network-Based Lubrication Flow Distribution Calculator

This package provides a comprehensive solution for analyzing lubrication flow
distribution in tree-like branching structures with component-based system building.

Key features:
- Tree-like network topology support
- Component-based architecture (channels, connectors, nozzles)
- Intuitive system building with connections
- Advanced network flow analysis
- Mass conservation at all junctions
- Pressure drop calculations through component sequences
"""

# Import all main classes and functions to maintain backward compatibility
from .components import (
    FlowComponent, Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType
)

from .network import (
    Node, Connection, FlowNetwork
)

from .solvers import (
    NetworkFlowSolver, SolverConfig
)

from .cli import (
    create_simple_tree_example,
    demonstrate_hydraulic_approaches_comparison,
    demonstrate_proper_hydraulic_analysis,
    main
)

__version__ = "1.0.0"
__author__ = "Lubrication Flow Calculator Team"

__all__ = [
    # Components
    'FlowComponent', 'Channel', 'Connector', 'Nozzle',
    'ComponentType', 'ConnectorType', 'NozzleType',
    
    # Network
    'Node', 'Connection', 'FlowNetwork',
    
    # Solvers
    'NetworkFlowSolver', 'SolverConfig',
    
    # CLI/Demo
    'create_simple_tree_example',
    'demonstrate_hydraulic_approaches_comparison', 
    'demonstrate_proper_hydraulic_analysis',
    'main'
]