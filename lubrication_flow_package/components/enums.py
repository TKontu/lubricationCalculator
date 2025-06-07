"""
Enums for flow components
"""

from enum import Enum


class ComponentType(Enum):
    """Types of flow components"""
    CHANNEL = "channel"
    CONNECTOR = "connector"
    NOZZLE = "nozzle"


class ConnectorType(Enum):
    """Types of connectors"""
    T_JUNCTION = "t_junction"
    X_JUNCTION = "x_junction"
    ELBOW_90 = "elbow_90"
    REDUCER = "reducer"
    STRAIGHT = "straight"


class NozzleType(Enum):
    """Types of nozzles"""
    SHARP_EDGED = "sharp_edged"
    ROUNDED = "rounded"
    VENTURI = "venturi"
    FLOW_NOZZLE = "flow_nozzle"