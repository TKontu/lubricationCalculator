"""
Base classes and enums for flow components
"""

import uuid
from enum import Enum
from typing import Dict


class ComponentType(Enum):
    """Types of flow components"""
    CHANNEL = "channel"
    CONNECTOR = "connector"
    NOZZLE = "nozzle"


class ConnectorType(Enum):
    """Types of connectors"""
    # Basic connectors
    T_JUNCTION = "t_junction"
    X_JUNCTION = "x_junction"
    STRAIGHT = "straight"
    
    # Elbows
    ELBOW_90 = "elbow_90"
    ELBOW_45 = "elbow_45"
    ELBOW_30 = "elbow_30"
    ELBOW_SMOOTH = "elbow_smooth"
    ELBOW_MITERED = "elbow_mitered"
    
    # Reducers and expanders
    REDUCER_GRADUAL = "reducer_gradual"
    REDUCER_SUDDEN = "reducer_sudden"
    EXPANDER_GRADUAL = "expander_gradual"
    EXPANDER_SUDDEN = "expander_sudden"
    
    # Valves
    GATE_VALVE = "gate_valve"
    BALL_VALVE = "ball_valve"
    GLOBE_VALVE = "globe_valve"
    CHECK_VALVE = "check_valve"
    BUTTERFLY_VALVE = "butterfly_valve"
    
    # Fittings
    UNION = "union"
    COUPLING = "coupling"
    ADAPTER = "adapter"
    
    # Complex junctions
    WYE_JUNCTION = "wye_junction"
    LATERAL_TEE = "lateral_tee"
    
    # Bends
    RETURN_BEND = "return_bend"
    LONG_RADIUS_ELBOW = "long_radius_elbow"
    SHORT_RADIUS_ELBOW = "short_radius_elbow"


class NozzleType(Enum):
    """Types of nozzles"""
    SHARP_EDGED = "sharp_edged"
    ROUNDED = "rounded"
    VENTURI = "venturi"
    FLOW_NOZZLE = "flow_nozzle"


class FlowComponent:
    """Base class for all flow components"""
    
    def __init__(self, component_id: str = None, name: str = ""):
        self.id = component_id or str(uuid.uuid4())[:8]
        self.name = name or f"{self.__class__.__name__}_{self.id}"
        self.component_type = ComponentType.CHANNEL  # Override in subclasses
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop through this component"""
        raise NotImplementedError("Subclasses must implement calculate_pressure_drop")
    
    def get_flow_area(self) -> float:
        """Get the flow area of this component"""
        raise NotImplementedError("Subclasses must implement get_flow_area")
    
    def validate_flow_rate(self, flow_rate: float) -> bool:
        """Validate if the flow rate is acceptable for this component"""
        return flow_rate >= 0
    
    def get_max_recommended_velocity(self) -> float:
        """Get maximum recommended velocity for this component type"""
        # Default conservative velocity limit
        return 10.0  # m/s