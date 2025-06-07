"""
Base flow component class
"""

import uuid
from typing import Dict
from .enums import ComponentType


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