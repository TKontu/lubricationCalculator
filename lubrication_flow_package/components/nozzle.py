"""
Nozzle component - Represents flow nozzles or orifices
"""

import math
from typing import Dict
from .base import FlowComponent, ComponentType, NozzleType


class Nozzle(FlowComponent):
    """Represents a flow nozzle or orifice"""
    
    def __init__(self, diameter: float, nozzle_type: NozzleType = NozzleType.SHARP_EDGED,
                 discharge_coeff: float = None, component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.NOZZLE
        self.diameter = diameter  # m
        self.nozzle_type = nozzle_type
        
        # Set default discharge coefficient if not provided
        if discharge_coeff is None:
            self.discharge_coeff = self._get_default_discharge_coeff()
        else:
            self.discharge_coeff = discharge_coeff
        
        # Validation
        if diameter <= 0:
            raise ValueError("Nozzle diameter must be positive")
        if not (0 < self.discharge_coeff <= 1):
            raise ValueError("Discharge coefficient must be between 0 and 1")
    
    def _get_default_discharge_coeff(self) -> float:
        """Get default discharge coefficient based on nozzle type"""
        defaults = {
            NozzleType.SHARP_EDGED: 0.6,
            NozzleType.ROUNDED: 0.8,
            NozzleType.VENTURI: 0.95,
            NozzleType.FLOW_NOZZLE: 0.98
        }
        return defaults[self.nozzle_type]
    
    def get_flow_area(self) -> float:
        """Get the flow area"""
        return math.pi * (self.diameter / 2) ** 2
    
    def get_max_recommended_velocity(self) -> float:
        """Get maximum recommended velocity for nozzles"""
        # Nozzles can handle higher velocities than pipes
        velocity_limits = {
            NozzleType.SHARP_EDGED: 15.0,    # m/s
            NozzleType.ROUNDED: 20.0,        # m/s  
            NozzleType.VENTURI: 30.0,        # m/s
            NozzleType.FLOW_NOZZLE: 25.0     # m/s
        }
        return velocity_limits.get(self.nozzle_type, 15.0)
    
    def validate_flow_rate(self, flow_rate: float) -> bool:
        """Validate if the flow rate is acceptable for this nozzle"""
        if flow_rate <= 0:
            return True  # Zero flow is always acceptable
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        max_velocity = self.get_max_recommended_velocity()
        
        return velocity <= max_velocity
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using orifice flow equation"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        
        # Orifice pressure drop calculation
        if self.nozzle_type == NozzleType.VENTURI:
            # Venturi has lower permanent pressure loss due to diffuser recovery
            K = ((1 / self.discharge_coeff ** 2) - 1) * 0.1  # 10% permanent loss
        else:
            # Standard orifice equation
            K = (1 / self.discharge_coeff ** 2) - 1
        
        pressure_drop = K * density * velocity ** 2 / 2
        
        return pressure_drop