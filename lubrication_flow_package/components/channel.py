"""
Channel flow component
"""

import math
from typing import Dict
from .base import FlowComponent
from .enums import ComponentType


class Channel(FlowComponent):
    """Represents a pipe or drilling channel"""
    
    def __init__(self, diameter: float, length: float, roughness: float = 0.00015,
                 component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.CHANNEL
        self.diameter = diameter  # m
        self.length = length      # m
        self.roughness = roughness  # m
        
        # Validation
        if diameter <= 0:
            raise ValueError("Channel diameter must be positive")
        if length <= 0:
            raise ValueError("Channel length must be positive")
        if roughness < 0:
            raise ValueError("Channel roughness cannot be negative")
    
    def get_flow_area(self) -> float:
        """Get the flow area"""
        return math.pi * (self.diameter / 2) ** 2
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using Darcy-Weisbach equation"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        viscosity = fluid_properties['viscosity']
        
        area = self.get_flow_area()
        velocity = flow_rate / area
        reynolds = (density * velocity * self.diameter) / viscosity
        
        # Calculate friction factor
        relative_roughness = self.roughness / self.diameter
        friction_factor = self._calculate_friction_factor(reynolds, relative_roughness)
        
        # Darcy-Weisbach equation
        pressure_drop = friction_factor * (self.length / self.diameter) * \
                       (density * velocity ** 2) / 2
        
        return pressure_drop
    
    def _calculate_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate friction factor using Churchill's full-range explicit formula"""
        if reynolds <= 0:
            return 0.0

        # Churchill's full-range formula (Churchill, 1977)
        # A = [2.457 * ln(1/((7/Re)^0.9 + 0.27 * ε/D))]^16
        # B = (37530 / Re)^16
        # f = 8 * [ (8/Re)^12 + (A + B)^(-1.5) ]^(1/12)
        A = (2.457 * math.log(1.0 / ((7.0 / reynolds) ** 0.9 + 0.27 * relative_roughness))) ** 16
        B = (37530.0 / reynolds) ** 16
        term = (8.0 / reynolds) ** 12 + (A + B) ** -1.5
        friction_factor = 8.0 * term ** (1.0 / 12.0)

        return friction_factor
    
    def _turbulent_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate turbulent friction factor using Swamee-Jain approximation"""
        if relative_roughness <= 0:  # Smooth pipe
            if reynolds < 100000:
                return 0.316 / (reynolds ** 0.25)
            else:
                return 0.0032 + 0.221 / (reynolds ** 0.237)
        else:
            # Swamee-Jain approximation
            term1 = relative_roughness / 3.7
            term2 = 5.74 / (reynolds ** 0.9)
            log_arg = max(term1 + term2, 1e-10)
            denominator = math.log10(log_arg)
            
            if abs(denominator) < 1e-10:
                return 0.02
                
            return 0.25 / (denominator ** 2)