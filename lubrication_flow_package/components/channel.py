"""
Channel component - Represents pipes or drilling channels
"""

import math
from typing import Dict
from .base import FlowComponent, ComponentType
from lubrication_flow_package.utils.friction import churchill_friction_factor

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
    
    def calculate_pressure_drop(self, Q: float, fluid: dict) -> float:
        """
        Unified ΔP–Q via Darcy–Weisbach with Churchill’s f(Re),
        falling back to 64/Re for laminar.
        """
        if Q == 0:
            return 0.0

        ρ = fluid['density']
        μ = fluid['viscosity']
        # cross‐sectional area
        A = math.pi * (self.diameter / 2.0)**2
        V = Q / A
        Re = ρ * abs(V) * self.diameter / μ

        # friction factor
        if Re < 2000:
            f = 64.0 / Re
        else:
            f = churchill_friction_factor(Re, self.roughness / self.diameter)

        # Darcy–Weisbach pressure drop (signed)
        dp_pipe = f * (self.length / self.diameter) * ρ * V * abs(V) / 2.0
        return dp_pipe
