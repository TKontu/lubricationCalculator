"""
Connector flow component
"""

import math
from typing import Dict
from .base import FlowComponent
from .enums import ComponentType, ConnectorType


class Connector(FlowComponent):
    """Represents a connector (junction, elbow, reducer, etc.)"""
    
    def __init__(self, connector_type: ConnectorType, diameter: float,
                 diameter_out: float = None, loss_coefficient: float = None,
                 component_id: str = None, name: str = ""):
        super().__init__(component_id, name)
        self.component_type = ComponentType.CONNECTOR
        self.connector_type = connector_type
        self.diameter = diameter  # m (inlet diameter)
        self.diameter_out = diameter_out or diameter  # m (outlet diameter)
        
        # Set default loss coefficients if not provided
        if loss_coefficient is None:
            self.loss_coefficient = self._get_default_loss_coefficient()
        else:
            self.loss_coefficient = loss_coefficient
        
        # Validation
        if diameter <= 0:
            raise ValueError("Connector diameter must be positive")
        if self.diameter_out <= 0:
            raise ValueError("Connector outlet diameter must be positive")
    
    def _get_default_loss_coefficient(self) -> float:
        """Get default loss coefficient based on connector type"""
        defaults = {
            ConnectorType.T_JUNCTION: 1.5,    # Branch tee
            ConnectorType.X_JUNCTION: 2.0,    # Cross junction
            ConnectorType.ELBOW_90: 0.9,      # 90-degree elbow
            ConnectorType.REDUCER: 0.5,       # Gradual reducer
            ConnectorType.STRAIGHT: 0.0       # Straight connector
        }
        return defaults.get(self.connector_type, 1.0)
    
    def get_flow_area(self) -> float:
        """Get the flow area (use smaller diameter)"""
        return math.pi * (min(self.diameter, self.diameter_out) / 2) ** 2
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Calculate pressure drop using loss coefficient method"""
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        
        # Use inlet diameter for velocity calculation
        area = math.pi * (self.diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Minor loss equation: ΔP = K * ρ * v² / 2
        pressure_drop = self.loss_coefficient * density * velocity ** 2 / 2
        
        # Add expansion/contraction losses for reducers
        if self.connector_type == ConnectorType.REDUCER and self.diameter != self.diameter_out:
            β = self.diameter_out / self.diameter
            area_ratio = β**2

            if area_ratio < 1:  # sudden contraction
                # loss coefficient Kc = (1 - β²)²
                Kc = (1 - area_ratio)**2
                pressure_drop += Kc * density * velocity**2 / 2
            else:               # sudden expansion
                # loss coefficient Ke = (1 - A1/A2)² = (1 - 1/area_ratio)²
                Ke = (1 - 1/area_ratio)**2
                pressure_drop += Ke * density * velocity**2 / 2
        
        return pressure_drop