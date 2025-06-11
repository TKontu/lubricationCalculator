"""
Connector component - Represents junctions, elbows, reducers, etc.
Enhanced with geometry-based and Reynolds number-dependent loss coefficients.
"""

import math
from typing import Dict, Optional, Union
from .base import FlowComponent, ComponentType, ConnectorType


class LossCoefficientCalculator:
    """Utility class for calculating loss coefficients based on geometry and flow conditions"""
    
    @staticmethod
    def calculate_reynolds_number(velocity: float, diameter: float, 
                                density: float, viscosity: float) -> float:
        """Calculate Reynolds number: Re = ρ * v * D / μ"""
        return density * velocity * diameter / viscosity
    
    @staticmethod
    def elbow_loss_coefficient(angle_deg: float, radius_ratio: float = 1.0, 
                             reynolds_number: float = 1e5) -> float:
        """
        Calculate loss coefficient for elbows based on angle and radius ratio.
        
        Args:
            angle_deg: Bend angle in degrees
            radius_ratio: R/D ratio (bend radius / diameter)
            reynolds_number: Reynolds number
        """
        # Base coefficient for 90-degree elbow
        angle_rad = math.radians(angle_deg)
        
        # Angle factor
        angle_factor = math.sin(angle_rad / 2) ** 0.5
        
        # Radius factor (Crane's formula)
        if radius_ratio < 1:
            radius_ratio = 1  # Minimum for sharp elbow
        
        radius_factor = 0.21 * (radius_ratio ** -0.5) if radius_ratio > 1 else 1.0
        
        # Reynolds number correction for laminar flow
        re_factor = 1.0
        if reynolds_number < 2300:
            re_factor = 64 / reynolds_number * (angle_deg / 90)
        elif reynolds_number < 4000:
            # Transition region
            re_factor = 0.5 + 0.5 * (64 / reynolds_number * (angle_deg / 90))
        
        base_k = 0.3 + 0.6 * angle_factor * radius_factor
        return base_k * re_factor
    
    @staticmethod
    def reducer_loss_coefficient(diameter_ratio: float, taper_angle_deg: float = 60,
                               reynolds_number: float = 1e5) -> float:
        """
        Calculate loss coefficient for reducers/expanders.
        
        Args:
            diameter_ratio: D2/D1 (outlet/inlet diameter ratio)
            taper_angle_deg: Total included angle of taper
            reynolds_number: Reynolds number
        """
        β = diameter_ratio
        area_ratio = β ** 2
        
        if area_ratio < 1:  # Contraction
            if taper_angle_deg <= 45:  # Gradual contraction
                K = 0.8 * (1 - area_ratio) * math.sin(math.radians(taper_angle_deg / 2))
            else:  # Sudden contraction
                K = 0.5 * (1 - area_ratio)
        else:  # Expansion
            # A₁/A₂ = 1/area_ratio
            ar_inv = 1.0 / area_ratio
            
            if taper_angle_deg <= 45:  # gradual expansion
                # empirical Crane‐handbook style
                K = 2.6 * math.sin(math.radians(taper_angle_deg / 2)) * (1 - ar_inv) ** 2
            else:  # sudden expansion
                K = (1 - ar_inv) ** 2
        
        # Reynolds number correction for laminar flow
        if reynolds_number < 2300:
            K *= (64 / reynolds_number) / 0.02  # Normalize to turbulent friction factor
        
        return K
    
    @staticmethod
    def valve_loss_coefficient(valve_type: ConnectorType, opening_fraction: float = 1.0) -> float:
        """
        Calculate loss coefficient for valves based on type and opening.
        
        Args:
            valve_type: Type of valve
            opening_fraction: Fraction open (0-1)
        """
        # Base loss coefficients for fully open valves
        base_coefficients = {
            ConnectorType.GATE_VALVE: 0.15,
            ConnectorType.BALL_VALVE: 0.05,
            ConnectorType.GLOBE_VALVE: 10.0,
            ConnectorType.CHECK_VALVE: 2.0,
            ConnectorType.BUTTERFLY_VALVE: 0.3
        }
        
        base_k = base_coefficients.get(valve_type, 1.0)
        
        # Opening factor (simplified model)
        if opening_fraction <= 0:
            return float('inf')  # Closed valve
        elif opening_fraction >= 1:
            return base_k
        else:
            # Approximate relationship for partially open valves
            return base_k / (opening_fraction ** 2)
    
    @staticmethod
    def junction_loss_coefficient(junction_type: ConnectorType, flow_split_ratio: float = 0.5) -> float:
        """
        Calculate loss coefficient for junctions.
        
        Args:
            junction_type: Type of junction
            flow_split_ratio: Fraction of flow going to branch (0-1)
        """
        base_coefficients = {
            ConnectorType.T_JUNCTION: 1.8,
            ConnectorType.X_JUNCTION: 2.5,
            ConnectorType.WYE_JUNCTION: 1.2,
            ConnectorType.LATERAL_TEE: 2.0
        }
        
        base_k = base_coefficients.get(junction_type, 1.5)
        
        # Adjust for flow split (simplified model)
        split_factor = 1 + abs(flow_split_ratio - 0.5)
        return base_k * split_factor


class Connector(FlowComponent):
    """
    Enhanced connector component with geometry-based and Reynolds number-dependent 
    loss coefficient calculations.
    """
    
    def __init__(self, connector_type: ConnectorType, diameter: float,
                 diameter_out: float = None, loss_coefficient: float = None,
                 component_id: str = None, name: str = "",
                 # Geometric parameters
                 bend_angle: float = 90.0,
                 bend_radius_ratio: float = 1.0,
                 taper_angle: float = 60.0,
                 valve_opening: float = 1.0,
                 flow_split_ratio: float = 0.5,
                 # Advanced options
                 auto_calculate_k: bool = True):
        """
        Initialize connector with enhanced geometric parameters.
        
        Args:
            connector_type: Type of connector
            diameter: Inlet diameter (m)
            diameter_out: Outlet diameter (m), defaults to inlet diameter
            loss_coefficient: Fixed loss coefficient (overrides calculation if provided)
            component_id: Unique identifier
            name: Component name
            bend_angle: Bend angle for elbows (degrees)
            bend_radius_ratio: R/D ratio for bends
            taper_angle: Total included angle for reducers/expanders (degrees)
            valve_opening: Valve opening fraction (0-1)
            flow_split_ratio: Flow split ratio for junctions (0-1)
            auto_calculate_k: Whether to automatically calculate K based on geometry and Re
        """
        super().__init__(component_id, name)
        self.component_type = ComponentType.CONNECTOR
        self.connector_type = connector_type
        self.diameter = diameter  # m (inlet diameter)
        self.diameter_out = diameter_out or diameter  # m (outlet diameter)
        
        # Geometric parameters
        self.bend_angle = bend_angle
        self.bend_radius_ratio = bend_radius_ratio
        self.taper_angle = taper_angle
        self.valve_opening = valve_opening
        self.flow_split_ratio = flow_split_ratio
        self.auto_calculate_k = auto_calculate_k
        
        # Loss coefficient
        self._fixed_loss_coefficient = loss_coefficient
        self.loss_coefficient = loss_coefficient or self._get_default_loss_coefficient()
        
        # Validation
        if diameter <= 0:
            raise ValueError("Connector diameter must be positive")
        if self.diameter_out <= 0:
            raise ValueError("Connector outlet diameter must be positive")
        if not 0 <= valve_opening <= 1:
            raise ValueError("Valve opening must be between 0 and 1")
        if not 0 <= flow_split_ratio <= 1:
            raise ValueError("Flow split ratio must be between 0 and 1")
    
    def _get_default_loss_coefficient(self) -> float:
        """Get default loss coefficient based on connector type"""
        defaults = {
            # Basic connectors
            ConnectorType.T_JUNCTION: 1.8,
            ConnectorType.X_JUNCTION: 2.5,
            ConnectorType.STRAIGHT: 0.0,
            
            # Elbows
            ConnectorType.ELBOW_90: 0.9,
            ConnectorType.ELBOW_45: 0.4,
            ConnectorType.ELBOW_30: 0.2,
            ConnectorType.ELBOW_SMOOTH: 0.3,
            ConnectorType.ELBOW_MITERED: 1.1,
            ConnectorType.LONG_RADIUS_ELBOW: 0.3,
            ConnectorType.SHORT_RADIUS_ELBOW: 0.9,
            
            # Reducers and expanders
            ConnectorType.REDUCER_GRADUAL: 0.3,
            ConnectorType.REDUCER_SUDDEN: 0.5,
            ConnectorType.EXPANDER_GRADUAL: 0.6,
            ConnectorType.EXPANDER_SUDDEN: 1.0,
            
            # Valves
            ConnectorType.GATE_VALVE: 0.15,
            ConnectorType.BALL_VALVE: 0.05,
            ConnectorType.GLOBE_VALVE: 10.0,
            ConnectorType.CHECK_VALVE: 2.0,
            ConnectorType.BUTTERFLY_VALVE: 0.3,
            
            # Fittings
            ConnectorType.UNION: 0.08,
            ConnectorType.COUPLING: 0.04,
            ConnectorType.ADAPTER: 0.1,
            
            # Complex junctions
            ConnectorType.WYE_JUNCTION: 1.2,
            ConnectorType.LATERAL_TEE: 2.0,
            
            # Bends
            ConnectorType.RETURN_BEND: 2.2,
        }
        return defaults.get(self.connector_type, 1.0)
    
    def calculate_loss_coefficient(self, velocity: float, fluid_properties: Dict) -> float:
        """
        Calculate loss coefficient based on geometry and flow conditions.
        
        Args:
            velocity: Flow velocity (m/s)
            fluid_properties: Dictionary containing density and viscosity
            
        Returns:
            Loss coefficient K
        """
        # Use fixed coefficient if provided and auto-calculation is disabled
        if self._fixed_loss_coefficient is not None and not self.auto_calculate_k:
            return self._fixed_loss_coefficient
        
        # Calculate Reynolds number
        density = fluid_properties['density']
        viscosity = fluid_properties.get('viscosity', 1e-3)  # Default to water viscosity
        reynolds_number = LossCoefficientCalculator.calculate_reynolds_number(
            velocity, self.diameter, density, viscosity
        )
        
        # Calculate K based on connector type
        if self.connector_type in [ConnectorType.ELBOW_90, ConnectorType.ELBOW_45, 
                                 ConnectorType.ELBOW_30, ConnectorType.ELBOW_SMOOTH,
                                 ConnectorType.ELBOW_MITERED, ConnectorType.LONG_RADIUS_ELBOW,
                                 ConnectorType.SHORT_RADIUS_ELBOW]:
            return LossCoefficientCalculator.elbow_loss_coefficient(
                self.bend_angle, self.bend_radius_ratio, reynolds_number
            )
        
        elif self.connector_type in [ConnectorType.REDUCER_GRADUAL, ConnectorType.REDUCER_SUDDEN,
                                   ConnectorType.EXPANDER_GRADUAL, ConnectorType.EXPANDER_SUDDEN]:
            diameter_ratio = self.diameter_out / self.diameter
            return LossCoefficientCalculator.reducer_loss_coefficient(
                diameter_ratio, self.taper_angle, reynolds_number
            )
        
        elif self.connector_type in [ConnectorType.GATE_VALVE, ConnectorType.BALL_VALVE,
                                   ConnectorType.GLOBE_VALVE, ConnectorType.CHECK_VALVE,
                                   ConnectorType.BUTTERFLY_VALVE]:
            return LossCoefficientCalculator.valve_loss_coefficient(
                self.connector_type, self.valve_opening
            )
        
        elif self.connector_type in [ConnectorType.T_JUNCTION, ConnectorType.X_JUNCTION,
                                   ConnectorType.WYE_JUNCTION, ConnectorType.LATERAL_TEE]:
            return LossCoefficientCalculator.junction_loss_coefficient(
                self.connector_type, self.flow_split_ratio
            )
        
        else:
            # Use default coefficient for other types
            return self._get_default_loss_coefficient()
    
    def get_flow_area(self) -> float:
        """Get the flow area (use smaller diameter for restriction)"""
        return math.pi * (min(self.diameter, self.diameter_out) / 2) ** 2
    
    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """
        Calculate pressure drop using enhanced loss coefficient method.
        
        Args:
            flow_rate: Volumetric flow rate (m³/s)
            fluid_properties: Dictionary containing density and viscosity
            
        Returns:
            Pressure drop (Pa)
        """
        if flow_rate <= 0:
            return 0
        
        density = fluid_properties['density']
        
        # Use inlet diameter for velocity calculation
        area = math.pi * (self.diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Calculate loss coefficient
        if self.auto_calculate_k:
            k = self.calculate_loss_coefficient(velocity, fluid_properties)
        else:
            k = self.loss_coefficient
        
        # Minor loss equation: ΔP = K * ρ * v² / 2
        pressure_drop = k * density * velocity ** 2 / 2
        
        return pressure_drop
    
    def set_geometric_parameters(self, **kwargs):
        """
        Update geometric parameters.
        
        Accepted parameters:
        - bend_angle: Bend angle for elbows (degrees)
        - bend_radius_ratio: R/D ratio for bends
        - taper_angle: Total included angle for reducers/expanders (degrees)
        - valve_opening: Valve opening fraction (0-1)
        - flow_split_ratio: Flow split ratio for junctions (0-1)
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Unknown geometric parameter: {param}")
    
    def get_connector_info(self) -> Dict:
        """Get comprehensive information about the connector"""
        return {
            'type': self.connector_type.value,
            'diameter_in': self.diameter,
            'diameter_out': self.diameter_out,
            'bend_angle': self.bend_angle,
            'bend_radius_ratio': self.bend_radius_ratio,
            'taper_angle': self.taper_angle,
            'valve_opening': self.valve_opening,
            'flow_split_ratio': self.flow_split_ratio,
            'auto_calculate_k': self.auto_calculate_k,
            'fixed_loss_coefficient': self._fixed_loss_coefficient,
            'current_loss_coefficient': self.loss_coefficient
        }
