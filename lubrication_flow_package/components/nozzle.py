"""
Nozzle component - Represents flow nozzles or orifices
"""

import math
from typing import Dict
from .base import FlowComponent, ComponentType, NozzleType
from .channel import Channel
from lubrication_flow_package.utils.friction import churchill_friction_factor


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
        
        stub_length = 0.000  # m, tune per your nozzle geometry
        if stub_length > 0:
            self._stub_channel = Channel(
                diameter=self.diameter,
                length=stub_length,
                roughness=getattr(self, "roughness", 0.00015)
            )
        else:
            self._stub_channel = None

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
            NozzleType.FLOW_NOZZLE: 0.98,
            NozzleType.STANDARD_ANGLE: 0.9, # A reasonable default
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
        
        # 1) stub‐pipe friction if configured
        dp_pipe = 0.0
        if self._stub_channel is not None:
            dp_pipe = self._stub_channel.calculate_pressure_drop(
                flow_rate, fluid_properties
            )

        # 2) orifice/minor loss
        density = fluid_properties['density']
        area = self.get_flow_area()
        velocity = flow_rate / area
        if self.nozzle_type == NozzleType.VENTURI:
            K = ((1.0 / self.discharge_coeff ** 2) - 1.0) * 0.1
        else:
            K = (1.0 / self.discharge_coeff ** 2) - 1.0
        dp_minor = K * density * velocity * velocity / 2.0

        return dp_pipe + dp_minor

    def calculate_flow_rate(self, pressure_drop: float, fluid_properties: Dict) -> float:
        """
        Calculate flow rate for a given pressure drop.
        This is the inverse of the pressure drop calculation.
        """
        if pressure_drop <= 0:
            return 0.0

        density = fluid_properties['density']
        area = self.get_flow_area()

        if self.nozzle_type == NozzleType.VENTURI:
            K = ((1.0 / self.discharge_coeff ** 2) - 1.0) * 0.1
        else:
            K = (1.0 / self.discharge_coeff ** 2) - 1.0

        if K <= 0:
            return float('inf')

        # Invert the orifice equation: dP = K * rho * Q^2 / (2 * A^2)
        # Q = A * sqrt(2 * dP / (K * rho))
        flow_rate = area * math.sqrt(2 * pressure_drop / (K * density))
        return flow_rate

    def get_differential_resistance(self, flow_rate: float, fluid_properties: Dict) -> float:
        """
        Calculate the differential resistance d(ΔP)/dQ.
        For a nozzle, ΔP = K * Q^2, so d(ΔP)/dQ = 2 * K * Q.
        """
        if flow_rate == 0:
            return 1e-9 # Avoid division by zero, return a small resistance

        pressure_drop = self.calculate_pressure_drop(flow_rate, fluid_properties)
        
        # d(ΔP)/dQ = 2 * ΔP / Q
        differential_resistance = 2 * pressure_drop / flow_rate if flow_rate != 0 else 0
        
        return max(differential_resistance, 1e-9)

    @classmethod
    def from_config(cls, config: Dict) -> 'Nozzle':
        """Create a Nozzle instance from a configuration dictionary."""
        nozzle_type_str = config.get('nozzle_type')
        if nozzle_type_str:
            nozzle_type = NozzleType(nozzle_type_str)
            if nozzle_type == NozzleType.STANDARD_ANGLE:
                return StandardAngleSprayNozzle.from_config(config)

        # Default to a standard Nozzle if not a special type
        return Nozzle(
            diameter=config['diameter'],
            nozzle_type=NozzleType(config.get('nozzle_type', 'sharp_edged')),
            discharge_coeff=config.get('discharge_coeff'),
            component_id=config.get('id'),
            name=config.get('name')
        )
    

class StandardAngleSprayNozzle(Nozzle):
    """Standard-angle spray nozzle (e.g. 95° spray at 3 bar) with empirical data."""
    MM_TO_M = 0.001
    LPM_TO_M3S = 1 / 60000  # 1 L/min = 1/60000 m³/s
    REF_PRESSURE_PA = 3e5  # 3 bar in Pascals

    # Chart data: size → (orifice dia in mm, flow @ 3 bar in L/min)
    PERFORMANCE: Dict[int, Dict[str, float]] = {
        10:  {"d_mm": 2.0, "q_lpm": 3.9},
        15:  {"d_mm": 2.4, "q_lpm": 5.9},
        20:  {"d_mm": 2.8, "q_lpm": 7.9},
        30:  {"d_mm": 3.4, "q_lpm": 11.8},
        40:  {"d_mm": 3.9, "q_lpm": 15.8},
        50:  {"d_mm": 4.4, "q_lpm": 19.7},
        60:  {"d_mm": 4.8, "q_lpm": 24.0},
        70:  {"d_mm": 5.2, "q_lpm": 28.0},
        80:  {"d_mm": 5.5, "q_lpm": 32.0},
        100: {"d_mm": 6.2, "q_lpm": 39.0},
        150: {"d_mm": 7.5, "q_lpm": 59.0},
        400: {"d_mm": 12.0, "q_lpm": 158.0},
    }

    def __init__(
        self,
        size: int,
        spray_angle: float = 95.0,
        component_id: str = None,
        name: str = ""
    ):
        data = self.PERFORMANCE[size]
        diameter_m = data["d_mm"] * self.MM_TO_M
        super().__init__(
            diameter=diameter_m,
            nozzle_type=NozzleType.STANDARD_ANGLE,
            discharge_coeff=None,
            component_id=component_id,
            name=name or f"SprayNozzle_{size}"
        )
        # Reference flow at 3 bar in m³/s:
        self.size = size
        self.spray_angle = spray_angle
        self._q3bar_m3s = data["q_lpm"] * self.LPM_TO_M3S

    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """Invert Q = Q_ref * sqrt(P / Pref) → P = Pref * (Q / Q_ref)^2."""
        if flow_rate <= 0:
            return 0.0

        dp_pipe = 0.0
        if hasattr(self, "_stub_channel") and self._stub_channel:
            dp_pipe = self._stub_channel.calculate_pressure_drop(flow_rate, fluid_properties)

        ratio = flow_rate / self._q3bar_m3s
        dp_orifice = self.REF_PRESSURE_PA * (ratio ** 2)
        return dp_pipe + dp_orifice

    def get_flow_rate_for_pressure(self, pressure_drop: float) -> float:
        """Q = Q_ref * sqrt(P / Pref)"""
        return self._q3bar_m3s * math.sqrt(max(pressure_drop, 0.0) / self.REF_PRESSURE_PA)

    def calculate_flow_rate(self, pressure_drop: float, fluid_properties: Dict) -> float:
        """
        Calculate flow rate for a given pressure drop.
        """
        return self.get_flow_rate_for_pressure(pressure_drop)

    @classmethod
    def from_config(cls, config: Dict) -> 'StandardAngleSprayNozzle':
        """Create a StandardAngleSprayNozzle instance from a configuration dictionary."""
        return StandardAngleSprayNozzle(
            size=config['size'],
            spray_angle=config.get('spray_angle', 95.0),
            component_id=config.get('id'),
            name=config.get('name')
        )

