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
    

class StandardAngleSprayNozzle(Nozzle):
    """Standard-angle spray nozzle (e.g. 95° spray at 40 psi) with empirical data."""
    PSI_TO_PA   = 6894.76
    IN_TO_M     = 0.0254
    GPM_TO_M3S  = 0.0000630902

    # Chart data: size → (orifice dia in inches, flow @ 40 psi in GPM)
    PERFORMANCE: Dict[int, Dict[str, float]] = {
        10:  {"d_in": 0.079, "q40": 1.0},
        15:  {"d_in": 0.094, "q40": 1.5},
        20:  {"d_in": 0.109, "q40": 2.0},
        30:  {"d_in": 0.133, "q40": 3.0},
        40:  {"d_in": 0.153, "q40": 4.0},
        50:  {"d_in": 0.172, "q40": 5.0},
        60:  {"d_in": 0.188, "q40": 6.0},
        70:  {"d_in": 0.203, "q40": 7.0},
        80:  {"d_in": 0.217, "q40": 8.0},
        100: {"d_in": 0.243, "q40": 10.0},
        150: {"d_in": 0.297, "q40": 15.0},
        400: {"d_in": 0.472, "q40": 40.0},
    }

    def __init__(
        self,
        size: int,
        spray_angle: float = 95.0,
        component_id: str = None,
        name: str = ""
    ):
        data = self.PERFORMANCE[size]
        diameter_m = data["d_in"] * self.IN_TO_M
        super().__init__(
            diameter=diameter_m,
            nozzle_type=NozzleType.STANDARD_ANGLE,
            discharge_coeff=None,  # we'll derive from data
            component_id=component_id,
            name=name or f"SprayNozzle_{size}"
        )
        # reference flow at 40 psi:
        self.size = size
        self.spray_angle = spray_angle
        self._q40_m3s = data["q40"] * self.GPM_TO_M3S

    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        """
        Invert Q = Q40 * sqrt(Ppsi/40) → P = 40 * (Q/Q40)^2 [psi].
        Then convert to Pa. Adds any stub‐pipe friction if configured.
        """
        if flow_rate <= 0:
            return 0.0

        # 1) stub‐pipe friction (if any)
        dp_pipe = 0.0
        if hasattr(self, "_stub_channel") and self._stub_channel:
            dp_pipe = self._stub_channel.calculate_pressure_drop(flow_rate, fluid_properties)

        # 2) spray‐orifice loss
        ratio = flow_rate / self._q40_m3s
        p_psi = 40.0 * (ratio ** 2)
        dp_orifice = p_psi * self.PSI_TO_PA
        return dp_pipe + dp_orifice

    def get_flow_rate_for_pressure(self, pressure_drop: float) -> float:
        """
        Q = Q40 * sqrt(Ppsi/40)
        """
        p_psi = pressure_drop / self.PSI_TO_PA
        return self._q40_m3s * math.sqrt(max(p_psi, 0.0) / 40.0)
