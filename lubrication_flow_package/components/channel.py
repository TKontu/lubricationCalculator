"""
Channel component - Represents pipes or drilling channels
"""

import math
from typing import Dict
from .base import FlowComponent, ComponentType
from lubrication_flow_package.utils.friction import churchill_friction_factor
from scipy.optimize import newton

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

    def calculate_flow_rate(self, pressure_drop: float, fluid_properties: Dict) -> float:
        """
        Calculate flow rate for a given pressure drop using a numerical root-finder.
        """
        if pressure_drop == 0:
            return 0.0

        def residual(q):
            # The pressure drop should be signed, so we match its sign
            return self.calculate_pressure_drop(q, fluid_properties) - pressure_drop

        # Initial guess based on a simplified linear model (Poiseuille flow)
        A = self.get_flow_area()
        μ = fluid_properties['viscosity']
        # Use absolute pressure drop for initial guess magnitude
        initial_guess = (abs(pressure_drop) * math.pi * self.diameter**4) / (128 * μ * self.length)
        if pressure_drop < 0:
            initial_guess *= -1

        try:
            flow_rate = newton(residual, initial_guess, tol=1e-6, maxiter=50)
        except (RuntimeError, ValueError):
            # If Newton's method fails, try a more robust bracketing solver
            from scipy.optimize import brentq
            
            # Improved bracketing logic
            if pressure_drop > 0:
                lower_bound = 0.0
                # Start with a reasonable upper bound and expand systematically
                upper_bound = max(initial_guess, 1e-5) * 1.1 
                for _ in range(15):
                    if residual(lower_bound) * residual(upper_bound) < 0:
                        break
                    upper_bound *= 2.0
                else:
                    raise ValueError("Could not bracket the root for positive pressure drop.")
            else: # pressure_drop < 0
                upper_bound = 0.0
                lower_bound = min(initial_guess, -1e-5) * 1.1
                for _ in range(15):
                    if residual(lower_bound) * residual(upper_bound) < 0:
                        break
                    lower_bound *= 2.0
                else:
                    raise ValueError("Could not bracket the root for negative pressure drop.")

            flow_rate = brentq(residual, lower_bound, upper_bound, xtol=1e-6)

        return flow_rate

    def get_differential_resistance(self, flow_rate: float, fluid_properties: Dict) -> float:
        """
        Calculate the differential resistance d(ΔP)/dQ using a finite difference method.
        """
        if flow_rate == 0:
            # For zero flow, resistance is based on laminar flow (Poiseuille's law)
            # ΔP = (128 * μ * L * Q) / (π * D^4) => d(ΔP)/dQ = (128 * μ * L) / (π * D^4)
            μ = fluid_properties['viscosity']
            return (128 * μ * self.length) / (math.pi * self.diameter**4)

        # Use a small perturbation for the finite difference calculation
        # Use a more robust perturbation size that balances accuracy and numerical stability
        delta_q = max(abs(flow_rate) * 1e-6, 1e-9)
        
        p1 = self.calculate_pressure_drop(flow_rate, fluid_properties)
        p2 = self.calculate_pressure_drop(flow_rate + delta_q, fluid_properties)
        
        differential_resistance = 2 * p1 / flow_rate if flow_rate != 0 else 0
        
        return max(differential_resistance, 1e-9)

    @classmethod
    def from_config(cls, config: Dict) -> 'Channel':
        """Create a Channel instance from a configuration dictionary."""
        return Channel(
            length=config['length'],
            diameter=config['diameter'],
            roughness=config.get('roughness', 0.00015),
            component_id=config.get('id'),
            name=config.get('name')
        )
