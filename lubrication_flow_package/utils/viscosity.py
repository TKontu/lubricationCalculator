"""
Centralized fluid property calculations.
"""

import math
from typing import Dict, Optional

# This data can be moved to an external file (e.g., fluids.yaml) in the future.
VISCOSITY_PARAMS = {
    "SAE10": {"A": 0.00004, "B": 950, "C": 135},
    "SAE20": {"A": 0.00006, "B": 1050, "C": 138},
    "SAE30": {"A": 0.0001, "B": 1200, "C": 140},
    "SAE40": {"A": 0.00015, "B": 1300, "C": 142},
    "SAE50": {"A": 0.0002, "B": 1400, "C": 145},
    "SAE60": {"A": 0.00025, "B": 1500, "C": 148},
    "VG220": {"A": 0.000064, "B": 1455, "C": 131},
    "VG320": {"A": 0.000064, "B": 1520, "C": 131},
    "VG460": {"A": 0.000064, "B": 1576, "C": 131}
}

def calculate_viscosity(
    temperature: float,
    oil_type: str,
    viscosity_model: str = 'vogel',
    viscosity_parameters: Optional[Dict] = None
) -> float:
    """
    Calculate dynamic viscosity using the Vogel equation.

    Args:
        temperature: Temperature in Celsius.
        oil_type: The type of oil (e.g., "SAE30").
        viscosity_model: The viscosity model to use (currently only 'vogel').
        viscosity_parameters: Optional dictionary with custom Vogel parameters (A, B, C).

    Returns:
        The dynamic viscosity in Pa·s.
    """
    T = temperature + 273.15  # Convert to Kelvin

    if viscosity_model == 'vogel':
        if viscosity_parameters:
            params = viscosity_parameters
        else:
            if oil_type not in VISCOSITY_PARAMS:
                raise ValueError(f"Oil type '{oil_type}' not supported in internal database.")
            params = VISCOSITY_PARAMS[oil_type]
        
        # Vogel's equation is undefined at or below the pole temperature C.
        # Add a small epsilon to prevent math errors if temperature is too close.
        if T <= params["C"]:
            T = params["C"] + 1e-6
        
        # Vogel's Equation: µ = A * exp(B / (T - C))
        viscosity = params["A"] * math.exp(params["B"] / (T - params["C"]))
        
        # Clamp the viscosity to a reasonable range to avoid numerical instability
        return max(1e-6, min(viscosity, 10.0))
    
    else:
        raise NotImplementedError(f"Viscosity model '{viscosity_model}' is not implemented.")
