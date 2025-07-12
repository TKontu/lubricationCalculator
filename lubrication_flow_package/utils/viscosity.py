"""
Centralized fluid property calculations.
"""

import math
from typing import Dict, Optional
import yaml
import os

import yaml
import os

def load_fluid_properties() -> Dict:
    """Loads fluid properties from the fluids.yaml file."""
    # Construct an absolute path to the config file
    # Assuming the script is run from the root of the project
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'fluids.yaml')
    if not os.path.exists(config_path):
        # Fallback for different execution contexts (like tests)
        config_path = os.path.join(os.getcwd(), 'config', 'fluids.yaml')
        if not os.path.exists(config_path):
             config_path = os.path.join(os.getcwd(), '..', 'config', 'fluids.yaml')
             if not os.path.exists(config_path):
                raise FileNotFoundError("Could not find fluids.yaml in expected locations.")

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('fluids', {})

VISCOSITY_PARAMS = load_fluid_properties()

def calculate_viscosity(
    temperature: float,
    oil_type: str,
    viscosity_model: str = 'vogel'
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
