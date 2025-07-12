"""
Tests for the centralized viscosity calculation.
"""

import pytest
from lubrication_flow_package.utils.viscosity import calculate_viscosity, VISCOSITY_PARAMS

def test_calculate_viscosity_known_values():
    """
    Tests the calculate_viscosity function with known oil types and temperatures.
    """
    # Test with a known oil type and temperature
    viscosity = calculate_viscosity(temperature=40, oil_type="SAE30")
    assert viscosity == pytest.approx(0.10229, rel=1e-3)

    viscosity = calculate_viscosity(temperature=100, oil_type="SAE30")
    assert viscosity == pytest.approx(0.017189, rel=1e-3)

def test_calculate_viscosity_unsupported_oil():
    """
    Tests that the function raises a ValueError for an unsupported oil type.
    """
    with pytest.raises(ValueError):
        calculate_viscosity(temperature=40, oil_type="UNSUPPORTED_OIL")

def test_calculate_viscosity_custom_params():
    """
    Tests the function with custom viscosity parameters.
    """
    custom_params = {"A": 0.0001, "B": 1200, "C": 140}
    viscosity = calculate_viscosity(
        temperature=40,
        oil_type="CUSTOM",
        viscosity_parameters=custom_params
    )
    assert viscosity == pytest.approx(0.10229, rel=1e-3)
