"""
Unit tests for core fluid-mechanics functions.

This module contains comprehensive tests for hydraulic calculations including:
1. Darcy-Weisbach pressure drop calculations for pipes (Channel)
2. Orifice/Nozzle pressure drop calculations 
3. Minor losses for connectors (elbows, tees, reducers)

All tests verify against analytical formulas with specified tolerances.
"""

import pytest
import math
from typing import Dict, Tuple

from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.components.connector import Connector, ConnectorType


# Test tolerance constants
PRESSURE_TOLERANCE_PA = 20000  # ±0.2 bar = ±20000 Pa
RELATIVE_TOLERANCE = 0.05      # 5% relative tolerance for edge cases

# Standard fluid properties (typical hydraulic oil at 40°C)
STANDARD_FLUID = {
    'density': 850.0,      # kg/m³
    'viscosity': 0.032     # Pa·s (32 cP)
}

# Water properties for comparison
WATER_FLUID = {
    'density': 1000.0,     # kg/m³  
    'viscosity': 0.001     # Pa·s (1 cP)
}


def convert_flow_rate(flow_lpm: float) -> float:
    """Convert flow rate from L/min to m³/s"""
    return flow_lpm / 60000.0


def convert_diameter(diameter_mm: float) -> float:
    """Convert diameter from mm to m"""
    return diameter_mm / 1000.0


def convert_pressure(pressure_pa: float) -> float:
    """Convert pressure from Pa to bar"""
    return pressure_pa / 100000.0


class AnalyticalFormulas:
    """Reference analytical formulas for verification"""
    
    @staticmethod
    def darcy_weisbach_pressure_drop(flow_rate: float, diameter: float, length: float,
                                   density: float, viscosity: float, roughness: float = 0.00015) -> float:
        """
        Calculate pressure drop using Darcy-Weisbach equation.
        
        Args:
            flow_rate: Volumetric flow rate (m³/s)
            diameter: Pipe diameter (m)
            length: Pipe length (m)
            density: Fluid density (kg/m³)
            viscosity: Dynamic viscosity (Pa·s)
            roughness: Pipe roughness (m)
            
        Returns:
            Pressure drop (Pa)
        """
        if flow_rate <= 0:
            return 0.0
            
        area = math.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        reynolds = (density * velocity * diameter) / viscosity
        
        # Calculate friction factor using Churchill's formula
        relative_roughness = roughness / diameter
        friction_factor = AnalyticalFormulas._churchill_friction_factor(reynolds, relative_roughness)
        
        # Darcy-Weisbach equation: Δp = f * (L/D) * (ρv²/2)
        pressure_drop = friction_factor * (length / diameter) * (density * velocity ** 2) / 2
        
        return pressure_drop
    
    @staticmethod
    def _churchill_friction_factor(reynolds: float, relative_roughness: float) -> float:
        """Calculate friction factor using Churchill's full-range formula"""
        if reynolds <= 0:
            return 0.0
            
        # Churchill's formula
        A = (2.457 * math.log(1.0 / ((7.0 / reynolds) ** 0.9 + 0.27 * relative_roughness))) ** 16
        B = (37530.0 / reynolds) ** 16
        term = (8.0 / reynolds) ** 12 + (A + B) ** -1.5
        friction_factor = 8.0 * term ** (1.0 / 12.0)
        
        return friction_factor
    
    @staticmethod
    def orifice_pressure_drop(flow_rate: float, diameter: float, density: float, 
                            discharge_coeff: float) -> float:
        """
        Calculate orifice pressure drop.
        
        Args:
            flow_rate: Volumetric flow rate (m³/s)
            diameter: Orifice diameter (m)
            density: Fluid density (kg/m³)
            discharge_coeff: Discharge coefficient
            
        Returns:
            Pressure drop (Pa)
        """
        if flow_rate <= 0:
            return 0.0
            
        area = math.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Standard orifice equation: K = (1/Cd²) - 1
        K = (1 / discharge_coeff ** 2) - 1
        
        # Pressure drop: Δp = K * ρv²/2
        pressure_drop = K * density * velocity ** 2 / 2
        
        return pressure_drop
    
    @staticmethod
    def venturi_pressure_drop(flow_rate: float, diameter: float, density: float,
                            discharge_coeff: float) -> float:
        """
        Calculate venturi pressure drop (with recovery).
        
        Args:
            flow_rate: Volumetric flow rate (m³/s)
            diameter: Venturi throat diameter (m)
            density: Fluid density (kg/m³)
            discharge_coeff: Discharge coefficient
            
        Returns:
            Permanent pressure drop (Pa)
        """
        if flow_rate <= 0:
            return 0.0
            
        area = math.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Venturi has lower permanent loss due to diffuser recovery
        K = ((1 / discharge_coeff ** 2) - 1) * 0.1  # 10% permanent loss
        
        # Pressure drop: Δp = K * ρv²/2
        pressure_drop = K * density * velocity ** 2 / 2
        
        return pressure_drop
    
    @staticmethod
    def minor_loss_pressure_drop(flow_rate: float, diameter: float, density: float,
                               loss_coefficient: float) -> float:
        """
        Calculate minor loss pressure drop.
        
        Args:
            flow_rate: Volumetric flow rate (m³/s)
            diameter: Pipe diameter (m)
            density: Fluid density (kg/m³)
            loss_coefficient: Loss coefficient K
            
        Returns:
            Pressure drop (Pa)
        """
        if flow_rate <= 0:
            return 0.0
            
        area = math.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Minor loss equation: Δp = K * ρv²/2
        pressure_drop = loss_coefficient * density * velocity ** 2 / 2
        
        return pressure_drop


class TestChannelDarcyWeisbach:
    """Test Channel.calculate_pressure_drop against Darcy-Weisbach analytical formula"""
    
    @pytest.mark.parametrize("flow_lpm,diameter_mm", [
        # Test matrix: Q = 40-400 L/min, D = 10-80 mm
        (40, 10), (40, 25), (40, 50), (40, 80),
        (100, 10), (100, 25), (100, 50), (100, 80),
        (200, 10), (200, 25), (200, 50), (200, 80),
        (400, 10), (400, 25), (400, 50), (400, 80),
        # Additional edge cases
        (50, 15), (150, 35), (300, 65)
    ])
    def test_darcy_weisbach_pressure_drop(self, flow_lpm: float, diameter_mm: float):
        """
        Test Channel pressure drop calculation against analytical Darcy-Weisbach formula.
        
        Verifies: Δp = f * (L/D) * (ρv²/2) within ±0.2 bar tolerance
        """
        # Convert units
        flow_rate = convert_flow_rate(flow_lpm)
        diameter = convert_diameter(diameter_mm)
        length = 10.0  # 10 meter test length
        
        # Create channel component
        channel = Channel(diameter=diameter, length=length, roughness=0.00015)
        
        # Calculate pressure drop using component
        calculated_dp = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Calculate expected pressure drop using analytical formula
        expected_dp = AnalyticalFormulas.darcy_weisbach_pressure_drop(
            flow_rate, diameter, length, 
            STANDARD_FLUID['density'], STANDARD_FLUID['viscosity']
        )
        
        # Assert within tolerance
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA, (
            f"Pressure drop error {convert_pressure(error):.3f} bar exceeds tolerance "
            f"±{convert_pressure(PRESSURE_TOLERANCE_PA):.1f} bar for "
            f"Q={flow_lpm} L/min, D={diameter_mm} mm. "
            f"Calculated: {convert_pressure(calculated_dp):.3f} bar, "
            f"Expected: {convert_pressure(expected_dp):.3f} bar"
        )
    
    @pytest.mark.parametrize("roughness", [0.0, 0.00015, 0.001, 0.005])
    def test_roughness_effect(self, roughness: float):
        """Test effect of pipe roughness on pressure drop"""
        flow_rate = convert_flow_rate(100)  # 100 L/min
        diameter = convert_diameter(25)     # 25 mm
        length = 5.0                        # 5 m
        
        channel = Channel(diameter=diameter, length=length, roughness=roughness)
        calculated_dp = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        expected_dp = AnalyticalFormulas.darcy_weisbach_pressure_drop(
            flow_rate, diameter, length,
            STANDARD_FLUID['density'], STANDARD_FLUID['viscosity'], roughness
        )
        
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA
    
    def test_zero_flow_rate(self):
        """Test that zero flow rate gives zero pressure drop"""
        channel = Channel(diameter=0.025, length=10.0)
        dp = channel.calculate_pressure_drop(0.0, STANDARD_FLUID)
        assert dp == 0.0
    
    def test_different_fluids(self):
        """Test pressure drop calculation with different fluid properties"""
        flow_rate = convert_flow_rate(150)  # 150 L/min
        diameter = convert_diameter(30)     # 30 mm
        length = 8.0                        # 8 m
        
        channel = Channel(diameter=diameter, length=length)
        
        # Test with hydraulic oil
        dp_oil = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        expected_oil = AnalyticalFormulas.darcy_weisbach_pressure_drop(
            flow_rate, diameter, length,
            STANDARD_FLUID['density'], STANDARD_FLUID['viscosity']
        )
        
        # Test with water
        dp_water = channel.calculate_pressure_drop(flow_rate, WATER_FLUID)
        expected_water = AnalyticalFormulas.darcy_weisbach_pressure_drop(
            flow_rate, diameter, length,
            WATER_FLUID['density'], WATER_FLUID['viscosity']
        )
        
        assert abs(dp_oil - expected_oil) <= PRESSURE_TOLERANCE_PA
        assert abs(dp_water - expected_water) <= PRESSURE_TOLERANCE_PA


class TestNozzleOrifice:
    """Test Nozzle.calculate_pressure_drop for orifices and venturis"""
    
    @pytest.mark.parametrize("flow_lpm,diameter_mm", [
        (50, 5), (50, 10), (50, 15),
        (100, 5), (100, 10), (100, 15),
        (200, 8), (200, 12), (200, 20),
        (300, 10), (300, 15), (300, 25)
    ])
    def test_sharp_edged_orifice(self, flow_lpm: float, diameter_mm: float):
        """
        Test sharp-edged orifice pressure drop calculation.
        
        Verifies: Δp = K * (ρv²/2) where K = (1/Cd²) - 1, Cd = 0.6
        """
        flow_rate = convert_flow_rate(flow_lpm)
        diameter = convert_diameter(diameter_mm)
        
        # Create sharp-edged orifice
        nozzle = Nozzle(diameter=diameter, nozzle_type=NozzleType.SHARP_EDGED)
        
        # Calculate pressure drop
        calculated_dp = nozzle.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Calculate expected pressure drop
        expected_dp = AnalyticalFormulas.orifice_pressure_drop(
            flow_rate, diameter, STANDARD_FLUID['density'], 0.6  # Cd = 0.6
        )
        
        # Assert within tolerance
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA, (
            f"Sharp-edged orifice pressure drop error {convert_pressure(error):.3f} bar "
            f"exceeds tolerance ±{convert_pressure(PRESSURE_TOLERANCE_PA):.1f} bar for "
            f"Q={flow_lpm} L/min, D={diameter_mm} mm. "
            f"Calculated: {convert_pressure(calculated_dp):.3f} bar, "
            f"Expected: {convert_pressure(expected_dp):.3f} bar"
        )
    
    @pytest.mark.parametrize("flow_lpm,diameter_mm", [
        (50, 8), (50, 12), (50, 18),
        (100, 8), (100, 12), (100, 18),
        (200, 10), (200, 15), (200, 22),
        (300, 12), (300, 18), (300, 28)
    ])
    def test_venturi_nozzle(self, flow_lpm: float, diameter_mm: float):
        """
        Test venturi nozzle pressure drop calculation.
        
        Verifies: Δp = K * (ρv²/2) where K = ((1/Cd²) - 1) * 0.1, Cd = 0.95
        """
        flow_rate = convert_flow_rate(flow_lpm)
        diameter = convert_diameter(diameter_mm)
        
        # Create venturi nozzle
        nozzle = Nozzle(diameter=diameter, nozzle_type=NozzleType.VENTURI)
        
        # Calculate pressure drop
        calculated_dp = nozzle.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Calculate expected pressure drop
        expected_dp = AnalyticalFormulas.venturi_pressure_drop(
            flow_rate, diameter, STANDARD_FLUID['density'], 0.95  # Cd = 0.95
        )
        
        # Assert within tolerance
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA, (
            f"Venturi pressure drop error {convert_pressure(error):.3f} bar "
            f"exceeds tolerance ±{convert_pressure(PRESSURE_TOLERANCE_PA):.1f} bar for "
            f"Q={flow_lpm} L/min, D={diameter_mm} mm. "
            f"Calculated: {convert_pressure(calculated_dp):.3f} bar, "
            f"Expected: {convert_pressure(expected_dp):.3f} bar"
        )
    
    @pytest.mark.parametrize("nozzle_type,expected_cd", [
        (NozzleType.SHARP_EDGED, 0.6),
        (NozzleType.ROUNDED, 0.8),
        (NozzleType.VENTURI, 0.95),
        (NozzleType.FLOW_NOZZLE, 0.98)
    ])
    def test_discharge_coefficients(self, nozzle_type: NozzleType, expected_cd: float):
        """Test that nozzles use correct discharge coefficients"""
        diameter = convert_diameter(15)  # 15 mm
        nozzle = Nozzle(diameter=diameter, nozzle_type=nozzle_type)
        
        assert abs(nozzle.discharge_coeff - expected_cd) < 0.01
    
    def test_custom_discharge_coefficient(self):
        """Test nozzle with custom discharge coefficient"""
        diameter = convert_diameter(12)  # 12 mm
        custom_cd = 0.75
        
        nozzle = Nozzle(diameter=diameter, discharge_coeff=custom_cd)
        flow_rate = convert_flow_rate(80)  # 80 L/min
        
        calculated_dp = nozzle.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        expected_dp = AnalyticalFormulas.orifice_pressure_drop(
            flow_rate, diameter, STANDARD_FLUID['density'], custom_cd
        )
        
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA


class TestConnectorMinorLosses:
    """Test Connector.calculate_pressure_drop for minor losses"""
    
    # Published K-values for validation (from Crane Technical Paper 410)
    PUBLISHED_K_VALUES = {
        ConnectorType.ELBOW_90: 0.9,
        ConnectorType.ELBOW_45: 0.4,
        ConnectorType.T_JUNCTION: 1.8,
        ConnectorType.GATE_VALVE: 0.15,
        ConnectorType.BALL_VALVE: 0.05,
        ConnectorType.GLOBE_VALVE: 10.0,
        ConnectorType.CHECK_VALVE: 2.0,
        ConnectorType.REDUCER_SUDDEN: 0.5,
        ConnectorType.EXPANDER_SUDDEN: 1.0
    }
    
    @pytest.mark.parametrize("connector_type,expected_k", [
        (ConnectorType.ELBOW_90, 0.9),
        (ConnectorType.ELBOW_45, 0.4),
        (ConnectorType.T_JUNCTION, 1.8),
        (ConnectorType.GATE_VALVE, 0.15),
        (ConnectorType.BALL_VALVE, 0.05),
        (ConnectorType.GLOBE_VALVE, 10.0),
        (ConnectorType.CHECK_VALVE, 2.0)
    ])
    def test_published_k_values(self, connector_type: ConnectorType, expected_k: float):
        """
        Test connector pressure drop using published K-values.
        
        Verifies: Δp = K * (ρv²/2) within ±0.2 bar tolerance
        """
        diameter = convert_diameter(25)  # 25 mm
        flow_rate = convert_flow_rate(100)  # 100 L/min
        
        # Create connector with fixed K-value (disable auto-calculation)
        connector = Connector(
            connector_type=connector_type,
            diameter=diameter,
            loss_coefficient=expected_k,
            auto_calculate_k=False
        )
        
        # Calculate pressure drop
        calculated_dp = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Calculate expected pressure drop using analytical formula
        expected_dp = AnalyticalFormulas.minor_loss_pressure_drop(
            flow_rate, diameter, STANDARD_FLUID['density'], expected_k
        )
        
        # Assert within tolerance
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA, (
            f"Minor loss pressure drop error {convert_pressure(error):.3f} bar "
            f"exceeds tolerance ±{convert_pressure(PRESSURE_TOLERANCE_PA):.1f} bar for "
            f"{connector_type.value} with K={expected_k}. "
            f"Calculated: {convert_pressure(calculated_dp):.3f} bar, "
            f"Expected: {convert_pressure(expected_dp):.3f} bar"
        )
    
    @pytest.mark.parametrize("flow_lpm,diameter_mm", [
        (50, 15), (50, 25), (50, 40),
        (100, 15), (100, 25), (100, 40),
        (200, 20), (200, 30), (200, 50),
        (300, 25), (300, 35), (300, 60)
    ])
    def test_elbow_90_various_conditions(self, flow_lpm: float, diameter_mm: float):
        """Test 90-degree elbow under various flow and diameter conditions"""
        flow_rate = convert_flow_rate(flow_lpm)
        diameter = convert_diameter(diameter_mm)
        
        connector = Connector(
            connector_type=ConnectorType.ELBOW_90,
            diameter=diameter,
            loss_coefficient=0.9,
            auto_calculate_k=False
        )
        
        calculated_dp = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        expected_dp = AnalyticalFormulas.minor_loss_pressure_drop(
            flow_rate, diameter, STANDARD_FLUID['density'], 0.9
        )
        
        error = abs(calculated_dp - expected_dp)
        assert error <= PRESSURE_TOLERANCE_PA
    
    def test_reducer_diameter_ratio_effect(self):
        """Test reducer with different diameter ratios"""
        inlet_diameter = convert_diameter(30)   # 30 mm
        outlet_diameter = convert_diameter(20)  # 20 mm
        flow_rate = convert_flow_rate(120)      # 120 L/min
        
        # Create reducer
        connector = Connector(
            connector_type=ConnectorType.REDUCER_SUDDEN,
            diameter=inlet_diameter,
            diameter_out=outlet_diameter,
            auto_calculate_k=True  # Let it calculate K based on geometry
        )
        
        calculated_dp = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Should be positive pressure drop
        assert calculated_dp > 0
        
        # Should be reasonable magnitude (not too high or too low)
        assert convert_pressure(calculated_dp) < 5.0  # Less than 5 bar
    
    def test_valve_opening_effect(self):
        """Test valve pressure drop with different opening fractions"""
        diameter = convert_diameter(20)  # 20 mm
        flow_rate = convert_flow_rate(80)  # 80 L/min
        
        # Test fully open valve
        valve_open = Connector(
            connector_type=ConnectorType.GATE_VALVE,
            diameter=diameter,
            valve_opening=1.0,
            auto_calculate_k=True
        )
        
        # Test partially open valve
        valve_half = Connector(
            connector_type=ConnectorType.GATE_VALVE,
            diameter=diameter,
            valve_opening=0.5,
            auto_calculate_k=True
        )
        
        dp_open = valve_open.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        dp_half = valve_half.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Partially open valve should have higher pressure drop
        assert dp_half > dp_open
        
        # Both should be positive
        assert dp_open > 0
        assert dp_half > 0
    
    def test_zero_flow_rate_connectors(self):
        """Test that zero flow rate gives zero pressure drop for connectors"""
        connector = Connector(
            connector_type=ConnectorType.ELBOW_90,
            diameter=convert_diameter(25)
        )
        
        dp = connector.calculate_pressure_drop(0.0, STANDARD_FLUID)
        assert dp == 0.0
    
    @pytest.mark.parametrize("connector_type", [
        ConnectorType.T_JUNCTION,
        ConnectorType.X_JUNCTION,
        ConnectorType.WYE_JUNCTION,
        ConnectorType.LATERAL_TEE
    ])
    def test_junction_types(self, connector_type: ConnectorType):
        """Test various junction types"""
        diameter = convert_diameter(25)  # 25 mm
        flow_rate = convert_flow_rate(150)  # 150 L/min
        
        connector = Connector(
            connector_type=connector_type,
            diameter=diameter,
            auto_calculate_k=True
        )
        
        dp = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Should have positive pressure drop
        assert dp > 0
        
        # Should be reasonable magnitude
        assert convert_pressure(dp) < 10.0  # Less than 10 bar


class TestIntegrationAndEdgeCases:
    """Integration tests and edge case validation"""
    
    def test_component_consistency(self):
        """Test that all components handle the same inputs consistently"""
        flow_rate = convert_flow_rate(100)  # 100 L/min
        diameter = convert_diameter(20)     # 20 mm
        
        # Create components
        channel = Channel(diameter=diameter, length=1.0)  # 1 m length
        nozzle = Nozzle(diameter=diameter, nozzle_type=NozzleType.SHARP_EDGED)
        connector = Connector(connector_type=ConnectorType.ELBOW_90, diameter=diameter)
        
        # All should handle zero flow
        assert channel.calculate_pressure_drop(0.0, STANDARD_FLUID) == 0.0
        assert nozzle.calculate_pressure_drop(0.0, STANDARD_FLUID) == 0.0
        assert connector.calculate_pressure_drop(0.0, STANDARD_FLUID) == 0.0
        
        # All should handle positive flow
        dp_channel = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        dp_nozzle = nozzle.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        dp_connector = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        assert dp_channel > 0
        assert dp_nozzle > 0
        assert dp_connector > 0
    
    def test_high_flow_rates(self):
        """Test components at high flow rates"""
        high_flow = convert_flow_rate(500)  # 500 L/min
        diameter = convert_diameter(50)     # 50 mm
        
        channel = Channel(diameter=diameter, length=5.0)
        dp = channel.calculate_pressure_drop(high_flow, STANDARD_FLUID)
        
        # Should handle high flow without errors
        assert dp > 0
        assert not math.isnan(dp)
        assert not math.isinf(dp)
    
    def test_small_diameters(self):
        """Test components with small diameters"""
        flow_rate = convert_flow_rate(20)   # 20 L/min
        small_diameter = convert_diameter(5)  # 5 mm
        
        channel = Channel(diameter=small_diameter, length=2.0)
        dp = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Should handle small diameter without errors
        assert dp > 0
        assert not math.isnan(dp)
        assert not math.isinf(dp)
    
    def test_pressure_units_consistency(self):
        """Verify pressure calculations are in correct units (Pa)"""
        flow_rate = convert_flow_rate(100)  # 100 L/min
        diameter = convert_diameter(25)     # 25 mm
        
        channel = Channel(diameter=diameter, length=10.0)  # 10 m
        dp_pa = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        dp_bar = convert_pressure(dp_pa)
        
        # Pressure drop should be reasonable in bar (0.1 to 10 bar range)
        assert 0.01 < dp_bar < 20.0
        
        # Should be much larger in Pa
        assert dp_pa > 1000  # At least 1000 Pa


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])