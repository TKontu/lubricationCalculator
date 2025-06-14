#!/usr/bin/env python3
"""
Verification script to demonstrate the hydraulics test results
"""

import math
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.components.connector import Connector, ConnectorType

# Standard fluid properties (hydraulic oil at 40°C)
STANDARD_FLUID = {
    'density': 850.0,      # kg/m³
    'viscosity': 0.032     # Pa·s (32 cP)
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

def analytical_darcy_weisbach(flow_rate: float, diameter: float, length: float,
                            density: float, viscosity: float, roughness: float = 0.00015) -> float:
    """Reference analytical Darcy-Weisbach calculation"""
    if flow_rate <= 0:
        return 0.0
        
    area = math.pi * (diameter / 2) ** 2
    velocity = flow_rate / area
    reynolds = (density * velocity * diameter) / viscosity
    
    # Churchill's friction factor formula
    relative_roughness = roughness / diameter
    A = (2.457 * math.log(1.0 / ((7.0 / reynolds) ** 0.9 + 0.27 * relative_roughness))) ** 16
    B = (37530.0 / reynolds) ** 16
    term = (8.0 / reynolds) ** 12 + (A + B) ** -1.5
    friction_factor = 8.0 * term ** (1.0 / 12.0)
    
    # Darcy-Weisbach equation: Δp = f * (L/D) * (ρv²/2)
    pressure_drop = friction_factor * (length / diameter) * (density * velocity ** 2) / 2
    
    return pressure_drop

def main():
    print("=== Hydraulics Test Verification ===\n")
    
    # Test 1: Darcy-Weisbach Channel
    print("1. DARCY-WEISBACH CHANNEL TEST")
    print("-" * 40)
    
    test_cases = [
        (100, 25),  # 100 L/min, 25 mm
        (200, 50),  # 200 L/min, 50 mm
        (400, 80),  # 400 L/min, 80 mm
    ]
    
    for flow_lpm, diameter_mm in test_cases:
        flow_rate = convert_flow_rate(flow_lpm)
        diameter = convert_diameter(diameter_mm)
        length = 10.0  # 10 m
        
        # Component calculation
        channel = Channel(diameter=diameter, length=length, roughness=0.00015)
        calculated_dp = channel.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Analytical calculation
        expected_dp = analytical_darcy_weisbach(
            flow_rate, diameter, length,
            STANDARD_FLUID['density'], STANDARD_FLUID['viscosity']
        )
        
        error = abs(calculated_dp - expected_dp)
        error_bar = convert_pressure(error)
        
        print(f"Flow: {flow_lpm} L/min, Diameter: {diameter_mm} mm")
        print(f"  Calculated: {convert_pressure(calculated_dp):.3f} bar")
        print(f"  Expected:   {convert_pressure(expected_dp):.3f} bar")
        print(f"  Error:      {error_bar:.4f} bar (tolerance: ±0.2 bar)")
        print(f"  Status:     {'✓ PASS' if error_bar <= 0.2 else '✗ FAIL'}")
        print()
    
    # Test 2: Orifice/Nozzle
    print("2. ORIFICE/NOZZLE TEST")
    print("-" * 40)
    
    flow_rate = convert_flow_rate(150)  # 150 L/min
    diameter = convert_diameter(12)     # 12 mm
    
    # Sharp-edged orifice
    nozzle = Nozzle(diameter=diameter, nozzle_type=NozzleType.SHARP_EDGED)
    calculated_dp = nozzle.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
    
    # Analytical calculation
    area = math.pi * (diameter / 2) ** 2
    velocity = flow_rate / area
    K = (1 / 0.6 ** 2) - 1  # Cd = 0.6 for sharp-edged
    expected_dp = K * STANDARD_FLUID['density'] * velocity ** 2 / 2
    
    error = abs(calculated_dp - expected_dp)
    error_bar = convert_pressure(error)
    
    print(f"Sharp-edged orifice: {convert_diameter(diameter)*1000:.0f} mm, {flow_rate*60000:.0f} L/min")
    print(f"  Calculated: {convert_pressure(calculated_dp):.3f} bar")
    print(f"  Expected:   {convert_pressure(expected_dp):.3f} bar")
    print(f"  Error:      {error_bar:.4f} bar (tolerance: ±0.2 bar)")
    print(f"  Status:     {'✓ PASS' if error_bar <= 0.2 else '✗ FAIL'}")
    print()
    
    # Venturi
    venturi = Nozzle(diameter=diameter, nozzle_type=NozzleType.VENTURI)
    calculated_dp = venturi.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
    
    # Analytical calculation for venturi
    K_venturi = ((1 / 0.95 ** 2) - 1) * 0.1  # Cd = 0.95, 10% permanent loss
    expected_dp = K_venturi * STANDARD_FLUID['density'] * velocity ** 2 / 2
    
    error = abs(calculated_dp - expected_dp)
    error_bar = convert_pressure(error)
    
    print(f"Venturi nozzle: {convert_diameter(diameter)*1000:.0f} mm, {flow_rate*60000:.0f} L/min")
    print(f"  Calculated: {convert_pressure(calculated_dp):.3f} bar")
    print(f"  Expected:   {convert_pressure(expected_dp):.3f} bar")
    print(f"  Error:      {error_bar:.4f} bar (tolerance: ±0.2 bar)")
    print(f"  Status:     {'✓ PASS' if error_bar <= 0.2 else '✗ FAIL'}")
    print()
    
    # Test 3: Minor Losses
    print("3. MINOR LOSSES TEST")
    print("-" * 40)
    
    flow_rate = convert_flow_rate(120)  # 120 L/min
    diameter = convert_diameter(20)     # 20 mm
    
    test_connectors = [
        (ConnectorType.ELBOW_90, 0.9),
        (ConnectorType.T_JUNCTION, 1.8),
        (ConnectorType.GATE_VALVE, 0.15),
    ]
    
    for connector_type, expected_k in test_connectors:
        connector = Connector(
            connector_type=connector_type,
            diameter=diameter,
            loss_coefficient=expected_k,
            auto_calculate_k=False
        )
        
        calculated_dp = connector.calculate_pressure_drop(flow_rate, STANDARD_FLUID)
        
        # Analytical calculation
        area = math.pi * (diameter / 2) ** 2
        velocity = flow_rate / area
        expected_dp = expected_k * STANDARD_FLUID['density'] * velocity ** 2 / 2
        
        error = abs(calculated_dp - expected_dp)
        error_bar = convert_pressure(error)
        
        print(f"{connector_type.value}: K = {expected_k}")
        print(f"  Calculated: {convert_pressure(calculated_dp):.3f} bar")
        print(f"  Expected:   {convert_pressure(expected_dp):.3f} bar")
        print(f"  Error:      {error_bar:.4f} bar (tolerance: ±0.2 bar)")
        print(f"  Status:     {'✓ PASS' if error_bar <= 0.2 else '✗ FAIL'}")
        print()
    
    print("=== All tests completed ===")

if __name__ == "__main__":
    main()