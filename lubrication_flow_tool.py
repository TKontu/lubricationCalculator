#!/usr/bin/env python3
"""
ALubrication Piping Flow Distribution Calculation Tool

This tool calculates the flow distribution in a piping system with branches and nozzles.
It takes into account:
- Total flow rate
- Temperature (affects viscosity)
- Viscosity
- Piping geometry (diameters of main line and branches)
- Nozzle characteristics (flow restriction coefficients)

The system is analyzed as a whole, so changing a nozzle in one branch affects the entire distribution.
"""

import math

# Constants
GRAVITY = 9.81  # m/s^2
WATER_DENSITY = 1000  # kg/m^3 at 20°C

def calculate_viscosity(temperature, oil_type="SAE30"):
    """
    Calculate dynamic viscosity of lubrication oil based on temperature using the Andrade equation.

    Args:
        temperature (float): Temperature in Celsius
        oil_type (str): Type of oil (default is SAE30)

    Returns:
        float: Dynamic viscosity in Pa·s
    """
    # Convert to Kelvin
    T = temperature + 273.15

    # Constants for the Andrade equation (values are approximate and depend on oil type)
    if oil_type == "SAE30":
        A = 0.0001  # Pa·s (approximate value for SAE 30 oil)
        B = 2500    # K (approximate value for SAE 30 oil)
    else:
        raise ValueError(f"Oil type {oil_type} not supported")

    # Andrade equation: μ = A * e^(B/T)
    viscosity = A * math.exp(B / T)

    return viscosity

def darcy_weisbach(flow_rate, diameter, length, roughness, viscosity):
    """
    Calculate pressure drop using Darcy-Weisbach equation.

    Args:
        flow_rate (float): Volumetric flow rate in m³/s
        diameter (float): Pipe diameter in meters
        length (float): Pipe length in meters
        roughness (float): Pipe roughness in meters
        viscosity (float): Dynamic viscosity in Pa·s

    Returns:
        float: Pressure drop in Pascals
    """
    area = math.pi * (diameter / 2) ** 2
    velocity = flow_rate / area

    # Reynolds number
    reynolds = (velocity * diameter * WATER_DENSITY) / viscosity

    # Relative roughness
    epsilon_d = roughness / diameter

    # Friction factor (using Colebrook-White equation approximation)
    if reynolds < 2000:  # Laminar flow
        friction_factor = 64 / reynolds
    else:  # Turbulent flow
        # Using Swamee-Jain approximation for turbulent flow
        friction_factor = 1.325 / (math.log10(epsilon_d + 5.74 / (reynolds ** 0.9)) ** 2)

    # Darcy-Weisbach equation
    pressure_drop = friction_factor * (length / diameter) * (WATER_DENSITY * velocity**2) / 2

    return pressure_drop

def calculate_branch_flow(main_flow_rate, main_diameter, branch_lengths, branch_diameters,
                         nozzle_kv_values, roughness=0.00015, max_iterations=100, tolerance=1e-6):
    """
    Calculate flow distribution in branches using an iterative approach.

    Args:
        main_flow_rate (float): Total flow rate in the main pipe in m³/s
        main_diameter (float): Diameter of the main pipe in meters
        branch_lengths (list of float): Lengths of each branch in meters
        branch_diameters (list of float): Diameters of each branch in meters
        nozzle_kv_values (list of float): Flow coefficient for each nozzle
        roughness (float): Pipe roughness in meters (default is for steel pipes)
        max_iterations (int): Maximum number of iterations for convergence
        tolerance (float): Tolerance for convergence

    Returns:
        list of float: Flow rates in each branch in m³/s
    """
    num_branches = len(branch_diameters)

    # Initial guess - equal distribution
    branch_flow_rates = [main_flow_rate / num_branches] * num_branches

    for iteration in range(max_iterations):
        # Calculate pressure drops in each branch
        branch_pressure_drops = []
        total_pressure_drop = 0

        for i, (flow_rate, diameter, length) in enumerate(zip(branch_flow_rates, branch_diameters, branch_lengths)):
            # Get viscosity based on flow rate (this is a simplification)
            # In a real scenario, we'd need to consider temperature and oil type
            viscosity = calculate_viscosity(40)  # Using 40°C as an example

            # Calculate pressure drop in this branch using Darcy-Weisbach
            pressure_drop = darcy_weisbach(
                flow_rate,
                diameter,
                length,
                roughness,
                viscosity
            )

            # Add nozzle restriction effect (simplified model)
            if nozzle_kv_values[i] > 0:
                pressure_drop += (flow_rate / (nozzle_kv_values[i] * 1e-6))**2

            branch_pressure_drops.append(pressure_drop)
            total_pressure_drop += pressure_drop

        # Calculate new flow rates based on pressure drops
        new_branch_flow_rates = []
        for i, (flow_rate, pressure_drop) in enumerate(zip(branch_flow_rates, branch_pressure_drops)):
            # Adjust flow rate proportionally to the inverse of pressure drop
            if pressure_drop == 0:
                new_flow = 0
            else:
                new_flow = main_flow_rate * (1 / pressure_drop) / sum(1 / pd for pd in branch_pressure_drops)

            new_branch_flow_rates.append(new_flow)

        # Check for convergence
        if all(abs(new - old) < tolerance for new, old in zip(new_branch_flow_rates, branch_flow_rates)):
            print(f"Converged after {iteration+1} iterations")
            break

        branch_flow_rates = new_branch_flow_rates

    else:
        print(f"Maximum iterations ({max_iterations}) reached without convergence")

    return branch_flow_rates

def main():
    # Example input parameters
    total_flow_rate = 0.01  # m³/s (10 L/s)
    temperature = 40  # °C

    main_pipe_diameter = 0.1  # meters
    main_pipe_length = 10  # meters
    main_pipe_roughness = 0.00015  # meters (typical for steel pipes)

    # Branch parameters
    branch_lengths = [5, 6, 7]  # meters
    branch_diameters = [0.05, 0.04, 0.03]  # meters
    nozzle_kv_values = [20, 15, 10]  # Flow coefficient values

    # Calculate viscosity based on temperature
    viscosity = calculate_viscosity(temperature)
    print(f"Viscosity at {temperature}°C: {viscosity:.6f} Pa·s")

    # Calculate pressure drop in main pipe
    main_pressure_drop = darcy_weisbach(
        total_flow_rate,
        main_pipe_diameter,
        main_pipe_length,
        main_pipe_roughness,
        viscosity
    )

    print(f"Main Pipe Pressure Drop: {main_pressure_drop:.2f} Pa")

    # Calculate flow distribution in branches
    branch_flow_rates = calculate_branch_flow(
        total_flow_rate,
        main_pipe_diameter,
        branch_lengths,
        branch_diameters,
        nozzle_kv_values
    )

    # Print results
    print("\nBranch Flow Rates:")
    for i, flow_rate in enumerate(branch_flow_rates):
        print(f"  Branch {i+1}: {flow_rate:.6f} m³/s ({flow_rate * 1000:.2f} L/s)")

    # Verify total flow
    total_calculated = sum(branch_flow_rates)
    print(f"\nTotal Calculated Flow: {total_calculated:.6f} m³/s")
    print(f"Difference from Input: {(total_calculated - total_flow_rate):.8f} m³/s")

if __name__ == "__main__":
    main()