#!/usr/bin/env python3
"""
AImproved Lubrication Piping Flow Distribution Calculation Tool

This tool calculates the flow distribution in a piping system with branches and nozzles
using proper hydraulic network analysis principles.

Key improvements:
- Proper hydraulic network analysis with equal pressure drops at junctions
- Correct nozzle flow coefficient implementation
- Oil density and viscosity properties
- Robust iterative solver with proper convergence criteria
- Support for complex branching geometries
- Temperature-dependent fluid properties

Physics principles implemented:
1. Conservation of mass (continuity equation)
2. Equal pressure drops across parallel branches at junctions
3. Darcy-Weisbach equation for pipe friction losses
4. Orifice flow equations for nozzle restrictions
5. Temperature-dependent viscosity using Vogel equation
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class PipeSegment:
    """Represents a pipe segment with its properties"""
    diameter: float  # meters
    length: float    # meters
    roughness: float = 0.00015  # meters (default for steel)
    
    
@dataclass
class Nozzle:
    """Represents a nozzle with flow restriction"""
    diameter: float      # meters
    discharge_coeff: float = 0.6  # typical for sharp-edged orifice
    
    
@dataclass
class Branch:
    """Represents a complete branch with pipe and optional nozzle"""
    pipe: PipeSegment
    nozzle: Optional[Nozzle] = None
    name: str = ""


class LubricationFlowCalculator:
    """Main calculator class for lubrication flow distribution"""
    
    def __init__(self, oil_density: float = 900.0, oil_type: str = "SAE30"):
        """
        Initialize the calculator
        
        Args:
            oil_density: Oil density in kg/m³ (typical range 850-950 for lubricating oils)
            oil_type: Type of oil for viscosity calculations
        """
        self.oil_density = oil_density
        self.oil_type = oil_type
        self.gravity = 9.81  # m/s²
        
    def calculate_viscosity(self, temperature: float) -> float:
        """
        Calculate dynamic viscosity using Vogel equation (more accurate than Andrade)
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Dynamic viscosity in Pa·s
        """
        # Convert to Kelvin
        T = temperature + 273.15
        
        # Vogel equation parameters for different oil types
        # μ = A * exp(B / (T - C))
        if self.oil_type == "SAE30":
            A = 0.0001    # Pa·s
            B = 1200      # K
            C = 140       # K
        elif self.oil_type == "SAE10":
            A = 0.00005
            B = 1000
            C = 140
        elif self.oil_type == "SAE50":
            A = 0.0002
            B = 1400
            C = 140
        else:
            raise ValueError(f"Oil type {self.oil_type} not supported")
        
        # Vogel equation
        viscosity = A * math.exp(B / (T - C))
        return viscosity
    
    def calculate_reynolds_number(self, velocity: float, diameter: float, 
                                viscosity: float) -> float:
        """Calculate Reynolds number"""
        return (self.oil_density * velocity * diameter) / viscosity
    
    def calculate_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """
        Calculate friction factor using appropriate correlations
        
        Args:
            reynolds: Reynolds number
            relative_roughness: ε/D ratio
            
        Returns:
            Darcy friction factor
        """
        if reynolds < 2300:  # Laminar flow
            return 64 / reynolds
        elif reynolds < 4000:  # Transition region
            # Linear interpolation between laminar and turbulent
            f_lam = 64 / 2300
            f_turb = self._turbulent_friction_factor(4000, relative_roughness)
            return f_lam + (f_turb - f_lam) * (reynolds - 2300) / (4000 - 2300)
        else:  # Turbulent flow
            return self._turbulent_friction_factor(reynolds, relative_roughness)
    
    def _turbulent_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate turbulent friction factor using Colebrook-White equation"""
        # Using Swamee-Jain approximation for computational efficiency
        if relative_roughness == 0:  # Smooth pipe
            return 0.316 / (reynolds ** 0.25)
        else:
            numerator = math.log10(relative_roughness / 3.7 + 5.74 / (reynolds ** 0.9))
            return 0.25 / (numerator ** 2)
    
    def calculate_pipe_pressure_drop(self, flow_rate: float, pipe: PipeSegment, 
                                   viscosity: float) -> float:
        """
        Calculate pressure drop through a pipe using Darcy-Weisbach equation
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            pipe: PipeSegment object
            viscosity: Dynamic viscosity in Pa·s
            
        Returns:
            Pressure drop in Pascals
        """
        if flow_rate <= 0:
            return 0
            
        area = math.pi * (pipe.diameter / 2) ** 2
        velocity = flow_rate / area
        reynolds = self.calculate_reynolds_number(velocity, pipe.diameter, viscosity)
        relative_roughness = pipe.roughness / pipe.diameter
        friction_factor = self.calculate_friction_factor(reynolds, relative_roughness)
        
        # Darcy-Weisbach equation
        pressure_drop = friction_factor * (pipe.length / pipe.diameter) * \
                       (self.oil_density * velocity ** 2) / 2
        
        return pressure_drop
    
    def calculate_nozzle_pressure_drop(self, flow_rate: float, nozzle: Nozzle) -> float:
        """
        Calculate pressure drop through a nozzle using orifice flow equation
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            nozzle: Nozzle object
            
        Returns:
            Pressure drop in Pascals
        """
        if flow_rate <= 0:
            return 0
            
        area = math.pi * (nozzle.diameter / 2) ** 2
        velocity = flow_rate / area
        
        # Orifice pressure drop: ΔP = (1/Cd² - 1) * ρ * v² / 2
        # For sharp-edged orifice, this simplifies to approximately ΔP = K * ρ * v² / 2
        K = (1 / nozzle.discharge_coeff ** 2) - 1
        pressure_drop = K * self.oil_density * velocity ** 2 / 2
        
        return pressure_drop
    
    def calculate_branch_pressure_drop(self, flow_rate: float, branch: Branch, 
                                     viscosity: float) -> float:
        """
        Calculate total pressure drop through a branch (pipe + nozzle)
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            branch: Branch object
            viscosity: Dynamic viscosity in Pa·s
            
        Returns:
            Total pressure drop in Pascals
        """
        pipe_dp = self.calculate_pipe_pressure_drop(flow_rate, branch.pipe, viscosity)
        nozzle_dp = 0
        
        if branch.nozzle:
            nozzle_dp = self.calculate_nozzle_pressure_drop(flow_rate, branch.nozzle)
            
        return pipe_dp + nozzle_dp
    
    def solve_flow_distribution(self, total_flow_rate: float, branches: List[Branch], 
                              temperature: float, max_iterations: int = 100, 
                              tolerance: float = 1e-6) -> Tuple[List[float], Dict]:
        """
        Solve flow distribution using iterative method based on equal pressure drops
        
        Args:
            total_flow_rate: Total flow rate in m³/s
            branches: List of Branch objects
            temperature: Temperature in Celsius
            max_iterations: Maximum iterations for convergence
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (branch_flow_rates, solution_info)
        """
        viscosity = self.calculate_viscosity(temperature)
        num_branches = len(branches)
        
        # Initial guess - equal distribution
        branch_flows = np.array([total_flow_rate / num_branches] * num_branches)
        
        solution_info = {
            'converged': False,
            'iterations': 0,
            'viscosity': viscosity,
            'temperature': temperature,
            'pressure_drops': [],
            'reynolds_numbers': []
        }
        
        for iteration in range(max_iterations):
            # Calculate pressure drops for current flow distribution
            pressure_drops = []
            for i, (flow, branch) in enumerate(zip(branch_flows, branches)):
                dp = self.calculate_branch_pressure_drop(flow, branch, viscosity)
                pressure_drops.append(dp)
            
            # Target pressure drop (average of all branches)
            target_dp = np.mean(pressure_drops)
            
            # Adjust flows to equalize pressure drops
            new_flows = np.zeros(num_branches)
            
            for i, (flow, branch, dp) in enumerate(zip(branch_flows, branches, pressure_drops)):
                if dp > 0:
                    # Use Newton-Raphson-like approach to adjust flow
                    # For small changes: ΔP ≈ (∂ΔP/∂Q) * ΔQ
                    # We want new_dp = target_dp, so ΔQ = (target_dp - dp) / (∂ΔP/∂Q)
                    
                    # Estimate derivative numerically
                    delta_q = flow * 0.001  # 0.1% change
                    if delta_q > 0:
                        dp_plus = self.calculate_branch_pressure_drop(flow + delta_q, branch, viscosity)
                        dpdq = (dp_plus - dp) / delta_q
                        
                        if abs(dpdq) > 1e-12:  # Avoid division by zero
                            flow_correction = (target_dp - dp) / dpdq
                            new_flows[i] = max(0, flow + 0.5 * flow_correction)  # Damping factor
                        else:
                            new_flows[i] = flow
                    else:
                        new_flows[i] = total_flow_rate / num_branches
                else:
                    new_flows[i] = total_flow_rate / num_branches
            
            # Normalize to maintain total flow
            flow_sum = np.sum(new_flows)
            if flow_sum > 0:
                new_flows = new_flows * total_flow_rate / flow_sum
            
            # Check convergence
            flow_change = np.max(np.abs(new_flows - branch_flows))
            if flow_change < tolerance:
                solution_info['converged'] = True
                solution_info['iterations'] = iteration + 1
                break
                
            branch_flows = new_flows.copy()
        
        # Calculate final pressure drops and Reynolds numbers
        final_pressure_drops = []
        reynolds_numbers = []
        
        for i, (flow, branch) in enumerate(zip(branch_flows, branches)):
            dp = self.calculate_branch_pressure_drop(flow, branch, viscosity)
            final_pressure_drops.append(dp)
            
            if flow > 0:
                area = math.pi * (branch.pipe.diameter / 2) ** 2
                velocity = flow / area
                re = self.calculate_reynolds_number(velocity, branch.pipe.diameter, viscosity)
                reynolds_numbers.append(re)
            else:
                reynolds_numbers.append(0)
        
        solution_info['pressure_drops'] = final_pressure_drops
        solution_info['reynolds_numbers'] = reynolds_numbers
        
        return branch_flows.tolist(), solution_info
    
    def print_results(self, branch_flows: List[float], branches: List[Branch], 
                     solution_info: Dict):
        """Print detailed results of the flow calculation"""
        print(f"\n{'='*60}")
        print("LUBRICATION FLOW DISTRIBUTION RESULTS")
        print(f"{'='*60}")
        
        print(f"Temperature: {solution_info['temperature']:.1f}°C")
        print(f"Oil Type: {self.oil_type}")
        print(f"Oil Density: {self.oil_density:.1f} kg/m³")
        print(f"Dynamic Viscosity: {solution_info['viscosity']:.6f} Pa·s")
        print(f"Converged: {solution_info['converged']} (in {solution_info['iterations']} iterations)")
        
        print(f"\n{'Branch':<10} {'Flow Rate':<12} {'Pressure Drop':<15} {'Reynolds':<10} {'Flow Type'}")
        print(f"{'Name':<10} {'(L/s)':<12} {'(Pa)':<15} {'Number':<10}")
        print("-" * 60)
        
        total_flow = 0
        for i, (flow, branch) in enumerate(zip(branch_flows, branches)):
            name = branch.name if branch.name else f"Branch {i+1}"
            flow_lps = flow * 1000  # Convert to L/s
            pressure_drop = solution_info['pressure_drops'][i]
            reynolds = solution_info['reynolds_numbers'][i]
            
            # Determine flow type
            if reynolds < 2300:
                flow_type = "Laminar"
            elif reynolds < 4000:
                flow_type = "Transition"
            else:
                flow_type = "Turbulent"
            
            print(f"{name:<10} {flow_lps:<12.3f} {pressure_drop:<15.1f} {reynolds:<10.0f} {flow_type}")
            total_flow += flow
        
        print("-" * 60)
        print(f"{'Total':<10} {total_flow*1000:<12.3f}")
        
        # Print branch details
        print(f"\n{'BRANCH DETAILS'}")
        print("-" * 40)
        for i, branch in enumerate(branches):
            name = branch.name if branch.name else f"Branch {i+1}"
            print(f"{name}:")
            print(f"  Pipe: D={branch.pipe.diameter*1000:.1f}mm, L={branch.pipe.length:.1f}m")
            if branch.nozzle:
                print(f"  Nozzle: D={branch.nozzle.diameter*1000:.1f}mm, Cd={branch.nozzle.discharge_coeff:.2f}")
            print()


def create_example_system() -> Tuple[float, List[Branch], float]:
    """Create an example lubrication system for demonstration"""
    
    # System parameters
    total_flow_rate = 0.01  # m³/s (10 L/s)
    temperature = 40  # °C
    
    # Define branches
    branches = [
        Branch(
            pipe=PipeSegment(diameter=0.05, length=5.0),  # 50mm diameter, 5m long
            nozzle=Nozzle(diameter=0.008, discharge_coeff=0.6),  # 8mm nozzle
            name="Main Bearing"
        ),
        Branch(
            pipe=PipeSegment(diameter=0.04, length=6.0),  # 40mm diameter, 6m long
            nozzle=Nozzle(diameter=0.006, discharge_coeff=0.6),  # 6mm nozzle
            name="Aux Bearing"
        ),
        Branch(
            pipe=PipeSegment(diameter=0.03, length=7.0),  # 30mm diameter, 7m long
            nozzle=Nozzle(diameter=0.004, discharge_coeff=0.6),  # 4mm nozzle
            name="Gear Box"
        ),
        Branch(
            pipe=PipeSegment(diameter=0.025, length=4.0),  # 25mm diameter, 4m long
            nozzle=None,  # No nozzle restriction
            name="Cooler Return"
        )
    ]
    
    return total_flow_rate, branches, temperature


def main():
    """Main function demonstrating the lubrication flow calculator"""
    
    # Create calculator instance
    calculator = LubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
    
    # Create example system
    total_flow_rate, branches, temperature = create_example_system()
    
    print("Calculating lubrication flow distribution...")
    print(f"Total flow rate: {total_flow_rate * 1000:.1f} L/s")
    print(f"Number of branches: {len(branches)}")
    
    # Solve flow distribution
    branch_flows, solution_info = calculator.solve_flow_distribution(
        total_flow_rate, branches, temperature
    )
    
    # Print results
    calculator.print_results(branch_flows, branches, solution_info)
    
    # Demonstrate effect of changing nozzle size
    print(f"\n{'='*60}")
    print("DEMONSTRATING NOZZLE SIZE EFFECT")
    print(f"{'='*60}")
    
    # Increase nozzle size in first branch
    original_nozzle_diameter = branches[0].nozzle.diameter
    branches[0].nozzle.diameter = 0.012  # Increase from 8mm to 12mm
    
    print(f"Increasing {branches[0].name} nozzle from {original_nozzle_diameter*1000:.0f}mm to {branches[0].nozzle.diameter*1000:.0f}mm")
    
    # Recalculate
    new_branch_flows, new_solution_info = calculator.solve_flow_distribution(
        total_flow_rate, branches, temperature
    )
    
    # Show comparison
    print(f"\n{'Branch':<15} {'Original (L/s)':<15} {'New (L/s)':<12} {'Change (%)'}")
    print("-" * 55)
    
    for i, branch in enumerate(branches):
        name = branch.name if branch.name else f"Branch {i+1}"
        original_flow = branch_flows[i] * 1000
        new_flow = new_branch_flows[i] * 1000
        change_percent = ((new_flow - original_flow) / original_flow) * 100 if original_flow > 0 else 0
        
        print(f"{name:<15} {original_flow:<15.3f} {new_flow:<12.3f} {change_percent:>8.1f}%")


if __name__ == "__main__":
    main()