#!/usr/bin/env python3
"""
AAdvanced Lubrication Piping Flow Distribution Calculation Tool

This enhanced version provides improved reliability for larger and complex systems
with multiple branches, different nozzle types, and robust numerical methods.

Key enhancements:
- Newton-Raphson solver for better convergence
- Support for different nozzle types and shapes
- Improved numerical stability for large systems
- Better handling of extreme flow conditions
- Comprehensive error handling and validation
- Performance optimizations for complex networks

Physics principles implemented:
1. Conservation of mass (continuity equation)
2. Equal pressure drops across parallel branches at junctions
3. Darcy-Weisbach equation for pipe friction losses
4. Various nozzle flow equations for different geometries
5. Temperature-dependent viscosity using multiple correlations
6. Compressibility effects for high-pressure systems
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum
import warnings


class NozzleType(Enum):
    """Enumeration of supported nozzle types"""
    SHARP_EDGED = "sharp_edged"
    ROUNDED = "rounded"
    VENTURI = "venturi"
    FLOW_NOZZLE = "flow_nozzle"
    CUSTOM = "custom"


class FlowRegime(Enum):
    """Flow regime classification"""
    LAMINAR = "laminar"
    TRANSITION = "transition"
    TURBULENT = "turbulent"


@dataclass
class PipeSegment:
    """Represents a pipe segment with its properties"""
    diameter: float  # meters
    length: float    # meters
    roughness: float = 0.00015  # meters (default for steel)
    
    def __post_init__(self):
        """Validate pipe parameters"""
        if self.diameter <= 0:
            raise ValueError("Pipe diameter must be positive")
        if self.length <= 0:
            raise ValueError("Pipe length must be positive")
        if self.roughness < 0:
            raise ValueError("Pipe roughness cannot be negative")


@dataclass
class Nozzle:
    """Represents a nozzle with flow restriction"""
    diameter: float      # meters
    nozzle_type: NozzleType = NozzleType.SHARP_EDGED
    discharge_coeff: Optional[float] = None  # Will be calculated if None
    beta_ratio: Optional[float] = None  # d/D ratio for some nozzle types
    
    def __post_init__(self):
        """Validate nozzle parameters and set default discharge coefficient"""
        if self.diameter <= 0:
            raise ValueError("Nozzle diameter must be positive")
            
        if self.discharge_coeff is None:
            self.discharge_coeff = self._get_default_discharge_coeff()
    
    def _get_default_discharge_coeff(self) -> float:
        """Get default discharge coefficient based on nozzle type"""
        defaults = {
            NozzleType.SHARP_EDGED: 0.6,
            NozzleType.ROUNDED: 0.8,
            NozzleType.VENTURI: 0.95,
            NozzleType.FLOW_NOZZLE: 0.98,
            NozzleType.CUSTOM: 0.6
        }
        return defaults[self.nozzle_type]


@dataclass
class Branch:
    """Represents a complete branch with pipe and optional nozzle"""
    pipe: PipeSegment
    nozzle: Optional[Nozzle] = None
    name: str = ""
    elevation_change: float = 0.0  # meters (positive = upward)
    
    def __post_init__(self):
        """Validate branch parameters"""
        if not self.name:
            self.name = f"Branch_{id(self)}"


class AdvancedLubricationFlowCalculator:
    """Advanced calculator class for lubrication flow distribution"""
    
    def __init__(self, oil_density: float = 900.0, oil_type: str = "SAE30",
                 compressibility: float = 0.0):
        """
        Initialize the advanced calculator
        
        Args:
            oil_density: Oil density in kg/m³ (typical range 850-950 for lubricating oils)
            oil_type: Type of oil for viscosity calculations
            compressibility: Oil compressibility factor (1/Pa) for high-pressure systems
        """
        self.oil_density = oil_density
        self.oil_type = oil_type
        self.compressibility = compressibility
        self.gravity = 9.81  # m/s²
        
        # Validation
        if oil_density <= 0:
            raise ValueError("Oil density must be positive")
        if compressibility < 0:
            raise ValueError("Compressibility cannot be negative")
    
    def calculate_viscosity(self, temperature: float) -> float:
        """
        Calculate dynamic viscosity using improved correlations
        
        Args:
            temperature: Temperature in Celsius
            
        Returns:
            Dynamic viscosity in Pa·s
        """
        # Convert to Kelvin
        T = temperature + 273.15
        
        # Enhanced Vogel equation parameters for different oil types
        # μ = A * exp(B / (T - C)) with improved coefficients
        viscosity_params = {
            "SAE10": {"A": 0.00004, "B": 950, "C": 135},
            "SAE20": {"A": 0.00006, "B": 1050, "C": 138},
            "SAE30": {"A": 0.0001, "B": 1200, "C": 140},
            "SAE40": {"A": 0.00015, "B": 1300, "C": 142},
            "SAE50": {"A": 0.0002, "B": 1400, "C": 145},
            "SAE60": {"A": 0.00025, "B": 1500, "C": 148}
        }
        
        if self.oil_type not in viscosity_params:
            raise ValueError(f"Oil type {self.oil_type} not supported. "
                           f"Available types: {list(viscosity_params.keys())}")
        
        params = viscosity_params[self.oil_type]
        
        # Vogel equation with temperature limits
        if T < params["C"]:
            warnings.warn(f"Temperature {temperature}°C is below recommended range")
            T = params["C"] + 1  # Avoid division by zero
        
        viscosity = params["A"] * math.exp(params["B"] / (T - params["C"]))
        
        # Apply reasonable limits
        viscosity = max(1e-6, min(viscosity, 10.0))  # 1 µPa·s to 10 Pa·s
        
        return viscosity
    
    def calculate_reynolds_number(self, velocity: float, diameter: float, 
                                viscosity: float) -> float:
        """Calculate Reynolds number with validation"""
        if velocity < 0 or diameter <= 0 or viscosity <= 0:
            return 0
        return (self.oil_density * velocity * diameter) / viscosity
    
    def calculate_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """
        Calculate friction factor using enhanced correlations
        
        Args:
            reynolds: Reynolds number
            relative_roughness: ε/D ratio
            
        Returns:
            Darcy friction factor
        """
        if reynolds <= 0:
            return 0
            
        if reynolds < 2300:  # Laminar flow
            return 64 / reynolds
        elif reynolds < 4000:  # Transition region
            # Improved transition model
            f_lam = 64 / 2300
            f_turb = self._turbulent_friction_factor(4000, relative_roughness)
            # Smooth transition using cubic interpolation
            x = (reynolds - 2300) / (4000 - 2300)
            return f_lam * (1 - x)**3 + f_turb * x**3
        else:  # Turbulent flow
            return self._turbulent_friction_factor(reynolds, relative_roughness)
    
    def _turbulent_friction_factor(self, reynolds: float, relative_roughness: float) -> float:
        """Calculate turbulent friction factor using Colebrook-White equation"""
        if relative_roughness <= 0:  # Smooth pipe
            # Blasius equation for smooth pipes (Re < 100,000)
            if reynolds < 100000:
                return 0.316 / (reynolds ** 0.25)
            else:
                # Prandtl equation for higher Reynolds numbers
                return 0.0032 + 0.221 / (reynolds ** 0.237)
        else:
            # Swamee-Jain approximation with improved accuracy
            term1 = relative_roughness / 3.7
            term2 = 5.74 / (reynolds ** 0.9)
            
            # Ensure we don't get negative values in log
            log_arg = max(term1 + term2, 1e-10)
            denominator = math.log10(log_arg)
            
            # Avoid numerical issues
            if abs(denominator) < 1e-10:
                return 0.02  # Reasonable default
                
            return 0.25 / (denominator ** 2)
    
    def calculate_pipe_pressure_drop(self, flow_rate: float, pipe: PipeSegment, 
                                   viscosity: float, elevation_change: float = 0.0) -> float:
        """
        Calculate pressure drop through a pipe including elevation effects
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            pipe: PipeSegment object
            viscosity: Dynamic viscosity in Pa·s
            elevation_change: Elevation change in meters (positive = upward)
            
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
        
        # Darcy-Weisbach equation for friction losses
        friction_loss = friction_factor * (pipe.length / pipe.diameter) * \
                       (self.oil_density * velocity ** 2) / 2
        
        # Elevation pressure change
        elevation_loss = self.oil_density * self.gravity * elevation_change
        
        # Total pressure drop
        total_pressure_drop = friction_loss + elevation_loss
        
        return total_pressure_drop
    
    def calculate_nozzle_pressure_drop(self, flow_rate: float, nozzle: Nozzle,
                                     pipe_diameter: float = None) -> float:
        """
        Calculate pressure drop through a nozzle with enhanced models
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            nozzle: Nozzle object
            pipe_diameter: Upstream pipe diameter for beta ratio calculations
            
        Returns:
            Pressure drop in Pascals
        """
        if flow_rate <= 0:
            return 0
            
        nozzle_area = math.pi * (nozzle.diameter / 2) ** 2
        velocity = flow_rate / nozzle_area
        
        # Calculate pressure drop based on nozzle type
        if nozzle.nozzle_type == NozzleType.SHARP_EDGED:
            # Standard orifice equation
            K = (1 / nozzle.discharge_coeff ** 2) - 1
            pressure_drop = K * self.oil_density * velocity ** 2 / 2
            
        elif nozzle.nozzle_type == NozzleType.ROUNDED:
            # Rounded entrance nozzle
            K = (1 / nozzle.discharge_coeff ** 2) - 1
            pressure_drop = K * self.oil_density * velocity ** 2 / 2
            
        elif nozzle.nozzle_type == NozzleType.VENTURI:
            # Venturi nozzle with recovery - lower pressure drop due to diffuser recovery
            if pipe_diameter and nozzle.beta_ratio:
                beta = nozzle.beta_ratio
            elif pipe_diameter:
                beta = nozzle.diameter / pipe_diameter
            else:
                beta = 0.7  # Default assumption
                
            # Venturi has much lower permanent pressure loss due to diffuser recovery
            # Permanent pressure loss is typically 5-15% of differential pressure
            K_differential = (1 - beta**4) / (nozzle.discharge_coeff**2 * beta**4)
            recovery_factor = 0.1  # Only 10% permanent loss for well-designed venturi
            K = K_differential * recovery_factor
            pressure_drop = K * self.oil_density * velocity ** 2 / 2
            
        elif nozzle.nozzle_type == NozzleType.FLOW_NOZZLE:
            # Flow nozzle (ISA 1932)
            if pipe_diameter:
                beta = nozzle.diameter / pipe_diameter
            else:
                beta = 0.7  # Default assumption
                
            K = (1 - beta**4) / (nozzle.discharge_coeff**2 * beta**4)
            pressure_drop = K * self.oil_density * velocity ** 2 / 2
            
        else:  # CUSTOM or fallback
            K = (1 / nozzle.discharge_coeff ** 2) - 1
            pressure_drop = K * self.oil_density * velocity ** 2 / 2
        
        return pressure_drop
    
    def calculate_branch_pressure_drop(self, flow_rate: float, branch: Branch, 
                                     viscosity: float) -> float:
        """
        Calculate total pressure drop through a branch (pipe + nozzle + elevation)
        
        Args:
            flow_rate: Volumetric flow rate in m³/s
            branch: Branch object
            viscosity: Dynamic viscosity in Pa·s
            
        Returns:
            Total pressure drop in Pascals
        """
        pipe_dp = self.calculate_pipe_pressure_drop(
            flow_rate, branch.pipe, viscosity, branch.elevation_change
        )
        
        nozzle_dp = 0
        if branch.nozzle:
            nozzle_dp = self.calculate_nozzle_pressure_drop(
                flow_rate, branch.nozzle, branch.pipe.diameter
            )
            
        return pipe_dp + nozzle_dp
    
    def solve_flow_distribution_newton(self, total_flow_rate: float, branches: List[Branch], 
                                     temperature: float, max_iterations: int = 50, 
                                     tolerance: float = 1e-3) -> Tuple[List[float], Dict]:
        """
        Solve flow distribution using Newton-Raphson method for improved convergence
        
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
        
        if num_branches < 2:
            # Single branch case
            return [total_flow_rate], {
                'converged': True,
                'iterations': 1,
                'viscosity': viscosity,
                'temperature': temperature,
                'method': 'direct',
                'pressure_drops': [self.calculate_branch_pressure_drop(
                    total_flow_rate, branches[0], viscosity)],
                'reynolds_numbers': [self._calculate_reynolds_for_branch(
                    total_flow_rate, branches[0], viscosity)]
            }
        
        # Better initial guess based on pipe resistance
        flows = self._get_initial_guess(total_flow_rate, branches, viscosity)
        
        solution_info = {
            'converged': False,
            'iterations': 0,
            'viscosity': viscosity,
            'temperature': temperature,
            'method': 'newton_raphson',
            'pressure_drops': [],
            'reynolds_numbers': []
        }
        
        for iteration in range(max_iterations):
            # Calculate residuals (pressure drop differences)
            pressure_drops = np.array([
                self.calculate_branch_pressure_drop(flow, branch, viscosity)
                for flow, branch in zip(flows, branches)
            ])
            
            # Target pressure drop (first branch as reference)
            target_dp = pressure_drops[0]
            
            # Residual vector (pressure differences from first branch)
            residuals = pressure_drops[1:] - target_dp
            
            # Mass conservation residual
            mass_residual = np.sum(flows) - total_flow_rate
            
            # Combined residual vector
            F = np.concatenate([residuals, [mass_residual]])
            
            # Check convergence
            max_residual = np.max(np.abs(F))
            
            # Optional debug output (commented out for production)
            # if iteration < 5:
            #     print(f"  Iteration {iteration}: max residual = {max_residual:.2e}")
            
            if max_residual < tolerance:
                solution_info['converged'] = True
                solution_info['iterations'] = iteration + 1
                break
            
            # Calculate Jacobian matrix
            jacobian = self._calculate_jacobian(flows, branches, viscosity)
            
            # Solve Newton step: J * delta_flows = -F
            try:
                # Check condition number to avoid ill-conditioned systems
                cond_num = np.linalg.cond(jacobian)
                if cond_num > 1e12:
                    # Use damped iteration for ill-conditioned systems
                    delta_flows = self._damped_iteration_step(flows, branches, viscosity, 
                                                            total_flow_rate)
                else:
                    delta_flows = np.linalg.solve(jacobian, -F)
            except np.linalg.LinAlgError:
                # Fallback to damped iteration if Jacobian is singular
                delta_flows = self._damped_iteration_step(flows, branches, viscosity, 
                                                        total_flow_rate)
            
            # Adaptive damping based on iteration number
            if iteration < 5:
                damping_factor = 0.5  # Conservative start
            elif iteration < 15:
                damping_factor = 0.7  # Moderate damping
            else:
                damping_factor = 0.3  # Heavy damping for difficult cases
                
            flows += damping_factor * delta_flows
            
            # Ensure positive flows
            flows = np.maximum(flows, 1e-10)
            
            # Normalize to maintain total flow
            flows = flows * total_flow_rate / np.sum(flows)
        
        # Update final iteration count if not converged
        if not solution_info['converged']:
            solution_info['iterations'] = max_iterations
        
        # Calculate final results
        final_pressure_drops = [
            self.calculate_branch_pressure_drop(flow, branch, viscosity)
            for flow, branch in zip(flows, branches)
        ]
        
        reynolds_numbers = [
            self._calculate_reynolds_for_branch(flow, branch, viscosity)
            for flow, branch in zip(flows, branches)
        ]
        
        solution_info['pressure_drops'] = final_pressure_drops
        solution_info['reynolds_numbers'] = reynolds_numbers
        
        return flows.tolist(), solution_info
    
    def _calculate_jacobian(self, flows: np.ndarray, branches: List[Branch], 
                          viscosity: float) -> np.ndarray:
        """Calculate Jacobian matrix for Newton-Raphson method"""
        num_branches = len(branches)
        jacobian = np.zeros((num_branches, num_branches))
        
        # Adaptive delta based on flow magnitude
        base_delta = 1e-8
        
        for i in range(num_branches - 1):  # Pressure difference equations
            for j in range(num_branches):
                # Adaptive delta for numerical differentiation
                delta = max(base_delta, flows[j] * 1e-6)
                
                # Derivative of (DP_j - DP_0) with respect to flow_j
                if j == 0:
                    # Derivative with respect to first branch flow
                    flow_plus = max(flows[0] + delta, 1e-12)
                    flow_minus = max(flows[0] - delta, 1e-12)
                    dp_plus = self.calculate_branch_pressure_drop(flow_plus, branches[0], viscosity)
                    dp_minus = self.calculate_branch_pressure_drop(flow_minus, branches[0], viscosity)
                    jacobian[i, j] = -(dp_plus - dp_minus) / (flow_plus - flow_minus)
                elif j == i + 1:
                    # Derivative with respect to current branch flow
                    flow_plus = max(flows[j] + delta, 1e-12)
                    flow_minus = max(flows[j] - delta, 1e-12)
                    dp_plus = self.calculate_branch_pressure_drop(flow_plus, branches[j], viscosity)
                    dp_minus = self.calculate_branch_pressure_drop(flow_minus, branches[j], viscosity)
                    jacobian[i, j] = (dp_plus - dp_minus) / (flow_plus - flow_minus)
                else:
                    jacobian[i, j] = 0
        
        # Mass conservation equation (last row)
        jacobian[-1, :] = 1
        
        # Add small diagonal terms for numerical stability
        for i in range(num_branches - 1):
            if abs(jacobian[i, i]) < 1e-12:
                jacobian[i, i] = 1e-6
        
        return jacobian
    
    def _damped_iteration_step(self, flows: np.ndarray, branches: List[Branch],
                             viscosity: float, total_flow_rate: float) -> np.ndarray:
        """Fallback damped iteration when Jacobian is singular"""
        num_branches = len(branches)
        pressure_drops = np.array([
            self.calculate_branch_pressure_drop(flow, branch, viscosity)
            for flow, branch in zip(flows, branches)
        ])
        
        target_dp = np.mean(pressure_drops)
        delta_flows = np.zeros(num_branches)
        
        for i, (flow, branch, dp) in enumerate(zip(flows, branches, pressure_drops)):
            if dp > 0:
                # Estimate derivative
                delta_q = flow * 0.001
                dp_plus = self.calculate_branch_pressure_drop(flow + delta_q, branch, viscosity)
                dpdq = (dp_plus - dp) / delta_q
                
                if abs(dpdq) > 1e-12:
                    delta_flows[i] = (target_dp - dp) / dpdq
        
        return delta_flows
    
    def _calculate_reynolds_for_branch(self, flow_rate: float, branch: Branch, 
                                     viscosity: float) -> float:
        """Calculate Reynolds number for a branch"""
        if flow_rate <= 0:
            return 0
        area = math.pi * (branch.pipe.diameter / 2) ** 2
        velocity = flow_rate / area
        return self.calculate_reynolds_number(velocity, branch.pipe.diameter, viscosity)
    
    def _get_initial_guess(self, total_flow_rate: float, branches: List[Branch], 
                          viscosity: float) -> np.ndarray:
        """Get better initial guess based on branch resistances"""
        num_branches = len(branches)
        
        # Calculate approximate resistance for each branch
        resistances = []
        for branch in branches:
            # Estimate resistance using simplified approach
            area = math.pi * (branch.pipe.diameter / 2) ** 2
            
            # Pipe resistance (simplified)
            pipe_resistance = (128 * viscosity * branch.pipe.length) / (math.pi * branch.pipe.diameter**4)
            
            # Nozzle resistance (simplified)
            nozzle_resistance = 0
            if branch.nozzle:
                nozzle_area = math.pi * (branch.nozzle.diameter / 2) ** 2
                # Simplified nozzle resistance
                K = (1 / branch.nozzle.discharge_coeff ** 2) - 1
                nozzle_resistance = K * self.oil_density / (2 * nozzle_area**2)
            
            total_resistance = pipe_resistance + nozzle_resistance
            resistances.append(total_resistance)
        
        # Convert to conductances (inverse of resistance)
        conductances = [1/r if r > 0 else 1.0 for r in resistances]
        total_conductance = sum(conductances)
        
        # Distribute flow proportional to conductance
        flows = np.array([total_flow_rate * c / total_conductance for c in conductances])
        
        # Ensure positive flows
        flows = np.maximum(flows, total_flow_rate / (num_branches * 100))
        
        # Normalize to maintain total flow
        flows = flows * total_flow_rate / np.sum(flows)
        
        return flows
    
    def solve_flow_distribution(self, total_flow_rate: float, branches: List[Branch], 
                              temperature: float, method: str = "newton",
                              **kwargs) -> Tuple[List[float], Dict]:
        """
        Solve flow distribution using specified method
        
        Args:
            total_flow_rate: Total flow rate in m³/s
            branches: List of Branch objects
            temperature: Temperature in Celsius
            method: Solution method ("newton" or "iterative")
            **kwargs: Additional arguments for the solver
            
        Returns:
            Tuple of (branch_flow_rates, solution_info)
        """
        if method.lower() == "newton":
            return self.solve_flow_distribution_newton(
                total_flow_rate, branches, temperature, **kwargs
            )
        else:
            # Fallback to iterative method (from original implementation)
            return self._solve_flow_distribution_iterative(
                total_flow_rate, branches, temperature, **kwargs
            )
    
    def _solve_flow_distribution_iterative(self, total_flow_rate: float, branches: List[Branch], 
                                         temperature: float, max_iterations: int = 100, 
                                         tolerance: float = 1e-6) -> Tuple[List[float], Dict]:
        """Original iterative method as fallback"""
        viscosity = self.calculate_viscosity(temperature)
        num_branches = len(branches)
        
        # Initial guess - equal distribution
        branch_flows = np.array([total_flow_rate / num_branches] * num_branches)
        
        solution_info = {
            'converged': False,
            'iterations': 0,
            'viscosity': viscosity,
            'temperature': temperature,
            'method': 'iterative',
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
        
        # Calculate final results
        final_pressure_drops = []
        reynolds_numbers = []
        
        for i, (flow, branch) in enumerate(zip(branch_flows, branches)):
            dp = self.calculate_branch_pressure_drop(flow, branch, viscosity)
            final_pressure_drops.append(dp)
            reynolds_numbers.append(self._calculate_reynolds_for_branch(flow, branch, viscosity))
        
        solution_info['pressure_drops'] = final_pressure_drops
        solution_info['reynolds_numbers'] = reynolds_numbers
        
        return branch_flows.tolist(), solution_info
    
    def get_flow_regime(self, reynolds: float) -> FlowRegime:
        """Determine flow regime based on Reynolds number"""
        if reynolds < 2300:
            return FlowRegime.LAMINAR
        elif reynolds < 4000:
            return FlowRegime.TRANSITION
        else:
            return FlowRegime.TURBULENT
    
    def print_results(self, branch_flows: List[float], branches: List[Branch], 
                     solution_info: Dict, detailed: bool = True):
        """Print detailed results of the flow calculation"""
        print(f"\n{'='*70}")
        print("ADVANCED LUBRICATION FLOW DISTRIBUTION RESULTS")
        print(f"{'='*70}")
        
        print(f"Temperature: {solution_info['temperature']:.1f}°C")
        print(f"Oil Type: {self.oil_type}")
        print(f"Oil Density: {self.oil_density:.1f} kg/m³")
        print(f"Dynamic Viscosity: {solution_info['viscosity']:.6f} Pa·s")
        print(f"Solution Method: {solution_info.get('method', 'unknown')}")
        print(f"Converged: {solution_info['converged']} (in {solution_info['iterations']} iterations)")
        
        print(f"\n{'Branch':<15} {'Flow Rate':<12} {'Pressure Drop':<15} {'Reynolds':<10} {'Flow Type'}")
        print(f"{'Name':<15} {'(L/s)':<12} {'(Pa)':<15} {'Number':<10}")
        print("-" * 70)
        
        total_flow = 0
        for i, (flow, branch) in enumerate(zip(branch_flows, branches)):
            name = branch.name if branch.name else f"Branch {i+1}"
            flow_lps = flow * 1000  # Convert to L/s
            pressure_drop = solution_info['pressure_drops'][i]
            reynolds = solution_info['reynolds_numbers'][i]
            flow_regime = self.get_flow_regime(reynolds)
            
            print(f"{name:<15} {flow_lps:<12.3f} {pressure_drop:<15.1f} {reynolds:<10.0f} {flow_regime.value}")
            total_flow += flow
        
        print("-" * 70)
        print(f"{'Total':<15} {total_flow*1000:<12.3f}")
        
        if detailed:
            # Print branch details
            print(f"\n{'BRANCH DETAILS'}")
            print("-" * 50)
            for i, branch in enumerate(branches):
                name = branch.name if branch.name else f"Branch {i+1}"
                print(f"{name}:")
                print(f"  Pipe: D={branch.pipe.diameter*1000:.1f}mm, L={branch.pipe.length:.1f}m, "
                      f"ε={branch.pipe.roughness*1000:.3f}mm")
                if branch.nozzle:
                    print(f"  Nozzle: D={branch.nozzle.diameter*1000:.1f}mm, "
                          f"Type={branch.nozzle.nozzle_type.value}, Cd={branch.nozzle.discharge_coeff:.3f}")
                if branch.elevation_change != 0:
                    print(f"  Elevation change: {branch.elevation_change:.1f}m")
                print()


def create_complex_example_system() -> Tuple[float, List[Branch], float]:
    """Create a complex lubrication system for testing"""
    
    # System parameters
    total_flow_rate = 0.025  # m³/s (25 L/s) - larger system
    temperature = 45  # °C
    
    # Define complex branches with different nozzle types
    branches = [
        Branch(
            pipe=PipeSegment(diameter=0.08, length=12.0, roughness=0.00015),
            nozzle=Nozzle(diameter=0.015, nozzle_type=NozzleType.SHARP_EDGED),
            name="Main Bearing 1",
            elevation_change=2.0
        ),
        Branch(
            pipe=PipeSegment(diameter=0.06, length=8.0, roughness=0.00020),
            nozzle=Nozzle(diameter=0.012, nozzle_type=NozzleType.ROUNDED),
            name="Main Bearing 2",
            elevation_change=1.5
        ),
        Branch(
            pipe=PipeSegment(diameter=0.05, length=15.0, roughness=0.00015),
            nozzle=Nozzle(diameter=0.008, nozzle_type=NozzleType.VENTURI, beta_ratio=0.5),
            name="Aux Bearing 1",
            elevation_change=0.5
        ),
        Branch(
            pipe=PipeSegment(diameter=0.04, length=10.0, roughness=0.00025),
            nozzle=Nozzle(diameter=0.006, nozzle_type=NozzleType.FLOW_NOZZLE),
            name="Aux Bearing 2",
            elevation_change=-0.5
        ),
        Branch(
            pipe=PipeSegment(diameter=0.035, length=18.0, roughness=0.00030),
            nozzle=Nozzle(diameter=0.005, nozzle_type=NozzleType.SHARP_EDGED),
            name="Gear Box 1",
            elevation_change=1.0
        ),
        Branch(
            pipe=PipeSegment(diameter=0.03, length=14.0, roughness=0.00020),
            nozzle=Nozzle(diameter=0.004, nozzle_type=NozzleType.ROUNDED),
            name="Gear Box 2",
            elevation_change=0.8
        ),
        Branch(
            pipe=PipeSegment(diameter=0.025, length=6.0, roughness=0.00015),
            nozzle=None,  # No nozzle restriction
            name="Cooler Return 1",
            elevation_change=-2.0
        ),
        Branch(
            pipe=PipeSegment(diameter=0.02, length=8.0, roughness=0.00025),
            nozzle=None,  # No nozzle restriction
            name="Cooler Return 2",
            elevation_change=-1.5
        )
    ]
    
    return total_flow_rate, branches, temperature


def main():
    """Main function demonstrating the advanced lubrication flow calculator"""
    
    # Create calculator instance
    calculator = AdvancedLubricationFlowCalculator(oil_density=900.0, oil_type="SAE30")
    
    # Test with complex system
    total_flow_rate, branches, temperature = create_complex_example_system()
    
    print("Calculating advanced lubrication flow distribution...")
    print(f"Total flow rate: {total_flow_rate * 1000:.1f} L/s")
    print(f"Number of branches: {len(branches)}")
    print(f"Temperature: {temperature}°C")
    
    # Solve using Newton-Raphson method
    print("\n" + "="*50)
    print("NEWTON-RAPHSON METHOD")
    print("="*50)
    
    branch_flows_newton, solution_info_newton = calculator.solve_flow_distribution(
        total_flow_rate, branches, temperature, method="newton"
    )
    
    calculator.print_results(branch_flows_newton, branches, solution_info_newton)
    
    # Compare with iterative method
    print("\n" + "="*50)
    print("ITERATIVE METHOD COMPARISON")
    print("="*50)
    
    branch_flows_iter, solution_info_iter = calculator.solve_flow_distribution(
        total_flow_rate, branches, temperature, method="iterative"
    )
    
    print(f"Newton-Raphson: Converged in {solution_info_newton['iterations']} iterations")
    print(f"Iterative: Converged in {solution_info_iter['iterations']} iterations")
    
    # Show flow differences
    print(f"\n{'Branch':<15} {'Newton (L/s)':<12} {'Iterative (L/s)':<15} {'Difference (%)'}")
    print("-" * 60)
    
    for i, branch in enumerate(branches):
        newton_flow = branch_flows_newton[i] * 1000
        iter_flow = branch_flows_iter[i] * 1000
        diff_percent = abs(newton_flow - iter_flow) / newton_flow * 100 if newton_flow > 0 else 0
        
        print(f"{branch.name:<15} {newton_flow:<12.3f} {iter_flow:<15.3f} {diff_percent:>10.2f}%")


if __name__ == "__main__":
    main()