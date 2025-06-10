"""
Pump characteristic curves for hydraulic system modeling
"""

import numpy as np
from typing import Union, List, Tuple, Optional
from scipy.interpolate import interp1d
import warnings


class PumpCharacteristic:
    """
    Represents a pump characteristic curve P(Q) that can be defined using:
    - Polynomial coefficients
    - Lookup table (Q, P) points
    - Mathematical functions
    """
    
    def __init__(self, 
                 curve_type: str = "polynomial",
                 coefficients: Optional[List[float]] = None,
                 flow_points: Optional[List[float]] = None,
                 pressure_points: Optional[List[float]] = None,
                 max_flow: Optional[float] = None,
                 max_pressure: Optional[float] = None):
        """
        Initialize pump characteristic curve
        
        Args:
            curve_type: Type of curve definition ("polynomial", "table", "linear")
            coefficients: Polynomial coefficients [a0, a1, a2, ...] for P = a0 + a1*Q + a2*Q^2 + ...
            flow_points: Flow rate points for lookup table (m³/s)
            pressure_points: Pressure points for lookup table (Pa)
            max_flow: Maximum flow rate (m³/s) - used for bounds checking
            max_pressure: Maximum pressure (Pa) - used for bounds checking
        """
        self.curve_type = curve_type.lower()
        self.coefficients = coefficients or []
        self.flow_points = np.array(flow_points) if flow_points else None
        self.pressure_points = np.array(pressure_points) if pressure_points else None
        self.max_flow = max_flow
        self.max_pressure = max_pressure
        
        # Validate inputs
        self._validate_inputs()
        
        # Create interpolation function for table-based curves
        if self.curve_type == "table" and self.flow_points is not None:
            self._create_interpolator()
    
    def _validate_inputs(self):
        """Validate input parameters"""
        if self.curve_type == "polynomial":
            if not self.coefficients:
                raise ValueError("Polynomial coefficients must be provided for polynomial curve type")
        
        elif self.curve_type == "table":
            if self.flow_points is None or self.pressure_points is None:
                raise ValueError("Flow and pressure points must be provided for table curve type")
            if len(self.flow_points) != len(self.pressure_points):
                raise ValueError("Flow and pressure points must have the same length")
            if len(self.flow_points) < 2:
                raise ValueError("At least 2 points required for table interpolation")
        
        elif self.curve_type == "linear":
            if len(self.coefficients) != 2:
                raise ValueError("Linear curve requires exactly 2 coefficients [intercept, slope]")
        
        else:
            raise ValueError(f"Unsupported curve type: {self.curve_type}")
    
    def _create_interpolator(self):
        """Create interpolation function for table-based curves"""
        # Sort by flow rate to ensure monotonic interpolation
        sorted_indices = np.argsort(self.flow_points)
        sorted_flow = self.flow_points[sorted_indices]
        sorted_pressure = self.pressure_points[sorted_indices]
        
        # Create interpolator with extrapolation bounds
        self.interpolator = interp1d(
            sorted_flow, sorted_pressure,
            kind='linear',
            bounds_error=False,
            fill_value=(sorted_pressure[0], sorted_pressure[-1])
        )
        
        # Store bounds for validation
        self.min_flow = sorted_flow[0]
        self.max_flow_table = sorted_flow[-1]
    
    def get_pressure(self, flow_rate: float) -> float:
        """
        Get pressure for a given flow rate using the pump characteristic
        
        Args:
            flow_rate: Flow rate (m³/s)
            
        Returns:
            Pressure (Pa)
        """
        if flow_rate < 0:
            warnings.warn("Negative flow rate provided to pump characteristic")
            return 0.0
        
        if self.curve_type == "polynomial":
            pressure = sum(coeff * (flow_rate ** i) for i, coeff in enumerate(self.coefficients))
        
        elif self.curve_type == "table":
            pressure = float(self.interpolator(flow_rate))
        
        elif self.curve_type == "linear":
            pressure = self.coefficients[0] + self.coefficients[1] * flow_rate
        
        else:
            raise ValueError(f"Unsupported curve type: {self.curve_type}")
        
        # Apply bounds checking
        if self.max_pressure is not None:
            pressure = min(pressure, self.max_pressure)
        
        # Ensure non-negative pressure
        pressure = max(0.0, pressure)
        
        return pressure
    
    def get_flow_for_pressure(self, target_pressure: float, 
                             flow_range: Tuple[float, float] = (0.0, 1.0),
                             tolerance: float = 1e-6) -> Optional[float]:
        """
        Find flow rate that produces the target pressure (inverse lookup)
        
        Args:
            target_pressure: Target pressure (Pa)
            flow_range: Search range for flow rate (min_flow, max_flow)
            tolerance: Convergence tolerance for pressure matching
            
        Returns:
            Flow rate (m³/s) or None if no solution found
        """
        if target_pressure < 0:
            return None
        
        # Use binary search for inverse lookup
        min_flow, max_flow = flow_range
        
        # Check bounds
        p_min = self.get_pressure(min_flow)
        p_max = self.get_pressure(max_flow)
        
        if target_pressure > max(p_min, p_max) or target_pressure < min(p_min, p_max):
            return None
        
        # Binary search
        for _ in range(100):  # Maximum iterations
            mid_flow = (min_flow + max_flow) / 2
            mid_pressure = self.get_pressure(mid_flow)
            
            if abs(mid_pressure - target_pressure) < tolerance:
                return mid_flow
            
            if mid_pressure > target_pressure:
                if p_min < p_max:  # Normal decreasing curve
                    min_flow = mid_flow
                else:  # Increasing curve
                    max_flow = mid_flow
            else:
                if p_min < p_max:  # Normal decreasing curve
                    max_flow = mid_flow
                else:  # Increasing curve
                    min_flow = mid_flow
        
        return None
    
    def find_operating_point(self, system_resistance: float, 
                           flow_range: Tuple[float, float] = (0.0, 1.0),
                           tolerance: float = 1e-6) -> Tuple[float, float]:
        """
        Find the operating point where pump curve intersects system curve
        System curve: P = system_resistance * Q^2 (typical for hydraulic systems)
        
        Args:
            system_resistance: System resistance coefficient (Pa·s²/m⁶)
            flow_range: Search range for flow rate (min_flow, max_flow)
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of (operating_flow, operating_pressure)
        """
        min_flow, max_flow = flow_range
        
        # Binary search for intersection point
        for _ in range(100):  # Maximum iterations
            mid_flow = (min_flow + max_flow) / 2
            
            pump_pressure = self.get_pressure(mid_flow)
            system_pressure = system_resistance * (mid_flow ** 2)
            
            pressure_diff = pump_pressure - system_pressure
            
            if abs(pressure_diff) < tolerance:
                return mid_flow, pump_pressure
            
            if pressure_diff > 0:  # Pump pressure higher than system requirement
                min_flow = mid_flow
            else:  # System pressure higher than pump can provide
                max_flow = mid_flow
            
            # Check for convergence failure
            if max_flow - min_flow < 1e-12:
                break
        
        # Return best estimate
        final_flow = (min_flow + max_flow) / 2
        final_pressure = self.get_pressure(final_flow)
        return final_flow, final_pressure
    
    def get_curve_points(self, num_points: int = 50, 
                        flow_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate points along the pump curve for plotting
        
        Args:
            num_points: Number of points to generate
            flow_range: Flow range (min, max). If None, uses reasonable defaults
            
        Returns:
            Tuple of (flow_array, pressure_array)
        """
        if flow_range is None:
            if self.curve_type == "table":
                flow_range = (self.min_flow, self.max_flow_table)
            else:
                # Use reasonable defaults
                max_flow_est = self.max_flow if self.max_flow else 0.1  # 0.1 m³/s default
                flow_range = (0.0, max_flow_est)
        
        flow_array = np.linspace(flow_range[0], flow_range[1], num_points)
        pressure_array = np.array([self.get_pressure(q) for q in flow_array])
        
        return flow_array, pressure_array
    
    @classmethod
    def create_typical_centrifugal_pump(cls, max_pressure: float, max_flow: float,
                                      efficiency_point: Tuple[float, float] = (0.7, 0.8)) -> 'PumpCharacteristic':
        """
        Create a typical centrifugal pump characteristic curve
        
        Args:
            max_pressure: Shutoff pressure (Pa) at zero flow
            max_flow: Maximum flow rate (m³/s) at zero pressure
            efficiency_point: (flow_fraction, pressure_fraction) for best efficiency point
            
        Returns:
            PumpCharacteristic instance
        """
        # Create a quadratic curve that passes through (0, max_pressure) and (max_flow, 0)
        # with a specified efficiency point
        
        q_eff = efficiency_point[0] * max_flow
        p_eff = efficiency_point[1] * max_pressure
        
        # Solve for quadratic coefficients: P = a + b*Q + c*Q^2
        # Constraints: P(0) = max_pressure, P(max_flow) = 0, P(q_eff) = p_eff
        
        # From P(0) = max_pressure: a = max_pressure
        a = max_pressure
        
        # From P(max_flow) = 0: a + b*max_flow + c*max_flow^2 = 0
        # From P(q_eff) = p_eff: a + b*q_eff + c*q_eff^2 = p_eff
        
        # Solve the 2x2 system for b and c
        A = np.array([[max_flow, max_flow**2],
                     [q_eff, q_eff**2]])
        B = np.array([-a, p_eff - a])
        
        b, c = np.linalg.solve(A, B)
        
        return cls(
            curve_type="polynomial",
            coefficients=[a, b, c],
            max_flow=max_flow,
            max_pressure=max_pressure
        )
    
    @classmethod
    def create_from_manufacturer_data(cls, flow_points: List[float], 
                                    pressure_points: List[float]) -> 'PumpCharacteristic':
        """
        Create pump characteristic from manufacturer's performance data
        
        Args:
            flow_points: Flow rates from manufacturer data (m³/s)
            pressure_points: Pressures from manufacturer data (Pa)
            
        Returns:
            PumpCharacteristic instance
        """
        return cls(
            curve_type="table",
            flow_points=flow_points,
            pressure_points=pressure_points,
            max_flow=max(flow_points),
            max_pressure=max(pressure_points)
        )