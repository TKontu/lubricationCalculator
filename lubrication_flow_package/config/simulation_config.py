"""
Simulation configuration for network flow analysis
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters"""
    
    # Flow parameters
    total_flow_rate: float  # m³/s
    temperature: float  # °C
    inlet_pressure: float  # Pa
    outlet_pressure: Optional[float] = None  # Pa, if None uses atmospheric
    
    # Fluid properties
    oil_density: float = 900.0  # kg/m³
    oil_type: str = "SAE30"
    
    # Solver settings
    max_iterations: int = 100
    tolerance: float = 1e-6
    relaxation_factor: float = 0.8
    
    # Output settings
    output_units: str = "metric"  # "metric" or "imperial"
    detailed_output: bool = True
    save_results: bool = False
    results_file: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'flow_parameters': {
                'total_flow_rate': self.total_flow_rate,
                'temperature': self.temperature,
                'inlet_pressure': self.inlet_pressure,
                'outlet_pressure': self.outlet_pressure
            },
            'fluid_properties': {
                'oil_density': self.oil_density,
                'oil_type': self.oil_type
            },
            'solver_settings': {
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance,
                'relaxation_factor': self.relaxation_factor
            },
            'output_settings': {
                'output_units': self.output_units,
                'detailed_output': self.detailed_output,
                'save_results': self.save_results,
                'results_file': self.results_file
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """Create from dictionary (JSON deserialization)"""
        flow_params = data.get('flow_parameters', {})
        fluid_props = data.get('fluid_properties', {})
        solver_settings = data.get('solver_settings', {})
        output_settings = data.get('output_settings', {})
        
        return cls(
            total_flow_rate=flow_params.get('total_flow_rate', 0.015),
            temperature=flow_params.get('temperature', 40.0),
            inlet_pressure=flow_params.get('inlet_pressure', 200000.0),
            outlet_pressure=flow_params.get('outlet_pressure'),
            oil_density=fluid_props.get('oil_density', 900.0),
            oil_type=fluid_props.get('oil_type', 'SAE30'),
            max_iterations=solver_settings.get('max_iterations', 100),
            tolerance=solver_settings.get('tolerance', 1e-6),
            relaxation_factor=solver_settings.get('relaxation_factor', 0.8),
            output_units=output_settings.get('output_units', 'metric'),
            detailed_output=output_settings.get('detailed_output', True),
            save_results=output_settings.get('save_results', False),
            results_file=output_settings.get('results_file')
        )