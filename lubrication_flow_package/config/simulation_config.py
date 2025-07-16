"""
Simulation configuration for network flow analysis
"""

from dataclasses import dataclass, field
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
    viscosity_model: str = "vogel"

    # Solver settings
    max_iterations: int = 100
    tolerance: float = 1e-6
    min_resistance: float = 1e-12
    dq_absolute: float = 1e-8
    relaxation_factor: float = 0.5
    
    # For RobustNonLinearSolver
    convergence: Dict[str, Any] = field(default_factory=lambda: {
        "residual_tolerance": 1.0e-8,
        "relative_tolerance": 1.0e-6,
        "component_tolerance": 1.0e-4,
    })
    line_search: Dict[str, Any] = field(default_factory=lambda: {
        "method": "armijo",
        "c1": 1.0e-4,
        "alpha_min": 1.0e-10,
        "max_backtracks": 20,
    })
    jacobian: Dict[str, Any] = field(default_factory=lambda: {
        "update_method": "analytical",
        "finite_difference_step": 1.0e-8,
        "sparsity_detection": True,
    })

    # Output settings
    output_units: str = "metric"  # "metric" or "imperial"
    input_pressure_unit: str = "Pa"
    input_flow_rate_unit: str = "m3/s"
    output_pressure_unit: str = "Pa"
    output_flow_rate_unit: str = "m3/s"
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
                'oil_type': self.oil_type,
                'viscosity_model': self.viscosity_model
            },
            'solver_settings': {
                'max_iterations': self.max_iterations,
                'tolerance': self.tolerance,
                'min_resistance': self.min_resistance,
                'dq_absolute': self.dq_absolute,
                'relaxation_factor': self.relaxation_factor,
                'convergence': self.convergence,
                'line_search': self.line_search,
                'jacobian': self.jacobian
            },
            'output_settings': {
                'output_units': self.output_units,
                'input_pressure_unit': self.input_pressure_unit,
                'input_flow_rate_unit': self.input_flow_rate_unit,
                'output_pressure_unit': self.output_pressure_unit,
                'output_flow_rate_unit': self.output_flow_rate_unit,
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
        solver_settings_data = data.get('solver_settings', {})
        output_settings = data.get('output_settings', {})

        # Create a temporary default instance to get default dicts
        temp_defaults = cls(total_flow_rate=0, temperature=0, inlet_pressure=0)
        
        # Merge the nested dictionaries
        convergence = temp_defaults.convergence
        if solver_settings_data.get('convergence'):
            convergence.update(solver_settings_data['convergence'])
            
        line_search = temp_defaults.line_search
        if solver_settings_data.get('line_search'):
            line_search.update(solver_settings_data['line_search'])
            
        jacobian = temp_defaults.jacobian
        if solver_settings_data.get('jacobian'):
            jacobian.update(solver_settings_data['jacobian'])

        return cls(
            total_flow_rate=flow_params.get('total_flow_rate', 0.015),
            temperature=flow_params.get('temperature', 40.0),
            inlet_pressure=flow_params.get('inlet_pressure', 200000.0),
            outlet_pressure=flow_params.get('outlet_pressure'),
            
            oil_density=fluid_props.get('oil_density', 900.0),
            oil_type=fluid_props.get('oil_type', 'SAE30'),
            viscosity_model=fluid_props.get('viscosity_model', 'vogel'),
            
            max_iterations=solver_settings_data.get('max_iterations', temp_defaults.max_iterations),
            tolerance=solver_settings_data.get('tolerance', temp_defaults.tolerance),
            min_resistance=solver_settings_data.get('min_resistance', temp_defaults.min_resistance),
            dq_absolute=solver_settings_data.get('dq_absolute', temp_defaults.dq_absolute),
            relaxation_factor=solver_settings_data.get('relaxation_factor', temp_defaults.relaxation_factor),
            
            convergence=convergence,
            line_search=line_search,
            jacobian=jacobian,

            output_units=output_settings.get('output_units', 'metric'),
            input_pressure_unit=output_settings.get('input_pressure_unit', 'Pa'),
            input_flow_rate_unit=output_settings.get('input_flow_rate_unit', 'm3/s'),
            output_pressure_unit=output_settings.get('output_pressure_unit', 'Pa'),
            output_flow_rate_unit=output_settings.get('output_flow_rate_unit', 'm3/s'),
            detailed_output=output_settings.get('detailed_output', True),
            save_results=output_settings.get('save_results', False),
            results_file=output_settings.get('results_file')
        )
