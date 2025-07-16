"""
Base classes for all hydraulic network solvers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Callable

from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..utils.viscosity import calculate_viscosity


class SolverBase(ABC):
    """
    An abstract base class that defines the common interface for all hydraulic solvers.

    This class ensures that any solver, whether it's a linear nodal solver or a
    non-linear Newton-Raphson solver, can be used interchangeably by the
    calling application (e.g., the CLI or GUI).
    """

    def __init__(self, sim_config: SimulationConfig, progress_callback: Optional[Callable[[str], None]] = None):
        """
        Initializes the solver.

        Args:
            sim_config: The simulation configuration object, containing physical
                        parameters of the system (e.g., flow rate, fluid properties)
                        and solver settings.
            progress_callback: An optional callable to report progress updates.
        """
        self.sim_config = sim_config
        self.progress_callback = progress_callback
        
        # Centralized fluid property calculation
        viscosity = calculate_viscosity(
            temperature=self.sim_config.temperature,
            oil_type=self.sim_config.oil_type,
            viscosity_model=self.sim_config.viscosity_model
        )
        self.fluid_properties = {
            'density': self.sim_config.oil_density,
            'viscosity': viscosity
        }

    def _report_progress(self, message: str):
        """Reports progress if a callback is provided."""
        if self.progress_callback:
            self.progress_callback(message)

    @abstractmethod
    def solve(self, network: FlowNetwork) -> Dict:
        """
        Main entry point for solving the hydraulic network.

        Args:
            network: The FlowNetwork object to be solved.

        Returns:
            A dictionary containing the complete solution, including flows,
            pressures, and convergence information. The structure of this
            dictionary should be standardized across all solvers.
        """
        raise NotImplementedError("Subclasses must implement the solve method.")

    def print_results(self, network: FlowNetwork, solution: Dict, pressure_unit: str = 'kPa', flow_rate_unit: str = 'L/s'):
        """Print detailed results in a structured and clear format."""
        
        def convert_pressure(p_pa, unit):
            if unit.lower() == 'bar':
                return p_pa / 100000, 'bar'
            return p_pa / 1000, 'kPa'

        def convert_flow_rate(q_m3s, unit):
            if unit.lower() == 'l/min':
                return q_m3s * 60000, 'L/min'
            return q_m3s * 1000, 'L/s'

        p_unit_str = pressure_unit
        q_unit_str = flow_rate_unit
        
        connection_flows = solution.get('component_flows', {})
        solution_info = solution

        print(f"\n{'='*80}")
        print(f"NETWORK FLOW SIMULATION RESULTS")
        print(f"{'='*80}")
        
        # --- General Information ---
        print(f"  Network Name:      {network.name}")
        print(f"  Temperature:       {solution_info.get('temperature', 'N/A'):.1f}°C")
        print(f"  Oil Type:          {self.sim_config.oil_type}")
        print(f"  Oil Density:       {self.sim_config.oil_density:.1f} kg/m³")
        print(f"  Dynamic Viscosity: {solution_info.get('viscosity', 'N/A'):.6f} Pa·s")
        
        # --- Simulation Summary ---
        flow_rate_key = 'total_flow_rate' if 'total_flow_rate' in solution_info else 'actual_flow_rate'
        total_flow_rate = solution_info.get(flow_rate_key, 0.0)
        total_flow_rate_disp, q_unit_str_disp = convert_flow_rate(total_flow_rate, q_unit_str)
        print(f"\n  Total System Flow Rate: {total_flow_rate_disp:.2f} {q_unit_str_disp}")
        
        inlet_pressure_key = 'inlet_pressure' if 'inlet_pressure' in solution_info else 'required_inlet_pressure'
        inlet_pressure = solution_info.get(inlet_pressure_key, 0.0)
        inlet_pressure_disp, p_unit_str_disp = convert_pressure(inlet_pressure, p_unit_str)
        print(f"  Inlet Pressure:         {inlet_pressure_disp:.2f} {p_unit_str_disp}")
        
        converged = solution_info.get('converged', False)
        iterations = solution_info.get('iterations', 'N/A')
        print(f"  Solver Converged:       {'Yes' if converged else 'No'} (in {iterations} iterations)")
        
        # --- Outlet Flow Distribution ---
        print(f"\n{'='*80}")
        print("OUTLET FLOW DISTRIBUTION")
        print(f"{'='*80}")
        print(f"  {'Outlet Node':<25} {'Flow Rate (' + q_unit_str + ')':<20} {'Percentage of Total':<25}")
        print(f"  {'-'*25} {'-'*20} {'-'*25}")
        
        outlet_nodes = network.outlet_nodes
        total_outlet_flow = 0
        
        if outlet_nodes:
            for outlet_node in outlet_nodes:
                for conn in network.connections:
                    if conn.to_node.id == outlet_node.id:
                        flow = connection_flows.get(conn.component.id, 0.0)
                        total_outlet_flow += flow
                        flow_disp, _ = convert_flow_rate(flow, q_unit_str)
                        percentage = (flow / total_flow_rate * 100) if total_flow_rate > 0 else 0
                        print(f"  {outlet_node.name:<25} {flow_disp:<20.3f} {percentage:>24.1f}%")
        
        total_outlet_flow_disp, _ = convert_flow_rate(total_outlet_flow, q_unit_str)
        print(f"  {'-'*25} {'-'*20} {'-'*25}")
        print(f"  {'Total Outlet Flow':<25} {total_outlet_flow_disp:<20.3f}")

        # --- Path Analysis ---
        print(f"\n{'='*80}")
        print("PATH ANALYSIS")
        print(f"{'='*80}")
        print(f"  {'Path to Outlet':<25} {'Length (m)':<15} {'Weighted Dia (mm)':<20} {'Min Dia (mm)':<15}")
        print(f"  {'-'*25} {'-'*15} {'-'*20} {'-'*15}")

        paths = network.get_paths_to_outlets()
        for path in paths:
            if not path:
                continue
            
            outlet_name = path[-1].to_node.name
            path_length = sum(getattr(conn.component, 'length', 0) for conn in path)
            
            # Calculate weighted diameter
            total_length = 0
            weighted_dia_sum = 0
            min_diameter = float('inf')

            for conn in path:
                comp = conn.component
                length = getattr(comp, 'length', 0)
                diameter = getattr(comp, 'diameter', float('inf'))
                
                if length > 0 and diameter != float('inf'):
                    total_length += length
                    weighted_dia_sum += length * diameter
                
                if diameter != float('inf'):
                    min_diameter = min(min_diameter, diameter)

            weighted_diameter = weighted_dia_sum / total_length if total_length > 0 else 0
            
            print(f"  {outlet_name:<25} {path_length:<15.2f} {weighted_diameter * 1000:<20.2f} {min_diameter * 1000:<15.2f}")


        # --- Pressure and Flow Details ---
        print(f"\n{'='*80}")
        print("PRESSURE AND FLOW DETAILS")
        print(f"{'='*80}")
        
        if 'pressure_drops' not in solution_info:
            solution_info['pressure_drops'] = {}
            for connection in network.connections:
                component = connection.component
                flow_rate = connection_flows.get(component.id, 0.0)
                dp = component.calculate_pressure_drop(flow_rate, self.fluid_properties)
                solution_info['pressure_drops'][component.id] = dp
        
        print(f"  {'Component':<20} {'Type':<15} {'Flow Rate (' + q_unit_str + ')':<20} {'Pressure Drop (' + p_unit_str + ')':<20}")
        print(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*20}")
        
        for connection in network.connections:
            component = connection.component
            flow_rate = connection_flows.get(component.id, 0.0)
            pressure_drop = solution_info['pressure_drops'].get(component.id, 0)
            flow_rate_disp, _ = convert_flow_rate(flow_rate, q_unit_str)
            pressure_drop_disp, _ = convert_pressure(pressure_drop, p_unit_str)
            
            comp_type_str = getattr(component, 'component_type', type(component).__name__)
            if hasattr(comp_type_str, 'value'):
                comp_type_str = comp_type_str.value

            print(f"  {component.name:<20} {comp_type_str:<15} "
                  f"{flow_rate_disp:<20.3f} {pressure_drop_disp:<20.2f}")
        
        print(f"\n  {'Node':<20} {'Pressure (' + p_unit_str + ')':<20} {'Elevation (m)':<15}")
        print(f"  {'-'*20} {'-'*20} {'-'*15}")
        
        sorted_nodes = sorted(solution_info.get('node_pressures', {}).items(), key=lambda item: item[1], reverse=True)
        
        for node_id, pressure in sorted_nodes:
            node = network.nodes.get(node_id)
            if node:
                pressure_disp, _ = convert_pressure(pressure, p_unit_str)
                print(f"  {node.name:<20} {pressure_disp:<20.2f} {node.elevation:<15.1f}")
        
        if 'warnings' in solution_info and solution_info['warnings']:
            print(f"\n{'='*80}")
            print("WARNINGS")
            print(f"{'='*80}")
            for warning in solution_info['warnings']:
                print(f"  - {warning}")
        
        print(f"\n{'='*80}\n")

    

    
