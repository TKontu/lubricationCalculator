"""
Tests for the unified print_results method in SolverBase.
"""

import pytest
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.config.simulation_config import SimulationConfig

@pytest.fixture
def solved_network():
    """
    Provides a simple, solved network to test the print_results method.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.001,
        oil_density=850,
        oil_type="SAE30",
        temperature=40,
        inlet_pressure=200000
    )
    solver = NodalMatrixSolver(sim_config)
    
    net = FlowNetwork("test_print")
    n1 = net.create_node("N1")
    n2 = net.create_node("N2")
    net.set_inlet(n1)
    net.add_outlet(n2)
    
    channel = Channel(length=1.0, diameter=0.01, component_id="C1")
    net.connect_components(n1, n2, channel)
    
    solution = solver.solve(net)
    return solver, net, solution

def test_print_results_output(solved_network, capsys):
    """
    Tests that the print_results method produces the expected output.
    """
    solver, network, solution = solved_network
    
    solver.print_results(network, solution, pressure_unit='bar', flow_rate_unit='L/min')
    
    captured = capsys.readouterr()
    output = captured.out
    
    # Check for key sections
    assert "NETWORK FLOW SIMULATION RESULTS" in output
    assert "OUTLET FLOW DISTRIBUTION" in output
    assert "PRESSURE AND FLOW DETAILS" in output
    
    # Check for specific results with units
    assert "Inlet Pressure:" in output
    assert "bar" in output
    assert "Total System Flow Rate:" in output
    assert "L/min" in output
    assert "Component" in output
    assert "Pressure Drop (bar)" in output
    assert "Flow Rate (L/min)" in output
    assert "Node" in output
    assert "Pressure (bar)" in output
