"""
Tests for the NonLinearTreeSolver.
"""

import pytest
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.components.connector import Connector, ConnectorType
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.solvers.tree_solver import NonLinearTreeSolver

from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.config.simulation_config import SimulationConfig


def test_component_flow_calculation():
    """
    Test the inverse flow calculation for components.
    This test must pass before the solver implementation begins.
    """
    fluid_properties = {'density': 900, 'viscosity': 0.015}
    
    # Test Channel
    channel = Channel(diameter=0.05, length=10)
    pressure_drop = 50000  # Pa
    
    # Add a placeholder for the new method
    if hasattr(channel, 'calculate_flow_rate'):
        flow_rate = channel.calculate_flow_rate(pressure_drop, fluid_properties)
        # Verify that the calculated flow rate produces the original pressure drop
        calculated_dp = channel.calculate_pressure_drop(flow_rate, fluid_properties)
        assert calculated_dp == pytest.approx(pressure_drop, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Channel")

    # Test Nozzle
    nozzle = Nozzle(diameter=0.01, nozzle_type=NozzleType.ROUNDED)
    pressure_drop_nozzle = 20000 # Pa

    if hasattr(nozzle, 'calculate_flow_rate'):
        flow_rate_nozzle = nozzle.calculate_flow_rate(pressure_drop_nozzle, fluid_properties)
        calculated_dp_nozzle = nozzle.calculate_pressure_drop(flow_rate_nozzle, fluid_properties)
        assert calculated_dp_nozzle == pytest.approx(pressure_drop_nozzle, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Nozzle")

    # Test Connector
    connector = Connector(diameter=0.05, connector_type=ConnectorType.ELBOW_90)
    pressure_drop_connector = 10000 # Pa

    if hasattr(connector, 'calculate_flow_rate'):
        flow_rate_connector = connector.calculate_flow_rate(pressure_drop_connector, fluid_properties)
        calculated_dp_connector = connector.calculate_pressure_drop(flow_rate_connector, fluid_properties)
        assert calculated_dp_connector == pytest.approx(pressure_drop_connector, rel=1e-3)
    else:
        pytest.skip("calculate_flow_rate method not yet implemented on Connector")


def test_solver_on_simple_tree():
    """
    Test the solver on a simple tree network.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.01,
        temperature=40.0,
        inlet_pressure=200000.0,
        oil_type="SAE30",
        outlet_pressure=101325.0
    )

    builder = NetworkBuilder(sim_config)
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "j1", length=10, diameter=0.05)
        .add_pipe("j1", "out1", length=5, diameter=0.03)
        .add_nozzle("out1", "nozzle1", diameter=0.01)
        .add_outlet("nozzle1")
        .build()
    )

    solver = NonLinearTreeSolver(sim_config)
    solution = solver.solve(network)

    assert solution['converged']
    assert solution['node_pressures'][network.get_node('inlet').id] > solution['node_pressures'][network.get_node('j1').id]
    assert solution['node_pressures'][network.get_node('j1').id] > solution['node_pressures'][network.get_node('out1').id]
    assert solution['node_pressures'][network.get_node('out1').id] > solution['node_pressures'][network.get_node('nozzle1').id]
    assert solution['node_pressures'][network.get_node('nozzle1').id] > 0


def test_solver_against_linear_solver():
    """
    Compare the new solver's results against the existing NodalMatrixSolver.
    """
    sim_config = SimulationConfig(
        total_flow_rate=0.0001, # Low flow rate to minimize non-linear effects
        temperature=40.0,
        inlet_pressure=200000.0,
        oil_type="SAE30",
        outlet_pressure=101325.0
    )

    builder = NetworkBuilder(sim_config)
    network = (builder
        .set_inlet("inlet")
        .add_pipe("inlet", "j1", length=10, diameter=0.05)
        .add_pipe("j1", "out1", length=5, diameter=0.03)
        .add_nozzle("out1", "nozzle1", diameter=0.01)
        .add_outlet("nozzle1")
        .build()
    )

    # Solve with the new non-linear solver
    nonlinear_solver = NonLinearTreeSolver(sim_config)
    nonlinear_solution = nonlinear_solver.solve(network)

    # Solve with the linear solver
    linear_solver = NodalMatrixSolver(sim_config)
    linear_solution = linear_solver.solve(network)

    # Compare the results
    assert nonlinear_solution['converged']
    for node_id in linear_solution['node_pressures']:
        assert nonlinear_solution['node_pressures'][node_id] == pytest.approx(
            linear_solution['node_pressures'][node_id], rel=1e-2
        )
    for comp_id in linear_solution['component_flows']:
        assert nonlinear_solution['component_flows'][comp_id] == pytest.approx(
            linear_solution['component_flows'][comp_id], rel=1e-2
        )
