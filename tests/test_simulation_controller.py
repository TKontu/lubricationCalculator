"""
Tests for the SimulationController.
"""
import unittest
from unittest.mock import Mock, patch, MagicMock

import pytest

from lubrication_flow_package.simulation.simulation_controller import SimulationController
from lubrication_flow_package.config.network_config import NetworkConfig
from lubrication_flow_package.config.simulation_config import SimulationConfig

@pytest.fixture
def mock_configs():
    """Fixture to create mock configuration objects."""
    network_config = Mock(spec=NetworkConfig)
    network_config.network_name = "Test Network"
    network_config.simulation = {
        'total_flow_rate': 0.02,
        'temperature': 50,
        'inlet_pressure': 300000,
        'oil_type': 'SAE30',
        'viscosity_model': 'vogel',
        'input_flow_rate_unit': 'm3/s',
        'input_pressure_unit': 'Pa',
        'output_pressure_unit': 'Pa',
        'output_flow_rate_unit': 'm3/s'
    }
    network_config.nodes = [
        {'id': 'inlet', 'type': 'inlet'},
        {'id': 'outlet', 'type': 'outlet'}
    ]
    network_config.components = [
        {'id': 'pipe', 'type': 'channel', 'length': 1, 'diameter': 0.1}
    ]
    network_config.connections = [
        {'from_node': 'inlet', 'to_node': 'outlet', 'component': 'pipe'}
    ]
    sim_config = SimulationConfig.from_dict(network_config.simulation)
    return network_config, sim_config

@pytest.fixture
def controller():
    """Fixture to create a SimulationController instance."""
    return SimulationController()

def test_initialization(controller: SimulationController):
    """Test that the controller initializes with default values."""
    assert controller.network_config is None
    assert controller.sim_config is None
    assert controller.network is None
    assert controller.solver is None
    assert controller.results is None

@patch('lubrication_flow_package.simulation.simulation_controller.NetworkConfigLoader')
def test_load_configuration(mock_loader, controller: SimulationController, mock_configs):
    """Test that loading a configuration builds the network."""
    network_config, sim_config = mock_configs
    mock_network = MagicMock()
    mock_loader.build_network.return_value = (mock_network, sim_config)

    controller.load_configuration(network_config, sim_config)

    assert controller.network_config == network_config
    mock_loader.build_network.assert_called_once_with(network_config)
    assert controller.network == mock_network

def test_set_solver(controller: SimulationController, mock_configs):
    """Test that the correct solver is created."""
    _, sim_config = mock_configs
    controller.sim_config = sim_config

    controller.set_solver("nodal")
    assert controller.solver is not None
    assert "NodalMatrixSolver" in str(type(controller.solver))

    controller.set_solver("tree_nonlinear")
    assert controller.solver is not None
    assert "TreeSolver" in str(type(controller.solver))

    with pytest.raises(ValueError, match="Unknown solver: invalid_solver"):
        controller.set_solver("invalid_solver")

@patch('lubrication_flow_package.simulation.simulation_controller.TreeSolver')
def test_run_simulation_success(mock_solver_cls, controller: SimulationController, mock_configs):
    """Test a successful simulation run."""
    network_config, sim_config = mock_configs
    
    # Mock the solver instance and its solve method
    mock_solver_instance = mock_solver_cls.return_value
    mock_solver_instance.solve.return_value = {"converged": True, "iterations": 10}
    
    # Setup controller
    controller.load_configuration(network_config, sim_config)
    controller.solver = mock_solver_instance

    converged = controller.run_simulation()

    assert converged is True
    mock_solver_instance.solve.assert_called_once_with(controller.network)
    assert controller.results is not None
    assert controller.results["converged"] is True

@patch('lubrication_flow_package.simulation.simulation_controller.NodalMatrixSolver')
def test_run_simulation_failure(mock_solver_cls, controller: SimulationController, mock_configs):
    """Test a failed simulation run."""
    network_config, sim_config = mock_configs
    
    mock_solver_instance = mock_solver_cls.return_value
    mock_solver_instance.solve.return_value = {"converged": False, "warnings": ["Did not converge"]}
    
    controller.load_configuration(network_config, sim_config)
    controller.solver = mock_solver_instance

    converged = controller.run_simulation()

    assert converged is False
    assert controller.results["converged"] is False

def test_run_simulation_no_network(controller: SimulationController):
    """Test running simulation without a network loaded."""
    controller.solver = Mock()
    converged = controller.run_simulation()
    assert converged is False

def test_get_results(controller: SimulationController):
    """Test that get_results returns the stored results."""
    mock_results = {"converged": True, "data": "some_data"}
    controller.results = mock_results
    assert controller.get_results() == mock_results

def test_progress_callback(controller: SimulationController):
    """Test that the progress callback is called correctly."""
    mock_callback = Mock()
    controller.set_progress_callback(mock_callback)

    controller._report_progress("Test message")
    
    mock_callback.assert_called_once_with("Test message")

def test_get_available_solvers(controller: SimulationController):
    """Test that the list of available solvers is correct."""
    assert controller.get_available_solvers() == ["nodal", "tree_nonlinear"]
