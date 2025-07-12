"""
Tests for the NetworkBuilder utility.

This test suite follows a TDD approach to verify the functionality
of the NetworkBuilder before and after its implementation.
"""

import pytest
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.base import ConnectorType, NozzleType
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.connector import Connector
from lubrication_flow_package.components.nozzle import Nozzle

@pytest.fixture
def dummy_config():
    """Provides a dummy SimulationConfig for tests."""
    return SimulationConfig(
        total_flow_rate=0.01,
        temperature=40.0,
        inlet_pressure=200000.0
    )

@pytest.fixture
def builder(dummy_config):
    """Provides a fresh NetworkBuilder instance for each test."""
    return NetworkBuilder(sim_config=dummy_config)

# --- Test Group 1: Core Builder Functionality ---

def test_builder_initialization(builder: NetworkBuilder):
    """Test that the builder initializes correctly and can build an empty network."""
    assert builder._network is not None
    builder.set_inlet("in")
    builder.add_outlet("out")
    builder.add_pipe("in", "out", length=1, diameter=0.1)
    network = builder.build()
    assert isinstance(network, FlowNetwork)
    assert len(network.nodes) == 2
    assert len(network.connections) == 1

def test_set_inlet_and_outlet(builder: NetworkBuilder):
    """Test setting inlet and outlet nodes."""
    builder.set_inlet("in")
    builder.add_outlet("out")
    builder.add_pipe("in", "out", length=1, diameter=0.1) # Add a path
    network = builder.build()
    
    assert network.inlet_node is not None
    assert network.inlet_node.name == "in"
    assert len(network.outlet_nodes) == 1
    assert network.outlet_nodes[0].name == "out"

def test_get_or_create_node_uniqueness(builder: NetworkBuilder):
    """Test that nodes are created only once."""
    builder.set_inlet("A")
    builder.add_pipe("A", "B", length=1, diameter=0.1)
    builder.add_pipe("B", "C", length=1, diameter=0.1)
    builder.add_outlet("C")
    network = builder.build()
    
    assert len(network.nodes) == 3  # Should not create 4 nodes

# --- Test Group 2: Component-Adding Methods ---

def test_add_pipe(builder: NetworkBuilder):
    """Test the add_pipe method."""
    builder.set_inlet("N1")
    builder.add_outlet("N2")
    builder.add_pipe("N1", "N2", length=5.0, diameter=0.05, roughness=0.001)
    network = builder.build()
    
    assert len(network.nodes) == 2
    assert len(network.connections) == 1
    
    connection = network.connections[0]
    pipe = connection.component
    
    assert isinstance(pipe, Channel)
    assert pipe.length == 5.0
    assert pipe.diameter == 0.05
    assert pipe.roughness == 0.001
    assert connection.from_node.name == "N1"
    assert connection.to_node.name == "N2"

def test_add_nozzle(builder: NetworkBuilder):
    """Test the add_nozzle method."""
    builder.set_inlet("N1")
    builder.add_outlet("N2")
    builder.add_nozzle("N1", "N2", diameter=0.01, nozzle_type=NozzleType.ROUNDED)
    network = builder.build()
    
    assert len(network.connections) == 1
    nozzle = network.connections[0].component
    
    assert isinstance(nozzle, Nozzle)
    assert nozzle.diameter == 0.01
    assert nozzle.nozzle_type == NozzleType.ROUNDED

def test_add_fitting(builder: NetworkBuilder):
    """Test the generic add_fitting method."""
    builder.set_inlet("N1")
    builder.add_outlet("N2")
    builder.add_fitting("N1", "N2", ConnectorType.ELBOW_90, diameter=0.1, loss_coefficient=0.9)
    network = builder.build()
    
    assert len(network.connections) == 1
    fitting = network.connections[0].component
    
    assert isinstance(fitting, Connector)
    assert fitting.connector_type == ConnectorType.ELBOW_90
    assert fitting.loss_coefficient == 0.9

# --- Test Group 3: T-Junction Logic ---

def test_add_tee_junction_topology(builder: NetworkBuilder):
    """Test that add_tee_junction creates the correct number of nodes and connections."""
    builder.set_inlet("main_in")
    builder.add_outlet("main_out")
    builder.add_outlet("branch_out")
    builder.add_tee_junction("main_in", "main_out", "branch_out", "T1", diameter=0.1)
    network = builder.build()
    
    assert len(network.nodes) == 4  # main_in, main_out, branch_out, T1
    assert len(network.connections) == 3

def test_add_tee_junction_physics(builder: NetworkBuilder):
    """Test that the T-junction has correct, asymmetric loss coefficients."""
    builder.set_inlet("main_in")
    builder.add_outlet("main_out")
    builder.add_outlet("branch_out")
    builder.add_tee_junction("main_in", "main_out", "branch_out", "T1", diameter=0.1)
    network = builder.build()
    
    # Find the three connectors associated with the tee
    tee_connectors = [c.component for c in network.connections]
    
    k_values = sorted([conn.loss_coefficient for conn in tee_connectors])
    
    # Assert that the K-factors match the physically-aware model
    expected_k_values = sorted([0.05, 0.2, 1.0])
    assert k_values == pytest.approx(expected_k_values)

# --- Test Group 4: Integration and Error Handling ---

def test_network_chaining(builder: NetworkBuilder):
    """Test building a simple network by chaining calls."""
    builder.set_inlet("in")
    builder.add_pipe("in", "T1", length=2, diameter=0.1)
    builder.add_tee_junction("T1", "mid", "branch", "TeeNode", diameter=0.1)
    builder.add_nozzle("mid", "out1", diameter=0.05)
    builder.add_nozzle("branch", "out2", diameter=0.05)
    builder.add_outlet("out1")
    builder.add_outlet("out2")
    
    network = builder.build()
    
    assert len(network.nodes) == 7
    assert len(network.connections) == 6 # 1 pipe + 3 for tee + 2 nozzles
    assert network.inlet_node.name == "in"
    assert len(network.outlet_nodes) == 2

def test_build_invalid_network_raises_error(builder: NetworkBuilder):
    """Test that building an invalid network raises a ValueError."""
    builder.set_inlet("A")
    builder.add_outlet("C")
    builder.add_pipe("A", "B", length=1, diameter=0.1)
    # Node "C" is an outlet but is not connected to anything.
    
    with pytest.raises(ValueError, match="Constructed network is invalid"):
        builder.build()
