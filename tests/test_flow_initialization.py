"""
Tests for the solver flow initialization methods.
"""

import pytest
import numpy as np

from lubrication_flow_package.solvers.nonlinear_loop_solver import RobustNonLinearSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.network.connection import Connection
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.config.simulation_config import SimulationConfig

# Mock Component for testing purposes
class MockComponent(FlowComponent):
    """A mock component with a configurable pressure drop for testing."""
    def __init__(self, name, resistance_val=1.0):
        super().__init__(name)
        self.resistance_val = resistance_val
        self.id = name

    def calculate_pressure_drop(self, q, fluid_props):
        # A simple quadratic relationship: dP = R * q^2
        return self.resistance_val * q * np.abs(q)

# Fixture to provide a solver instance for tests
@pytest.fixture
def solver():
    """Provides a default RobustNonLinearSolver instance."""
    sim_config = SimulationConfig(
        total_flow_rate=1.0,  # Use 1.0 for easy-to-calculate ratios
        oil_density=850,
        oil_type="SAE30",
        temperature=40,
        inlet_pressure=101325
    )
    solver_instance = RobustNonLinearSolver(sim_config)
    return solver_instance

# Helper function to create a FlowNetwork from simplified data
def create_network(nodes_data, connections_data):
    """Creates a FlowNetwork for testing."""
    nodes = {node_id: Node(id=node_id, name=name) for node_id, name in nodes_data}
    connections = []
    for from_id, to_id, comp_name, comp_val in connections_data:
        comp = MockComponent(name=comp_name, resistance_val=comp_val)
        conn = Connection(
            from_node=nodes[from_id],
            to_node=nodes[to_id],
            component=comp
        )
        connections.append(conn)
    
    network = FlowNetwork(name="TestNetwork")
    network.nodes = nodes
    network.connections = connections
    
    if nodes_data:
        network.inlet_node = nodes[nodes_data[0][0]]
        # Assume the last node is the outlet for simplicity in these tests
        network.outlet_nodes = [nodes[nodes_data[-1][0]]]
        
    return network

def test_series_initialization(solver):
    """
    Tests that for a simple series network, the initial flow in all
    components is equal to the total system flow rate.
    """
    nodes_data = [("N1", "Inlet"), ("N2", "Mid"), ("N3", "Outlet")]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 10)
    ]
    network = create_network(nodes_data, connections_data)
    
    # The new initialization method will be called `_initialize_flows_linear`
    # to distinguish it from the old one.
    q_initial = solver._initialize_flows(network)
    
    # In a series circuit, flow should be the same everywhere
    expected_flow = solver.sim_config.total_flow_rate
    np.testing.assert_allclose(q_initial, expected_flow, rtol=1e-6)


def test_parallel_equal_resistance(solver):
    """
    Tests that for a network with two identical parallel branches,
    the initial flow is split exactly in half.
    """
    nodes_data = [("N1", "Inlet"), ("N2", "Junction"), ("N3", "Outlet")]
    connections_data = [
        ("N1", "N2", "C_in", 1),
        ("N2", "N3", "C_p1", 10), # Parallel branch 1
        ("N2", "N3", "C_p2", 10)  # Parallel branch 2
    ]
    network = create_network(nodes_data, connections_data)
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}
    
    q_initial = solver._initialize_flows(network)
    
    # Check that the flow in the parallel branches is half of the total flow
    total_flow = solver.sim_config.total_flow_rate
    flow_p1 = q_initial[comp_to_idx['C_p1']]
    flow_p2 = q_initial[comp_to_idx['C_p2']]
    
    assert np.isclose(flow_p1, total_flow / 2, rtol=1e-6)
    assert np.isclose(flow_p2, total_flow / 2, rtol=1e-6)
    

def test_parallel_unequal_resistance(solver):
    """
    Tests that for parallel branches of unequal resistance, flow is
    distributed inversely proportional to the resistance.
    """
    nodes_data = [("N1", "Inlet"), ("N2", "Junction"), ("N3", "Outlet")]
    # Branch 2 has 4 times the resistance of Branch 1
    connections_data = [
        ("N1", "N2", "C_in", 1),
        ("N2", "N3", "C_p1", 10), # Low resistance branch
        ("N2", "N3", "C_p2", 40)  # High resistance branch
    ]
    network = create_network(nodes_data, connections_data)
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}
    
    q_initial = solver._initialize_flows(network)
    
    # Flow should be split 4:1
    # Q1/Q2 = R2/R1. Q1+Q2 = Q_total.
    # Q1 = Q_total * R2/(R1+R2)
    total_flow = solver.sim_config.total_flow_rate
    flow_p1 = q_initial[comp_to_idx['C_p1']]
    flow_p2 = q_initial[comp_to_idx['C_p2']]
    
    expected_flow_p1 = total_flow * 40 / (10 + 40) # 0.8
    expected_flow_p2 = total_flow * 10 / (10 + 40) # 0.2
    
    assert np.isclose(flow_p1, expected_flow_p1, rtol=1e-6)
    assert np.isclose(flow_p2, expected_flow_p2, rtol=1e-6)


def test_initialization_mass_conservation(solver):
    """
    Tests that the initialized flows conserve mass at a junction.
    """
    nodes_data = [
        ("N1", "Inlet"), ("N2", "Junction1"), 
        ("N3", "Mid"), ("N4", "Junction2"), 
        ("N5", "Outlet")
    ]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 20),
        ("N2", "N4", "C3", 30),
        ("N3", "N4", "C4", 5),
        ("N4", "N5", "C5", 15)
    ]
    network = create_network(nodes_data, connections_data)
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}
    
    q_initial = solver._initialize_flows(network)
    
    # Check mass conservation at Junction 2 (N4)
    flow_in_c3 = q_initial[comp_to_idx['C3']]
    flow_in_c4 = q_initial[comp_to_idx['C4']]
    flow_out_c5 = q_initial[comp_to_idx['C5']]
    
    assert np.isclose(flow_in_c3 + flow_in_c4, flow_out_c5, rtol=1e-6)
