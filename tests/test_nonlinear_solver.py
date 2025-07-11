"""
Tests for the RobustNonLinearSolver, focusing on the Jacobian matrix construction.
"""

import pytest
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from lubrication_flow_package.solvers.nonlinear_solver import RobustNonLinearSolver
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
        self.id = name # Make sure component has an ID

    def calculate_pressure_drop(self, q, fluid_props):
        # A simple non-linear relationship: dP = R * q * |q|
        return self.resistance_val * q * np.abs(q)

# Fixture to provide a solver instance for tests
@pytest.fixture
def solver():
    """Provides a default RobustNonLinearSolver instance."""
    sim_config = SimulationConfig(
        total_flow_rate=0.1, 
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
    # Inlet/outlet are not needed for Jacobian tests
    return network

def test_jacobian_pressure_simple_square(solver, mocker):
    """
    Tests the pressure Jacobian for a simple square network with one cycle.
    All component connections are aligned in a clockwise direction.
    The test verifies that the Jacobian row for the cycle equation correctly
    contains the differential resistance of each component in the cycle.
    """
    nodes_data = [("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3"), ("N4", "Node 4")]
    connections_data = [
        ("N1", "N2", "C1", 10),  # Corresponds to Q_vector[0]
        ("N2", "N3", "C2", 20),  # Corresponds to Q_vector[1]
        ("N3", "N4", "C3", 30),  # Corresponds to Q_vector[2]
        ("N4", "N1", "C4", 40)   # Corresponds to Q_vector[3]
    ]
    network = create_network(nodes_data, connections_data)
    
    # Mock the differential resistance calculation to return predictable values.
    # We'll make it return a unique value for each component (10, 20, 30, 40).
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val 
    )

    q_vector = np.array([1.0, 1.0, 1.0, 1.0])
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # The system has 4 nodes and 1 cycle. The pressure equation corresponds to the last row.
    pressure_jacobian_row = jacobian.toarray()[4, :]
    
    # The expected row depends on the cycle found by networkx. We must build it
    # dynamically to ensure the test is robust.
    expected_row = np.zeros(len(connections_data))
    cycle = cycles[0] # There is only one cycle
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        conn = network.get_connection_by_nodes(u, v)
        comp_idx = comp_to_idx[conn.component.id]
        
        # The mocked resistance is just the component's configured value
        resistance = conn.component.resistance_val

        # The sign depends on whether the connection direction matches the cycle traversal
        if conn.from_node.id == u:
            expected_row[comp_idx] = resistance
        else:
            expected_row[comp_idx] = -resistance
            
    np.testing.assert_allclose(pressure_jacobian_row, expected_row, atol=1e-9)

def test_jacobian_pressure_square_with_reversed_edge(solver, mocker):
    """
    Tests the pressure Jacobian for a square network where one component's
    connection is oriented against the direction of cycle traversal.
    The test verifies that the corresponding entry in the Jacobian has a
    negative sign.
    """
    nodes_data = [("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3"), ("N4", "Node 4")]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 20),
        ("N4", "N3", "C3", 30),  # Reversed connection (N4 -> N3)
        ("N4", "N1", "C4", 40)
    ]
    network = create_network(nodes_data, connections_data)
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.array([1.0, 1.0, 1.0, 1.0])
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    pressure_jacobian_row = jacobian.toarray()[4, :]
    
    expected_row = np.zeros(len(connections_data))
    cycle = cycles[0]
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

    for i in range(len(cycle)):
        u, v = cycle[i], cycle[(i + 1) % len(cycle)]
        conn = network.get_connection_by_nodes(u, v)
        comp_idx = comp_to_idx[conn.component.id]
        resistance = conn.component.resistance_val

        if conn.from_node.id == u:
            expected_row[comp_idx] = resistance
        else:
            expected_row[comp_idx] = -resistance
            
    np.testing.assert_allclose(pressure_jacobian_row, expected_row, atol=1e-9)


def test_jacobian_pressure_with_branch(solver, mocker):
    """
    Tests the pressure Jacobian for a network with a cycle and a branch.
    The test verifies that the component in the branch (not part of the cycle)
    has a zero entry in the cycle's Jacobian row.
    """
    nodes_data = [("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3"), ("N4", "Node 4"), ("N5", "Node 5")]
    connections_data = [
        ("N1", "N2", "C1", 10), # Cycle component
        ("N2", "N3", "C2", 20), # Cycle component
        ("N3", "N4", "C3", 30), # Cycle component
        ("N4", "N1", "C4", 40), # Cycle component
        ("N2", "N5", "C5", 50)  # Branch component
    ]
    network = create_network(nodes_data, connections_data)
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # The branch component "C5" should not be in the cycle.
    # Its column in the pressure jacobian should be 0.
    pressure_jacobian_row = jacobian.toarray()[5, :] # 5 nodes, 1 cycle -> row index 5
    
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}
    branch_comp_idx = comp_to_idx["C5"]

    assert pressure_jacobian_row[branch_comp_idx] == 0.0


def test_jacobian_pressure_two_cycles(solver, mocker):
    """
    Tests the pressure Jacobian for a "figure-8" network with two cycles
    sharing a common component.
    The test verifies that two pressure equations are generated and that the
    shared component appears correctly in both Jacobian rows.
    """
    nodes_data = [
        ("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3"),
        ("N4", "Node 4"), ("N5", "Node 5"), ("N6", "Node 6")
    ]
    connections_data = [
        ("N1", "N2", "C1", 10), # Left cycle
        ("N2", "N4", "C2", 20), # Left cycle
        ("N4", "N1", "C3", 30), # Left cycle
        ("N2", "N3", "C4", 40), # Right cycle
        ("N3", "N5", "C5", 50), # Right cycle
        ("N5", "N2", "C6", 60)  # Right cycle
    ]
    network = create_network(nodes_data, connections_data)
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.ones(len(connections_data))
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # 6 nodes, 2 cycles. Pressure rows are the last two, indices 6 and 7.
    assert len(cycles) == 2
    assert jacobian.shape[0] == 6 + 2

    pressure_jacobian = jacobian.toarray()[6:, :]

    # Dynamically build the expected Jacobian for both cycles
    expected_jacobian = np.zeros((2, len(connections_data)))
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

    for i, cycle in enumerate(cycles):
        for j in range(len(cycle)):
            u, v = cycle[j], cycle[(j + 1) % len(cycle)]
            conn = network.get_connection_by_nodes(u, v)
            comp_idx = comp_to_idx[conn.component.id]
            resistance = conn.component.resistance_val

            if conn.from_node.id == u:
                expected_jacobian[i, comp_idx] = resistance
            else:
                expected_jacobian[i, comp_idx] = -resistance

    # Check that the calculated pressure jacobian matches the expected one.
    # The order of cycles from networkx is not guaranteed, so we must check
    # that the set of rows is the same.
    # Convert rows to tuples to make them hashable for set comparison.
    calculated_rows = set(map(tuple, pressure_jacobian))
    expected_rows = set(map(tuple, expected_jacobian))

    assert calculated_rows == expected_rows
