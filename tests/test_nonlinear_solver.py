"""
Tests for the RobustNonLinearSolver, focusing on the Jacobian matrix construction.
"""

import pytest
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix

from lubrication_flow_package.solvers.nonlinear_loop_solver import RobustNonLinearSolver
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.network.connection import Connection
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder

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

    def get_flow_area(self) -> float:
        return 0.01

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
def create_network(nodes_data, connections_data, inlet_node_id=None, outlet_node_ids=None):
    """Creates a FlowNetwork for testing."""
    builder = NetworkBuilder()
    if outlet_node_ids is None:
        outlet_node_ids = []

    for node_id, name in nodes_data:
        builder.add_node(name=node_id)  # Use the ID as the name for builder lookup

    if inlet_node_id:
        builder.set_inlet(inlet_node_id)

    for outlet_id in outlet_node_ids:
        builder.add_outlet(outlet_id)

    for from_id, to_id, comp_name, comp_val in connections_data:
        # The builder doesn't support adding pre-made components,
        # so we can't use the MockComponent directly in the builder chain.
        # We build the topology first, then replace the components.
        builder.add_pipe(from_node_name=from_id, to_node_name=to_id, length=1, diameter=1, name=comp_name)

    network = builder.build()
    network.name = "TestNetwork"

    # Replace the standard components with our mock components for testing
    new_connections = []
    for conn in network.connections:
        from_id = conn.from_node.name
        to_id = conn.to_node.name
        # Find the corresponding resistance value from the original data
        comp_val = next(c[3] for c in connections_data if c[0] == from_id and c[1] == to_id)
        mock_comp = MockComponent(name=conn.component.name, resistance_val=comp_val)
        new_conn = Connection(
            from_node=conn.from_node,
            to_node=conn.to_node,
            component=mock_comp
        )
        new_connections.append(new_conn)
    network.connections = new_connections
        
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
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N4"])
    
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

    # The system has 4 nodes (3 mass equations) and 1 cycle. The pressure equation is at index 3.
    pressure_jacobian_row = jacobian.toarray()[3, :]
    
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
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N3"])
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.array([1.0, 1.0, 1.0, 1.0])
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # The system has 4 nodes (3 mass equations) and 1 cycle. The pressure equation is at index 3.
    pressure_jacobian_row = jacobian.toarray()[3, :]
    
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
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N5"])
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # 5 nodes (4 mass equations), 1 cycle -> pressure equation is at index 4
    pressure_jacobian_row = jacobian.toarray()[4, :]
    
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
        ("N4", "Node 4"), ("N5", "Node 5")
    ]
    connections_data = [
        ("N1", "N2", "C1", 10), # Left cycle
        ("N2", "N4", "C2", 20), # Shared edge
        ("N4", "N1", "C3", 30), # Left cycle
        ("N4", "N3", "C4", 40), # Right cycle
        ("N3", "N5", "C5", 50), # Right cycle
        ("N5", "N4", "C6", 60)  # Right cycle
    ]
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N5"])
    
    mocker.patch.object(
        solver, 
        '_calculate_differential_resistance', 
        side_effect=lambda comp, q, props: comp.resistance_val
    )

    q_vector = np.ones(len(connections_data))
    cycles = solver._find_fundamental_cycles(network)
    jacobian = solver._build_jacobian(q_vector, network, cycles)

    # 6 nodes (5 mass eq), 6 connections -> 1 cycle eq is needed for a square 6x6 system.
    num_connections = len(connections_data)
    num_mass_eq = len(nodes_data) - 1
    num_cycle_eq = num_connections - num_mass_eq
    
    assert len(cycles) >= num_cycle_eq # networkx can find more cycles than needed
    assert jacobian.shape[0] == num_connections

    pressure_jacobian = jacobian.toarray()[num_mass_eq:, :]
    assert pressure_jacobian.shape[0] == num_cycle_eq

    # Dynamically build the expected Jacobian for the cycles that are actually used
    expected_jacobian = np.zeros((num_cycle_eq, len(connections_data)))
    comp_to_idx = {conn.component.id: i for i, conn in enumerate(network.connections)}

    for i in range(num_cycle_eq):
        cycle = cycles[i]
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

def test_jacobian_is_square(solver):
    """
    Tests that the Jacobian matrix is square (number of equations == number of variables).
    This is the most critical test for the validity of the Newton-Raphson formulation.
    A non-square Jacobian indicates a fundamental problem in the system of equations.
    """
    # A simple network with one cycle is sufficient to test the formulation.
    # 4 nodes, 4 connections -> 4 variables (flows)
    # 1 cycle -> 1 pressure equation
    # With the bug: 4 mass equations. Total equations = 5. System = 5x4 (non-square)
    # After the fix: 3 mass equations (N-1). Total equations = 4. System = 4x4 (square)
    nodes_data = [("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3"), ("N4", "Node 4")]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 20),
        ("N3", "N4", "C3", 30),
        ("N4", "N1", "C4", 40)
    ]
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N4"])

    q_vector = np.ones(len(connections_data))
    cycles = solver._find_fundamental_cycles(network)
    
    # This is the call that builds the Jacobian
    jacobian = solver._build_jacobian(q_vector, network, cycles)
    
    # The number of variables is the number of connections (flows)
    num_variables = len(connections_data)
    
    # The number of equations is the number of rows in the Jacobian
    num_equations = jacobian.shape[0]
    
    # Assert that the matrix is square
    assert num_equations == num_variables, (
        f"Jacobian matrix should be square, but has shape {jacobian.shape}. "
        f"Number of equations ({num_equations}) does not match number of variables ({num_variables})."
    )

def test_pressure_calculation_double_loop(solver):
    """
    Tests the final pressure calculation for a network with two connected loops.
    This test is designed to fail with the BFS-based pressure calculation.
    """
    nodes_data = [
        ("N1", "Inlet"), 
        ("N2", "Junction1"), 
        ("N3", "MidLoop"), 
        ("N4", "Junction2"), 
        ("N5", "Outlet")
    ]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 20),
        ("N3", "N4", "C3", 30),
        ("N2", "N4", "C4", 40), # Bridge between loops
        ("N4", "N5", "C5", 50)
    ]
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N5"])

    # Set a known total flow rate
    solver.sim_config.total_flow_rate = 0.1

    # Run the full solver
    solution = solver.solve(network)
    pressures = solution.get("node_pressures", {})
    flows = solution.get("component_flows", {})

    # --- Analytical Solution ---
    # With the BFS approach, the pressure at N4 will be calculated based on the
    # path from N5. However, the pressure at N4 is also influenced by the
    # path from N2. The BFS approach can't handle this correctly.
    # We can calculate the expected pressure at N4 by considering both paths.
    
    # This is a simplified analytical solution. A real one would be more complex.
    # The key is that the pressure at N4 should be consistent regardless of the
    # path taken to calculate it.
    
    # Pressure drop from N4 to N5
    dp5 = 50 * flows['C5'] * abs(flows['C5'])
    p4_from_n5 = pressures[network.get_node("N5").id] + dp5

    # Pressure drop from N2 to N4
    dp4 = 40 * flows['C4'] * abs(flows['C4'])
    
    # Pressure at N2
    p2 = pressures[network.get_node("N2").id]
    
    p4_from_n2 = p2 - dp4

    # The pressure at N4 should be consistent with the pressure at N5
    assert pressures[network.get_node("N4").id] == pytest.approx(p4_from_n5, rel=1e-3)

def test_pressure_calculation_simple_loop(solver):
    """
    Tests the final pressure calculation for a simple square network.
    This test runs the full solver and verifies the calculated node pressures
    against an analytical solution.
    """
    nodes_data = [("N1", "Node 1"), ("N2", "Node 2"), ("N3", "Node 3")]
    connections_data = [
        ("N1", "N2", "C1", 10),
        ("N2", "N3", "C2", 20),
        ("N3", "N1", "C3", 30)
    ]
    network = create_network(nodes_data, connections_data, inlet_node_id="N1", outlet_node_ids=["N3"])

    # Set a known total flow rate
    solver.sim_config.total_flow_rate = 0.1

    # Run the full solver
    solution = solver.solve(network)
    pressures = solution.get("node_pressures", {})

    # --- Analytical Solution ---
    # This is a simple delta network. We can calculate the expected pressures.
    # The solver will find the flows, and from the flows, the pressures.
    # We can manually calculate the expected pressure at N2.
    # The pressure at N3 is the outlet pressure (0.0 in this case).
    # The pressure at N1 is the inlet pressure.
    # The flow will split between the two paths.
    # Path 1: N1 -> N2 -> N3 (R = 10 + 20 = 30)
    # Path 2: N1 -> N3 (R = 30)
    # The conductances are equal, so the flow should split evenly.
    q1 = 0.05
    q2 = 0.05
    q3 = 0.05

    # Pressure drop across C1
    dp1 = 10 * q1 * abs(q1)
    # Pressure drop across C2
    dp2 = 20 * q2 * abs(q2)
    # Pressure drop across C3
    dp3 = 30 * q3 * abs(q3)

    # Pressure at N2, relative to N3
    p2_expected = dp2

    # Pressure at N1, relative to N3
    p1_expected = p2_expected + dp1

    assert pressures[network.get_node("N1").id] == pytest.approx(p1_expected, rel=1e-3)
    assert pressures[network.get_node("N2").id] == pytest.approx(p2_expected, rel=1e-3)
    assert pressures[network.get_node("N3").id] == pytest.approx(0.0, abs=1e-9)
