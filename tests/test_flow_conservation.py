import pytest
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.connector import Connector
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.solvers.config import SolverConfig

# Helper to create a simple two-node one-channel network
def create_two_node_network(pressure_A, pressure_B, conductance):
    network = FlowNetwork("Test Network")

    node_A = Node(name="A", pressure=pressure_A, elevation=0.0)
    node_B = Node(name="B", pressure=pressure_B, elevation=0.0)

    network.add_node(node_A)
    network.add_node(node_B)
    network.set_inlet(node_A)
    network.add_outlet(node_B)

    # Resistance R = 1/G
    R = 1.0 / conductance
    channel = Channel(diameter=0.1, length=1.0)
    # Monkeypatch pressure_drop method for predictable behavior
    channel.calculate_pressure_drop = lambda Q, props: R * Q

    connection = network.connect_components(node_A, node_B, channel)

    return network, connection.component.id

# Fixtures for re-use
@pytest.fixture
def solver():
    return NodalMatrixSolver()

@pytest.fixture
def fluid_properties():
    return {"density": 900.0, "viscosity": 0.01}

def test_flat_network_flow_conservation(solver, fluid_properties):
    network, comp_id = create_two_node_network(200000, 190000, conductance=2.0)
    flow, info = solver.solve_nodal_network(
        network,
        total_flow_rate=0.005,
        temperature=40.0,
        inlet_pressure=200000,
        outlet_pressure=190000
    )
    assert comp_id in flow
    assert pytest.approx(flow[comp_id], rel=1e-3) == 0.005

def test_zero_flow_when_no_pressure_difference(solver, fluid_properties):
    network, comp_id = create_two_node_network(100000, 100000, conductance=5.0)
    flow, info = solver.solve_nodal_network(
        network,
        total_flow_rate=0.0,
        temperature=40.0,
        inlet_pressure=100000,
        outlet_pressure=100000
    )
    assert comp_id in flow
    assert abs(flow[comp_id]) < 1e-8

def test_inclined_network_hydrostatic_adjustment(solver, fluid_properties):
    network = FlowNetwork("Inclined Network")
    node_A = Node(name="A", elevation=0.0)
    node_B = Node(name="B", elevation=2.0)  # 2 m height difference

    network.add_node(node_A)
    network.add_node(node_B)
    network.set_inlet(node_A)
    network.add_outlet(node_B)

    channel = Channel(diameter=0.1, length=1.0)
    channel.calculate_pressure_drop = lambda Q, props: 10000 * Q  # R = 10000 Pa·s/m³

    network.connect_components(node_A, node_B, channel)

    flow, info = solver.solve_nodal_network(
        network,
        total_flow_rate=0.002,
        temperature=40.0,
        inlet_pressure=200000,
        outlet_pressure=101325
    )

    # Pressure diff from ρgΔz = 900 * 9.81 * 2 = 17658 Pa
    dp_expected = 17658 + channel.calculate_pressure_drop(0.002, fluid_properties)
    actual_dp = info["node_pressures"][node_A.id] - info["node_pressures"][node_B.id]

    assert actual_dp == pytest.approx(dp_expected, rel=1e-2)
