import pytest
import math

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder

# Physical constants for test
DENSITY = 900.0      # kg/m³
VISCOSITY = 1e-3     # Pa·s (use a known simple value)
TEMPERATURE = 20.0   # °C
INLET_P = 2e5        # Pa
OUTLET_P = 1e5       # Pa
Q_TOTAL = 1e-4       # m³/s

@pytest.fixture
def simple_pipe_network():
    """
    Two nodes connected by a single channel.
    Analytical ΔP = (128 μ L / (π D^4)) · Q
    """
    builder = NetworkBuilder()
    network = (builder
        .set_inlet("inlet")
        .add_outlet("outlet")
        .add_pipe("inlet", "outlet", length=1.0, diameter=0.02, name="ch0")
        .build()
    )
    inlet_id = network.get_node("inlet").id
    outlet_id = network.get_node("outlet").id
    ch = network.get_connection_by_nodes(inlet_id, outlet_id).component
    return network, ch

def test_two_node_case(simple_pipe_network):
    net, ch = simple_pipe_network
    sim_config = SimulationConfig(oil_density=DENSITY, oil_type="VG220", temperature=TEMPERATURE, total_flow_rate=Q_TOTAL, inlet_pressure=INLET_P, outlet_pressure=OUTLET_P)
    solver = NodalMatrixSolver(sim_config)
    # monkey-patch fluid properties for this test
    solver.fluid_properties = {'density': DENSITY, 'viscosity': VISCOSITY}

    # call unified interface
    info = solver.solve(net)
    flows = info.get("component_flows", {})

    # 1) only one connection
    assert ch.id in flows
    Q = flows[ch.id]

    # 2) flow matches requested total (within solver tolerance)
    assert math.isclose(Q, Q_TOTAL, rel_tol=1e-3)

    # 3) pressure drop matches the channel's own API
    fluid_props = {'density': DENSITY, 'viscosity': VISCOSITY}
    expected_dp = ch.calculate_pressure_drop(Q, fluid_props)
    dp_solver   = info["pressure_drops"][ch.id]
    assert math.isclose(
        dp_solver, expected_dp, rel_tol=1e-6
    ), f"DP mismatch: solver {dp_solver} vs channel {expected_dp}"

    # 4) inlet/outlet pressures honored
    expected_inlet_pressure = expected_dp + OUTLET_P
    assert math.isclose(info["inlet_pressure"], expected_inlet_pressure, rel_tol=1e-6)
    assert info["outlet_pressure"] == OUTLET_P

def test_mass_conservation_and_branching():
    """
    Simple T‐junction: inlet splits equally into two identical branches.
    """
    builder = NetworkBuilder()
    network = (builder
        .set_inlet("inlet")
        .add_outlet("out1")
        .add_outlet("out2")
        .add_pipe("inlet", "junction", length=0.5, diameter=0.02, name="main")
        .add_pipe("junction", "out1", length=0.8, diameter=0.015, name="b1")
        .add_pipe("junction", "out2", length=0.8, diameter=0.015, name="b2")
        .build()
    )
    inlet_id = network.get_node("inlet").id
    junction_id = network.get_node("junction").id
    out1_id = network.get_node("out1").id
    out2_id = network.get_node("out2").id

    ch_main = network.get_connection_by_nodes(inlet_id, junction_id).component
    ch1 = network.get_connection_by_nodes(junction_id, out1_id).component
    ch2 = network.get_connection_by_nodes(junction_id, out2_id).component

    sim_config = SimulationConfig(oil_density=DENSITY, oil_type="VG220", temperature=TEMPERATURE, total_flow_rate=Q_TOTAL, inlet_pressure=INLET_P, outlet_pressure=OUTLET_P)
    solver = NodalMatrixSolver(sim_config)
    solver.fluid_properties = {'density': DENSITY, 'viscosity': VISCOSITY}

    info = solver.solve(network)
    flows = info.get("component_flows", {})

    # mass conservation at junction: main = b1 + b2
    Q_main = flows[ch_main.id]
    Qb1 = flows[ch1.id]
    Qb2 = flows[ch2.id]
    assert math.isclose(Q_main, Qb1 + Qb2, rel_tol=1e-6)

    # branches split roughly equally (since identical geometry)
    assert math.isclose(Qb1, Qb2, rel_tol=5e-2)

    # pressures are monotonic: P0 > Pjunction > Poutlets
    P = info["node_pressures"]
    assert P[network.get_node("inlet").id] > P[network.get_node("junction").id] > P[network.get_node("out1").id]
    assert P[network.get_node("junction").id] > P[network.get_node("out2").id]
