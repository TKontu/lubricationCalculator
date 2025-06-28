import pytest
import math

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

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
    net = FlowNetwork("single_pipe") 
    n0 = net.create_node(name="inlet", elevation=0.0)
    n1 = net.create_node(name="outlet", elevation=0.0)
    net.set_inlet(n0)
    net.add_outlet(n1)

    # geometry chosen so that R = 128μL/(πD⁴) is nice
    dia = 0.02   # m
    length = 1.0 # m
    ch = Channel(diameter=dia, length=length, name="ch0")
    net.connect_components(n0, n1, ch)

    return net, ch

def test_two_node_case(simple_pipe_network):
    net, ch = simple_pipe_network
    solver = NodalMatrixSolver(oil_density=DENSITY, oil_type="VG220")
    # monkey‐patch viscosity to fixed value
    solver.calculate_viscosity = lambda T: VISCOSITY

    # call unified interface
    flows, info = solver.solve_nodal_network(
        network=net,
        total_flow_rate=Q_TOTAL,
        temperature=TEMPERATURE,
        inlet_pressure=INLET_P,
        outlet_pressure=OUTLET_P
    )

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
    net = FlowNetwork("t_junction")
    n0 = net.create_node(name="inlet", elevation=0.0)
    n1 = net.create_node(name="junction", elevation=0.0)
    n2 = net.create_node(name="out1", elevation=0.0)
    n3 = net.create_node(name="out2", elevation=0.0)
    net.set_inlet(n0)
    net.add_outlet(n2); net.add_outlet(n3)

    # identical channels
    ch_main = Channel(diameter=0.02, length=0.5, name="main")
    ch1 = Channel(diameter=0.015, length=0.8, name="b1")
    ch2 = Channel(diameter=0.015, length=0.8, name="b2")

    net.connect_components(n0, n1, ch_main)
    net.connect_components(n1, n2, ch1)
    net.connect_components(n1, n3, ch2)

    solver = NodalMatrixSolver(oil_density=DENSITY, oil_type="VG220")
    solver.calculate_viscosity = lambda T: VISCOSITY

    flows, info = solver.solve_nodal_network(
        network=net,
        total_flow_rate=Q_TOTAL,
        temperature=TEMPERATURE,
        inlet_pressure=INLET_P,
        outlet_pressure=OUTLET_P
    )

    # mass conservation at junction: main = b1 + b2
    Q_main = flows[ch_main.id]
    Qb1 = flows[ch1.id]
    Qb2 = flows[ch2.id]
    assert math.isclose(Q_main, Qb1 + Qb2, rel_tol=1e-6)

    # branches split roughly equally (since identical geometry)
    assert math.isclose(Qb1, Qb2, rel_tol=5e-2)

    # pressures are monotonic: P0 > Pjunction > Poutlets
    P = info["node_pressures"]
    assert P[n0.id] > P[n1.id] > P[n2.id]
    assert P[n1.id] > P[n3.id]
