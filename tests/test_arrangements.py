import pytest

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

@pytest.fixture
def solver():
    return NodalMatrixSolver()

# -----------------------------------------------------------------------------
# Case 1: Single 12 mm×1 m pipe → 2 mm sharp-edged nozzle @ 10 L/min
def test_single_pipe_nozzle_flow_driven(solver):
    Q = 10e-3 / 60.0      # 10 L/min → m³/s
    T = 40.0              # °C
    p_out = 101_325       # Pa

    # Build network
    net = FlowNetwork("Single-branch")
    n_in, n_mid, n_out = Node("Inlet"), Node("Mid"), Node("Outlet")
    for n in (n_in, n_mid, n_out):
        net.add_node(n)
    net.set_inlet(n_in)
    net.add_outlet(n_out)

    # Components
    pipe   = Channel(diameter=0.012, length=1.0, component_id="pipe")
    nozzle = Nozzle(
        diameter=0.002,
        nozzle_type=NozzleType.SHARP_EDGED,
        component_id="nozzle"
    )

    # Connections
    net.connect_components(n_in,  n_mid, pipe)
    net.connect_components(n_mid, n_out, nozzle)

    # Solve fixing the flow
    flows, info = solver.solve_nodal_network_with_pump_physics(
        network=net,
        pump_flow_rate=Q,
        temperature=T,
        pump_max_pressure=5e6,   # sufficiently high bound
        outlet_pressure=p_out
    )

    # 1) Check mass conservation
    assert pytest.approx(info["total_flow_rate"], rel=1e-4) == Q
    assert pytest.approx(flows["pipe"], rel=1e-4)   == Q
    assert pytest.approx(flows["nozzle"], rel=1e-4) == Q

    # 2) Reconstruct expected inlet pressure: ΔP_pipe + ΔP_nozzle + p_out
    dp_pipe   = pipe.calculate_pressure_drop(Q, info["fluid_properties"])
    dp_nozzle = nozzle.calculate_pressure_drop(Q, info["fluid_properties"])
    expected_pin = dp_pipe + dp_nozzle + p_out

    assert pytest.approx(info["inlet_pressure"], rel=0.05) == expected_pin


# -----------------------------------------------------------------------------
# Case 2: 24 mm inlet → T-junction → two 12 mm×1 m branches ending in 2 mm & 3 mm nozzles @ 32 L/min
def test_t_split_flow_driven(solver):
    Q_tot = 32e-3 / 60.0   # 32 L/min → m³/s
    T = 40.0               # °C
    p_out = 101_325        # Pa

    # Build network
    net = FlowNetwork("Parallel-branches")
    n_in, n_j = Node("Inlet"), Node("Junction")
    n_b1, n_out1 = Node("Branch1"), Node("Outlet1")
    n_b2, n_out2 = Node("Branch2"), Node("Outlet2")
    for n in (n_in, n_j, n_b1, n_b2, n_out1, n_out2):
        net.add_node(n)
    net.set_inlet(n_in)
    net.add_outlet(n_out1)
    net.add_outlet(n_out2)

    # Components
    inlet_pipe = Channel(diameter=0.024, length=1.0, component_id="inlet_pipe")
    pipe1      = Channel(diameter=0.012, length=1.0, component_id="pipe1")
    pipe2      = Channel(diameter=0.012, length=1.0, component_id="pipe2")
    nozzle1    = Nozzle(0.002, NozzleType.SHARP_EDGED, component_id="nozzle1")
    nozzle2    = Nozzle(0.003, NozzleType.SHARP_EDGED, component_id="nozzle2")

    # Connections
    net.connect_components(n_in,   n_j,    inlet_pipe)
    net.connect_components(n_j,    n_b1,   pipe1)
    net.connect_components(n_b1,   n_out1, nozzle1)
    net.connect_components(n_j,    n_b2,   pipe2)
    net.connect_components(n_b2,   n_out2, nozzle2)

    # Solve fixing the total flow
    flows, info = solver.solve_nodal_network_with_pump_physics(
        network=net,
        pump_flow_rate=Q_tot,
        temperature=T,
        pump_max_pressure=5e6,
        outlet_pressure=p_out
    )

    # 1) Total flow delivered
    assert pytest.approx(info["total_flow_rate"], rel=1e-4) == Q_tot

    # 2) Mass conservation in each leg
    assert pytest.approx(flows["pipe1"], rel=1e-4)  == flows["nozzle1"]
    assert pytest.approx(flows["pipe2"], rel=1e-4)  == flows["nozzle2"]

    # 3) Approximate split in L/min (2 mm vs 3 mm)
    Q1_Lpm = flows["pipe1"] * 60e3
    Q2_Lpm = flows["pipe2"] * 60e3
    assert pytest.approx(Q1_Lpm, rel=0.1) ==  9.9
    assert pytest.approx(Q2_Lpm, rel=0.1) == 22.1

    # 4) Reconstruct expected inlet pressure via one branch + inlet pipe
    dp_inlet = inlet_pipe.calculate_pressure_drop(flows["inlet_pipe"], info["fluid_properties"])
    dp_branch1 = (
        pipe1.calculate_pressure_drop(flows["pipe1"], info["fluid_properties"]) +
        nozzle1.calculate_pressure_drop(flows["nozzle1"], info["fluid_properties"])
    )
    expected_pin = dp_inlet + dp_branch1 + p_out

    assert pytest.approx(info["inlet_pressure"], rel=0.05) == expected_pin
