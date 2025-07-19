import pytest

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.components.nozzle import Nozzle, NozzleType
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder


@pytest.fixture
def solver():
    sim_config = SimulationConfig(total_flow_rate=0.1, oil_density=850, oil_type="SAE30", temperature=40, inlet_pressure=101325)
    return NodalMatrixSolver(sim_config)

# -----------------------------------------------------------------------------
# Case 1: Single 12 mm×1 m pipe → 2 mm sharp-edged nozzle @ 10 L/min
def test_single_pipe_nozzle_flow_driven(solver):
    Q = 10e-3 / 60.0      # 10 L/min → m³/s
    T = 40.0              # °C
    p_out = 101_325       # Pa

    # Build network using the builder
    builder = NetworkBuilder(solver.sim_config)
    network = (builder
        .set_inlet("Inlet")
        .add_pipe("Inlet", "Mid", length=1.0, diameter=0.012, name="pipe")
        .add_nozzle("Mid", "Outlet", diameter=0.002, nozzle_type=NozzleType.SHARP_EDGED, name="nozzle")
        .add_outlet("Outlet", pressure=p_out)
        .build()
    )
    network.name = "Single-branch"

    # Solve fixing the flow
    solver.sim_config.total_flow_rate = Q
    solver.sim_config.temperature = T
    solver.sim_config.inlet_pressure = 5e6
    solver.sim_config.outlet_pressure = p_out
    info = solver.solve(network)
    flows = info.get("component_flows", {})

    # Get components for pressure drop calculation
    pipe = network.get_component_by_name("pipe")
    nozzle = network.get_component_by_name("nozzle")

    # 1) Check mass conservation
    assert pytest.approx(info["total_flow_rate"], rel=1e-4) == Q
    assert pytest.approx(flows[pipe.id], rel=1e-4)   == Q
    assert pytest.approx(flows[nozzle.id], rel=1e-4) == Q

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

    # Build network using the builder
    builder = NetworkBuilder(solver.sim_config)
    network = (builder
        .set_inlet("Inlet")
        .add_pipe("Inlet", "Junction", length=1.0, diameter=0.024, name="inlet_pipe")
        .add_pipe("Junction", "Branch1", length=1.0, diameter=0.012, name="pipe1")
        .add_nozzle("Branch1", "Outlet1", diameter=0.002, nozzle_type=NozzleType.SHARP_EDGED, name="nozzle1")
        .add_pipe("Junction", "Branch2", length=1.0, diameter=0.012, name="pipe2")
        .add_nozzle("Branch2", "Outlet2", diameter=0.003, nozzle_type=NozzleType.SHARP_EDGED, name="nozzle2")
        .add_outlet("Outlet1", pressure=p_out)
        .add_outlet("Outlet2", pressure=p_out)
        .build()
    )
    network.name = "Parallel-branches"

    # Solve fixing the total flow
    solver.sim_config.total_flow_rate = Q_tot
    solver.sim_config.temperature = T
    solver.sim_config.inlet_pressure = 5e6
    solver.sim_config.outlet_pressure = p_out
    info = solver.solve(network)
    flows = info.get("component_flows", {})

    # Get components for checks
    inlet_pipe = network.get_component_by_name("inlet_pipe")
    pipe1 = network.get_component_by_name("pipe1")
    nozzle1 = network.get_component_by_name("nozzle1")
    pipe2 = network.get_component_by_name("pipe2")
    nozzle2 = network.get_component_by_name("nozzle2")

    # 1) Total flow delivered
    assert pytest.approx(info["total_flow_rate"], rel=1e-4) == Q_tot

    # 2) Mass conservation in each leg
    assert pytest.approx(flows[pipe1.id], rel=1e-4)  == flows[nozzle1.id]
    assert pytest.approx(flows[pipe2.id], rel=1e-4)  == flows[nozzle2.id]

    # 3) Approximate split in L/min (2 mm vs 3 mm)
    Q1_Lpm = flows[pipe1.id] * 60e3
    Q2_Lpm = flows[pipe2.id] * 60e3
    assert pytest.approx(Q1_Lpm, rel=0.1) ==  9.9
    assert pytest.approx(Q2_Lpm, rel=0.1) == 22.1

    # 4) Reconstruct expected inlet pressure via one branch + inlet pipe
    dp_inlet = inlet_pipe.calculate_pressure_drop(flows[inlet_pipe.id], info["fluid_properties"])
    dp_branch1 = (
        pipe1.calculate_pressure_drop(flows[pipe1.id], info["fluid_properties"]) +
        nozzle1.calculate_pressure_drop(flows[nozzle1.id], info["fluid_properties"])
    )
    expected_pin = dp_inlet + dp_branch1 + p_out

    assert pytest.approx(info["inlet_pressure"], rel=0.05) == expected_pin
