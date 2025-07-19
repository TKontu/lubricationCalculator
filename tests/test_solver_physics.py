"""
Tests for the physics implementation in the nodal matrix solver.

This suite validates the solver's handling of:
1.  Hydrostatic pressure effects in networks with elevation changes.
2.  Non-linear component resistances.
3.  Mass conservation in complex (series-parallel) networks.
4.  Basic network validation logic.
"""

import pytest
import math
from typing import Dict

from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.base import FlowComponent
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.network.connection import Connection
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder

# --- Constants and Fixtures ---

DENSITY = 900.0  # kg/m³
VISCOSITY = 0.01  # Pa·s
GRAVITY = 9.81   # m/s²
Q_TOTAL = 0.001  # m³/s

@pytest.fixture
def fluid_properties() -> Dict[str, float]:
    """Provides a standard set of fluid properties."""
    return {'density': DENSITY, 'viscosity': VISCOSITY}

@pytest.fixture
def solver() -> NodalMatrixSolver:
    """Provides a solver instance with fixed fluid properties."""
    sim_config = SimulationConfig(total_flow_rate=Q_TOTAL, oil_density=DENSITY, oil_type="SAE30", temperature=20, inlet_pressure=0)
    s = NodalMatrixSolver(sim_config)
    s.calculate_viscosity = lambda temp: VISCOSITY
    return s

class LinearResistance(FlowComponent):
    """A simple component with a fixed, linear resistance for testing."""
    def __init__(self, resistance: float, component_id: str):
        super().__init__(component_id=component_id)
        self.resistance = resistance

    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        return self.resistance * abs(flow_rate)

    def get_flow_area(self) -> float:
        return 0.01

class QuadraticResistance(FlowComponent):
    """A component with non-linear resistance (ΔP = aQ + bQ²) for testing."""
    def __init__(self, a: float, b: float, component_id: str):
        super().__init__(component_id=component_id)
        self.a = a
        self.b = b

    def calculate_pressure_drop(self, flow_rate: float, fluid_properties: Dict) -> float:
        q = abs(flow_rate)
        return self.a * q + self.b * q**2

    def get_flow_area(self) -> float:
        return 0.01

# --- Test Cases ---

def test_hydrostatic_pressure_simple_vertical_pipe(solver, fluid_properties):
    """
    Tests if the solver correctly accounts for hydrostatic pressure in a vertical pipe.
    With zero flow, the pressure difference should exactly equal ρgΔz.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("bottom", elevation=0.0)
        .add_outlet("top", elevation=10.0) # 10m height difference
        .add_pipe("bottom", "top", length=10, diameter=0.1, name="pipe")
        .build()
    )
    net.name = "vertical_pipe"

    # Replace the standard channel with a mock component
    comp = LinearResistance(resistance=1e12, component_id="R1")
    n_bottom = net.get_node("bottom")
    n_top = net.get_node("top")
    # Find the original connection and replace its component
    for conn in net.connections:
        if conn.from_node == n_bottom and conn.to_node == n_top:
            conn.component = comp
            break

    # We set Q_total to 0 to focus purely on the static pressure.
    solver.sim_config.total_flow_rate = 0.0
    solution = solver.solve(net)
    pressures = solution.get("node_pressures", {})
    flows = solution.get("component_flows", {})

    # Analytical hydrostatic pressure difference
    expected_dp_hydro = DENSITY * GRAVITY * (n_top.elevation - n_bottom.elevation)

    # Pressure at the bottom should be higher than the top by ρgΔz
    # The solver sets the sink pressure to 0, so the source pressure is the delta.
    actual_dp = pressures[n_bottom.id] - pressures[n_top.id]

    assert flows[comp.id] == pytest.approx(0.0, abs=1e-9)
    assert actual_dp == pytest.approx(expected_dp_hydro, rel=1e-6)




def test_resistance_calculation_methods(solver, fluid_properties):
    """
    Tests the differential resistance calculation method and compares
    it with analytical values for a non-linear component.
    """
    a, b = 1000.0, 500000.0
    q_test = 0.002  # m³/s
    comp = QuadraticResistance(a=a, b=b, component_id="NL_resistance_test")

    # Calculate average resistance manually: R_avg = ΔP/Q = (aQ + bQ²)/Q = a + bQ
    dp = comp.calculate_pressure_drop(q_test, fluid_properties)
    avg_res_calculated = dp / q_test
    avg_res_analytical = a + b * q_test
    assert avg_res_calculated == pytest.approx(avg_res_analytical)

    # Calculate differential resistance: R_diff = d(ΔP)/dQ
    diff_res_calculated = solver._calculate_component_resistance(comp, fluid_properties, q_test)
    # Analytical differential resistance: d(aQ + bQ²)/dQ = a + 2bQ
    diff_res_analytical = a + 2 * b * q_test
    assert diff_res_calculated == pytest.approx(diff_res_analytical)

    # Both _compute_resistance and _calculate_component_resistance now return differential resistance
    unified_res = solver._compute_resistance(comp, q_test, fluid_properties)
    assert unified_res == pytest.approx(diff_res_calculated)

    # Assert that for a non-linear component, average and differential resistances are different
    assert avg_res_calculated != pytest.approx(diff_res_calculated)




def test_nonlinear_residual_is_zero_after_convergence(solver, caplog):
    """
    Tests that the corrected solver converges with a near-zero residual
    between the physical and linearized pressure drops, indicating convergence
    to the true physical solution.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("in")
        .add_outlet("out")
        .add_pipe("in", "out", length=1, diameter=1, name="pipe")
        .build()
    )
    net.name = "nonlinear_residual_test"
    comp = QuadraticResistance(a=1000.0, b=500000.0, component_id="NL_resid_test")
    n_in = net.get_node("in")
    n_out = net.get_node("out")
    for conn in net.connections:
        if conn.from_node == n_in and conn.to_node == n_out:
            conn.component = comp
            break
    fluid_properties = {'density': DENSITY, 'viscosity': VISCOSITY}

    with caplog.at_level("DEBUG"):
        solver.solve(net)

    # Find the log message from the final iteration
    final_iter_log = None
    for rec in reversed(caplog.records):
        if "NL_resid_test" in rec.message and "Resid_DP" in rec.message:
            final_iter_log = rec.message
            break
    
    assert final_iter_log is not None, "Did not find residual DP log message"

    # Extract the residual DP value
    # Example: "  Conn NL_resid_test: Flow=0.0010, Phys_DP=1.50, Lin_DP=1.50, Resid_DP=0.00"
    parts = {p.split("=")[0].strip(): float(p.split("=")[1]) for p in final_iter_log.split(",") if "=" in p}
    residual_dp = parts["Resid_DP"]

    # Assert that the residual IS close to zero
    assert math.isclose(residual_dp, 0.0, abs_tol=1e-3)


def test_hydrostatic_utube_zero_flow(solver, fluid_properties):
    """
    Tests if a U-shaped pipe with equal inlet/outlet elevations and
    pressures results in zero flow, which would prove the hydrostatic
    contributions to the 'b' vector are correctly balanced.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("in", elevation=10.0)
        .add_node("mid", elevation=0.0)
        .add_outlet("out", elevation=10.0)
        .add_pipe("in", "mid", length=1, diameter=1, name="down_pipe")
        .add_pipe("mid", "out", length=1, diameter=1, name="up_pipe")
        .build()
    )
    net.name = "u_tube_test"

    # Two pipes forming the U-shape
    comp1 = LinearResistance(resistance=1000.0, component_id="down_pipe")
    comp2 = LinearResistance(resistance=1000.0, component_id="up_pipe")
    n_in = net.get_node("in")
    n_mid = net.get_node("mid")
    n_out = net.get_node("out")
    for conn in net.connections:
        if conn.from_node == n_in and conn.to_node == n_mid:
            conn.component = comp1
        if conn.from_node == n_mid and conn.to_node == n_out:
            conn.component = comp2

    # With zero total flow, the internal flows should also be zero
    # as the hydrostatic effects should cancel out.
    solver.sim_config.total_flow_rate = 0.0
    solution = solver.solve(net)
    flows = solution.get("component_flows", {})

    # The current, incorrect implementation will produce a non-zero flow.
    assert math.isclose(flows["down_pipe"], 0.0, abs_tol=1e-9)
    assert math.isclose(flows["up_pipe"], 0.0, abs_tol=1e-9)


def test_solver_with_correct_nonlinear_logic(solver, fluid_properties):
    """
    This test validates that the solver produces the correct final pressure
    drop for a non-linear component in a simple flow-controlled system.
    It is based on the validated logic from debug_solver2.py.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("in")
        .add_outlet("out")
        .add_pipe("in", "out", length=1, diameter=1, name="pipe")
        .build()
    )
    net.name = "correct_nonlinear_test"

    # Component: ΔP = 1000Q + 500000Q²
    a, b = 1000.0, 500000.0
    comp = QuadraticResistance(a=a, b=b, component_id="NL_correct")
    n_in = net.get_node("in")
    n_out = net.get_node("out")
    for conn in net.connections:
        if conn.from_node == n_in and conn.to_node == n_out:
            conn.component = comp
            break

    # Run the solver
    solution = solver.solve(net)
    pressures = solution.get("node_pressures", {})

    # The final pressure drop from the solver must match the true physical
    # pressure drop for the given total flow rate.
    actual_dp = pressures[n_in.id] - pressures[n_out.id]
    expected_dp = physical_pressure_drop(Q_TOTAL, a, b)

    # This test will fail with the current solver, which incorrectly
    # converges to a pressure drop of ~2.0 Pa instead of 1.5 Pa.
    assert actual_dp == pytest.approx(expected_dp, rel=1e-6)

def physical_pressure_drop(q, a, b):
    """Helper to calculate the true physical pressure drop."""
    return a * q + b * q**2

def test_nonlinear_resistance_component(solver, fluid_properties):
    """
    Tests if the solver converges to the correct pressure drop for a
    component with a non-linear (quadratic) resistance.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("inlet")
        .add_outlet("outlet")
        .add_pipe("inlet", "outlet", length=1, diameter=1, name="pipe")
        .build()
    )
    net.name = "nonlinear_test"

    # ΔP = 1000Q + 500000Q²
    comp = QuadraticResistance(a=1000.0, b=500000.0, component_id="NL1")
    n_in = net.get_node("inlet")
    n_out = net.get_node("outlet")
    for conn in net.connections:
        if conn.from_node == n_in and conn.to_node == n_out:
            conn.component = comp
            break

    solution = solver.solve(net)
    pressures = solution.get("node_pressures", {})
    flows = solution.get("component_flows", {})

    # Analytical pressure drop for the given flow
    expected_dp = comp.calculate_pressure_drop(Q_TOTAL, fluid_properties)
    actual_dp = pressures[n_in.id] - pressures[n_out.id]

    assert flows[comp.id] == pytest.approx(Q_TOTAL, rel=1e-6)
    assert actual_dp == pytest.approx(expected_dp, rel=1e-6)

def test_series_parallel_network_analytical_solution(solver, fluid_properties):
    """
    Tests a Y-network (one series element followed by two parallel branches)
    and compares the result to the analytical solution.
    """
    builder = NetworkBuilder()
    net = (builder
        .set_inlet("in")
        .add_node("junction")
        .add_outlet("out") # Both branches converge to the same outlet
        .add_pipe("in", "junction", length=1, diameter=1, name="pipe1")
        .add_pipe("junction", "out", length=1, diameter=1, name="pipe2")
        .add_pipe("junction", "out", length=1, diameter=1, name="pipe3")
        .build()
    )
    net.name = "y_network"

    R1, R2, R3 = 1000.0, 2000.0, 3000.0
    comp1 = LinearResistance(R1, "R1")
    comp2 = LinearResistance(R2, "R2")
    comp3 = LinearResistance(R3, "R3")

    n_in = net.get_node("in")
    n_j = net.get_node("junction")
    n_out = net.get_node("out")

    # Replace components
    conns = list(net.connections) # Make a copy to modify
    net.connections.clear()
    net.connections.append(Connection(n_in, n_j, comp1))
    net.connections.append(Connection(n_j, n_out, comp2))
    net.connections.append(Connection(n_j, n_out, comp3))


    solution = solver.solve(net)
    pressures = solution.get("node_pressures", {})
    flows = solution.get("component_flows", {})

    # --- Analytical Solution ---
    # Equivalent resistance of parallel branches
    r_parallel = 1.0 / (1.0/R2 + 1.0/R3)
    # Total equivalent resistance
    r_total = R1 + r_parallel
    # Total pressure drop
    dp_total = r_total * Q_TOTAL
    # Pressure at the junction
    p_junction = dp_total - (R1 * Q_TOTAL)
    # Flow through parallel branches
    q2 = p_junction / R2
    q3 = p_junction / R3

    # --- Assertions ---
    # Total pressure drop
    actual_dp = pressures[n_in.id] - pressures[n_out.id]
    assert actual_dp == pytest.approx(dp_total, rel=1e-6)

    # Flow conservation at junction
    assert flows[comp1.id] == pytest.approx(Q_TOTAL, rel=1e-6)
    assert (flows[comp2.id] + flows[comp3.id]) == pytest.approx(Q_TOTAL, rel=1e-6)

    # Flow distribution in parallel branches
    assert flows[comp2.id] == pytest.approx(q2, rel=1e-6)
    assert flows[comp3.id] == pytest.approx(q3, rel=1e-6)


def test_network_validation_logic():
    """
    Tests the FlowNetwork.validate_network() method for common errors.
    """
    # 1. No inlet defined
    builder1 = NetworkBuilder()
    builder1.add_node("n1").add_node("n2").add_pipe("n1", "n2", length=1, diameter=1, name="pipe").add_outlet("n2")
    is_valid, errors = builder1._network.validate_network()
    assert not is_valid
    assert "No inlet node defined" in errors

    # 2. No outlets defined
    builder2 = NetworkBuilder()
    builder2.add_node("n1").add_node("n2").add_pipe("n1", "n2", length=1, diameter=1, name="pipe").set_inlet("n1")
    is_valid, errors = builder2._network.validate_network()
    assert not is_valid
    assert "No outlet nodes defined" in errors

    # 3. Isolated (disconnected) node
    builder3 = NetworkBuilder()
    builder3.set_inlet("in").add_outlet("out").add_node("isolated").add_pipe("in", "out", length=1, diameter=1, name="R1")
    is_valid, errors = builder3._network.validate_network()
    assert not is_valid
    assert "Isolated nodes: ['isolated']" in errors

    # 4. Unreachable outlet
    builder4 = NetworkBuilder()
    builder4.set_inlet("in").add_outlet("out1").add_outlet("out2").add_pipe("in", "out1", length=1, diameter=1, name="R1")
    is_valid, errors = builder4._network.validate_network()
    assert not is_valid
    assert "Unreachable outlets: ['out2']" in errors