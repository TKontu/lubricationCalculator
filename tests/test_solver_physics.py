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
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

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
    s = NodalMatrixSolver(oil_density=DENSITY, oil_type="SAE30")
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
    net = FlowNetwork("vertical_pipe")
    n_bottom = net.create_node(name="bottom", elevation=0.0)
    n_top = net.create_node(name="top", elevation=10.0) # 10m height difference
    net.set_inlet(n_bottom)
    net.add_outlet(n_top)

    # Use a component with very high resistance to ensure flow is near zero
    comp = LinearResistance(resistance=1e12, component_id="R1")
    net.connect_components(n_bottom, n_top, comp)

    # The `solve_nodal_iterative` is used here to isolate the core physics calculation
    # from other parts of the `solve_nodal_network` wrapper.
    # We set Q_total to 0 to focus purely on the static pressure.
    pressures, flows = solver.solve_nodal_iterative(
        network=net,
        source_node_id=n_bottom.id,
        sink_node_ids=[n_top.id],
        Q_total=0.0,
        fluid_properties=fluid_properties
    )

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
    net = FlowNetwork("nonlinear_residual_test")
    n_in = net.create_node("in")
    n_out = net.create_node("out")
    net.set_inlet(n_in)
    net.add_outlet(n_out)
    comp = QuadraticResistance(a=1000.0, b=500000.0, component_id="NL_resid_test")
    net.connect_components(n_in, n_out, comp)
    fluid_properties = {'density': DENSITY, 'viscosity': VISCOSITY}

    with caplog.at_level("DEBUG"):
        solver.solve_nodal_iterative(
            network=net,
            source_node_id=n_in.id,
            sink_node_ids=[n_out.id],
            Q_total=Q_TOTAL,
            fluid_properties=fluid_properties,
            tol_pressure=1e-3 # Use a tight tolerance for this test
        )

    # Find the log message from the final iteration
    final_iter_log = None
    for rec in reversed(caplog.records):
        if "NL_resid_test" in rec.message and "Resid_DP" in rec.message:
            final_iter_log = rec.message
            break
    
    assert final_iter_log is not None, "Did not find residual DP log message"

    # Extract the residual DP value
    # Example: "  Conn NL_resid_test: Flow=0.0010, Phys_DP=1.50, Lin_DP=1.50, Resid_DP=0.00"
    parts = {p.split("=")[0].strip(): float(p.split("=")[1]) for p in final_iter_log.split(",")}
    residual_dp = parts["Resid_DP"]

    # Assert that the residual IS close to zero
    assert math.isclose(residual_dp, 0.0, abs_tol=1e-3)


def test_hydrostatic_utube_zero_flow(solver, fluid_properties):
    """
    Tests if a U-shaped pipe with equal inlet/outlet elevations and
    pressures results in zero flow, which would prove the hydrostatic
    contributions to the 'b' vector are correctly balanced.
    """
    net = FlowNetwork("u_tube_test")
    n_in = net.create_node("in", elevation=10.0)
    n_mid = net.create_node("mid", elevation=0.0)
    n_out = net.create_node("out", elevation=10.0)
    net.set_inlet(n_in)
    net.add_outlet(n_out)

    # Two pipes forming the U-shape
    comp1 = LinearResistance(resistance=1000.0, component_id="down_pipe")
    comp2 = LinearResistance(resistance=1000.0, component_id="up_pipe")
    net.connect_components(n_in, n_mid, comp1)
    net.connect_components(n_mid, n_out, comp2)

    # With zero total flow, the internal flows should also be zero
    # as the hydrostatic effects should cancel out.
    pressures, flows = solver.solve_nodal_iterative(
        network=net,
        source_node_id=n_in.id,
        sink_node_ids=[n_out.id],
        Q_total=0.0,
        fluid_properties=fluid_properties
    )

    # The current, incorrect implementation will produce a non-zero flow.
    assert math.isclose(flows["down_pipe"], 0.0, abs_tol=1e-9)
    assert math.isclose(flows["up_pipe"], 0.0, abs_tol=1e-9)


def test_solver_with_correct_nonlinear_logic(solver, fluid_properties):
    """
    This test validates that the solver produces the correct final pressure
    drop for a non-linear component in a simple flow-controlled system.
    It is based on the validated logic from debug_solver2.py.
    """
    net = FlowNetwork("correct_nonlinear_test")
    n_in = net.create_node("in")
    n_out = net.create_node("out")
    net.set_inlet(n_in)
    net.add_outlet(n_out)

    # Component: ΔP = 1000Q + 500000Q²
    a, b = 1000.0, 500000.0
    comp = QuadraticResistance(a=a, b=b, component_id="NL_correct")
    net.connect_components(n_in, n_out, comp)

    # Run the solver
    pressures, flows = solver.solve_nodal_iterative(
        network=net,
        source_node_id=n_in.id,
        sink_node_ids=[n_out.id],
        Q_total=Q_TOTAL,
        fluid_properties=fluid_properties
    )

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
    net = FlowNetwork("nonlinear_test")
    n_in = net.create_node(name="inlet")
    n_out = net.create_node(name="outlet")
    net.set_inlet(n_in)
    net.add_outlet(n_out)

    # ΔP = 1000Q + 500000Q²
    comp = QuadraticResistance(a=1000.0, b=500000.0, component_id="NL1")
    net.connect_components(n_in, n_out, comp)

    pressures, flows = solver.solve_nodal_iterative(
        network=net,
        source_node_id=n_in.id,
        sink_node_ids=[n_out.id],
        Q_total=Q_TOTAL,
        fluid_properties=fluid_properties
    )

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
    net = FlowNetwork("y_network")
    n_in = net.create_node("in")
    n_j = net.create_node("junction")
    n_out = net.create_node("out")
    net.set_inlet(n_in)
    net.add_outlet(n_out) # Both branches converge to the same outlet

    R1, R2, R3 = 1000.0, 2000.0, 3000.0
    comp1 = LinearResistance(R1, "R1")
    comp2 = LinearResistance(R2, "R2")
    comp3 = LinearResistance(R3, "R3")

    net.connect_components(n_in, n_j, comp1)
    net.connect_components(n_j, n_out, comp2)
    net.connect_components(n_j, n_out, comp3)

    pressures, flows = solver.solve_nodal_iterative(
        network=net,
        source_node_id=n_in.id,
        sink_node_ids=[n_out.id],
        Q_total=Q_TOTAL,
        fluid_properties=fluid_properties
    )

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
    net1 = FlowNetwork()
    n1 = net1.create_node("n1")
    net1.add_outlet(n1)
    is_valid, errors = net1.validate_network()
    assert not is_valid
    assert "No inlet node defined" in errors

    # 2. No outlets defined
    net2 = FlowNetwork()
    n2 = net2.create_node("n1")
    net2.set_inlet(n2)
    is_valid, errors = net2.validate_network()
    assert not is_valid
    assert "No outlet nodes defined" in errors

    # 3. Isolated (disconnected) node
    net3 = FlowNetwork()
    n_in = net3.create_node("in")
    n_out = net3.create_node("out")
    net3.create_node("isolated") # This node is not connected
    net3.set_inlet(n_in)
    net3.add_outlet(n_out)
    net3.connect_components(n_in, n_out, LinearResistance(100, "R1"))
    is_valid, errors = net3.validate_network()
    assert not is_valid
    assert "Isolated nodes: ['isolated']" in errors

    # 4. Unreachable outlet
    net4 = FlowNetwork()
    n_in = net4.create_node("in")
    n_out1 = net4.create_node("out1")
    n_out2 = net4.create_node("out2") # This outlet is not connected
    net4.set_inlet(n_in)
    net4.add_outlet(n_out1)
    net4.add_outlet(n_out2)
    net4.connect_components(n_in, n_out1, LinearResistance(100, "R1"))
    is_valid, errors = net4.validate_network()
    assert not is_valid
    assert "Unreachable outlets: ['out2']" in errors
