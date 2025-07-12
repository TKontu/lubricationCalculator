# tests/test_resistance_linearization.py

import math
import pytest
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig

FLUID = {'density': 1000.0, 'viscosity': 1e-3}
D, L = 0.01, 1.0
ch = Channel(diameter=D, length=L, name="test")
sim_config = SimulationConfig(oil_density=FLUID['density'], oil_type="SAE30", temperature=40, total_flow_rate=0, inlet_pressure=0)
solver = NodalMatrixSolver(sim_config)
solver.calculate_viscosity = lambda T: FLUID['viscosity']

def test_laminar_resistance():
    Q = 1e-6
    R_num = solver._calculate_component_resistance(ch, FLUID, Q)
    R_anal = (128.0 * FLUID['viscosity'] * L) / (math.pi * D**4)
    assert math.isclose(R_num, R_anal, rel_tol=1e-3)

def test_turbulent_resistance():
    Re_target = 1e5
    A = math.pi * (D/2)**2
    V = Re_target * FLUID['viscosity'] / (FLUID['density'] * D)
    Q = V * A
    dp = ch.calculate_pressure_drop(Q, FLUID)
    R_exp = 2 * dp / Q
    R_num = solver._calculate_component_resistance(ch, FLUID, Q)
    assert math.isclose(R_num, R_exp, rel_tol=0.01)
