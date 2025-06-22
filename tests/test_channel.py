# tests/test_channel_dp.py

import math
import pytest

from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.utils.friction import churchill_friction_factor

# Common fluid properties for tests
FLUID = {'density': 1000.0, 'viscosity': 1e-3}


def test_laminar_pressure_drop():
    """
    At low Re (<2000), ΔP should match Hagen–Poiseuille:
      ΔP = (128 μ L)/(π D^4) · Q
    """
    D = 0.01      # 10 mm pipe
    L = 1.0       # 1 m length
    Q = 1e-6      # m³/s → Re ≈ 0.013 < 2000
    ch = Channel(diameter=D, length=L, roughness=0.0, name="laminar_test")

    dp_calc = ch.calculate_pressure_drop(Q, FLUID)
    dp_hp = (128 * FLUID['viscosity'] * L) / (math.pi * D**4) * Q

    assert math.isclose(
        dp_calc, dp_hp, rel_tol=1e-6
    ), f"Laminar ΔP mismatch: got {dp_calc}, expected {dp_hp}"


@pytest.mark.parametrize("Re, eps_over_D", [
    (1e5,   0.00005),
    (5e5,   0.0001),
    (1e6,   0.00005),
])
def test_turbulent_pressure_drop(Re, eps_over_D):
    """
    At high Re, ΔP should follow Darcy–Weisbach with Churchill’s f:
      f = churchill_friction_factor(Re, ε/D)
      ΔP = f·(L/D)·(ρ·V²/2)
    """
    D = 0.05      # 50 mm pipe
    L = 2.0       # 2 m length
    eps = eps_over_D * D
    ch = Channel(diameter=D, length=L, roughness=eps, name="turb_test")

    # Compute Q that yields the target Re
    A = math.pi * (D/2)**2
    V = Re * FLUID['viscosity'] / (FLUID['density'] * D)
    Q = V * A

    dp_calc = ch.calculate_pressure_drop(Q, FLUID)

    f = churchill_friction_factor(Re, eps_over_D)
    dp_exp = f * (L / D) * (FLUID['density'] * V * abs(V) / 2)

    assert math.isclose(
        dp_calc, dp_exp, rel_tol=1e-3
    ), (
        f"Turbulent ΔP mismatch (Re={Re:.1e}, ε/D={eps_over_D}): "
        f"got {dp_calc}, expected {dp_exp}"
    )


@pytest.mark.parametrize("D,L,Q,eps_over_D", [
    # Transitional around Re ≈ 2000
    (0.02, 1.0, 2e-5, 0.00005),  
    # Small diameter, moderate flow → Re ≈ 10000
    (0.005, 0.5, 5e-4, 0.0001),
    # Rough pipe, moderate Re
    (0.03, 1.0, 1e-3, 0.005),
])
def test_various_flow_conditions(D, L, Q, eps_over_D):
    """
    Spot-check across transitional and varied geometries.
    """
    fluid = {'density': 900.0, 'viscosity': 1e-3}
    eps = eps_over_D * D
    ch = Channel(diameter=D, length=L, roughness=eps, name="var_test")

    dp_calc = ch.calculate_pressure_drop(Q, fluid)

    # Recompute Re & f
    A = math.pi * (D/2)**2
    V = Q / A
    Re = fluid['density'] * abs(V) * D / fluid['viscosity']
    if Re < 2000:
        f = 64.0 / Re
    else:
        f = churchill_friction_factor(Re, eps_over_D)
    dp_exp = f * (L / D) * (fluid['density'] * V * abs(V) / 2)

    assert math.isclose(
        dp_calc, dp_exp, rel_tol=1e-3
    ), (
        f"ΔP mismatch (D={D}, L={L}, Q={Q}, eps/D={eps_over_D}): "
        f"got {dp_calc}, expected {dp_exp}"
    )


def test_zero_and_negative_flow():
    """
    ΔP(0) must be zero; ΔP(-Q) = -ΔP(Q).
    """
    D = 0.02
    L = 1.0
    Q = 1e-4
    ch = Channel(diameter=D, length=L, roughness=0.0, name="sign_test")

    dp_zero = ch.calculate_pressure_drop(0.0, FLUID)
    dp_pos = ch.calculate_pressure_drop(Q, FLUID)
    dp_neg = ch.calculate_pressure_drop(-Q, FLUID)

    assert dp_zero == 0.0
    assert math.isclose(dp_neg, -dp_pos, rel_tol=1e-9), (
        f"Sign symmetry failed: dp(+Q)={dp_pos}, dp(-Q)={dp_neg}"
    )


@pytest.mark.parametrize("Re_target", [0.1, 2e3, 1e8])
def test_edge_reynolds_limits(Re_target):
    """
    Verify behavior at the limits of the Churchill formula:
      - Re ~ 0.1   (very low)
      - Re ~ 2000  (transition)
      - Re ~ 1e8   (upper bound)
    """
    D = 0.05
    L = 1.0
    fluid = {'density': 1000.0, 'viscosity': 1e-3}
    ch = Channel(diameter=D, length=L, roughness=0.0, name="edge_test")

    # Compute Q for the target Re
    A = math.pi * (D/2)**2
    V = Re_target * fluid['viscosity'] / (fluid['density'] * D)
    Q = V * A

    dp_calc = ch.calculate_pressure_drop(Q, fluid)

    # Expect dp = f·(L/D)·(ρ·V²/2), where f is exact 64/Re for low Re,
    # or churchill_friction_factor otherwise
    if Re_target < 2000:
        f_exp = 64.0 / Re_target
    else:
        f_exp = churchill_friction_factor(Re_target, 0.0)

    dp_exp = f_exp * (L / D) * (fluid['density'] * V * abs(V) / 2)
    assert math.isclose(
        dp_calc, dp_exp, rel_tol=1e-3
    ), (
        f"Edge-Re ΔP mismatch (Re={Re_target}): "
        f"got {dp_calc}, expected {dp_exp}"
    )
