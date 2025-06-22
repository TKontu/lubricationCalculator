# tests/test_friction.py

import math
import pytest
from lubrication_flow_package.utils.friction import churchill_friction_factor

# We’ll use SciPy’s root‐finder to get reference Colebrook–White values
from scipy.optimize import brentq

def _colebrook(Re: float, eps_over_D: float) -> float:
    """
    Solve the implicit Colebrook–White equation for f:
      1/sqrt(f) + 2.0*log10(eps/D/3.7 + 2.51/(Re*sqrt(f))) = 0
    """
    def cw_eq(f):
        return 1.0/math.sqrt(f) + 2.0*math.log10(eps_over_D/3.7 + 2.51/(Re*math.sqrt(f)))
    # bracket f between a tiny number and 1.0
    return brentq(cw_eq, 1e-6, 1.0)

def test_laminar_limit():
    """
    For Re < 2000, the explicit Churchill formula is within
    ~0.2% of the exact f = 64/Re laminar solution.
    """
    for Re in [50, 500, 1500, 1999]:
        f_ch = churchill_friction_factor(Re, eps_over_D=0.0)
        f_lin = 64.0 / Re
        # allow ~0.2% error
        assert math.isclose(
            f_ch, f_lin, rel_tol=2e-3
        ), f"Laminar mismatch: Re={Re}, f_ch={f_ch}, 64/Re={f_lin}"

@pytest.mark.parametrize("Re, eps_over_D", [
    (1e5, 0.00005),
    (1e6, 0.00005),
    (1e7, 0.00005),
    (5e5, 0.0002),
])
def test_turbulent_vs_colebrook(Re, eps_over_D):
    """
    Compare churchill_friction_factor against a reference
    Colebrook–White solution for a few turbulent cases.
    """
    f_ch = churchill_friction_factor(Re, eps_over_D)
    f_ref = _colebrook(Re, eps_over_D)
    # Allow ~3% error between explicit and implicit formulas
    assert math.isclose(
        f_ch, f_ref, rel_tol=3e-2
    ), f"Turbulent mismatch: Re={Re}, eps/D={eps_over_D}, f_ch={f_ch}, f_ref={f_ref}"
