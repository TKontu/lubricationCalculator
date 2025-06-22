import math

def churchill_friction_factor(
    Re: float,
    eps_over_D: float
) -> float:
    """
    Compute the full‐range Darcy–Weisbach friction factor f using
    Churchill’s explicit formula [Churchill, 1977]. Valid for
    0.1 ≲ Re ≲ 1e8.  If Re < 2000, you may optionally use f=64/Re.

    Parameters
    ----------
    Re : float
        Reynolds number (must be > 0).
    eps_over_D : float
        Relative roughness (ε / D).

    Returns
    -------
    f : float
        Darcy–Weisbach friction factor.
    """
    if Re <= 0:
        raise ValueError(f"Reynolds number must be positive; got Re={Re}")

    # Churchill’s formula
    A = (2.457 * math.log(1/((7/Re)**0.9 + 0.27 * eps_over_D)))**16
    B = (37530.0/Re)**16
    f = 8.0 * ((8.0/Re)**12 + 1.0/(A + B)**1.5)**(1.0/12.0)
    return f
