import numpy as np

Q_total = 0.001
a = 1000.0
b = 500000.0

def physical_pressure_drop(q):
    return a * q + b * q**2

def differential_resistance(q):
    return a + 2 * b * q

# Initialization
q_current = Q_total  # Fixed at target flow (flow-controlled system)

print("--- Starting Manual Solver ---")

for i in range(10):  # Max 10 iterations
    print(f"\n--- Iteration {i+1} ---")
    
    # 1. Calculate linearized resistance and conductance
    R_diff = differential_resistance(q_current)
    G = 1.0 / R_diff
    print(f"  q_current = {q_current:.6f} (fixed at Q_total)")
    print(f"  R_diff    = {R_diff:.2f}")
    print(f"  G         = {G:.6f}")

    # 2. Calculate non-linear residuals
    phys_dp = physical_pressure_drop(q_current)
    lin_dp = R_diff * q_current
    resid_dp = phys_dp - lin_dp
    resid_flow = G * resid_dp
    print(f"  Phys_DP   = {phys_dp:.2f}")
    print(f"  Lin_DP    = {lin_dp:.2f}")
    print(f"  Resid_DP  = {resid_dp:.2f}")
    print(f"  Resid_Flow= {resid_flow:.6f}")

    # 3. Assemble system: G * p_node0 = Q_total + resid_flow
    A = np.array([[G]])
    b_vec = np.array([Q_total + resid_flow])  # CORRECTED: Add residual flow
    print(f"  A matrix  = {A}")
    print(f"  b vector  = {b_vec}")

    # 4. Solve for pressure
    p = np.linalg.solve(A, b_vec)
    p_node0 = p[0]
    print(f"  Solved P0 = {p_node0:.2f}")

    # 5. Skip flow update (q_new unused)
    q_new = G * p_node0  # Not used for updating q_current
    print(f"  q_new     = {q_new:.6f} (not used)")

    # 6. Check pressure convergence (since flow is fixed)
    expected_pressure = physical_pressure_drop(Q_total)
    pressure_error = abs(p_node0 - expected_pressure)
    print(f"  Pressure Error = {pressure_error:.6f}")
    if pressure_error < 1e-6:
        print("\n--- Converged ---")
        break

    # 7. q_current remains fixed at Q_total (no update)

print(f"\nFinal Result: P_drop = {p_node0:.2f} Pa, Flow = {Q_total:.6f} m³/s")
print(f"Expected Result: P_drop = {physical_pressure_drop(Q_total):.2f} Pa, Flow = {Q_total:.6f} m³/s")