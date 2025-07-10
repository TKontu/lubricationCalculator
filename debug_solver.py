import numpy as np

# --- Problem Definition ---
# Component: QuadraticResistance(a=1000, b=500000)
# ΔP = 1000Q + 500000Q²
# Target Flow (Q_total): 0.001 m³/s
# Expected Physical ΔP: 1000*(0.001) + 500000*(0.001)² = 1.5 Pa

Q_total = 0.001
a = 1000.0
b = 500000.0

def physical_pressure_drop(q):
    return a * q + b * q**2

def differential_resistance(q):
    # R_diff = d(aQ + bQ²)/dQ = a + 2bQ
    return a + 2 * b * q

# --- Manual Solver ---

# Initialization
q_current = Q_total # Initial guess for flow

print("--- Starting Manual Solver ---")

for i in range(10): # Max 10 iterations
    print(f"\n--- Iteration {i+1} ---")
    
    # 1. Calculate linearized resistance and conductance
    R_diff = differential_resistance(q_current)
    G = 1.0 / R_diff
    print(f"  q_current = {q_current:.6f}")
    print(f"  R_diff    = {R_diff:.2f}")
    print(f"  G         = {G:.6f}")

    # 2. Calculate the non-linear residual pressure and flow
    phys_dp = physical_pressure_drop(q_current)
    lin_dp = R_diff * q_current
    resid_dp = phys_dp - lin_dp
    resid_flow = G * resid_dp
    print(f"  Phys_DP   = {phys_dp:.2f}")
    print(f"  Lin_DP    = {lin_dp:.2f}")
    print(f"  Resid_DP  = {resid_dp:.2f}")
    print(f"  Resid_Flow= {resid_flow:.6f}")

    # 3. Assemble A and b for a 2-node system where node 1 is the sink (P=0)
    # The system is A * p = b, where p is just [p_node0]
    # Equation at node 0: G * (p_node0 - p_node1) = Q_total - Correction
    # Since p_node1 = 0: G * p_node0 = Q_total - Correction
    # Correction = resid_flow (since dp_hydro is 0)
    
    A = np.array([[G]])
    # The b vector must include the total flow AND the non-linear correction
    # The user correctly pointed out the residual flow must be subtracted.
    b_vec = np.array([Q_total - resid_flow])
    print(f"  A matrix  = {A}")
    print(f"  b vector  = {b_vec}")

    # 4. Solve for pressure
    p = np.linalg.solve(A, b_vec)
    p_node0 = p[0]
    print(f"  Solved P0 = {p_node0:.2f}")

    # 5. Calculate new flow based on the solved pressure
    q_new = G * p_node0
    print(f"  q_new     = {q_new:.6f}")

    # 6. Check for convergence
    flow_change = abs(q_new - q_current)
    print(f"  Flow Change = {flow_change:.6f}")
    if flow_change < 1e-6:
        print("\n--- Converged ---")
        break
    
    # 7. Update for next iteration
    q_current = q_new

print(f"\nFinal Result: P_drop = {p_node0:.2f} Pa, Flow = {q_current:.6f} m³/s")
print(f"Expected Result: P_drop = {physical_pressure_drop(Q_total):.2f} Pa, Flow = {Q_total:.6f} m³/s")
