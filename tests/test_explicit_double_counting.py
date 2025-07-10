#!/usr/bin/env python3
"""
Explicit test to demonstrate the double-counting bug in path-based flow initialization
"""

import pytest
import numpy as np
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver

class TestExplicitDoubleCounting:
    
    def test_explicit_double_counting(self):
        """
        Test the exact problematic code pattern:
        for path in all_paths:
            path_flow = calculate_path_flow(path)
            for comp_id in path:
                edge_flows[comp_id] += path_flow  # BUG: double-counting!
        """
        
        # Create simple Y-network
        net = FlowNetwork("double_count_explicit")
        
        source = net.create_node("source")
        junction = net.create_node("junction")
        sink1 = net.create_node("sink1")
        sink2 = net.create_node("sink2")
        
        net.set_inlet(source)
        net.add_outlet(sink1)
        net.add_outlet(sink2)
        
        # Equal resistance components for predictable behavior
        shared = Channel(diameter=0.015, length=1.0, component_id="shared")
        branch1 = Channel(diameter=0.015, length=1.0, component_id="branch1")
        branch2 = Channel(diameter=0.015, length=1.0, component_id="branch2")
        
        net.connect_components(source, junction, shared)
        net.connect_components(junction, sink1, branch1)
        net.connect_components(junction, sink2, branch2)
        
        Q_total = 0.001  # 1 L/s
        
        # Manually implement the problematic algorithm to show the bug
        print("=== Manual Implementation of Problematic Algorithm ===")
        
        # Step 1: Find all paths (this part works correctly)
        all_paths = []
        sink_ids = [sink1.id, sink2.id]
        
        for sink_id in sink_ids:
            queue = [(source.id, [])]
            visited = set()
            
            while queue:
                curr_node_id, path = queue.pop(0)
                
                if curr_node_id == sink_id:
                    all_paths.append(path)
                    print(f"Found path to {sink_id}: {' → '.join(path)}")
                    continue
                    
                if curr_node_id in visited:
                    continue
                visited.add(curr_node_id)
                
                for conn in net.connections:
                    if conn.from_node.id == curr_node_id and conn.to_node.id not in visited:
                        new_path = path + [conn.component.id]
                        queue.append((conn.to_node.id, new_path))
        
        # Step 2: Calculate path resistances (simplified - assume equal)
        print("\n=== Path Analysis ===")
        path_flows = []
        for i, path in enumerate(all_paths):
            # For equal resistances, each path gets equal flow
            path_flow = Q_total / len(all_paths)
            path_flows.append(path_flow)
            print(f"Path {i+1} ({' → '.join(path)}): {path_flow:.6f} m³/s")
        
        # Step 3: Apply the problematic algorithm
        print("\n=== Applying Problematic Algorithm ===")
        edge_flows_buggy = {"shared": 0.0, "branch1": 0.0, "branch2": 0.0}
        
        for i, path in enumerate(all_paths):
            path_flow = path_flows[i]
            print(f"Processing path {i+1} with flow {path_flow:.6f}")
            
            for comp_id in path:
                edge_flows_buggy[comp_id] += path_flow  # THE BUG!
                print(f"  Adding {path_flow:.6f} to {comp_id}: new total = {edge_flows_buggy[comp_id]:.6f}")
        
        print("\n=== Results of Buggy Algorithm ===")
        total_flow_assigned = sum(edge_flows_buggy.values())
        for comp, flow in edge_flows_buggy.items():
            print(f"{comp}: {flow:.6f} m³/s")
        print(f"Total flow assigned: {total_flow_assigned:.6f} m³/s (should be {Q_total:.6f})")
        print(f"Double-counting factor: {total_flow_assigned / Q_total:.2f}")
        
        # Expected flows for physically correct solution
        print("\n=== Expected Physically Correct Flows ===")
        expected_flows = {
            "shared": Q_total,  # All flow passes through shared pipe
            "branch1": Q_total / 2,  # Half to each branch (equal resistance)
            "branch2": Q_total / 2
        }
        
        for comp, flow in expected_flows.items():
            print(f"{comp}: {flow:.6f} m³/s")
        
        # NOTE: The physical total flow in the network is Q_total, but when we sum
        # all edge flows, we count the shared component plus the branches = 2*Q_total
        # This is NOT double-counting - it's just how flow accounting works in networks
        expected_total_accounting = expected_flows["shared"] + expected_flows["branch1"] + expected_flows["branch2"]
        print(f"Total expected (accounting): {expected_total_accounting:.6f} m³/s")
        print(f"Actual network flow: {Q_total:.6f} m³/s")
        
        # Compare buggy vs correct
        print("\n=== Error Analysis ===")
        for comp in ["shared", "branch1", "branch2"]:
            error = edge_flows_buggy[comp] - expected_flows[comp]
            error_pct = (error / expected_flows[comp]) * 100 if expected_flows[comp] > 0 else 0
            print(f"{comp}: error = {error:.6f} ({error_pct:+.1f}%)")
        
        # Check if the buggy algorithm produces the same result as expected
        # If so, then either the algorithm isn't buggy, or the bug happens to cancel out
        all_correct = True
        for comp in ["shared", "branch1", "branch2"]:
            error = abs(edge_flows_buggy[comp] - expected_flows[comp])
            if error > 1e-6:
                all_correct = False
                break
        
        if all_correct:
            print("⚠️  The 'buggy' algorithm produces correct results for this simple case!")
            print("   This suggests either:")
            print("   1. The algorithm isn't actually buggy, or")
            print("   2. The bug cancels out for equal-resistance symmetric networks")
        
        if not all_correct:
            print(f"\n✓ CONFIRMED: Double-counting bug detected!")
            print(f"  Shared component has {edge_flows_buggy['shared']:.6f} instead of {expected_flows['shared']:.6f}")
        else:
            print(f"\n⚠️  Algorithm appears correct for this test case")
        
        # Now test the actual solver
        print("\n" + "="*60)
        print("=== Testing Actual Solver Implementation ===")
        
        solver = NodalMatrixSolver()
        actual_flows = solver._initialize_flows(net, source.id, [sink1.id, sink2.id], Q_total)
        
        print("Actual solver results:")
        for comp, flow in actual_flows.items():
            print(f"{comp}: {flow:.6f} m³/s")
        
        actual_total = sum(actual_flows.values())
        print(f"Actual total assigned: {actual_total:.6f} m³/s")
        
        # Check if actual solver has the bug
        shared_actual = actual_flows["shared"]
        shared_expected = expected_flows["shared"]
        
        if abs(shared_actual - shared_expected) > 1e-4:
            print(f"⚠️  ACTUAL SOLVER HAS BUG: shared={shared_actual:.6f}, expected={shared_expected:.6f}")
            assert False, f"Solver has double-counting bug: shared={shared_actual:.6f}, expected={shared_expected:.6f}"
        else:
            print(f"✅ Actual solver appears correct: shared={shared_actual:.6f}")
            # This is a passing assertion
            assert abs(shared_actual - shared_expected) < 1e-4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])