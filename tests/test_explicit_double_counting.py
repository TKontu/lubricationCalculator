#!/usr/bin/env python3
"""
Explicit test to demonstrate the double-counting bug in path-based flow initialization
"""

import pytest
import numpy as np
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig
from lubrication_flow_package.utils.network_builder import NetworkBuilder

class TestExplicitDoubleCounting:
    
    def test_explicit_double_counting(self):
        """
        Test the exact problematic code pattern:
        for path in all_paths:
            path_flow = calculate_path_flow(path)
            for comp_id in path:
                edge_flows[comp_id] += path_flow  # BUG: double-counting!
        """
        
        # Create simple Y-network using the builder
        builder = NetworkBuilder()
        net = (builder
            .set_inlet("source")
            .add_pipe("source", "junction", length=1.0, diameter=0.015, name="shared")
            .add_pipe("junction", "sink1", length=1.0, diameter=0.015, name="branch1")
            .add_pipe("junction", "sink2", length=1.0, diameter=0.015, name="branch2")
            .add_outlet("sink1")
            .add_outlet("sink2")
            .build()
        )
        net.name = "double_count_explicit"

        source = net.get_node("source")
        sink1 = net.get_node("sink1")
        sink2 = net.get_node("sink2")
        
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
        shared_comp = net.get_component_by_name("shared")
        branch1_comp = net.get_component_by_name("branch1")
        branch2_comp = net.get_component_by_name("branch2")
        edge_flows_buggy = {
            shared_comp.id: 0.0,
            branch1_comp.id: 0.0,
            branch2_comp.id: 0.0
        }
        
        for i, path in enumerate(all_paths):
            path_flow = path_flows[i]
            print(f"Processing path {i+1} with flow {path_flow:.6f}")
            
            for comp_id in path:
                edge_flows_buggy[comp_id] += path_flow  # THE BUG!
                comp_name = net.connections[0].component.id # A bit of a hack to get the name back for printing
                for conn in net.connections:
                    if conn.component.id == comp_id:
                        comp_name = conn.component.name
                        break
                print(f"  Adding {path_flow:.6f} to {comp_name} ({comp_id[:8]}...): new total = {edge_flows_buggy[comp_id]:.6f}")
        
        print("\n=== Results of Buggy Algorithm ===")
        total_flow_assigned = sum(edge_flows_buggy.values())
        for comp_id, flow in edge_flows_buggy.items():
            comp_name = "unknown"
            for conn in net.connections:
                if conn.component.id == comp_id:
                    comp_name = conn.component.name
                    break
            print(f"{comp_name}: {flow:.6f} m³/s")
        print(f"Total flow assigned: {total_flow_assigned:.6f} m³/s (should be {Q_total:.6f})")
        print(f"Double-counting factor: {total_flow_assigned / Q_total:.2f}")
        
        # Expected flows for physically correct solution
        print("\n=== Expected Physically Correct Flows ===")
        expected_flows = {
            shared_comp.id: Q_total,  # All flow passes through shared pipe
            branch1_comp.id: Q_total / 2,  # Half to each branch (equal resistance)
            branch2_comp.id: Q_total / 2
        }
        
        for comp_id, flow in expected_flows.items():
            comp_name = "unknown"
            for conn in net.connections:
                if conn.component.id == comp_id:
                    comp_name = conn.component.name
                    break
            print(f"{comp_name}: {flow:.6f} m³/s")
        
        # NOTE: The physical total flow in the network is Q_total, but when we sum
        # all edge flows, we count the shared component plus the branches = 2*Q_total
        # This is NOT double-counting - it's just how flow accounting works in networks
        expected_total_accounting = expected_flows[shared_comp.id] + expected_flows[branch1_comp.id] + expected_flows[branch2_comp.id]
        print(f"Total expected (accounting): {expected_total_accounting:.6f} m³/s")
        print(f"Actual network flow: {Q_total:.6f} m³/s")
        
        # Compare buggy vs correct
        print("\n=== Error Analysis ===")
        for comp_id in [shared_comp.id, branch1_comp.id, branch2_comp.id]:
            error = edge_flows_buggy[comp_id] - expected_flows[comp_id]
            error_pct = (error / expected_flows[comp_id]) * 100 if expected_flows[comp_id] > 0 else 0
            comp_name = "unknown"
            for conn in net.connections:
                if conn.component.id == comp_id:
                    comp_name = conn.component.name
                    break
            print(f"{comp_name}: error = {error:.6f} ({error_pct:+.1f}%)")
        
        # Check if the buggy algorithm produces the same result as expected
        # If so, then either the algorithm isn't buggy, or the bug happens to cancel out
        all_correct = True
        for comp_id in [shared_comp.id, branch1_comp.id, branch2_comp.id]:
            error = abs(edge_flows_buggy[comp_id] - expected_flows[comp_id])
            if error > 1e-6:
                all_correct = False
                break
        
        if all_correct:
            print("⚠️  The 'buggy' algorithm produces correct results for this simple case!")
            print("   This suggests either:")
            print("   1. The algorithm isn't actually buggy, or")
            print("   2. The bug cancels out for equal-resistance symmetric networks")
        
        if not all_correct:
            print("\n✓ CONFIRMED: Double-counting bug detected!")
            print(f"  Shared component has {edge_flows_buggy[shared_comp.id]:.6f} instead of {expected_flows[shared_comp.id]:.6f}")
        else:
            print("\n⚠️  Algorithm appears correct for this test case")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])