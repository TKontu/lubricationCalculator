#!/usr/bin/env python3
"""
Test script to assess if flow initialization correctly conserves mass
and handles shared upstream components without double-counting.
"""

import pytest
import numpy as np
from lubrication_flow_package.network.flow_network import FlowNetwork
from lubrication_flow_package.components.channel import Channel
from lubrication_flow_package.network.node import Node
from lubrication_flow_package.solvers.nodal_matrix_solver import NodalMatrixSolver
from lubrication_flow_package.config.simulation_config import SimulationConfig

class TestFlowInitialization:
    
    def setup_method(self):
        """Set up solver for each test"""
        sim_config = SimulationConfig(total_flow_rate=0.001, oil_density=900.0, oil_type="SAE30", temperature=40, inlet_pressure=101325)
        self.solver = NodalMatrixSolver(sim_config)
        self.solver.calculate_viscosity = lambda T: 1e-3  # Fixed viscosity
        self.Q_total = 0.001  # 1 L/s total flow
    
    def test_simple_parallel_branches(self):
        """
        Test Case 1: Simple parallel branches (no shared upstream)
        
        Source → Junction → Branch1 → Sink1
                       └→ Branch2 → Sink2
        
        Expected: Each branch should get appropriate flow split, no double-counting
        """
        net = FlowNetwork("parallel_test")
        
        # Nodes
        source = net.create_node("source")
        junction = net.create_node("junction") 
        sink1 = net.create_node("sink1")
        sink2 = net.create_node("sink2")
        
        net.set_inlet(source)
        net.add_outlet(sink1)
        net.add_outlet(sink2)
        
        # Components - make branch1 easier (lower resistance)
        feeder = Channel(diameter=0.02, length=1.0, component_id="feeder")
        branch1 = Channel(diameter=0.015, length=1.0, component_id="branch1")  # Larger diameter
        branch2 = Channel(diameter=0.010, length=1.0, component_id="branch2")  # Smaller diameter
        
        # Connections
        net.connect_components(source, junction, feeder)
        net.connect_components(junction, sink1, branch1)
        net.connect_components(junction, sink2, branch2)
        
        # Initialize flows
        edge_flows = self.solver._initialize_flows(net, source.id, [sink1.id, sink2.id], self.Q_total)
        
        # Test 1: Mass conservation at junction
        flow_in = edge_flows["feeder"]
        flow_out = edge_flows["branch1"] + edge_flows["branch2"]
        
        print(f"Flow into junction: {flow_in:.6f}")
        print(f"Flow out of junction: {flow_out:.6f}")
        print(f"Mass conservation error: {abs(flow_in - flow_out):.6e}")
        
        assert abs(flow_in - flow_out) < 1e-6, f"Mass conservation violated at junction: in={flow_in}, out={flow_out}"
        
        # Test 2: Total flow conservation
        total_source_flow = edge_flows["feeder"]
        total_sink_flow = edge_flows["branch1"] + edge_flows["branch2"]
        
        assert abs(total_source_flow - self.Q_total) < 1e-6, f"Source flow error: {total_source_flow} vs {self.Q_total}"
        assert abs(total_sink_flow - self.Q_total) < 1e-6, f"Sink flow error: {total_sink_flow} vs {self.Q_total}"
        
        # Test 3: Flow split should favor lower resistance (larger diameter)
        assert edge_flows["branch1"] > edge_flows["branch2"], "Flow should favor lower resistance branch"
    
    def test_shared_upstream_component(self):
        """
        Test Case 2: Shared upstream component (the critical case)
        
        Source → [Pipe A] → Junction → [Pipe B] → Sink1
                                   └→ [Pipe C] → Sink2
        
        This tests if Pipe A gets double-counted in the path-based algorithm.
        Expected: Pipe A flow = total flow, Pipe B + Pipe C = total flow
        """
        net = FlowNetwork("shared_upstream_test")
        
        # Nodes
        source = net.create_node("source")
        junction = net.create_node("junction")
        sink1 = net.create_node("sink1")
        sink2 = net.create_node("sink2")
        
        net.set_inlet(source)
        net.add_outlet(sink1)
        net.add_outlet(sink2)
        
        # Components
        pipe_a = Channel(diameter=0.020, length=1.0, component_id="pipe_a")  # Shared upstream
        pipe_b = Channel(diameter=0.015, length=1.0, component_id="pipe_b")  # Branch 1
        pipe_c = Channel(diameter=0.015, length=1.0, component_id="pipe_c")  # Branch 2
        
        # Connections
        net.connect_components(source, junction, pipe_a)
        net.connect_components(junction, sink1, pipe_b)
        net.connect_components(junction, sink2, pipe_c)
        
        # Initialize flows
        edge_flows = self.solver._initialize_flows(net, source.id, [sink1.id, sink2.id], self.Q_total)
        
        print(f"Pipe A (shared): {edge_flows['pipe_a']:.6f}")
        print(f"Pipe B (branch1): {edge_flows['pipe_b']:.6f}")
        print(f"Pipe C (branch2): {edge_flows['pipe_c']:.6f}")
        print(f"Total flow assigned: {sum(edge_flows.values()):.6f}")
        
        # Test 1: Shared pipe should carry exactly the total flow
        assert abs(edge_flows["pipe_a"] - self.Q_total) < 1e-6, \
            f"Shared pipe A should carry total flow: {edge_flows['pipe_a']} vs {self.Q_total}"
        
        # Test 2: Branch flows should sum to total flow
        branch_total = edge_flows["pipe_b"] + edge_flows["pipe_c"]
        assert abs(branch_total - self.Q_total) < 1e-6, \
            f"Branch flows should sum to total: {branch_total} vs {self.Q_total}"
        
        # Test 3: Mass conservation at junction
        flow_in = edge_flows["pipe_a"]
        flow_out = edge_flows["pipe_b"] + edge_flows["pipe_c"]
        assert abs(flow_in - flow_out) < 1e-6, \
            f"Mass conservation at junction: in={flow_in}, out={flow_out}"
        
        # Test 4: Total flow assignment should not exceed physical total
        # This is the critical test for double-counting
        total_assigned = sum(edge_flows.values())
        max_allowed = 2 * self.Q_total  # Upper bound for this topology
        assert total_assigned <= max_allowed, \
            f"Total flow assignment suggests double-counting: {total_assigned} vs max {max_allowed}"
    
    def test_complex_tree_network(self):
        """
        Test Case 3: Complex tree with multiple shared components
        
        Source → [A] → J1 → [B] → J2 → [D] → Sink1
                          ↓        ↓
                        [C] → J3   [E] → Sink2
                              ↓
                            [F] → Sink3
        
        Tests multiple levels of shared components
        """
        net = FlowNetwork("complex_tree_test")
        
        # Nodes
        source = net.create_node("source")
        j1 = net.create_node("j1")
        j2 = net.create_node("j2") 
        j3 = net.create_node("j3")
        sink1 = net.create_node("sink1")
        sink2 = net.create_node("sink2")
        sink3 = net.create_node("sink3")
        
        net.set_inlet(source)
        net.add_outlet(sink1)
        net.add_outlet(sink2)
        net.add_outlet(sink3)
        
        # Components (all same size for simplicity)
        pipes = {}
        for name in ['A', 'B', 'C', 'D', 'E', 'F']:
            pipes[name] = Channel(diameter=0.015, length=1.0, component_id=f"pipe_{name}")
        
        # Connections
        net.connect_components(source, j1, pipes['A'])
        net.connect_components(j1, j2, pipes['B'])
        net.connect_components(j1, j3, pipes['C'])
        net.connect_components(j2, sink1, pipes['D'])
        net.connect_components(j2, sink2, pipes['E'])
        net.connect_components(j3, sink3, pipes['F'])
        
        # Initialize flows
        edge_flows = self.solver._initialize_flows(net, source.id, [sink1.id, sink2.id, sink3.id], self.Q_total)
        
        print("Complex tree flows:")
        for name, flow in edge_flows.items():
            print(f"  {name}: {flow:.6f}")
        
        # Test 1: Pipe A should carry total flow (all paths pass through it)
        assert abs(edge_flows["pipe_A"] - self.Q_total) < 1e-6, \
            f"Pipe A should carry total flow: {edge_flows['pipe_A']}"
        
        # Test 2: Mass conservation at each junction
        # J1: A_in = B_out + C_out
        j1_in = edge_flows["pipe_A"]
        j1_out = edge_flows["pipe_B"] + edge_flows["pipe_C"]
        assert abs(j1_in - j1_out) < 1e-6, f"Mass conservation at J1: {j1_in} vs {j1_out}"
        
        # J2: B_in = D_out + E_out
        j2_in = edge_flows["pipe_B"]
        j2_out = edge_flows["pipe_D"] + edge_flows["pipe_E"]
        assert abs(j2_in - j2_out) < 1e-6, f"Mass conservation at J2: {j2_in} vs {j2_out}"
        
        # J3: C_in = F_out
        j3_in = edge_flows["pipe_C"]
        j3_out = edge_flows["pipe_F"]
        assert abs(j3_in - j3_out) < 1e-6, f"Mass conservation at J3: {j3_in} vs {j3_out}"
        
        # Test 3: Total sink flows
        total_sinks = edge_flows["pipe_D"] + edge_flows["pipe_E"] + edge_flows["pipe_F"]
        assert abs(total_sinks - self.Q_total) < 1e-6, \
            f"Total sink flows: {total_sinks} vs {self.Q_total}"
    
    def test_double_counting_detection(self):
        """
        Test Case 4: Explicit test for double-counting in current implementation
        
        Creates a simple case where double-counting is obvious and measures it.
        """
        net = FlowNetwork("double_count_test")
        
        # Simple Y-network: one shared pipe feeding two branches
        source = net.create_node("source")
        junction = net.create_node("junction")
        sink1 = net.create_node("sink1")
        sink2 = net.create_node("sink2")
        
        net.set_inlet(source)
        net.add_outlet(sink1)
        net.add_outlet(sink2)
        
        # Equal resistance components
        shared = Channel(diameter=0.015, length=1.0, component_id="shared")
        branch1 = Channel(diameter=0.015, length=1.0, component_id="branch1")
        branch2 = Channel(diameter=0.015, length=1.0, component_id="branch2")
        
        net.connect_components(source, junction, shared)
        net.connect_components(junction, sink1, branch1)
        net.connect_components(junction, sink2, branch2)
        
        # Initialize flows
        edge_flows = self.solver._initialize_flows(net, source.id, [sink1.id, sink2.id], self.Q_total)
        
        # Calculate double-counting factor
        shared_flow = edge_flows["shared"]
        expected_shared_flow = self.Q_total
        double_count_factor = shared_flow / expected_shared_flow
        
        print(f"Expected shared flow: {expected_shared_flow:.6f}")
        print(f"Actual shared flow: {shared_flow:.6f}")
        print(f"Double-counting factor: {double_count_factor:.3f}")
        
        # If double-counting exists, shared flow will be > Q_total
        if double_count_factor > 1.1:  # Allow 10% tolerance
            pytest.fail(f"Double-counting detected: shared pipe has {double_count_factor:.2f}x expected flow")
        
        # Mass conservation check
        flow_balance_error = abs(shared_flow - (edge_flows["branch1"] + edge_flows["branch2"]))
        if flow_balance_error > 1e-6:
            pytest.fail(f"Mass conservation violated: balance error = {flow_balance_error:.2e}")

def test_pathfinding_algorithm_directly():
    """
    Test the path-finding algorithm in isolation to show double-counting
    """
    # Create simple network for manual analysis
    net = FlowNetwork("path_test")
    
    source = net.create_node("source")
    junction = net.create_node("junction")
    sink1 = net.create_node("sink1")
    sink2 = net.create_node("sink2")
    
    net.set_inlet(source)
    net.add_outlet(sink1)
    net.add_outlet(sink2)
    
    shared = Channel(diameter=0.015, length=1.0, component_id="shared")
    branch1 = Channel(diameter=0.015, length=1.0, component_id="branch1")
    branch2 = Channel(diameter=0.015, length=1.0, component_id="branch2")
    
    net.connect_components(source, junction, shared)
    net.connect_components(junction, sink1, branch1)
    net.connect_components(junction, sink2, branch2)
    
    sim_config = SimulationConfig(total_flow_rate=0.1, oil_density=850, oil_type="SAE30", temperature=40, inlet_pressure=101325)
    solver = NodalMatrixSolver(sim_config)
    
    # Test current path-finding logic manually
    all_paths = []
    
    # Find paths to sink1
    for sink_id in [sink1.id, sink2.id]:
        paths_to_sink = []
        queue = [(source.id, [])]
        visited = set()
        
        while queue:
            curr_node_id, path = queue.pop(0)
            
            if curr_node_id == sink_id:
                paths_to_sink.append(path)
                continue
                
            if curr_node_id in visited:
                continue
            visited.add(curr_node_id)
            
            for conn in net.connections:
                if conn.from_node.id == curr_node_id and conn.to_node.id not in visited:
                    new_path = path + [conn.component.id]
                    queue.append((conn.to_node.id, new_path))
        
        all_paths.extend(paths_to_sink)
    
    print("Found paths:")
    for i, path in enumerate(all_paths):
        print(f"  Path {i+1}: {' → '.join(path)}")
    
    # Count how many times each component appears
    component_count = {}
    for path in all_paths:
        for comp_id in path:
            component_count[comp_id] = component_count.get(comp_id, 0) + 1
    
    print("Component usage count:")
    for comp_id, count in component_count.items():
        print(f"  {comp_id}: {count} times")
    
    # The shared component should appear in multiple paths (this is the bug!)
    assert component_count["shared"] > 1, "Shared component should appear in multiple paths (demonstrating the bug)"
    print(f"CONFIRMED: Shared component appears {component_count['shared']} times (double-counting bug)")

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])