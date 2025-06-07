#!/usr/bin/env python3
"""
Enhanced Web-Based Network Lubrication Flow Distribution GUI

A modern, feature-rich web interface for the network-based lubrication flow
distribution calculator with advanced functionality including:

- Modern, responsive web UI design
- Interactive network visualization with D3.js
- Drag & drop component building
- Real-time validation and feedback
- Advanced results visualization
- Component library and templates
- Export capabilities (PDF, PNG, CSV)
- Network analysis tools
- Auto-save functionality
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import os
import datetime
import uuid
import math
import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import pandas as pd

# Import the backend functionality
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType
)

app = Flask(__name__)
app.secret_key = 'lubrication_calculator_secret_key'

# Global variables for the application state
current_network = FlowNetwork("New Network")
current_solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
current_results = None
node_positions = {}

class ComponentLibrary:
    """Library of predefined components and templates"""
    
    @staticmethod
    def get_channel_presets():
        return {
            "Small Pipe": {"diameter": 0.025, "length": 5.0, "roughness": 0.00015},
            "Medium Pipe": {"diameter": 0.050, "length": 10.0, "roughness": 0.00015},
            "Large Pipe": {"diameter": 0.100, "length": 15.0, "roughness": 0.00015},
            "Smooth Tube": {"diameter": 0.020, "length": 3.0, "roughness": 0.000001},
            "Rough Pipe": {"diameter": 0.075, "length": 12.0, "roughness": 0.0005}
        }
    
    @staticmethod
    def get_connector_presets():
        return {
            "Standard T-Junction": {"type": "T_JUNCTION", "diameter": 0.050},
            "Large T-Junction": {"type": "T_JUNCTION", "diameter": 0.100},
            "90° Elbow": {"type": "ELBOW_90", "diameter": 0.050},
            "Cross Junction": {"type": "X_JUNCTION", "diameter": 0.075},
            "Reducer": {"type": "REDUCER", "diameter": 0.050, "diameter_out": 0.025}
        }
    
    @staticmethod
    def get_nozzle_presets():
        return {
            "Small Sharp Nozzle": {"type": "SHARP_EDGED", "diameter": 0.008},
            "Medium Sharp Nozzle": {"type": "SHARP_EDGED", "diameter": 0.012},
            "Large Sharp Nozzle": {"type": "SHARP_EDGED", "diameter": 0.020},
            "Rounded Nozzle": {"type": "ROUNDED", "diameter": 0.010},
            "Venturi Nozzle": {"type": "VENTURI", "diameter": 0.015}
        }
    
    @staticmethod
    def get_network_templates():
        return {
            "Simple Branch": {
                "description": "Basic 2-outlet branching system",
                "nodes": 3,
                "complexity": "Simple"
            },
            "Tree Network": {
                "description": "Multi-level tree with 4 outlets",
                "nodes": 7,
                "complexity": "Medium"
            },
            "Complex Distribution": {
                "description": "Complex network with 8 outlets",
                "nodes": 15,
                "complexity": "Complex"
            }
        }

# Utility functions
def serialize_network(network):
    """Serialize network to JSON-compatible format"""
    network_data = {
        'name': network.name,
        'nodes': [],
        'connections': [],
        'inlet_node_id': network.inlet_node.id if network.inlet_node and hasattr(network.inlet_node, 'id') else None,
        'outlet_node_ids': [node.id for node in network.outlet_nodes if hasattr(node, 'id')]
    }
    
    # Serialize nodes
    for node in network.nodes.values():
        node_data = {
            'id': node.id,
            'name': node.name,
            'elevation': node.elevation,
            'pressure': getattr(node, 'pressure', 0.0)
        }
        network_data['nodes'].append(node_data)
    
    # Serialize connections
    for connection in network.connections:
        comp = connection.component
        comp_data = {
            'id': comp.id,
            'type': comp.component_type.value,
            'name': getattr(comp, 'name', None)
        }
        
        # Add component-specific data
        if comp.component_type == ComponentType.CHANNEL:
            comp_data.update({
                'diameter': comp.diameter,
                'length': comp.length,
                'roughness': comp.roughness
            })
        elif comp.component_type == ComponentType.CONNECTOR:
            comp_data.update({
                'connector_type': comp.connector_type.value,
                'diameter': comp.diameter,
                'diameter_out': getattr(comp, 'diameter_out', None)
            })
        elif comp.component_type == ComponentType.NOZZLE:
            comp_data.update({
                'diameter': comp.diameter,
                'nozzle_type': comp.nozzle_type.value,
                'discharge_coeff': getattr(comp, 'discharge_coeff', 0.6)
            })
        
        connection_data = {
            'from_node_id': connection.from_node.id,
            'to_node_id': connection.to_node.id,
            'component': comp_data,
            'flow_rate': connection.flow_rate
        }
        network_data['connections'].append(connection_data)
    
    return network_data

def deserialize_network(network_data):
    """Deserialize network from JSON-compatible format"""
    network = FlowNetwork(network_data.get('name', 'Loaded Network'))
    
    # Create nodes
    node_map = {}
    for node_data in network_data['nodes']:
        node = network.create_node(node_data['name'], elevation=node_data['elevation'])
        node.id = node_data['id']  # Preserve original ID
        node.pressure = node_data.get('pressure', 0.0)
        node_map[node_data['id']] = node
    
    # Create connections
    for conn_data in network_data['connections']:
        from_node = node_map[conn_data['from_node_id']]
        to_node = node_map[conn_data['to_node_id']]
        comp_data = conn_data['component']
        
        # Create component based on type
        if comp_data['type'] == 'channel':
            component = Channel(
                diameter=comp_data['diameter'],
                length=comp_data['length'],
                roughness=comp_data['roughness'],
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'connector':
            component = Connector(
                ConnectorType(comp_data['connector_type']),
                diameter=comp_data['diameter'],
                diameter_out=comp_data.get('diameter_out'),
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'nozzle':
            component = Nozzle(
                diameter=comp_data['diameter'],
                nozzle_type=NozzleType(comp_data['nozzle_type']),
                discharge_coeff=comp_data.get('discharge_coeff', 0.6),
                name=comp_data.get('name')
            )
        
        component.id = comp_data['id']  # Preserve original ID
        connection = network.connect_components(from_node, to_node, component)
        connection.flow_rate = conn_data.get('flow_rate', 0.0)
    
    # Set inlet and outlets
    if network_data.get('inlet_node_id'):
        inlet_node = node_map[network_data['inlet_node_id']]
        network.set_inlet(inlet_node)
    
    for outlet_id in network_data.get('outlet_node_ids', []):
        outlet_node = node_map[outlet_id]
        network.add_outlet(outlet_node)
    
    return network

def create_example_network():
    """Create an example network for demonstration"""
    network = FlowNetwork("Example Network")
    
    # Create nodes
    inlet = network.create_node("Inlet", elevation=0.0)
    junction1 = network.create_node("Junction1", elevation=0.0)
    junction2 = network.create_node("Junction2", elevation=0.0)
    outlet1 = network.create_node("Outlet1", elevation=0.0)
    outlet2 = network.create_node("Outlet2", elevation=0.0)
    outlet3 = network.create_node("Outlet3", elevation=0.0)
    
    # Create components
    main_channel = Channel(diameter=0.050, length=10.0, roughness=0.00015, name="Main Channel")
    branch_channel = Channel(diameter=0.040, length=8.0, roughness=0.00015, name="Branch Channel")
    sub_branch1 = Channel(diameter=0.025, length=5.0, roughness=0.00015, name="Sub Branch 1")
    sub_branch2 = Channel(diameter=0.025, length=5.0, roughness=0.00015, name="Sub Branch 2")
    outlet_channel = Channel(diameter=0.030, length=6.0, roughness=0.00015, name="Outlet Channel")
    
    # Connect components
    network.connect_components(inlet, junction1, main_channel)
    network.connect_components(junction1, junction2, branch_channel)
    network.connect_components(junction2, outlet1, sub_branch1)
    network.connect_components(junction2, outlet2, sub_branch2)
    network.connect_components(junction1, outlet3, outlet_channel)
    
    # Set inlet and outlets
    network.set_inlet(inlet)
    network.add_outlet(outlet1)
    network.add_outlet(outlet2)
    network.add_outlet(outlet3)
    
    return network

def auto_layout_nodes(network):
    """Automatically layout nodes for visualization"""
    positions = {}
    
    if not network.nodes:
        return positions
    
    # Get all nodes from the dictionary values
    nodes = list(network.nodes.values())
    if not nodes:
        return positions
    
    if network.inlet_node:
        # Hierarchical layout starting from inlet
        positions[network.inlet_node.id] = {'x': 100, 'y': 300}
        positioned = {network.inlet_node.id}
        
        # Build adjacency list
        adjacency = {}
        for node in nodes:
            adjacency[node.id] = []
        
        for connection in network.connections:
            if hasattr(connection.from_node, 'id') and hasattr(connection.to_node, 'id'):
                adjacency[connection.from_node.id].append(connection.to_node)
                adjacency[connection.to_node.id].append(connection.from_node)
        
        # BFS to position nodes level by level
        queue = [(network.inlet_node, 0)]
        level_nodes = {}
        
        while queue:
            node, level = queue.pop(0)
            if level not in level_nodes:
                level_nodes[level] = []
            level_nodes[level].append(node)
            
            for neighbor in adjacency[node.id]:
                if neighbor.id not in positioned:
                    positioned.add(neighbor.id)
                    queue.append((neighbor, level + 1))
        
        # Position nodes at each level
        for level, nodes in level_nodes.items():
            if len(nodes) == 1:
                y_positions = [300]
            else:
                y_start = 150
                y_end = 450
                y_positions = [y_start + i * (y_end - y_start) / (len(nodes) - 1) for i in range(len(nodes))]
            
            for i, node in enumerate(nodes):
                positions[node.id] = {
                    'x': 100 + level * 200,
                    'y': y_positions[i] if len(nodes) > 1 else 300
                }
    else:
        # Simple grid layout
        for i, node in enumerate(nodes):
            x = 100 + (i % 3) * 200
            y = 150 + (i // 3) * 150
            positions[node.id] = {'x': x, 'y': y}
    
    return positions

def generate_network_visualization(network, results=None):
    """Generate network visualization data for D3.js"""
    nodes_data = []
    links_data = []
    
    # Generate node positions if not available
    global node_positions
    if not node_positions:
        node_positions = auto_layout_nodes(network)
    
    # Prepare nodes data
    for node in network.nodes.values():
        node_type = "junction"
        if node == network.inlet_node:
            node_type = "inlet"
        elif node in network.outlet_nodes:
            node_type = "outlet"
        
        pressure = 0
        if results and node.id in results['solution_info']['node_pressures']:
            pressure = results['solution_info']['node_pressures'][node.id]
        
        position = node_positions.get(node.id, {'x': 100, 'y': 100})
        
        node_data = {
            'id': node.id,
            'name': node.name,
            'type': node_type,
            'elevation': node.elevation,
            'pressure': pressure,
            'x': position['x'],
            'y': position['y']
        }
        nodes_data.append(node_data)
    
    # Prepare links data
    for connection in network.connections:
        comp = connection.component
        flow_rate = 0
        if results and comp.id in results['connection_flows']:
            flow_rate = results['connection_flows'][comp.id] * 1000  # Convert to L/s
        
        link_data = {
            'id': comp.id,
            'source': connection.from_node.id,
            'target': connection.to_node.id,
            'type': comp.component_type.value,
            'name': getattr(comp, 'name', f"{comp.component_type.value}_{comp.id[:8]}"),
            'flow_rate': flow_rate,
            'diameter': getattr(comp, 'diameter', 0) * 1000,  # Convert to mm
            'length': getattr(comp, 'length', 0) if hasattr(comp, 'length') else 0
        }
        links_data.append(link_data)
    
    return {
        'nodes': nodes_data,
        'links': links_data
    }

def create_results_charts(results):
    """Create charts for results visualization"""
    if not results:
        return None
    
    # Create pressure distribution chart
    plt.figure(figsize=(12, 8))
    
    # Pressure chart
    plt.subplot(2, 2, 1)
    node_names = []
    pressures = []
    colors = []
    
    for node in current_network.nodes:
        node_names.append(node.name)
        pressure = results['solution_info']['node_pressures'].get(node.id, 0) / 1000  # kPa
        pressures.append(pressure)
        
        if node == current_network.inlet_node:
            colors.append('green')
        elif node in current_network.outlet_nodes:
            colors.append('red')
        else:
            colors.append('steelblue')
    
    bars = plt.bar(node_names, pressures, color=colors, alpha=0.7)
    plt.title('Pressure Distribution', fontweight='bold')
    plt.ylabel('Pressure (kPa)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Flow distribution chart
    plt.subplot(2, 2, 2)
    comp_names = []
    flows = []
    comp_colors = []
    
    for connection in current_network.connections:
        comp = connection.component
        comp_name = getattr(comp, 'name', f"{comp.component_type.value}_{comp.id[:8]}")
        comp_names.append(comp_name)
        
        flow = results['connection_flows'].get(comp.id, 0) * 1000  # L/s
        flows.append(flow)
        
        if comp.component_type == ComponentType.CHANNEL:
            comp_colors.append('dodgerblue')
        elif comp.component_type == ComponentType.CONNECTOR:
            comp_colors.append('darkorchid')
        elif comp.component_type == ComponentType.NOZZLE:
            comp_colors.append('tomato')
        else:
            comp_colors.append('gray')
    
    plt.bar(comp_names, flows, color=comp_colors, alpha=0.7)
    plt.title('Flow Distribution', fontweight='bold')
    plt.ylabel('Flow Rate (L/s)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Velocity distribution chart
    plt.subplot(2, 2, 3)
    velocities = []
    for connection in current_network.connections:
        comp = connection.component
        flow = results['connection_flows'].get(comp.id, 0)  # m³/s
        
        if hasattr(comp, 'diameter') and comp.diameter > 0:
            area = math.pi * (comp.diameter / 2) ** 2
            velocity = flow / area  # m/s
        else:
            velocity = 0
        
        velocities.append(velocity)
    
    plt.bar(comp_names, velocities, color=comp_colors, alpha=0.7)
    plt.title('Velocity Distribution', fontweight='bold')
    plt.ylabel('Velocity (m/s)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # System overview
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    info = results['solution_info']
    overview_text = f"""
System Overview

Total Flow Rate: {results['total_flow']*1000:.1f} L/s
Temperature: {info['temperature']:.1f}°C
Oil Type: {current_solver.oil_type}
Viscosity: {info['viscosity']:.6f} Pa·s

Network Statistics:
Nodes: {len(current_network.nodes)}
Components: {len(current_network.connections)}
Outlets: {len(current_network.outlet_nodes)}

Convergence: {'✓' if info['converged'] else '✗'}
Iterations: {info['iterations']}
    """
    
    plt.text(0.1, 0.9, overview_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save to base64 string
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return img_base64

# Flask routes
@app.route('/')
def index():
    """Main application page"""
    return render_template('enhanced_network_index.html')

@app.route('/api/network', methods=['GET'])
def get_network():
    """Get current network data"""
    global current_network, current_results, node_positions
    
    network_data = serialize_network(current_network)
    network_data['node_positions'] = node_positions
    
    visualization_data = generate_network_visualization(current_network, current_results)
    
    return jsonify({
        'network': network_data,
        'visualization': visualization_data,
        'has_results': current_results is not None
    })

@app.route('/api/network', methods=['POST'])
def save_network():
    """Save network data"""
    global current_network, node_positions
    
    try:
        data = request.json
        current_network = deserialize_network(data['network'])
        node_positions = data.get('node_positions', {})
        
        return jsonify({'success': True, 'message': 'Network saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/node', methods=['POST'])
def add_node():
    """Add a new node"""
    global current_network, node_positions
    
    try:
        data = request.json
        node = current_network.create_node(
            data['name'], 
            elevation=data.get('elevation', 0.0)
        )
        
        # Set position if provided
        if 'position' in data:
            node_positions[node.id] = data['position']
        else:
            # Auto-position
            max_x = max([pos.get('x', 0) for pos in node_positions.values()], default=0)
            node_positions[node.id] = {'x': max_x + 200, 'y': 300}
        
        return jsonify({
            'success': True, 
            'node': {
                'id': node.id,
                'name': node.name,
                'elevation': node.elevation
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/node/<node_id>', methods=['PUT'])
def update_node(node_id):
    """Update a node"""
    global current_network, node_positions
    
    try:
        data = request.json
        
        # Find node
        node = None
        for n in current_network.nodes:
            if n.id == node_id:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'}), 404
        
        # Update properties
        if 'name' in data:
            node.name = data['name']
        if 'elevation' in data:
            node.elevation = data['elevation']
        if 'position' in data:
            node_positions[node.id] = data['position']
        
        return jsonify({'success': True, 'message': 'Node updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/node/<node_id>', methods=['DELETE'])
def delete_node(node_id):
    """Delete a node"""
    global current_network
    
    try:
        # Find node
        node = None
        for n in current_network.nodes:
            if n.id == node_id:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'}), 404
        
        # Remove connections involving this node
        connections_to_remove = []
        for conn in current_network.connections:
            if conn.from_node == node or conn.to_node == node:
                connections_to_remove.append(conn)
        
        for conn in connections_to_remove:
            current_network.connections.remove(conn)
        
        # Remove node
        current_network.nodes.remove(node)
        
        # Remove from inlet/outlets
        if current_network.inlet_node == node:
            current_network.inlet_node = None
        if node in current_network.outlet_nodes:
            current_network.outlet_nodes.remove(node)
        
        return jsonify({'success': True, 'message': 'Node deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/component', methods=['POST'])
def add_component():
    """Add a new component"""
    global current_network
    
    try:
        data = request.json
        
        # Find nodes
        from_node = None
        to_node = None
        
        for node in current_network.nodes:
            if node.id == data['from_node_id']:
                from_node = node
            if node.id == data['to_node_id']:
                to_node = node
        
        if not from_node or not to_node:
            return jsonify({'success': False, 'error': 'Invalid node IDs'}), 400
        
        # Create component
        comp_data = data['component']
        component_type = comp_data['type']
        
        if component_type == 'channel':
            component = Channel(
                diameter=comp_data['diameter'] / 1000,  # Convert mm to m
                length=comp_data['length'],
                roughness=comp_data['roughness'] / 1000,  # Convert mm to m
                name=comp_data.get('name')
            )
        elif component_type == 'connector':
            component = Connector(
                ConnectorType(comp_data['connector_type']),
                diameter=comp_data['diameter'] / 1000,  # Convert mm to m
                diameter_out=comp_data.get('diameter_out', comp_data['diameter']) / 1000,
                name=comp_data.get('name')
            )
        elif component_type == 'nozzle':
            component = Nozzle(
                diameter=comp_data['diameter'] / 1000,  # Convert mm to m
                nozzle_type=NozzleType(comp_data['nozzle_type']),
                discharge_coeff=comp_data.get('discharge_coeff', 0.6),
                name=comp_data.get('name')
            )
        else:
            return jsonify({'success': False, 'error': 'Invalid component type'}), 400
        
        # Connect components
        connection = current_network.connect_components(from_node, to_node, component)
        
        return jsonify({
            'success': True,
            'component': {
                'id': component.id,
                'type': component.component_type.value,
                'name': getattr(component, 'name', None)
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/network/inlet', methods=['POST'])
def set_inlet():
    """Set inlet node"""
    global current_network
    
    try:
        data = request.json
        node_id = data['node_id']
        
        # Find node
        node = None
        for n in current_network.nodes:
            if n.id == node_id:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'}), 404
        
        current_network.set_inlet(node)
        return jsonify({'success': True, 'message': 'Inlet set successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/network/outlet', methods=['POST'])
def add_outlet():
    """Add outlet node"""
    global current_network
    
    try:
        data = request.json
        node_id = data['node_id']
        
        # Find node
        node = None
        for n in current_network.nodes:
            if n.id == node_id:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'}), 404
        
        current_network.add_outlet(node)
        return jsonify({'success': True, 'message': 'Outlet added successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/network/validate', methods=['POST'])
def validate_network():
    """Validate the network"""
    global current_network
    
    try:
        is_valid, errors = current_network.validate_network()
        
        return jsonify({
            'valid': is_valid,
            'errors': errors
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/calculate', methods=['POST'])
def calculate_flow():
    """Calculate flow distribution"""
    global current_network, current_solver, current_results
    
    try:
        data = request.json
        
        # Validate network first
        is_valid, errors = current_network.validate_network()
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Network validation failed',
                'validation_errors': errors
            }), 400
        
        # Update solver properties
        current_solver.oil_density = data.get('oil_density', 900.0)
        current_solver.oil_type = data.get('oil_type', 'SAE30')
        
        # Calculate flow distribution
        total_flow = data.get('total_flow', 15.0) / 1000  # Convert L/s to m³/s
        temperature = data.get('temperature', 40.0)
        
        connection_flows, solution_info = current_solver.solve_network_flow(
            current_network, total_flow, temperature
        )
        
        current_results = {
            'connection_flows': connection_flows,
            'solution_info': solution_info,
            'total_flow': total_flow,
            'temperature': temperature
        }
        
        # Generate charts
        charts_base64 = create_results_charts(current_results)
        
        # Format results for display
        results_text = format_results_text(current_results)
        
        return jsonify({
            'success': True,
            'results': current_results,
            'results_text': results_text,
            'charts': charts_base64,
            'visualization': generate_network_visualization(current_network, current_results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

def format_results_text(results):
    """Format results as text for display"""
    if not results:
        return ""
    
    text = "NETWORK FLOW DISTRIBUTION RESULTS\n"
    text += "=" * 70 + "\n\n"
    
    info = results['solution_info']
    text += f"📊 SYSTEM OVERVIEW\n"
    text += f"{'─' * 50}\n"
    text += f"Network Name:        {current_network.name}\n"
    text += f"Analysis Date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text += f"Temperature:         {info['temperature']:.1f}°C\n"
    text += f"Oil Type:            {current_solver.oil_type}\n"
    text += f"Oil Density:         {current_solver.oil_density:.1f} kg/m³\n"
    text += f"Dynamic Viscosity:   {info['viscosity']:.6f} Pa·s\n"
    text += f"Total Flow Rate:     {results['total_flow']*1000:.2f} L/s\n"
    text += f"Convergence:         {'✓ Yes' if info['converged'] else '✗ No'} ({info['iterations']} iterations)\n\n"
    
    text += f"🔧 COMPONENT ANALYSIS\n"
    text += f"{'─' * 70}\n"
    text += f"{'Component Name':<25} {'Type':<12} {'Flow Rate':<12} {'Velocity':<12}\n"
    text += f"{'':25} {'':12} {'(L/s)':<12} {'(m/s)':<12}\n"
    text += f"{'─' * 70}\n"
    
    # Component flows with enhanced details
    for connection in current_network.connections:
        comp = connection.component
        flow = results['connection_flows'].get(comp.id, 0) * 1000  # Convert to L/s
        
        # Calculate velocity
        if hasattr(comp, 'diameter') and comp.diameter > 0:
            area = math.pi * (comp.diameter / 2) ** 2
            velocity = (flow / 1000) / area  # m/s
        else:
            velocity = 0
        
        comp_name = getattr(comp, 'name', f"{comp.component_type.value}_{comp.id[:8]}")
        text += f"{comp_name:<25} {comp.component_type.value:<12} {flow:<12.3f} {velocity:<12.2f}\n"
    
    text += f"{'─' * 70}\n"
    
    # Node pressures
    text += f"\n🏗️ NODE PRESSURES\n"
    text += f"{'─' * 50}\n"
    text += f"{'Node Name':<20} {'Pressure':<15} {'Elevation':<12} {'Type':<10}\n"
    text += f"{'':20} {'(kPa)':<15} {'(m)':<12} {'':10}\n"
    text += f"{'─' * 50}\n"
    
    for node in current_network.nodes:
        pressure = info['node_pressures'].get(node.id, 0) / 1000  # Convert to kPa
        
        if node == current_network.inlet_node:
            node_type = "Inlet"
        elif node in current_network.outlet_nodes:
            node_type = "Outlet"
        else:
            node_type = "Junction"
        
        text += f"{node.name:<20} {pressure:<15.1f} {node.elevation:<12.1f} {node_type:<10}\n"
    
    return text

@app.route('/api/example', methods=['POST'])
def load_example():
    """Load example network"""
    global current_network, node_positions
    
    try:
        current_network = create_example_network()
        node_positions = auto_layout_nodes(current_network)
        
        return jsonify({
            'success': True,
            'message': 'Example network loaded',
            'visualization': generate_network_visualization(current_network)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/clear', methods=['POST'])
def clear_network():
    """Clear the current network"""
    global current_network, current_results, node_positions
    
    try:
        current_network = FlowNetwork("New Network")
        current_results = None
        node_positions = {}
        
        return jsonify({
            'success': True,
            'message': 'Network cleared'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/presets/<component_type>')
def get_presets(component_type):
    """Get component presets"""
    try:
        if component_type == 'channel':
            presets = ComponentLibrary.get_channel_presets()
        elif component_type == 'connector':
            presets = ComponentLibrary.get_connector_presets()
        elif component_type == 'nozzle':
            presets = ComponentLibrary.get_nozzle_presets()
        else:
            return jsonify({'success': False, 'error': 'Invalid component type'}), 400
        
        return jsonify({'presets': presets})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/export/results')
def export_results():
    """Export results as text file"""
    global current_results
    
    if not current_results:
        return jsonify({'success': False, 'error': 'No results to export'}), 400
    
    try:
        results_text = format_results_text(current_results)
        
        # Create file-like object
        output = BytesIO()
        output.write(results_text.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f'lubrication_results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
            mimetype='text/plain'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/export/network')
def export_network_data():
    """Export network as JSON file"""
    global current_network, node_positions
    
    try:
        network_data = serialize_network(current_network)
        network_data['node_positions'] = node_positions
        
        # Create file-like object
        output = BytesIO()
        output.write(json.dumps(network_data, indent=2).encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            as_attachment=True,
            download_name=f'network_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            mimetype='application/json'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    # Initialize with example network
    current_network = create_example_network()
    node_positions = auto_layout_nodes(current_network)
    
    # Run the application
    app.run(host='0.0.0.0', port=50960, debug=True)