#!/usr/bin/env python3
"""
Network-Based Lubrication Flow Distribution Web Application

A modern web application for the network-based lubrication flow distribution
calculator. Features include:
- Interactive network topology building
- Component-based system construction (channels, connectors, nozzles)
- Visual network representation with flow visualization
- Tree-like branching support
- Real-time flow distribution calculation
- Export capabilities for results and network configurations
"""

from flask import Flask, render_template, request, jsonify, send_file
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
# import networkx as nx  # Not available, using custom layout
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType,
    create_simple_tree_example
)

app = Flask(__name__)

# Global solver instance
solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")

@app.route('/')
def index():
    """Main page"""
    return render_template('network_index.html')

@app.route('/api/create_example', methods=['POST'])
def create_example():
    """Create an example network"""
    try:
        network, total_flow_rate, temperature = create_simple_tree_example()
        
        # Serialize network for frontend
        network_data = serialize_network(network)
        network_data['system_params'] = {
            'total_flow': total_flow_rate * 1000,  # Convert to L/s
            'temperature': temperature,
            'oil_type': 'SAE30',
            'oil_density': 900.0
        }
        
        return jsonify({'success': True, 'network': network_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/calculate', methods=['POST'])
def calculate_flow():
    """API endpoint for flow calculation"""
    try:
        data = request.json
        
        # Deserialize network
        network = deserialize_network(data['network'])
        
        # Update solver properties
        solver.oil_density = data['system_params']['oil_density']
        solver.oil_type = data['system_params']['oil_type']
        
        # Calculate flow distribution
        total_flow = data['system_params']['total_flow'] / 1000  # Convert L/s to m³/s
        temperature = data['system_params']['temperature']
        
        connection_flows, solution_info = solver.solve_network_flow(
            network, total_flow, temperature
        )
        
        # Prepare response
        results = {
            'success': True,
            'connection_flows': {comp_id: flow * 1000 for comp_id, flow in connection_flows.items()},  # Convert to L/s
            'solution_info': solution_info,
            'total_flow': total_flow * 1000,  # Convert to L/s
            'temperature': temperature
        }
        
        # Generate network visualization
        network_plot = generate_network_plot(network, connection_flows)
        results['network_plot'] = network_plot
        
        # Generate results plots
        results_plots = generate_results_plots(network, connection_flows, solution_info)
        results['results_plots'] = results_plots
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/validate_network', methods=['POST'])
def validate_network():
    """Validate network topology"""
    try:
        data = request.json
        network = deserialize_network(data['network'])
        
        is_valid, errors = network.validate_network()
        
        return jsonify({
            'success': True,
            'is_valid': is_valid,
            'errors': errors
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_node', methods=['POST'])
def add_node():
    """Add a node to the network"""
    try:
        data = request.json
        network = deserialize_network(data['network'])
        
        node = network.create_node(data['name'], elevation=data['elevation'])
        
        network_data = serialize_network(network)
        return jsonify({'success': True, 'network': network_data, 'node_id': node.id})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_component', methods=['POST'])
def add_component():
    """Add a component to the network"""
    try:
        data = request.json
        network = deserialize_network(data['network'])
        
        # Find nodes
        from_node = None
        to_node = None
        for node in network.nodes:
            if node.id == data['from_node_id']:
                from_node = node
            if node.id == data['to_node_id']:
                to_node = node
        
        if not from_node or not to_node:
            return jsonify({'success': False, 'error': 'Nodes not found'})
        
        # Create component
        comp_data = data['component']
        if comp_data['type'] == 'channel':
            component = Channel(
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                length=comp_data['length'],
                roughness=comp_data.get('roughness', 0.00015),
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'connector':
            component = Connector(
                ConnectorType(comp_data['connector_type']),
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'nozzle':
            component = Nozzle(
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                nozzle_type=NozzleType(comp_data['nozzle_type']),
                name=comp_data.get('name')
            )
        
        network.connect_components(from_node, to_node, component)
        
        network_data = serialize_network(network)
        return jsonify({'success': True, 'network': network_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/set_inlet', methods=['POST'])
def set_inlet():
    """Set inlet node"""
    try:
        data = request.json
        network = deserialize_network(data['network'])
        
        # Find node
        node = None
        for n in network.nodes:
            if n.id == data['node_id']:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'})
        
        network.set_inlet(node)
        
        network_data = serialize_network(network)
        return jsonify({'success': True, 'network': network_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/add_outlet', methods=['POST'])
def add_outlet():
    """Add outlet node"""
    try:
        data = request.json
        network = deserialize_network(data['network'])
        
        # Find node
        node = None
        for n in network.nodes:
            if n.id == data['node_id']:
                node = n
                break
        
        if not node:
            return jsonify({'success': False, 'error': 'Node not found'})
        
        network.add_outlet(node)
        
        network_data = serialize_network(network)
        return jsonify({'success': True, 'network': network_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export_results', methods=['POST'])
def export_results():
    """Export results to text file"""
    try:
        data = request.json
        
        # Format results text
        text = format_results_text(data)
        
        # Create file-like object
        output = io.StringIO()
        output.write(text)
        output.seek(0)
        
        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/plain',
            as_attachment=True,
            download_name='network_lubrication_flow_results.txt'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def serialize_network(network):
    """Serialize network to JSON-compatible format"""
    network_data = {
        'name': network.name,
        'nodes': [],
        'connections': [],
        'inlet_node_id': network.inlet_node.id if network.inlet_node else None,
        'outlet_node_ids': [node.id for node in network.outlet_nodes]
    }
    
    # Serialize nodes
    for node in network.nodes:
        node_data = {
            'id': node.id,
            'name': node.name,
            'elevation': node.elevation
        }
        network_data['nodes'].append(node_data)
    
    # Serialize connections and components
    for connection in network.connections:
        comp = connection.component
        comp_data = {
            'type': comp.component_type.value,
            'id': comp.id
        }
        
        if hasattr(comp, 'name') and comp.name:
            comp_data['name'] = comp.name
        
        if comp.component_type == ComponentType.CHANNEL:
            comp_data.update({
                'diameter': comp.diameter * 1000,  # Convert to mm
                'length': comp.length,
                'roughness': comp.roughness * 1000  # Convert to mm
            })
        elif comp.component_type == ComponentType.CONNECTOR:
            comp_data.update({
                'connector_type': comp.connector_type.value,
                'diameter': comp.diameter * 1000,  # Convert to mm
                'diameter_out': comp.diameter_out * 1000  # Convert to mm
            })
        elif comp.component_type == ComponentType.NOZZLE:
            comp_data.update({
                'diameter': comp.diameter * 1000,  # Convert to mm
                'nozzle_type': comp.nozzle_type.value,
                'discharge_coeff': comp.discharge_coeff
            })
        
        connection_data = {
            'from_node_id': connection.from_node.id,
            'to_node_id': connection.to_node.id,
            'component': comp_data
        }
        network_data['connections'].append(connection_data)
    
    return network_data

def deserialize_network(network_data):
    """Deserialize network from JSON format"""
    network = FlowNetwork(network_data['name'])
    
    # Create nodes
    node_map = {}
    for node_data in network_data['nodes']:
        node = network.create_node(node_data['name'], elevation=node_data['elevation'])
        node.id = node_data['id']  # Preserve original ID
        node_map[node_data['id']] = node
    
    # Create connections
    for conn_data in network_data['connections']:
        from_node = node_map[conn_data['from_node_id']]
        to_node = node_map[conn_data['to_node_id']]
        comp_data = conn_data['component']
        
        # Create component based on type
        if comp_data['type'] == 'channel':
            component = Channel(
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                length=comp_data['length'],
                roughness=comp_data['roughness'] / 1000,  # Convert to meters
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'connector':
            component = Connector(
                ConnectorType(comp_data['connector_type']),
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                diameter_out=comp_data.get('diameter_out', comp_data['diameter']) / 1000,  # Convert to meters
                name=comp_data.get('name')
            )
        elif comp_data['type'] == 'nozzle':
            component = Nozzle(
                diameter=comp_data['diameter'] / 1000,  # Convert to meters
                nozzle_type=NozzleType(comp_data['nozzle_type']),
                discharge_coeff=comp_data.get('discharge_coeff'),
                name=comp_data.get('name')
            )
        
        component.id = comp_data['id']  # Preserve original ID
        network.connect_components(from_node, to_node, component)
    
    # Set inlet and outlets
    if network_data.get('inlet_node_id'):
        inlet_node = node_map[network_data['inlet_node_id']]
        network.set_inlet(inlet_node)
    
    for outlet_id in network_data.get('outlet_node_ids', []):
        outlet_node = node_map[outlet_id]
        network.add_outlet(outlet_node)
    
    return network

def generate_network_plot(network, connection_flows=None):
    """Generate network topology visualization"""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        if not network.nodes:
            ax.text(0.5, 0.5, 'No network defined', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
            img_str = plot_to_base64(fig)
            plt.close(fig)
            return img_str
        
        # Simple custom layout algorithm
        pos = {}
        
        if network.inlet_node:
            # Hierarchical layout starting from inlet
            pos[network.inlet_node.id] = (1, 4)
            positioned = {network.inlet_node.id}
            level = 1
            
            while len(positioned) < len(network.nodes):
                current_level_nodes = []
                
                # Find nodes connected to positioned nodes
                for connection in network.connections:
                    if (connection.from_node.id in positioned and 
                        connection.to_node.id not in positioned):
                        current_level_nodes.append(connection.to_node)
                    elif (connection.to_node.id in positioned and 
                          connection.from_node.id not in positioned):
                        current_level_nodes.append(connection.from_node)
                
                # Position nodes at current level
                if current_level_nodes:
                    y_spacing = 6 / max(1, len(current_level_nodes) - 1) if len(current_level_nodes) > 1 else 0
                    y_start = 1 if len(current_level_nodes) > 1 else 4
                    
                    for i, node in enumerate(current_level_nodes):
                        y_pos = y_start + i * y_spacing
                        pos[node.id] = (level * 3 + 1, y_pos)
                        positioned.add(node.id)
                
                level += 1
                if level > 10:  # Prevent infinite loop
                    break
            
            # Position any remaining nodes
            for node in network.nodes:
                if node.id not in pos:
                    pos[node.id] = (level * 3 + 1, 4)
        else:
            # Simple grid layout if no inlet defined
            for i, node in enumerate(network.nodes):
                x = (i % 5) * 2 + 1
                y = (i // 5) * 2 + 1
                pos[node.id] = (x, y)
        
        # Draw connections first
        for connection in network.connections:
            from_id = connection.from_node.id
            to_id = connection.to_node.id
            
            if from_id in pos and to_id in pos:
                x1, y1 = pos[from_id]
                x2, y2 = pos[to_id]
                
                comp = connection.component
                
                # Determine line style based on component type
                if comp.component_type == ComponentType.CHANNEL:
                    color = 'blue'
                    linewidth = 3
                    linestyle = '-'
                elif comp.component_type == ComponentType.CONNECTOR:
                    color = 'purple'
                    linewidth = 4
                    linestyle = '-'
                elif comp.component_type == ComponentType.NOZZLE:
                    color = 'red'
                    linewidth = 2
                    linestyle = '--'
                else:
                    color = 'gray'
                    linewidth = 2
                    linestyle = '-'
                
                # Draw connection line
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                       linestyle=linestyle, alpha=0.7, zorder=1)
                
                # Add flow rate if available
                if connection_flows and comp.id in connection_flows:
                    flow = connection_flows[comp.id] * 1000  # Convert to L/s
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{flow:.1f}", ha='center', va='center', 
                           fontsize=8, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
                
                # Add component type indicator
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                if comp.component_type == ComponentType.CONNECTOR:
                    # Draw small square for connector
                    rect = patches.Rectangle((mid_x-0.15, mid_y-0.15), 0.3, 0.3, 
                                           facecolor=color, alpha=0.8, zorder=2)
                    ax.add_patch(rect)
                elif comp.component_type == ComponentType.NOZZLE:
                    # Draw small triangle for nozzle
                    triangle = patches.RegularPolygon((mid_x, mid_y), 3, 0.15, 
                                                    facecolor=color, alpha=0.8, zorder=2)
                    ax.add_patch(triangle)
        
        # Draw nodes
        for node in network.nodes:
            if node.id in pos:
                x, y = pos[node.id]
                
                # Determine node color and style
                if node == network.inlet_node:
                    color = 'green'
                    marker = 's'  # square
                    size = 200
                elif node in network.outlet_nodes:
                    color = 'red'
                    marker = '^'  # triangle
                    size = 200
                else:
                    color = 'lightblue'
                    marker = 'o'
                    size = 150
                
                # Draw node
                ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', 
                          linewidth=2, zorder=3)
                
                # Add node label
                ax.text(x, y-0.4, node.name, ha='center', va='top', fontsize=9, weight='bold')
                
                # Add elevation info
                ax.text(x, y+0.4, f"h={node.elevation:.1f}m", ha='center', va='bottom', fontsize=8)
        
        # Set axis properties
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Network Topology: {network.name}", fontsize=14, weight='bold')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=3, label='Channel'),
            plt.Line2D([0], [0], color='purple', linewidth=4, label='Connector'),
            plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Nozzle'),
            plt.scatter([], [], c='green', marker='s', s=100, label='Inlet'),
            plt.scatter([], [], c='red', marker='^', s=100, label='Outlet'),
            plt.scatter([], [], c='lightblue', marker='o', s=100, label='Junction')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        img_str = plot_to_base64(fig)
        plt.close(fig)
        return img_str
        
    except Exception as e:
        print(f"Error generating network plot: {e}")
        return None

def generate_results_plots(network, connection_flows, solution_info):
    """Generate results visualization plots"""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Component flows
        component_names = []
        flow_rates = []
        component_types = []
        
        for connection in network.connections:
            comp = connection.component
            flow = connection_flows.get(comp.id, 0) * 1000  # Convert to L/s
            
            comp_name = comp.name if hasattr(comp, 'name') and comp.name else f"{comp.component_type.value}_{comp.id[:8]}"
            component_names.append(comp_name)
            flow_rates.append(flow)
            component_types.append(comp.component_type.value)
        
        # Flow rate bar chart
        colors = {'channel': 'skyblue', 'connector': 'lightcoral', 'nozzle': 'lightgreen'}
        bar_colors = [colors.get(ct, 'gray') for ct in component_types]
        
        bars1 = ax1.bar(range(len(component_names)), flow_rates, color=bar_colors, alpha=0.7)
        ax1.set_ylabel('Flow Rate (L/s)')
        ax1.set_title('Flow Distribution by Component')
        ax1.set_xticks(range(len(component_names)))
        ax1.set_xticklabels(component_names, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars1, flow_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        # Node pressures
        node_names = [node.name for node in network.nodes]
        pressures = [solution_info['node_pressures'].get(node.id, 0) / 1000 for node in network.nodes]  # Convert to kPa
        
        bars2 = ax2.bar(node_names, pressures, color='orange', alpha=0.7)
        ax2.set_ylabel('Pressure (kPa)')
        ax2.set_title('Node Pressures')
        ax2.tick_params(axis='x', rotation=45)
        
        # Flow distribution pie chart
        outlet_flows = []
        outlet_names = []
        
        for outlet in network.outlet_nodes:
            total_flow = 0
            for connection in network.connections:
                if connection.to_node == outlet:
                    total_flow += connection_flows.get(connection.component.id, 0) * 1000
            outlet_flows.append(total_flow)
            outlet_names.append(outlet.name)
        
        if outlet_flows:
            ax3.pie(outlet_flows, labels=outlet_names, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Flow Distribution to Outlets')
        else:
            ax3.text(0.5, 0.5, 'No outlet flows', ha='center', va='center')
            ax3.set_title('Flow Distribution to Outlets')
        
        # Component type summary
        type_flows = {}
        for i, ct in enumerate(component_types):
            if ct not in type_flows:
                type_flows[ct] = 0
            type_flows[ct] += flow_rates[i]
        
        if type_flows:
            ax4.bar(type_flows.keys(), type_flows.values(), 
                   color=[colors.get(ct, 'gray') for ct in type_flows.keys()], alpha=0.7)
            ax4.set_ylabel('Total Flow (L/s)')
            ax4.set_title('Flow by Component Type')
        else:
            ax4.text(0.5, 0.5, 'No components', ha='center', va='center')
            ax4.set_title('Flow by Component Type')
        
        plt.tight_layout()
        
        img_str = plot_to_base64(fig)
        plt.close(fig)
        return img_str
        
    except Exception as e:
        print(f"Error generating results plots: {e}")
        return None

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    return img_str

def format_results_text(data):
    """Format results as text for export"""
    text = "NETWORK LUBRICATION FLOW DISTRIBUTION RESULTS\n"
    text += "=" * 60 + "\n\n"
    
    info = data['solution_info']
    text += f"Network: {data.get('network_name', 'User Network')}\n"
    text += f"Temperature: {info['temperature']:.1f}°C\n"
    text += f"Oil Type: {solver.oil_type}\n"
    text += f"Oil Density: {solver.oil_density:.1f} kg/m³\n"
    text += f"Dynamic Viscosity: {info['viscosity']:.6f} Pa·s\n"
    text += f"Total Flow Rate: {data['total_flow']:.1f} L/s\n"
    text += f"Converged: {info['converged']} (in {info['iterations']} iterations)\n\n"
    
    text += f"{'Component':<20} {'Type':<12} {'Flow Rate':<12} {'Pressure Drop':<15}\n"
    text += f"{'ID':<20} {'':<12} {'(L/s)':<12} {'(Pa)':<15}\n"
    text += "-" * 65 + "\n"
    
    # Display component flows
    for comp_id, flow in data['connection_flows'].items():
        text += f"{comp_id[:20]:<20} {'component':<12} {flow:<12.3f} {'N/A':<15}\n"
    
    text += "-" * 65 + "\n"
    
    # Node pressures
    text += f"\n{'Node':<20} {'Pressure (Pa)':<15} {'Elevation (m)':<12}\n"
    text += "-" * 50 + "\n"
    
    for node_id, pressure in info['node_pressures'].items():
        text += f"{node_id[:20]:<20} {pressure:<15.1f} {'N/A':<12}\n"
    
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=51252, debug=True)