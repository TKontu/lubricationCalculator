#!/usr/bin/env python3
"""
Network-Based Lubrication Flow Distribution GUI

An advanced graphical user interface for the network-based lubrication flow
distribution calculator. Features include:
- Interactive network topology building
- Component-based system construction (channels, connectors, nozzles)
- Visual network representation with flow visualization
- Tree-like branching support
- Real-time flow distribution calculation
- Export capabilities for results and network configurations
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np
import json
import uuid
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType
)


class NetworkFlowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Network-Based Lubrication Flow Distribution Calculator")
        self.root.geometry("1400x900")
        
        # Initialize solver
        self.solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
        
        # Data storage
        self.network = FlowNetwork("User Network")
        self.results = None
        self.selected_node = None
        self.selected_component = None
        
        # Visual elements for network display
        self.node_positions = {}
        self.visual_elements = {}
        
        # Create GUI
        self.create_widgets()
        self.load_example_network()
        
    def create_widgets(self):
        """Create the main GUI widgets"""
        
        # Create main frames
        self.create_control_frame()
        self.create_network_frame()
        self.create_results_frame()
        
    def create_control_frame(self):
        """Create the control panel frame"""
        
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # System parameters
        sys_frame = ttk.LabelFrame(control_frame, text="System Parameters")
        sys_frame.pack(fill=tk.X, pady=5)
        
        # Total flow rate
        ttk.Label(sys_frame, text="Total Flow Rate (L/s):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_flow_var = tk.DoubleVar(value=15.0)
        ttk.Entry(sys_frame, textvariable=self.total_flow_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        # Temperature
        ttk.Label(sys_frame, text="Temperature (°C):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.temperature_var = tk.DoubleVar(value=40.0)
        ttk.Entry(sys_frame, textvariable=self.temperature_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # Oil type
        ttk.Label(sys_frame, text="Oil Type:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.oil_type_var = tk.StringVar(value="SAE30")
        oil_combo = ttk.Combobox(sys_frame, textvariable=self.oil_type_var, 
                                values=["SAE10", "SAE20", "SAE30", "SAE40", "SAE50", "SAE60"], width=8)
        oil_combo.grid(row=2, column=1, padx=5, pady=2)
        
        # Oil density
        ttk.Label(sys_frame, text="Oil Density (kg/m³):").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.oil_density_var = tk.DoubleVar(value=900.0)
        ttk.Entry(sys_frame, textvariable=self.oil_density_var, width=10).grid(row=3, column=1, padx=5, pady=2)
        
        # Network building frame
        network_frame = ttk.LabelFrame(control_frame, text="Network Building")
        network_frame.pack(fill=tk.X, pady=5)
        
        # Node operations
        node_frame = ttk.LabelFrame(network_frame, text="Nodes")
        node_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(node_frame, text="Add Node", command=self.add_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(node_frame, text="Edit Node", command=self.edit_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(node_frame, text="Delete Node", command=self.delete_node).pack(side=tk.LEFT, padx=2)
        
        # Component operations
        comp_frame = ttk.LabelFrame(network_frame, text="Components")
        comp_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(comp_frame, text="Add Channel", command=lambda: self.add_component("channel")).pack(side=tk.LEFT, padx=1)
        ttk.Button(comp_frame, text="Add Connector", command=lambda: self.add_component("connector")).pack(side=tk.LEFT, padx=1)
        ttk.Button(comp_frame, text="Add Nozzle", command=lambda: self.add_component("nozzle")).pack(side=tk.LEFT, padx=1)
        
        # Network operations
        net_ops_frame = ttk.LabelFrame(network_frame, text="Network Operations")
        net_ops_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(net_ops_frame, text="Set Inlet", command=self.set_inlet).pack(side=tk.LEFT, padx=2)
        ttk.Button(net_ops_frame, text="Add Outlet", command=self.add_outlet).pack(side=tk.LEFT, padx=2)
        ttk.Button(net_ops_frame, text="Validate", command=self.validate_network).pack(side=tk.LEFT, padx=2)
        
        # Network info
        info_frame = ttk.LabelFrame(control_frame, text="Network Information")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_text = tk.Text(info_frame, height=8, width=35, font=('Courier', 9))
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(action_frame, text="Calculate Flow", command=self.calculate_flow, 
                  style="Accent.TButton").pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Load Example", command=self.load_example_network).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Clear Network", command=self.clear_network).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Save Network", command=self.save_network).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Load Network", command=self.load_network).pack(fill=tk.X, pady=2)
        ttk.Button(action_frame, text="Export Results", command=self.export_results).pack(fill=tk.X, pady=2)
        
    def create_network_frame(self):
        """Create the network visualization frame"""
        
        network_frame = ttk.LabelFrame(self.root, text="Network Topology")
        network_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for network visualization
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, network_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for interaction
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        
        # Instructions
        instr_frame = ttk.Frame(network_frame)
        instr_frame.pack(fill=tk.X, pady=2)
        ttk.Label(instr_frame, text="Click nodes to select • Drag to move • Right-click for options", 
                 font=('Arial', 9)).pack()
        
    def create_results_frame(self):
        """Create the results display frame"""
        
        results_frame = ttk.LabelFrame(self.root, text="Flow Distribution Results")
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results text widget with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(text_frame, font=('Courier', 9))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def add_node(self):
        """Add a new node to the network"""
        NodeDialog(self.root, self.on_node_added)
        
    def edit_node(self):
        """Edit selected node"""
        if not self.selected_node:
            messagebox.showwarning("Warning", "Please select a node to edit")
            return
        
        NodeDialog(self.root, self.on_node_edited, self.selected_node)
        
    def delete_node(self):
        """Delete selected node"""
        if not self.selected_node:
            messagebox.showwarning("Warning", "Please select a node to delete")
            return
            
        if messagebox.askyesno("Confirm", f"Delete node '{self.selected_node.name}'?"):
            # Remove from network
            if self.selected_node in self.network.nodes:
                self.network.nodes.remove(self.selected_node)
            
            # Remove connections involving this node
            connections_to_remove = []
            for conn in self.network.connections:
                if conn.from_node == self.selected_node or conn.to_node == self.selected_node:
                    connections_to_remove.append(conn)
            
            for conn in connections_to_remove:
                self.network.connections.remove(conn)
            
            # Clear selection
            self.selected_node = None
            
            # Update displays
            self.update_network_display()
            self.update_info_display()
    
    def add_component(self, component_type):
        """Add a component between two nodes"""
        if len(self.network.nodes) < 2:
            messagebox.showwarning("Warning", "Need at least 2 nodes to add a component")
            return
        
        ComponentDialog(self.root, self.on_component_added, component_type)
    
    def set_inlet(self):
        """Set selected node as inlet"""
        if not self.selected_node:
            messagebox.showwarning("Warning", "Please select a node to set as inlet")
            return
        
        self.network.set_inlet(self.selected_node)
        self.update_network_display()
        self.update_info_display()
        
    def add_outlet(self):
        """Add selected node as outlet"""
        if not self.selected_node:
            messagebox.showwarning("Warning", "Please select a node to add as outlet")
            return
        
        self.network.add_outlet(self.selected_node)
        self.update_network_display()
        self.update_info_display()
    
    def validate_network(self):
        """Validate the current network"""
        is_valid, errors = self.network.validate_network()
        
        if is_valid:
            messagebox.showinfo("Validation", "Network is valid!")
        else:
            error_msg = "Network validation failed:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showerror("Validation Error", error_msg)
    
    def calculate_flow(self):
        """Calculate flow distribution"""
        # Validate network first
        is_valid, errors = self.network.validate_network()
        if not is_valid:
            error_msg = "Cannot calculate: Network validation failed:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showerror("Validation Error", error_msg)
            return
        
        try:
            # Update solver properties
            self.solver.oil_density = self.oil_density_var.get()
            self.solver.oil_type = self.oil_type_var.get()
            
            # Calculate flow distribution
            total_flow = self.total_flow_var.get() / 1000  # Convert L/s to m³/s
            temperature = self.temperature_var.get()
            
            connection_flows, solution_info = self.solver.solve_network_flow(
                self.network, total_flow, temperature
            )
            
            self.results = {
                'connection_flows': connection_flows,
                'solution_info': solution_info,
                'total_flow': total_flow,
                'temperature': temperature
            }
            
            # Display results
            self.display_results()
            self.update_network_display()  # Update with flow visualization
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
    
    def display_results(self):
        """Display calculation results"""
        if not self.results:
            return
        
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        text = "NETWORK FLOW DISTRIBUTION RESULTS\n"
        text += "=" * 60 + "\n\n"
        
        info = self.results['solution_info']
        text += f"Network: {self.network.name}\n"
        text += f"Temperature: {info['temperature']:.1f}°C\n"
        text += f"Oil Type: {self.solver.oil_type}\n"
        text += f"Oil Density: {self.solver.oil_density:.1f} kg/m³\n"
        text += f"Dynamic Viscosity: {info['viscosity']:.6f} Pa·s\n"
        text += f"Total Flow Rate: {self.results['total_flow']*1000:.1f} L/s\n"
        text += f"Converged: {info['converged']} (in {info['iterations']} iterations)\n\n"
        
        text += f"{'Component':<20} {'Type':<12} {'Flow Rate':<12} {'Pressure Drop':<15}\n"
        text += f"{'Name':<20} {'':<12} {'(L/s)':<12} {'(Pa)':<15}\n"
        text += "-" * 65 + "\n"
        
        # Display component flows
        for connection in self.network.connections:
            comp = connection.component
            flow = self.results['connection_flows'].get(comp.id, 0) * 1000  # Convert to L/s
            
            # Get pressure drop (simplified - would need component-specific calculation)
            pressure_drop = 0  # Placeholder
            
            comp_name = comp.name if hasattr(comp, 'name') and comp.name else f"{comp.component_type.value}_{comp.id[:8]}"
            text += f"{comp_name:<20} {comp.component_type.value:<12} {flow:<12.3f} {pressure_drop:<15.1f}\n"
        
        text += "-" * 65 + "\n"
        
        # Node pressures
        text += f"\n{'Node':<20} {'Pressure (Pa)':<15} {'Elevation (m)':<12}\n"
        text += "-" * 50 + "\n"
        
        for node in self.network.nodes:
            pressure = info['node_pressures'].get(node.id, 0)
            text += f"{node.name:<20} {pressure:<15.1f} {node.elevation:<12.1f}\n"
        
        self.results_text.insert(1.0, text)
    
    def update_network_display(self):
        """Update the network visualization"""
        self.ax.clear()
        
        if not self.network.nodes:
            self.ax.text(0.5, 0.5, 'No network defined\nClick "Load Example" to start', 
                        ha='center', va='center', transform=self.ax.transAxes, fontsize=12)
            self.canvas.draw()
            return
        
        # Auto-layout nodes if positions not set
        if not self.node_positions:
            self.auto_layout_nodes()
        
        # Draw connections first (so they appear behind nodes)
        for connection in self.network.connections:
            self.draw_connection(connection)
        
        # Draw nodes
        for node in self.network.nodes:
            self.draw_node(node)
        
        # Set axis properties
        self.ax.set_xlim(-1, 11)
        self.ax.set_ylim(-1, 8)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title(f"Network: {self.network.name}")
        
        self.canvas.draw()
    
    def auto_layout_nodes(self):
        """Automatically layout nodes in a tree-like structure"""
        if not self.network.nodes:
            return
        
        # Simple layout algorithm
        if self.network.inlet_node:
            # Start with inlet at left
            self.node_positions[self.network.inlet_node.id] = (1, 4)
            
            # Layout other nodes based on network topology
            positioned = {self.network.inlet_node.id}
            level = 2
            
            while len(positioned) < len(self.network.nodes):
                current_level_nodes = []
                
                # Find nodes connected to positioned nodes
                for connection in self.network.connections:
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
                        self.node_positions[node.id] = (level * 2, y_pos)
                        positioned.add(node.id)
                
                level += 1
                if level > 10:  # Prevent infinite loop
                    break
            
            # Position any remaining nodes
            for node in self.network.nodes:
                if node.id not in self.node_positions:
                    self.node_positions[node.id] = (level * 2, 4)
        else:
            # Simple grid layout if no inlet defined
            for i, node in enumerate(self.network.nodes):
                x = (i % 5) * 2 + 1
                y = (i // 5) * 2 + 1
                self.node_positions[node.id] = (x, y)
    
    def draw_node(self, node):
        """Draw a node on the network display"""
        if node.id not in self.node_positions:
            return
        
        x, y = self.node_positions[node.id]
        
        # Determine node color and style
        if node == self.network.inlet_node:
            color = 'green'
            marker = 's'  # square
            size = 150
        elif node in self.network.outlet_nodes:
            color = 'red'
            marker = '^'  # triangle
            size = 150
        elif node == self.selected_node:
            color = 'orange'
            marker = 'o'
            size = 120
        else:
            color = 'lightblue'
            marker = 'o'
            size = 100
        
        # Draw node
        self.ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', linewidth=2, zorder=3)
        
        # Add node label
        self.ax.text(x, y-0.4, node.name, ha='center', va='top', fontsize=8, weight='bold')
        
        # Add elevation info
        self.ax.text(x, y+0.4, f"h={node.elevation:.1f}m", ha='center', va='bottom', fontsize=7)
    
    def draw_connection(self, connection):
        """Draw a connection between nodes"""
        from_pos = self.node_positions.get(connection.from_node.id)
        to_pos = self.node_positions.get(connection.to_node.id)
        
        if not from_pos or not to_pos:
            return
        
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        # Determine line style based on component type
        comp = connection.component
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
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                    linestyle=linestyle, alpha=0.7, zorder=1)
        
        # Add flow rate if results available
        if self.results and comp.id in self.results['connection_flows']:
            flow = self.results['connection_flows'][comp.id] * 1000  # Convert to L/s
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.ax.text(mid_x, mid_y, f"{flow:.1f}", ha='center', va='center', 
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        # Add component type indicator
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if comp.component_type == ComponentType.CONNECTOR:
            # Draw small square for connector
            rect = patches.Rectangle((mid_x-0.1, mid_y-0.1), 0.2, 0.2, 
                                   facecolor=color, alpha=0.8, zorder=2)
            self.ax.add_patch(rect)
        elif comp.component_type == ComponentType.NOZZLE:
            # Draw small triangle for nozzle
            triangle = patches.RegularPolygon((mid_x, mid_y), 3, 0.1, 
                                            facecolor=color, alpha=0.8, zorder=2)
            self.ax.add_patch(triangle)
    
    def on_canvas_click(self, event):
        """Handle mouse clicks on the network canvas"""
        if event.inaxes != self.ax:
            return
        
        # Find closest node
        closest_node = None
        min_distance = float('inf')
        
        for node in self.network.nodes:
            if node.id in self.node_positions:
                x, y = self.node_positions[node.id]
                distance = ((event.xdata - x) ** 2 + (event.ydata - y) ** 2) ** 0.5
                if distance < 0.5 and distance < min_distance:  # Within 0.5 units
                    closest_node = node
                    min_distance = distance
        
        # Update selection
        self.selected_node = closest_node
        self.update_network_display()
        self.update_info_display()
    
    def on_canvas_motion(self, event):
        """Handle mouse motion for dragging nodes"""
        # TODO: Implement node dragging
        pass
    
    def update_info_display(self):
        """Update the network information display"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"Network: {self.network.name}\n"
        info += f"Nodes: {len(self.network.nodes)}\n"
        info += f"Connections: {len(self.network.connections)}\n"
        info += f"Inlet: {self.network.inlet_node.name if self.network.inlet_node else 'None'}\n"
        info += f"Outlets: {len(self.network.outlet_nodes)}\n\n"
        
        if self.selected_node:
            info += f"Selected Node: {self.selected_node.name}\n"
            info += f"Elevation: {self.selected_node.elevation:.1f} m\n"
            
            # Count connections
            incoming = sum(1 for conn in self.network.connections if conn.to_node == self.selected_node)
            outgoing = sum(1 for conn in self.network.connections if conn.from_node == self.selected_node)
            info += f"Incoming: {incoming}, Outgoing: {outgoing}\n"
        
        self.info_text.insert(1.0, info)
    
    def on_node_added(self, node_data):
        """Callback when a node is added"""
        node = self.network.create_node(node_data['name'], elevation=node_data['elevation'])
        
        # Position new node
        if not self.node_positions:
            self.node_positions[node.id] = (5, 4)
        else:
            # Position near other nodes
            max_x = max(pos[0] for pos in self.node_positions.values())
            self.node_positions[node.id] = (max_x + 2, 4)
        
        self.update_network_display()
        self.update_info_display()
    
    def on_node_edited(self, node_data):
        """Callback when a node is edited"""
        if self.selected_node:
            self.selected_node.name = node_data['name']
            self.selected_node.elevation = node_data['elevation']
            self.update_network_display()
            self.update_info_display()
    
    def on_component_added(self, component_data):
        """Callback when a component is added"""
        from_node = component_data['from_node']
        to_node = component_data['to_node']
        component = component_data['component']
        
        self.network.connect_components(from_node, to_node, component)
        self.update_network_display()
        self.update_info_display()
    
    def load_example_network(self):
        """Load an example network"""
        self.clear_network()
        
        # Create simple tree example
        inlet = self.network.create_node("Inlet", elevation=0.0)
        junction = self.network.create_node("Junction", elevation=1.0)
        outlet1 = self.network.create_node("Outlet1", elevation=2.0)
        outlet2 = self.network.create_node("Outlet2", elevation=1.5)
        
        self.network.set_inlet(inlet)
        self.network.add_outlet(outlet1)
        self.network.add_outlet(outlet2)
        
        # Create components
        main_channel = Channel(diameter=0.08, length=10.0, name="Main_Channel")
        branch1 = Channel(diameter=0.05, length=8.0, name="Branch1")
        branch2 = Channel(diameter=0.04, length=6.0, name="Branch2")
        nozzle1 = Nozzle(diameter=0.012, nozzle_type=NozzleType.VENTURI, name="Nozzle1")
        nozzle2 = Nozzle(diameter=0.008, nozzle_type=NozzleType.SHARP_EDGED, name="Nozzle2")
        
        # Connect components
        self.network.connect_components(inlet, junction, main_channel)
        self.network.connect_components(junction, outlet1, branch1)
        self.network.connect_components(junction, outlet2, branch2)
        
        # Add nozzles at outlets
        final1 = self.network.create_node("Final1", elevation=2.0)
        final2 = self.network.create_node("Final2", elevation=1.5)
        self.network.connect_components(outlet1, final1, nozzle1)
        self.network.connect_components(outlet2, final2, nozzle2)
        
        # Update outlets
        self.network.outlet_nodes = [final1, final2]
        
        # Set positions
        self.node_positions = {
            inlet.id: (1, 4),
            junction.id: (4, 4),
            outlet1.id: (7, 5.5),
            outlet2.id: (7, 2.5),
            final1.id: (9, 5.5),
            final2.id: (9, 2.5)
        }
        
        self.update_network_display()
        self.update_info_display()
    
    def clear_network(self):
        """Clear the current network"""
        self.network = FlowNetwork("User Network")
        self.node_positions = {}
        self.selected_node = None
        self.results = None
        self.update_network_display()
        self.update_info_display()
        self.results_text.delete(1.0, tk.END)
    
    def save_network(self):
        """Save current network configuration"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                # Serialize network data
                network_data = {
                    'name': self.network.name,
                    'system_params': {
                        'total_flow': self.total_flow_var.get(),
                        'temperature': self.temperature_var.get(),
                        'oil_type': self.oil_type_var.get(),
                        'oil_density': self.oil_density_var.get()
                    },
                    'nodes': [],
                    'connections': [],
                    'inlet_node_id': self.network.inlet_node.id if self.network.inlet_node else None,
                    'outlet_node_ids': [node.id for node in self.network.outlet_nodes],
                    'node_positions': self.node_positions
                }
                
                # Serialize nodes
                for node in self.network.nodes:
                    node_data = {
                        'id': node.id,
                        'name': node.name,
                        'elevation': node.elevation
                    }
                    network_data['nodes'].append(node_data)
                
                # Serialize connections and components
                for connection in self.network.connections:
                    comp = connection.component
                    comp_data = {
                        'type': comp.component_type.value,
                        'id': comp.id
                    }
                    
                    if hasattr(comp, 'name'):
                        comp_data['name'] = comp.name
                    
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
                            'diameter_out': comp.diameter_out
                        })
                    elif comp.component_type == ComponentType.NOZZLE:
                        comp_data.update({
                            'diameter': comp.diameter,
                            'nozzle_type': comp.nozzle_type.value,
                            'discharge_coeff': comp.discharge_coeff
                        })
                    
                    connection_data = {
                        'from_node_id': connection.from_node.id,
                        'to_node_id': connection.to_node.id,
                        'component': comp_data
                    }
                    network_data['connections'].append(connection_data)
                
                with open(filename, 'w') as f:
                    json.dump(network_data, f, indent=2)
                
                messagebox.showinfo("Success", "Network saved successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save network: {str(e)}")
    
    def load_network(self):
        """Load network configuration"""
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    network_data = json.load(f)
                
                # Clear current network
                self.clear_network()
                
                # Load system parameters
                if 'system_params' in network_data:
                    params = network_data['system_params']
                    self.total_flow_var.set(params.get('total_flow', 15.0))
                    self.temperature_var.set(params.get('temperature', 40.0))
                    self.oil_type_var.set(params.get('oil_type', 'SAE30'))
                    self.oil_density_var.set(params.get('oil_density', 900.0))
                
                # Create network
                self.network.name = network_data.get('name', 'Loaded Network')
                
                # Create nodes
                node_map = {}
                for node_data in network_data['nodes']:
                    node = self.network.create_node(node_data['name'], elevation=node_data['elevation'])
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
                            discharge_coeff=comp_data.get('discharge_coeff'),
                            name=comp_data.get('name')
                        )
                    
                    component.id = comp_data['id']  # Preserve original ID
                    self.network.connect_components(from_node, to_node, component)
                
                # Set inlet and outlets
                if network_data.get('inlet_node_id'):
                    inlet_node = node_map[network_data['inlet_node_id']]
                    self.network.set_inlet(inlet_node)
                
                for outlet_id in network_data.get('outlet_node_ids', []):
                    outlet_node = node_map[outlet_id]
                    self.network.add_outlet(outlet_node)
                
                # Load node positions
                self.node_positions = network_data.get('node_positions', {})
                
                self.update_network_display()
                self.update_info_display()
                messagebox.showinfo("Success", "Network loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load network: {str(e)}")
    
    def export_results(self):
        """Export calculation results"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Please calculate first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results exported successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")


class NodeDialog:
    def __init__(self, parent, callback, node=None):
        self.callback = callback
        self.node = node
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Node Configuration")
        self.dialog.geometry("300x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
        
        if node:
            self.load_node_data()
    
    def create_widgets(self):
        """Create dialog widgets"""
        
        # Node name
        ttk.Label(self.dialog, text="Node Name:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(self.dialog, textvariable=self.name_var, width=20).grid(row=0, column=1, padx=10, pady=5)
        
        # Elevation
        ttk.Label(self.dialog, text="Elevation (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.elevation_var = tk.DoubleVar(value=0.0)
        ttk.Entry(self.dialog, textvariable=self.elevation_var, width=20).grid(row=1, column=1, padx=10, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def load_node_data(self):
        """Load existing node data"""
        self.name_var.set(self.node.name)
        self.elevation_var.set(self.node.elevation)
    
    def ok_clicked(self):
        """Handle OK button click"""
        try:
            node_data = {
                'name': self.name_var.get(),
                'elevation': self.elevation_var.get()
            }
            
            self.callback(node_data)
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")


class ComponentDialog:
    def __init__(self, parent, callback, component_type):
        self.callback = callback
        self.component_type = component_type
        
        # Create dialog window
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Add {component_type.title()}")
        self.dialog.geometry("400x350")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Get available nodes
        self.nodes = parent.master.network.nodes if hasattr(parent, 'master') else []
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create dialog widgets"""
        
        # Node selection
        ttk.Label(self.dialog, text="From Node:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.from_node_var = tk.StringVar()
        from_combo = ttk.Combobox(self.dialog, textvariable=self.from_node_var, 
                                 values=[node.name for node in self.nodes], width=18)
        from_combo.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(self.dialog, text="To Node:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.to_node_var = tk.StringVar()
        to_combo = ttk.Combobox(self.dialog, textvariable=self.to_node_var, 
                               values=[node.name for node in self.nodes], width=18)
        to_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Component parameters frame
        comp_frame = ttk.LabelFrame(self.dialog, text=f"{self.component_type.title()} Parameters")
        comp_frame.grid(row=2, column=0, columnspan=2, sticky=tk.EW, padx=10, pady=10)
        
        if self.component_type == "channel":
            self.create_channel_widgets(comp_frame)
        elif self.component_type == "connector":
            self.create_connector_widgets(comp_frame)
        elif self.component_type == "nozzle":
            self.create_nozzle_widgets(comp_frame)
        
        # Buttons
        btn_frame = ttk.Frame(self.dialog)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.LEFT, padx=5)
    
    def create_channel_widgets(self, parent):
        """Create channel-specific widgets"""
        ttk.Label(parent, text="Diameter (mm):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.diameter_var = tk.DoubleVar(value=50.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=15).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(parent, text="Length (m):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.length_var = tk.DoubleVar(value=10.0)
        ttk.Entry(parent, textvariable=self.length_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(parent, text="Roughness (mm):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.roughness_var = tk.DoubleVar(value=0.15)
        ttk.Entry(parent, textvariable=self.roughness_var, width=15).grid(row=2, column=1, padx=5, pady=2)
    
    def create_connector_widgets(self, parent):
        """Create connector-specific widgets"""
        ttk.Label(parent, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.connector_type_var = tk.StringVar(value="T_JUNCTION")
        type_combo = ttk.Combobox(parent, textvariable=self.connector_type_var,
                                 values=["T_JUNCTION", "X_JUNCTION", "ELBOW_90", "REDUCER", "STRAIGHT"], width=12)
        type_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(parent, text="Diameter (mm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.diameter_var = tk.DoubleVar(value=50.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=15).grid(row=1, column=1, padx=5, pady=2)
    
    def create_nozzle_widgets(self, parent):
        """Create nozzle-specific widgets"""
        ttk.Label(parent, text="Type:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.nozzle_type_var = tk.StringVar(value="SHARP_EDGED")
        type_combo = ttk.Combobox(parent, textvariable=self.nozzle_type_var,
                                 values=["SHARP_EDGED", "ROUNDED", "VENTURI", "FLOW_NOZZLE"], width=12)
        type_combo.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(parent, text="Diameter (mm):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.diameter_var = tk.DoubleVar(value=8.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=15).grid(row=1, column=1, padx=5, pady=2)
    
    def ok_clicked(self):
        """Handle OK button click"""
        try:
            # Find selected nodes
            from_node = None
            to_node = None
            
            for node in self.nodes:
                if node.name == self.from_node_var.get():
                    from_node = node
                if node.name == self.to_node_var.get():
                    to_node = node
            
            if not from_node or not to_node:
                messagebox.showerror("Error", "Please select both from and to nodes")
                return
            
            if from_node == to_node:
                messagebox.showerror("Error", "From and to nodes must be different")
                return
            
            # Create component
            if self.component_type == "channel":
                component = Channel(
                    diameter=self.diameter_var.get() / 1000,  # Convert to meters
                    length=self.length_var.get(),
                    roughness=self.roughness_var.get() / 1000  # Convert to meters
                )
            elif self.component_type == "connector":
                component = Connector(
                    ConnectorType(self.connector_type_var.get()),
                    diameter=self.diameter_var.get() / 1000  # Convert to meters
                )
            elif self.component_type == "nozzle":
                component = Nozzle(
                    diameter=self.diameter_var.get() / 1000,  # Convert to meters
                    nozzle_type=NozzleType(self.nozzle_type_var.get())
                )
            
            component_data = {
                'from_node': from_node,
                'to_node': to_node,
                'component': component
            }
            
            self.callback(component_data)
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")


def main():
    """Main function to run the GUI application"""
    root = tk.Tk()
    app = NetworkFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()