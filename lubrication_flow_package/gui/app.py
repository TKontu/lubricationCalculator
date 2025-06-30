import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.collections import PathCollection
import matplotlib.pyplot as plt

from ..solvers.nodal_matrix_solver import NodalMatrixSolver
from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..components.channel import Channel
from ..components.nozzle import Nozzle
from ..components.connector import Connector
from .canvas import NetworkGraph
from .sidebar import Sidebar
from .dialogs import ComponentDialog

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lubrication Flow Calculator")
        self.geometry("1200x800")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.network_graph = NetworkGraph(canvas_frame)
        self.sidebar = Sidebar(main_frame, self)

        self.node_counter = 0
        self.component_counter = 0
        self.selected_node = None
        self.connecting = False
        self.adding_node = False
        self.node_map = {}

        self.network_graph.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.network_graph.figure.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.network_graph.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.network_graph.zoom_to_fit()

    def update_element_lists(self):
        self.nodes_listbox.delete(0, tk.END)
        for node in self.network_graph.graph.nodes():
            self.nodes_listbox.insert(tk.END, node)

        self.components_listbox.delete(0, tk.END)
        for u, v, data in self.network_graph.graph.edges(data=True):
            self.components_listbox.insert(tk.END, data.get('label', f"{u}-{v}"))

    def on_node_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            node_id = event.widget.get(index)
            self.selected_node = node_id
            self.properties_editor.show_properties(node_id, self.network_graph.graph.nodes[node_id])

    def on_component_select(self, event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            label = event.widget.get(index)
            for u, v, data in self.network_graph.graph.edges(data=True):
                if data.get('label') == label:
                    self.properties_editor.show_properties(label, data)
                    break

    def on_canvas_click(self, event):
        if self.adding_node:
            if event.button == 1:
                self.add_node_on_click(event)
            elif event.button == 3:
                self.cancel_add_node()
            return

        if event.button == 3: # Right-click
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                min_dist = float('inf')
                selected_node = None
                for node, data in self.network_graph.graph.nodes(data=True):
                    dist = (data['x'] - x)**2 + (data['y'] - y)**2
                    if dist < min_dist:
                        min_dist = dist
                        selected_node = node
                
                if min_dist < 200: # Only show context menu if a node is clicked
                    self.selected_node = selected_node
                    self.show_context_menu(event)

    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, PathCollection):
            indices = event.ind
            if not indices.any():
                return
            
            nodes = list(self.network_graph.graph.nodes)
            picked_node = nodes[indices[0]]
            
            if self.connecting:
                self.complete_connection(picked_node)
            else:
                self.selected_node = picked_node
                self.properties_editor.show_properties(picked_node, self.network_graph.graph.nodes[picked_node])

    def show_context_menu(self, event):
        context_menu = tk.Menu(self, tearoff=0)
        if self.selected_node:
            node_data = self.network_graph.graph.nodes[self.selected_node]
            context_menu.add_command(label="Set as Inlet", command=lambda: self.set_node_type('inlet'))
            context_menu.add_command(label="Set as Outlet", command=lambda: self.set_node_type('outlet'))
            context_menu.add_command(label="Start Connection", command=self.start_connection)
            if node_data.get('type') == 'outlet':
                context_menu.add_command(label="Add Nozzle", command=self.add_nozzle)
            context_menu.add_separator()
            context_menu.add_command(label="Delete Node", command=self.delete_node)
        
        context_menu.tk_popup(int(event.x), int(event.y))

    def set_node_type(self, node_type):
        if self.selected_node:
            self.network_graph.graph.nodes[self.selected_node]['type'] = node_type
            color = 'red' if node_type == 'inlet' else 'green' if node_type == 'outlet' else 'skyblue'
            self.network_graph.graph.nodes[self.selected_node]['color'] = color
            self.network_graph.draw_graph(self)

    def start_connection(self):
        self.connecting = True

    def complete_connection(self, to_node):
        if self.connecting and self.selected_node and self.selected_node != to_node:
            dialog = ComponentDialog(self, "Channel Properties", "Channel", {"diameter": 0.1, "length": 10})
            if dialog.result:
                self.component_counter += 1
                comp_name = f"Channel {self.component_counter}"
                self.network_graph.graph.add_edge(self.selected_node, to_node, label=comp_name, type="Channel", properties=dialog.result)
                self.network_graph.draw_graph(self)
                self.update_element_lists()
        self.connecting = False

    def add_nozzle(self):
        if self.selected_node:
            dialog = ComponentDialog(self, "Nozzle Properties", "Nozzle", {"diameter": 0.02})
            if dialog.result:
                self.component_counter += 1
                comp_name = f"Nozzle {self.component_counter}"
                self.network_graph.graph.nodes[self.selected_node]['nozzle'] = {"label": comp_name, "properties": dialog.result}
                self.network_graph.graph.nodes[self.selected_node]['color'] = 'purple'
                self.network_graph.draw_graph(self)
                self.update_element_lists()

    def delete_node(self):
        if self.selected_node:
            self.network_graph.graph.remove_node(self.selected_node)
            self.selected_node = None
            self.network_graph.draw_graph(self)
            self.update_element_lists()

    def start_add_node(self):
        self.adding_node = True
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="crosshair")

    def on_key_press(self, event):
        if event.key == 'escape' and self.adding_node:
            self.cancel_add_node()

    def cancel_add_node(self):
        self.adding_node = False
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="")

    def add_node_on_click(self, event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        grid_size = self.network_graph.grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size

        self.node_counter += 1
        node_name = f"Node {self.node_counter}"
        self.network_graph.graph.add_node(node_name, x=snapped_x, y=snapped_y, type='Node', elevation=0.0)
        self.network_graph.draw_graph(self)
        self.update_element_lists()
        
        self.cancel_add_node()

    def update_element_properties(self, element_id, properties):
        if element_id in self.network_graph.graph.nodes:
            for key, value in properties.items():
                try:
                    # Convert to float if possible, otherwise keep as string
                    self.network_graph.graph.nodes[element_id][key] = float(value)
                except (ValueError, TypeError):
                    self.network_graph.graph.nodes[element_id][key] = value
        else:
            for u, v, data in self.network_graph.graph.edges(data=True):
                if data.get('label') == element_id:
                    for key, value in properties.items():
                        try:
                            self.network_graph.graph.edges[u, v][key] = float(value)
                        except (ValueError, TypeError):
                            self.network_graph.graph.edges[u, v][key] = value
                    break
        self.network_graph.draw_graph(self)

    def create_flow_network(self):
        flow_network = FlowNetwork("GUI Network")
        node_map = {}

        for node_id, data in self.network_graph.graph.nodes(data=True):
            node = flow_network.create_node(name=node_id, elevation=float(data.get('elevation', 0.0)))
            node_map[node_id] = node
            if data.get('type') == 'inlet':
                flow_network.set_inlet(node)
            elif data.get('type') == 'outlet':
                flow_network.add_outlet(node)

        for u, v, data in self.network_graph.graph.edges(data=True):
            comp_type = data.get('type')
            comp_name = data.get('label')
            properties = data.get('properties', {})
            component = None
            if comp_type == 'Channel':
                component = Channel(diameter=properties.get('diameter', 0.1), length=properties.get('length', 10), name=comp_name)
            elif comp_type == 'Nozzle':
                component = Nozzle(diameter=properties.get('diameter', 0.02), name=comp_name)
            elif comp_type == 'Connector':
                component = Connector(diameter=properties.get('diameter', 0.1), name=comp_name)
            
            if component:
                flow_network.connect_components(node_map[u], node_map[v], component)
        
        return flow_network

    def run_simulation(self):
        try:
            flow_network = self.create_flow_network()
            
            sim_config_data = {key: float(entry.get()) for key, entry in self.sim_entries.items()}
            sim_config = SimulationConfig(**sim_config_data)
            
            solver = NodalMatrixSolver(
                oil_density=sim_config.oil_density,
                oil_type=sim_config.oil_type
            )
            
            self.connection_flows, self.solution_info = solver.solve_nodal_network(
                flow_network,
                total_flow_rate=sim_config.total_flow_rate,
                temperature=sim_config.temperature,
                inlet_pressure=sim_config.inlet_pressure
            )
            
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                solver.print_results(flow_network, self.connection_flows, self.solution_info)
            results = f.getvalue()
            
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results)
            self.results_text.config(state=tk.DISABLED)
            
            self.visualize_results()

        except Exception as e:
            messagebox.showerror("Simulation Error", f"An error occurred during simulation:\n{e}")

    def visualize_results(self):
        if not hasattr(self, 'solution_info'):
            return

        pressures = self.solution_info['node_pressures']
        flows = self.connection_flows

        # Create a new pressures dictionary with GUI node names as keys
        gui_pressures = {gui_name: pressures[solver_node.id] for gui_name, solver_node in self.node_map.items()}

        min_p, max_p = min(gui_pressures.values()), max(gui_pressures.values())
        norm_p = {node: (p - min_p) / (max_p - min_p) if (max_p - min_p) > 0 else 0.5 for node, p in gui_pressures.items()}
        
        min_f, max_f = min(flows.values()), max(flows.values())
        norm_f = {comp: (f - min_f) / (max_f - min_f) if (max_f - min_f) > 0 else 0.5 for comp, f in flows.items()}

        cmap = plt.get_cmap('viridis')
        node_colors = [cmap(norm_p[node_name]) for node_name in self.network_graph.graph.nodes()]
        edge_colors = [cmap(norm_f[data['label']]) for u, v, data in self.network_graph.graph.edges(data=True)]

        self.network_graph.draw_graph(self, node_colors=node_colors, edge_colors=edge_colors)
