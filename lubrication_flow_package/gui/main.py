import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import PathCollection

from ..solvers.nodal_matrix_solver import NodalMatrixSolver
from ..config.simulation_config import SimulationConfig
from ..network.flow_network import FlowNetwork
from ..components.channel import Channel
from ..components.nozzle import Nozzle
from ..components.connector import Connector

class PropertiesEditor(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.entries = {}

    def show_properties(self, element_id, properties):
        for widget in self.winfo_children():
            widget.destroy()
        self.entries.clear()

        ttk.Label(self, text=f"Properties for {element_id}").pack(pady=5)

        for key, value in properties.items():
            frame = ttk.Frame(self)
            frame.pack(fill=tk.X, padx=5, pady=2)
            label = ttk.Label(frame, text=f"{key}:")
            label.pack(side=tk.LEFT)
            if key == 'type':
                entry = ttk.Combobox(frame, values=["Node", "inlet", "outlet"])
                entry.set(value)
            else:
                entry = ttk.Entry(frame)
                entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries[key] = entry

        save_button = ttk.Button(self, text="Save", command=lambda: self.save_properties(element_id))
        save_button.pack(pady=5)

    def save_properties(self, element_id):
        new_properties = {key: entry.get() for key, entry in self.entries.items()}
        self.app.update_element_properties(element_id, new_properties)

class ComponentDialog(simpledialog.Dialog):
    def __init__(self, parent, title, component_type, properties):
        self.component_type = component_type
        self.properties = properties
        super().__init__(parent, title=title)

    def body(self, master):
        self.entries = {}
        for key, value in self.properties.items():
            ttk.Label(master, text=f"{key}:").grid(row=len(self.entries), sticky=tk.W)
            entry = ttk.Entry(master)
            entry.insert(0, str(value))
            entry.grid(row=len(self.entries), column=1, padx=5, pady=5)
            self.entries[key] = entry
        return self.entries[list(self.properties.keys())[0]]

    def apply(self):
        self.result = {key: float(entry.get()) for key, entry in self.entries.items()}

class NetworkGraph:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.graph = nx.Graph()
        self.figure = plt.figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.graph_canvas, self.parent_frame)
        self.toolbar.update()
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.grid_size = 50
        self.draw_grid()

    def draw_grid(self):
        self.ax.clear()
        self.ax.set_xlim(0, 800)
        self.ax.set_ylim(0, 600)
        self.ax.set_xticks(range(0, 800, self.grid_size))
        self.ax.set_yticks(range(0, 600, self.grid_size))
        self.ax.grid(True)

    def draw_graph(self, app, node_colors=None, edge_colors=None):
        self.draw_grid()
        pos = {node: (data['x'], data['y']) for node, data in self.graph.nodes(data=True)}
        
        if node_colors is None:
            node_colors = [data.get('color', 'skyblue') for node, data in self.graph.nodes(data=True)]
        if edge_colors is None:
            edge_colors = 'gray'
            
        nodes = nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, node_color=node_colors, node_size=700)
        if nodes:
            nodes.set_picker(5)

        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, edge_color=edge_colors)
        nx.draw_networkx_labels(self.graph, pos, ax=self.ax, font_size=10)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=self.ax)
        self.graph_canvas.draw()

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lubrication Flow Calculator")
        self.geometry("1200x800")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        sidebar_frame = ttk.Frame(main_frame, width=350)
        sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        sidebar_frame.pack_propagate(False)

        toolbar = ttk.LabelFrame(sidebar_frame, text="Toolbar")
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        self.add_node_button = ttk.Button(toolbar, text="Add Node", command=self.start_add_node)
        self.add_node_button.pack(side=tk.LEFT, padx=5, pady=5)

        sim_settings_frame = ttk.LabelFrame(sidebar_frame, text="Simulation Settings")
        sim_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.sim_entries = {}
        sim_params = {
            "total_flow_rate": 0.02,
            "temperature": 50,
            "inlet_pressure": 300000
        }
        for name, value in sim_params.items():
            frame = ttk.Frame(sim_settings_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            label = ttk.Label(frame, text=f"{name.replace('_', ' ').title()}:")
            label.pack(side=tk.LEFT)
            entry = ttk.Entry(frame)
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.sim_entries[name] = entry

        elements_frame = ttk.LabelFrame(sidebar_frame, text="Elements")
        elements_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.nodes_listbox = tk.Listbox(elements_frame)
        self.nodes_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.nodes_listbox.bind('<<ListboxSelect>>', self.on_node_select)

        self.components_listbox = tk.Listbox(elements_frame)
        self.components_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.components_listbox.bind('<<ListboxSelect>>', self.on_component_select)

        self.properties_editor = PropertiesEditor(sidebar_frame, self)

        results_frame = ttk.LabelFrame(sidebar_frame, text="Results", height=200)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        results_frame.pack_propagate(False)

        self.results_text = tk.Text(results_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.run_button = ttk.Button(sidebar_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(fill=tk.X, padx=5, pady=5)

        self.network_graph = NetworkGraph(canvas_frame)
        self.node_counter = 0
        self.component_counter = 0
        self.selected_node = None
        self.connecting = False

        self.network_graph.figure.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.network_graph.figure.canvas.mpl_connect('pick_event', self.on_pick)

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
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="crosshair")
        self.figure_canvas_cid = self.network_graph.figure.canvas.mpl_connect('button_press_event', self.add_node_on_click)

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
        
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="")
        self.network_graph.figure.canvas.mpl_disconnect(self.figure_canvas_cid)

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

        min_p, max_p = min(pressures.values()), max(pressures.values())
        norm_p = {node: (p - min_p) / (max_p - min_p) if (max_p - min_p) > 0 else 0.5 for node, p in pressures.items()}
        
        min_f, max_f = min(flows.values()), max(flows.values())
        norm_f = {comp: (f - min_f) / (max_f - min_f) if (max_f - min_f) > 0 else 0.5 for comp, f in flows.items()}

        cmap = plt.get_cmap('viridis')
        node_colors = [cmap(norm_p[node]) for node in self.network_graph.graph.nodes()]
        edge_colors = [cmap(norm_f[data['label']]) for u, v, data in self.network_graph.graph.edges(data=True)]

        self.network_graph.draw_graph(self, node_colors=node_colors, edge_colors=edge_colors)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()