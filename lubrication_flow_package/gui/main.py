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
            entry = ttk.Entry(frame)
            entry.insert(0, str(value))
            entry.pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.entries[key] = entry

        save_button = ttk.Button(self, text="Save", command=lambda: self.save_properties(element_id))
        save_button.pack(pady=5)

    def save_properties(self, element_id):
        new_properties = {key: entry.get() for key, entry in self.entries.items()}
        self.app.update_element_properties(element_id, new_properties)

class PlotOptionsDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Plot Options")
        
        ttk.Label(master, text="Plot Type:").grid(row=0, sticky=tk.W)
        self.plot_type = tk.StringVar()
        self.plot_type_combo = ttk.Combobox(master, textvariable=self.plot_type,
                                            values=["Bar Chart", "Line Chart"])
        self.plot_type_combo.grid(row=0, column=1, padx=5, pady=5)
        self.plot_type_combo.current(0)

        ttk.Label(master, text="Data to Plot:").grid(row=1, sticky=tk.W)
        self.data_to_plot = tk.StringVar()
        self.data_to_plot_combo = ttk.Combobox(master, textvariable=self.data_to_plot,
                                               values=["Pressure", "Flow Rate"])
        self.data_to_plot_combo.grid(row=1, column=1, padx=5, pady=5)
        self.data_to_plot_combo.current(0)

        return self.plot_type_combo

    def apply(self):
        self.result = {
            "plot_type": self.plot_type.get(),
            "data_to_plot": self.data_to_plot.get()
        }

class ComponentDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Select Component")
        ttk.Label(master, text="Component Type:").grid(row=0, sticky=tk.W)
        self.component_type = tk.StringVar()
        self.component_type_combo = ttk.Combobox(master, textvariable=self.component_type,
                                                 values=["Channel", "Nozzle", "Connector"])
        self.component_type_combo.grid(row=0, column=1, padx=5, pady=5)
        self.component_type_combo.current(0)

        self.properties_frame = ttk.Frame(master)
        self.properties_frame.grid(row=1, columnspan=2, sticky=tk.W)
        self.entries = {}

        self.component_type.trace_add('write', self.update_properties)
        self.update_properties()

        return self.component_type_combo

    def update_properties(self, *args):
        for widget in self.properties_frame.winfo_children():
            widget.destroy()
        self.entries.clear()

        comp_type = self.component_type.get()
        if comp_type == "Channel":
            self.add_property("diameter", 0.1)
            self.add_property("length", 10)
        elif comp_type == "Nozzle":
            self.add_property("diameter", 0.02)
        elif comp_type == "Connector":
            self.add_property("diameter", 0.1)

    def add_property(self, name, default_value):
        frame = ttk.Frame(self.properties_frame)
        frame.pack(fill=tk.X, padx=5, pady=2)
        label = ttk.Label(frame, text=f"{name}:")
        label.pack(side=tk.LEFT)
        entry = ttk.Entry(frame)
        entry.insert(0, str(default_value))
        entry.pack(side=tk.RIGHT)
        self.entries[name] = entry

    def apply(self):
        self.result = {
            "type": self.component_type.get(),
            "properties": {key: float(entry.get()) for key, entry in self.entries.items()}
        }

class NetworkGraph:
    def __init__(self, parent_frame, app):
        self.parent_frame = parent_frame
        self.app = app
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
        self.ax.set_xticks(range(0, 800, self.grid_size))
        self.ax.set_yticks(range(0, 600, self.grid_size))
        self.ax.grid(True)

    def draw_graph(self, node_colors=None, edge_colors=None):
        self.draw_grid()
        pos = {node: (data['x'], data['y']) for node, data in self.graph.nodes(data=True)}
        
        if node_colors is None:
            node_colors = 'skyblue'
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
        self.add_component_button = ttk.Button(toolbar, text="Add Component", command=self.add_component)
        self.add_component_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.plot_button = ttk.Button(toolbar, text="Plot Results", command=self.plot_results, state=tk.DISABLED)
        self.plot_button.pack(side=tk.LEFT, padx=5, pady=5)

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

        self.properties_editor = PropertiesEditor(sidebar_frame, self)

        results_frame = ttk.LabelFrame(sidebar_frame, text="Results", height=200)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        results_frame.pack_propagate(False)

        self.results_text = tk.Text(results_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.run_button = ttk.Button(sidebar_frame, text="Run Simulation", command=self.run_simulation)
        self.run_button.pack(fill=tk.X, padx=5, pady=5)

        self.network_graph = NetworkGraph(canvas_frame, self)
        self.node_counter = 0
        self.component_counter = 0
        self.selected_nodes = []
        self.adding_node = False
        self.figure_canvas_cid = None

    def start_add_node(self):
        self.adding_node = True
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="crosshair")
        self.figure_canvas_cid = self.network_graph.figure.canvas.mpl_connect('button_press_event', self.add_node_on_click)

    def add_node_on_click(self, event):
        if not self.adding_node:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        grid_size = self.network_graph.grid_size
        snapped_x = round(x / grid_size) * grid_size
        snapped_y = round(y / grid_size) * grid_size

        self.node_counter += 1
        node_name = f"Node {self.node_counter}"
        self.network_graph.graph.add_node(node_name, x=snapped_x, y=snapped_y, type='Node', elevation=0.0)
        self.network_graph.draw_graph()
        self.properties_editor.show_properties(node_name, self.network_graph.graph.nodes[node_name])
        
        self.adding_node = False
        self.network_graph.graph_canvas.get_tk_widget().config(cursor="")
        self.network_graph.figure.canvas.mpl_disconnect(self.figure_canvas_cid)

    def add_component(self):
        dialog = ComponentDialog(self)
        if dialog.result:
            self.component_data = dialog.result
            self.selected_nodes = []
            self.figure_canvas_cid = self.network_graph.figure.canvas.mpl_connect('pick_event', self.on_pick)

    def on_pick(self, event):
        artist = event.artist
        if isinstance(artist, PathCollection):
            indices = event.ind
            if not indices.any():
                return
            
            nodes = list(self.network_graph.graph.nodes())
            picked_node = nodes[indices[0]]
            
            self.select_node_for_connection(picked_node)

    def select_node_for_connection(self, node):
        if node not in self.selected_nodes:
            self.selected_nodes.append(node)
        
        if len(self.selected_nodes) == 2:
            self.component_counter += 1
            comp_name = f"{self.component_data['type']} {self.component_counter}"
            self.network_graph.graph.add_edge(self.selected_nodes[0], self.selected_nodes[1], label=comp_name, **self.component_data)
            self.network_graph.draw_graph()
            self.properties_editor.show_properties(comp_name, self.network_graph.graph.edges[self.selected_nodes[0], self.selected_nodes[1]])
            
            self.network_graph.figure.canvas.mpl_disconnect(self.figure_canvas_cid)
            self.selected_nodes = []

    def update_element_properties(self, element_id, properties):
        if element_id in self.network_graph.graph.nodes:
            for key, value in properties.items():
                self.network_graph.graph.nodes[element_id][key] = value
        else:
            for u, v, data in self.network_graph.graph.edges(data=True):
                if data.get('label') == element_id:
                    for key, value in properties.items():
                        self.network_graph.graph.edges[u, v][key] = value
                    break
        self.network_graph.draw_graph()

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
            self.plot_button.config(state=tk.NORMAL)
            
            self.visualize_results()

        except Exception as e:
            messagebox.showerror("Simulation Error", f"An error occurred during simulation:\n{e}")
            self.plot_button.config(state=tk.DISABLED)

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

        self.network_graph.draw_graph(node_colors=node_colors, edge_colors=edge_colors)

    def plot_results(self):
        if not hasattr(self, 'solution_info'):
            return

        dialog = PlotOptionsDialog(self)
        if not dialog.result:
            return

        plot_type = dialog.result['plot_type']
        data_to_plot = dialog.result['data_to_plot']

        plot_window = tk.Toplevel(self)
        plot_window.title("Results Plot")
        fig, ax = plt.subplots(figsize=(8, 6))

        if data_to_plot == "Pressure":
            data = self.solution_info['node_pressures']
            ax.set_ylabel("Pressure (Pa)")
        else:
            data = self.connection_flows
            ax.set_ylabel("Flow Rate (m³/s)")

        names = list(data.keys())
        values = list(data.values())

        if plot_type == "Bar Chart":
            ax.bar(names, values)
        else:
            ax.plot(names, values)

        ax.set_title(f"{data_to_plot} Plot")
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()