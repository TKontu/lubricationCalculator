import tkinter as tk
from tkinter import ttk
from .dialogs import PropertiesEditor

class Sidebar(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, width=350)
        self.app = app
        self.pack(side=tk.RIGHT, fill=tk.Y, expand=False)
        self.pack_propagate(False)

        self.create_widgets()

    def create_widgets(self):
        toolbar = ttk.LabelFrame(self, text="Toolbar")
        toolbar.pack(fill=tk.X, padx=5, pady=5)

        self.add_node_button = ttk.Button(toolbar, text="Add Node", command=self.app.start_add_node)
        self.add_node_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.zoom_fit_button = ttk.Button(toolbar, text="Zoom to Fit", command=self.app.network_graph.zoom_to_fit)
        self.zoom_fit_button.pack(side=tk.LEFT, padx=5, pady=5)

        sim_settings_frame = ttk.LabelFrame(self, text="Simulation Settings")
        sim_settings_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.app.sim_entries = {}
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
            self.app.sim_entries[name] = entry

        elements_frame = ttk.LabelFrame(self, text="Elements")
        elements_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.app.nodes_listbox = tk.Listbox(elements_frame)
        self.app.nodes_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.app.nodes_listbox.bind('<<ListboxSelect>>', self.app.on_node_select)

        self.app.components_listbox = tk.Listbox(elements_frame)
        self.app.components_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.app.components_listbox.bind('<<ListboxSelect>>', self.app.on_component_select)

        self.app.properties_editor = PropertiesEditor(self, self.app)

        results_frame = ttk.LabelFrame(self, text="Results", height=200)
        results_frame.pack(fill=tk.X, padx=5, pady=5)
        results_frame.pack_propagate(False)

        self.app.results_text = tk.Text(results_frame, wrap=tk.WORD, state=tk.DISABLED)
        self.app.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.run_button = ttk.Button(self, text="Run Simulation", command=self.app.run_simulation)
        self.run_button.pack(fill=tk.X, padx=5, pady=5)
