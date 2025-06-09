#!/usr/bin/env python3
"""
Enhanced Network-Based Lubrication Flow Distribution GUI

A modern, feature-rich graphical user interface for the network-based lubrication flow
distribution calculator with advanced functionality including:

- Modern, responsive UI design with themes
- Enhanced network visualization with zoom/pan
- Drag & drop component building
- Real-time validation and feedback
- Advanced results visualization
- Component library and templates
- Undo/redo functionality
- Keyboard shortcuts
- Context menus
- Auto-save and recent files
- Export capabilities (PDF, PNG, CSV)
- Network analysis tools
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Arrow
import matplotlib.patches as patches
import numpy as np
import json
import uuid
import os
import datetime
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# Import the backend functionality
from network_lubrication_flow_tool import (
    FlowNetwork, NetworkFlowSolver, Node, Connection,
    Channel, Connector, Nozzle,
    ComponentType, ConnectorType, NozzleType
)


@dataclass
class UIState:
    """Manages the UI state for undo/redo functionality"""
    network_data: Dict
    node_positions: Dict
    timestamp: datetime.datetime
    description: str


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


class EnhancedNetworkFlowGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Network-Based Lubrication Flow Calculator")
        self.root.geometry("1600x1000")
        self.root.state('zoomed' if os.name == 'nt' else 'normal')
        
        # Initialize backend
        self.solver = NetworkFlowSolver(oil_density=900.0, oil_type="SAE30")
        self.network = FlowNetwork("New Network")
        self.results = None
        
        # UI state management
        self.selected_nodes = set()
        self.selected_components = set()
        self.clipboard = None
        self.undo_stack = deque(maxlen=50)
        self.redo_stack = deque(maxlen=50)
        
        # Visual elements
        self.node_positions = {}
        self.visual_elements = {}
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        
        # Component library
        self.component_library = ComponentLibrary()
        
        # Recent files
        self.recent_files = []
        self.max_recent_files = 10
        
        # Auto-save
        self.auto_save_enabled = True
        self.auto_save_interval = 300000  # 5 minutes in milliseconds
        
        # Setup GUI
        self.setup_styles()
        self.create_menu_bar()
        self.create_toolbar()
        self.create_main_layout()
        self.create_status_bar()
        self.setup_keyboard_shortcuts()
        self.load_settings()
        
        # Load example network
        self.load_example_network()
        
        # Start auto-save timer
        if self.auto_save_enabled:
            self.root.after(self.auto_save_interval, self.auto_save)
    
    def setup_styles(self):
        """Setup modern styling for the application"""
        style = ttk.Style()
        
        # Configure modern theme
        style.theme_use('clam')
        
        # Custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 12, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 10, 'bold'))
        style.configure('Action.TButton', font=('Segoe UI', 9, 'bold'))
        style.configure('Tool.TButton', font=('Segoe UI', 8))
        
        # Configure colors
        style.configure('Success.TLabel', foreground='green')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Error.TLabel', foreground='red')
    
    def create_menu_bar(self):
        """Create the application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Network", command=self.new_network, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.load_network, accelerator="Ctrl+O")
        file_menu.add_separator()
        
        # Recent files submenu
        self.recent_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_menu)
        
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_network, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_network_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_command(label="Export Network Image...", command=self.export_network_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self.undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self.redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Copy", command=self.copy_selection, accelerator="Ctrl+C")
        edit_menu.add_command(label="Paste", command=self.paste_selection, accelerator="Ctrl+V")
        edit_menu.add_command(label="Delete", command=self.delete_selection, accelerator="Delete")
        edit_menu.add_separator()
        edit_menu.add_command(label="Select All", command=self.select_all, accelerator="Ctrl+A")
        edit_menu.add_command(label="Clear Selection", command=self.clear_selection, accelerator="Escape")
        
        # Network menu
        network_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Network", menu=network_menu)
        network_menu.add_command(label="Validate Network", command=self.validate_network, accelerator="F5")
        network_menu.add_command(label="Calculate Flow", command=self.calculate_flow, accelerator="F9")
        network_menu.add_separator()
        network_menu.add_command(label="Auto Layout", command=self.auto_layout_network)
        network_menu.add_command(label="Reset View", command=self.reset_view)
        network_menu.add_separator()
        network_menu.add_command(label="Network Properties...", command=self.show_network_properties)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Component Library...", command=self.show_component_library)
        tools_menu.add_command(label="Network Templates...", command=self.show_network_templates)
        tools_menu.add_separator()
        tools_menu.add_command(label="Pressure Analysis...", command=self.show_pressure_analysis)
        tools_menu.add_command(label="Flow Analysis...", command=self.show_flow_analysis)
        tools_menu.add_separator()
        tools_menu.add_command(label="Settings...", command=self.show_settings)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        
        # File operations
        file_frame = ttk.LabelFrame(toolbar_frame, text="File")
        file_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(file_frame, text="New", command=self.new_network, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="Open", command=self.load_network, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(file_frame, text="Save", command=self.save_network, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        
        # Edit operations
        edit_frame = ttk.LabelFrame(toolbar_frame, text="Edit")
        edit_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(edit_frame, text="Undo", command=self.undo, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(edit_frame, text="Redo", command=self.redo, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        
        # Network building
        build_frame = ttk.LabelFrame(toolbar_frame, text="Build")
        build_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(build_frame, text="Add Node", command=self.add_node, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(build_frame, text="Channel", command=lambda: self.add_component("channel"), style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(build_frame, text="Connector", command=lambda: self.add_component("connector"), style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(build_frame, text="Nozzle", command=lambda: self.add_component("nozzle"), style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        
        # Analysis
        analysis_frame = ttk.LabelFrame(toolbar_frame, text="Analysis")
        analysis_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(analysis_frame, text="Validate", command=self.validate_network, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(analysis_frame, text="Calculate", command=self.calculate_flow, style='Action.TButton').pack(side=tk.LEFT, padx=1)
        
        # View controls
        view_frame = ttk.LabelFrame(toolbar_frame, text="View")
        view_frame.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(view_frame, text="Zoom In", command=self.zoom_in, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(view_frame, text="Zoom Out", command=self.zoom_out, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
        ttk.Button(view_frame, text="Reset", command=self.reset_view, style='Tool.TButton').pack(side=tk.LEFT, padx=1)
    
    def create_main_layout(self):
        """Create the main application layout"""
        # Create main paned window
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        self.create_control_panel(main_paned)
        
        # Center panel for network visualization
        self.create_network_panel(main_paned)
        
        # Right panel for results and properties
        self.create_results_panel(main_paned)
    
    def create_control_panel(self, parent):
        """Create the control panel"""
        control_frame = ttk.Frame(parent)
        parent.add(control_frame, weight=1)
        
        # System parameters
        sys_frame = ttk.LabelFrame(control_frame, text="System Parameters")
        sys_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a scrollable frame for parameters
        canvas = tk.Canvas(sys_frame, height=200)
        scrollbar = ttk.Scrollbar(sys_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Parameters
        row = 0
        
        # Total flow rate
        ttk.Label(scrollable_frame, text="Total Flow Rate (L/s):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.total_flow_var = tk.DoubleVar(value=15.0)
        flow_entry = ttk.Entry(scrollable_frame, textvariable=self.total_flow_var, width=12)
        flow_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # Temperature
        ttk.Label(scrollable_frame, text="Temperature (°C):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.temperature_var = tk.DoubleVar(value=40.0)
        temp_entry = ttk.Entry(scrollable_frame, textvariable=self.temperature_var, width=12)
        temp_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # Oil type
        ttk.Label(scrollable_frame, text="Oil Type:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.oil_type_var = tk.StringVar(value="SAE30")
        oil_combo = ttk.Combobox(scrollable_frame, textvariable=self.oil_type_var, 
                                values=["SAE10", "SAE20", "SAE30", "SAE40", "SAE50", "SAE60"], width=10)
        oil_combo.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        # Oil density
        ttk.Label(scrollable_frame, text="Oil Density (kg/m³):").grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)
        self.oil_density_var = tk.DoubleVar(value=900.0)
        density_entry = ttk.Entry(scrollable_frame, textvariable=self.oil_density_var, width=12)
        density_entry.grid(row=row, column=1, padx=5, pady=2)
        row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Network operations
        ops_frame = ttk.LabelFrame(control_frame, text="Network Operations")
        ops_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Node operations
        node_frame = ttk.Frame(ops_frame)
        node_frame.pack(fill=tk.X, pady=2)
        ttk.Label(node_frame, text="Nodes:", style='Heading.TLabel').pack(anchor=tk.W)
        
        node_btn_frame = ttk.Frame(node_frame)
        node_btn_frame.pack(fill=tk.X)
        ttk.Button(node_btn_frame, text="Add", command=self.add_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(node_btn_frame, text="Edit", command=self.edit_selected_node).pack(side=tk.LEFT, padx=2)
        ttk.Button(node_btn_frame, text="Delete", command=self.delete_selected_nodes).pack(side=tk.LEFT, padx=2)
        
        # Component operations
        comp_frame = ttk.Frame(ops_frame)
        comp_frame.pack(fill=tk.X, pady=2)
        ttk.Label(comp_frame, text="Components:", style='Heading.TLabel').pack(anchor=tk.W)
        
        comp_btn_frame = ttk.Frame(comp_frame)
        comp_btn_frame.pack(fill=tk.X)
        ttk.Button(comp_btn_frame, text="Channel", command=lambda: self.add_component("channel")).pack(side=tk.LEFT, padx=1)
        ttk.Button(comp_btn_frame, text="Connector", command=lambda: self.add_component("connector")).pack(side=tk.LEFT, padx=1)
        ttk.Button(comp_btn_frame, text="Nozzle", command=lambda: self.add_component("nozzle")).pack(side=tk.LEFT, padx=1)
        
        # Network setup
        setup_frame = ttk.Frame(ops_frame)
        setup_frame.pack(fill=tk.X, pady=2)
        ttk.Label(setup_frame, text="Setup:", style='Heading.TLabel').pack(anchor=tk.W)
        
        setup_btn_frame = ttk.Frame(setup_frame)
        setup_btn_frame.pack(fill=tk.X)
        ttk.Button(setup_btn_frame, text="Set Inlet", command=self.set_inlet).pack(side=tk.LEFT, padx=2)
        ttk.Button(setup_btn_frame, text="Add Outlet", command=self.add_outlet).pack(side=tk.LEFT, padx=2)
        
        # Network information
        info_frame = ttk.LabelFrame(control_frame, text="Network Information")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(info_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(text_frame, height=10, font=('Consolas', 9), wrap=tk.WORD)
        info_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Quick actions
        action_frame = ttk.LabelFrame(control_frame, text="Quick Actions")
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(action_frame, text="Load Example", command=self.load_example_network).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Clear Network", command=self.clear_network).pack(fill=tk.X, pady=1)
        ttk.Button(action_frame, text="Component Library", command=self.show_component_library).pack(fill=tk.X, pady=1)
    
    def create_network_panel(self, parent):
        """Create the network visualization panel"""
        network_frame = ttk.LabelFrame(parent, text="Network Topology")
        parent.add(network_frame, weight=3)
        
        # Create matplotlib figure with enhanced features
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 8))
        self.fig.patch.set_facecolor('white')
        
        # Create canvas with navigation toolbar
        canvas_frame = ttk.Frame(network_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add navigation toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        toolbar.update()
        
        # Bind enhanced mouse events
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.canvas.mpl_connect('scroll_event', self.on_canvas_scroll)
        self.canvas.mpl_connect('key_press_event', self.on_canvas_key)
        
        # Instructions
        instr_frame = ttk.Frame(network_frame)
        instr_frame.pack(fill=tk.X, pady=2)
        
        instructions = [
            "• Left-click: Select nodes/components",
            "• Ctrl+Click: Multi-select",
            "• Right-click: Context menu",
            "• Drag: Move selected items",
            "• Mouse wheel: Zoom"
        ]
        
        for i, instr in enumerate(instructions):
            ttk.Label(instr_frame, text=instr, font=('Arial', 8)).grid(row=i//2, column=i%2, sticky=tk.W, padx=5)
    
    def create_results_panel(self, parent):
        """Create the results and properties panel"""
        results_frame = ttk.Frame(parent)
        parent.add(results_frame, weight=2)
        
        # Create notebook for tabbed interface
        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results tab
        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results")
        
        # Results text with enhanced formatting
        results_text_frame = ttk.Frame(results_tab)
        results_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_text_frame, font=('Consolas', 9), wrap=tk.WORD)
        results_scrollbar = ttk.Scrollbar(results_text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        results_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Properties tab
        properties_tab = ttk.Frame(notebook)
        notebook.add(properties_tab, text="Properties")
        
        self.properties_tree = ttk.Treeview(properties_tab, columns=('Property', 'Value'), show='tree headings')
        self.properties_tree.heading('#0', text='Item')
        self.properties_tree.heading('Property', text='Property')
        self.properties_tree.heading('Value', text='Value')
        
        prop_scrollbar = ttk.Scrollbar(properties_tab, orient=tk.VERTICAL, command=self.properties_tree.yview)
        self.properties_tree.configure(yscrollcommand=prop_scrollbar.set)
        
        self.properties_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        prop_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Analysis tab
        analysis_tab = ttk.Frame(notebook)
        notebook.add(analysis_tab, text="Analysis")
        
        # Create matplotlib figure for analysis charts
        self.analysis_fig, (self.pressure_ax, self.flow_ax) = plt.subplots(2, 1, figsize=(6, 8))
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, analysis_tab)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Log tab
        log_tab = ttk.Frame(notebook)
        notebook.add(log_tab, text="Log")
        
        log_text_frame = ttk.Frame(log_tab)
        log_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.log_text = tk.Text(log_text_frame, font=('Consolas', 8), wrap=tk.WORD)
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Add initial log entry
        self.log_message("Application started", "INFO")
    
    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        
        # Network stats
        self.stats_var = tk.StringVar(value="Nodes: 0 | Components: 0")
        self.stats_label = ttk.Label(status_frame, textvariable=self.stats_var)
        self.stats_label.pack(side=tk.RIGHT, padx=10)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        shortcuts = {
            '<Control-n>': self.new_network,
            '<Control-o>': self.load_network,
            '<Control-s>': self.save_network,
            '<Control-Shift-S>': self.save_network_as,
            '<Control-z>': self.undo,
            '<Control-y>': self.redo,
            '<Control-c>': self.copy_selection,
            '<Control-v>': self.paste_selection,
            '<Control-a>': self.select_all,
            '<Delete>': self.delete_selection,
            '<Escape>': self.clear_selection,
            '<F5>': self.validate_network,
            '<F9>': self.calculate_flow,
            '<Control-q>': self.root.quit
        }
        
        for key, command in shortcuts.items():
            self.root.bind(key, lambda e, cmd=command: cmd())
    
    def log_message(self, message: str, level: str = "INFO"):
        """Add a message to the log"""
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        
        # Update status bar
        self.status_var.set(message)
    
    def update_stats(self):
        """Update network statistics in status bar"""
        node_count = len(self.network.nodes)
        component_count = len(self.network.connections)
        self.stats_var.set(f"Nodes: {node_count} | Components: {component_count}")
    
    # Placeholder methods for new functionality
    def new_network(self):
        """Create a new network"""
        if messagebox.askyesno("New Network", "Create a new network? Unsaved changes will be lost."):
            self.save_state("New Network")
            self.network = FlowNetwork("New Network")
            self.node_positions.clear()
            self.selected_nodes.clear()
            self.selected_components.clear()
            self.results = None
            self.update_network_display()
            self.update_info_display()
            self.update_stats()
            self.log_message("New network created")
    
    def save_network_as(self):
        """Save network with new filename"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.save_network_to_file(filename)
    
    def save_state(self, description: str):
        """Save current state for undo functionality"""
        state = UIState(
            network_data=self.serialize_network(),
            node_positions=self.node_positions.copy(),
            timestamp=datetime.datetime.now(),
            description=description
        )
        self.undo_stack.append(state)
        self.redo_stack.clear()
    
    def undo(self):
        """Undo last action"""
        if self.undo_stack:
            current_state = UIState(
                network_data=self.serialize_network(),
                node_positions=self.node_positions.copy(),
                timestamp=datetime.datetime.now(),
                description="Current state"
            )
            self.redo_stack.append(current_state)
            
            previous_state = self.undo_stack.pop()
            self.restore_state(previous_state)
            self.log_message(f"Undone: {previous_state.description}")
    
    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            current_state = UIState(
                network_data=self.serialize_network(),
                node_positions=self.node_positions.copy(),
                timestamp=datetime.datetime.now(),
                description="Current state"
            )
            self.undo_stack.append(current_state)
            
            next_state = self.redo_stack.pop()
            self.restore_state(next_state)
            self.log_message(f"Redone: {next_state.description}")
    
    def copy_selection(self):
        """Copy selected items to clipboard"""
        if self.selected_nodes or self.selected_components:
            self.clipboard = {
                'nodes': list(self.selected_nodes),
                'components': list(self.selected_components)
            }
            self.log_message(f"Copied {len(self.selected_nodes)} nodes and {len(self.selected_components)} components")
    
    def paste_selection(self):
        """Paste items from clipboard"""
        if self.clipboard:
            # Implementation for pasting
            self.log_message("Paste functionality not yet implemented")
    
    def select_all(self):
        """Select all items in the network"""
        self.selected_nodes = set(self.network.nodes)
        self.selected_components = set(conn.component for conn in self.network.connections)
        self.update_network_display()
        self.log_message(f"Selected all items")
    
    def clear_selection(self):
        """Clear current selection"""
        self.selected_nodes.clear()
        self.selected_components.clear()
        self.update_network_display()
    
    def delete_selection(self):
        """Delete selected items"""
        if self.selected_nodes or self.selected_components:
            if messagebox.askyesno("Delete", "Delete selected items?"):
                self.save_state("Delete selection")
                # Implementation for deletion
                self.log_message("Delete functionality not yet implemented")
    
    def zoom_in(self):
        """Zoom in on the network view"""
        self.zoom_level *= 1.2
        self.update_network_display()
    
    def zoom_out(self):
        """Zoom out of the network view"""
        self.zoom_level /= 1.2
        self.update_network_display()
    
    def reset_view(self):
        """Reset the network view"""
        self.zoom_level = 1.0
        self.pan_offset = [0, 0]
        self.update_network_display()
    
    def auto_save(self):
        """Auto-save the current network"""
        if self.auto_save_enabled and self.network.nodes:
            try:
                auto_save_dir = os.path.join(os.path.expanduser("~"), ".lubrication_calculator")
                os.makedirs(auto_save_dir, exist_ok=True)
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(auto_save_dir, f"autosave_{timestamp}.json")
                
                self.save_network_to_file(filename)
                self.log_message("Auto-saved network", "DEBUG")
            except Exception as e:
                self.log_message(f"Auto-save failed: {str(e)}", "ERROR")
        
        # Schedule next auto-save
        self.root.after(self.auto_save_interval, self.auto_save)
    
    # Enhanced event handlers
    def on_canvas_click(self, event):
        """Enhanced canvas click handler"""
        if event.inaxes != self.ax:
            return
        
        # Convert to data coordinates
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        # Find clicked item
        clicked_node = self.find_node_at_position(x, y)
        clicked_component = self.find_component_at_position(x, y)
        
        if event.button == 1:  # Left click
            if event.key == 'control':  # Multi-select
                if clicked_node:
                    if clicked_node in self.selected_nodes:
                        self.selected_nodes.remove(clicked_node)
                    else:
                        self.selected_nodes.add(clicked_node)
                elif clicked_component:
                    if clicked_component in self.selected_components:
                        self.selected_components.remove(clicked_component)
                    else:
                        self.selected_components.add(clicked_component)
            else:  # Single select
                self.selected_nodes.clear()
                self.selected_components.clear()
                if clicked_node:
                    self.selected_nodes.add(clicked_node)
                elif clicked_component:
                    self.selected_components.add(clicked_component)
        
        elif event.button == 3:  # Right click
            self.show_context_menu(event, clicked_node, clicked_component)
        
        self.update_network_display()
        self.update_properties_display()
    
    def on_canvas_release(self, event):
        """Canvas mouse release handler"""
        pass
    
    def on_canvas_motion(self, event):
        """Enhanced canvas motion handler"""
        if event.inaxes != self.ax:
            return
        
        # Update cursor based on hover
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        
        hovered_node = self.find_node_at_position(x, y)
        hovered_component = self.find_component_at_position(x, y)
        
        if hovered_node or hovered_component:
            self.canvas.get_tk_widget().config(cursor="hand2")
        else:
            self.canvas.get_tk_widget().config(cursor="")
    
    def on_canvas_scroll(self, event):
        """Canvas scroll handler for zooming"""
        if event.inaxes != self.ax:
            return
        
        # Zoom in/out
        if event.step > 0:
            self.zoom_level *= 1.1
        else:
            self.zoom_level /= 1.1
        
        self.update_network_display()
    
    def on_canvas_key(self, event):
        """Canvas key press handler"""
        if event.key == 'delete':
            self.delete_selection()
        elif event.key == 'escape':
            self.clear_selection()
    
    def find_node_at_position(self, x: float, y: float, tolerance: float = 0.3) -> Optional[Node]:
        """Find node at given position"""
        for node in self.network.nodes:
            if node.id in self.node_positions:
                nx, ny = self.node_positions[node.id]
                if abs(nx - x) < tolerance and abs(ny - y) < tolerance:
                    return node
        return None
    
    def find_component_at_position(self, x: float, y: float, tolerance: float = 0.2) -> Optional[Any]:
        """Find component at given position"""
        for connection in self.network.connections:
            from_pos = self.node_positions.get(connection.from_node.id)
            to_pos = self.node_positions.get(connection.to_node.id)
            
            if from_pos and to_pos:
                # Check if point is near the line
                x1, y1 = from_pos
                x2, y2 = to_pos
                
                # Distance from point to line
                A = y2 - y1
                B = x1 - x2
                C = x2 * y1 - x1 * y2
                
                distance = abs(A * x + B * y + C) / math.sqrt(A * A + B * B)
                
                if distance < tolerance:
                    # Check if point is between the endpoints
                    dot_product = (x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)
                    squared_length = (x2 - x1) ** 2 + (y2 - y1) ** 2
                    
                    if 0 <= dot_product <= squared_length:
                        return connection.component
        
        return None
    
    def show_context_menu(self, event, node=None, component=None):
        """Show context menu"""
        context_menu = tk.Menu(self.root, tearoff=0)
        
        if node:
            context_menu.add_command(label=f"Edit {node.name}", command=lambda: self.edit_node(node))
            context_menu.add_command(label=f"Delete {node.name}", command=lambda: self.delete_node(node))
            context_menu.add_separator()
            context_menu.add_command(label="Set as Inlet", command=lambda: self.set_node_as_inlet(node))
            context_menu.add_command(label="Add as Outlet", command=lambda: self.add_node_as_outlet(node))
        elif component:
            context_menu.add_command(label="Edit Component", command=lambda: self.edit_component(component))
            context_menu.add_command(label="Delete Component", command=lambda: self.delete_component(component))
        else:
            context_menu.add_command(label="Add Node Here", command=lambda: self.add_node_at_position(event.xdata, event.ydata))
            context_menu.add_separator()
            context_menu.add_command(label="Paste", command=self.paste_selection, state=tk.NORMAL if self.clipboard else tk.DISABLED)
        
        try:
            context_menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
        finally:
            context_menu.grab_release()
    
    # Placeholder methods for dialogs and advanced features
    def show_component_library(self):
        """Show component library dialog"""
        ComponentLibraryDialog(self.root, self.component_library, self.on_component_selected)
    
    def show_network_templates(self):
        """Show network templates dialog"""
        TemplateDialog(self.root, self.component_library, self.on_template_selected)
    
    def show_pressure_analysis(self):
        """Show pressure analysis dialog"""
        if self.results:
            PressureAnalysisDialog(self.root, self.network, self.results)
        else:
            messagebox.showwarning("Warning", "Please calculate flow first")
    
    def show_flow_analysis(self):
        """Show flow analysis dialog"""
        if self.results:
            FlowAnalysisDialog(self.root, self.network, self.results)
        else:
            messagebox.showwarning("Warning", "Please calculate flow first")
    
    def show_settings(self):
        """Show settings dialog"""
        SettingsDialog(self.root, self)
    
    def show_user_guide(self):
        """Show user guide"""
        UserGuideDialog(self.root)
    
    def show_shortcuts(self):
        """Show keyboard shortcuts"""
        ShortcutsDialog(self.root)
    
    def show_about(self):
        """Show about dialog"""
        AboutDialog(self.root)
    
    def show_network_properties(self):
        """Show network properties dialog"""
        NetworkPropertiesDialog(self.root, self.network)
    
    # Implement core functionality methods (simplified versions for now)
    def add_node(self):
        """Add a new node"""
        dialog = EnhancedNodeDialog(self.root, self.on_node_added)
    
    def add_component(self, component_type):
        """Add a component"""
        if len(self.network.nodes) < 2:
            messagebox.showwarning("Warning", "Need at least 2 nodes to add a component")
            return
        
        dialog = EnhancedComponentDialog(self.root, self.on_component_added, component_type, self.network.nodes)
    
    def set_inlet(self):
        """Set selected node as inlet"""
        if len(self.selected_nodes) == 1:
            node = list(self.selected_nodes)[0]
            self.save_state("Set inlet")
            self.network.set_inlet(node)
            self.update_network_display()
            self.update_info_display()
            self.log_message(f"Set {node.name} as inlet")
        else:
            messagebox.showwarning("Warning", "Please select exactly one node to set as inlet")
    
    def add_outlet(self):
        """Add selected node as outlet"""
        if len(self.selected_nodes) == 1:
            node = list(self.selected_nodes)[0]
            self.save_state("Add outlet")
            self.network.add_outlet(node)
            self.update_network_display()
            self.update_info_display()
            self.log_message(f"Added {node.name} as outlet")
        else:
            messagebox.showwarning("Warning", "Please select exactly one node to add as outlet")
    
    def validate_network(self):
        """Validate the network"""
        is_valid, errors = self.network.validate_network()
        
        if is_valid:
            messagebox.showinfo("Validation", "Network is valid!")
            self.log_message("Network validation passed")
        else:
            error_msg = "Network validation failed:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showerror("Validation Error", error_msg)
            self.log_message("Network validation failed", "ERROR")
    
    def calculate_flow(self):
        """Calculate flow distribution"""
        # Validate network first
        is_valid, errors = self.network.validate_network()
        if not is_valid:
            error_msg = "Cannot calculate: Network validation failed:\n\n" + "\n".join(f"• {error}" for error in errors)
            messagebox.showerror("Validation Error", error_msg)
            return
        
        try:
            self.progress_var.set(10)
            self.root.update()
            
            # Update solver properties
            self.solver.oil_density = self.oil_density_var.get()
            self.solver.oil_type = self.oil_type_var.get()
            
            self.progress_var.set(30)
            self.root.update()
            
            # Calculate flow distribution
            total_flow = self.total_flow_var.get() / 1000  # Convert L/s to m³/s
            temperature = self.temperature_var.get()
            
            self.progress_var.set(50)
            self.root.update()
            
            connection_flows, solution_info = self.solver.solve_network_flow(
                self.network, total_flow, temperature
            )
            
            self.progress_var.set(80)
            self.root.update()
            
            self.results = {
                'connection_flows': connection_flows,
                'solution_info': solution_info,
                'total_flow': total_flow,
                'temperature': temperature
            }
            
            # Display results
            self.display_results()
            self.update_network_display()
            self.update_analysis_charts()
            
            self.progress_var.set(100)
            self.root.update()
            
            self.log_message("Flow calculation completed successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {str(e)}")
            self.log_message(f"Calculation failed: {str(e)}", "ERROR")
        finally:
            self.progress_var.set(0)
    
    def clear_network(self):
        """Clear the current network"""
        if messagebox.askyesno("Clear Network", "Clear the current network? This cannot be undone."):
            self.save_state("Clear network")
            self.network = FlowNetwork("New Network")
            self.node_positions.clear()
            self.selected_nodes.clear()
            self.selected_components.clear()
            self.results = None
            self.update_network_display()
            self.update_info_display()
            self.update_stats()
            self.log_message("Network cleared")
    
    def load_example_network(self):
        """Load an example network"""
        self.save_state("Load example")
        
        # Create a simple example network
        self.network = FlowNetwork("Example Network")
        
        # Create nodes
        inlet = self.network.create_node("Inlet", elevation=0.0)
        junction1 = self.network.create_node("Junction1", elevation=0.0)
        outlet1 = self.network.create_node("Outlet1", elevation=0.0)
        outlet2 = self.network.create_node("Outlet2", elevation=0.0)
        
        # Create components
        main_channel = Channel(diameter=0.050, length=10.0, roughness=0.00015, name="Main Channel")
        branch1 = Channel(diameter=0.025, length=5.0, roughness=0.00015, name="Branch 1")
        branch2 = Channel(diameter=0.025, length=5.0, roughness=0.00015, name="Branch 2")
        
        # Connect components
        self.network.connect_components(inlet, junction1, main_channel)
        self.network.connect_components(junction1, outlet1, branch1)
        self.network.connect_components(junction1, outlet2, branch2)
        
        # Set inlet and outlets
        self.network.set_inlet(inlet)
        self.network.add_outlet(outlet1)
        self.network.add_outlet(outlet2)
        
        # Set positions
        self.node_positions = {
            inlet.id: (1, 4),
            junction1.id: (5, 4),
            outlet1.id: (9, 6),
            outlet2.id: (9, 2)
        }
        
        self.update_network_display()
        self.update_info_display()
        self.update_stats()
        self.log_message("Example network loaded")
    
    # Callback methods
    def on_node_added(self, node_data):
        """Handle node addition"""
        self.save_state("Add node")
        node = self.network.create_node(node_data['name'], elevation=node_data['elevation'])
        
        # Position new node
        if not self.node_positions:
            self.node_positions[node.id] = (1, 4)
        else:
            # Find a good position for the new node
            max_x = max(pos[0] for pos in self.node_positions.values())
            self.node_positions[node.id] = (max_x + 2, 4)
        
        self.update_network_display()
        self.update_info_display()
        self.update_stats()
        self.log_message(f"Added node: {node.name}")
    
    def on_component_added(self, component_data):
        """Handle component addition"""
        self.save_state("Add component")
        self.network.connect_components(
            component_data['from_node'],
            component_data['to_node'],
            component_data['component']
        )
        
        self.update_network_display()
        self.update_info_display()
        self.update_stats()
        self.log_message(f"Added component: {component_data['component'].component_type.value}")
    
    def on_component_selected(self, component_data):
        """Handle component selection from library"""
        # Implementation for component library selection
        pass
    
    def on_template_selected(self, template_data):
        """Handle template selection"""
        # Implementation for template selection
        pass
    
    # Display update methods
    def update_network_display(self):
        """Update the network visualization with enhanced graphics"""
        self.ax.clear()
        
        if not self.network.nodes:
            self.ax.text(0.5, 0.5, 'No network defined\nUse "Load Example" or add nodes to start', 
                        ha='center', va='center', transform=self.ax.transAxes, 
                        fontsize=14, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
            self.canvas.draw()
            return
        
        # Auto-layout nodes if positions not set
        if not self.node_positions:
            self.auto_layout_nodes()
        
        # Draw connections with enhanced styling
        for connection in self.network.connections:
            self.draw_enhanced_connection(connection)
        
        # Draw nodes with enhanced styling
        for node in self.network.nodes:
            self.draw_enhanced_node(node)
        
        # Set axis properties with zoom and pan
        if self.node_positions:
            x_coords = [pos[0] for pos in self.node_positions.values()]
            y_coords = [pos[1] for pos in self.node_positions.values()]
            
            margin = 1.0
            x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
            y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
            
            # Apply zoom and pan
            x_center = (x_min + x_max) / 2 + self.pan_offset[0]
            y_center = (y_min + y_max) / 2 + self.pan_offset[1]
            x_range = (x_max - x_min) / self.zoom_level
            y_range = (y_max - y_min) / self.zoom_level
            
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_title(f"Network: {self.network.name}", fontsize=14, fontweight='bold')
        
        # Add legend
        self.add_network_legend()
        
        self.canvas.draw()
    
    def draw_enhanced_node(self, node):
        """Draw a node with enhanced styling"""
        if node.id not in self.node_positions:
            return
        
        x, y = self.node_positions[node.id]
        
        # Determine node appearance
        if node == self.network.inlet_node:
            color = '#2E8B57'  # Sea green
            marker = 's'
            size = 200
            edge_color = '#1F5F3F'
        elif node in self.network.outlet_nodes:
            color = '#DC143C'  # Crimson
            marker = '^'
            size = 200
            edge_color = '#8B0000'
        else:
            color = '#4169E1'  # Royal blue
            marker = 'o'
            size = 150
            edge_color = '#191970'
        
        # Highlight if selected
        if node in self.selected_nodes:
            edge_color = '#FFD700'  # Gold
            edge_width = 4
        else:
            edge_width = 2
        
        # Draw node
        self.ax.scatter(x, y, c=color, marker=marker, s=size, 
                       edgecolors=edge_color, linewidth=edge_width, zorder=5, alpha=0.9)
        
        # Add node label with background
        self.ax.text(x, y-0.5, node.name, ha='center', va='top', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray'))
        
        # Add elevation info
        if node.elevation != 0:
            self.ax.text(x, y+0.5, f"h={node.elevation:.1f}m", ha='center', va='bottom', 
                        fontsize=8, style='italic', color='darkblue')
        
        # Add pressure info if results available
        if self.results and node.id in self.results['solution_info']['node_pressures']:
            pressure = self.results['solution_info']['node_pressures'][node.id]
            self.ax.text(x+0.6, y, f"P={pressure/1000:.1f}kPa", ha='left', va='center', 
                        fontsize=7, color='purple',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='lavender', alpha=0.7))
    
    def draw_enhanced_connection(self, connection):
        """Draw a connection with enhanced styling"""
        from_pos = self.node_positions.get(connection.from_node.id)
        to_pos = self.node_positions.get(connection.to_node.id)
        
        if not from_pos or not to_pos:
            return
        
        x1, y1 = from_pos
        x2, y2 = to_pos
        
        comp = connection.component
        
        # Determine line style based on component type
        if comp.component_type == ComponentType.CHANNEL:
            color = '#1E90FF'  # Dodger blue
            linewidth = 4
            linestyle = '-'
            alpha = 0.8
        elif comp.component_type == ComponentType.CONNECTOR:
            color = '#9932CC'  # Dark orchid
            linewidth = 5
            linestyle = '-'
            alpha = 0.8
        elif comp.component_type == ComponentType.NOZZLE:
            color = '#FF6347'  # Tomato
            linewidth = 3
            linestyle = '--'
            alpha = 0.8
        else:
            color = 'gray'
            linewidth = 2
            linestyle = '-'
            alpha = 0.6
        
        # Highlight if selected
        if comp in self.selected_components:
            linewidth += 2
            alpha = 1.0
        
        # Draw connection line
        self.ax.plot([x1, x2], [y1, y2], color=color, linewidth=linewidth, 
                    linestyle=linestyle, alpha=alpha, zorder=2)
        
        # Add flow rate if results available
        if self.results and comp.id in self.results['connection_flows']:
            flow = self.results['connection_flows'][comp.id] * 1000  # Convert to L/s
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            # Add flow arrow
            dx, dy = x2 - x1, y2 - y1
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx, dy = dx/length, dy/length
                arrow_length = 0.3
                self.ax.arrow(mid_x - dx*arrow_length/2, mid_y - dy*arrow_length/2,
                             dx*arrow_length, dy*arrow_length,
                             head_width=0.1, head_length=0.1, fc=color, ec=color, alpha=0.8)
            
            # Add flow rate label
            self.ax.text(mid_x, mid_y+0.2, f"{flow:.1f} L/s", ha='center', va='center', 
                        fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor=color))
        
        # Add component type indicator
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        if comp.component_type == ComponentType.CONNECTOR:
            # Draw connector symbol
            self.ax.scatter(mid_x, mid_y, c=color, marker='s', s=50, zorder=4, alpha=0.8)
        elif comp.component_type == ComponentType.NOZZLE:
            # Draw nozzle symbol
            self.ax.scatter(mid_x, mid_y, c=color, marker='>', s=40, zorder=4, alpha=0.8)
    
    def add_network_legend(self):
        """Add a legend to the network display"""
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2E8B57', markersize=10, label='Inlet'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#DC143C', markersize=10, label='Outlet'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#4169E1', markersize=8, label='Junction'),
            plt.Line2D([0], [0], color='#1E90FF', linewidth=3, label='Channel'),
            plt.Line2D([0], [0], color='#9932CC', linewidth=3, label='Connector'),
            plt.Line2D([0], [0], color='#FF6347', linewidth=2, linestyle='--', label='Nozzle')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), 
                      frameon=True, fancybox=True, shadow=True)
    
    def auto_layout_nodes(self):
        """Automatically layout nodes with improved algorithm"""
        if not self.network.nodes:
            return
        
        if self.network.inlet_node:
            # Use hierarchical layout starting from inlet
            self.hierarchical_layout()
        else:
            # Use force-directed layout
            self.force_directed_layout()
    
    def hierarchical_layout(self):
        """Hierarchical layout starting from inlet"""
        if not self.network.inlet_node:
            return
        
        # Start with inlet at origin
        self.node_positions[self.network.inlet_node.id] = (0, 0)
        positioned = {self.network.inlet_node.id}
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for connection in self.network.connections:
            adjacency[connection.from_node.id].append(connection.to_node)
            adjacency[connection.to_node.id].append(connection.from_node)
        
        # BFS to position nodes level by level
        queue = deque([(self.network.inlet_node, 0)])
        level_nodes = defaultdict(list)
        
        while queue:
            node, level = queue.popleft()
            level_nodes[level].append(node)
            
            for neighbor in adjacency[node.id]:
                if neighbor.id not in positioned:
                    positioned.add(neighbor.id)
                    queue.append((neighbor, level + 1))
        
        # Position nodes at each level
        for level, nodes in level_nodes.items():
            if len(nodes) == 1:
                y_positions = [0]
            else:
                y_positions = np.linspace(-len(nodes)/2, len(nodes)/2, len(nodes))
            
            for i, node in enumerate(nodes):
                self.node_positions[node.id] = (level * 3, y_positions[i] * 2)
    
    def force_directed_layout(self):
        """Simple force-directed layout"""
        nodes = list(self.network.nodes)
        n = len(nodes)
        
        if n == 0:
            return
        
        # Initialize random positions
        positions = np.random.rand(n, 2) * 10
        
        # Simple spring-mass simulation
        for iteration in range(100):
            forces = np.zeros((n, 2))
            
            # Repulsive forces between all nodes
            for i in range(n):
                for j in range(i + 1, n):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist > 0:
                        force = diff / (dist ** 3) * 10
                        forces[i] += force
                        forces[j] -= force
            
            # Attractive forces for connected nodes
            for connection in self.network.connections:
                i = nodes.index(connection.from_node)
                j = nodes.index(connection.to_node)
                
                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    force = diff * dist * 0.01
                    forces[i] += force
                    forces[j] -= force
            
            # Update positions
            positions += forces * 0.1
        
        # Store positions
        for i, node in enumerate(nodes):
            self.node_positions[node.id] = tuple(positions[i])
    
    def update_info_display(self):
        """Update the network information display"""
        self.info_text.delete(1.0, tk.END)
        
        info = f"NETWORK INFORMATION\n"
        info += "=" * 40 + "\n\n"
        
        info += f"Network Name: {self.network.name}\n"
        info += f"Total Nodes: {len(self.network.nodes)}\n"
        info += f"Total Components: {len(self.network.connections)}\n\n"
        
        if self.network.inlet_node:
            info += f"Inlet: {self.network.inlet_node.name}\n"
        else:
            info += "Inlet: Not set\n"
        
        if self.network.outlet_nodes:
            info += f"Outlets: {', '.join(node.name for node in self.network.outlet_nodes)}\n"
        else:
            info += "Outlets: None\n"
        
        info += "\n" + "NODES" + "\n" + "-" * 20 + "\n"
        for node in self.network.nodes:
            info += f"{node.name}: elevation={node.elevation:.1f}m\n"
        
        info += "\n" + "COMPONENTS" + "\n" + "-" * 20 + "\n"
        for connection in self.network.connections:
            comp = connection.component
            comp_name = getattr(comp, 'name', f"{comp.component_type.value}_{comp.id[:8]}")
            info += f"{comp_name}: {connection.from_node.name} → {connection.to_node.name}\n"
        
        self.info_text.insert(1.0, info)
    
    def update_properties_display(self):
        """Update the properties display"""
        # Clear existing items
        for item in self.properties_tree.get_children():
            self.properties_tree.delete(item)
        
        # Add selected items properties
        if self.selected_nodes:
            nodes_item = self.properties_tree.insert('', 'end', text='Selected Nodes')
            for node in self.selected_nodes:
                node_item = self.properties_tree.insert(nodes_item, 'end', text=node.name)
                self.properties_tree.insert(node_item, 'end', text='', values=('ID', node.id))
                self.properties_tree.insert(node_item, 'end', text='', values=('Elevation', f"{node.elevation:.2f} m"))
                if self.results and node.id in self.results['solution_info']['node_pressures']:
                    pressure = self.results['solution_info']['node_pressures'][node.id]
                    self.properties_tree.insert(node_item, 'end', text='', values=('Pressure', f"{pressure:.1f} Pa"))
        
        if self.selected_components:
            comps_item = self.properties_tree.insert('', 'end', text='Selected Components')
            for comp in self.selected_components:
                comp_name = getattr(comp, 'name', f"{comp.component_type.value}_{comp.id[:8]}")
                comp_item = self.properties_tree.insert(comps_item, 'end', text=comp_name)
                self.properties_tree.insert(comp_item, 'end', text='', values=('Type', comp.component_type.value))
                self.properties_tree.insert(comp_item, 'end', text='', values=('ID', comp.id))
                
                if hasattr(comp, 'diameter'):
                    self.properties_tree.insert(comp_item, 'end', text='', values=('Diameter', f"{comp.diameter*1000:.1f} mm"))
                if hasattr(comp, 'length'):
                    self.properties_tree.insert(comp_item, 'end', text='', values=('Length', f"{comp.length:.2f} m"))
                if hasattr(comp, 'roughness'):
                    self.properties_tree.insert(comp_item, 'end', text='', values=('Roughness', f"{comp.roughness*1000:.3f} mm"))
                
                if self.results and comp.id in self.results['connection_flows']:
                    flow = self.results['connection_flows'][comp.id] * 1000
                    self.properties_tree.insert(comp_item, 'end', text='', values=('Flow Rate', f"{flow:.2f} L/s"))
    
    def display_results(self):
        """Display calculation results with enhanced formatting"""
        if not self.results:
            return
        
        self.results_text.delete(1.0, tk.END)
        
        # Enhanced results formatting
        text = "NETWORK FLOW DISTRIBUTION RESULTS\n"
        text += "=" * 70 + "\n\n"
        
        info = self.results['solution_info']
        text += f"📊 SYSTEM OVERVIEW\n"
        text += f"{'─' * 50}\n"
        text += f"Network Name:        {self.network.name}\n"
        text += f"Analysis Date:       {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Temperature:         {info['temperature']:.1f}°C\n"
        text += f"Oil Type:            {self.solver.oil_type}\n"
        text += f"Oil Density:         {self.solver.oil_density:.1f} kg/m³\n"
        text += f"Dynamic Viscosity:   {info['viscosity']:.6f} Pa·s\n"
        text += f"Total Flow Rate:     {self.results['total_flow']*1000:.2f} L/s\n"
        text += f"Convergence:         {'✓ Yes' if info['converged'] else '✗ No'} ({info['iterations']} iterations)\n\n"
        
        text += f"🔧 COMPONENT ANALYSIS\n"
        text += f"{'─' * 70}\n"
        text += f"{'Component Name':<25} {'Type':<12} {'Flow Rate':<12} {'Velocity':<12}\n"
        text += f"{'':25} {'':12} {'(L/s)':<12} {'(m/s)':<12}\n"
        text += f"{'─' * 70}\n"
        
        # Component flows with enhanced details
        for connection in self.network.connections:
            comp = connection.component
            flow = self.results['connection_flows'].get(comp.id, 0) * 1000  # Convert to L/s
            
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
        
        for node in self.network.nodes:
            pressure = info['node_pressures'].get(node.id, 0) / 1000  # Convert to kPa
            
            if node == self.network.inlet_node:
                node_type = "Inlet"
            elif node in self.network.outlet_nodes:
                node_type = "Outlet"
            else:
                node_type = "Junction"
            
            text += f"{node.name:<20} {pressure:<15.1f} {node.elevation:<12.1f} {node_type:<10}\n"
        
        text += f"{'─' * 50}\n"
        
        # Summary statistics
        text += f"\n📈 SUMMARY STATISTICS\n"
        text += f"{'─' * 30}\n"
        
        flows = [self.results['connection_flows'].get(conn.component.id, 0) * 1000 
                for conn in self.network.connections]
        pressures = [info['node_pressures'].get(node.id, 0) / 1000 
                    for node in self.network.nodes]
        
        if flows:
            text += f"Flow Rate Range:     {min(flows):.2f} - {max(flows):.2f} L/s\n"
            text += f"Average Flow Rate:   {np.mean(flows):.2f} L/s\n"
        
        if pressures:
            text += f"Pressure Range:      {min(pressures):.1f} - {max(pressures):.1f} kPa\n"
            text += f"Average Pressure:    {np.mean(pressures):.1f} kPa\n"
        
        text += f"Total Outlets:       {len(self.network.outlet_nodes)}\n"
        text += f"Total Components:    {len(self.network.connections)}\n"
        
        self.results_text.insert(1.0, text)
    
    def update_analysis_charts(self):
        """Update the analysis charts"""
        if not self.results:
            return
        
        # Clear previous plots
        self.pressure_ax.clear()
        self.flow_ax.clear()
        
        # Pressure distribution chart
        node_names = [node.name for node in self.network.nodes]
        pressures = [self.results['solution_info']['node_pressures'].get(node.id, 0) / 1000 
                    for node in self.network.nodes]
        
        bars = self.pressure_ax.bar(node_names, pressures, color='steelblue', alpha=0.7)
        self.pressure_ax.set_title('Pressure Distribution', fontweight='bold')
        self.pressure_ax.set_ylabel('Pressure (kPa)')
        self.pressure_ax.tick_params(axis='x', rotation=45)
        
        # Highlight inlet and outlets
        for i, node in enumerate(self.network.nodes):
            if node == self.network.inlet_node:
                bars[i].set_color('green')
            elif node in self.network.outlet_nodes:
                bars[i].set_color('red')
        
        # Flow distribution chart
        comp_names = [getattr(conn.component, 'name', f"{conn.component.component_type.value}_{conn.component.id[:8]}")
                     for conn in self.network.connections]
        flows = [self.results['connection_flows'].get(conn.component.id, 0) * 1000 
                for conn in self.network.connections]
        
        colors = []
        for conn in self.network.connections:
            if conn.component.component_type == ComponentType.CHANNEL:
                colors.append('dodgerblue')
            elif conn.component.component_type == ComponentType.CONNECTOR:
                colors.append('darkorchid')
            elif conn.component.component_type == ComponentType.NOZZLE:
                colors.append('tomato')
            else:
                colors.append('gray')
        
        self.flow_ax.bar(comp_names, flows, color=colors, alpha=0.7)
        self.flow_ax.set_title('Flow Distribution', fontweight='bold')
        self.flow_ax.set_ylabel('Flow Rate (L/s)')
        self.flow_ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout
        self.analysis_fig.tight_layout()
        self.analysis_canvas.draw()
    
    # Placeholder methods for file operations
    def save_network(self):
        """Save the current network"""
        # Implementation for saving
        self.log_message("Save functionality not yet implemented")
    
    def load_network(self):
        """Load a network from file"""
        # Implementation for loading
        self.log_message("Load functionality not yet implemented")
    
    def export_results(self):
        """Export results to file"""
        if not self.results:
            messagebox.showwarning("Warning", "No results to export. Please calculate first.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", "Results exported successfully")
                self.log_message(f"Results exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")
                self.log_message(f"Export failed: {str(e)}", "ERROR")
    
    def export_network_image(self):
        """Export network visualization as image"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")]
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", "Network image exported successfully")
                self.log_message(f"Network image exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export image: {str(e)}")
                self.log_message(f"Image export failed: {str(e)}", "ERROR")
    
    # Placeholder methods for advanced functionality
    def serialize_network(self):
        """Serialize network for undo/redo"""
        # Implementation for network serialization
        return {}
    
    def restore_state(self, state):
        """Restore network state"""
        # Implementation for state restoration
        pass
    
    def load_settings(self):
        """Load application settings"""
        # Implementation for loading settings
        pass
    
    def save_settings(self):
        """Save application settings"""
        # Implementation for saving settings
        pass
    
    def save_network_to_file(self, filename):
        """Save network to specified file"""
        # Implementation for file saving
        pass
    
    def edit_selected_node(self):
        """Edit the selected node"""
        if len(self.selected_nodes) == 1:
            node = list(self.selected_nodes)[0]
            dialog = EnhancedNodeDialog(self.root, self.on_node_edited, node)
        else:
            messagebox.showwarning("Warning", "Please select exactly one node to edit")
    
    def delete_selected_nodes(self):
        """Delete selected nodes"""
        if self.selected_nodes:
            if messagebox.askyesno("Delete Nodes", f"Delete {len(self.selected_nodes)} selected nodes?"):
                self.save_state("Delete nodes")
                # Implementation for node deletion
                self.log_message(f"Deleted {len(self.selected_nodes)} nodes")
    
    def on_node_edited(self, node_data):
        """Handle node editing"""
        # Implementation for node editing
        self.log_message("Node editing not yet implemented")


# Enhanced dialog classes
class EnhancedNodeDialog:
    def __init__(self, parent, callback, node=None):
        self.callback = callback
        self.node = node
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Node Configuration")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
        
        if node:
            self.load_node_data()
    
    def create_widgets(self):
        """Create enhanced dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="Node Configuration", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Form frame
        form_frame = ttk.LabelFrame(main_frame, text="Node Properties")
        form_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Node name
        ttk.Label(form_frame, text="Node Name:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        self.name_var = tk.StringVar()
        name_entry = ttk.Entry(form_frame, textvariable=self.name_var, width=25)
        name_entry.grid(row=0, column=1, padx=10, pady=10)
        name_entry.focus()
        
        # Elevation
        ttk.Label(form_frame, text="Elevation (m):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        self.elevation_var = tk.DoubleVar(value=0.0)
        ttk.Entry(form_frame, textvariable=self.elevation_var, width=25).grid(row=1, column=1, padx=10, pady=10)
        
        # Description
        ttk.Label(form_frame, text="Description:").grid(row=2, column=0, sticky=tk.NW, padx=10, pady=10)
        self.description_text = tk.Text(form_frame, height=4, width=25)
        self.description_text.grid(row=2, column=1, padx=10, pady=10)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="OK", command=self.ok_clicked, style='Action.TButton').pack(side=tk.RIGHT)
    
    def load_node_data(self):
        """Load existing node data"""
        self.name_var.set(self.node.name)
        self.elevation_var.set(self.node.elevation)
    
    def ok_clicked(self):
        """Handle OK button click"""
        try:
            name = self.name_var.get().strip()
            if not name:
                messagebox.showerror("Error", "Node name cannot be empty")
                return
            
            node_data = {
                'name': name,
                'elevation': self.elevation_var.get(),
                'description': self.description_text.get(1.0, tk.END).strip()
            }
            
            self.callback(node_data)
            self.dialog.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")


class EnhancedComponentDialog:
    def __init__(self, parent, callback, component_type, nodes):
        self.callback = callback
        self.component_type = component_type
        self.nodes = nodes
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Add {component_type.title()}")
        self.dialog.geometry("500x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create enhanced component dialog widgets"""
        # Main frame
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text=f"Add {self.component_type.title()}", style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Connection frame
        conn_frame = ttk.LabelFrame(main_frame, text="Connection")
        conn_frame.pack(fill=tk.X, pady=(0, 10))
        
        # From node
        ttk.Label(conn_frame, text="From Node:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.from_node_var = tk.StringVar()
        from_combo = ttk.Combobox(conn_frame, textvariable=self.from_node_var, 
                                 values=[node.name for node in self.nodes], width=20)
        from_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # To node
        ttk.Label(conn_frame, text="To Node:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.to_node_var = tk.StringVar()
        to_combo = ttk.Combobox(conn_frame, textvariable=self.to_node_var, 
                               values=[node.name for node in self.nodes], width=20)
        to_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Component parameters
        params_frame = ttk.LabelFrame(main_frame, text=f"{self.component_type.title()} Parameters")
        params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        if self.component_type == "channel":
            self.create_channel_widgets(params_frame)
        elif self.component_type == "connector":
            self.create_connector_widgets(params_frame)
        elif self.component_type == "nozzle":
            self.create_nozzle_widgets(params_frame)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X)
        
        ttk.Button(btn_frame, text="Cancel", command=self.dialog.destroy).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="Add Component", command=self.ok_clicked, style='Action.TButton').pack(side=tk.RIGHT)
    
    def create_channel_widgets(self, parent):
        """Create channel-specific widgets"""
        # Preset selection
        ttk.Label(parent, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(parent, textvariable=self.preset_var, 
                                   values=list(ComponentLibrary.get_channel_presets().keys()), width=20)
        preset_combo.grid(row=0, column=1, padx=10, pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_selected)
        
        # Parameters
        ttk.Label(parent, text="Diameter (mm):").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.diameter_var = tk.DoubleVar(value=50.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=20).grid(row=1, column=1, padx=10, pady=5)
        
        ttk.Label(parent, text="Length (m):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.length_var = tk.DoubleVar(value=10.0)
        ttk.Entry(parent, textvariable=self.length_var, width=20).grid(row=2, column=1, padx=10, pady=5)
        
        ttk.Label(parent, text="Roughness (mm):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.roughness_var = tk.DoubleVar(value=0.15)
        ttk.Entry(parent, textvariable=self.roughness_var, width=20).grid(row=3, column=1, padx=10, pady=5)
        
        # Name
        ttk.Label(parent, text="Name (optional):").grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.name_var, width=20).grid(row=4, column=1, padx=10, pady=5)
    
    def create_connector_widgets(self, parent):
        """Create connector-specific widgets"""
        # Preset selection
        ttk.Label(parent, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(parent, textvariable=self.preset_var, 
                                   values=list(ComponentLibrary.get_connector_presets().keys()), width=20)
        preset_combo.grid(row=0, column=1, padx=10, pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_selected)
        
        # Type
        ttk.Label(parent, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.connector_type_var = tk.StringVar(value="T_JUNCTION")
        type_combo = ttk.Combobox(parent, textvariable=self.connector_type_var,
                                 values=["T_JUNCTION", "X_JUNCTION", "ELBOW_90", "REDUCER", "STRAIGHT"], width=20)
        type_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Diameter
        ttk.Label(parent, text="Diameter (mm):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.diameter_var = tk.DoubleVar(value=50.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=20).grid(row=2, column=1, padx=10, pady=5)
        
        # Name
        ttk.Label(parent, text="Name (optional):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.name_var, width=20).grid(row=3, column=1, padx=10, pady=5)
    
    def create_nozzle_widgets(self, parent):
        """Create nozzle-specific widgets"""
        # Preset selection
        ttk.Label(parent, text="Preset:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.preset_var = tk.StringVar()
        preset_combo = ttk.Combobox(parent, textvariable=self.preset_var, 
                                   values=list(ComponentLibrary.get_nozzle_presets().keys()), width=20)
        preset_combo.grid(row=0, column=1, padx=10, pady=5)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_selected)
        
        # Type
        ttk.Label(parent, text="Type:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.nozzle_type_var = tk.StringVar(value="SHARP_EDGED")
        type_combo = ttk.Combobox(parent, textvariable=self.nozzle_type_var,
                                 values=["SHARP_EDGED", "ROUNDED", "VENTURI", "FLOW_NOZZLE"], width=20)
        type_combo.grid(row=1, column=1, padx=10, pady=5)
        
        # Diameter
        ttk.Label(parent, text="Diameter (mm):").grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)
        self.diameter_var = tk.DoubleVar(value=8.0)
        ttk.Entry(parent, textvariable=self.diameter_var, width=20).grid(row=2, column=1, padx=10, pady=5)
        
        # Name
        ttk.Label(parent, text="Name (optional):").grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(parent, textvariable=self.name_var, width=20).grid(row=3, column=1, padx=10, pady=5)
    
    def on_preset_selected(self, event):
        """Handle preset selection"""
        preset_name = self.preset_var.get()
        if not preset_name:
            return
        
        if self.component_type == "channel":
            presets = ComponentLibrary.get_channel_presets()
            if preset_name in presets:
                preset = presets[preset_name]
                self.diameter_var.set(preset['diameter'] * 1000)  # Convert to mm
                self.length_var.set(preset['length'])
                self.roughness_var.set(preset['roughness'] * 1000)  # Convert to mm
        elif self.component_type == "connector":
            presets = ComponentLibrary.get_connector_presets()
            if preset_name in presets:
                preset = presets[preset_name]
                self.connector_type_var.set(preset['type'])
                self.diameter_var.set(preset['diameter'] * 1000)  # Convert to mm
        elif self.component_type == "nozzle":
            presets = ComponentLibrary.get_nozzle_presets()
            if preset_name in presets:
                preset = presets[preset_name]
                self.nozzle_type_var.set(preset['type'])
                self.diameter_var.set(preset['diameter'] * 1000)  # Convert to mm
    
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
            name = self.name_var.get().strip() or None
            
            if self.component_type == "channel":
                component = Channel(
                    diameter=self.diameter_var.get() / 1000,  # Convert to meters
                    length=self.length_var.get(),
                    roughness=self.roughness_var.get() / 1000,  # Convert to meters
                    name=name
                )
            elif self.component_type == "connector":
                component = Connector(
                    ConnectorType(self.connector_type_var.get()),
                    diameter=self.diameter_var.get() / 1000,  # Convert to meters
                    name=name
                )
            elif self.component_type == "nozzle":
                component = Nozzle(
                    diameter=self.diameter_var.get() / 1000,  # Convert to meters
                    nozzle_type=NozzleType(self.nozzle_type_var.get()),
                    name=name
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


# Placeholder dialog classes for advanced features
class ComponentLibraryDialog:
    def __init__(self, parent, library, callback):
        self.library = library
        self.callback = callback
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Component Library")
        self.dialog.geometry("600x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Component Library", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class TemplateDialog:
    def __init__(self, parent, library, callback):
        self.library = library
        self.callback = callback
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Network Templates")
        self.dialog.geometry("500x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Network Templates", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class PressureAnalysisDialog:
    def __init__(self, parent, network, results):
        self.network = network
        self.results = results
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Pressure Analysis")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Pressure Analysis", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class FlowAnalysisDialog:
    def __init__(self, parent, network, results):
        self.network = network
        self.results = results
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Flow Analysis")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Flow Analysis", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class SettingsDialog:
    def __init__(self, parent, app):
        self.app = app
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Settings")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Settings", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class UserGuideDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("User Guide")
        self.dialog.geometry("600x500")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="User Guide", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


class ShortcutsDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Keyboard Shortcuts")
        self.dialog.geometry("400x400")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Create shortcuts display
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Keyboard Shortcuts", style='Title.TLabel').pack(pady=(0, 20))
        
        shortcuts = [
            ("File Operations", ""),
            ("Ctrl+N", "New Network"),
            ("Ctrl+O", "Open Network"),
            ("Ctrl+S", "Save Network"),
            ("Ctrl+Shift+S", "Save As"),
            ("", ""),
            ("Edit Operations", ""),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Ctrl+C", "Copy"),
            ("Ctrl+V", "Paste"),
            ("Delete", "Delete Selection"),
            ("Escape", "Clear Selection"),
            ("Ctrl+A", "Select All"),
            ("", ""),
            ("Network Operations", ""),
            ("F5", "Validate Network"),
            ("F9", "Calculate Flow"),
            ("", ""),
            ("Application", ""),
            ("Ctrl+Q", "Exit")
        ]
        
        # Create scrollable text widget
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(text_frame, font=('Consolas', 10), wrap=tk.NONE)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        for shortcut, description in shortcuts:
            if not shortcut and not description:
                text_widget.insert(tk.END, "\n")
            elif not shortcut:
                text_widget.insert(tk.END, f"{description}\n", "header")
            else:
                text_widget.insert(tk.END, f"{shortcut:<20} {description}\n")
        
        text_widget.tag_configure("header", font=('Consolas', 10, 'bold'), foreground='blue')
        text_widget.config(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        ttk.Button(main_frame, text="Close", command=self.dialog.destroy).pack(pady=20)


class AboutDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("About")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        main_frame = ttk.Frame(self.dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ttk.Label(main_frame, text="Enhanced Network Flow Calculator", style='Title.TLabel').pack(pady=10)
        ttk.Label(main_frame, text="Version 2.0", font=('Arial', 12)).pack()
        ttk.Label(main_frame, text="Advanced Lubrication Flow Distribution Analysis", font=('Arial', 10)).pack(pady=10)
        
        info_text = """
A modern, feature-rich application for analyzing
lubrication flow distribution in complex networks.

Features:
• Interactive network building
• Advanced flow calculations
• Real-time visualization
• Component library
• Export capabilities
• Modern user interface

Built with Python, Tkinter, and Matplotlib
        """
        
        ttk.Label(main_frame, text=info_text, justify=tk.CENTER).pack(pady=20)
        ttk.Button(main_frame, text="Close", command=self.dialog.destroy).pack()


class NetworkPropertiesDialog:
    def __init__(self, parent, network):
        self.network = network
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Network Properties")
        self.dialog.geometry("400x300")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Placeholder implementation
        ttk.Label(self.dialog, text="Network Properties", style='Title.TLabel').pack(pady=20)
        ttk.Label(self.dialog, text="Feature coming soon...").pack()
        ttk.Button(self.dialog, text="Close", command=self.dialog.destroy).pack(pady=20)


def main():
    """Main function to run the enhanced GUI application"""
    root = tk.Tk()
    app = EnhancedNetworkFlowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()