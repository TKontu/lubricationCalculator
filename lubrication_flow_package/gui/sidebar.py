"""
Sidebar for the Lubrication Flow Network GUI.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QComboBox, QFormLayout, 
    QLineEdit, QTextEdit, QLabel, QListWidget, QListWidgetItem
)
from PyQt5.QtCore import pyqtSignal
from typing import Dict

class Sidebar(QWidget):
    """
    Sidebar widget containing simulation controls, solver selection,
    and results display.
    """
    run_simulation_requested = pyqtSignal()
    solver_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(350)
        
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        self._create_widgets()

    def _create_widgets(self):
        """Create all the widgets for the sidebar."""
        # Simulation Settings
        sim_settings_group = QGroupBox("Simulation Settings")
        sim_settings_layout = QFormLayout()
        
        self.sim_entries = {}
        sim_params = {
            "total_flow_rate": "0.02",
            "temperature": "50",
            "inlet_pressure": "300000"
        }
        for name, value in sim_params.items():
            entry = QLineEdit(value)
            self.sim_entries[name] = entry
            sim_settings_layout.addRow(QLabel(f"{name.replace('_', ' ').title()}:"), entry)
            
        sim_settings_group.setLayout(sim_settings_layout)
        self.layout.addWidget(sim_settings_group)

        # Solver Selection
        solver_group = QGroupBox("Solver")
        solver_layout = QVBoxLayout()
        
        self.solver_combo = QComboBox()
        self.solver_combo.currentTextChanged.connect(self.solver_changed)
        solver_layout.addWidget(self.solver_combo)
        solver_group.setLayout(solver_layout)
        self.layout.addWidget(solver_group)

        # Run Button
        self.run_button = QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.run_simulation_requested)
        self.layout.addWidget(self.run_button)

        # Elements
        elements_group = QGroupBox("Elements")
        elements_layout = QVBoxLayout()
        self.nodes_list = QListWidget()
        self.components_list = QListWidget()
        elements_layout.addWidget(QLabel("Nodes:"))
        elements_layout.addWidget(self.nodes_list)
        elements_layout.addWidget(QLabel("Components:"))
        elements_layout.addWidget(self.components_list)
        elements_group.setLayout(elements_layout)
        self.layout.addWidget(elements_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        self.layout.addWidget(results_group)

        self.layout.addStretch()

    def populate_solvers(self, solvers: list[str]):
        """Populates the solver selection dropdown."""
        self.solver_combo.clear()
        self.solver_combo.addItems(solvers)

    def get_simulation_settings(self) -> Dict[str, float]:
        """Returns the current simulation settings from the UI."""
        settings = {}
        for name, entry in self.sim_entries.items():
            try:
                settings[name] = float(entry.text())
            except ValueError:
                settings[name] = 0.0
        return settings

    def display_results(self, results: Dict):
        """Displays the simulation results in the text area."""
        if not results:
            self.results_text.setPlainText("No results to display.")
            return

        text = f"Converged: {results.get('converged', False)}\n"
        text += f"Iterations: {results.get('iterations', 'N/A')}\n"
        text += f"Final Residual: {results.get('final_residual_norm', 'N/A'):.4e}\n\n"
        
        text += "Node Pressures (Pa):\n"
        if results.get('node_pressures'):
            for node, pressure in results.get('node_pressures', {}).items():
                text += f"  {node}: {pressure:,.0f}\n"
            
        text += "\nComponent Flows (m³/s):\n"
        if results.get('component_flows'):
            for comp, flow in results.get('component_flows', {}).items():
                text += f"  {comp}: {flow:.6f}\n"
            
        self.results_text.setPlainText(text)

    def clear_results(self):
        """Clears the results text area."""
        self.results_text.clear()

    def update_element_lists(self, network_config):
        """Updates the node and component lists from the network config."""
        self.nodes_list.clear()
        self.components_list.clear()
        
        if not network_config:
            return
            
        for node in network_config.nodes:
            self.nodes_list.addItem(QListWidgetItem(node['id']))
            
        for component in network_config.components:
            self.components_list.addItem(QListWidgetItem(component['id']))