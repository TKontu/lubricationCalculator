"""
Sidebar for the Lubrication Flow Network GUI.
"""
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QPushButton, QComboBox, QFormLayout, 
    QLineEdit, QTextEdit, QLabel, QListWidget, QListWidgetItem, QHBoxLayout,
    QMenu
)
from PyQt5.QtCore import pyqtSignal, Qt
from typing import Dict

class Sidebar(QWidget):
    """
    Sidebar widget containing simulation controls, solver selection,
    and results display.
    """
    run_simulation_requested = pyqtSignal()
    solver_changed = pyqtSignal(str)
    add_node_requested = pyqtSignal()
    delete_node_requested = pyqtSignal(str)
    add_connection_requested = pyqtSignal()
    edit_node_requested = pyqtSignal(str)
    edit_component_requested = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(350)
        
        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        
        self._create_widgets()
        self.controller = None
        self.canvas = None

    def set_controller_and_canvas(self, controller, canvas):
        self.controller = controller
        self.canvas = canvas

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

        # Node editing buttons
        node_button_layout = QHBoxLayout()
        self.add_node_button = QPushButton("Add Node")
        self.add_node_button.clicked.connect(self.add_node_requested)
        self.delete_node_button = QPushButton("Delete Node")
        self.delete_node_button.clicked.connect(self.on_delete_node)
        node_button_layout.addWidget(self.add_node_button)
        node_button_layout.addWidget(self.delete_node_button)
        elements_layout.addLayout(node_button_layout)
        
        self.nodes_list = QListWidget()
        self.nodes_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.nodes_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.nodes_list.customContextMenuRequested.connect(self.show_node_context_menu)
        elements_layout.addWidget(QLabel("Nodes:"))
        elements_layout.addWidget(self.nodes_list)

        # Component editing buttons
        component_button_layout = QHBoxLayout()
        self.add_connection_button = QPushButton("Add Connection")
        self.add_connection_button.clicked.connect(self.add_connection_requested)
        self.delete_component_button = QPushButton("Delete Component")
        # self.delete_component_button.clicked.connect(self.on_delete_component) # To be implemented
        component_button_layout.addWidget(self.add_connection_button)
        component_button_layout.addWidget(self.delete_component_button)
        elements_layout.addLayout(component_button_layout)

        self.components_list = QListWidget()
        self.components_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.components_list.customContextMenuRequested.connect(self.show_component_context_menu)
        elements_layout.addWidget(QLabel("Components:"))
        elements_layout.addWidget(self.components_list)
        elements_group.setLayout(elements_layout)
        self.layout.addWidget(elements_group)

        # Properties Editor
        self.properties_group = QGroupBox("Properties")
        self.properties_layout = QFormLayout()
        self.properties_group.setLayout(self.properties_layout)
        self.layout.addWidget(self.properties_group)
        self.properties_group.setVisible(False) # Initially hidden

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        results_group.setLayout(results_layout)
        self.layout.addWidget(results_group)

        self.layout.addStretch()

        # Connect signals for property editor
        self.nodes_list.itemSelectionChanged.connect(self.on_element_selection_changed)
        self.components_list.itemSelectionChanged.connect(self.on_element_selection_changed)

    def show_node_context_menu(self, position):
        """Shows a context menu for nodes."""
        item = self.nodes_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        edit_action = menu.addAction("Edit Properties")
        action = menu.exec_(self.nodes_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_node_requested.emit(item.text())

    def show_component_context_menu(self, position):
        """Shows a context menu for components."""
        item = self.components_list.itemAt(position)
        if not item:
            return
            
        menu = QMenu()
        edit_action = menu.addAction("Edit Properties")
        action = menu.exec_(self.components_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_component_requested.emit(item.text())

    def on_delete_node(self):
        """Emits a signal with the ID of the selected node to be deleted."""
        selected_item = self.nodes_list.currentItem()
        if selected_item:
            self.delete_node_requested.emit(selected_item.text())

    def on_element_selection_changed(self):
        """Update the properties editor when a node or component is selected."""
        sender = self.sender()
        
        # Deselect items in the other list
        if sender == self.nodes_list:
            self.components_list.blockSignals(True)
            self.components_list.clearSelection()
            self.components_list.blockSignals(False)
        else:
            self.nodes_list.blockSignals(True)
            self.nodes_list.clearSelection()
            self.nodes_list.blockSignals(False)

        selected_items = sender.selectedItems()
        if not selected_items:
            self.clear_properties()
            self.properties_group.setVisible(False)
            return

        item = selected_items[0]
        item_id = item.text()
        
        if not self.controller or not self.controller.network_config:
            return

        data = None
        item_type = None
        if sender == self.nodes_list:
            item_type = 'node'
            for n in self.controller.network_config.nodes:
                if n['id'] == item_id:
                    data = n
                    break
        else: # components_list
            item_type = 'component'
            for c in self.controller.network_config.components:
                if c['id'] == item_id:
                    data = c
                    break
        
        if data:
            self.display_properties(data, item_type)
            self.properties_group.setVisible(True)

    def clear_properties(self):
        """Clears the properties editor form."""
        while self.properties_layout.count():
            child = self.properties_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def display_properties(self, data: dict, item_type: str):
        """Populates the properties editor with the data of the selected element."""
        self.clear_properties()
        
        for key, value in data.items():
            label = QLabel(f"{key.replace('_', ' ').title()}:")
            editor = QLineEdit(str(value))
            
            # Make ID and type read-only for now
            if key in ['id', 'type']:
                editor.setReadOnly(True)
            
            # Use a lambda with default arguments to capture loop variables
            editor.editingFinished.connect(
                lambda k=key, e=editor, d=data: self.on_property_changed(d['id'], item_type, k, e.text())
            )
            self.properties_layout.addRow(label, editor)

    def on_property_changed(self, item_id: str, item_type: str, key: str, new_value_str: str):
        """Updates the network configuration when a property is edited."""
        if not self.controller or not self.controller.network_config:
            return

        config_list = self.controller.network_config.nodes if item_type == 'node' else self.controller.network_config.components
        
        for item in config_list:
            if item['id'] == item_id:
                original_value = item.get(key)
                try:
                    # Convert new value to the type of the original value
                    if isinstance(original_value, float):
                        item[key] = float(new_value_str)
                    elif isinstance(original_value, int):
                        item[key] = int(new_value_str)
                    else:
                        item[key] = new_value_str
                except (ValueError, TypeError):
                    # If conversion fails, do not update. Maybe show a message.
                    print(f"Invalid value '{new_value_str}' for property '{key}'.")
                    # Revert editor text
                    # This is tricky, we don't have the editor handle here.
                    # For now, we just don't update the model.
                    pass 
                
                # If position changed, redraw network
                if key in ['x', 'y'] and self.canvas:
                    self.canvas.draw_network(self.controller.network_config)
                
                break

    def populate_solvers(self, solvers: list[str]):
        """Populates the solver selection dropdown."""
        self.solver_combo.blockSignals(True)
        self.solver_combo.clear()
        self.solver_combo.addItems(solvers)
        self.solver_combo.blockSignals(False)

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
            
        text += "\nComponent Flows (m 3/s):\n"
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