"""
Main application window for the Lubrication Flow Network GUI.
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QAction, QFileDialog, QMessageBox, QInputDialog
from PyQt5.QtCore import pyqtSlot, Qt

from .sidebar import Sidebar
from .canvas import NetworkCanvas
from .dialogs import PropertiesDialog
from ..simulation.simulation_controller import SimulationController
from ..config.network_config import NetworkConfigLoader, NetworkConfigSaver, NetworkConfig
from ..config.simulation_config import SimulationConfig

class App(QMainWindow):
    """
    The main application window.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lubrication Flow Network Calculator")
        self.setGeometry(100, 100, 1200, 800)

        self.controller = SimulationController()
        self.current_file_path = None
        self.add_node_mode = False
        
        self._create_widgets()
        self._create_menu()
        self._connect_signals()

    def _create_widgets(self):
        """Create the main widgets and layout."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout(self.central_widget)
        
        self.canvas = NetworkCanvas()
        
        self.sidebar = Sidebar()
        self.sidebar.set_controller_and_canvas(self.controller, self.canvas)
        
        self.layout.addWidget(self.canvas, 1) # Give more space to canvas
        self.layout.addWidget(self.sidebar)

    def _create_menu(self):
        """Create the main menu bar."""
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("&File")

        open_action = QAction("&Open Configuration...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        save_action = QAction("&Save", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        save_as_action = QAction("Save &As...", self)
        save_as_action.triggered.connect(self.save_file_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        exit_action = QAction("&Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def _connect_signals(self):
        """Connect signals between the controller and GUI components."""
        self.sidebar.run_simulation_requested.connect(self.run_simulation)
        self.sidebar.solver_changed.connect(self.controller.set_solver)
        self.sidebar.add_node_requested.connect(self.enter_add_node_mode)
        self.sidebar.delete_node_requested.connect(self.delete_node)
        self.sidebar.add_connection_requested.connect(self.add_connection)
        self.sidebar.edit_node_requested.connect(self.edit_node)
        self.canvas.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.controller.set_progress_callback(self.sidebar.results_text.append)

    def open_file(self):
        """Open a network configuration file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Network Configuration", "", 
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            try:
                network_config = NetworkConfigLoader.load_json(file_name)
                sim_settings = network_config.simulation
                sim_config = SimulationConfig.from_dict(sim_settings)
                
                self.controller.load_configuration(network_config, sim_config)
                
                self.sidebar.update_element_lists(network_config)
                self.sidebar.populate_solvers(self.controller.get_available_solvers())
                # Set the first solver as the default
                if self.controller.get_available_solvers():
                    default_solver = self.controller.get_available_solvers()[0]
                    self.sidebar.solver_combo.setCurrentText(default_solver)
                    self.controller.set_solver(default_solver)
                
                self.sidebar.clear_results()
                self.canvas.draw_network(network_config)
                self.current_file_path = file_name
                self.setWindowTitle(f"Lubrication Flow Calculator - {file_name}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file: {e}")

    def save_file(self):
        """Save the current network configuration."""
        if self.current_file_path:
            self._save_to_path(self.current_file_path)
        else:
            self.save_file_as()

    def save_file_as(self):
        """Save the current network configuration to a new file."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Network Configuration", "",
                                                   "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            self.current_file_path = file_name
            self.setWindowTitle(f"Lubrication Flow Calculator - {file_name}")
            self._save_to_path(file_name)

    def _save_to_path(self, file_path: str):
        """Helper to save the configuration to a specific path."""
        if not self.controller.network_config:
            QMessageBox.warning(self, "Warning", "No network configuration loaded to save.")
            return
        
        try:
            # Update the config from the UI before saving
            sim_settings = self.sidebar.get_simulation_settings()
            self.controller.network_config.simulation.update(sim_settings)
            
            NetworkConfigSaver.save_json(self.controller.network_config, file_path)
            self.sidebar.results_text.append(f"Configuration saved to {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving file: {e}")

    @pyqtSlot()
    def run_simulation(self):
        """Slot to run the simulation."""
        self.sidebar.clear_results()
        
        if not self.controller.sim_config:
            QMessageBox.warning(self, "Warning", "Please load a configuration before running a simulation.")
            return

        # Get settings from UI and update sim_config
        sim_settings = self.sidebar.get_simulation_settings()
        for key, value in sim_settings.items():
            if hasattr(self.controller.sim_config, key):
                setattr(self.controller.sim_config, key, value)

        # Set the solver from the dropdown
        selected_solver = self.sidebar.solver_combo.currentText()
        if not selected_solver:
            QMessageBox.warning(self, "Warning", "Please select a solver.")
            return
        self.controller.set_solver(selected_solver)

        # Run the simulation
        self.controller.run_simulation()
        
        # Display results
        results = self.controller.get_results()
        self.sidebar.display_results(results)
        self.canvas.update_visuals(results)

    def enter_add_node_mode(self):
        """Activates 'add node' mode."""
        self.add_node_mode = True
        self.setCursor(Qt.CrossCursor)
        self.statusBar().showMessage("Click on the canvas to add a new node.")

    def on_canvas_click(self, event):
        """Handles clicks on the canvas, for adding nodes."""
        if self.add_node_mode and event.inaxes == self.canvas.ax:
            x, y = event.xdata, event.ydata
            self.add_node(x, y)
            self.add_node_mode = False
            self.setCursor(Qt.ArrowCursor)
            self.statusBar().clearMessage()

    def add_node(self, x, y):
        """Adds a new node to the network configuration."""
        if not self.controller.network_config:
            QMessageBox.warning(self, "Warning", "Load a configuration first.")
            return

        node_id, ok = QInputDialog.getText(self, "New Node", "Enter Node ID:")
        if ok and node_id:
            # Check for duplicate ID
            if any(n['id'] == node_id for n in self.controller.network_config.nodes):
                QMessageBox.warning(self, "Warning", f"Node ID '{node_id}' already exists.")
                return

            new_node = {'id': node_id, 'x': round(x), 'y': round(y), 'type': 'internal'}
            self.controller.network_config.nodes.append(new_node)
            
            # Refresh UI
            self.sidebar.update_element_lists(self.controller.network_config)
            self.canvas.draw_network(self.controller.network_config)

    def delete_node(self, node_id: str):
        """Deletes a node and its connected components from the network."""
        if not self.controller.network_config:
            return

        # Find connections and components associated with the node
        components_to_delete = set()
        connections_to_delete = []
        for c in self.controller.network_config.connections:
            if c['from_node'] == node_id or c['to_node'] == node_id:
                connections_to_delete.append(c)
                if 'component' in c:
                    components_to_delete.add(c['component'])

        # Find and remove the node
        self.controller.network_config.nodes = [
            n for n in self.controller.network_config.nodes if n['id'] != node_id
        ]

        # Remove the identified connections
        self.controller.network_config.connections = [
            c for c in self.controller.network_config.connections if c not in connections_to_delete
        ]

        # Remove the identified components
        self.controller.network_config.components = [
            comp for comp in self.controller.network_config.components 
            if comp['id'] not in components_to_delete
        ]
        
        # Refresh UI
        self.sidebar.update_element_lists(self.controller.network_config)
        self.canvas.draw_network(self.controller.network_config)
        self.sidebar.results_text.append(f"Deleted node '{node_id}' and connected components.")

    def add_connection(self):
        """Adds a new connection between two nodes."""
        if not self.controller.network_config or len(self.controller.network_config.nodes) < 2:
            QMessageBox.warning(self, "Warning", "Please add at least two nodes before adding a connection.")
            return

        node_ids = [n['id'] for n in self.controller.network_config.nodes]
        
        from_node, ok1 = QInputDialog.getItem(self, "Add Connection", "From Node:", node_ids, 0, False)
        if not ok1: return
        
        to_node, ok2 = QInputDialog.getItem(self, "Add Connection", "To Node:", node_ids, 0, False)
        if not ok2: return

        if from_node == to_node:
            QMessageBox.warning(self, "Warning", "Cannot connect a node to itself.")
            return

        # Check if a connection already exists
        for c in self.controller.network_config.connections:
            if (c['from_node'] == from_node and c['to_node'] == to_node) or \
               (c['from_node'] == to_node and c['to_node'] == from_node):
                QMessageBox.warning(self, "Warning", f"A connection between '{from_node}' and '{to_node}' already exists.")
                return

        comp_id, ok3 = QInputDialog.getText(self, "Add Connection", "Enter Component ID for this connection:")
        if not ok3 or not comp_id: return

        # Check for duplicate component ID
        if any(c['id'] == comp_id for c in self.controller.network_config.components):
            QMessageBox.warning(self, "Warning", f"Component ID '{comp_id}' already exists.")
            return

        # For simplicity, we'll create a 'channel' type component by default.
        # A more advanced implementation would ask for component type and properties.
        new_component = {'id': comp_id, 'type': 'channel', 'length': 1.0, 'diameter': 0.01}
        new_connection = {'from_node': from_node, 'to_node': to_node, 'component': comp_id}

        self.controller.network_config.components.append(new_component)
        self.controller.network_config.connections.append(new_connection)

        # Refresh UI
        self.sidebar.update_element_lists(self.controller.network_config)
        self.canvas.draw_network(self.controller.network_config)

    def edit_node(self, node_id: str):
        """Opens a dialog to edit the properties of a node."""
        if not self.controller.network_config:
            return

        node_data = None
        for n in self.controller.network_config.nodes:
            if n['id'] == node_id:
                node_data = n
                break
        
        if not node_data:
            return

        dialog = PropertiesDialog(node_id, node_data)
        if dialog.exec_() == QDialog.Accepted:
            updated_props = dialog.get_properties()
            
            # Update the node in the config
            for i, n in enumerate(self.controller.network_config.nodes):
                if n['id'] == node_id:
                    self.controller.network_config.nodes[i] = updated_props
                    break
            
            # Refresh UI
            self.sidebar.update_element_lists(self.controller.network_config)
            self.canvas.draw_network(self.controller.network_config)

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()