"""
Main application window for the Lubrication Flow Network GUI.
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QAction, QFileDialog, QMessageBox
from PyQt5.QtCore import pyqtSlot

from .sidebar import Sidebar
from .canvas import NetworkCanvas
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

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()