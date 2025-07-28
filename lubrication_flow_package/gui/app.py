"""
Main application window for the Lubrication Flow Network GUI.
"""
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QAction, QFileDialog
from PyQt5.QtCore import pyqtSlot

from .sidebar import Sidebar
from ..simulation.simulation_controller import SimulationController
from ..config.network_config import NetworkConfigLoader
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
        
        self._create_widgets()
        self._create_menu()
        self._connect_signals()

    def _create_widgets(self):
        """Create the main widgets and layout."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout(self.central_widget)
        
        # In a real application, the network canvas would be here.
        # For now, we'll use a placeholder.
        self.canvas_placeholder = QWidget()
        self.canvas_placeholder.setStyleSheet("background-color: #f0f0f0;")
        
        self.sidebar = Sidebar()
        
        self.layout.addWidget(self.canvas_placeholder, 1) # Give more space to canvas
        self.layout.addWidget(self.sidebar)

    def _create_menu(self):
        """Create the main menu bar."""
        self.menu_bar = self.menuBar()
        file_menu = self.menu_bar.addMenu("&File")

        open_action = QAction("&Open Configuration...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

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
                self.sidebar.clear_results()
                
            except Exception as e:
                # In a real app, show a message box
                print(f"Error loading file: {e}")

    @pyqtSlot()
    def run_simulation(self):
        """Slot to run the simulation."""
        self.sidebar.clear_results()
        
        # Get settings from UI and update sim_config
        sim_settings = self.sidebar.get_simulation_settings()
        for key, value in sim_settings.items():
            if hasattr(self.controller.sim_config, key):
                setattr(self.controller.sim_config, key, value)

        # Set the solver from the dropdown
        selected_solver = self.sidebar.solver_combo.currentText()
        self.controller.set_solver(selected_solver)

        # Run the simulation
        converged = self.controller.run_simulation()
        
        # Display results
        results = self.controller.get_results()
        self.sidebar.display_results(results)

def main():
    """Main entry point for the GUI application."""
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()