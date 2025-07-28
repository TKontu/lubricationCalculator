"""
Tests for the PyQt5 GUI components.
"""
import pytest
from PyQt5.QtCore import Qt
from unittest.mock import Mock, MagicMock

# Mark all tests in this file as needing the qt_bot fixture
pytestmark = pytest.mark.qt

from lubrication_flow_package.gui.sidebar import Sidebar

@pytest.fixture
def sidebar(qtbot):
    """Fixture to create a Sidebar widget."""
    widget = Sidebar()
    qtbot.addWidget(widget)
    widget.show()
    return widget

def test_sidebar_initialization(sidebar: Sidebar):
    """Test that the sidebar initializes correctly."""
    assert sidebar.solver_combo.count() == 0
    assert sidebar.results_text.toPlainText() == ""
    assert sidebar.run_button.isEnabled()

def test_populate_solvers(sidebar: Sidebar):
    """Test populating the solver dropdown."""
    solvers = ["solver1", "solver2", "solver3"]
    sidebar.populate_solvers(solvers)
    assert sidebar.solver_combo.count() == 3
    assert sidebar.solver_combo.itemText(0) == "solver1"
    assert sidebar.solver_combo.currentText() == "solver1"

def test_solver_changed_signal(sidebar: Sidebar, qtbot):
    """Test that the solver_changed signal is emitted."""
    solvers = ["nodal", "tree_nonlinear"]
    sidebar.populate_solvers(solvers)
    
    with qtbot.waitSignal(sidebar.solver_changed, raising=True) as blocker:
        sidebar.solver_combo.setCurrentIndex(1)
        
    assert blocker.args == ["tree_nonlinear"]

def test_run_simulation_requested_signal(sidebar: Sidebar, qtbot):
    """Test that the run_simulation_requested signal is emitted."""
    with qtbot.waitSignal(sidebar.run_simulation_requested, raising=True) as blocker:
        qtbot.mouseClick(sidebar.run_button, Qt.LeftButton)

def test_get_simulation_settings(sidebar: Sidebar):
    """Test reading simulation settings from the UI."""
    sidebar.sim_entries["total_flow_rate"].setText("0.05")
    sidebar.sim_entries["temperature"].setText("60")
    sidebar.sim_entries["inlet_pressure"].setText("400000")

    settings = sidebar.get_simulation_settings()
    
    assert settings["total_flow_rate"] == 0.05
    assert settings["temperature"] == 60
    assert settings["inlet_pressure"] == 400000

def test_get_simulation_settings_invalid_input(sidebar: Sidebar):
    """Test that non-numeric input is handled gracefully."""
    sidebar.sim_entries["total_flow_rate"].setText("invalid")
    settings = sidebar.get_simulation_settings()
    assert settings["total_flow_rate"] == 0.0

def test_display_results(sidebar: Sidebar):
    """Test that results are displayed correctly."""
    results = {
        "converged": True,
        "iterations": 15,
        "final_residual_norm": 1.2345e-7,
        "node_pressures": {"N1": 300000, "N2": 250000},
        "component_flows": {"C1": 0.01, "C2": 0.005}
    }
    sidebar.display_results(results)
    
    text = sidebar.results_text.toPlainText()
    assert "Converged: True" in text
    assert "Iterations: 15" in text
    assert "Final Residual: 1.2345e-07" in text
    assert "N1: 300,000" in text
    assert "C1: 0.010000" in text

def test_clear_results(sidebar: Sidebar):
    """Test clearing the results display."""
    sidebar.results_text.setPlainText("Some results")
    sidebar.clear_results()
    assert sidebar.results_text.toPlainText() == ""

def test_update_element_lists(sidebar: Sidebar):
    """Test updating the node and component list widgets."""
    mock_config = MagicMock()
    mock_config.nodes = [{'id': 'Node1'}, {'id': 'Node2'}]
    mock_config.components = [{'id': 'Comp1'}]
    
    sidebar.update_element_lists(mock_config)
    
    assert sidebar.nodes_list.count() == 2
    assert sidebar.nodes_list.item(0).text() == 'Node1'
    assert sidebar.components_list.count() == 1
    assert sidebar.components_list.item(0).text() == 'Comp1'
