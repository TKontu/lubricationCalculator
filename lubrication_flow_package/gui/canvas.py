"""
Matplotlib Canvas for PyQt5 GUI.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx
import matplotlib.pyplot as plt

class NetworkCanvas(QWidget):
    """
    A PyQt5 widget that embeds a matplotlib Figure.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax = self.figure.add_subplot(111)
        self.graph = nx.Graph()
        
    def draw_network(self, network_config):
        """Draws the network graph."""
        self.ax.clear()
        self.graph.clear()
        
        if not network_config:
            self.canvas.draw()
            return

        # Add nodes
        pos = {}
        for i, node_data in enumerate(network_config.nodes):
            node_id = node_data['id']
            self.graph.add_node(node_id, **node_data)
            # Use x, y from config if available, otherwise generate position
            pos[node_id] = (node_data.get('x', i * 10), node_data.get('y', 0))

        # Add edges
        for conn_data in network_config.connections:
            self.graph.add_edge(conn_data['from_node'], conn_data['to_node'], **conn_data)
            
        nx.draw(self.graph, pos, ax=self.ax, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
        self.canvas.draw()

    def update_visuals(self, results):
        """Updates the graph with simulation results."""
        if not results or not self.graph:
            return
            
        node_pressures = results.get('node_pressures', {})
        component_flows = results.get('component_flows', {})
        
        # Normalize pressures for color mapping
        min_p = min(node_pressures.values()) if node_pressures else 0
        max_p = max(node_pressures.values()) if node_pressures else 0
        
        def normalize(value, min_val, max_val):
            if max_val - min_val > 0:
                return (value - min_val) / (max_val - min_val)
            return 0.5

        node_colors = [normalize(node_pressures.get(n, 0), min_p, max_p) for n in self.graph.nodes]
        
        # Get positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos:
             pos = nx.spring_layout(self.graph)

        self.ax.clear()
        nx.draw(self.graph, pos, ax=self.ax, with_labels=True, node_color=node_colors, 
                cmap=plt.get_cmap('viridis'), node_size=700, edge_color='gray')
        self.canvas.draw()