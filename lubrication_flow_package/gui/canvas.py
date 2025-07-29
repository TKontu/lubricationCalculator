"""
Matplotlib Canvas for PyQt5 GUI.
"""
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
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
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        self.ax = self.figure.add_subplot(111)
        self.graph = nx.Graph()
        
    def draw_network(self, network_config):
        """Draws the network graph with a stable, smart grid, and auto-zoomed layout."""
        self.ax.clear()
        self.graph.clear()
        
        if not network_config or not network_config.nodes:
            self.canvas.draw()
            return

        for node_data in network_config.nodes:
            self.graph.add_node(node_data['id'])
        
        for conn_data in network_config.connections:
            self.graph.add_edge(conn_data['from_node'], conn_data['to_node'], **conn_data)

        # Check if positions are defined in the config
        has_positions = all('x' in n and 'y' in n for n in network_config.nodes)
        
        pos = {}
        if has_positions:
            for node_data in network_config.nodes:
                pos[node_data['id']] = (node_data['x'], node_data['y'])
        else:
            # Auto-generate a layout, then snap to a grid.
            # Use a layout that respects topology to avoid overlaps.
            if self.graph.number_of_edges() > 0:
                try:
                    # Start with a layout that spaces nodes well.
                    base_pos = nx.kamada_kawai_layout(self.graph)
                except nx.NetworkXError:
                    base_pos = nx.spring_layout(self.graph, seed=42)
            else:
                base_pos = nx.random_layout(self.graph, seed=42)

            # Snap the positions to a grid
            for node_id, (x, y) in base_pos.items():
                pos[node_id] = (round(x * 20), round(y * 20))


        # Store positions in the graph nodes
        for node_id, position in pos.items():
            self.graph.nodes[node_id]['pos'] = position
                    
        # Draw the graph elements
        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, node_color='skyblue', node_size=700)
        nx.draw_networkx_labels(self.graph, pos, ax=self.ax)
        self._draw_orthogonal_edges(pos)
        
        self.ax.grid(True) # Ensure grid is visible
        self._autoscale_view(pos)
        self.canvas.draw()

    def _draw_orthogonal_edges(self, pos, edge_color='gray'):
        """Draws edges as orthogonal lines."""
        for u, v in self.graph.edges():
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            # Draw horizontal segment first, then vertical
            self.ax.plot([x1, x2], [y1, y1], color=edge_color, zorder=1)
            self.ax.plot([x2, x2], [y1, y2], color=edge_color, zorder=1)

    def _autoscale_view(self, pos):
        """Zooms the view to fit the graph with a margin."""
        if not pos:
            return
            
        x_coords = [p[0] for p in pos.values()]
        y_coords = [p[1] for p in pos.values()]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        dx = max_x - min_x
        dy = max_y - min_y
        
        margin_x = dx * 0.15 or 10
        margin_y = dy * 0.15 or 10
        
        self.ax.set_xlim(min_x - margin_x, max_x + margin_x)
        self.ax.set_ylim(min_y - margin_y, max_y + margin_y)
        
        self.ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        self.ax.grid(True)
        self.ax.set_aspect('equal', adjustable='box')
        self.figure.tight_layout()

    def update_visuals(self, results):
        """Updates the graph with simulation results without changing the layout."""
        if not results or not self.graph:
            return
            
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos:
            self.canvas.draw()
            return

        node_pressures = results.get('node_pressures', {})
        component_flows = results.get('component_flows', {})
        
        # Normalize pressures for color mapping
        min_p = min(node_pressures.values()) if node_pressures else 0
        max_p = max(node_pressures.values()) if node_pressures else 1
        
        def normalize(value, min_val, max_val):
            if max_val > min_val:
                return (value - min_val) / (max_val - min_val)
            return 0.5

        node_color_values = [normalize(node_pressures.get(n, 0), min_p, max_p) for n in self.graph.nodes]
        
        self.ax.clear()
        
        # Redraw the graph with the new colors but same positions
        cmap = plt.get_cmap('viridis')
        nodes = nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, node_color=node_color_values, cmap=cmap, node_size=700)
        if nodes:
            nodes.set_zorder(2)

        # Draw high-contrast node labels
        node_colors_rgba = nodes.get_facecolor()
        
        # Determine if a single color was returned for all nodes
        use_single_color = False
        single_color = None
        if node_colors_rgba.ndim == 1:
            # Shape is (4,), one color for all nodes
            use_single_color = True
            single_color = node_colors_rgba
        elif node_colors_rgba.shape[0] == 1:
            # Shape is (1, 4), one color for all nodes
            use_single_color = True
            single_color = node_colors_rgba[0]

        for i, node_id in enumerate(self.graph.nodes()):
            rgba = single_color if use_single_color else node_colors_rgba[i]
            
            # Using a standard luminance formula to determine text color
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            text_color = 'white' if luminance < 0.5 else 'black'
            x, y = pos[node_id]
            self.ax.text(x, y, node_id, ha='center', va='center', color=text_color, zorder=3, fontsize=9)

        self._draw_orthogonal_edges(pos)

        # Draw edge labels on the horizontal segment with a solid background
        for u, v, d in self.graph.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            label = f"{component_flows.get(d.get('component'), 0):.4f}"
            self.ax.text((x1 + x2) / 2, y1, label, ha='center', va='center', fontsize=8, color='blue', zorder=4,
                         bbox=dict(facecolor='white', alpha=1.0, edgecolor='none', boxstyle='round,pad=0.2'))
        
        self.ax.grid(True)
        self._autoscale_view(pos)
        self.canvas.draw()