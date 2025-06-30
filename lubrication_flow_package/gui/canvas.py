import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class NetworkGraph:
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.graph = nx.DiGraph()
        self.figure = plt.figure(figsize=(8, 6))
        self.ax = self.figure.add_subplot(111)
        
        self.graph_canvas = FigureCanvasTkAgg(self.figure, master=self.parent_frame)
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.graph_canvas, self.parent_frame)
        self.toolbar.update()
        self.graph_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.grid_size = 50
        self.setup_pan_zoom()
        self.draw_grid()

    def setup_pan_zoom(self):
        self.figure.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self._pan_start = None

    def on_press(self, event):
        if event.button == 2:  # Middle mouse button
            self._pan_start = (event.x, event.y)

    def on_release(self, event):
        if event.button == 2:
            self._pan_start = None

    def on_motion(self, event):
        if self._pan_start is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        dx_data = (cur_xlim[1] - cur_xlim[0]) * (dx / self.ax.get_window_extent().width)
        dy_data = (cur_ylim[1] - cur_ylim[0]) * (dy / self.ax.get_window_extent().height)

        self.ax.set_xlim(cur_xlim[0] - dx_data, cur_xlim[1] - dx_data)
        self.ax.set_ylim(cur_ylim[0] - dy_data, cur_ylim[1] - dy_data)
        self.graph_canvas.draw()

    def on_scroll(self, event):
        if event.xdata is None or event.ydata is None:
            return
        
        scale_factor = 1.1 if event.button == 'up' else 1 / 1.1
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

        relx = (cur_xlim[1] - event.xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - event.ydata) / (cur_ylim[1] - cur_ylim[0])

        self.ax.set_xlim([event.xdata - new_width * (1 - relx), event.xdata + new_width * relx])
        self.ax.set_ylim([event.ydata - new_height * (1 - rely), event.ydata + new_height * rely])
        self.graph_canvas.draw()

    def zoom_to_fit(self):
        if not self.graph.nodes:
            self.ax.set_xlim(0, 800)
            self.ax.set_ylim(0, 600)
            self.graph_canvas.draw()
            return

        pos = {node: (data['x'], data['y']) for node, data in self.graph.nodes(data=True)}
        if not pos:
            self.ax.set_xlim(0, 800)
            self.ax.set_ylim(0, 600)
            self.graph_canvas.draw()
            return
            
        x_coords, y_coords = zip(*pos.values())
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        padding = 50
        self.ax.set_xlim(min_x - padding, max_x + padding)
        self.ax.set_ylim(min_y - padding, max_y + padding)
        self.graph_canvas.draw()

    def draw_grid(self):
        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

    def draw_graph(self, app, node_colors=None, edge_colors=None):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        self.ax.clear()
        self.ax.grid(True)
        self.ax.set_xlabel("X (mm)")
        self.ax.set_ylabel("Y (mm)")
        self.ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)

        self.ax.set_xlim(cur_xlim)
        self.ax.set_ylim(cur_ylim)

        pos = {node: (data['x'], data['y']) for node, data in self.graph.nodes(data=True)}
        
        if not pos: # No nodes, just draw the grid
            self.zoom_to_fit()
            return
        
        if node_colors is None:
            node_colors = [data.get('color', 'skyblue') for node, data in self.graph.nodes(data=True)]
        if edge_colors is None:
            edge_colors = 'gray'
            
        nodes = nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, node_color=node_colors, node_size=700)
        if nodes:
            nodes.set_picker(5)

        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, edge_color=edge_colors)
        nx.draw_networkx_labels(self.graph, pos, ax=self.ax, font_size=10)
        
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, ax=self.ax)
        self.graph_canvas.draw()
