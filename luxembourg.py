import xml.etree.ElementTree as ET
import tkinter as tk
import tkinter.messagebox as messagebox
from Models.graph import Graph
from Models.node_edge import Node, Edge
import heapq
import numpy as np

class GraphVisualizer:
    def __init__(self, nodes, edges, max_lat, min_lat, max_long, min_long):
        self.nodes = nodes
        self.edges = edges
        self.max_lat = max_lat
        self.min_lat = min_lat
        self.max_long = max_long
        self.min_long = min_long
        self.start_node = None
        self.end_node = None
        self.start_marker = None
        self.end_marker = None
        self.graph = self.build_graph()
        self.path_lines = []

    def build_graph(self):
        graph = {}
        for edge in self.edges:
            graph.setdefault(edge.from_node_id, []).append((edge.to_node_id, edge.weight))
        return graph

    def scale(self, value, min_value, max_value, size):
        return int((value - min_value) / (max_value - min_value) * size)

    def draw_graph(self):
        self.window = tk.Tk()
        self.window.title("Graph Visualization")
        self.window.resizable(False, False)
        self.window.geometry("1000x600")

        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Control-Button-1>", self.on_right_click)

        control_frame = tk.Frame(self.window, width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, expand=True)

        self.start_label = tk.Label(control_frame, text="Start Node: None", font=("Arial", 12))
        self.start_label.pack(pady=10)
        self.end_label = tk.Label(control_frame, text="End Node: None", font=("Arial", 12))
        self.end_label.pack(pady=10)

        tk.Button(control_frame, text="Run Dijkstra", bg="gray", font=("Arial", 12),
                  command=self.run_dijkstra).pack(pady=10)
        tk.Button(control_frame, text="Run Bellman-Ford", bg="gray", font=("Arial", 12),
                  command=self.run_bellman_ford).pack(pady=10)

        self.node_positions = {}
        for edge in self.edges:
            from_node = self.nodes[edge.from_node_id]
            to_node = self.nodes[edge.to_node_id]

            x1 = self.scale(from_node.x, self.min_long, self.max_long, self.canvas_width)
            y1 = self.scale(from_node.y, self.min_lat, self.max_lat, self.canvas_height)
            x2 = self.scale(to_node.x, self.min_long, self.max_long, self.canvas_width)
            y2 = self.scale(to_node.y, self.min_lat, self.max_lat, self.canvas_height)

            self.canvas.create_line(x1, y1, x2, y2, fill="black", width=2)

            self.node_positions[from_node.id] = (x1, y1)
            self.node_positions[to_node.id] = (x2, y2)

        self.window.mainloop()

    def on_left_click(self, event):
        closest_node = self.get_closest_node(event)
        if closest_node is not None:
            self.start_node = closest_node
            self.start_label.config(text=f"Start Node: {closest_node}")
            self.remove_marker(self.start_marker)
            self.start_marker = self.mark_selection(closest_node, color="green")

    def on_right_click(self, event):
        closest_node = self.get_closest_node(event)
        if closest_node is not None:
            self.end_node = closest_node
            self.end_label.config(text=f"End Node: {closest_node}")
            self.remove_marker(self.end_marker)
            self.end_marker = self.mark_selection(closest_node, color="red")

    def get_closest_node(self, event):
        closest_node = None
        closest_distance = float("inf")

        for node_id, (x, y) in self.node_positions.items():
            distance = ((event.x - x) ** 2 + (event.y - y) ** 2) ** 0.5
            if distance < closest_distance:
                closest_node = node_id
                closest_distance = distance

        return closest_node

    def mark_selection(self, node_id, color):
        x, y = self.node_positions[node_id]
        marker_id = self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill=color, outline=color)
        return marker_id

    def remove_marker(self, marker):
        if marker is not None:
            self.canvas.delete(marker)

    def draw_path(self, path):
        self.clear_path_lines()

        for i in range(len(path) - 1):
            x1, y1 = self.node_positions[path[i]]
            x2, y2 = self.node_positions[path[i + 1]]
            line_id = self.canvas.create_line(x1, y1, x2, y2, fill="orange", width=3)
            self.path_lines.append(line_id)

    def clear_path_lines(self):
        for line_id in self.path_lines:
            self.canvas.delete(line_id)
        self.path_lines = []

    def run_dijkstra(self):
        self.clear_path_lines()

        if self.start_node is None or self.end_node is None:
            messagebox.showerror("No start/end node(s)", "Please select both start and end nodes.")
            return

        distances = {node: float('inf') for node in self.graph}
        distances[self.start_node] = 0
        previous = {}
        visited = set()

        priority_queue = []
        heapq.heappush(priority_queue, (0, self.start_node))

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_node in visited:
                continue
            visited.add(current_node)

            if current_node == self.end_node:
                break

            for neighbor, weight in self.graph.get(current_node, []):
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_distance, neighbor))

        path = []
        current = self.end_node
        while current in previous:
            path.append(current)
            current = previous[current]
        if current == self.start_node:
            path.append(self.start_node)
            path.reverse()

        print(f"Dikstra Path: {path}")
        self.draw_path(path)

    def run_bellman_ford(self):
        self.clear_path_lines()

        if self.start_node is None or self.end_node is None:
            messagebox.showerror("No start/end node(s)", "Please select both start and end nodes.")
            return

        node_indices = {node: idx for idx, node in enumerate(self.graph.keys())}
        index_to_node = {idx: node for node, idx in node_indices.items()}
        num_nodes = len(node_indices)

        distances = np.full(num_nodes, np.inf, dtype=np.float64)
        previous = np.full(num_nodes, -1, dtype=np.int32)

        start_index = node_indices[self.start_node]
        end_index = node_indices[self.end_node]

        distances[start_index] = 0

        edges = []
        for node, neighbors in self.graph.items():
            for neighbor, weight in neighbors:
                edges.append((node_indices[node], node_indices[neighbor], weight))
        edges = np.array(edges, dtype=np.float64)

        for _ in range(num_nodes - 1):
            u = edges[:, 0].astype(np.int32)
            v = edges[:, 1].astype(np.int32)
            w = edges[:, 2]

            relax_mask = (distances[u] + w) < distances[v]
            distances[v[relax_mask]] = distances[u[relax_mask]] + w[relax_mask]
            previous[v[relax_mask]] = u[relax_mask]

            if not relax_mask.any():
                break

        path_indices = []
        current = end_index

        if distances[end_index] == np.inf:
            print("No path found")
            return

        while current != -1:
            path_indices.append(current)
            current = previous[current]

        path_indices.reverse()
        path = [index_to_node[i] for i in path_indices]

        print(f"Bellman-Ford Path: {path}")
        self.draw_path(path)

def build_graph_from_xml(xml_file):
    graph = Graph()
    nodes = {}
    edges = []

    max_lat, min_lat = float("-inf"), float("inf")
    max_long, min_long = float("-inf"), float("inf")

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for node in root.find('nodes'):
        node_id = int(node.attrib['id'])
        longitude = int(node.attrib['longitude'])
        latitude = int(node.attrib['latitude'])
        nodes[node_id] = Node(node_id, longitude, latitude, None, None)
        max_lat = max(max_lat, latitude)
        min_lat = min(min_lat, latitude)
        max_long = max(max_long, longitude)
        min_long = min(min_long, longitude)

    for arc in root.find('arcs'):
        from_node = int(arc.attrib['from'])
        to_node = int(arc.attrib['to'])
        weight = int(arc.attrib['length'])
        edges.append(Edge(from_node, to_node, line_id=f"line_{from_node}_{to_node}", weight=weight))
        graph.add_edge(from_node, to_node)

    return graph, nodes, edges, max_lat, min_lat, max_long, min_long

if __name__ == '__main__':
    xml_file = 'Resources/Harta_Luxemburg.xml'
    graph, nodes, edges, max_lat, min_lat, max_long, min_long = build_graph_from_xml(xml_file)

    visualizer = GraphVisualizer(nodes, edges, max_lat, min_lat, max_long, min_long)
    visualizer.draw_graph()
