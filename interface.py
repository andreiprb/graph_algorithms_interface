import tkinter as tk
from tkinter import ttk
import tkinter.messagebox as messagebox
from collections import deque
from Models.graph import Graph
from Models.node_edge import Node, Edge
import math
import random
import heapq

class GraphGUI:
    def __init__(self, master):
        self.master = master
        master.title("Graph Algorithms Visualization")
        self.mode = tk.StringVar()
        self.mode.set("draw_node")
        self.algorithm = tk.StringVar()
        self.algorithm.set("Recursive DFS")
        self.start_node = tk.IntVar()
        self.end_node = tk.IntVar()
        self.graph_type = tk.StringVar()
        self.graph_type.set("directed")
        self.graph_type.trace("w", self.on_graph_type_change)
        self.node_counter = 0
        self.nodes = {}
        self.edges = []
        self.graph = Graph()
        self.create_widgets()
        self.topo_text_ids = []

    def create_widgets(self):
        self.main_frame = tk.Frame(self.master)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.main_frame, bg="white", width=400)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        modes_frame = tk.LabelFrame(control_frame, text="Mode")
        modes_frame.pack(pady=10)
        tk.Radiobutton(modes_frame, text="Draw Nodes", variable=self.mode, value="draw_node").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Draw Edges", variable=self.mode, value="draw_edge").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Move Nodes", variable=self.mode, value="move_node").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Delete Nodes", variable=self.mode, value="delete_node").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Delete Edges", variable=self.mode, value="delete_edge").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Change Weight", variable=self.mode, value="change_edge_weight").pack(anchor=tk.W)
        tk.Radiobutton(modes_frame, text="Change Flux", variable=self.mode, value="change_edge_flux").pack(anchor=tk.W)

        weight_frame = tk.LabelFrame(control_frame, text="Edge Weight")
        weight_frame.pack(pady=10)
        self.weight_entry = tk.Entry(weight_frame, width=5)
        self.weight_entry.insert(0, "1")
        self.weight_entry.pack(pady=5)
        self.weight_entry.bind("<Return>", self.unfocus_weight_flux_entry)

        flux_frame = tk.LabelFrame(control_frame, text="Edge Flux")
        flux_frame.pack(pady=10)
        self.flux_entry = tk.Entry(flux_frame, width=5)
        self.flux_entry.insert(0, "1")
        self.flux_entry.pack(pady=5)
        self.flux_entry.bind("<Return>", self.unfocus_weight_flux_entry)

        graph_type_frame = tk.LabelFrame(control_frame, text="Graph Type")
        graph_type_frame.pack(pady=10)
        tk.Radiobutton(graph_type_frame, text="Undirected", variable=self.graph_type, value="undirected").pack(anchor=tk.W)
        tk.Radiobutton(graph_type_frame, text="Directed", variable=self.graph_type, value="directed").pack(anchor=tk.W)

        alg_frame = tk.LabelFrame(control_frame, text="Algorithm")
        alg_frame.pack(pady=10)
        algorithms = ["Recursive DFS", "DFS", "BFS", "CC/SCC", "Topological Sort", "Check Tree", "Find Center",
                      "Build Tree from Center", "Prim MST", "Kruskal MST", "Boruvka MST", "Dijkstra", "Bellman-Ford",
                      "Ford-Fulkerson", "Cycle Cancelling"]
        alg_menu = ttk.OptionMenu(alg_frame, self.algorithm, self.algorithm.get(), *algorithms)
        alg_menu.pack()

        start_node_frame = tk.LabelFrame(control_frame, text="Start Node")
        start_node_frame.pack(pady=10)
        self.start_node_menu = ttk.OptionMenu(start_node_frame, self.start_node, None)
        self.start_node_menu.pack()

        end_node_frame = tk.LabelFrame(control_frame, text="End Node")
        end_node_frame.pack(pady=10)
        self.end_node_menu = ttk.OptionMenu(end_node_frame, self.end_node, None)
        self.end_node_menu.pack()

        run_button = tk.Button(control_frame, text="Run", command=self.run_algorithm)
        run_button.pack(pady=10)
        clear_button = tk.Button(control_frame, text="Clear Canvas", command=self.clear_canvas)
        clear_button.pack(pady=10)

        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.selected_node_id = None
        self.dragging = False
        self.edge_start_node = None

    def unfocus_weight_flux_entry(self, event):
        self.master.focus()

    def on_graph_type_change(self, *args):
        self.clear_canvas()

    def update_start_node_menu(self):
        menu = self.start_node_menu["menu"]
        menu.delete(0, "end")
        for node_id in self.nodes.keys():
            menu.add_command(label=str(node_id), command=lambda value=node_id: self.start_node.set(value))
        if self.nodes:
            self.start_node.set(next(iter(self.nodes)))
        else:
            self.start_node.set(None)

    def update_end_node_menu(self):
        menu = self.end_node_menu["menu"]
        menu.delete(0, "end")
        for node_id in self.nodes.keys():
            menu.add_command(label=str(node_id), command=lambda value=node_id: self.end_node.set(value))
        if self.nodes:
            self.end_node.set(next(iter(self.nodes)))
        else:
            self.end_node.set(None)

    def canvas_click(self, event):
        if self.mode.get() == "draw_node":
            self.add_node(event.x, event.y)
        elif self.mode.get() == "draw_edge":
            self.select_edge_node(event.x, event.y)
        elif self.mode.get() == "move_node":
            self.start_move_node(event.x, event.y)
        elif self.mode.get() == "delete_node":
            self.delete_node_at(event.x, event.y)
        elif self.mode.get() == "delete_edge":
            self.delete_edge_at(event.x, event.y)
        elif self.mode.get() == "change_edge_weight":
            self.change_edge_weight_at(event.x, event.y)
        elif self.mode.get() == "change_edge_flux":
            self.change_edge_flux_at(event.x, event.y)

    def change_edge_flux_at(self, x, y):
        closest_edge = None
        min_distance = float('inf')

        for edge in self.edges:
            from_node = self.nodes[edge.from_node_id]
            to_node = self.nodes[edge.to_node_id]

            if self.graph_type.get() == "directed":
                mid_x = (from_node.x + to_node.x) / 2
                mid_y = (from_node.y + to_node.y) / 2
                dx = to_node.x - from_node.x
                dy = to_node.y - from_node.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                offset = 30
                offset_x = mid_x - (dy / (distance + 1)) * offset
                offset_y = mid_y + (dx / (distance + 1)) * offset

                distance = self.distance_to_bezier(x, y, from_node.x, from_node.y, offset_x, offset_y,
                                                   to_node.x, to_node.y)
            else:
                distance = self.distance_to_line(x, y, from_node.x, from_node.y, to_node.x, to_node.y)

            if distance < min_distance and distance < 10:
                min_distance = distance
                closest_edge = edge

        if closest_edge:
            new_flux = self.flux_entry.get()
            try:
                new_flux = int(new_flux)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer flux.")
                return

            closest_edge.flux = new_flux
            weight = closest_edge.weight

            combined_text = f"{new_flux}, {weight}"
            self.canvas.itemconfig(closest_edge.weight_flux_text_id, text=str(combined_text))

    def change_edge_weight_at(self, x, y):
        closest_edge = None
        min_distance = float('inf')

        for edge in self.edges:
            from_node = self.nodes[edge.from_node_id]
            to_node = self.nodes[edge.to_node_id]

            if self.graph_type.get() == "directed":
                mid_x = (from_node.x + to_node.x) / 2
                mid_y = (from_node.y + to_node.y) / 2
                dx = to_node.x - from_node.x
                dy = to_node.y - from_node.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                offset = 30
                offset_x = mid_x - (dy / (distance + 1)) * offset
                offset_y = mid_y + (dx / (distance + 1)) * offset

                distance = self.distance_to_bezier(x, y, from_node.x, from_node.y, offset_x, offset_y, to_node.x,
                                                   to_node.y)
            else:
                distance = self.distance_to_line(x, y, from_node.x, from_node.y, to_node.x, to_node.y)

            if distance < min_distance and distance < 10:
                min_distance = distance
                closest_edge = edge

        if closest_edge:
            new_weight = self.weight_entry.get()
            try:
                new_weight = int(new_weight)
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid integer weight.")
                return

            closest_edge.weight = new_weight
            flux = closest_edge.flux

            combined_text = f"{flux}, {new_weight}"
            self.canvas.itemconfig(closest_edge.weight_flux_text_id, text=str(combined_text))

    def delete_node_at(self, x, y):
        node_id = self.get_node_at_position(x, y)
        if node_id is not None:
            node = self.nodes.pop(node_id)
            self.canvas.delete(node.circle_id)
            self.canvas.delete(node.text_id)

            edges_to_delete = [edge for edge in self.edges if
                               edge.from_node_id == node_id or edge.to_node_id == node_id]
            for edge in edges_to_delete:
                self.canvas.delete(edge.line_id)
                self.canvas.delete(edge.weight_flux_text_id)
                if hasattr(edge, 'bg_text_id') and edge.bg_text_id:
                    self.canvas.delete(edge.bg_text_id)
                self.edges.remove(edge)

            self.graph.graph.pop(node_id, None)
            if self.graph_type.get() == "undirected":
                for neighbors in self.graph.graph.values():
                    if node_id in neighbors:
                        neighbors.remove(node_id)
            else:
                for neighbor in self.graph.graph:
                    if node_id in self.graph.graph[neighbor]:
                        self.graph.graph[neighbor].remove(node_id)

            self.update_start_node_menu()
            self.update_end_node_menu()

    def delete_edge_at(self, x, y):
        closest_edge = None
        min_distance = float('inf')

        for edge in self.edges:
            from_node = self.nodes[edge.from_node_id]
            to_node = self.nodes[edge.to_node_id]

            if self.graph_type.get() == "directed":
                mid_x = (from_node.x + to_node.x) / 2
                mid_y = (from_node.y + to_node.y) / 2
                dx = to_node.x - from_node.x
                dy = to_node.y - from_node.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                offset = 30
                offset_x = mid_x - (dy / (distance + 1)) * offset
                offset_y = mid_y + (dx / (distance + 1)) * offset

                distance = self.distance_to_bezier(x, y, from_node.x, from_node.y, offset_x, offset_y, to_node.x,
                                                   to_node.y)
            else:
                distance = self.distance_to_line(x, y, from_node.x, from_node.y, to_node.x, to_node.y)

            if distance < min_distance and distance < 10:
                min_distance = distance
                closest_edge = edge

        if closest_edge:
            self.canvas.delete(closest_edge.line_id)
            self.canvas.delete(closest_edge.weight_flux_text_id)
            self.canvas.delete(closest_edge.bg_text_id)
            self.edges.remove(closest_edge)
            self.graph.graph[closest_edge.from_node_id].remove(closest_edge.to_node_id)

            if self.graph_type.get() == "undirected":
                self.graph.graph[closest_edge.to_node_id].remove(closest_edge.from_node_id)

    def distance_to_line(self, px, py, x1, y1, x2, y2):
        line_mag = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_mag < 0.000001:
            return float('inf')

        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag ** 2)
        if u < 0 or u > 1:
            dist1 = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            dist2 = math.sqrt((px - x2) ** 2 + (py - y2) ** 2)
            return min(dist1, dist2)
        else:
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            return math.sqrt((px - ix) ** 2 + (py - iy) ** 2)

    def distance_to_bezier(self, px, py, x1, y1, cx, cy, x2, y2):
        def bezier_point(t, p0, p1, p2):
            return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

        def distance_to_point(px, py, bx, by):
            return math.sqrt((px - bx) ** 2 + (py - by) ** 2)

        min_distance = float('inf')
        steps = 100
        for i in range(steps + 1):
            t = i / steps
            bx = bezier_point(t, x1, cx, x2)
            by = bezier_point(t, y1, cy, y2)
            distance = distance_to_point(px, py, bx, by)
            if distance < min_distance:
                min_distance = distance

        return min_distance

    def add_node(self, x, y):
        node_id = self.node_counter
        self.node_counter += 1

        r = 20
        circle_id = self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="lightblue")
        text_id = self.canvas.create_text(x, y, text=str(node_id))

        node = Node(node_id, x, y, circle_id, text_id)
        self.nodes[node_id] = node
        self.update_start_node_menu()
        self.update_end_node_menu()

    def select_edge_node(self, x, y):
        node_id = self.get_node_at_position(x, y)
        if node_id is not None:
            if self.edge_start_node is None:
                self.edge_start_node = node_id
                self.canvas.itemconfig(self.nodes[node_id].circle_id, outline="red", width=2)
            else:
                from_node_id = self.edge_start_node
                to_node_id = node_id
                weight = int(self.weight_entry.get())
                flux = int(self.flux_entry.get())
                self.add_edge(from_node_id, to_node_id, weight, flux)
                self.canvas.itemconfig(self.nodes[from_node_id].circle_id, outline="black", width=1)
                self.edge_start_node = None

    def get_node_at_position(self, x, y):
        overlapping = self.canvas.find_overlapping(x - 1, y - 1, x + 1, y + 1)
        for item in overlapping:
            for node_id, node in self.nodes.items():
                if item == node.circle_id or item == node.text_id:
                    return node_id
        return None

    def add_edge(self, from_node_id, to_node_id, weight=1, flux=0):
        for edge in self.edges:
            if (edge.from_node_id == from_node_id and edge.to_node_id == to_node_id) or \
                    (self.graph_type.get() == "undirected" and
                     edge.from_node_id == to_node_id and edge.to_node_id == from_node_id):
                return

        from_node = self.nodes[from_node_id]
        to_node = self.nodes[to_node_id]

        if self.graph_type.get() == "directed":
            mid_x = (from_node.x + to_node.x) / 2
            mid_y = (from_node.y + to_node.y) / 2

            dx = to_node.x - from_node.x
            dy = to_node.y - from_node.y
            distance = math.sqrt(dx ** 2 + dy ** 2)
            offset = 30
            offset_x = mid_x - (dy / (distance + 1)) * offset
            offset_y = mid_y + (dx / (distance + 1)) * offset

            line_id = self.canvas.create_line(from_node.x, from_node.y, offset_x, offset_y,
                                              to_node.x, to_node.y, smooth=True, fill="black", arrow=tk.LAST)

            t = 0.5
            curve_x = (1 - t) ** 2 * from_node.x + 2 * (1 - t) * t * offset_x + t ** 2 * to_node.x
            curve_y = (1 - t) ** 2 * from_node.y + 2 * (1 - t) * t * offset_y + t ** 2 * to_node.y
        else:
            line_id = self.canvas.create_line(from_node.x, from_node.y, to_node.x, to_node.y, fill="black")

            curve_x = (from_node.x + to_node.x) / 2
            curve_y = (from_node.y + to_node.y) / 2

        combined_text = f"{flux}, {weight}"

        rect_padding_x = 5 * len(combined_text)
        rect_padding_y = 10
        rect_id = self.canvas.create_rectangle(curve_x - rect_padding_x, curve_y - rect_padding_y,
                                               curve_x + rect_padding_x, curve_y + rect_padding_y,
                                               fill="white", outline="")

        weight_text_id = self.canvas.create_text(curve_x, curve_y, text=combined_text, fill="blue")

        edge = Edge(from_node_id, to_node_id, line_id, weight, flux)
        edge.weight_flux_text_id = weight_text_id
        edge.bg_text_id = rect_id
        self.edges.append(edge)

        self.graph.add_edge(from_node_id, to_node_id)

        if self.graph_type.get() == "undirected":
            self.graph.add_edge(to_node_id, from_node_id)

    def start_move_node(self, x, y):
        node_id = self.get_node_at_position(x, y)
        if node_id is not None:
            self.selected_node_id = node_id
            self.dragging = True

    def canvas_drag(self, event):
        if self.mode.get() == "move_node" and self.dragging and self.selected_node_id is not None:
            node = self.nodes[self.selected_node_id]
            dx = event.x - node.x
            dy = event.y - node.y
            self.canvas.move(node.circle_id, dx, dy)
            self.canvas.move(node.text_id, dx, dy)
            node.x = event.x
            node.y = event.y
            self.update_edges(self.selected_node_id)

    def canvas_release(self, event):
        if self.mode.get() == "move_node" and self.dragging:
            self.dragging = False
            self.selected_node_id = None

    def update_edges(self, node_id):
        for text_id in self.topo_text_ids:
            self.canvas.delete(text_id)
        self.topo_text_ids.clear()

        for edge in self.edges:
            combined_text = f"{edge.flux}, {edge.weight}"
            self.canvas.itemconfig(edge.weight_flux_text_id, text=str(combined_text))

        for node_id in self.nodes:
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="lightblue")

        for edge in self.edges:
            self.canvas.itemconfig(edge.line_id, fill="black", width=1)

        for edge in self.edges:
            from_node = self.nodes[edge.from_node_id]
            to_node = self.nodes[edge.to_node_id]

            if self.graph_type.get() == "directed":
                mid_x = (from_node.x + to_node.x) / 2
                mid_y = (from_node.y + to_node.y) / 2
                dx = to_node.x - from_node.x
                dy = to_node.y - from_node.y
                distance = math.sqrt(dx ** 2 + dy ** 2)
                offset = 30
                offset_x = mid_x - (dy / (distance + 1)) * offset
                offset_y = mid_y + (dx / (distance + 1)) * offset

                self.canvas.coords(edge.line_id, from_node.x, from_node.y, offset_x, offset_y, to_node.x, to_node.y)

                t = 0.5
                curve_x = (1 - t) ** 2 * from_node.x + 2 * (1 - t) * t * offset_x + t ** 2 * to_node.x
                curve_y = (1 - t) ** 2 * from_node.y + 2 * (1 - t) * t * offset_y + t ** 2 * to_node.y

            else:
                self.canvas.coords(edge.line_id, from_node.x, from_node.y, to_node.x, to_node.y)

                curve_x = (from_node.x + to_node.x) / 2
                curve_y = (from_node.y + to_node.y) / 2

            combined_text = f"{edge.flux}, {edge.weight}"

            if edge.weight_flux_text_id:
                self.canvas.itemconfig(edge.weight_flux_text_id, text=combined_text)
                self.canvas.coords(edge.weight_flux_text_id, curve_x, curve_y)

            if edge.bg_text_id:
                rect_padding_x = 5 * len(combined_text)
                rect_padding_y = 10
                self.canvas.coords(edge.bg_text_id,
                                   curve_x - rect_padding_x, curve_y - rect_padding_y,
                                   curve_x + rect_padding_x, curve_y + rect_padding_y)

    def run_algorithm(self):
        for text_id in self.topo_text_ids:
            self.canvas.delete(text_id)
        self.topo_text_ids.clear()

        for edge in self.edges:
            combined_text = f"{edge.flux}, {edge.weight}"
            self.canvas.itemconfig(edge.weight_flux_text_id, text=str(combined_text))

        for node_id in self.nodes:
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="lightblue")

        for edge in self.edges:
            self.canvas.itemconfig(edge.line_id, fill="black", width=1)

        if self.algorithm.get() == "Dijkstra":
            start_node_id = self.start_node.get()
            if start_node_id is None:
                messagebox.showerror("No starting node", "Please select a starting node")
                return

            self.run_dijkstra(start_node_id)

        elif self.algorithm.get() == "Bellman-Ford":
            start_node_id = self.start_node.get()
            if start_node_id is None:
                messagebox.showinfo("No starting node", "Please select a starting node")
                return

            self.run_bellman_ford(start_node_id)

        elif self.algorithm.get() == "Ford-Fulkerson":
            start_node_id = self.start_node.get()
            end_node_id = self.end_node.get()
            if start_node_id is None or end_node_id is None:
                messagebox.showerror("Invalid Nodes", "Please select both start and end nodes.")
                return

            self.run_ford_fulkerson()

        elif self.algorithm.get() == "Cycle Cancelling":
            start_node_id = self.start_node.get()
            end_node_id = self.end_node.get()
            if start_node_id is None or end_node_id is None:
                messagebox.showerror("Invalid Nodes", "Please select both start and end nodes.")
                return

            self.run_cycle_cancelling()

        elif self.algorithm.get() == "Prim MST":
            self.run_prim_mst()
        elif self.algorithm.get() == "Kruskal MST":
            self.run_kruskal_mst()
        elif self.algorithm.get() == "Boruvka MST":
            self.run_boruvka_mst()

        elif self.algorithm.get() == "Check Tree":
            if self.graph_type.get() == "undirected":
                is_tree = self.is_undirected_tree()
            else:
                is_tree =  self.is_directed_tree()

            if is_tree:
                for node in self.nodes.values():
                    self.canvas.itemconfig(node.circle_id, fill="green")
            else:
                for node in self.nodes.values():
                    self.canvas.itemconfig(node.circle_id, fill="red")

        elif self.algorithm.get() == "Find Center":
            self.find_center()

        elif self.algorithm.get() == "Build Tree from Center":
            if self.is_undirected_tree():
                self.build_tree_from_center()

        elif self.algorithm.get() == "CC/SCC":
            if self.graph_type == "undirected":
                self.run_connected_components()
            else:
                self.run_strongly_connected_components()

        elif self.algorithm.get() == "Topological Sort" and self.is_directed_tree():
            if self.graph_type.get() == "directed":
                sorted_nodes = self.topological_sort()
                for i, node_id in enumerate(sorted_nodes, start=1):
                    x, y = self.nodes[node_id].x, self.nodes[node_id].y
                    order_text = f"{i}"
                    text_id = self.canvas.create_text(x + 25, y, text=order_text, fill="black")
                    self.topo_text_ids.append(text_id)
        else:
            start_node_id = self.start_node.get()
            if start_node_id is None:
                messagebox.showinfo("No starting node", "Please select a starting node")
                return

            algorithm = self.algorithm.get()
            for node in self.nodes.values():
                self.canvas.itemconfig(node.circle_id, fill="lightblue")
            for edge in self.edges:
                self.canvas.itemconfig(edge.line_id, fill="black")

            if algorithm == "Recursive DFS":
                visited = {}
                self.recursive_dfs(start_node_id, visited)
            elif algorithm == "DFS":
                self.dfs(start_node_id)
            elif algorithm == "BFS":
                self.bfs(start_node_id)

    def run_cycle_cancelling(self):
        start_node_id = self.start_node.get()
        end_node_id = self.end_node.get()

        def ford_fulkerson():
            residual = {}
            flows = {}

            for edge in self.edges:
                residual[(edge.from_node_id, edge.to_node_id)] = edge.flux
                residual[(edge.to_node_id, edge.from_node_id)] = 0
                flows[(edge.from_node_id, edge.to_node_id)] = 0

            def bfs(source, sink, parent):
                visited = set()
                queue = deque([source])
                visited.add(source)

                while queue:
                    u = queue.popleft()
                    for v in self.graph.graph.get(u, []):
                        if v not in visited and residual.get((u, v), 0) > 0:
                            parent[v] = u
                            if v == sink:
                                return True
                            queue.append(v)
                            visited.add(v)
                return False

            max_flow = 0
            parent = {}

            while bfs(start_node_id, end_node_id, parent):
                path_flow = float('inf')
                v = end_node_id
                while v != start_node_id:
                    u = parent[v]
                    path_flow = min(path_flow, residual[(u, v)])
                    v = u

                v = end_node_id
                while v != start_node_id:
                    u = parent[v]
                    residual[(u, v)] -= path_flow
                    residual[(v, u)] += path_flow
                    flows[(u, v)] += path_flow
                    v = u

                max_flow += path_flow

            return max_flow, flows

        max_flow, flows = ford_fulkerson()

        def detect_negative_cycle():
            residual = {}
            residual_edges = []
            for edge in self.edges:
                flow = flows.get((edge.from_node_id, edge.to_node_id), 0)

                if flow < edge.flux:
                    residual[(edge.from_node_id, edge.to_node_id)] = edge.flux - flow
                    residual_edges.append((edge.from_node_id, edge.to_node_id, edge.weight))

                reverse_flux = flow
                if reverse_flux > 0:
                    residual[(edge.to_node_id, edge.from_node_id)] = reverse_flux
                    residual_edges.append((edge.to_node_id, edge.from_node_id, -edge.weight))

            distances = {node: float('inf') for node in self.nodes}
            parent = {node: None for node in self.nodes}

            for start_node in self.nodes:
                distances[start_node] = 0

                for _ in range(len(self.nodes) - 1):
                    for u, v, weight in residual_edges:
                        if distances[u] + weight < distances[v]:
                            distances[v] = distances[u] + weight
                            parent[v] = u

                for u, v, weight in residual_edges:
                    if distances[u] + weight < distances[v]:
                        visited = set()
                        cycle = []
                        current = v

                        for _ in range(len(self.nodes)):
                            current = parent[current]

                        cycle_start = current
                        while current not in visited:
                            visited.add(current)
                            current = parent[current]

                        cycle = [cycle_start]
                        node = parent[cycle_start]
                        while node != cycle_start:
                            cycle.append(node)
                            node = parent[node]
                        cycle.append(cycle_start)

                        return cycle[::-1], residual

            return None, residual

        total_cost = 0
        while True:
            negative_cycle, residual = detect_negative_cycle()
            if not negative_cycle:
                break

            min_capacity = float('inf')
            while True:
                negative_cycle, residual = detect_negative_cycle()
                if not negative_cycle:
                    break

                min_capacity = float('inf')
                for i in range(len(negative_cycle) - 1):
                    from_node = negative_cycle[i]
                    to_node = negative_cycle[i + 1]

                    residual_capacity = residual.get((from_node, to_node), 0)
                    if residual_capacity > 0:
                        min_capacity = min(min_capacity, residual_capacity)

                for i in range(len(negative_cycle) - 1):
                    from_node = negative_cycle[i]
                    to_node = negative_cycle[i + 1]

                    if (to_node, from_node) in residual:
                        residual[(to_node, from_node)] += min_capacity
                    else:
                        residual[(to_node, from_node)] = min_capacity

                    flows[(from_node, to_node)] = flows.get((from_node, to_node), 0) + min_capacity
                    flows[(to_node, from_node)] = flows.get((to_node, from_node), 0) - min_capacity

            for edge in self.edges:
                capacity = edge.flux
                flow = flows.get((edge.from_node_id, edge.to_node_id), 0)
                combined_text = f"{flow}/{capacity}, {edge.weight}"
                total_cost += flow * edge.weight
                self.canvas.itemconfig(edge.weight_flux_text_id, text=str(combined_text))
                if edge.bg_text_id:
                    rect_padding_x = 5 * len(combined_text)
                    rect_padding_y = 10
                    self.canvas.coords(edge.bg_text_id,
                                       (self.canvas.coords(edge.weight_flux_text_id)[0] - rect_padding_x,
                                        self.canvas.coords(edge.weight_flux_text_id)[1] - rect_padding_y,
                                        self.canvas.coords(edge.weight_flux_text_id)[0] + rect_padding_x,
                                        self.canvas.coords(edge.weight_flux_text_id)[1] + rect_padding_y))

        full_text = f"{max_flow}, {total_cost}"

        x, y = self.nodes[end_node_id].x, self.nodes[end_node_id].y
        text_id = self.canvas.create_text(x + 25, y, text=full_text, fill="black")
        self.topo_text_ids.append(text_id)

    def run_ford_fulkerson(self):
        start_node_id = self.start_node.get()
        end_node_id = self.end_node.get()

        residual = {}
        for edge in self.edges:
            residual[(edge.from_node_id, edge.to_node_id)] = edge.flux
            residual[(edge.to_node_id, edge.from_node_id)] = 0

        def bfs(source, sink, parent):
            visited = set()
            queue = deque([source])
            visited.add(source)

            while queue:
                u = queue.popleft()
                for v in self.graph.graph.get(u, []):
                    if v not in visited and residual[(u, v)] > 0:
                        parent[v] = u
                        if v == sink:
                            return True
                        queue.append(v)
                        visited.add(v)
            return False

        max_flow = 0
        parent = {}

        while bfs(start_node_id, end_node_id, parent):
            path_flow = float('inf')
            v = end_node_id
            while v != start_node_id:
                u = parent[v]
                path_flow = min(path_flow, residual[(u, v)])
                v = u

            v = end_node_id
            while v != start_node_id:
                u = parent[v]
                residual[(u, v)] -= path_flow
                residual[(v, u)] += path_flow
                v = u

            max_flow += path_flow

            v = end_node_id
            while v != start_node_id:
                u = parent[v]
                edge = next((e for e in self.edges if e.from_node_id == u and e.to_node_id == v), None)
                if edge:
                    self.canvas.itemconfig(edge.line_id, fill="orange", width=2)
                v = u
            self.master.update()
            self.master.after(500)

            for edge in self.edges:
                u, w = edge.from_node_id, edge.to_node_id
                capacitate = edge.flux
                rez = residual.get((u, w), 0)
                flux_curent = capacitate - rez
                combined_text = f"{flux_curent}/{edge.flux}, {edge.weight}"
                self.canvas.itemconfig(edge.weight_flux_text_id, text=str(combined_text))
                if edge.bg_text_id:
                    rect_padding_x = 5 * len(combined_text)
                    rect_padding_y = 10
                    self.canvas.coords(edge.bg_text_id,
                                       (self.canvas.coords(edge.weight_flux_text_id)[0] - rect_padding_x,
                                        self.canvas.coords(edge.weight_flux_text_id)[1] - rect_padding_y,
                                        self.canvas.coords(edge.weight_flux_text_id)[0] + rect_padding_x,
                                        self.canvas.coords(edge.weight_flux_text_id)[1] + rect_padding_y))

        x, y = self.nodes[end_node_id].x, self.nodes[end_node_id].y
        text_id = self.canvas.create_text(x + 25, y, text=max_flow, fill="black")
        self.topo_text_ids.append(text_id)

    def run_bellman_ford(self, start_node_id):
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_node_id] = 0

        for _ in range(len(self.nodes) - 1):
            for edge in self.edges:
                if distances[edge.from_node_id] != float('inf') and \
                        distances[edge.from_node_id] + edge.weight < distances[edge.to_node_id]:
                    distances[edge.to_node_id] = distances[edge.from_node_id] + edge.weight

        for edge in self.edges:
            if distances[edge.from_node_id] != float('inf') and \
                    distances[edge.from_node_id] + edge.weight < distances[edge.to_node_id]:
                messagebox.showerror("Negative Cycle Detected",
                                     "Graph contains a negative weight cycle. Bellman-Ford cannot proceed.")
                return

        for node_id, distance in distances.items():
            x, y = self.nodes[node_id].x, self.nodes[node_id].y
            distance_text = f"{distance}" if distance != float('inf') else "∞"
            text_id = self.canvas.create_text(x + 25, y, text=distance_text, fill="black")
            self.topo_text_ids.append(text_id)

        end_node_id = self.end_node.get()
        if distances[end_node_id] != float('inf'):
            current_node = end_node_id
            path = []

            while current_node != start_node_id:
                for edge in self.edges:
                    if edge.to_node_id == current_node and distances[edge.from_node_id] + edge.weight == distances[current_node]:
                        path.append(edge)
                        current_node = edge.from_node_id
                        break

            for edge in path:
                self.canvas.itemconfig(edge.line_id, fill="orange", width=2)

        return distances

    def run_dijkstra(self, start_node_id):
        for edge in self.edges:
            if edge.weight < 0:
                messagebox.showerror("Negative Edge Weight",
                                     "Graph contains negative edge weights. Dijkstra's algorithm cannot handle negative weights.")
                return

        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_node_id] = 0
        pq = [(0, start_node_id)]

        while pq:
            current_distance, current_node = heapq.heappop(pq)

            if current_distance > distances[current_node]:
                continue

            for neighbor in self.graph.graph.get(current_node, []):
                edge = next((e for e in self.edges if
                             (e.from_node_id == current_node and e.to_node_id == neighbor) or
                             (self.graph_type.get() == "undirected" and
                              e.from_node_id == neighbor and e.to_node_id == current_node)), None)

                if edge:
                    new_distance = current_distance + edge.weight
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        heapq.heappush(pq, (new_distance, neighbor))

        for node_id, distance in distances.items():
            x, y = self.nodes[node_id].x, self.nodes[node_id].y
            distance_text = f"{distance}" if distance != float('inf') else "∞"
            text_id = self.canvas.create_text(x + 25, y, text=distance_text, fill="black")
            self.topo_text_ids.append(text_id)

        end_node_id = self.end_node.get()
        if distances[end_node_id] != float('inf'):
            current_node = end_node_id
            path = []

            while current_node != start_node_id:
                for edge in self.edges:
                    if edge.to_node_id == current_node and distances[edge.from_node_id] + edge.weight == distances[
                        current_node]:
                        path.append(edge)
                        current_node = edge.from_node_id
                        break

            for edge in path:
                self.canvas.itemconfig(edge.line_id, fill="orange", width=2)

        return distances

    def dfs_topo(self, node, visited, stack):
        visited[node] = True
        for neighbor in self.graph.graph.get(node, []):
            if not visited[neighbor]:
                self.dfs_topo(neighbor, visited, stack)
        stack.append(node)

    def topological_sort(self):
        visited = {node: False for node in self.graph.graph}

        for node in self.graph.graph:
            for neighbor in self.graph.graph[node]:
                if neighbor not in visited:
                    visited[neighbor] = False

        stack = []
        for node in self.graph.graph:
            if not visited[node]:
                self.dfs_topo(node, visited, stack)

        return stack[::-1]

    def dfs_connected(self, node, visited, component):
        visited.add(node)
        component.append(node)
        for neighbor in self.graph.graph.get(node, []):
            if neighbor not in visited:
                self.dfs_connected(neighbor, visited, component)

    def find_connected_components(self):
        visited = set()
        components = []
        for node in self.graph.graph:
            if node not in visited:
                component = []
                self.dfs_connected(node, visited, component)
                components.append(component)
        return components

    def run_connected_components(self):
        components = self.find_connected_components()
        colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(len(components))]

        for idx, component in enumerate(components):
            color = colors[idx]
            for node_id in component:
                self.canvas.itemconfig(self.nodes[node_id].circle_id, fill=color)

            for edge in self.edges:
                if edge.from_node_id in component and edge.to_node_id in component:
                    self.canvas.itemconfig(edge.line_id, fill=color)

    def dfs_strong(self, node, visited, stack=None):
        visited[node] = True
        for neighbor in self.graph.graph.get(node, []):
            if not visited[neighbor]:
                self.dfs_strong(neighbor, visited, stack)
        if stack is not None:
            stack.append(node)

    def transpose_graph(self):
        transposed = {}
        for node in self.graph.graph:
            for neighbor in self.graph.graph[node]:
                transposed.setdefault(neighbor, []).append(node)
        return transposed

    def dfs_on_transposed(self, graph, v, visited, component):
        visited[v] = True
        component.append(v)
        for neighbor in graph.get(v, []):
            if not visited[neighbor]:
                self.dfs_on_transposed(graph, neighbor, visited, component)

    def kosaraju_scc(self):
        stack = []
        visited = {node: False for node in self.graph.graph}
        for node in self.graph.graph:
            if not visited[node]:
                self.dfs_strong(node, visited, stack)

        g_t = self.transpose_graph()
        visited = {node: False for node in g_t}
        scc_list = []

        while stack:
            node = stack.pop()
            if not visited[node]:
                component = []
                self.dfs_on_transposed(g_t, node, visited, component)
                scc_list.append(component)

        return scc_list

    def run_strongly_connected_components(self):
        components = self.kosaraju_scc()
        colors = ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(len(components))]

        for idx, component in enumerate(components):
            color = colors[idx]
            for node_id in component:
                self.canvas.itemconfig(self.nodes[node_id].circle_id, fill=color)

            for edge in self.edges:
                if edge.from_node_id in component and edge.to_node_id in component:
                    self.canvas.itemconfig(edge.line_id, fill=color)

    def recursive_dfs(self, node_id, visited):
        visited[node_id] = True
        self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
        self.master.update()
        self.master.after(500)

        for neighbor in self.graph.graph.get(node_id, []):
            if neighbor not in visited:
                self.recursive_dfs(neighbor, visited)

    def dfs(self, start_node_id):
        stack = [start_node_id]
        visited = {}

        while stack:
            node_id = stack.pop()
            if node_id not in visited:
                visited[node_id] = True
                self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
                self.master.update()
                self.master.after(500)

                for neighbor in reversed(self.graph.graph.get(node_id, [])):
                    if neighbor not in visited:
                        stack.append(neighbor)

    def bfs(self, start_node_id):
        visited = {start_node_id: True}
        queue = deque([start_node_id])

        while queue:
            node_id = queue.popleft()
            self.canvas.itemconfig(self.nodes[node_id].circle_id, fill="yellow")
            self.master.update()
            self.master.after(500)

            for neighbor in self.graph.graph.get(node_id, []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited[neighbor] = True

    def is_undirected_tree(self):
        if len(self.nodes) == 0:
            return True

        if len(self.edges) != len(self.nodes) - 1:
            return False

        visited = set()
        self.dfs_connected(next(iter(self.nodes)), visited, [])

        return len(visited) == len(self.nodes)

    def is_directed_tree(self):
        if len(self.nodes) == 0:
            return True

        visited = {node: False for node in self.nodes}
        in_degree = {node: 0 for node in self.nodes}

        for edge in self.edges:
            in_degree[edge.to_node_id] += 1

        root_count = sum(1 for degree in in_degree.values() if degree == 0)
        if root_count != 1:
            return False

        visited = {}
        for node_id in self.nodes:
            if node_id not in visited:
                if self.dfs_cycle_check(node_id, visited, set()):
                    return False

        return True

    def dfs_cycle_check(self, node_id, visited, rec_stack):
        visited[node_id] = True
        rec_stack.add(node_id)

        for neighbor in self.graph.graph.get(node_id, []):
            if neighbor not in visited:
                if self.dfs_cycle_check(neighbor, visited, rec_stack):
                    return True
            elif neighbor in rec_stack:
                return True

        rec_stack.remove(node_id)
        return False

    def find_center(self):
        if self.graph_type.get() != "undirected":
            return None

        max_distances = {}

        for node_id in self.nodes:
            distances = self.bfs_distances(node_id)
            max_distances[node_id] = max(distances.values())

        center_node = min(max_distances, key=max_distances.get)

        self.canvas.itemconfig(self.nodes[center_node].circle_id, fill="yellow")
        return center_node

    def bfs_distances(self, start_node_id):
        distances = {node_id: float('inf') for node_id in self.nodes}
        distances[start_node_id] = 0
        queue = deque([start_node_id])

        while queue:
            node_id = queue.popleft()
            current_distance = distances[node_id]

            for neighbor in self.graph.graph.get(node_id, []):
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        return distances

    def run_prim_mst(self):
        if self.graph_type.get() != "undirected":
            return

        if not self.nodes:
            return

        start_node = next(iter(self.nodes))
        visited = set()
        mst_edges = []
        min_heap = []

        def add_edges(node):
            visited.add(node)
            for neighbor in self.graph.graph.get(node, []):
                edge = next((e for e in self.edges if
                             (e.from_node_id == node and e.to_node_id == neighbor) or
                             (e.from_node_id == neighbor and e.to_node_id == node)), None)
                if edge and neighbor not in visited:
                    heapq.heappush(min_heap, (edge.weight, node, neighbor, edge))

        add_edges(start_node)

        while min_heap and len(visited) < len(self.nodes):
            weight, from_node, to_node, edge = heapq.heappop(min_heap)

            if to_node in visited:
                continue

            self.canvas.itemconfig(edge.line_id, fill="orange", width=2)
            self.master.update()
            self.master.after(500)

            mst_edges.append(edge)
            add_edges(to_node)

        for edge in mst_edges:
            self.canvas.itemconfig(edge.line_id, fill="red", width=2)

        total_weight = sum(edge.weight for edge in mst_edges)

        start_node_id = self.start_node.get()
        x, y = self.nodes[start_node_id].x, self.nodes[start_node_id].y
        text_id = self.canvas.create_text(x + 25, y, text=total_weight, fill="black")
        self.topo_text_ids.append(text_id)

    def run_kruskal_mst(self):
        if self.graph_type.get() != "undirected":
            return

        if not self.nodes:
            return

        parent = {node_id: node_id for node_id in self.nodes}
        rank = {node_id: 0 for node_id in self.nodes}

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    rank[root1] += 1

        sorted_edges = sorted(self.edges, key=lambda edge: edge.weight)
        mst_edges = []
        total_weight = 0

        for edge in sorted_edges:
            if find(edge.from_node_id) != find(edge.to_node_id):
                union(edge.from_node_id, edge.to_node_id)
                mst_edges.append(edge)
                total_weight += edge.weight

                self.canvas.itemconfig(edge.line_id, fill="orange", width=2)
                self.master.update()
                self.master.after(500)

        for edge in mst_edges:
            self.canvas.itemconfig(edge.line_id, fill="red", width=2)

        start_node_id = self.start_node.get()
        x, y = self.nodes[start_node_id].x, self.nodes[start_node_id].y
        text_id = self.canvas.create_text(x + 25, y, text=total_weight, fill="black")
        self.topo_text_ids.append(text_id)

    def run_boruvka_mst(self):
        if self.graph_type.get() != "undirected":
            return

        if not self.nodes:
            return

        parent = {node_id: node_id for node_id in self.nodes}
        rank = {node_id: 0 for node_id in self.nodes}

        def find(node):
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        def union(node1, node2):
            root1 = find(node1)
            root2 = find(node2)
            if root1 != root2:
                if rank[root1] > rank[root2]:
                    parent[root2] = root1
                elif rank[root1] < rank[root2]:
                    parent[root1] = root2
                else:
                    parent[root2] = root1
                    rank[root1] += 1

        mst_edges = []
        total_weight = 0

        num_components = len(self.nodes)
        while num_components > 1:
            cheapest = {}

            for edge in self.edges:
                root1 = find(edge.from_node_id)
                root2 = find(edge.to_node_id)
                if root1 != root2:
                    if root1 not in cheapest or cheapest[root1].weight > edge.weight:
                        cheapest[root1] = edge
                    if root2 not in cheapest or cheapest[root2].weight > edge.weight:
                        cheapest[root2] = edge

            for edge in cheapest.values():
                root1 = find(edge.from_node_id)
                root2 = find(edge.to_node_id)
                if root1 != root2:
                    union(root1, root2)
                    mst_edges.append(edge)
                    total_weight += edge.weight
                    num_components -= 1

                    self.canvas.itemconfig(edge.line_id, fill="orange", width=2)
                    self.master.update()
                    self.master.after(500)

        for edge in mst_edges:
            self.canvas.itemconfig(edge.line_id, fill="red", width=2)

        start_node_id = self.start_node.get()
        x, y = self.nodes[start_node_id].x, self.nodes[start_node_id].y
        text_id = self.canvas.create_text(x + 25, y, text=total_weight, fill="black")
        self.topo_text_ids.append(text_id)

    def build_tree_from_center(self):
        center_node = self.find_center()
        if center_node is None:
            return

        visited = set()
        tree_edges = []

        queue = deque([center_node])
        visited.add(center_node)

        while queue:
            node_id = queue.popleft()

            for neighbor in self.graph.graph.get(node_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)

                    edge = next((e for e in self.edges if
                                 (e.from_node_id == node_id and e.to_node_id == neighbor) or
                                 (e.from_node_id == neighbor and e.to_node_id == node_id)), None)
                    weight = edge.weight if edge else 1
                    flux = edge.flux if edge else 1

                    tree_edges.append((node_id, neighbor, weight, flux))
                    queue.append(neighbor)

        self.semi_clear_canvas()
        self.display_tree(tree_edges)

    def display_tree(self, tree_edges):
        levels = {}
        root = tree_edges[0][0]
        self.calculate_levels(root, levels)

        level_spacing_y = 100
        node_spacing_x = 80

        level_positions = {}
        for node_id, level in levels.items():
            if level not in level_positions:
                level_positions[level] = []
            level_positions[level].append(node_id)

        x_center = self.canvas.winfo_width() // 2
        y_start = 50

        for level, nodes in level_positions.items():
            level_width = (len(nodes) - 1) * node_spacing_x
            x_start = x_center - level_width // 2

            for i, node_id in enumerate(nodes):
                node = self.nodes[node_id]
                new_x = x_start + i * node_spacing_x
                new_y = y_start + level * level_spacing_y

                node.circle_id = self.canvas.create_oval(new_x - 20, new_y - 20, new_x + 20, new_y + 20,
                                                         fill="lightgreen")
                node.text_id = self.canvas.create_text(new_x, new_y, text=str(node_id))
                node.x, node.y = new_x, new_y

        for from_node_id, to_node_id, weight, flux in tree_edges:
            from_node = self.nodes[from_node_id]
            to_node = self.nodes[to_node_id]

            line_id = self.canvas.create_line(from_node.x, from_node.y, to_node.x, to_node.y, fill="black")

            mid_x = (from_node.x + to_node.x) / 2
            mid_y = (from_node.y + to_node.y) / 2

            rect_padding_x = 5
            rect_padding_y = 10
            rect_id = self.canvas.create_rectangle(mid_x - rect_padding_x * 2, mid_y - rect_padding_y,
                                                   mid_x + rect_padding_x * 2, mid_y + rect_padding_y,
                                                   fill="white", outline="")

            combined_text = f"{weight}, {flux}"

            weight_flux_text_id = self.canvas.create_text(mid_x, mid_y, text=str(combined_text), fill="blue")

            self.edges.append(Edge(from_node_id, to_node_id, line_id, weight, flux))
            self.edges[-1].weight_flux_text_id = weight_flux_text_id
            self.edges[-1].bg_text_id = rect_id

    def calculate_levels(self, root, levels):
        queue = deque([(root, 0)])
        levels[root] = 0

        while queue:
            node_id, level = queue.popleft()
            for neighbor in self.graph.graph.get(node_id, []):
                if neighbor not in levels:
                    levels[neighbor] = level + 1
                    queue.append((neighbor, level + 1))

    def semi_clear_canvas(self):
        self.canvas.delete("all")
        self.edges.clear()
        self.update_start_node_menu()
        self.update_end_node_menu()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.nodes.clear()
        self.edges.clear()
        self.graph = Graph()
        self.node_counter = 0
        self.update_start_node_menu()
        self.update_end_node_menu()

if __name__ == '__main__':
    root = tk.Tk()
    app = GraphGUI(root)
    root.mainloop()