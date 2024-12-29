import numpy as np
from collections import deque
import tkinter as tk
import tkinter.messagebox as messagebox
import math
import heapq
from heapq import heappop, heappush
import time
import matplotlib.pyplot as plt

VERBOSE = True
NUM_NODES = 50
DENSITY = 0.5
DIRECTED = False
CONNEX = True

def is_connected(adj_list, n):
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for (neighbor, _) in adj_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)
    return np.all(visited)

def generate_erdos_renyi_graph(n, p, isDirected=False, isConnected=False, max_iter=1000, weightLimit=10, fluxLimit=5):
    for _ in range(max_iter):
        A = np.random.rand(n, n)
        np.fill_diagonal(A, 1.0)
        edges = (A < p).astype(np.uint8)
        np.fill_diagonal(edges, 0)
        if not isDirected:
            edges = np.bitwise_or(edges, edges.T)
        if not isConnected:
            weights = np.zeros((n, n), dtype=int)
            fluxes = np.zeros((n, n), dtype=int)
            mask = (edges == 1)
            weights[mask] = np.random.randint(1, weightLimit + 1, size=np.count_nonzero(mask))
            fluxes[mask] = np.random.randint(1, fluxLimit + 1, size=np.count_nonzero(mask))
            return edges, weights, fluxes

        undirected_edges = np.bitwise_or(edges, edges.T)
        adj_list = [np.where(undirected_edges[i] == 1)[0] for i in range(n)]
        if is_connected([[(x,0) for x in row] for row in adj_list], n):
            break
    else:
        raise ValueError("Could not generate a connected graph within the allowed number of attempts.")

    weights = np.zeros((n, n), dtype=int)
    fluxes = np.zeros((n, n), dtype=int)
    mask = (edges == 1)
    weights[mask] = np.random.randint(1, weightLimit + 1, size=np.count_nonzero(mask))
    fluxes[mask] = np.random.randint(1, fluxLimit + 1, size=np.count_nonzero(mask))

    if not isDirected:
        weights = np.minimum(weights, weights.T)

    return edges, weights, fluxes

def generate_node_coordinates(n, width=600, height=800):
    cx, cy = width / 2, height / 2
    r = min(width, height) / 2 - 50
    angle_step = 2 * math.pi / n
    coords = []
    for i in range(n):
        angle = i * angle_step
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        coords.append((x, y))
    return np.array(coords)

def build_adj_list(n, edges, weights):
    adj_list = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if edges[i, j] == 1:
                w = weights[i, j]
                adj_list[i].append((j, w))
    return adj_list

def build_undirected_edge_list():
    global edges, weights, fluxes
    edge_list = []
    n = len(fluxes)
    for i in range(n):
        for j in range(i+1, n):
            if edges[i, j] == 1 or edges[j, i] == 1:
                w = weights[i, j] if weights[i, j] != 0 else weights[j, i]
                f = fluxes[i, j] if fluxes[i, j] != 0 else fluxes[j, i]
                edge_list.append((i, j, f, w))
    return edge_list

def eager_prim(n, adj_list):
    mst_edges = []
    visited = [False]*n
    visited[0] = True
    pq = []
    for (nbr, w) in adj_list[0]:
        heapq.heappush(pq, (w, 0, nbr))
    total_weight = 0
    while pq and len(mst_edges) < n-1:
        w, u, v = heapq.heappop(pq)
        if not visited[v]:
            visited[v] = True
            mst_edges.append((u, v))
            total_weight += w
            for (x, ww) in adj_list[v]:
                if not visited[x]:
                    heapq.heappush(pq, (ww, v, x))
    return mst_edges, total_weight

def kruskal(n, edge_list):
    edge_list.sort(key=lambda x: x[2])

    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if rank[root_a] < rank[root_b]:
                parent[root_a] = root_b
            elif rank[root_a] > rank[root_b]:
                parent[root_b] = root_a
            else:
                parent[root_b] = root_a
                rank[root_a] += 1
            return True
        return False

    mst_edges = []
    total_weight = 0

    for u, v, w, _ in edge_list:
        if union(u, v):
            mst_edges.append((u, v))
            total_weight += w
            if len(mst_edges) == n - 1:
                break

    return mst_edges, total_weight

def boruvka(n, edge_list):
    parent = list(range(n))
    rank = [0] * n

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a, b):
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            if rank[root_a] < rank[root_b]:
                parent[root_a] = root_b
            elif rank[root_b] < rank[root_a]:
                parent[root_b] = root_a
            else:
                parent[root_b] = root_a
                rank[root_a] += 1
            return True
        return False

    mst_edges = []
    total_weight = 0
    components = n

    while components > 1:
        cheapest = [-1] * n
        cheapest_weight = [float('inf')] * n

        for u, v, w, _ in edge_list:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                if w < cheapest_weight[root_u]:
                    cheapest_weight[root_u] = w
                    cheapest[root_u] = (u, v, w)
                if w < cheapest_weight[root_v]:
                    cheapest_weight[root_v] = w
                    cheapest[root_v] = (u, v, w)

        for i in range(n):
            if cheapest[i] != -1:
                u, v, w = cheapest[i]
                if union(u, v):
                    mst_edges.append((u, v))
                    total_weight += w
                    components -= 1

    return mst_edges, total_weight

def clear_path_lines(canvas):
    for line_id in path_lines:
        canvas.itemconfig(line_id, fill="black", width=1)
    path_lines.clear()

def dijkstra():
    n = len(fluxes)
    dist = np.full(n, np.inf, dtype=float)
    prev = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)

    dist[start_node] = 0
    pq = [(0, start_node)]

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True

        if u == end_node:
            break

        neighbors = np.array([v for v, w in adj_list[u]], dtype=int)
        weights = np.array([w for _, w in adj_list[u]], dtype=float)

        new_dist = dist[u] + weights
        relax_mask = ~visited[neighbors] & (new_dist < dist[neighbors])

        dist[neighbors[relax_mask]] = new_dist[relax_mask]
        prev[neighbors[relax_mask]] = u
        for v, new_d in zip(neighbors[relax_mask], new_dist[relax_mask]):
            heapq.heappush(pq, (new_d, v))

    if dist[end_node] == np.inf:
        if verbose: messagebox.showerror("No path", "No path found by Dijkstra.")
        return None, None

    path = []
    cur = end_node
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return path, dist[end_node]

def bellman_ford():
    n = len(fluxes)
    all_edges = [(u, v, w) for u in range(n) for v, w in adj_list[u]]
    edges_arr = np.array(all_edges, dtype=np.int64)

    dist = np.full(n, np.inf)
    prev = np.full(n, -1, dtype=np.int64)
    dist[start_node] = 0

    for _ in range(n - 1):
        u = edges_arr[:, 0]
        v = edges_arr[:, 1]
        w = edges_arr[:, 2]
        relax_mask = (dist[u] + w < dist[v])
        dist[v[relax_mask]] = dist[u[relax_mask]] + w[relax_mask]
        prev[v[relax_mask]] = u[relax_mask]

    if dist[end_node] == np.inf:
        if verbose: messagebox.showerror("No path", "No path found by Bellman-Ford.")
        return None, None

    path = []
    cur = end_node
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return path, dist[end_node]

def ford_fulkerson():
    n = len(fluxes)
    capacities = np.array(fluxes, dtype=int)
    residual_capacities = capacities.copy()
    parent = np.full(n, -1, dtype=int)

    def bfs():
        visited = np.zeros(n, dtype=bool)
        queue = deque([start_node])
        visited[start_node] = True
        while queue:
            u = queue.popleft()
            for v in range(n):
                if not visited[v] and residual_capacities[u, v] > 0:
                    parent[v] = u
                    if v == end_node:
                        return True
                    queue.append(v)
                    visited[v] = True
        return False

    max_flow = 0
    while bfs():
        path_flow = float('Inf')
        s = end_node
        while s != start_node:
            path_flow = min(path_flow, residual_capacities[parent[s], s])
            s = parent[s]

        v = end_node
        while v != start_node:
            u = parent[v]
            residual_capacities[u, v] -= path_flow
            residual_capacities[v, u] += path_flow
            v = parent[v]

        max_flow += path_flow

    return max_flow, residual_capacities

def find_negative_cycle_with_priority_queue(residual_capacities, residual_costs, n):
    dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)
    dist[0] = 0
    pq = [(0, 0)]

    while pq:
        d, u = heappop(pq)
        if d > dist[u]:
            continue

        for v in range(n):
            if residual_capacities[u, v] > 0 and dist[u] + residual_costs[u, v] < dist[v]:
                dist[v] = dist[u] + residual_costs[u, v]
                parent[v] = u
                heappush(pq, (dist[v], v))

    for u in range(n):
        for v in range(n):
            if residual_capacities[u, v] > 0 and dist[u] + residual_costs[u, v] < dist[v]:
                return parent, v
    return None, None

def cycle_cancelling():
    n = len(fluxes)
    capacities = np.array(fluxes, dtype=int)
    costs = np.array(weights, dtype=int)
    max_flow, residual_capacities = ford_fulkerson()
    residual_costs = costs.copy()
    total_cost = np.sum((capacities - residual_capacities) * residual_costs)

    def find_negative_cycle():
        return find_negative_cycle_with_priority_queue(residual_capacities, residual_costs, n)

    while True:
        parent, cycle_node = find_negative_cycle()
        if cycle_node is None:
            break

        cycle = []
        visited = set()
        while cycle_node not in visited:
            visited.add(cycle_node)
            cycle_node = parent[cycle_node]

        start = cycle_node
        cycle.append(start)
        cycle_node = parent[start]
        while cycle_node != start:
            cycle.append(cycle_node)
            cycle_node = parent[cycle_node]

        cycle_flow = min(residual_capacities[cycle[i], cycle[(i + 1) % len(cycle)]] for i in range(len(cycle)))

        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            residual_capacities[u, v] -= cycle_flow
            residual_capacities[v, u] += cycle_flow
            total_cost += cycle_flow * residual_costs[u, v]

    return max_flow, total_cost

def run_alg(alg_type, canvas, node_positions, line_refs, adj_list, edge_list):
    for line_id in mst_lines:
        canvas.itemconfig(line_id, fill="black", width=1)
    mst_lines.clear()
    clear_path_lines(canvas)

    n = len(node_positions)
    if alg_type in ["Prim", "Kruskal", "Boruvka"]:
        if not is_connected(adj_list, n):
            messagebox.showerror("Not Connected", "MST Algorithms cant run on this graph.")
            return

        if alg_type == "Prim":
            mst_edges, total_weight = eager_prim(n, adj_list)
        elif alg_type == "Kruskal":
            mst_edges, total_weight = kruskal(n, edge_list)
        elif alg_type == "Boruvka":
            mst_edges, total_weight = boruvka(n, edge_list)

        if total_weight == None and mst_edges == None:
            return

        print("Total MST Weight ({}) = {}".format(alg_type, total_weight))
        for (u, v) in mst_edges:
            if (u, v) in line_refs:
                line_id = line_refs[(u, v)]
                canvas.itemconfig(line_id, fill="orange", width=2)
                canvas.tag_raise(line_id)
                mst_lines.append(line_id)
            elif (v, u) in line_refs:
                line_id = line_refs[(v, u)]
                canvas.itemconfig(line_id, fill="orange", width=2)
                canvas.tag_raise(line_id)
                mst_lines.append(line_id)

    elif alg_type in ["Dijkstra", "Bellman-Ford"]:
        if start_node is None or end_node is None:
            messagebox.showerror("No start/end node(s)", "Select start and end nodes first.")
            return

        if alg_type == "Dijkstra":
            path, dist = dijkstra()
        elif alg_type == "Bellman-Ford":
            path, dist = bellman_ford()

        if path == None and dist == None:
            return

        print("Path distance ({}) = {}".format(alg_type, int(dist)))

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            if (u, v) in line_refs:
                line_id = line_refs[(u, v)]
                canvas.itemconfig(line_id, fill="orange", width=2)
                canvas.tag_raise(line_id)
                path_lines.append(line_id)
            elif (v, u) in line_refs:
                line_id = line_refs[(v, u)]
                canvas.itemconfig(line_id, fill="orange", width=2)
                canvas.tag_raise(line_id)
                path_lines.append(line_id)


    elif alg_type in ["Ford-Fulkerson", "Cycle Cancelling"]:
        if start_node is None or end_node is None:
            messagebox.showerror("No start/end node(s)", "Select start and end nodes first.")
            return

        if alg_type == "Ford-Fulkerson":
            max_flow, _ = ford_fulkerson()
            print("Max Flow (Ford-Fulkerson) = {}".format(max_flow))
        elif alg_type == "Cycle Cancelling":
            min_cost_flow, total_cost = cycle_cancelling()
            print("Min Cost Flow (Cycle Cancelling) = {}, Total Cost = {}".format(min_cost_flow, total_cost))

if __name__ == "__main__":
    np.random.seed(42)
    verbose = VERBOSE

    if verbose:
        mst_lines = []
        path_lines = []

        n = NUM_NODES
        p = DENSITY
        isDirected = DIRECTED
        isConnected = CONNEX
        verbose = False
        edges, weights, fluxes = generate_erdos_renyi_graph(n, p, isDirected, isConnected, weightLimit=10, fluxLimit=5)
        node_positions = generate_node_coordinates(n, 600, 800)

        adj_list = build_adj_list(n, edges, weights)
        undirected_edge_list = build_undirected_edge_list()

        root = tk.Tk()
        root.title("Graph Generator")

        control_frame = tk.Frame(root, width=200, height=800)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        if isDirected:
            start_label = tk.Label(control_frame, text="Start Node: None", font=("Arial", 12))
            start_label.pack(pady=5)
            end_label = tk.Label(control_frame, text="End Node: None", font=("Arial", 12))
            end_label.pack(pady=5)

        canvas = tk.Canvas(root, width=600, height=800, bg="white")
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        line_refs = {}
        for i in range(n):
            for (j, w) in adj_list[i]:
                line_id = canvas.create_line(node_positions[i][0], node_positions[i][1],
                                             node_positions[j][0], node_positions[j][1],
                                             width=1, fill="black")
                line_refs[(i,j)] = line_id

        node_refs = []
        for i in range(n):
            x, y = node_positions[i]

        start_node = None
        end_node = None
        start_marker = None
        end_marker = None

        def find_nearest_node(x_click, y_click, threshold=15):
            nearest = None
            min_dist = float('inf')
            for i, (x, y) in enumerate(node_positions):
                dist = ((x - x_click)**2 + (y - y_click)**2)**0.5
                if dist < min_dist and dist <= threshold:
                    min_dist = dist
                    nearest = i
            return nearest

        def on_left_click(event):
            if not isDirected:
                return

            global start_node, end_node, start_marker
            i = find_nearest_node(event.x, event.y)
            if i is not None:
                if start_marker is not None:
                    canvas.delete(start_marker)
                    start_marker = None
                start_node = i
                start_label.config(text=f"Start Node: {start_node}")
                x, y = node_positions[i]
                start_marker = canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="green", fill="green")

        def on_right_click(event):
            if not isDirected:
                return

            global start_node, end_node, end_marker
            i = find_nearest_node(event.x, event.y)
            if i is not None:
                if end_marker is not None:
                    canvas.delete(end_marker)
                    end_marker = None
                end_node = i
                end_label.config(text=f"End Node: {end_node}")
                x, y = node_positions[i]
                end_marker = canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="red", fill="red")

        def on_ctrl_left_click(event):
            on_right_click(event)

        canvas.bind("<Button-1>", on_left_click)
        canvas.bind("<Button-3>", on_right_click)
        canvas.bind("<Control-Button-1>", on_ctrl_left_click)

        def run_prim():
            run_alg("Prim", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_kruskal():
            run_alg("Kruskal", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_boruvka():
            run_alg("Boruvka", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_dijkstra():
            run_alg("Dijkstra", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_bellman_ford():
            run_alg("Bellman-Ford", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_ford_fulkerson():
            run_alg("Ford-Fulkerson", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        def run_cycle_cancelling():
            run_alg("Cycle Cancelling", canvas, node_positions, line_refs, adj_list, undirected_edge_list)

        if not isDirected:
            prim_button = tk.Button(control_frame, text="Eager Prim", command=run_prim)
            prim_button.pack(pady=5)

            kruskal_button = tk.Button(control_frame, text="Kruskal", command=run_kruskal)
            kruskal_button.pack(pady=5)

            boruvka_button = tk.Button(control_frame, text="Boruvka", command=run_boruvka)
            boruvka_button.pack(pady=5)

        else:
            dijkstra_button = tk.Button(control_frame, text="Dijkstra", command=run_dijkstra)
            dijkstra_button.pack(pady=5)

            bellman_button = tk.Button(control_frame, text="Bellman-Ford", command=run_bellman_ford)
            bellman_button.pack(pady=5)

            fulkerson_button = tk.Button(control_frame, text="Ford-Fulkerson", command=run_ford_fulkerson)
            fulkerson_button.pack(pady=5)

            cycle_button = tk.Button(control_frame, text="Cycle Cancelling", command=run_cycle_cancelling)
            cycle_button.pack(pady=5)

        root.mainloop()


    else:

        def test_algorithm(algorithm, *args):
            start_time = time.time()
            result = algorithm(*args)
            end_time = time.time()
            return end_time - start_time, result


        def plot_results(results, title):
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
            densities = [0.3, 0.6, 1]

            for idx, density in enumerate(densities):
                ax = axs[idx]
                for alg_name, data in results.items():
                    filtered_data = [entry for entry in data if entry['density'] == density]
                    x = [entry['nodes'] for entry in filtered_data]
                    y = [entry['time'] for entry in filtered_data]
                    ax.plot(x, y, label=alg_name)

                ax.set_xlabel('Numărul de noduri')
                ax.set_ylabel('Timp de execuție (secunde)')
                ax.set_title(f'Densitate = {density}')
                ax.legend()
                ax.grid(True)

            plt.suptitle(title)
            plt.tight_layout()
            plt.show()

        results_undirected = {
            'Eager Prim': [],
            'Kruskal': [],
            'Boruvka': []
        }

        for n in [50, 100, 500, 1000]:
            for density in [0.3, 0.6, 1]:
                edges, weights, fluxes = generate_erdos_renyi_graph(n, density, False, True, weightLimit=10,
                                                                    fluxLimit=5)
                adj_list = build_adj_list(n, edges, weights)
                edge_list = build_undirected_edge_list()

                times = []
                for _ in range(10):
                    time_taken, _ = test_algorithm(eager_prim, n, adj_list)
                    times.append(time_taken)
                mean_time = np.mean(times)
                results_undirected['Eager Prim'].append({'nodes': n, 'time': mean_time, 'density': density})

                times = []
                for _ in range(10):
                    time_taken, _ = test_algorithm(kruskal, n, edge_list)
                    times.append(time_taken)
                mean_time = np.mean(times)
                results_undirected['Kruskal'].append({'nodes': n, 'time': mean_time, 'density': density})

                times = []
                for _ in range(10):
                    time_taken, _ = test_algorithm(boruvka, n, edge_list)
                    times.append(time_taken)
                mean_time = np.mean(times)
                results_undirected['Boruvka'].append({'nodes': n, 'time': mean_time, 'density': density})

        plot_results(results_undirected, 'Performanța algoritmilor pe grafuri neorientate')

        results_directed = {
            'Dijkstra': [],
            'Bellman-Ford': [],
            'Ford-Fulkerson': [],
            'Cycle Cancelling': []
        }

        for n in [50, 100, 500, 1000]:
            for density in [0.3, 0.6, 1]:
                edges, weights, fluxes = generate_erdos_renyi_graph(n, density, True, True, weightLimit=10,
                                                                    fluxLimit=5)
                adj_list = build_adj_list(n, edges, weights)
                start_node = np.random.randint(0, n - 1)
                end_node = np.random.randint(0, n - 1)

                time_taken, _ = test_algorithm(dijkstra)
                results_directed['Dijkstra'].append({'nodes': n, 'time': time_taken, 'density': density})

                time_taken, _ = test_algorithm(bellman_ford)
                results_directed['Bellman-Ford'].append({'nodes': n, 'time': time_taken, 'density': density})

                time_taken, _ = test_algorithm(ford_fulkerson)
                results_directed['Ford-Fulkerson'].append({'nodes': n, 'time': time_taken, 'density': density})

                time_taken, _ = test_algorithm(cycle_cancelling)
                results_directed['Cycle Cancelling'].append({'nodes': n, 'time': time_taken, 'density': density})

        plot_results(results_directed, 'Performanța algoritmilor pe grafuri orientate')

'''
'''