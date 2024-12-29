class Node:
    def __init__(self, node_id, x, y, circle_id, text_id):
        self.id = node_id
        self.x = x
        self.y = y
        self.circle_id = circle_id
        self.text_id = text_id


class Edge:
    def __init__(self, from_node_id, to_node_id, line_id, weight=1, flux=0):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.line_id = line_id
        self.weight = weight
        self.flux = flux
        self.weight_flux_text_id = None
        self.bg_text_id = None

    def __lt__(self, other):
        return self.weight < other.weight