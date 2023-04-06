from collections import defaultdict


class Graph:

    def __init__(self):
        self.edges = defaultdict(lambda: 0)
        self.weights = defaultdict(lambda: 0)

    def add_var(self, pred, var, weight):
        self.edges[(pred, var)] = weight
        self.edges[(var, pred)] = weight

    def traverse_graph_update_weights(self, src, weight, visited=None):
        if visited is None:
            visited = {src}
        elif src in visited:
            return
        else:
            visited.add(src)
        self.weights[src] += weight
        for (s, t), w in self.edges.items():
            if s == src:
                self.traverse_graph_update_weights(t, weight * w, visited)
