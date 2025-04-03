import pickle
import networkx as nx

class GraphRetriever:
    def __init__(self, graph_path="graph_data.gpickle"):
        with open(graph_path, "rb") as f:
            self.graph = pickle.load(f)

    def query(self, entity, depth=1):
        """Returns neighbors (1-hop or 2-hop) and the edge relations."""
        if entity not in self.graph:
            print(f"❌ Entity '{entity}' not in graph.")
            return []

        visited = set()
        results = []

        def dfs(node, d):
            if d > depth or node in visited:
                return
            visited.add(node)
            for neighbor in self.graph.neighbors(node):
                edge_data = self.graph.get_edge_data(node, neighbor)
                results.append({
                    "source": node,
                    "target": neighbor,
                    "relation": edge_data.get("relation", "related_to"),
                    "sentence": edge_data.get("sentence", "")
                })
                dfs(neighbor, d + 1)

        dfs(entity, 0)
        return results
