# graph_builder.py
import json
import networkx as nx
import pickle

class GraphBuilder:
    def __init__(self, triplet_file="graph_triplets.json"):
        self.triplet_file = triplet_file
        self.graph = nx.DiGraph()

    def build_graph(self):
        print(f"📥 Loading triplets from {self.triplet_file}")
        with open(self.triplet_file, "r", encoding="utf-8") as f:
            triplets = json.load(f)

        for triplet in triplets:
            subj = triplet["subject"]
            obj = triplet["object"]
            rel = triplet["relation"]
            sentence = triplet["sentence"]

            # Add nodes with metadata
            self.graph.add_node(subj)
            self.graph.add_node(obj)

            # Add edge with attributes
            self.graph.add_edge(subj, obj, relation=rel, sentence=sentence)

        print(f"✅ Graph created with {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

    def save_graph(self, path="graph_data.gpickle"):
        with open(path, "wb") as f:
            pickle.dump(self.graph, f)
        print(f"💾 Graph saved to {path}")


if __name__ == "__main__":
    builder = GraphBuilder()
    builder.build_graph()
    builder.save_graph()
