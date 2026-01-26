import pickle
from graph import Graph

def save_graph(graph: Graph, filepath: str) -> None:
    with open(filepath, 'wb') as f:
        pickle.dump(graph, f)

def load_graph(filepath: str) -> Graph:
    with open(filepath, 'rb') as f:
        return pickle.load(f)