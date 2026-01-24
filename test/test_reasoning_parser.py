"""
Test for reasoning_parser.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import ReasoningParser
from graph import Graph


def test_parser():
    """Test the reasoning parser with the example from the user."""
    
    reasoning_text = """<thinking>

The estimated number of characters for this story is 3000. Since 3000 is lower than 5000, I conclude that the story is [SHORT]. (SHORT : True)

Since the story starts with an adjective ("Wet raindrops fall...") it is not [NOUN]. (NOUN : False)

The story is also [NON-NOUN] because it is not [NOUN]. (NON-NOUN : True)

Therefore, the story is [DENSE] because both [SHORT] and [NON-NOUN] are true. (DENSE : True)

</thinking>
Yes, the story is dense."""

    parser = ReasoningParser()
    graph = parser.parse(reasoning_text)
    
    print("Parsed Graph:")
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    print("\nNodes:")
    for node in graph.nodes:
        print(f"  {node.id}: value={node.value}")
    print("\nEdges:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}")
    
    # Verify expected structure
    node_ids = {node.id for node in graph.nodes}
    expected_nodes = {"SHORT", "NOUN", "NON-NOUN", "DENSE"}
    assert node_ids == expected_nodes, f"Expected nodes {expected_nodes}, got {node_ids}"
    
    # Verify node values
    node_values = {node.id: node.value for node in graph.nodes}
    assert node_values["SHORT"] == True
    assert node_values["NOUN"] == False
    assert node_values["NON-NOUN"] == True
    assert node_values["DENSE"] == True
    
    # Verify edges
    edge_set = {(edge.source, edge.target) for edge in graph.edges}
    # SHORT -> DENSE, NOUN -> NON-NOUN, NON-NOUN -> DENSE
    expected_edges = {("SHORT", "DENSE"), ("NOUN", "NON-NOUN"), ("NON-NOUN", "DENSE")}
    assert edge_set == expected_edges, f"Expected edges {expected_edges}, got {edge_set}"
    
    # Verify no self-connections
    for edge in graph.edges:
        assert edge.source != edge.target, f"Self-connection found: {edge.source} -> {edge.target}"
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_parser()

