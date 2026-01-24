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


def test_parser_with_categorical_values():
    """Test the reasoning parser with categorical (string) values."""
    
    reasoning_text = """<thinking>

The story genre is determined by [THEME] and [SETTING]. (GENRE : "science-fiction")

The theme is [FUTURISTIC] and the setting is [SPACE]. (THEME : "futuristic")

The setting location is [OUTER_SPACE]. (SETTING : "space")

The story takes place in [OUTER_SPACE]. (FUTURISTIC : True)

The story location is [OUTER_SPACE]. (SPACE : True)

</thinking>"""

    parser = ReasoningParser()
    graph = parser.parse(reasoning_text)
    
    print("\n" + "="*50)
    print("Testing with categorical values:")
    print("="*50)
    print(f"Number of nodes: {len(graph.nodes)}")
    print(f"Number of edges: {len(graph.edges)}")
    print("\nNodes:")
    for node in graph.nodes:
        print(f"  {node.id}: value={node.value} (type: {type(node.value).__name__})")
    print("\nEdges:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}")
    
    # Verify expected structure
    node_ids = {node.id for node in graph.nodes}
    expected_nodes = {"GENRE", "THEME", "SETTING", "FUTURISTIC", "SPACE", "OUTER_SPACE"}
    assert node_ids == expected_nodes, f"Expected nodes {expected_nodes}, got {node_ids}"
    
    # Verify node values (mix of boolean and categorical)
    node_values = {node.id: node.value for node in graph.nodes}
    assert node_values["GENRE"] == "science-fiction", f"Expected 'science-fiction', got {node_values['GENRE']}"
    assert node_values["THEME"] == "futuristic", f"Expected 'futuristic', got {node_values['THEME']}"
    assert node_values["SETTING"] == "space", f"Expected 'space', got {node_values['SETTING']}"
    assert node_values["FUTURISTIC"] == True
    assert node_values["SPACE"] == True
    assert node_values["OUTER_SPACE"] is None  # Source node without value
    
    # Verify edges
    edge_set = {(edge.source, edge.target) for edge in graph.edges}
    expected_edges = {
        ("THEME", "GENRE"),
        ("SETTING", "GENRE"),
        ("FUTURISTIC", "THEME"),
        ("SPACE", "THEME"),
        ("OUTER_SPACE", "SETTING"),
        ("OUTER_SPACE", "FUTURISTIC"),
        ("OUTER_SPACE", "SPACE")
    }
    assert edge_set == expected_edges, f"Expected edges {expected_edges}, got {edge_set}"
    
    # Verify no self-connections
    for edge in graph.edges:
        assert edge.source != edge.target, f"Self-connection found: {edge.source} -> {edge.target}"
    
    print("\n✓ All categorical value tests passed!")


def test_parser_mixed_values():
    """Test the parser with a mix of boolean and categorical values."""
    
    reasoning_text = """<thinking>

The story length is [SHORT]. (LENGTH : "short")

The story is [SHORT]. (SHORT : True)

The story style is [DENSE]. (STYLE : "dense")

The story is [DENSE]. (DENSE : False)

</thinking>"""

    parser = ReasoningParser()
    graph = parser.parse(reasoning_text)
    
    print("\n" + "="*50)
    print("Testing with mixed boolean and categorical values:")
    print("="*50)
    print(f"Number of nodes: {len(graph.nodes)}")
    print("\nNodes:")
    for node in graph.nodes:
        print(f"  {node.id}: value={node.value} (type: {type(node.value).__name__})")
    
    # Verify mixed value types
    node_values = {node.id: node.value for node in graph.nodes}
    assert isinstance(node_values["LENGTH"], str), "LENGTH should be a string"
    assert node_values["LENGTH"] == "short"
    assert isinstance(node_values["SHORT"], bool), "SHORT should be a boolean"
    assert node_values["SHORT"] == True
    assert isinstance(node_values["STYLE"], str), "STYLE should be a string"
    assert node_values["STYLE"] == "dense"
    assert isinstance(node_values["DENSE"], bool), "DENSE should be a boolean"
    assert node_values["DENSE"] == False
    
    print("\n✓ All mixed value tests passed!")


if __name__ == "__main__":
    test_parser()
    test_parser_with_categorical_values()
    test_parser_mixed_values()

