"""
Tests for graph serialization and deserialization
"""

import unittest
import sys
import os
import tempfile

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.formulas import Not, And, Or, Xor, Equal, In
from serializer import save_graph, load_graph


class TestGraphSerializer(unittest.TestCase):
    """Tests for graph serialization and deserialization"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_graph.json")
    
    def tearDown(self):
        """Clean up test files"""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
        os.rmdir(self.temp_dir)
    
    def test_save_load_simple_graph(self):
        """Test saving and loading a simple graph with no formulas"""
        nodes = [
            Node("a"),
            Node("b"),
            Node("c")
        ]
        edges = [
            Edge("a", "b"),
            Edge("b", "c")
        ]
        original_graph = Graph(nodes, edges)
        
        # Set some values
        original_graph.get_node_by_id("a").set_value(True)
        original_graph.get_node_by_id("b").set_value(False)
        original_graph.get_node_by_id("c").set_value(True)
        
        # Save and load
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        # Verify structure
        self.assertEqual(len(loaded_graph.nodes), len(original_graph.nodes))
        self.assertEqual(len(loaded_graph.edges), len(original_graph.edges))
        
        # Verify node attributes
        for original_node in original_graph.nodes:
            loaded_node = loaded_graph.get_node_by_id(original_node.id)
            self.assertIsNotNone(loaded_node)
            self.assertEqual(loaded_node.id, original_node.id)
            self.assertEqual(loaded_node.label, original_node.label)
            self.assertEqual(loaded_node.value, original_node.value)
            self.assertEqual(loaded_node.valid_path_parents, original_node.valid_path_parents)
            self.assertEqual(loaded_node.formula, original_node.formula)
        
        # Verify edges
        original_edge_sources = {e.source for e in original_graph.edges}
        original_edge_targets = {e.target for e in original_graph.edges}
        loaded_edge_sources = {e.source for e in loaded_graph.edges}
        loaded_edge_targets = {e.target for e in loaded_graph.edges}
        
        self.assertEqual(original_edge_sources, loaded_edge_sources)
        self.assertEqual(original_edge_targets, loaded_edge_targets)
    
    def test_save_load_graph_with_formulas(self):
        """Test saving and loading a graph with formula objects"""
        nodes = [
            Node("short"),
            Node("noun"),
            Node("magical"),
            Node("serious"),
            Node("non-noun", formula=Not("noun")),
            Node("dense", formula=And("short", "non-noun")),
            Node("thrilling", formula=Or("magical", "serious")),
            Node("engaging", formula=And("dense", "thrilling")),
        ]
        
        edges = [
            Edge("short", "dense"),
            Edge("noun", "non-noun"),
            Edge("non-noun", "dense"),
            Edge("dense", "engaging"),
            Edge("magical", "thrilling"),
            Edge("serious", "thrilling"),
            Edge("thrilling", "engaging")
        ]
        
        original_graph = Graph(nodes, edges)
        
        # Set leaf values and auto-infer
        original_graph.get_node_by_id("short").set_value(True)
        original_graph.get_node_by_id("noun").set_value(False)
        original_graph.get_node_by_id("magical").set_value(True)
        original_graph.get_node_by_id("serious").set_value(False)
        original_graph.auto_infer_values()
        
        # Save and load
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        # Verify all nodes
        self.assertEqual(len(loaded_graph.nodes), len(original_graph.nodes))
        
        # Verify formulas are preserved
        for original_node in original_graph.nodes:
            loaded_node = loaded_graph.get_node_by_id(original_node.id)
            self.assertIsNotNone(loaded_node)
            
            # Check if formula exists (both should be None or both should exist)
            if original_node.formula is None:
                self.assertIsNone(loaded_node.formula)
            else:
                self.assertIsNotNone(loaded_node.formula)
                # Verify formula type
                self.assertEqual(type(loaded_node.formula), type(original_node.formula))
            
            # Verify values
            self.assertEqual(loaded_node.value, original_node.value)
            
            # Verify valid_path_parents
            self.assertEqual(loaded_node.valid_path_parents, original_node.valid_path_parents)
        
        # Verify formulas still work
        loaded_graph.get_node_by_id("short").set_value(True)
        loaded_graph.get_node_by_id("noun").set_value(False)
        loaded_graph.get_node_by_id("magical").set_value(True)
        loaded_graph.get_node_by_id("serious").set_value(False)
        loaded_graph.auto_infer_values()
        
        # Check that computed values match
        self.assertEqual(loaded_graph.get_node_by_id("non-noun").value, True)
        self.assertEqual(loaded_graph.get_node_by_id("dense").value, True)
        self.assertEqual(loaded_graph.get_node_by_id("thrilling").value, True)
        self.assertEqual(loaded_graph.get_node_by_id("engaging").value, True)
    
    def test_save_load_graph_with_nested_formulas(self):
        """Test saving and loading a graph with nested formula objects"""
        nodes = [
            Node("a"),
            Node("b"),
            Node("c"),
            Node("not_a", formula=Not("a")),
            Node("not_b", formula=Not("b")),
            Node("complex", formula=And(Not("a"), Or("b", "c"))),
        ]
        
        edges = [
            Edge("a", "not_a"),
            Edge("a", "complex"),
            Edge("b", "not_b"),
            Edge("b", "complex"),
            Edge("c", "complex"),
        ]
        
        original_graph = Graph(nodes, edges)
        
        # Set values
        original_graph.get_node_by_id("a").set_value(False)
        original_graph.get_node_by_id("b").set_value(True)
        original_graph.get_node_by_id("c").set_value(True)
        original_graph.auto_infer_values()
        
        # Save and load
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        # Verify nested formulas are preserved
        complex_node_original = original_graph.get_node_by_id("complex")
        complex_node_loaded = loaded_graph.get_node_by_id("complex")
        
        self.assertIsNotNone(complex_node_loaded.formula)
        self.assertEqual(type(complex_node_loaded.formula), type(complex_node_original.formula))
        self.assertEqual(complex_node_loaded.value, complex_node_original.value)
    
    def test_save_load_graph_with_all_formula_types(self):
        """Test saving and loading a graph with all formula types"""
        nodes = [
            Node("x", value=True),
            Node("y", value=False),
            Node("z", value=2),
            Node("w", value="test"),
            Node("not_x", formula=Not("x")),
            Node("and_xy", formula=And("x", "y")),
            Node("or_xy", formula=Or("x", "y")),
            Node("xor_xy", formula=Xor("x", "y")),
            Node("equal_z", formula=Equal("z", 2)),
            Node("in_w", formula=In("w", ["test", "example"])),
        ]
        
        edges = [
            Edge("x", "not_x"),
            Edge("x", "and_xy"),
            Edge("y", "and_xy"),
            Edge("x", "or_xy"),
            Edge("y", "or_xy"),
            Edge("x", "xor_xy"),
            Edge("y", "xor_xy"),
            Edge("z", "equal_z"),
            Edge("w", "in_w"),
        ]
        
        original_graph = Graph(nodes, edges)
        original_graph.auto_infer_values()
        
        # Save and load
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        # Verify all formula types are preserved
        formula_types = ["not_x", "and_xy", "or_xy", "xor_xy", "equal_z", "in_w"]
        for node_id in formula_types:
            original_node = original_graph.get_node_by_id(node_id)
            loaded_node = loaded_graph.get_node_by_id(node_id)
            
            self.assertIsNotNone(loaded_node.formula)
            self.assertEqual(type(loaded_node.formula), type(original_node.formula))
            self.assertEqual(loaded_node.value, original_node.value)
    
    def test_save_load_empty_graph(self):
        """Test saving and loading an empty graph"""
        original_graph = Graph([], [])
        
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        self.assertEqual(len(loaded_graph.nodes), 0)
        self.assertEqual(len(loaded_graph.edges), 0)
    
    def test_save_load_graph_with_none_values(self):
        """Test saving and loading a graph with None values"""
        nodes = [
            Node("a", value=None),
            Node("b", value=None),
        ]
        edges = [
            Edge("a", "b")
        ]
        original_graph = Graph(nodes, edges)
        
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        self.assertEqual(loaded_graph.get_node_by_id("a").value, None)
        self.assertEqual(loaded_graph.get_node_by_id("b").value, None)
    
    def test_save_load_graph_with_custom_labels(self):
        """Test saving and loading a graph with custom node labels"""
        nodes = [
            Node("id1", label="Custom Label 1"),
            Node("id2", label="Custom Label 2"),
        ]
        edges = [
            Edge("id1", "id2")
        ]
        original_graph = Graph(nodes, edges)
        
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        self.assertEqual(loaded_graph.get_node_by_id("id1").label, "Custom Label 1")
        self.assertEqual(loaded_graph.get_node_by_id("id2").label, "Custom Label 2")
    
    def test_save_load_graph_preserves_functionality(self):
        """Test that loaded graph maintains all functionality"""
        nodes = [
            Node("a"),
            Node("b"),
            Node("c", formula=And("a", "b")),
        ]
        edges = [
            Edge("a", "c"),
            Edge("b", "c"),
        ]
        original_graph = Graph(nodes, edges)
        
        # Set values and compute
        original_graph.get_node_by_id("a").set_value(True)
        original_graph.get_node_by_id("b").set_value(True)
        original_graph.auto_infer_values()
        
        save_graph(original_graph, self.test_file)
        loaded_graph = load_graph(self.test_file)
        
        # Test that loaded graph methods still work
        self.assertIsNotNone(loaded_graph.get_node_by_id("a"))
        self.assertIsNotNone(loaded_graph.get_node_by_id("c"))
        
        incoming_edges = loaded_graph.get_incoming_edges(loaded_graph.get_node_by_id("c"))
        self.assertEqual(len(incoming_edges), 2)
        
        outgoing_edges = loaded_graph.get_outgoing_edges(loaded_graph.get_node_by_id("a"))
        self.assertEqual(len(outgoing_edges), 1)
        
        # Test topological sort
        sorted_nodes = loaded_graph.topological_sort()
        self.assertEqual(len(sorted_nodes), 3)
        
        # Test that we can still compute values
        loaded_graph.get_node_by_id("a").set_value(False)
        loaded_graph.get_node_by_id("b").set_value(True)
        loaded_graph.auto_infer_values()
        self.assertEqual(loaded_graph.get_node_by_id("c").value, False)


if __name__ == '__main__':
    unittest.main()

