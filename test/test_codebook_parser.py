"""
Test for codebook parser using cb-1.txt codebook.

Compares the parsed graph with the expected graph structure from test_graph_original.py
"""

import unittest
import sys
import os
import tempfile

# Add parent directory to path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.formulas import Not, And, Or
from parser import CodebookParser


class TestCodebookParser(unittest.TestCase):
    """Test codebook parser with cb-1.txt"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Get the expected graph structure from test_graph_original.py
        LEAF_NODES = ["short", "noun", "magical", "serious"]
        
        self.expected_nodes = [
            Node("short"),
            Node("noun"),
            Node("magical"),
            Node("serious"),
            Node("non-noun", formula=Not("noun")),
            Node("dense", formula=And("short", "non-noun")),
            Node("thrilling", formula=Or("magical", "serious")),
            Node("engaging", formula=And("dense", "thrilling")),
        ]
        
        self.expected_edges = [
            Edge("short", "dense"),
            Edge("noun", "non-noun"),
            Edge("non-noun", "dense"),
            Edge("dense", "engaging"),
            Edge("magical", "thrilling"),
            Edge("serious", "thrilling"),
            Edge("thrilling", "engaging")
        ]
        
        self.expected_graph = Graph(self.expected_nodes, self.expected_edges)
        
        # Path to codebook
        self.codebook_path = os.path.join(
            os.path.dirname(__file__), '..', 'codebooks', 'cb-1.txt'
        )
        
        # Create temp directory for output files
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_parse_codebook_structure(self):
        """Test that parsing codebook produces correct graph structure"""
        # Check if API key is available
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not set in environment")
        
        # Parse codebook
        parser = CodebookParser()
        output_path = os.path.join(self.temp_dir, "cb-1.pkl")
        parsed_graph = parser.parse_codebook(self.codebook_path, output_path)
        
        # Check that all expected nodes exist
        expected_node_ids = {node.id for node in self.expected_nodes}
        parsed_node_ids = {node.id for node in parsed_graph.nodes}
        
        for expected_id in expected_node_ids:
            self.assertIn(
                expected_id, parsed_node_ids,
                f"Expected node '{expected_id}' not found in parsed graph"
            )
        
        # Check node formulas match
        for expected_node in self.expected_nodes:
            parsed_node = parsed_graph.get_node_by_id(expected_node.id)
            self.assertIsNotNone(
                parsed_node,
                f"Node '{expected_node.id}' not found in parsed graph"
            )
            
            # Check formula type matches
            if expected_node.formula is None:
                self.assertIsNone(
                    parsed_node.formula,
                    f"Node '{expected_node.id}' should have no formula"
                )
            else:
                self.assertIsNotNone(
                    parsed_node.formula,
                    f"Node '{expected_node.id}' should have a formula"
                )
                self.assertEqual(
                    type(parsed_node.formula),
                    type(expected_node.formula),
                    f"Node '{expected_node.id}' formula type mismatch"
                )
        
        # Check that all expected edges exist
        expected_edge_set = {(e.source, e.target) for e in self.expected_edges}
        parsed_edge_set = {(e.source, e.target) for e in parsed_graph.edges}
        
        for expected_edge in expected_edge_set:
            self.assertIn(
                expected_edge, parsed_edge_set,
                f"Expected edge '{expected_edge[0]} -> {expected_edge[1]}' not found"
            )
    
    def test_parse_codebook_formulas(self):
        """Test that formulas are correctly parsed and work"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not set in environment")
        
        parser = CodebookParser()
        output_path = os.path.join(self.temp_dir, "cb-1.pkl")
        parsed_graph = parser.parse_codebook(self.codebook_path, output_path)
        
        # Test specific formulas
        # Check non-noun formula
        non_noun_node = parsed_graph.get_node_by_id("non-noun")
        self.assertIsNotNone(non_noun_node.formula)
        self.assertIsInstance(non_noun_node.formula, Not)
        
        # Check dense formula
        dense_node = parsed_graph.get_node_by_id("dense")
        self.assertIsNotNone(dense_node.formula)
        self.assertIsInstance(dense_node.formula, And)
        
        # Check thrilling formula
        thrilling_node = parsed_graph.get_node_by_id("thrilling")
        self.assertIsNotNone(thrilling_node.formula)
        self.assertIsInstance(thrilling_node.formula, Or)
        
        # Check engaging formula
        engaging_node = parsed_graph.get_node_by_id("engaging")
        self.assertIsNotNone(engaging_node.formula)
        self.assertIsInstance(engaging_node.formula, And)
    
    def test_parse_codebook_value_computation(self):
        """Test that parsed graph computes values correctly"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not set in environment")
        
        parser = CodebookParser()
        output_path = os.path.join(self.temp_dir, "cb-1.pkl")
        parsed_graph = parser.parse_codebook(self.codebook_path, output_path)
        
        # Get all leaf nodes from the parsed graph and set values for the ones we care about
        leaf_nodes = parsed_graph.get_leaf_nodes()
        leaf_node_ids = {node.id for node in leaf_nodes}
        
        # Test case from test_graph_original.py: (True, False, True, False)
        # short=True, noun=False, magical=True, serious=False
        # Expected: non-noun=True, dense=True, thrilling=True, engaging=True
        
        # Set values for our test nodes (only if they exist as leaf nodes)
        if "short" in leaf_node_ids:
            parsed_graph.get_node_by_id("short").set_value(True)
        if "noun" in leaf_node_ids:
            parsed_graph.get_node_by_id("noun").set_value(False)
        if "magical" in leaf_node_ids:
            parsed_graph.get_node_by_id("magical").set_value(True)
        if "serious" in leaf_node_ids:
            parsed_graph.get_node_by_id("serious").set_value(False)
        
        # Set default values for any other leaf nodes (like random-seed) to avoid errors
        for leaf_node in leaf_nodes:
            if leaf_node.value is None and leaf_node.id not in ["short", "noun", "magical", "serious"]:
                leaf_node.set_value(False)  # Default to False for other leaf nodes
        
        parsed_graph.auto_infer_values()
        
        # Verify computed values
        self.assertEqual(parsed_graph.get_node_by_id("non-noun").value, True)
        self.assertEqual(parsed_graph.get_node_by_id("dense").value, True)
        self.assertEqual(parsed_graph.get_node_by_id("thrilling").value, True)
        self.assertEqual(parsed_graph.get_node_by_id("engaging").value, True)
        
        # Test another case: (True, True, True, True)
        # short=True, noun=True, magical=True, serious=True
        # Expected: non-noun=False, dense=False, thrilling=True, engaging=False
        
        # Reset all leaf nodes
        for leaf_node in parsed_graph.get_leaf_nodes():
            leaf_node.set_value(None)
        
        if "short" in leaf_node_ids:
            parsed_graph.get_node_by_id("short").set_value(True)
        if "noun" in leaf_node_ids:
            parsed_graph.get_node_by_id("noun").set_value(True)
        if "magical" in leaf_node_ids:
            parsed_graph.get_node_by_id("magical").set_value(True)
        if "serious" in leaf_node_ids:
            parsed_graph.get_node_by_id("serious").set_value(True)
        
        # Set default values for any other leaf nodes
        for leaf_node in parsed_graph.get_leaf_nodes():
            if leaf_node.value is None and leaf_node.id not in ["short", "noun", "magical", "serious"]:
                leaf_node.set_value(False)
        
        parsed_graph.auto_infer_values()
        
        self.assertEqual(parsed_graph.get_node_by_id("non-noun").value, False)
        self.assertEqual(parsed_graph.get_node_by_id("dense").value, False)
        self.assertEqual(parsed_graph.get_node_by_id("thrilling").value, True)
        self.assertEqual(parsed_graph.get_node_by_id("engaging").value, False)
    
    def test_parse_codebook_output_files(self):
        """Test that both pickle and JSON files are created"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            self.skipTest("OPENAI_API_KEY not set in environment")
        
        parser = CodebookParser()
        output_path = os.path.join(self.temp_dir, "cb-1.pkl")
        parsed_graph = parser.parse_codebook(self.codebook_path, output_path)
        
        # Check pickle file exists
        self.assertTrue(os.path.exists(output_path), "Pickle file not created")
        
        # Check JSON file exists
        json_path = os.path.join(self.temp_dir, "cb-1.json")
        self.assertTrue(os.path.exists(json_path), "JSON file not created")
        
        # Verify JSON file is valid
        import json
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            self.assertIn("nodes", json_data)
            self.assertIn("edges", json_data)
            self.assertIsInstance(json_data["nodes"], list)
            self.assertIsInstance(json_data["edges"], list)


if __name__ == '__main__':
    unittest.main()

