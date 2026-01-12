"""
Tests for basic graph functionality: Node, Edge, Graph
"""

import unittest
import sys
import os

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.formulas import Not, And, Or


class TestNode(unittest.TestCase):
    """Tests for Node class"""
    
    def test_node_creation(self):
        """Test basic node creation"""
        node = Node('test')
        self.assertEqual(node.id, 'test')
        self.assertEqual(node.label, 'test')
        self.assertIsNone(node.value)
        self.assertIsNone(node.formula)
        self.assertIsNone(node.valid_path_parents)
    
    def test_node_with_label(self):
        """Test node with custom label"""
        node = Node('id', label='Custom Label')
        self.assertEqual(node.id, 'id')
        self.assertEqual(node.label, 'Custom Label')
    
    def test_node_with_formula(self):
        """Test node with formula and auto-inferred valid_path_parents"""
        node = Node('test', formula=Not('a'))
        self.assertIsNotNone(node.formula)
        self.assertEqual(node.valid_path_parents, [['a']])
    
    def test_node_with_explicit_valid_path_parents(self):
        """Test node with explicit valid_path_parents"""
        node = Node('test', formula=Not('a'), valid_path_parents=[['b']])
        self.assertEqual(node.valid_path_parents, [['b']])  # Explicit overrides auto-inference
    
    def test_node_set_value(self):
        """Test setting node value"""
        node = Node('test')
        node.set_value(True)
        self.assertEqual(node.value, True)
        node.set_value(False)
        self.assertEqual(node.value, False)
    
    def test_node_compute_value(self):
        """Test computing node value from formula"""
        node = Node('test', formula=And('a', 'b'))
        incoming = {'a': True, 'b': True}
        result = node.compute_value(incoming)
        self.assertEqual(result, True)
        
        incoming = {'a': True, 'b': False}
        result = node.compute_value(incoming)
        self.assertEqual(result, False)
    
    def test_node_copy(self):
        """Test copying a node"""
        node = Node('test', label='Test', value=True, formula=Not('a'))
        copied = node.copy()
        
        self.assertEqual(copied.id, node.id)
        self.assertEqual(copied.label, node.label)
        self.assertEqual(copied.value, node.value)
        self.assertEqual(copied.formula, node.formula)
        self.assertEqual(copied.valid_path_parents, node.valid_path_parents)
        
        # Modify copy and ensure original unchanged
        copied.set_value(False)
        self.assertEqual(node.value, True)


class TestEdge(unittest.TestCase):
    """Tests for Edge class"""
    
    def test_edge_creation(self):
        """Test basic edge creation"""
        edge = Edge('source', 'target')
        self.assertEqual(edge.source, 'source')
        self.assertEqual(edge.target, 'target')
    
    def test_edge_iteration(self):
        """Test edge iteration"""
        edge = Edge('a', 'b')
        source, target = list(edge)
        self.assertEqual(source, 'a')
        self.assertEqual(target, 'b')
    
    def test_edge_equality(self):
        """Test edge equality"""
        edge1 = Edge('a', 'b')
        edge2 = Edge('a', 'b')
        edge3 = Edge('a', 'c')
        
        self.assertEqual(edge1, edge2)
        self.assertNotEqual(edge1, edge3)


class TestGraph(unittest.TestCase):
    """Tests for Graph class"""
    
    def setUp(self):
        """Set up test graphs"""
        self.nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        self.edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        self.graph = Graph(self.nodes, self.edges)
    
    def test_graph_creation(self):
        """Test graph creation"""
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertEqual(len(self.graph.edges), 2)
    
    def test_add_node(self):
        """Test adding a node"""
        new_node = Node('d')
        self.graph.add_node(new_node)
        self.assertEqual(len(self.graph.nodes), 4)
        self.assertIn(new_node, self.graph.nodes)
    
    def test_add_edge(self):
        """Test adding an edge"""
        new_edge = Edge('c', 'd')
        self.graph.add_edge(new_edge)
        self.assertEqual(len(self.graph.edges), 3)
        self.assertIn(new_edge, self.graph.edges)
    
    def test_get_node_by_id(self):
        """Test getting node by ID"""
        node = self.graph.get_node_by_id('a')
        self.assertIsNotNone(node)
        self.assertEqual(node.id, 'a')
        
        node = self.graph.get_node_by_id('nonexistent')
        self.assertIsNone(node)
    
    def test_get_incoming_edges(self):
        """Test getting incoming edges"""
        node_c = self.graph.get_node_by_id('c')
        incoming = self.graph.get_incoming_edges(node_c)
        self.assertEqual(len(incoming), 2)
        self.assertEqual({e.source for e in incoming}, {'a', 'b'})
    
    def test_get_outgoing_edges(self):
        """Test getting outgoing edges"""
        node_a = self.graph.get_node_by_id('a')
        outgoing = self.graph.get_outgoing_edges(node_a)
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(outgoing[0].target, 'c')
    
    def test_get_incoming_nodes(self):
        """Test getting incoming nodes"""
        node_c = self.graph.get_node_by_id('c')
        incoming = self.graph.get_incoming_nodes(node_c)
        self.assertEqual(len(incoming), 2)
        self.assertEqual({n.id for n in incoming}, {'a', 'b'})
    
    def test_is_leaf_node(self):
        """Test checking if node is a leaf"""
        node_a = self.graph.get_node_by_id('a')
        node_c = self.graph.get_node_by_id('c')
        
        self.assertTrue(self.graph.is_leaf_node(node_a))
        self.assertFalse(self.graph.is_leaf_node(node_c))
    
    def test_get_leaf_nodes(self):
        """Test getting all leaf nodes"""
        leaves = self.graph.get_leaf_nodes()
        self.assertEqual(len(leaves), 2)
        self.assertEqual({n.id for n in leaves}, {'a', 'b'})
    
    def test_topological_sort(self):
        """Test topological sorting"""
        sorted_nodes = self.graph.topological_sort()
        # Leaf nodes should come before nodes that depend on them
        node_ids = [n.id for n in sorted_nodes]
        self.assertLess(node_ids.index('a'), node_ids.index('c'))
        self.assertLess(node_ids.index('b'), node_ids.index('c'))
    
    def test_leaf_values_set(self):
        """Test checking if all leaf values are set"""
        self.assertFalse(self.graph.leaf_values_set())
        
        self.graph.get_node_by_id('a').set_value(True)
        self.graph.get_node_by_id('b').set_value(True)
        self.assertTrue(self.graph.leaf_values_set())
    
    def test_non_leaf_formula_set(self):
        """Test checking if all non-leaf formulas are set"""
        # 'c' has a formula, so should return True
        self.assertTrue(self.graph.non_leaf_formula_set())
        
        # Remove formula from 'c'
        self.graph.get_node_by_id('c').formula = None
        self.assertFalse(self.graph.non_leaf_formula_set())
    
    def test_auto_infer_values(self):
        """Test auto-inferring values from formulas"""
        self.graph.get_node_by_id('a').set_value(True)
        self.graph.get_node_by_id('b').set_value(True)
        
        self.graph.auto_infer_values()
        
        node_c = self.graph.get_node_by_id('c')
        self.assertEqual(node_c.value, True)
    
    def test_auto_infer_values_fails_without_leaves(self):
        """Test that auto_infer_values fails if leaves not set"""
        with self.assertRaises(ValueError):
            self.graph.auto_infer_values()
    
    def test_graph_copy(self):
        """Test copying a graph"""
        self.graph.get_node_by_id('a').set_value(True)
        copied = self.graph.copy()
        
        self.assertEqual(len(copied.nodes), len(self.graph.nodes))
        self.assertEqual(len(copied.edges), len(self.graph.edges))
        
        # Modify copy and ensure original unchanged
        copied.get_node_by_id('a').set_value(False)
        self.assertEqual(self.graph.get_node_by_id('a').value, True)
    
    def test_complex_graph(self):
        """Test a more complex graph with multiple levels"""
        nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b')),
            Node('d', formula=Or('a', 'b')),
            Node('e', formula=And('c', 'd'))
        ]
        edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        graph = Graph(nodes, edges)
        
        # Set leaf values
        graph.get_node_by_id('a').set_value(True)
        graph.get_node_by_id('b').set_value(True)
        
        # Auto-infer
        graph.auto_infer_values()
        
        # Check computed values
        self.assertEqual(graph.get_node_by_id('c').value, True)
        self.assertEqual(graph.get_node_by_id('d').value, True)
        self.assertEqual(graph.get_node_by_id('e').value, True)


if __name__ == '__main__':
    unittest.main()

