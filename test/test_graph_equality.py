"""
Tests for Graph equality with check_ids parameter
"""

import unittest
import sys
import os

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.formulas import Not, And, Or, Xor, Equal, In


class TestGraphEqualityWithIds(unittest.TestCase):
    """Tests for Graph equality with check_ids=True (default)"""
    
    def test_identical_graphs_equal(self):
        """Test that identical graphs are equal with check_ids=True"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges2 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=True))
    
    def test_different_node_ids_not_equal(self):
        """Test that graphs with different node IDs are not equal with check_ids=True"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y'))
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1 == graph2)
        self.assertFalse(graph1.__eq__(graph2, check_ids=True))
    
    def test_different_formulas_not_equal(self):
        """Test that graphs with different formulas are not equal"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b'))  # Different formula
        ]
        edges2 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1 == graph2)
    
    def test_different_edges_not_equal(self):
        """Test that graphs with different edges are not equal"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges2 = [
            Edge('a', 'c')
            # Missing edge b -> c
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1 == graph2)
    
    def test_complex_formulas_with_ids(self):
        """Test equality with complex formulas (Not, Xor, Equal, In)"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=Not('a')),
            Node('d', formula=Xor('a', 'b')),
            Node('e', formula=Equal('a', True)),
            Node('f', formula=In('a', ['x', 'y']))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('a', 'e'),
            Edge('a', 'f')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('a'),
            Node('b'),
            Node('c', formula=Not('a')),
            Node('d', formula=Xor('a', 'b')),
            Node('e', formula=Equal('a', True)),
            Node('f', formula=In('a', ['x', 'y']))
        ]
        edges2 = [
            Edge('a', 'c'),
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('a', 'e'),
            Edge('a', 'f')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1 == graph2)


class TestGraphEqualityWithoutIds(unittest.TestCase):
    """Tests for Graph equality with check_ids=False (structural comparison)"""
    
    def test_same_structure_different_ids_equal(self):
        """Test that graphs with same structure but different IDs are equal with check_ids=False"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y'))
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_obfuscated_vs_non_obfuscated_equal(self):
        """Test that obfuscated and non-obfuscated graphs are equal structurally"""
        # Non-obfuscated graph
        nodes1 = [
            Node('noun'),
            Node('verb'),
            Node('well-selling', formula=And('noun', 'verb'))
        ]
        edges1 = [
            Edge('noun', 'well-selling'),
            Edge('verb', 'well-selling')
        ]
        graph1 = Graph(nodes1, edges1)
        
        # Obfuscated graph (same structure, different IDs)
        nodes2 = [
            Node('attr-1'),
            Node('attr-2'),
            Node('attr-3', formula=And('attr-1', 'attr-2'))
        ]
        edges2 = [
            Edge('attr-1', 'attr-3'),
            Edge('attr-2', 'attr-3')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
        self.assertFalse(graph1 == graph2)  # Should not be equal with check_ids=True
    
    def test_different_structure_not_equal(self):
        """Test that graphs with different structure are not equal even with check_ids=False"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y'))
        ]
        edges2 = [
            Edge('x', 'z')
            # Missing edge y -> z
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1.__eq__(graph2, check_ids=False))
    
    def test_different_formulas_not_equal(self):
        """Test that graphs with different formulas are not equal even with check_ids=False"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=Or('x', 'y'))  # Different formula
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1.__eq__(graph2, check_ids=False))
    
    def test_complex_formulas_without_ids(self):
        """Test structural equality with complex formulas"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=Not('a')),
            Node('d', formula=Xor('a', 'b')),
            Node('e', formula=Equal('a', True)),
            Node('f', formula=In('a', ['x', 'y']))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('a', 'e'),
            Edge('a', 'f')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=Not('x')),
            Node('w', formula=Xor('x', 'y')),
            Node('v', formula=Equal('x', True)),
            Node('u', formula=In('x', ['x', 'y']))
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('x', 'w'),
            Edge('y', 'w'),
            Edge('x', 'v'),
            Edge('x', 'u')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_nested_formulas_without_ids(self):
        """Test structural equality with nested formulas"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b')),
            Node('d', formula=Or(And('a', 'b'), 'c'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('c', 'd')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y')),
            Node('w', formula=Or(And('x', 'y'), 'z'))
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z'),
            Edge('x', 'w'),
            Edge('y', 'w'),
            Edge('z', 'w')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_different_number_of_nodes_not_equal(self):
        """Test that graphs with different number of nodes are not equal"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y')
        ]
        edges2 = []
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1.__eq__(graph2, check_ids=False))
    
    def test_different_topological_order_same_structure(self):
        """Test that graphs with same structure but potentially different topological order are equal"""
        # Graph 1: a -> c <- b
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        graph1 = Graph(nodes1, edges1)
        
        # Graph 2: x -> z <- y (same structure, different IDs)
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y'))
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))


class TestGraphEqualityEdgeCases(unittest.TestCase):
    """Edge cases for graph equality"""
    
    def test_empty_graphs_equal(self):
        """Test that empty graphs are equal"""
        graph1 = Graph([], [])
        graph2 = Graph([], [])
        
        self.assertTrue(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_single_node_equal(self):
        """Test single node graphs"""
        graph1 = Graph([Node('a')], [])
        graph2 = Graph([Node('a')], [])
        
        self.assertTrue(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_single_node_different_ids(self):
        """Test single node graphs with different IDs"""
        graph1 = Graph([Node('a')], [])
        graph2 = Graph([Node('x')], [])
        
        self.assertFalse(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))  # Structurally same
    
    def test_graph_with_no_formulas(self):
        """Test graphs with no formulas"""
        nodes1 = [Node('a'), Node('b'), Node('c')]
        edges1 = [Edge('a', 'b'), Edge('b', 'c')]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [Node('x'), Node('y'), Node('z')]
        edges2 = [Edge('x', 'y'), Edge('y', 'z')]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))
    
    def test_graph_with_mixed_formulas(self):
        """Test graphs with some nodes having formulas and some not"""
        nodes1 = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b')),
            Node('d')
        ]
        edges1 = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'd')
        ]
        graph1 = Graph(nodes1, edges1)
        
        nodes2 = [
            Node('x'),
            Node('y'),
            Node('z', formula=And('x', 'y')),
            Node('w')
        ]
        edges2 = [
            Edge('x', 'z'),
            Edge('y', 'z'),
            Edge('z', 'w')
        ]
        graph2 = Graph(nodes2, edges2)
        
        self.assertFalse(graph1 == graph2)
        self.assertTrue(graph1.__eq__(graph2, check_ids=False))


if __name__ == '__main__':
    unittest.main()

