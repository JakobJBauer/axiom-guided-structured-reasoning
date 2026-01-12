"""
Comprehensive tests for GraphMetrics class
"""

import unittest
import sys
import os

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.graph_metrics import GraphMetrics
from graph.formulas import Not, And, Or


class TestGraphMetricsBasic(unittest.TestCase):
    """Basic tests for GraphMetrics"""
    
    def setUp(self):
        """Set up reference and predicted graphs"""
        # Reference graph
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        self.reference = Graph(ref_nodes, ref_edges)
        
        # Predicted graph (same as reference)
        pred_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        self.predicted = Graph(pred_nodes, pred_edges)
        
        self.metrics = GraphMetrics(self.reference, self.predicted)
    
    def test_get_end_node(self):
        """Test getting end node"""
        end_node = self.metrics._get_end_node(self.reference)
        self.assertEqual(end_node.id, 'c')
    
    def test_get_end_node_multiple_ends_fails(self):
        """Test that get_end_node fails with multiple end nodes"""
        nodes = [Node('a'), Node('b')]
        edges = []
        graph = Graph(nodes, edges)
        metrics = GraphMetrics(graph, graph)
        
        with self.assertRaises(ValueError):
            metrics._get_end_node(graph)


class TestGraphMetricsStructure(unittest.TestCase):
    """Tests for structure-based metrics"""
    
    def setUp(self):
        """Set up test graphs"""
        # Reference: a -> c <- b
        ref_nodes = [Node('a'), Node('b'), Node('c', formula=And('a', 'b'))]
        ref_edges = [Edge('a', 'c'), Edge('b', 'c')]
        self.reference = Graph(ref_nodes, ref_edges)
    
    def test_correct_reasoning_edges_perfect_match(self):
        """Test correct edges when graphs match perfectly"""
        pred_nodes = [Node('a'), Node('b'), Node('c', formula=And('a', 'b'))]
        pred_edges = [Edge('a', 'c'), Edge('b', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics = GraphMetrics(self.reference, predicted)
        self.assertEqual(metrics.correct_reasoning_edges(), 2)
        self.assertEqual(metrics.missing_reasoning_edges(), 0)
        self.assertEqual(metrics.hallucinated_reasoning_edges(), 0)
        self.assertTrue(metrics.full_graph_match())
    
    def test_correct_reasoning_edges_missing_edge(self):
        """Test when predicted graph is missing an edge"""
        pred_nodes = [Node('a'), Node('b'), Node('c', formula=And('a', 'b'))]
        pred_edges = [Edge('a', 'c')]  # Missing b -> c
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics = GraphMetrics(self.reference, predicted)
        self.assertEqual(metrics.correct_reasoning_edges(), 1)
        self.assertEqual(metrics.missing_reasoning_edges(), 1)
        self.assertEqual(metrics.hallucinated_reasoning_edges(), 0)
        self.assertFalse(metrics.full_graph_match())
    
    def test_correct_reasoning_edges_hallucinated_edge(self):
        """Test when predicted graph has extra edge"""
        pred_nodes = [Node('a'), Node('b'), Node('c', formula=And('a', 'b')), Node('d')]
        pred_edges = [Edge('a', 'c'), Edge('b', 'c'), Edge('a', 'd')]  # Extra edge
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics = GraphMetrics(self.reference, predicted)
        self.assertEqual(metrics.correct_reasoning_edges(), 2)
        self.assertEqual(metrics.missing_reasoning_edges(), 0)
        self.assertEqual(metrics.hallucinated_reasoning_edges(), 1)
        # full_graph_match should return False when there are hallucinations
        # This is the correct behavior - hallucinations indicate incorrect reasoning
        self.assertFalse(metrics.full_graph_match())
    
    def test_longest_correct_reasoning_path(self):
        """Test longest correct reasoning path"""
        # Reference: a -> b -> c
        ref_nodes = [Node('a'), Node('b', formula=Not('a')), Node('c', formula=And('a', 'b'))]
        ref_edges = [Edge('a', 'b'), Edge('b', 'c')]
        reference = Graph(ref_nodes, ref_edges)
        
        # Predicted: same structure
        pred_nodes = [Node('a'), Node('b', formula=Not('a')), Node('c', formula=And('a', 'b'))]
        pred_edges = [Edge('a', 'b'), Edge('b', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics = GraphMetrics(reference, predicted)
        self.assertEqual(metrics.longest_correct_reasoning_path(), 3)  # a -> b -> c = 3 nodes
    
    def test_longest_correct_reasoning_path_with_missing_node(self):
        """Test longest path when predicted graph is missing a node"""
        ref_nodes = [Node('a'), Node('b'), Node('c', formula=And('a', 'b'))]
        ref_edges = [Edge('a', 'b'), Edge('b', 'c')]
        reference = Graph(ref_nodes, ref_edges)
        
        # Predicted: missing node 'b'
        pred_nodes = [Node('a'), Node('c', formula=And('a', 'b'))]
        pred_edges = [Edge('a', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics = GraphMetrics(reference, predicted)
        # Longest path should be shorter due to missing node
        self.assertLess(metrics.longest_correct_reasoning_path(), 3)


class TestGraphMetricsValues(unittest.TestCase):
    """Tests for value-based metrics"""
    
    def setUp(self):
        """Set up test graphs with values"""
        # Reference graph
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        ref_edges = [Edge('a', 'c'), Edge('b', 'c')]
        self.reference = Graph(ref_nodes, ref_edges)
        
        # Set reference values
        self.reference.get_node_by_id('a').set_value(True)
        self.reference.get_node_by_id('b').set_value(True)
        self.reference.get_node_by_id('c').set_value(True)
    
    def test_correct_reasoning_edges_with_values_perfect_match(self):
        """Test correct edges with value checking when values match"""
        pred_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        pred_edges = [Edge('a', 'c'), Edge('b', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Set predicted values to match
        predicted.get_node_by_id('a').set_value(True)
        predicted.get_node_by_id('b').set_value(True)
        predicted.get_node_by_id('c').set_value(True)
        
        metrics = GraphMetrics(self.reference, predicted)
        self.assertEqual(metrics.correct_reasoning_edges(check_values=True), 2)
        self.assertEqual(metrics.missing_reasoning_edges(check_values=True), 0)
        self.assertTrue(metrics.full_graph_match(check_values=True))
    
    def test_correct_reasoning_edges_with_values_mismatch(self):
        """Test when edge structure matches but values don't"""
        pred_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        pred_edges = [Edge('a', 'c'), Edge('b', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Set predicted values - 'c' has wrong value
        predicted.get_node_by_id('a').set_value(True)
        predicted.get_node_by_id('b').set_value(True)
        predicted.get_node_by_id('c').set_value(False)  # Wrong!
        
        metrics = GraphMetrics(self.reference, predicted)
        # Without value checking, all edges are correct
        self.assertEqual(metrics.correct_reasoning_edges(check_values=False), 2)
        # With value checking, edges to 'c' are not correct
        self.assertEqual(metrics.correct_reasoning_edges(check_values=True), 0)
    
    def test_longest_correct_reasoning_path_with_values(self):
        """Test longest path with value checking"""
        # Reference: a -> b -> c (linear chain)
        ref_nodes = [Node('a'), Node('b'), Node('c')]
        ref_edges = [Edge('a', 'b'), Edge('b', 'c')]
        reference = Graph(ref_nodes, ref_edges)
        reference.get_node_by_id('a').set_value(True)
        reference.get_node_by_id('b').set_value(True)
        reference.get_node_by_id('c').set_value(True)
        
        # Predicted: same structure, but 'c' has wrong value
        pred_nodes = [Node('a'), Node('b'), Node('c')]
        pred_edges = [Edge('a', 'b'), Edge('b', 'c')]
        predicted = Graph(pred_nodes, pred_edges)
        predicted.get_node_by_id('a').set_value(True)
        predicted.get_node_by_id('b').set_value(True)
        predicted.get_node_by_id('c').set_value(False)  # Wrong value
        
        metrics = GraphMetrics(reference, predicted)
        # Without value checking, full path is correct (structure matches)
        self.assertEqual(metrics.longest_correct_reasoning_path(check_values=False), 3)
        # With value checking, 'c' (end node) has wrong value
        # The algorithm starts from end node 'c', and if it's wrong, it returns 0
        # This is the current behavior - if end node is wrong, no valid path
        result = metrics.longest_correct_reasoning_path(check_values=True)
        self.assertEqual(result, 0)  # End node has wrong value, so no valid path


class TestGraphMetricsBinaryMetrics(unittest.TestCase):
    """Tests for binary classification metrics"""
    
    def setUp(self):
        """Set up test graphs"""
        # Need at least one edge to have a single end node
        ref_nodes = [Node('a'), Node('b'), Node('c')]
        ref_edges = [Edge('a', 'c'), Edge('b', 'c')]  # c is the end node
        self.reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [Node('a'), Node('b'), Node('c')]
        pred_edges = [Edge('a', 'c'), Edge('b', 'c')]
        self.predicted = Graph(pred_nodes, pred_edges)
    
    def test_end_node_metrics_perfect_match(self):
        """Test end node metrics when values match perfectly"""
        self.reference.get_node_by_id('c').set_value(True)
        self.predicted.get_node_by_id('c').set_value(True)
        
        metrics = GraphMetrics(self.reference, self.predicted)
        acc, prec, rec, f1 = metrics.end_node_metrics()
        
        self.assertEqual(acc, 1.0)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)
    
    def test_end_node_metrics_false_positive(self):
        """Test end node metrics with false positive"""
        self.reference.get_node_by_id('c').set_value(False)
        self.predicted.get_node_by_id('c').set_value(True)  # False positive
        
        metrics = GraphMetrics(self.reference, self.predicted)
        acc, prec, rec, f1 = metrics.end_node_metrics()
        
        self.assertEqual(acc, 0.0)
        self.assertEqual(prec, 0.0)  # TP=0, FP=1
        self.assertEqual(rec, 0.0)
        self.assertEqual(f1, 0.0)
    
    def test_end_node_metrics_false_negative(self):
        """Test end node metrics with false negative"""
        self.reference.get_node_by_id('c').set_value(True)
        self.predicted.get_node_by_id('c').set_value(False)  # False negative
        
        metrics = GraphMetrics(self.reference, self.predicted)
        acc, prec, rec, f1 = metrics.end_node_metrics()
        
        self.assertEqual(acc, 0.0)
        self.assertEqual(prec, 0.0)  # TP=0, FP=0
        self.assertEqual(rec, 0.0)  # TP=0, FN=1
        self.assertEqual(f1, 0.0)
    
    def test_average_node_metrics(self):
        """Test average node metrics across all nodes"""
        # Set all reference values
        self.reference.get_node_by_id('a').set_value(True)
        self.reference.get_node_by_id('b').set_value(False)
        self.reference.get_node_by_id('c').set_value(True)
        
        # Set predicted values (2 correct, 1 wrong)
        self.predicted.get_node_by_id('a').set_value(True)  # Correct
        self.predicted.get_node_by_id('b').set_value(False)  # Correct
        self.predicted.get_node_by_id('c').set_value(False)  # Wrong
        
        metrics = GraphMetrics(self.reference, self.predicted)
        acc, prec, rec, f1 = metrics.average_node_metrics()
        
        # 2 out of 3 correct
        self.assertAlmostEqual(acc, 2.0/3.0, places=5)
    
    def test_metrics_with_none_values(self):
        """Test metrics when some values are None"""
        self.reference.get_node_by_id('c').set_value(True)
        self.predicted.get_node_by_id('c').set_value(None)  # None value
        
        metrics = GraphMetrics(self.reference, self.predicted)
        acc, prec, rec, f1 = metrics.end_node_metrics()
        
        # Should return (0, 0, 0, 0) when values are None
        self.assertEqual(acc, 0.0)
        self.assertEqual(prec, 0.0)
        self.assertEqual(rec, 0.0)
        self.assertEqual(f1, 0.0)


class TestGraphMetricsComplex(unittest.TestCase):
    """Tests for complex graph scenarios"""
    
    def test_codebook_example(self):
        """Test with the codebook example from test_graph.py"""
        from graph.formulas import Not, And, Or
        
        # Reference graph
        ref_nodes = [
            Node('short'),
            Node('noun'),
            Node('magical'),
            Node('serious'),
            Node('non-noun', formula=Not('noun')),
            Node('dense', formula=And('short', 'non-noun')),
            Node('thrilling', formula=Or('magical', 'serious')),
            Node('engaging', formula=And('dense', 'thrilling'))
        ]
        ref_edges = [
            Edge('short', 'dense'),
            Edge('noun', 'non-noun'),
            Edge('non-noun', 'dense'),
            Edge('dense', 'engaging'),
            Edge('magical', 'thrilling'),
            Edge('serious', 'thrilling'),
            Edge('thrilling', 'engaging')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        # Predicted graph (same structure)
        pred_nodes = [
            Node('short'),
            Node('noun'),
            Node('magical'),
            Node('serious'),
            Node('non-noun', formula=Not('noun')),
            Node('dense', formula=And('short', 'non-noun')),
            Node('thrilling', formula=Or('magical', 'serious')),
            Node('engaging', formula=And('dense', 'thrilling'))
        ]
        pred_edges = [
            Edge('short', 'dense'),
            Edge('noun', 'non-noun'),
            Edge('non-noun', 'dense'),
            Edge('dense', 'engaging'),
            Edge('magical', 'thrilling'),
            Edge('serious', 'thrilling'),
            Edge('thrilling', 'engaging')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Set values and compute
        predicted.get_node_by_id('short').set_value(True)
        predicted.get_node_by_id('noun').set_value(True)
        predicted.get_node_by_id('magical').set_value(True)
        predicted.get_node_by_id('serious').set_value(True)
        predicted.auto_infer_values()
        
        # Set reference values to match
        for node in reference.nodes:
            ref_node = reference.get_node_by_id(node.id)
            pred_node = predicted.get_node_by_id(node.id)
            if ref_node and pred_node:
                ref_node.set_value(pred_node.value)
        
        metrics = GraphMetrics(reference, predicted)
        
        # Should have perfect match
        self.assertTrue(metrics.full_graph_match(check_values=True))
        self.assertEqual(metrics.correct_reasoning_edges(check_values=True), 7)
        self.assertEqual(metrics.missing_reasoning_edges(check_values=True), 0)
        self.assertEqual(metrics.hallucinated_reasoning_edges(), 0)


if __name__ == '__main__':
    unittest.main()

