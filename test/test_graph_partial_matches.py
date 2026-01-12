"""
Tests comparing old vs new implementation for partial/minimal reasoning paths.

This test file demonstrates the difference between:
- Old implementation: Counts ALL missing edges, regardless of whether they're needed
- New implementation: Only counts missing edges that are actually required

Tests the three key scenarios:
1. Or with minimal path (should NOT penalize missing b->c)
2. And with incomplete path (SHOULD penalize missing b->c)
3. Complex minimal path (should NOT penalize missing b->c)
"""

import unittest
import sys
import os

# Add parent directory to path to import graph modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from graph.graph import Node, Edge, Graph
from graph.formulas import Or, And, Xor
from graph.graph_metrics import GraphMetrics as GraphMetricsNew


class GraphMetricsOld:
    """
    Old implementation of GraphMetrics for comparison.
    This is the original implementation that counted ALL missing edges.
    """
    def __init__(self, reference_graph, predicted_graph):
        self.reference = reference_graph
        self.predicted = predicted_graph

    def _get_end_node(self, graph):
        """Get the end node (node with no outgoing edges) from a graph."""
        end_nodes = [node for node in graph.nodes if graph.get_outgoing_edges(node) == []]
        if len(end_nodes) != 1: raise ValueError("Graph must have exactly one end node")
        return end_nodes[0]

    def correct_reasoning_edges(self, check_values=False):
        """Counts the number of edges that are in the correct position in the reasoning tree."""
        if not check_values:
            return sum(pred_edge in self.reference.edges for pred_edge in self.predicted.get_edges())
        else:
            return sum(
                edge in self.reference.edges and
                self.reference.get_node_by_id(edge.target).value == self.predicted.get_node_by_id(edge.target).value 
                for edge in self.predicted.get_edges()
            )

    def missing_reasoning_edges(self, check_values=False):
        """
        OLD IMPLEMENTATION: Counts ALL edges that are in the reference graph but not in the predicted graph.
        This penalizes missing edges even if they're not needed for a valid minimal path.
        """
        return len(self.reference.get_edges()) - self.correct_reasoning_edges(check_values=check_values)

    def hallucinated_reasoning_edges(self):
        """Counts the number of edges that are not in the reference graph."""
        return sum(pred_edge not in self.reference.edges for pred_edge in self.predicted.get_edges())

    def full_graph_match(self, check_values=False):
        """OLD IMPLEMENTATION: Requires all reference edges to be present."""
        return self.correct_reasoning_edges(check_values=check_values) == len(self.reference.edges) and self.missing_reasoning_edges(check_values=check_values) == 0
    
    def longest_correct_reasoning_path(self, check_values=False):
        """
        OLD IMPLEMENTATION: Finds the longest correct reasoning path in the predicted graph.
        When check_values=True, only considers paths where all nodes have correct values.
        """
        queue = [(self._get_end_node(self.reference), 1)]
        longest_path = 0
        
        while queue:
            node, path_length = queue.pop(0)
            
            pred_node = self.predicted.get_node_by_id(node.id)
            if pred_node is None: continue
            if check_values and pred_node.value != node.value: continue
            
            longest_path = max(longest_path, path_length)
            queue.extend((incoming_node, path_length + 1) for incoming_node in self.reference.get_incoming_nodes(node))
        return longest_path


class TestPartialMatchesComparison(unittest.TestCase):
    """Compare old vs new implementation for partial matches"""
    
    def test_scenario_1_or_minimal_path(self):
        """
        Scenario 1: Or with minimal path
        Reference: a->c, b->c (c = Or("a", "b"))
        Predicted: a->c (minimal path using only "a")
        
        Expected behavior:
        - Old: Should count b->c as missing (penalizes unnecessarily)
        - New: Should NOT count b->c as missing (a->c is sufficient for Or)
        """
        # Setup
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b'))
        ]
        pred_edges = [
            Edge('a', 'c')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Test old implementation
        metrics_old = GraphMetricsOld(reference, predicted)
        old_correct = metrics_old.correct_reasoning_edges()
        old_missing = metrics_old.missing_reasoning_edges()
        old_match = metrics_old.full_graph_match()
        
        # Test new implementation
        metrics_new = GraphMetricsNew(reference, predicted)
        new_correct = metrics_new.correct_reasoning_edges()
        new_missing = metrics_new.missing_reasoning_edges()
        new_match = metrics_new.full_graph_match()
        
        # Assertions
        print(f"\n=== Scenario 1: Or with minimal path ===")
        print(f"Reference edges: 2 (a->c, b->c)")
        print(f"Predicted edges: 1 (a->c)")
        print(f"\nOld Implementation:")
        print(f"  Correct edges: {old_correct}")
        print(f"  Missing edges: {old_missing} (should be 1 - counts b->c)")
        print(f"  Full match: {old_match} (should be False)")
        print(f"\nNew Implementation:")
        print(f"  Correct edges: {new_correct}")
        print(f"  Missing edges: {new_missing} (should be 0 - b->c not required)")
        print(f"  Full match: {new_match} (should be True)")
        
        # Both should have same correct edges
        self.assertEqual(old_correct, new_correct, "Both should count 1 correct edge")
        self.assertEqual(old_correct, 1)
        
        # Old counts missing edge, new doesn't
        self.assertEqual(old_missing, 1, "Old implementation should count b->c as missing")
        self.assertEqual(new_missing, 0, "New implementation should NOT count b->c as missing (not required)")
        
        # Old doesn't match, new does
        self.assertFalse(old_match, "Old implementation should return False (has missing edge)")
        self.assertTrue(new_match, "New implementation should return True (valid minimal path)")
    
    def test_scenario_2_and_incomplete_path(self):
        """
        Scenario 2: And with incomplete path
        Reference: a->c, b->c (c = And("a", "b"))
        Predicted: a->c (incomplete - missing b->c)
        
        Expected behavior:
        - Old: Should count b->c as missing (correctly)
        - New: Should ALSO count b->c as missing (correctly - And requires both)
        """
        # Setup
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=And('a', 'b'))
        ]
        pred_edges = [
            Edge('a', 'c')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Test old implementation
        metrics_old = GraphMetricsOld(reference, predicted)
        old_correct = metrics_old.correct_reasoning_edges()
        old_missing = metrics_old.missing_reasoning_edges()
        old_match = metrics_old.full_graph_match()
        
        # Test new implementation
        metrics_new = GraphMetricsNew(reference, predicted)
        new_correct = metrics_new.correct_reasoning_edges()
        new_missing = metrics_new.missing_reasoning_edges()
        new_match = metrics_new.full_graph_match()
        
        # Assertions
        print(f"\n=== Scenario 2: And with incomplete path ===")
        print(f"Reference edges: 2 (a->c, b->c)")
        print(f"Predicted edges: 1 (a->c)")
        print(f"\nOld Implementation:")
        print(f"  Correct edges: {old_correct}")
        print(f"  Missing edges: {old_missing} (should be 1 - counts b->c)")
        print(f"  Full match: {old_match} (should be False)")
        print(f"\nNew Implementation:")
        print(f"  Correct edges: {new_correct}")
        print(f"  Missing edges: {new_missing} (should be 1 - b->c IS required)")
        print(f"  Full match: {new_match} (should be False)")
        
        # Both should have same correct edges
        self.assertEqual(old_correct, new_correct, "Both should count 1 correct edge")
        self.assertEqual(old_correct, 1)
        
        # Both should count missing edge (And requires both)
        self.assertEqual(old_missing, 1, "Old implementation should count b->c as missing")
        self.assertEqual(new_missing, 1, "New implementation should ALSO count b->c as missing (required for And)")
        
        # Both should not match
        self.assertFalse(old_match, "Old implementation should return False")
        self.assertFalse(new_match, "New implementation should return False (missing required edge)")
    
    def test_scenario_3_complex_minimal_path(self):
        """
        Scenario 3: Complex minimal path
        Reference: a->c, b->c, c->e, d->e (c = Or("a","b"), e = And("c","d"))
        Predicted: a->c, c->e, d->e (minimal path using a instead of b)
        
        Expected behavior:
        - Old: Should count b->c as missing (penalizes unnecessarily)
        - New: Should NOT count b->c as missing (a->c is sufficient for Or)
        """
        # Setup
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Test old implementation
        metrics_old = GraphMetricsOld(reference, predicted)
        old_correct = metrics_old.correct_reasoning_edges()
        old_missing = metrics_old.missing_reasoning_edges()
        old_match = metrics_old.full_graph_match()
        
        # Test new implementation
        metrics_new = GraphMetricsNew(reference, predicted)
        new_correct = metrics_new.correct_reasoning_edges()
        new_missing = metrics_new.missing_reasoning_edges()
        new_match = metrics_new.full_graph_match()
        
        # Assertions
        print(f"\n=== Scenario 3: Complex minimal path ===")
        print(f"Reference edges: 4 (a->c, b->c, c->e, d->e)")
        print(f"Predicted edges: 3 (a->c, c->e, d->e)")
        print(f"\nOld Implementation:")
        print(f"  Correct edges: {old_correct}")
        print(f"  Missing edges: {old_missing} (should be 1 - counts b->c)")
        print(f"  Full match: {old_match} (should be False)")
        print(f"\nNew Implementation:")
        print(f"  Correct edges: {new_correct}")
        print(f"  Missing edges: {new_missing} (should be 0 - b->c not required)")
        print(f"  Full match: {new_match} (should be True)")
        
        # Both should have same correct edges
        self.assertEqual(old_correct, new_correct, "Both should count 3 correct edges")
        self.assertEqual(old_correct, 3)
        
        # Old counts missing edge, new doesn't
        self.assertEqual(old_missing, 1, "Old implementation should count b->c as missing")
        self.assertEqual(new_missing, 0, "New implementation should NOT count b->c as missing (not required)")
        
        # Old doesn't match, new does
        self.assertFalse(old_match, "Old implementation should return False (has missing edge)")
        self.assertTrue(new_match, "New implementation should return True (valid minimal path)")
    
    def test_scenario_4_xor_minimal_path(self):
        """
        Scenario 4: Xor with minimal path
        Reference: a->c, b->c (c = Xor("a", "b"))
        Predicted: a->c (minimal path)
        
        Expected behavior:
        - Old: Should count b->c as missing (penalizes unnecessarily)
        - New: Should NOT count b->c as missing (a->c is sufficient for Xor)
        """
        # Setup
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Xor('a', 'b'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Xor('a', 'b'))
        ]
        pred_edges = [
            Edge('a', 'c')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        # Test old implementation
        metrics_old = GraphMetricsOld(reference, predicted)
        old_correct = metrics_old.correct_reasoning_edges()
        old_missing = metrics_old.missing_reasoning_edges()
        old_match = metrics_old.full_graph_match()
        
        # Test new implementation
        metrics_new = GraphMetricsNew(reference, predicted)
        new_correct = metrics_new.correct_reasoning_edges()
        new_missing = metrics_new.missing_reasoning_edges()
        new_match = metrics_new.full_graph_match()
        
        # Assertions
        print(f"\n=== Scenario 4: Xor with minimal path ===")
        print(f"Reference edges: 2 (a->c, b->c)")
        print(f"Predicted edges: 1 (a->c)")
        print(f"\nOld Implementation:")
        print(f"  Correct edges: {old_correct}")
        print(f"  Missing edges: {old_missing} (should be 1 - counts b->c)")
        print(f"  Full match: {old_match} (should be False)")
        print(f"\nNew Implementation:")
        print(f"  Correct edges: {new_correct}")
        print(f"  Missing edges: {new_missing} (should be 0 - b->c not required)")
        print(f"  Full match: {new_match} (should be True)")
        
        # Both should have same correct edges
        self.assertEqual(old_correct, new_correct, "Both should count 1 correct edge")
        self.assertEqual(old_correct, 1)
        
        # Old counts missing edge, new doesn't
        self.assertEqual(old_missing, 1, "Old implementation should count b->c as missing")
        self.assertEqual(new_missing, 0, "New implementation should NOT count b->c as missing (not required)")
        
        # Old doesn't match, new does
        self.assertFalse(old_match, "Old implementation should return False (has missing edge)")
        self.assertTrue(new_match, "New implementation should return True (valid minimal path)")


class TestAdvancedScenarios(unittest.TestCase):
    """Advanced scenarios comparing old vs new implementation"""
    
    def test_longest_path_minimal_vs_complete(self):
        """
        Test longest path: minimal path may be shorter but still valid.
        Reference: a->c, b->c, c->e, d->e (c = Or("a","b"), e = And("c","d"))
        Predicted (minimal): a->c, c->e, d->e
        
        Both should find the same longest path (depth 3: a->c->e or d->e->e)
        But new recognizes this as valid, old doesn't.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        
        print(f"\n=== Longest Path: Minimal vs Complete ===")
        print(f"Old longest path: {old_path}")
        print(f"New longest path: {new_path}")
        print(f"Both should find same path length (structure matches)")
        
        # Both should find the same longest path (structure is the same)
        self.assertEqual(old_path, new_path, "Both should find same longest path length")
        self.assertEqual(old_path, 3)  # a->c->e or d->e (depth 3)
        
        # But new recognizes it as valid, old doesn't
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())
    
    def test_multiple_branching_paths(self):
        """
        Test multiple branching paths with Or operations.
        Reference: a->d, b->d, c->d, d->e (d = Or("a","b","c"), e = Not("d"))
        Predicted (minimal): a->d, d->e (using only one branch)
        
        New should recognize this as valid minimal path.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c'),
            Node('d', formula=Or('a', 'b', 'c')),
            Node('e', formula=And('d', 'd'))  # e depends on d
        ]
        ref_edges = [
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('c', 'd'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('d', formula=Or('a', 'b', 'c')),
            Node('e', formula=And('d', 'd'))
        ]
        pred_edges = [
            Edge('a', 'd'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Multiple Branching Paths ===")
        print(f"Reference: 4 edges (a->d, b->d, c->d, d->e)")
        print(f"Predicted: 2 edges (a->d, d->e) - minimal path")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (should be 2)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (should be 0)")
        print(f"Old match: {metrics_old.full_graph_match()}")
        print(f"New match: {metrics_new.full_graph_match()}")
        
        self.assertEqual(metrics_old.missing_reasoning_edges(), 2)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())
    
    def test_deeply_nested_minimal_path(self):
        """
        Test deeply nested formulas with minimal path.
        Reference: a->b, c->b, b->d, e->d, d->f (b=Or("a","c"), d=And("b","e"), f=Or("d"))
        Predicted (minimal): a->b, b->d, e->d, d->f (skipping c->b)
        
        New should recognize this as valid.
        """
        ref_nodes = [
            Node('a'),
            Node('c'),
            Node('b', formula=Or('a', 'c')),
            Node('e'),
            Node('d', formula=And('b', 'e')),
            Node('f', formula=Or('d'))
        ]
        ref_edges = [
            Edge('a', 'b'),
            Edge('c', 'b'),
            Edge('b', 'd'),
            Edge('e', 'd'),
            Edge('d', 'f')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('b', formula=Or('a', 'c')),
            Node('e'),
            Node('d', formula=And('b', 'e')),
            Node('f', formula=Or('d'))
        ]
        pred_edges = [
            Edge('a', 'b'),
            Edge('b', 'd'),
            Edge('e', 'd'),
            Edge('d', 'f')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Deeply Nested Minimal Path ===")
        print(f"Reference: 5 edges, depth 4 (a->b->d->f)")
        print(f"Predicted: 4 edges, same depth (a->b->d->f)")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()}")
        print(f"New missing: {metrics_new.missing_reasoning_edges()}")
        
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        
        self.assertEqual(old_path, new_path, "Both should find same longest path")
        self.assertEqual(old_path, 4)  # a->b->d->f = 4 nodes
        self.assertEqual(metrics_old.missing_reasoning_edges(), 1)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())
    
    def test_mixed_required_and_optional_edges(self):
        """
        Test scenario with both required (And) and optional (Or) edges.
        Reference: a->c, b->c, c->e, d->e (c=Or("a","b"), e=And("c","d"))
        Predicted: a->c, c->e, d->e (minimal - missing b->c which is optional)
        
        New should recognize: b->c is optional (Or), but c->e and d->e are required (And).
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Mixed Required and Optional Edges ===")
        print(f"Reference: 4 edges (a->c, b->c optional, c->e required, d->e required)")
        print(f"Predicted: 3 edges (a->c, c->e, d->e) - all required edges present")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (counts optional b->c)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (ignores optional b->c)")
        
        # Both should have same correct edges
        self.assertEqual(metrics_old.correct_reasoning_edges(), 3)
        self.assertEqual(metrics_new.correct_reasoning_edges(), 3)
        
        # Old counts optional edge, new doesn't
        self.assertEqual(metrics_old.missing_reasoning_edges(), 1)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        
        # Both should find same longest path
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        self.assertEqual(old_path, new_path)
        self.assertEqual(old_path, 3)  # a->c->e or d->e
    
    def test_longest_path_with_missing_intermediate_node(self):
        """
        Test longest path when intermediate node is missing but path is still valid.
        Reference: a->b, b->c, c->d (linear chain)
        Predicted: a->c, c->d (missing b, but a->c->d is valid if c accepts a directly)
        
        Actually, this scenario might not work if c requires b. Let me use a different one:
        Reference: a->c, b->c, c->d (c = Or("a","b"))
        Predicted: a->c, c->d (missing b, but a->c->d is valid)
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b')),
            Node('d', formula=Or('c'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'd')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b')),
            Node('d', formula=Or('c'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('c', 'd')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        
        print(f"\n=== Longest Path with Missing Intermediate Node ===")
        print(f"Reference: a->c, b->c, c->d (depth 3)")
        print(f"Predicted: a->c, c->d (same depth 3, but missing b)")
        print(f"Old longest path: {old_path}")
        print(f"New longest path: {new_path}")
        
        # Both should find same longest path (a->c->d = 3 nodes)
        self.assertEqual(old_path, new_path)
        self.assertEqual(old_path, 3)
        
        # But new recognizes it as valid
        self.assertEqual(metrics_old.missing_reasoning_edges(), 1)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
    
    def test_value_based_comparison_minimal_path(self):
        """
        Test value-based metrics with minimal paths.
        Both implementations should handle values the same way, but
        structural recognition differs.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=Or('a', 'b'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c')
        ]
        reference = Graph(ref_nodes, ref_edges)
        reference.get_node_by_id('a').set_value(True)
        reference.get_node_by_id('b').set_value(False)
        reference.get_node_by_id('c').set_value(True)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=Or('a', 'b'))
        ]
        pred_edges = [
            Edge('a', 'c')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        predicted.get_node_by_id('a').set_value(True)
        predicted.get_node_by_id('c').set_value(True)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Value-Based Comparison with Minimal Path ===")
        print(f"With value checking:")
        print(f"  Old correct edges: {metrics_old.correct_reasoning_edges(check_values=True)}")
        print(f"  New correct edges: {metrics_new.correct_reasoning_edges(check_values=True)}")
        print(f"  Old missing edges: {metrics_old.missing_reasoning_edges(check_values=True)}")
        print(f"  New missing edges: {metrics_new.missing_reasoning_edges(check_values=True)}")
        
        # Both should have same correct edges with values
        self.assertEqual(metrics_old.correct_reasoning_edges(check_values=True), 1)
        self.assertEqual(metrics_new.correct_reasoning_edges(check_values=True), 1)
        
        # But missing edges differ
        self.assertEqual(metrics_old.missing_reasoning_edges(check_values=True), 1)
        self.assertEqual(metrics_new.missing_reasoning_edges(check_values=True), 0)
    
    def test_complex_or_with_multiple_valid_paths(self):
        """
        Test complex Or with multiple valid path options.
        Reference: a->d, b->d, c->d, d->e (d = Or("a","b","c"), e = And("d"))
        Predicted: b->d, d->e (using middle option)
        
        New should recognize this as valid.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c'),
            Node('d', formula=Or('a', 'b', 'c')),
            Node('e', formula=And('d'))
        ]
        ref_edges = [
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('c', 'd'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('b'),
            Node('d', formula=Or('a', 'b', 'c')),
            Node('e', formula=And('d'))
        ]
        pred_edges = [
            Edge('b', 'd'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Complex Or with Multiple Valid Paths ===")
        print(f"Reference: 4 edges (a->d, b->d, c->d, d->e)")
        print(f"Predicted: 2 edges (b->d, d->e) - using middle option")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (should be 2)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (should be 0)")
        
        self.assertEqual(metrics_old.missing_reasoning_edges(), 2)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())
        
        # Longest path should be same (b->d->e = 3 nodes)
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        self.assertEqual(old_path, new_path)
        self.assertEqual(old_path, 3)
    
    def test_incomplete_and_should_fail_both(self):
        """
        Test that incomplete And correctly fails in both implementations.
        Reference: a->c, b->c, c->e, d->e (c=And("a","b"), e=And("c","d"))
        Predicted: a->c, c->e, d->e (missing b->c which is REQUIRED)
        
        Both should correctly identify this as incomplete.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c', formula=And('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        ref_edges = [
            Edge('a', 'c'),
            Edge('b', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('c', formula=And('a', 'b')),
            Node('d'),
            Node('e', formula=And('c', 'd'))
        ]
        pred_edges = [
            Edge('a', 'c'),
            Edge('c', 'e'),
            Edge('d', 'e')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Incomplete And Should Fail Both ===")
        print(f"Reference: 4 edges (a->c, b->c REQUIRED, c->e, d->e)")
        print(f"Predicted: 3 edges (a->c, c->e, d->e) - missing REQUIRED b->c")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (should be 1)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (should be 1)")
        
        # Both should correctly identify missing required edge
        self.assertEqual(metrics_old.missing_reasoning_edges(), 1)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 1)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertFalse(metrics_new.full_graph_match())
    
    def test_xor_with_three_options_minimal(self):
        """
        Test Xor with three options, using minimal path.
        Reference: a->d, b->d, c->d (d = Xor("a","b","c"))
        Predicted: a->d (minimal - only one option needed for Xor)
        
        New should recognize this as valid.
        """
        ref_nodes = [
            Node('a'),
            Node('b'),
            Node('c'),
            Node('d', formula=Xor('a', 'b', 'c'))
        ]
        ref_edges = [
            Edge('a', 'd'),
            Edge('b', 'd'),
            Edge('c', 'd')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('d', formula=Xor('a', 'b', 'c'))
        ]
        pred_edges = [
            Edge('a', 'd')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Xor with Three Options Minimal ===")
        print(f"Reference: 3 edges (a->d, b->d, c->d)")
        print(f"Predicted: 1 edge (a->d) - minimal, only one needed for Xor")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (should be 2)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (should be 0)")
        
        self.assertEqual(metrics_old.missing_reasoning_edges(), 2)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())
    
    def test_chain_of_or_operations(self):
        """
        Test chain of Or operations where each level can be minimal.
        Reference: a->b, c->b, b->d, e->d, d->f (b=Or("a","c"), d=Or("b","e"), f=Or("d"))
        Predicted: a->b, b->d, d->f (minimal path through all Or operations)
        
        New should recognize this as valid minimal path.
        """
        ref_nodes = [
            Node('a'),
            Node('c'),
            Node('b', formula=Or('a', 'c')),
            Node('e'),
            Node('d', formula=Or('b', 'e')),
            Node('f', formula=Or('d'))
        ]
        ref_edges = [
            Edge('a', 'b'),
            Edge('c', 'b'),
            Edge('b', 'd'),
            Edge('e', 'd'),
            Edge('d', 'f')
        ]
        reference = Graph(ref_nodes, ref_edges)
        
        pred_nodes = [
            Node('a'),
            Node('b', formula=Or('a', 'c')),
            Node('d', formula=Or('b', 'e')),
            Node('f', formula=Or('d'))
        ]
        pred_edges = [
            Edge('a', 'b'),
            Edge('b', 'd'),
            Edge('d', 'f')
        ]
        predicted = Graph(pred_nodes, pred_edges)
        
        metrics_old = GraphMetricsOld(reference, predicted)
        metrics_new = GraphMetricsNew(reference, predicted)
        
        print(f"\n=== Chain of Or Operations ===")
        print(f"Reference: 5 edges, depth 4 (a->b->d->f)")
        print(f"Predicted: 3 edges, same depth (a->b->d->f) - minimal")
        print(f"Old missing: {metrics_old.missing_reasoning_edges()} (should be 2)")
        print(f"New missing: {metrics_new.missing_reasoning_edges()} (should be 0)")
        
        old_path = metrics_old.longest_correct_reasoning_path()
        new_path = metrics_new.longest_correct_reasoning_path()
        
        self.assertEqual(old_path, new_path, "Both should find same longest path")
        self.assertEqual(old_path, 4)  # a->b->d->f = 4 nodes
        self.assertEqual(metrics_old.missing_reasoning_edges(), 2)
        self.assertEqual(metrics_new.missing_reasoning_edges(), 0)
        self.assertFalse(metrics_old.full_graph_match())
        self.assertTrue(metrics_new.full_graph_match())


class TestSummaryComparison(unittest.TestCase):
    """Summary comparison showing the key differences"""
    
    def test_summary_table(self):
        """
        Print a summary table comparing old vs new behavior
        """
        print("\n" + "="*70)
        print("SUMMARY: Old vs New Implementation Comparison")
        print("="*70)
        print("\nKey Difference:")
        print("  Old: missing_edges = total_reference_edges - correct_edges")
        print("  New: missing_edges = count of missing edges that are REQUIRED")
        print("\n" + "-"*70)
        print("Scenario                    | Old Missing | New Missing | Old Match | New Match")
        print("-"*70)
        print("Or minimal (a->c only)      |      1      |      0      |   False   |   True")
        print("And incomplete (a->c only) |      1      |      1      |   False   |   False")
        print("Complex minimal             |      1      |      0      |   False   |   True")
        print("Xor minimal (a->c only)     |      1      |      0      |   False   |   True")
        print("-"*70)
        print("\nConclusion:")
        print("  - Old implementation: Always penalizes missing edges, even if not needed")
        print("  - New implementation: Only penalizes missing edges that are actually required")
        print("  - For Or/Xor: New correctly allows minimal paths")
        print("  - For And: Both correctly require all edges")
        print("="*70 + "\n")


if __name__ == '__main__':
    unittest.main(verbosity=2)

