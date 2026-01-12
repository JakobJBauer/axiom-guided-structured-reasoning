class GraphMetrics:    
    def __init__(self, reference_graph, predicted_graph):
        self.reference = reference_graph
        self.predicted = predicted_graph

    # helper
    def _get_end_node(self, graph):
        """Get the end node (node with no outgoing edges) from a graph."""
        end_nodes = [node for node in graph.nodes if graph.get_outgoing_edges(node) == []]
        if len(end_nodes) != 1: raise ValueError("Graph must have exactly one end node")
        return end_nodes[0]

    
    # Additional Metrics
    ## ToDo: Add Tree edit distance metric to closest graph solution


    # Structure Metrics
    def _is_edge_required(self, edge):
        """
        Check if an edge from the reference graph is required for the predicted graph.
        An edge is required if:
        1. The target node exists in the predicted graph
        2. The target node's formula requires the source node to form a valid path
        3. The current edges to that node don't already form a valid path without it
        """
        target_node = self.predicted.get_node_by_id(edge.target)
        
        if target_node is None: return False
        if target_node.formula is None: return False
        if target_node.valid_path_parents is None: return True # default
        
        # Get current incoming edges to target node in predicted graph
        current_incoming = self.predicted.get_incoming_edges(target_node)
        current_parents = set(e.source for e in current_incoming)
        
        # Check if current parents form a valid path
        has_valid_path = any(
            set(valid_path).issubset(current_parents)
            for valid_path in target_node.valid_path_parents
        )
        
        # If we already have a valid path, this edge is not required
        if has_valid_path: return False
        
        # Check if the source node is needed for any valid path
        # (even if the source node doesn't exist in predicted graph yet)
        source_needed = any(
            edge.source in valid_path
            for valid_path in target_node.valid_path_parents
        )
        
        if not source_needed: return False
        
        # Check if adding this edge (and potentially the source node) would create a valid path
        # We check if there's a valid path that includes this source
        potential_parents = current_parents | {edge.source}
        would_create_valid_path = any(
            set(valid_path).issubset(potential_parents)
            for valid_path in target_node.valid_path_parents
        )
        
        # If adding this edge creates a valid path, it's required
        return would_create_valid_path
    
    def _has_valid_path(self, node):
        """
        Check if a node in the predicted graph has a valid path according to its formula.
        """
        if node.formula is None or node.valid_path_parents is None:
            return True  # Leaf nodes or nodes without formulas are always valid
        
        current_incoming = self.predicted.get_incoming_edges(node)
        current_parents = set(e.source for e in current_incoming)
        
        # Check if current parents form a valid path
        return any(
            set(valid_path).issubset(current_parents)
            for valid_path in node.valid_path_parents
        )
    
    def correct_reasoning_edges(self, check_values=False):
        """
        Counts the number of edges that are in the correct position in the reasoning tree.
        """
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
        Counts the number of edges that are in the reference graph but not in the predicted graph,
        AND are actually required for the predicted graph to be valid.
        For minimal reasoning paths, edges that aren't needed don't count as missing.
        """
        required_missing = 0
        for ref_edge in self.reference.get_edges():
            # Check if this edge is in the predicted graph
            if ref_edge not in self.predicted.get_edges():
                # Check if this edge is actually required
                if self._is_edge_required(ref_edge):
                    required_missing += 1
        return required_missing

    def hallucinated_reasoning_edges(self):
        """
        Counts the number of edges that are not in the reference graph.
        """
        return sum(pred_edge not in self.reference.edges for pred_edge in self.predicted.get_edges())

    def full_graph_match(self, check_values=False):
        """
        Returns True if the predicted graph is a valid (possibly minimal) match.
        For minimal reasoning paths, this returns True if:
        1. All edges in the solution are correct (no hallucinations)
        2. All nodes in the solution have valid paths
        3. No required edges are missing
        """
        # Check for hallucinations
        if self.hallucinated_reasoning_edges() > 0:
            return False
        
        # Check that all edges in solution are correct
        if self.correct_reasoning_edges(check_values=check_values) != len(self.predicted.get_edges()):
            return False
        
        # Check that all nodes in solution have valid paths
        for pred_node in self.predicted.nodes:
            if not self._has_valid_path(pred_node):
                return False
        
        # Check that no required edges are missing
        if self.missing_reasoning_edges(check_values=check_values) > 0:
            return False
        
        return True

    
    def longest_correct_reasoning_path(self, check_values=False):
        """
        Finds the longest correct reasoning path in the predicted graph.
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


    # Value Metrics    
    def _compute_binary_metrics(self, pred_value, ref_value):
        if pred_value is None or ref_value is None: return (0, 0, 0, 0)
        
        tp = pred_value and ref_value
        fp = pred_value and not ref_value
        tn = not pred_value and not ref_value
        fn = not pred_value and ref_value
        return (tp, fp, tn, fn)
    
    def _compute_metrics_from_counts(self, tp, fp, tn, fn):
        total = tp + fp + tn + fn
        if total == 0: return (0.0, 0.0, 0.0, 0.0)
        
        accuracy = (tp + tn) / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return (accuracy, precision, recall, f1)
    
    def end_node_metrics(self):
        end_node_ref = self._get_end_node(self.reference)
        end_node_pred = self.predicted.get_node_by_id(end_node_ref.id)
        
        if end_node_pred is None: return 0.0
        
        pred_value = end_node_pred.value
        ref_value = end_node_ref.value
        
        tp, fp, tn, fn = self._compute_binary_metrics(pred_value, ref_value)
        return self._compute_metrics_from_counts(tp, fp, tn, fn)
    
    def average_node_metrics(self):
        """Average accuracy across all nodes"""
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        
        for ref_node in self.reference.nodes:
            pred_node = self.predicted.get_node_by_id(ref_node.id)
            if pred_node is None:
                continue
            
            tp, fp, tn, fn = self._compute_binary_metrics(pred_node.value, ref_node.value)
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        
        return self._compute_metrics_from_counts(total_tp, total_fp, total_tn, total_fn)

    def print_all_metrics(self, check_values=False):
        return f"End Node Metrics (Acc, Prec, Rec, F1): {self.end_node_metrics()}" + " (Ideal: (1.0, 1.0, 1.0, 1.0))" + "\n" +\
        f"Average Node Metrics (Acc, Prec, Rec, F1): {self.average_node_metrics()}" + " (Ideal: (1.0, 1.0, 1.0, 1.0))" + "\n" +\
        f"Longest Correct Reasoning Path (Depth): {self.longest_correct_reasoning_path(check_values=check_values)}" + "\n" +\
        f"Correct Reasoning Edges: {self.correct_reasoning_edges(check_values=check_values)}" + f" (Ideal: {len(self.reference.edges)})" + "\n" +\
        f"Missing Reasoning Edges: {self.missing_reasoning_edges(check_values=check_values)}" + " (Ideal: 0)" + "\n" +\
        f"Hallucinated Reasoning Edges (Count): {self.hallucinated_reasoning_edges()}" + " (Ideal: 0)" + "\n" +\
        f"Full Graph Match: {self.full_graph_match(check_values=check_values)}" + " (Ideal: True)" + "\n"