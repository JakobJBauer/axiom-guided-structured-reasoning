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
    ## ToDo: Implement Subsumption relations (another lambda function for auto-inference)
    ## ToDo: Add Tree edit distance metric to closest graph solution


    # Structure Metrics
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
        Counts the number of edges that are in the reference graph but not in the predicted graph.
        """
        return len(self.reference.get_edges()) - self.correct_reasoning_edges(check_values=check_values)

    def hallucinated_reasoning_edges(self):
        """
        Counts the number of edges that are not in the reference graph.
        """
        return sum(pred_edge not in self.reference.edges for pred_edge in self.predicted.get_edges())

    def full_graph_match(self, check_values=False):
        return self.correct_reasoning_edges(check_values=check_values) == len(self.reference.edges) and self.missing_reasoning_edges(check_values=check_values) == 0

    
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