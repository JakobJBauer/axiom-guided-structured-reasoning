class Node:
    def __init__(self, id, label=None, value=None, formula=None, valid_path_parents=None):
        """
        id: The id of the node.
        label: The label of the node.
        value: The value of the node.
        formula: The formula of the node (can be a Formula object or callable function).
        valid_path_parents: The parent nodes that are required for the formula to be valid. List of lists, each inner list is a valid path of parent nodes.
                          If None and formula is a Formula object, will be auto-inferred from the formula.
        """
        self.id = id
        self.label = label if label is not None else id
        self.value = value
        self.formula = formula
        
        # Auto-infer valid_path_parents if not provided and formula is a Formula object
        if valid_path_parents is None and formula is not None:
            if hasattr(formula, 'get_valid_path_parents'):
                self.valid_path_parents = formula.get_valid_path_parents()
            else:
                self.valid_path_parents = None
        else:
            self.valid_path_parents = valid_path_parents
    
    def set_value(self, value):
        self.value = value
    
    def compute_value(self, incoming_values):
        if self.formula is None: return None
        return self.formula(incoming_values)

    def __eq__(self, other):
        if not isinstance(other, Node): return False
        return self.id == other.id
    
    def copy(self):
        """Create a copy of this node with the same attributes."""
        return Node(self.id, self.label, self.value, self.formula, self.valid_path_parents)


class Edge:
    def __init__(self, source, target):
        self.source = source
        self.target = target
    
    def __iter__(self):
        yield self.source
        yield self.target

    def __eq__(self, other):
        if not isinstance(other, Edge): return False
        return self.source == other.source and self.target == other.target

class Graph:
    def __init__(self, nodes=None, edges=None):
        self.nodes = []
        self.edges = []

        for node in nodes: self.add_node(node)
        for edge in edges: self.add_edge(edge)

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.id == id:
                return node
        return None
    
    def get_incoming_edges(self, node):
        return [edge for edge in self.edges if edge.target == node.id]
    
    def get_outgoing_edges(self, node):
        return [edge for edge in self.edges if edge.source == node.id]
    
    def get_incoming_nodes(self, node):
        incoming_edges = self.get_incoming_edges(node)
        return [self.get_node_by_id(edge.source) for edge in incoming_edges]
    
    def is_leaf_node(self, node):
        return len(self.get_incoming_edges(node)) == 0
    
    def get_leaf_nodes(self):
        return [node for node in self.nodes if self.is_leaf_node(node)]
    
    def topological_sort(self):
        # Build adjacency list for incoming edges
        in_degree = {node.id: 0 for node in self.nodes}
        for edge in self.edges:
            in_degree[edge.target] += 1
        
        # Find nodes with no incoming edges
        queue = [node.id for node in self.nodes if in_degree[node.id] == 0]
        result = []
        
        while queue:
            node_id = queue.pop(0)
            node = self.get_node_by_id(node_id)
            result.append(node)
            
            for edge in self.get_outgoing_edges(node):
                in_degree[edge.target] -= 1
                if in_degree[edge.target] == 0:
                    queue.append(edge.target)
        
        return result

    def leaf_values_set(self):
        "Returns True if all leaf values are set"
        leaves = self.get_leaf_nodes()
        return all(leave.value is not None for leave in leaves)

    def non_leaf_formula_set(self):
        leaf_node_ids = {node.id for node in self.get_leaf_nodes()}
        non_leaves = [node for node in self.nodes if node.id not in leaf_node_ids]
        return all(non_leaf.formula is not None for non_leaf in non_leaves)
    
    def auto_infer_values(self):
        if not self.leaf_values_set():
            raise ValueError("Graph has undefined leaf node values")

        # Get nodes in topological order
        sorted_nodes = self.topological_sort()
    
        for node in sorted_nodes:
            if node.formula is None: continue
            
            incoming_nodes = self.get_incoming_nodes(node)
            incoming_values = {n.id: n.value for n in incoming_nodes if n.value is not None}
            
            # Compute value using formula
            computed_value = node.compute_value(incoming_values)
            if computed_value is not None: node.value = computed_value

    def copy(self):
        """Create a deep copy of this graph with copied nodes and edges."""
        copied_nodes = [node.copy() for node in self.nodes]
        copied_edges = [Edge(edge.source, edge.target) for edge in self.edges]
        return Graph(copied_nodes, copied_edges)
    
    def __eq__(self, other, check_ids=True):
        """
        Compare two graphs for structural and formula equality.
        
        Args:
            other: The other graph to compare with
            check_ids: If True, node IDs must match exactly. If False, compares
                      structurally using topological order mapping (ignores node IDs).
                      Default is True for backward compatibility.
        """
        if not isinstance(other, Graph):
            return False
        
        if check_ids:
            # Original comparison: node IDs must match exactly
            node_ids1 = {node.id for node in self.nodes}
            node_ids2 = {node.id for node in other.nodes}
            if node_ids1 != node_ids2:
                return False
            
            # Compare edges
            edges1 = {(edge.source, edge.target) for edge in self.edges}
            edges2 = {(edge.source, edge.target) for edge in other.edges}
            if edges1 != edges2:
                return False
            
            # Compare formulas for each node
            for node_id in node_ids1:
                node1 = self.get_node_by_id(node_id)
                node2 = other.get_node_by_id(node_id)
                
                # Compare formulas
                if node1.formula is None and node2.formula is None:
                    continue
                if node1.formula is None or node2.formula is None:
                    return False
                
                # Convert formulas to strings for comparison (use repr for unambiguous comparison)
                formula1_str = repr(node1.formula)
                formula2_str = repr(node2.formula)
                if formula1_str != formula2_str:
                    return False
        else:
            # Structural comparison: ignore node IDs, use topological order mapping
            # Must have same number of nodes
            if len(self.nodes) != len(other.nodes):
                return False
            
            # Must have same number of edges
            if len(self.edges) != len(other.edges):
                return False
            
            # Get topological order for both graphs
            try:
                topo1 = self.topological_sort()
                topo2 = other.topological_sort()
            except Exception:
                # If topological sort fails, fall back to regular comparison
                return False
            
            if len(topo1) != len(topo2):
                return False
            
            # Create mapping from node position to node
            node_map1 = {i: node for i, node in enumerate(topo1)}
            node_map2 = {i: node for i, node in enumerate(topo2)}
            
            # Compare edge structure using positional indices
            def get_edge_set(graph, node_map):
                """Get edges as (source_pos, target_pos) tuples."""
                edge_set = set()
                pos_by_id = {node.id: pos for pos, node in node_map.items()}
                for edge in graph.edges:
                    source_pos = pos_by_id.get(edge.source)
                    target_pos = pos_by_id.get(edge.target)
                    if source_pos is not None and target_pos is not None:
                        edge_set.add((source_pos, target_pos))
                return edge_set
            
            edges1 = get_edge_set(self, node_map1)
            edges2 = get_edge_set(other, node_map2)
            if edges1 != edges2:
                return False
            
            # Compare formulas by position
            for pos in range(len(topo1)):
                node1 = node_map1[pos]
                node2 = node_map2[pos]
                
                # Both must have formula or both must not have formula
                if (node1.formula is None) != (node2.formula is None):
                    return False
                
                if node1.formula is None:
                    continue
                
                # Compare formulas by converting node IDs to positional indices
                formula1_str = self._normalize_formula_repr(node1.formula, node_map1)
                formula2_str = other._normalize_formula_repr(node2.formula, node_map2)
                
                if formula1_str != formula2_str:
                    return False
        
        return True
    
    def _normalize_formula_repr(self, formula, node_map):
        """
        Convert formula to string representation with node IDs replaced by positional indices.
        This allows comparing formulas from different graphs that may have different node IDs.
        """
        from graph.formulas.formula import Formula
        
        pos_by_id = {node.id: pos for pos, node in node_map.items()}
        
        def normalize_arg(arg):
            """Normalize a formula argument (node ID, value, or nested formula)."""
            if isinstance(arg, str):
                # Check if it's a node ID
                if arg in pos_by_id:
                    return f"node_{pos_by_id[arg]}"
                # Otherwise it's a value, keep as-is
                return repr(arg)
            elif isinstance(arg, Formula):
                # Recursively normalize nested formulas
                return self._normalize_formula_repr(arg, node_map)
            else:
                return repr(arg)
        
        from graph.formulas import Not, And, Or, Xor, Equal, In
        
        if isinstance(formula, Not):
            arg = formula.key_or_formula
            normalized_arg = normalize_arg(arg)
            return f"Not({normalized_arg})"
        
        elif isinstance(formula, And):
            args = [normalize_arg(arg) for arg in formula.keys_or_formulas]
            return f"And({', '.join(args)})"
        
        elif isinstance(formula, Or):
            args = [normalize_arg(arg) for arg in formula.keys_or_formulas]
            return f"Or({', '.join(args)})"
        
        elif isinstance(formula, Xor):
            args = [normalize_arg(arg) for arg in formula.keys_or_formulas]
            return f"Xor({', '.join(args)})"
        
        elif isinstance(formula, Equal):
            key = normalize_arg(formula.key)
            value = repr(formula.value)
            return f"Equal({key}, {value})"
        
        elif isinstance(formula, In):
            key = normalize_arg(formula.key)
            values = [repr(v) for v in formula.values]
            return f"In({key}, [{', '.join(values)}])"
        
        else:
            # Fallback to repr
            return repr(formula)
    
    def __str__(self) -> str:
        return f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges:\n" + \
        "\n".join([f"Node {node.id}: {node.label} ({node.value})" for node in self.nodes]) + "\n" +\
        "\n".join([f"Edge {edge.source} -> {edge.target}" for edge in self.edges]) + "\n" +\
        "="*50 + "\n"