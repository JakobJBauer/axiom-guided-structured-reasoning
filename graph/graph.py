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
    
    def __eq__(self, other):
        """Compare two graphs for structural and formula equality."""
        if not isinstance(other, Graph):
            return False
        
        # Compare node IDs
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
        
        return True
    
    def __str__(self) -> str:
        return f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges:\n" + \
        "\n".join([f"Node {node.id}: {node.label} ({node.value})" for node in self.nodes]) + "\n" +\
        "\n".join([f"Edge {edge.source} -> {edge.target}" for edge in self.edges]) + "\n" +\
        "="*50 + "\n"