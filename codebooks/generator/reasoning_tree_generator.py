from graph import Node, Edge, Graph, Formula, Not, And, Or, Equal, Xor, In
import random
from pathlib import Path
import json

class ReasoningTreeGenerator:
    def __init__(
        self,
        goal_depth: int = 3,
        min_branching_factor: int = 1,
        max_branching_factor: int = 2,
        branch_density_factor: float = 0.8, # percentage of how many nodes have higher than min branching factors
        available_formulas: "list[type[Formula]] | None" = None,
        leaf_node_json_path: Path | None = None,
        randomness_factor: float = 0.1, # skip nodes and introduce mix edges randomly
        max_leaf_nodes: int = None, 
        seed=None,
    ):
        if goal_depth <= 0: raise ValueError("goal_depth must be positive")
        if min_branching_factor < 1: raise ValueError("min_branching_factor must be at least 1")
        if max_branching_factor < min_branching_factor: raise ValueError("max_branching_factor cannot be less than min_branching_factor")
        if not 0.0 <= branch_density_factor <= 1.0: raise ValueError("branch_density_factor must be between 0.0 and 1.0")
        if not 0.0 <= randomness_factor <= 0.5: raise ValueError("randomness_factor must be between 0.0 and 0.5")

        self.goal_depth = goal_depth
        self.min_branching_factor = min_branching_factor
        self.max_branching_factor = max_branching_factor
        self.branch_density_factor = branch_density_factor
        self.available_formulas = available_formulas or [Not, And, Or, Xor, Equal, In]
        self.randomness_factor = randomness_factor
        self.max_leaf_nodes = max_leaf_nodes
        self.rng = random.Random(seed)

        leaf_node_json_path = leaf_node_json_path or Path(__file__).parent / "proposed_leaf_nodes.json"
        with open(leaf_node_json_path, "r") as f:
            self.available_leaf_nodes = json.load(f)

    def generate(self) -> Graph:
        graph = self._generate_structure()
        graph = self._populate_formulas(graph)
        graph = self._fill_leaf_nodes(graph)
        graph = self._rename_nodes(graph)
        return graph

    def _generate_structure(self) -> Graph:
        """
            Generates a structural reasoning tree.
            Formula population and id/label updates have to be done in a further step.
        """
        root = Node(id="ROOT")
        all_nodes = [root]
        all_edges = []
        last_row = [root]
        for current_depth in range(1, 1 + self.goal_depth):
            # first we create at least min_branching_factor nodes for the current depth
            new_row = [
                [
                    Node(id=f"NODE_{current_depth}_{i}_{j}")
                    for j in range(self.min_branching_factor)
                ] for i in range(len(last_row))
            ]

            if self.min_branching_factor < self.max_branching_factor: # if we can have denser branches
                # then we decide which nodes get higher branching factors
                denser_branches = self.rng.sample(
                    population=range(len(last_row)),
                    k=round(self.branch_density_factor * len(last_row)),
                )

                # for all denser branches we uniformly sample a higher branching factor, and add it to the new row
                for denser_branch_id in denser_branches:
                    total_nodes = self.rng.randint(self.min_branching_factor + 1, self.max_branching_factor)
                    new_row[denser_branch_id].extend([
                        Node(id=f"NODE_{current_depth}_{denser_branch_id}_{j}")
                        for j in range(self.min_branching_factor, total_nodes)
                    ])
            
            # add edges between the new nodes and the last row
            for i in range(len(new_row)):
                for j in range(len(new_row[i])):
                    target = last_row[i].id if self.rng.random() >= self.randomness_factor else self.rng.choice(last_row).id # choose random parent
                    if current_depth > 1 and self.rng.random() < self.randomness_factor: # skip a node. Pick a target from an edge where the current node is the source
                        relevant_edges = [edge for edge in all_edges if edge.source == target]
                        if relevant_edges: target = self.rng.choice(relevant_edges).target

                    all_edges.append(Edge(source=new_row[i][j].id, target=target))

            # add nodes to the tree and update last_row
            last_row = sum(new_row, [])
            all_nodes.extend(last_row)
        
        graph = Graph(nodes=all_nodes, edges=all_edges)
        graph = self._reduce_leaf_nodes(graph)

        return graph

    
    def _reduce_leaf_nodes(self, graph: Graph) -> Graph:
        if self.max_leaf_nodes is None: return graph
        
        leaf_nodes = graph.get_leaf_nodes()
        if len(leaf_nodes) <= self.max_leaf_nodes: return graph

        chosen_leaf_nodes = self.rng.sample(leaf_nodes, k=self.max_leaf_nodes)
        remaining_leaf_nodes = [n for n in leaf_nodes if n not in chosen_leaf_nodes]

        node_map = {node.id: node for node in chosen_leaf_nodes}
        
        for node in remaining_leaf_nodes: node_map[node.id] = self.rng.choice(chosen_leaf_nodes)

        for edge in graph.get_edges():
            if edge.source in node_map: edge.source = node_map[edge.source].id
            if edge.target in node_map: edge.target = node_map[edge.target].id

        graph.remove_nodes(remaining_leaf_nodes)

        return graph

    
    def _populate_formulas(self, graph: Graph) -> Graph:
        non_leaf_nodes = [node for node in graph.get_nodes() if not graph.is_leaf_node(node)]
        formula_candidates = {}
        for formula in self.available_formulas:
            required_parameters = formula.min_parameter_count()
            if required_parameters not in formula_candidates: formula_candidates[required_parameters] = []
            formula_candidates[required_parameters].append(formula)
        
        for node in non_leaf_nodes:
            parameter_count = len(graph.get_incoming_nodes(node))
            if parameter_count == 1:
                formula = self.rng.choice(formula_candidates[1])
                node.formula = formula(node.id)
            elif parameter_count >= 2:
                formula = self.rng.choice(formula_candidates[2])
                node.formula = formula(node.id)
            else:
                raise ValueError(f"Node {node.id} has {parameter_count} parameters, but only 1 and 2 are supported")
        return graph

    def _fill_leaf_nodes(self, graph: Graph) -> Graph:
        raise NotImplementedError("Not implemented yet")

    def _rename_nodes(self, graph: Graph) -> Graph:
        raise NotImplementedError("Not implemented yet")