from graph import Node, Edge, Graph, Formula, Not, And, Or, Equal, Xor
import random

class ReasoningTreeGenerator:
    def __init__(
        self,
        min_nodes: int = 5,
        max_nodes: int = 5,
        min_depth: int = 3,
        max_depth: int = 3,
        min_branching_factor: int = 1,
        max_branching_factor: int = 2,
        prefer_balanced: float = 0.8,
        available_formulas: "list[type[Formula]] | None" = None,
        seed=None,
    ):
        if min_nodes <= 0 or max_nodes <= 0: raise ValueError("min_nodes and max_nodes must be positive")
        if min_nodes > max_nodes: raise ValueError("min_nodes cannot be greater than max_nodes")
        if min_depth <= 0 or max_depth <= 0: raise ValueError("min_depth and max_depth must be positive")
        if min_depth > max_depth: raise ValueError("min_depth cannot be greater than max_depth")
        if min_branching_factor < 1: raise ValueError("min_branching_factor must be at least 1")
        if max_branching_factor < min_branching_factor: raise ValueError("max_branching_factor cannot be less than min_branching_factor")

        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.min_branching_factor = min_branching_factor
        self.max_branching_factor = max_branching_factor
        self.prefer_balanced = prefer_balanced
        self.available_formulas = available_formulas or [Not, And, Or, Xor, Equal, In]
        self.rng = random.Random(seed)

    def _choose_num_nodes_and_depth(self) -> tuple[int, int]:
        candidates = []
        for _ in range(32):
            n = self.rng.randint(self.min_nodes, self.max_nodes)
            d = self.rng.randint(self.min_depth, self.max_depth)

            # Minimal nodes for depth d is a simple chain of length d
            min_nodes_for_d = d + 1

            # Max nodes for depth d given max_branching_factor (full b-ary tree)
            b = self.max_branching_factor
            if b == 1:
                max_nodes_for_d = d + 1
            else:
                max_nodes_for_d = (b ** (d + 1) - 1) // (b - 1)

            if min_nodes_for_d <= n <= max_nodes_for_d:
                candidates.append((n, d))

        if not candidates:
            raise ValueError(
                "No feasible (num_nodes, depth) combination found for the given "
                "node and branching_factor constraints."
            )

        return self.rng.choice(candidates)

    def _build_tree_structure(self, num_nodes: int, depth: int):
        """
        Construct a rooted tree with:
        - exactly num_nodes nodes
        - exact depth (root-to-leaf distance) == depth
        - branching factor in [min_branching_factor, max_branching_factor] for all internal nodes

        The internal construction uses parent->children adjacency for convenience,
        but when we create the final Graph we will use child->parent edges
        (matching the rest of the codebase).
        """

        class TempNode:
            __slots__ = ("id", "depth", "children")

            def __init__(self, id_: str, depth_: int):
                self.id = id_
                self.depth = depth_
                self.children: list["TempNode"] = []

        nodes: list[TempNode] = []

        # 1) Build a backbone path of length `depth` to ensure the required depth
        root = TempNode("n0", 0)
        nodes.append(root)
        current = root
        for i in range(1, depth + 1):
            child = TempNode(f"n{i}", i)
            current.children.append(child)
            nodes.append(child)
            current = child

        if len(nodes) > num_nodes:
            raise ValueError("Requested num_nodes is smaller than minimal nodes for the chosen depth.")

        remaining_nodes = num_nodes - len(nodes)

        # 2) Ensure all nodes on the backbone that are internal (depth < depth)
        #    satisfy the minimum branching factor by adding leaf children.
        next_id = len(nodes)
        for node in nodes:
            if node.depth == depth:
                continue  # leaf of the backbone

            current_children = len(node.children)
            required_extra = max(0, self.min_branching_factor - current_children)
            capacity = self.max_branching_factor - current_children
            if required_extra > capacity:
                raise ValueError(
                    "Cannot satisfy min_branching_factor for backbone nodes with "
                    "the current max_branching_factor."
                )

            if required_extra > remaining_nodes:
                raise ValueError(
                    "Not enough nodes available to satisfy min_branching_factor "
                    "requirements on the backbone."
                )

            for _ in range(required_extra):
                child = TempNode(f"n{next_id}", node.depth + 1)
                next_id += 1
                node.children.append(child)
                nodes.append(child)
                remaining_nodes -= 1

        # 3) Distribute remaining nodes while preserving depth and branching constraints.
        #    Any new internal node will immediately receive at least min_branching_factor children.
        while remaining_nodes > 0:
            candidates: list[TempNode] = []
            for node in nodes:
                if node.depth >= depth:
                    continue  # cannot add children without exceeding depth
                current_children = len(node.children)
                if current_children >= self.max_branching_factor:
                    continue

                remaining_capacity = self.max_branching_factor - current_children

                if current_children == 0:
                    # Turning a leaf into an internal node requires at least min_branching_factor children
                    if remaining_capacity >= self.min_branching_factor and remaining_nodes >= self.min_branching_factor:
                        candidates.append(node)
                else:
                    # Already an internal node with >= min_branching_factor children
                    # can take at least one more child
                    if remaining_capacity >= 1 and remaining_nodes >= 1:
                        candidates.append(node)

            if not candidates:
                raise ValueError(
                    "Unable to place all nodes without violating branching or depth "
                    "constraints. Try relaxing constraints or increasing ranges."
                )

            parent = self.rng.choice(candidates)
            current_children = len(parent.children)
            remaining_capacity = self.max_branching_factor - current_children

            if current_children == 0:
                alloc_min = self.min_branching_factor
            else:
                alloc_min = 1

            alloc_max = min(remaining_capacity, remaining_nodes)
            if alloc_min > alloc_max:
                # This parent ended up unusable; retry the loop to pick another candidate
                continue

            # For some variability, choose a random number in [alloc_min, alloc_max]
            alloc = self.rng.randint(alloc_min, alloc_max)

            for _ in range(alloc):
                child = TempNode(f"n{next_id}", parent.depth + 1)
                next_id += 1
                parent.children.append(child)
                nodes.append(child)
                remaining_nodes -= 1
                if remaining_nodes == 0:
                    break

        # Sanity checks
        if len(nodes) != num_nodes:
            raise ValueError("Internal error: constructed tree does not have the requested number of nodes.")

        max_depth_actual = max(n.depth for n in nodes)
        if not (self.min_depth <= max_depth_actual <= self.max_depth):
            raise ValueError(
                f"Constructed tree depth {max_depth_actual} is outside allowed range "
                f"[{self.min_depth}, {self.max_depth}]."
            )

        return root, nodes

    def _make_formula_for_node(self, node_id: str, child_ids: list[str]) -> Formula:
        """Create a random formula for an internal node given its children."""
        if not child_ids:
            return None

        formula_cls = self.rng.choice(self.available_formulas)

        # Helper to pick a non-empty subset of child_ids
        def choose_subset(ids: list[str]) -> list[str]:
            k_min = 1
            k_max = len(ids)
            k = self.rng.randint(k_min, k_max)
            return self.rng.sample(ids, k)

        if formula_cls is Not:
            if len(child_ids) == 1 or self.rng.random() < 0.5:
                arg = self.rng.choice(child_ids)
                return Not(arg)
            else:
                inner_cls = self.rng.choice([And, Or, Xor])
                subset = choose_subset(child_ids)
                inner = inner_cls(*subset)
                return Not(inner)

        elif formula_cls in (And, Or, Xor):
            subset = choose_subset(child_ids)
            return formula_cls(*subset)

        elif formula_cls is Equal:
            key = self.rng.choice(child_ids)
            # We use boolean values throughout the graph
            value = self.rng.choice([True, False])
            return Equal(key, value)

        # Fallback in case of unexpected formula class
        subset = choose_subset(child_ids)
        return And(*subset)

    def generate_n_trees(self, n: int) -> list[Graph]:
        return [self.generate_reasoning_tree() for _ in range(n)]

    def generate_reasoning_tree(self) -> Graph:
        """
        Generate a single reasoning tree as a Graph.

        - Leaves have boolean values and no formulas.
        - Internal nodes have formulas constructed from their children.
        - Edges are oriented child -> parent, matching the rest of the codebase.
        """
        num_nodes, depth = self._choose_num_nodes_and_depth()
        root, temp_nodes = self._build_tree_structure(num_nodes, depth)

        # Map temp nodes to Node objects
        id_to_node: dict[str, Node] = {}

        # First, create all Node instances without formulas
        for temp in temp_nodes:
            id_to_node[temp.id] = Node(id=temp.id)

        # Determine which nodes are leaves vs internal
        leaf_ids = {n.id for n in temp_nodes if not n.children}

        # Assign random boolean values to leaves
        for leaf_id in leaf_ids:
            id_to_node[leaf_id].set_value(self.rng.choice([True, False]))

        # Assign formulas to internal nodes
        for temp in temp_nodes:
            if not temp.children:
                continue
            child_ids = [child.id for child in temp.children]
            formula = self._make_formula_for_node(temp.id, child_ids)
            id_to_node[temp.id].formula = formula

        # Build edges: child -> parent
        edges: list[Edge] = []
        for temp in temp_nodes:
            for child in temp.children:
                edges.append(Edge(source=child.id, target=temp.id))

        graph = Graph(nodes=list(id_to_node.values()), edges=edges)

        # Final validation: all constraints should hold, otherwise raise
        # 1) Node count
        if not (self.min_nodes <= len(graph.nodes) <= self.max_nodes):
            raise ValueError("Generated graph violates node count constraints.")

        # 2) Depth (in terms of longest leaf->root path)
        actual_depth = max(n.depth for n in temp_nodes)
        if not (self.min_depth <= actual_depth <= self.max_depth):
            raise ValueError("Generated graph violates depth constraints.")

        # 3) Branching factors for internal nodes (number of children)
        for temp in temp_nodes:
            child_count = len(temp.children)
            if child_count == 0:
                continue
            if not (self.min_branching_factor <= child_count <= self.max_branching_factor):
                raise ValueError("Generated graph violates branching factor constraints.")

        return graph

