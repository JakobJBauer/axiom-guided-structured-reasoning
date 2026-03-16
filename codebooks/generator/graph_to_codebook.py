from graph.graph import Graph
from graph.formulas import Not, And, Or, Xor, Equal, In


def _format_leaf_definition(node_id: str, description: str) -> str:
    """
    Turn a leaf node into a simple, one-line definition.

    Example:
    [TOPIC-HIDDEN-TREASURES]
    A story is [TOPIC-HIDDEN-TREASURES] if the story is about hidden treasures.
    """
    label = node_id.upper()
    # Ensure description starts lowercased and attaches to "the story ..."
    desc = description.strip()
    if desc.lower().startswith("the story"):
        clause = desc[0].lower() + desc[1:]
    else:
        clause = f"the story {desc[0].lower()}{desc[1:]}" if desc else "satisfies this property"
    return f"A story is [{label}] if {clause}."


def _format_compound_definition(node_id: str, formula, graph: Graph) -> list[str]:
    """
    Turn a non-leaf node with a formula into one or more lines of definition
    using cb-1 style templates.
    """
    label = node_id.upper()

    def fmt_child(child_id: str) -> str:
        return f"[{child_id.upper()}]"

    # NOT
    if isinstance(formula, Not):
        # Not only supports a single key or nested formula; we assume a single child id
        key = formula.key_or_formula
        if isinstance(key, str):
            return [
                f"A story is [{label}] if it is not {fmt_child(key)}."
            ]

    # EQUAL / IN (unary predicates)
    if isinstance(formula, (Equal, In)):
        keys = formula.get_required_keys()
        if len(keys) == 1:
            return [
                f"A story is [{label}] if it is {fmt_child(keys[0])}."
            ]

    # AND
    if isinstance(formula, And):
        keys = formula.get_required_keys()
        lines = [
            f"A story is [{label}] if all of the following are true:",
        ]
        for k in keys:
            lines.append(f"- The story is {fmt_child(k)}")
        return lines

    # OR
    if isinstance(formula, Or):
        keys = formula.get_required_keys()
        lines = [
            f"A story is [{label}] if any of the following is true:",
        ]
        for k in keys:
            lines.append(f"- The story is {fmt_child(k)}")
        return lines

    # XOR
    if isinstance(formula, Xor):
        keys = formula.get_required_keys()
        lines = [
            f"A story is [{label}] if exactly one of the following is true:",
        ]
        for k in keys:
            lines.append(f"- The story is {fmt_child(k)}")
        return lines

    # Fallback: describe in terms of required keys, regardless of formula type
    keys = getattr(formula, "get_required_keys", lambda: [])()
    if keys:
        joined = ", ".join(fmt_child(k) for k in keys)
        return [f"A story is [{label}] if it depends on {joined}."]
    else:
        return [f"A story is [{label}] if this condition holds (see formula definition)."]


def graph_to_codebook(graph: Graph) -> str:
    """
    Convert a reasoning graph into a codebook-style textual specification.

    The output roughly follows the cb-1.txt style:
    - One block per node.
    - Leaf nodes get simple, direct definitions based on their labels.
    - Non-leaf nodes get AND/OR/NOT/XOR-style templates referencing child labels.
    """
    lines: list[str] = []

    # Use topological order so that base labels (leaves) appear before composites.
    ordered_nodes = graph.topological_sort()
    leaf_ids = {n.id for n in graph.get_leaf_nodes()}

    for node in ordered_nodes:
        node_id = node.id
        label_line = f"[{node_id.upper()}]"
        lines.append(label_line)

        if node_id in leaf_ids or node.formula is None:
            # Leaf: use the description stored in node.label
            description = node.label or f"the story satisfies {label_line}"
            lines.append(_format_leaf_definition(node_id, description))
        else:
            # Non-leaf: use the formula structure
            for def_line in _format_compound_definition(node_id, node.formula, graph):
                lines.append(def_line)

        # Blank line between entries
        lines.append("")

    # Drop trailing blank line for cleanliness
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)