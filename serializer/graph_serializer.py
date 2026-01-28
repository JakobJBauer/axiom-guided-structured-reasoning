import json
from typing import Any, Dict, List

from graph import Graph, Node, Edge
from graph.formulas import Not, And, Or, Xor, Equal, In
from graph.formulas.formula import Formula


def _formula_to_json(formula: Formula) -> Dict[str, Any]:
    def arg_to_json(arg: Any) -> Any:
        if isinstance(arg, Formula):
            return _formula_to_json(arg)
        return arg

    if isinstance(formula, Not):
        return {
            "formula_type": "Not",
            "formula_args": [arg_to_json(formula.key_or_formula)],
        }
    elif isinstance(formula, And):
        return {
            "formula_type": "And",
            "formula_args": [arg_to_json(a) for a in formula.keys_or_formulas],
        }
    elif isinstance(formula, Or):
        return {
            "formula_type": "Or",
            "formula_args": [arg_to_json(a) for a in formula.keys_or_formulas],
        }
    elif isinstance(formula, Xor):
        return {
            "formula_type": "Xor",
            "formula_args": [arg_to_json(a) for a in formula.keys_or_formulas],
        }
    elif isinstance(formula, Equal):
        return {
            "formula_type": "Equal",
            "formula_args": [formula.key, formula.value],
        }
    elif isinstance(formula, In):
        return {
            "formula_type": "In",
            "formula_args": [formula.key, list(formula.values)],
        }
    else:
        # Fallback: store repr only (won't round-trip to the same object type)
        return {
            "formula_type": "Unknown",
            "formula_args": [repr(formula)],
        }


def _formula_from_json(formula_type: str, args: List[Any]) -> Formula:
    """
    Reconstruct a Formula object from the JSON representation produced by
    _formula_to_json.
    """

    def arg_from_json(arg: Any) -> Any:
        if isinstance(arg, dict) and "formula_type" in arg:
            return _formula_from_json(
                arg["formula_type"],
                arg.get("formula_args", []),
            )
        return arg

    processed_args: List[Any] = [arg_from_json(a) for a in args]

    if formula_type == "Not":
        if len(processed_args) != 1:
            raise ValueError(f"Not formula requires 1 argument, got {len(processed_args)}")
        return Not(processed_args[0])

    if formula_type == "And":
        return And(*processed_args)

    if formula_type == "Or":
        return Or(*processed_args)

    if formula_type == "Xor":
        return Xor(*processed_args)

    if formula_type == "Equal":
        if len(processed_args) != 2:
            raise ValueError(f"Equal formula requires 2 arguments, got {len(processed_args)}")
        return Equal(processed_args[0], processed_args[1])

    if formula_type == "In":
        if len(processed_args) < 2:
            raise ValueError(f"In formula requires at least 2 arguments, got {len(processed_args)}")
        key = processed_args[0]
        values = processed_args[1]
        # Values may come as a list or single value
        if not isinstance(values, list):
            values = [values]
        return In(key, values)

    raise ValueError(f"Unknown formula type: {formula_type}")


def save_graph(graph: Graph, filepath: str) -> None:
    nodes_data = []
    for node in graph.nodes:
        node_entry: Dict[str, Any] = {
            "id": node.id,
            "label": node.label,
            "value": node.value,
        }
        if node.formula is not None and isinstance(node.formula, Formula):
            fjson = _formula_to_json(node.formula)
            node_entry["formula_type"] = fjson.get("formula_type")
            node_entry["formula_args"] = fjson.get("formula_args", [])
        else:
            node_entry["formula_type"] = None
            node_entry["formula_args"] = []
        nodes_data.append(node_entry)

    edges_data = [{"source": e.source, "target": e.target} for e in graph.edges]

    data = {"nodes": nodes_data, "edges": edges_data}

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(filepath: str) -> Graph:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes: List[Node] = []
    id_to_node: Dict[str, Node] = {}

    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        label = node_data.get("label", node_id)
        value = node_data.get("value")
        formula_type = node_data.get("formula_type")
        formula_args = node_data.get("formula_args", [])

        formula = None
        if formula_type:
            formula = _formula_from_json(formula_type, formula_args)

        node = Node(node_id, label=label, value=value, formula=formula)
        nodes.append(node)
        id_to_node[node_id] = node

    edges: List[Edge] = []
    for edge_data in data.get("edges", []):
        source = edge_data["source"]
        target = edge_data["target"]
        edges.append(Edge(source, target))

    return Graph(nodes, edges)