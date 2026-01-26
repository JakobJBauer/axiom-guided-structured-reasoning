import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings
import graphviz
from .formulas import Formula


def _formula_to_string(formula: Formula) -> str:
    """Convert a formula to a readable string representation."""
    if formula is None:
        return ""
    
    formula_type = type(formula).__name__
    
    if formula_type == "Not":
        arg = formula.key_or_formula
        if isinstance(arg, Formula):
            arg_str = _formula_to_string(arg)
        else:
            arg_str = str(arg)
        return f"NOT {arg_str}"
    
    elif formula_type == "And":
        args = []
        for kf in formula.keys_or_formulas:
            if isinstance(kf, Formula):
                args.append(_formula_to_string(kf))
            else:
                args.append(str(kf))
        return " AND ".join(f"({arg})" if " " in arg else arg for arg in args)
    
    elif formula_type == "Or":
        args = []
        for kf in formula.keys_or_formulas:
            if isinstance(kf, Formula):
                args.append(_formula_to_string(kf))
            else:
                args.append(str(kf))
        return " OR ".join(f"({arg})" if " " in arg else arg for arg in args)
    
    elif formula_type == "Xor":
        args = []
        for kf in formula.keys_or_formulas:
            if isinstance(kf, Formula):
                args.append(_formula_to_string(kf))
            else:
                args.append(str(kf))
        return " XOR ".join(f"({arg})" if " " in arg else arg for arg in args)
    
    elif formula_type == "Equal":
        return f"{formula.key} == {formula.value!r}"
    
    elif formula_type == "In":
        values_str = ", ".join(str(v) for v in formula.values)
        return f"{formula.key} IN [{values_str}]"
    
    else:
        return str(formula)


def visualize_graph(
    graph,
    output_path: Optional[str] = None,
    format: str = "png",
    layout: str = "hierarchical",
    show_values: bool = True,
    node_shape: str = "box",
    engine: str = "dot"
) -> Optional[str]:
    return _visualize_with_graphviz(
        graph, output_path, format, layout, show_values, node_shape, engine
    )


def _visualize_with_graphviz(
    graph,
    output_path: Optional[str],
    format: str,
    layout: str,
    show_values: bool,
    node_shape: str,
    engine: str
) -> Optional[str]:
    # Create a directed graph
    dot = graphviz.Digraph(format=format, engine=engine)
    dot.attr(rankdir='TB')  # Top to bottom
    dot.attr('node', shape=node_shape, style='rounded,filled', fillcolor='lightblue')
    dot.attr('edge', arrowsize='0.8')
    
    # Add nodes
    node_ids = {node.id for node in graph.nodes}
    for node in graph.nodes:
        name = node.label if node.label else node.id
        
        # Build label with value if present
        if show_values and node.value is not None:
            label = f"{name} : {node.value}"
        else:
            label = name
        
        # Add formula as small text beneath the node if present
        formula_str = ""
        if node.formula is not None:
            formula_str = _formula_to_string(node.formula)
        
        # Use HTML-like label for multi-line formatting with different font sizes
        if formula_str:
            # Escape special characters for Graphviz HTML labels
            formula_str_escaped = formula_str.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            label_escaped = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Graphviz HTML labels use angle brackets and support font size
            html_label = f'<{label_escaped}<BR ALIGN="LEFT"/><FONT POINT-SIZE="8">{formula_str_escaped}</FONT>>'
        else:
            html_label = label
        
        # Determine if it's a leaf node
        is_leaf = graph.is_leaf_node(node)
        
        # Style leaf nodes differently
        if is_leaf:
            dot.node(
                str(node.id),
                label=html_label,
                fillcolor='lightgreen',
                shape='ellipse'
            )
        else:
            dot.node(str(node.id), label=html_label)
    
    # Add edges
    for edge in graph.edges:
        # Verify both nodes exist
        if edge.source in node_ids and edge.target in node_ids:
            dot.edge(str(edge.source), str(edge.target))
    
    # Render the graph
    if output_path is None:
        output_path = "graph_visualization"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove extension if present, graphviz will add it
    output_stem = output_path.stem
    output_dir = output_path.parent
    
    filepath = dot.render(
        filename=output_stem,
        directory=str(output_dir),
        cleanup=True  # Remove intermediate files
    )
    return filepath


def visualize_graph_from_file(
    filepath: str,
    output_path: Optional[str] = None,
    format: str = "png",
    **kwargs
) -> Optional[str]:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from serializer import load_graph
    
    graph = load_graph(filepath)
    
    if output_path is None:
        input_path = Path(filepath)
        output_path = input_path.with_suffix(f'.{format}')
    
    return visualize_graph(graph, output_path, format, **kwargs)

