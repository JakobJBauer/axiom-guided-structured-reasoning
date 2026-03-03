from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEAF_SPECS_PATH = REPO_ROOT / "codebooks" / "generator" / "proposed_leaf_nodes.json"


def load_leaf_specs(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """
    Load leaf node specifications (id -> spec) from proposed_leaf_nodes.json.

    Each spec has at least:
    - id: str
    - type: 'categorical_match' | 'numeric_threshold'
    - feature: dataset feature name
    - value / threshold / operator: how to compute the boolean.
    """
    specs_path = path or DEFAULT_LEAF_SPECS_PATH
    with specs_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {item["id"]: item for item in data}


def _eval_leaf_spec(spec: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """Evaluate a single leaf spec against one simplestory row."""
    leaf_type = spec.get("type")
    feature = spec.get("feature")
    if feature is None:
        return False

    value = row.get(feature)

    if leaf_type == "categorical_match":
        target = spec.get("value")
        return value == target

    if leaf_type == "numeric_threshold":
        threshold = spec.get("threshold")
        operator = spec.get("operator")
        if value is None or threshold is None or operator is None:
            return False
        try:
            v = float(value)
        except (TypeError, ValueError):
            return False

        if operator == "<":
            return v < float(threshold)
        if operator == ">":
            return v > float(threshold)

        # Unknown operator → do not fire the leaf
        return False

    # Unknown type → treat as not satisfied
    return False


def _normalize_text(text: str) -> str:
    """Normalize free-form text for fuzzy matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_spec_by_label(
    label: str,
    leaf_specs: Mapping[str, Mapping[str, Any]],
    allowed_leaf_ids: set[str] | None = None,
) -> Mapping[str, Any] | None:
    """
    Best-effort resolution of a leaf spec for obfuscated graphs, based on the node label.

    This is only used when the graph's node id does not directly correspond to a spec id,
    e.g. obfuscated graphs where ids like 'attr-5' replace 'topic-magical-lands', but
    the labels still contain phrases like 'About magical lands'.
    """
    norm_label = _normalize_text(label)
    if not norm_label:
        return None

    # First, try exact / substring matches on the categorical 'value' field
    for spec_id, spec in leaf_specs.items():
        if allowed_leaf_ids is not None and spec_id not in allowed_leaf_ids:
            continue
        if spec.get("type") != "categorical_match":
            continue
        value = spec.get("value")
        if not isinstance(value, str):
            continue
        norm_value = _normalize_text(value)
        if not norm_value:
            continue
        # If the value phrase appears in the label (or vice versa), assume it's a match
        if norm_value in norm_label or norm_label in norm_value:
            return spec

    # As a fallback, try matching on the description text if present
    for spec_id, spec in leaf_specs.items():
        if allowed_leaf_ids is not None and spec_id not in allowed_leaf_ids:
            continue
        desc = spec.get("description")
        if not isinstance(desc, str):
            continue
        norm_desc = _normalize_text(desc)
        if not norm_desc:
            continue
        if norm_desc in norm_label or norm_label in norm_desc:
            return spec

    return None


def compute_leaf_values_for_graph(
    row: Mapping[str, Any],
    graph: Mapping[str, Any],
    leaf_specs: Mapping[str, Mapping[str, Any]] | None = None,
    clear_graph: Mapping[str, Any] | None = None,
) -> Dict[str, bool]:
    """
    Compute boolean values for all leaf nodes in a graph, using only dataset features.

    - row: one simplestories example (e.g. a pandas Series or plain dict)
    - graph: dict with a 'nodes' list (as in serialized Graph JSON)
    - leaf_specs: mapping from leaf id to spec; if None, loaded from proposed_leaf_nodes.json

    Returns: {leaf_id: bool}
    """
    if leaf_specs is None:
        leaf_specs = load_leaf_specs()

    values: Dict[str, bool] = {}
    allowed_leaf_ids: set[str] | None = None
    if clear_graph is not None:
        allowed_leaf_ids = {
            n.get("id")
            for n in clear_graph.get("nodes", [])
            if n.get("id") and n.get("formula_type") is None
        }

    for node in graph.get("nodes", []):
        node_id = node.get("id")
        formula_type = node.get("formula_type")

        # Only leaf nodes (formula_type is null / None) are directly tied to features
        if not node_id or formula_type is not None:
            continue

        spec = leaf_specs.get(node_id)
        if spec is None:
            # Obfuscated graphs often rename ids (e.g. 'attr-5') but keep informative labels.
            label = node.get("label") or ""
            spec = _find_spec_by_label(
                label,
                leaf_specs,
                allowed_leaf_ids=allowed_leaf_ids,
            )
            if spec is None:
                # If we still cannot find a spec, this leaf cannot be populated
                raise KeyError(
                    f"No leaf spec found for node id '{node_id}' with label '{label}'"
                )

        values[node_id] = _eval_leaf_spec(spec, row)

    return values


def check_all_final_graph_leaves_have_specs(
    graphs_dir: Path | None = None,
    leaf_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, set[str]]:
    """
    Utility: scan all graphs in final_selection/graphs and report any leaf ids
    that cannot be mapped via proposed_leaf_nodes.json.

    Returns a mapping {graph_name: {missing_leaf_ids}}; empty sets mean fully populate-able.
    """
    if graphs_dir is None:
        graphs_dir = REPO_ROOT / "codebooks" / "2026-01-28" / "final_selection" / "graphs"

    if leaf_specs is None:
        leaf_specs = load_leaf_specs()

    missing_by_graph: Dict[str, set[str]] = {}

    for json_file in sorted(graphs_dir.glob("*.json")):
        with json_file.open("r", encoding="utf-8") as f:
            graph = json.load(f)

        missing: set[str] = set()
        clear_graph = None
        if json_file.name.endswith("-obfc.json"):
            clear_candidate = json_file.with_name(
                json_file.name.replace("-obfc.json", "-clear.json")
            )
            if clear_candidate.exists():
                with clear_candidate.open("r", encoding="utf-8") as cf:
                    clear_graph = json.load(cf)

        for node in graph.get("nodes", []):
            node_id = node.get("id")
            formula_type = node.get("formula_type")
            if not node_id or formula_type is not None:
                continue
            if node_id in leaf_specs:
                continue

            # Try resolving obfuscated ids via label matching, optionally constrained
            # by the paired clear graph (if available).
            label = node.get("label") or ""
            allowed_leaf_ids = None
            if clear_graph is not None:
                allowed_leaf_ids = {
                    n.get("id")
                    for n in clear_graph.get("nodes", [])
                    if n.get("id") and n.get("formula_type") is None
                }
            spec = _find_spec_by_label(
                label,
                leaf_specs,
                allowed_leaf_ids=allowed_leaf_ids,
            )
            if spec is None:
                missing.add(node_id)

        if missing:
            missing_by_graph[json_file.name] = missing

    return missing_by_graph

