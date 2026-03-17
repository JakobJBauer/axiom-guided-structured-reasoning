from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEAF_SPECS_PATH = REPO_ROOT / "codebooks" / "generator" / "proposed_leaf_nodes.json"


def load_leaf_specs(path: Path | None = None) -> Dict[str, Dict[str, Any]]:
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

        raise ValueError(f"Unknown operator: {operator}")

    raise ValueError(f"Unknown leaf type: {leaf_type}")


def compute_leaf_values_for_leaf_ids(
    row: Mapping[str, Any],
    leaf_ids: Iterable[str],
    leaf_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, bool]:
    """
    Compute boolean values for a set of leaf node IDs using proposed_leaf_nodes.json.

    This is the preferred path when leaf node IDs are the canonical spec IDs
    (i.e. not obfuscated).
    """
    if leaf_specs is None:
        leaf_specs = load_leaf_specs()

    out: Dict[str, bool] = {}
    for leaf_id in leaf_ids:
        spec = leaf_specs.get(leaf_id)
        if spec is None:
            raise KeyError(f"No leaf spec found for leaf id '{leaf_id}'")
        out[leaf_id] = _eval_leaf_spec(spec, row)
    return out
