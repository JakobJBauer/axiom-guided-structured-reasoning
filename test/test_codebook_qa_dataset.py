import sys
from pathlib import Path

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.leaf_values import (
    compute_leaf_values_for_leaf_ids,
    load_leaf_specs,
)
from codebooks.generator import ReasoningTreeGenerator


def test_compute_leaf_values_for_leaf_ids_smoke():
    """
    Smoke test: we can compute leaf values for a generated graph using a dummy row
    that contains all feature keys used by specs (values can be None).
    """
    leaf_specs = load_leaf_specs()
    all_features = {spec["feature"] for spec in leaf_specs.values() if "feature" in spec}
    dummy_row = {feature: None for feature in all_features}

    g = ReasoningTreeGenerator(goal_depth=2, max_leaf_nodes=6, seed=0).generate()
    leaf_ids = [n.id for n in g.get_leaf_nodes()]

    values = compute_leaf_values_for_leaf_ids(
        row=dummy_row,
        leaf_ids=leaf_ids,
        leaf_specs=leaf_specs,
    )

    assert isinstance(values, dict)
    assert set(values.keys()) == set(leaf_ids)
    for v in values.values():
        assert isinstance(v, bool)


