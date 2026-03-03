from pathlib import Path

import json

import sys

from pathlib import Path

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.leaf_values import (  # noqa: E402
    check_all_final_graph_leaves_have_specs,
    compute_leaf_values_for_graph,
    load_leaf_specs,
)
from dataloader.codebook_qa import CodebookQADataset  # noqa: E402


def test_all_final_graph_leaves_have_specs():
    """All leaf nodes in final_selection/graphs must be mappable from dataset features."""
    graphs_dir = (
        REPO_ROOT
        / "codebooks"
        / "2026-01-28"
        / "final_selection"
        / "graphs"
    )
    leaf_specs = load_leaf_specs()

    missing_by_graph = check_all_final_graph_leaves_have_specs(
        graphs_dir=graphs_dir,
        leaf_specs=leaf_specs,
    )

    assert (
        not missing_by_graph
    ), f"Some graphs reference leaf ids without specs: {missing_by_graph}"


def test_compute_leaf_values_for_graph_smoke():
    """
    Smoke test: we can compute leaf values for at least one graph
    using a dummy row with the right feature keys.
    """
    graphs_dir = (
        REPO_ROOT
        / "codebooks"
        / "2026-01-28"
        / "final_selection"
        / "graphs"
    )
    # Prefer a clear graph for the basic smoke test
    example_graph_path = next(iter(sorted(graphs_dir.glob("*-clear.json"))))
    with example_graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    leaf_specs = load_leaf_specs()

    # Build a minimal dummy row with all features that any leaf spec might touch
    all_features = {spec["feature"] for spec in leaf_specs.values() if "feature" in spec}
    dummy_row = {feature: None for feature in all_features}

    # This should not raise, and should return a dict of booleans
    values = compute_leaf_values_for_graph(
        row=dummy_row,
        graph=graph,
        leaf_specs=leaf_specs,
    )

    assert isinstance(values, dict)
    for k, v in values.items():
        assert isinstance(k, str)
        assert isinstance(v, bool)


def test_compute_leaf_values_for_obfuscated_graph_smoke():
    """Smoke test: obfuscated graphs can be populated using the paired clear graph."""
    graphs_dir = (
        REPO_ROOT
        / "codebooks"
        / "2026-01-28"
        / "final_selection"
        / "graphs"
    )
    obf_path = next(iter(sorted(graphs_dir.glob("*-obfc.json"))))
    clear_path = obf_path.with_name(obf_path.name.replace("-obfc.json", "-clear.json"))
    assert clear_path.exists(), f"Expected paired clear graph for {obf_path.name}"

    with obf_path.open("r", encoding="utf-8") as f:
        obf_graph = json.load(f)
    with clear_path.open("r", encoding="utf-8") as f:
        clear_graph = json.load(f)

    leaf_specs = load_leaf_specs()
    all_features = {spec["feature"] for spec in leaf_specs.values() if "feature" in spec}
    dummy_row = {feature: None for feature in all_features}

    values = compute_leaf_values_for_graph(
        row=dummy_row,
        graph=obf_graph,
        leaf_specs=leaf_specs,
        clear_graph=clear_graph,
    )

    assert isinstance(values, dict)
    for k, v in values.items():
        assert isinstance(k, str)
        assert isinstance(v, bool)


def test_codebook_qa_dataset_sample_smoke():
    """Smoke test: sampling from CodebookQADataset returns a coherent sample."""
    # Minimal simplestory row; leaf population only requires feature keys, not text content.
    stories = [
        {
            "story": "This is a dummy story used only for testing the dataloader.",
            # All other features are optional for the smoke test; they default to None
            # inside compute_leaf_values_for_graph and simply yield False leaves.
        }
    ]

    dataset = CodebookQADataset(
        stories=stories,
        stories_story_key="story",
        codebooks_root=REPO_ROOT / "codebooks" / "final_selection",
        seed=123,
    )

    sample = dataset.sample()

    # Basic structure checks
    assert isinstance(sample.story, str) and sample.story
    assert isinstance(sample.codebook_text, str) and sample.codebook_text
    assert sample.graph is not None
    assert sample.sink_id in {n.id for n in sample.graph.get_nodes()}
    assert isinstance(sample.question, str) and "Is the story" in sample.question

    # Reasoning graph should contain the sink and have values set on leaves
    sink_in_reasoning = any(
        n.id == sample.sink_id for n in sample.reasoning_graph.get_nodes()
    )
    assert sink_in_reasoning
    assert sample.reasoning_graph.leaf_values_set()

    # --- Check that leaf auto-population matches a fresh call to compute_leaf_values_for_graph ---
    # Reconstruct a minimal JSON-like graph dict from the sampled full graph
    graph_dict = {
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "formula_type": None if n.formula is None else "Formula",
                "formula_args": [],
            }
            for n in sample.graph.get_nodes()
        ],
        "edges": [
            {"source": e.source, "target": e.target}
            for e in sample.graph.get_edges()
        ],
    }

    clear_graph_data = getattr(sample.graph, "_clear_graph_data", None)
    leaf_specs = load_leaf_specs()

    recomputed_leaf_values = compute_leaf_values_for_graph(
        row=sample.story_row,
        graph=graph_dict,
        leaf_specs=leaf_specs,
        clear_graph=clear_graph_data,
    )

    # On the reasoning graph, each leaf node's value should match the recomputed one
    for leaf in sample.reasoning_graph.get_leaf_nodes():
        assert leaf.id in recomputed_leaf_values
        assert leaf.value == recomputed_leaf_values[leaf.id]


