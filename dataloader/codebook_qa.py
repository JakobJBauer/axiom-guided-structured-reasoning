from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from graph.graph import Graph
from serializer import load_graph

from .leaf_values import compute_leaf_values_for_graph, load_leaf_specs


@dataclass
class CodebookSample:
    story: str
    story_row: Mapping[str, Any]
    codebook_path: Path
    codebook_text: str
    graph: Graph
    sink_id: str
    question: str
    reasoning_graph: Graph
    leaf_values: Dict[str, bool]


class CodebookQADataset:
    """
    Simple dataloader for:
      - sampling a simplestory
      - pairing it with one codebook + graph
      - choosing a sink node as the question target
      - populating leaf nodes from the story metadata
      - running graph auto-inference to produce a gold reasoning tree.
    """

    def __init__(
        self,
        stories: Optional[Sequence[Mapping[str, Any]]] = None,
        stories_story_key: str = "story",
        codebooks_root: Path | str = Path("codebooks") / "final_selection",
        simplestories_split: str = "train",
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            stories: Sequence of simplestory rows (e.g. list of dicts, or
                     something indexable like a pandas DataFrame via .iloc).
                     If None, the SimpleStories dataset from Hugging Face
                     (\"SimpleStories/SimpleStories\") is loaded by default.
            stories_story_key: Key in each row mapping to the story text.
            codebooks_root: Root directory containing:
                - graphs/  (JSON graphs)
                - codebooks/  (text codebooks)
            seed: Optional RNG seed for reproducibility.
        """
        self._simplestories_split = simplestories_split

        if seed is not None:
            random.seed(seed)

        if stories is None:
            stories = self._load_default_simplestories()

        self._stories = stories
        self._story_key = stories_story_key

        self._root = Path(codebooks_root)
        self._graphs_dir = self._root / "graphs"
        self._codebooks_dir = self._root / "codebooks"

        if not self._graphs_dir.exists():
            raise FileNotFoundError(f"Graphs directory not found: {self._graphs_dir}")
        if not self._codebooks_dir.exists():
            raise FileNotFoundError(
                f"Codebooks directory not found: {self._codebooks_dir}"
            )

        # Pre-index graphs by base name and style presence
        self._base_to_graphs: Dict[str, Dict[str, Path]] = {}
        for path in sorted(self._graphs_dir.glob("*.json")):
            stem = path.stem  # e.g. cb-003-small-easy-clear, cb-003-small-easy-obfc
            if stem.endswith("-clear"):
                base = stem[: -len("-clear")]
                variant = "clear"
            elif stem.endswith("-obfc"):
                base = stem[: -len("-obfc")]
                variant = "obfc"
            else:
                base = stem
                variant = "base"

            self._base_to_graphs.setdefault(base, {})[variant] = path

        # Pre-build list of (codebook_path, base_name, is_obfuscated) entries
        self._codebook_entries: list[Tuple[Path, str, bool]] = []
        for path in sorted(self._codebooks_dir.glob("*.txt")):
            stem = path.stem
            base = self._infer_base_from_stem(stem)
            if base in self._base_to_graphs:
                is_obfuscated = "-obfc" in stem
                self._codebook_entries.append((path, base, is_obfuscated))

        if not self._codebook_entries:
            raise RuntimeError(
                f"No codebooks in {self._codebooks_dir} matched any base name "
                f"from graphs in {self._graphs_dir}"
            )

        self._leaf_specs = load_leaf_specs()

    def __len__(self) -> int:
        return len(self._stories) * len(self._codebook_entries)

    # --- Public API ---------------------------------------------------------

    def sample(self) -> CodebookSample:
        """Sample a single (story, codebook, question, reasoning tree) datapoint."""
        story_row = self._sample_story_row()
        story_text = story_row[self._story_key]

        codebook_path, base_name, is_obfuscated = random.choice(self._codebook_entries)
        codebook_text = codebook_path.read_text(encoding="utf-8")

        graph, sink_id = self._sample_graph_and_sink(base_name, is_obfuscated)

        # Populate leaf values from story features
        leaf_values = self._populate_leaf_values(story_row, graph)

        # Run auto-inference on a copy so we don't mutate cached graphs
        reasoning_graph = graph.copy()
        for node in reasoning_graph.get_leaf_nodes():
            if node.id in leaf_values:
                node.set_value(leaf_values[node.id])
        reasoning_graph.auto_infer_values()

        # Extract the reasoning subgraph that supports the sink node
        reasoning_subgraph = self._extract_reasoning_subgraph(reasoning_graph, sink_id)

        # Simple natural-language question
        sink_node = graph.get_node_by_id(sink_id)
        question = f"Is the story {sink_node.label}?"

        return CodebookSample(
            story=story_text,
            story_row=story_row,
            codebook_path=codebook_path,
            codebook_text=codebook_text,
            graph=graph,
            sink_id=sink_id,
            question=question,
            reasoning_graph=reasoning_subgraph,
            leaf_values=leaf_values,
        )

    # --- Internal helpers ---------------------------------------------------

    def _sample_story_row(self) -> Mapping[str, Any]:
        idx = random.randrange(len(self._stories))

        row = self._stories[idx]
        # Support pandas DataFrame via .iloc as well as plain sequences of dicts
        if hasattr(self._stories, "iloc"):
            row = self._stories.iloc[idx]

        if self._story_key not in row:
            raise KeyError(f"Story row missing key '{self._story_key}': {row}")
        return row

    def _load_default_simplestories(self) -> Sequence[Mapping[str, Any]]:
        """
        Load the SimpleStories dataset from Hugging Face as the default source
        of stories when none are provided explicitly.

        This requires the `datasets` library and internet access:

            pip install datasets

        The dataset ID is \"SimpleStories/SimpleStories\" and we map our
        requested split onto the underlying dataset's available splits.
        Currently, SimpleStories exposes \"train\" and \"test\"; we support:
        - \"train\" -> \"train\"
        - \"test\"  -> \"test\"
        - \"validation\" / \"val\" / \"eval\" / \"seval\" -> \"test\"
        """
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "To use CodebookQADataset without passing `stories`, install the "
                "`datasets` library (pip install datasets), or pass a sequence of "
                "story rows explicitly."
            ) from exc

        requested = (self._simplestories_split or "train").lower()
        if requested in {"train"}:
            hf_split = "train"
        elif requested in {"test"}:
            hf_split = "test"
        elif requested in {"validation", "val", "eval", "seval"}:
            hf_split = "test"
        else:
            raise ValueError(
                f"Unsupported SimpleStories split '{self._simplestories_split}'. "
                "Expected one of: train, test, validation, val, eval, seval."
            )

        ds = load_dataset("SimpleStories/SimpleStories", split=hf_split)
        return ds

    def _infer_base_from_stem(self, stem: str) -> str:
        """
        Infer a base name for a codebook stem by matching it against known base names
        from graphs. We pick the longest base that is a prefix of the stem.
        """
        candidates = [
            base for base in self._base_to_graphs.keys() if stem.startswith(base)
        ]
        if not candidates:
            return stem
        # Longest prefix to disambiguate e.g. ...-allf vs shorter bases
        return max(candidates, key=len)

    def _sample_graph_and_sink(self, base_name: str, want_obfuscated: bool) -> Tuple[Graph, str]:
        variants = self._base_to_graphs[base_name]
        # Choose graph variant consistent with the sampled codebook:
        # - if the codebook text is obfuscated, we prefer the obfuscated graph
        # - if the codebook text is clear, we prefer the clear (or base) graph.
        clear_graph_data = None

        if want_obfuscated and "obfc" in variants:
            graph_path = variants["obfc"]
            clear_graph_path = variants.get("clear")
            if clear_graph_path is not None:
                with clear_graph_path.open("r", encoding="utf-8") as f:
                    clear_graph_data = json.load(f)
        elif not want_obfuscated and "clear" in variants:
            graph_path = variants["clear"]
        elif not want_obfuscated and "base" in variants:
            graph_path = variants["base"]
        else:
            # Fallbacks: if our preferred variant is missing, use whatever exists
            graph_path = variants.get("obfc") or variants.get("clear") or variants.get("base")
            if graph_path is None:
                raise RuntimeError(f"No graph variants found for base '{base_name}'")
            if graph_path == variants.get("obfc"):
                clear_graph_path = variants.get("clear")
                if clear_graph_path is not None:
                    with clear_graph_path.open("r", encoding="utf-8") as f:
                        clear_graph_data = json.load(f)

        graph = load_graph(str(graph_path))

        # Choose a sink node (no outgoing edges) as the question target
        sinks = [
            node
            for node in graph.get_nodes()
            if not graph.get_outgoing_edges(node)
        ]
        if not sinks:
            raise RuntimeError(f"No sink nodes found in graph {graph_path}")
        sink_node = random.choice(sinks)

        # Attach clear_graph_data to the instance for leaf population if needed
        # (graph IDs may be obfuscated, but labels align with clear graphs).
        graph._clear_graph_data = clear_graph_data  # type: ignore[attr-defined]

        return graph, sink_node.id

    def _populate_leaf_values(
        self,
        story_row: Mapping[str, Any],
        graph: Graph,
    ) -> Dict[str, bool]:
        # Prepare clear graph dict if we have it
        clear_graph_data = getattr(graph, "_clear_graph_data", None)

        # For compute_leaf_values_for_graph we pass the raw JSON-like graph dict,
        # not the Graph object, so reconstruct that minimal structure.
        graph_dict = {
            "nodes": [
                {
                    "id": node.id,
                    "label": node.label,
                    "formula_type": None if node.formula is None else "Formula",
                    "formula_args": [],
                }
                for node in graph.get_nodes()
            ],
            "edges": [
                {"source": e.source, "target": e.target} for e in graph.get_edges()
            ],
        }

        return compute_leaf_values_for_graph(
            row=story_row,
            graph=graph_dict,
            leaf_specs=self._leaf_specs,
            clear_graph=clear_graph_data,
        )

    def _extract_reasoning_subgraph(self, graph: Graph, sink_id: str) -> Graph:
        """
        Return the induced subgraph containing the sink node and all its ancestors.
        """
        sink = graph.get_node_by_id(sink_id)
        if sink is None:
            raise ValueError(f"Sink node '{sink_id}' not found in graph")

        # Backward DFS from sink through incoming edges
        visited: set[str] = set()
        stack = [sink]
        while stack:
            node = stack.pop()
            if node.id in visited:
                continue
            visited.add(node.id)
            for parent in graph.get_incoming_nodes(node):
                stack.append(parent)

        # Build subgraph
        nodes = [graph.get_node_by_id(nid).copy() for nid in visited]  # type: ignore[arg-type]
        id_set = set(visited)
        edges = [
            e for e in graph.get_edges() if e.source in id_set and e.target in id_set
        ]
        return Graph(nodes, edges)

