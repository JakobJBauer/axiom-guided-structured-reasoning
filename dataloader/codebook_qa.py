from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Literal

from graph.graph import Graph
from codebooks.generator import ReasoningTreeGenerator

from .leaf_values import compute_leaf_values_for_leaf_ids, load_leaf_specs


@dataclass
class CodebookSample:
    story: str
    story_row: Mapping[str, Any]
    codebook_text: str
    graph: Graph
    sink_id: str
    question: str
    reasoning_graph: Graph
    leaf_values: Dict[str, bool]
    answer: bool


SplitName = Literal["train", "test", "eval"]


@dataclass(frozen=True)
class GraphDifficultyConfig:
    """
    Configuration for how "big/hard" a generated graph/codebook should be.
    These map directly onto ReasoningTreeGenerator parameters.
    """
    goal_depth: int
    min_branching_factor: int = 1
    max_branching_factor: int = 2
    branch_density_factor: float = 0.8
    randomness_factor: float = 0.05
    max_leaf_nodes: int | None = 5


class CodebookQADataset:
    """
    Dataloader for CodebookQA:
      - always pulls SimpleStories from Hugging Face
      - creates train/test/eval splits locally
      - generates a reasoning graph on-the-fly
      - generates a codebook from the graph
      - generates a question from the graph's (single) sink node
      - populates leaf node values from SimpleStories row metadata
      - runs auto-inference to get the gold answer
    """

    def __init__(
        self,
        split: SplitName = "train",
        train_fraction: float = 0.90,
        test_fraction: float = 0.09,
        eval_fraction: float = 0.01,
        difficulties: Optional[Sequence[tuple[GraphDifficultyConfig, float]]] = None,
        seed: Optional[int] = 42,
        story_key: str = "story",
    ) -> None:
        """
        Args:
            split: One of {"train","test","eval"} selecting which local split to serve.
            train_fraction/test_fraction/eval_fraction: Fractions for local split.
                SimpleStories only has train+test upstream, so we concatenate them and
                re-split locally. Defaults: 0.90 / 0.09 / 0.01.
            difficulties: Weighted distribution over graph-generation configs.
                If None, a reasonable default distribution is used.
            seed: RNG seed controlling both story split shuffling and per-sample graph RNG.
            story_key: Key in the HF row containing story text.
        """
        if split not in {"train", "test", "eval"}: raise ValueError(f"split must be one of train/test/eval, got: {split!r}")
        self._split: SplitName = split
        self._seed = seed
        self._rng = random.Random(self._seed)
        self._story_key = story_key

        # Validate fractions
        if any(x < 0 for x in (train_fraction, test_fraction, eval_fraction)): raise ValueError("Split fractions must be non-negative")
        total = train_fraction + test_fraction + eval_fraction
        if abs(total - 1.0) > 1e-9: raise ValueError(f"Split fractions must sum to 1.0, got {total}")
        self._train_fraction = train_fraction
        self._test_fraction = test_fraction
        self._eval_fraction = eval_fraction

        # Default difficulty distribution: small/medium/large with light randomness
        if difficulties is None:
            difficulties = [
                (GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=6, randomness_factor=0.00), 0.45),
                (GraphDifficultyConfig(goal_depth=3, max_leaf_nodes=8, randomness_factor=0.05), 0.40),
                (GraphDifficultyConfig(goal_depth=4, max_leaf_nodes=10, randomness_factor=0.10), 0.15),
            ]
        self._difficulties = list(difficulties)
        if not self._difficulties: raise ValueError("difficulties must be non-empty")
        if any(w <= 0 for _, w in self._difficulties): raise ValueError("All difficulty weights must be > 0")
        if sum(w for _, w in self._difficulties) - 1.0 > 1e-9: raise ValueError("Difficulty weights must sum to 1.0")

        self._stories = _load_simplestories_concat_shared()
        self._indices_by_split = self._make_split_indices(len(self._stories))

        self._leaf_specs = load_leaf_specs()

    def __len__(self) -> int:
        return len(self._indices_by_split[self._split])

    def story_row_index(self, idx: int) -> int:
        """Index into the concatenated SimpleStories table for split sample ``idx``."""
        return self._indices_by_split[self._split][idx]

    def __getitem__(self, idx: int) -> CodebookSample:
        indices = self._indices_by_split[self._split]
        story_idx = indices[idx]
        story_row = self._stories[story_idx]
        story_text = story_row[self._story_key]

        graph = self._generate_graph_for_sample(sample_idx=idx)

        codebook_text = graph.generate_codebok_representation()
        sink_id = graph.get_single_sink_node().id
        question = graph.generate_question()

        # Populate leaf values from story features
        leaf_ids = [n.id for n in graph.get_leaf_nodes()]
        leaf_values = compute_leaf_values_for_leaf_ids(
            row=story_row,
            leaf_ids=leaf_ids,
            leaf_specs=self._leaf_specs,
        )

        # Run auto-inference on a copy so we don't mutate the graph instance returned
        reasoning_graph = graph.copy()
        for node in reasoning_graph.get_leaf_nodes():
            node.set_value(leaf_values[node.id])
        reasoning_graph.auto_infer_values()

        # Gold answer is the inferred sink value
        inferred_sink = reasoning_graph.get_node_by_id(sink_id)
        if inferred_sink is None or inferred_sink.value is None:
            raise RuntimeError("Failed to infer sink node value")
        answer = bool(inferred_sink.value)

        return CodebookSample(
            story=story_text,
            story_row=story_row,
            codebook_text=codebook_text,
            graph=graph,
            sink_id=sink_id,
            question=question,
            reasoning_graph=reasoning_graph,
            leaf_values=leaf_values,
            answer=answer,
        )

    def sample(self) -> CodebookSample:
        """Random sample from the chosen split."""
        return self[self._rng.randrange(len(self))]

    def _make_split_indices(self, n: int) -> dict[SplitName, list[int]]:
        indices = list(range(n))
        self._rng.shuffle(indices)

        n_train = int(self._train_fraction * n)
        n_test = int(self._test_fraction * n)
        n_eval = n - n_train - n_test

        train_idx = indices[:n_train]
        test_idx = indices[n_train : n_train + n_test]
        eval_idx = indices[n_train + n_test :]

        return {"train": train_idx, "test": test_idx, "eval": eval_idx}

    def _sample_difficulty(self, rng: random.Random) -> GraphDifficultyConfig:
        configs = [c for c, _w in self._difficulties]
        weights = [w for _c, w in self._difficulties]
        return rng.choices(configs, weights=weights, k=1)[0]

    def _generate_graph_for_sample(self, sample_idx: int) -> Graph:
        # Distinct stream per split so graphs don't overlap across splits
        split_offset = {"train": 10_000_000, "test": 20_000_000, "eval": 30_000_000}[self._split]
        rng = random.Random(self._seed + split_offset + sample_idx)
        diff = self._sample_difficulty(rng)

        generator = ReasoningTreeGenerator(
            goal_depth=diff.goal_depth,
            seed=self._seed + split_offset + sample_idx,
            min_branching_factor=diff.min_branching_factor,
            max_branching_factor=diff.max_branching_factor,
            branch_density_factor=diff.branch_density_factor,
            randomness_factor=diff.randomness_factor,
            max_leaf_nodes=diff.max_leaf_nodes,
        )
        graph = generator.generate()

        # Sanity: we require a single sink node for consistent question generation
        _ = graph.get_single_sink_node()
        return graph


@lru_cache(maxsize=1)
def _load_simplestories_concat_shared():
    from datasets import load_dataset, concatenate_datasets

    def load_split(name: str):
        return load_dataset(
            "SimpleStories/SimpleStories",
            split=name,
        )

    train_ds = load_split("train")
    test_ds = load_split("test")
    return concatenate_datasets([train_ds, test_ds])
