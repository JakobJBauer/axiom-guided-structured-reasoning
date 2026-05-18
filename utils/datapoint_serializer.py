from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig, SplitName
from graph.graph import Graph
from serializer import load_graph, save_graph

DEFAULT_DATAPOINTS_FILENAME = "datapoints.jsonl"
GRAPHS_SUBDIR = "graphs"
PromptStyle = Literal["full", "abbr", "none"]


def _gold_attr_values_from_graph(graph: Graph) -> dict[str, bool]:
    return {
        node.id.upper(): bool(node.value)
        for node in graph.get_nodes()
        if node.value is not None
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _story_features_from_row(story_row: Any, *, story_key: str = "story") -> dict[str, Any]:
    row = dict(story_row)
    row.pop(story_key, None)
    return _json_safe(row)


@dataclass
class SerializedCodebookDatapoint:
    story: str
    codebook_text: str
    codebook_path: str
    question: str
    sink_id: str
    story_features: dict[str, Any]
    gold_answer: bool
    gold_attr_values: dict[str, bool]
    simplestories_row_id: int
    prompt_strategy: PromptStyle
    answer_model_id: str | None = None
    answer_model_response: str | None = None
    answer_model_answer: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SerializedCodebookDatapoint":
        return cls(
            story=data["story"],
            codebook_text=data["codebook_text"],
            codebook_path=data["codebook_path"],
            question=data["question"],
            sink_id=data["sink_id"],
            story_features=data["story_features"],
            gold_answer=bool(data["gold_answer"]),
            gold_attr_values={str(k): bool(v) for k, v in data["gold_attr_values"].items()},
            simplestories_row_id=int(data["simplestories_row_id"]),
            prompt_strategy=data["prompt_strategy"],
            answer_model_id=data.get("answer_model_id"),
            answer_model_response=data.get("answer_model_response"),
            answer_model_answer=(
                None if data.get("answer_model_answer") is None else bool(data["answer_model_answer"])
            ),
        )


class CodebookDatapointSerializer:
    """
    Serialize CodebookQA datapoints to JSONL plus per-example graph JSON files.

    Each record stores prompt-facing fields and supervision metadata. Graphs are
    written to ``<output_path>/graphs/``; the relative path is stored as
    ``codebook_path`` in the JSONL record.
    """

    def __init__(
        self,
        output_path: str | Path,
        difficulty: GraphDifficultyConfig,
        *,
        split: SplitName = "train",
        seed: int = 42,
        prompt_strategy: PromptStyle = "none",
        datapoints_filename: str = DEFAULT_DATAPOINTS_FILENAME,
    ) -> None:
        self.output_path = Path(output_path)
        self.difficulty = difficulty
        self.split = split
        self.seed = seed
        self.prompt_strategy = prompt_strategy
        self.datapoints_filename = datapoints_filename

    @property
    def graphs_dir(self) -> Path:
        return self.output_path / GRAPHS_SUBDIR

    @property
    def datapoints_path(self) -> Path:
        return self.output_path / self.datapoints_filename

    def _make_dataset(self) -> CodebookQADataset:
        return CodebookQADataset(
            split=self.split,
            difficulties=[(self.difficulty, 1.0)],
            seed=self.seed,
        )

    def _serialize_one(
        self,
        dataset: CodebookQADataset,
        idx: int,
        *,
        graphs_dir: Path,
    ) -> SerializedCodebookDatapoint:
        sample = dataset[idx]
        graph_filename = f"{idx:06d}.json"
        graph_rel_path = f"{GRAPHS_SUBDIR}/{graph_filename}"
        graph_abs_path = graphs_dir / graph_filename

        save_graph(sample.reasoning_graph, str(graph_abs_path))

        return SerializedCodebookDatapoint(
            story=sample.story,
            codebook_text=sample.codebook_text,
            codebook_path=graph_rel_path,
            question=sample.question,
            sink_id=sample.sink_id,
            story_features=_story_features_from_row(sample.story_row),
            gold_answer=bool(sample.answer),
            gold_attr_values=_gold_attr_values_from_graph(sample.reasoning_graph),
            simplestories_row_id=dataset.story_row_index(idx),
            prompt_strategy=self.prompt_strategy,
            answer_model_id=None,
            answer_model_response=None,
            answer_model_answer=None,
        )

    def serialize(
        self,
        num_examples: int,
        *,
        indices: Sequence[int] | None = None,
    ) -> Path:
        """
        Serialize ``num_examples`` datapoints (or the explicit ``indices``) to disk.

        Returns the path to the JSONL file.
        """
        dataset = self._make_dataset()
        if indices is None:
            indices = range(num_examples)
        else:
            indices = list(indices)

        self.output_path.mkdir(parents=True, exist_ok=True)
        self.graphs_dir.mkdir(parents=True, exist_ok=True)

        with self.datapoints_path.open("w", encoding="utf-8") as f_out:
            for idx in indices:
                record = self._serialize_one(dataset, idx, graphs_dir=self.graphs_dir)
                f_out.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

        return self.datapoints_path

    @classmethod
    def load_datapoints(
        cls,
        output_path: str | Path,
        *,
        datapoints_filename: str = DEFAULT_DATAPOINTS_FILENAME,
    ) -> list[SerializedCodebookDatapoint]:
        path = Path(output_path) / datapoints_filename
        records: list[SerializedCodebookDatapoint] = []
        with path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                records.append(SerializedCodebookDatapoint.from_dict(json.loads(line)))
        return records

    @classmethod
    def iter_datapoints(
        cls,
        output_path: str | Path,
        *,
        datapoints_filename: str = DEFAULT_DATAPOINTS_FILENAME,
    ) -> Iterator[SerializedCodebookDatapoint]:
        path = Path(output_path) / datapoints_filename
        with path.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                yield SerializedCodebookDatapoint.from_dict(json.loads(line))

    @classmethod
    def load_graph(
        cls,
        output_path: str | Path,
        record: SerializedCodebookDatapoint,
    ) -> Graph:
        graph_path = Path(output_path) / record.codebook_path
        return load_graph(str(graph_path))
