from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

from tqdm import tqdm

from dataloader.trl_adapters import build_task_prefix
from utils.answer_parsing import parse_final_boolean_answer
from utils.datapoint_serializer import (
    DEFAULT_DATAPOINTS_FILENAME,
    CodebookDatapointSerializer,
    SerializedCodebookDatapoint,
)
from utils.inference_backends import GenerationConfig, TransformersBatchBackend, VllmBatchBackend


DEFAULT_OUTPUT_FILENAME = "datapoints_inferred.jsonl"


def build_prompt_from_record(record: SerializedCodebookDatapoint) -> str:
    prefix = build_task_prefix(record.prompt_strategy)
    return prefix + (
        "Story:\n"
        f"{record.story}\n\n"
        "Codebook:\n"
        f"{record.codebook_text}\n\n"
        "Question:\n"
        f"{record.question}\n\n"
        "Assistant:\n"
    )


def _record_with_completion(
    record: SerializedCodebookDatapoint,
    *,
    model_id: str,
    completion: str,
) -> SerializedCodebookDatapoint:
    return SerializedCodebookDatapoint(
        story=record.story,
        codebook_text=record.codebook_text,
        codebook_path=record.codebook_path,
        question=record.question,
        sink_id=record.sink_id,
        story_features=record.story_features,
        gold_answer=record.gold_answer,
        gold_attr_values=record.gold_attr_values,
        simplestories_row_id=record.simplestories_row_id,
        prompt_strategy=record.prompt_strategy,
        answer_model_id=model_id,
        answer_model_response=completion,
        answer_model_answer=parse_final_boolean_answer(completion),
    )


class CodebookDatapointInferencer:
    """
    Run model inference on serialized CodebookQA datapoints and write a new JSONL
    file with ``answer_model_id``, ``answer_model_response``, and ``answer_model_answer``.

    Inference runs in batches; each batch is appended to the output file as soon as
    it finishes (so partial results survive if the job stops mid-run).
    """

    def __init__(
        self,
        base_dir: str | Path,
        model: str,
        *,
        input_filename: str = DEFAULT_DATAPOINTS_FILENAME,
        output_filename: str | None = None,
        use_vllm: bool = True,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
        batch_size: int = 8,
        do_sample: bool = True,
        show_progress: bool = True,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.model = model
        self.input_filename = input_filename
        self.output_filename = output_filename or DEFAULT_OUTPUT_FILENAME
        self.use_vllm = use_vllm
        self.batch_size = max(1, batch_size)
        self.show_progress = show_progress
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    @property
    def input_path(self) -> Path:
        return self.base_dir / self.input_filename

    @property
    def output_path(self) -> Path:
        return self.base_dir / self.output_filename

    def _iter_input_records(self) -> Iterator[SerializedCodebookDatapoint]:
        yield from CodebookDatapointSerializer.iter_datapoints(
            self.base_dir,
            datapoints_filename=self.input_filename,
        )

    def _load_records(self, *, limit: int | None = None) -> list[SerializedCodebookDatapoint]:
        records = list(self._iter_input_records())
        if limit is not None:
            records = records[:limit]
        return records

    def _with_backend(self, generate_batch: Callable[[list[str]], list[str]], records: list) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        n_batches = (len(records) + self.batch_size - 1) // self.batch_size
        batch_iter = range(0, len(records), self.batch_size)
        if self.show_progress:
            batch_iter = tqdm(
                batch_iter,
                total=n_batches,
                desc="Inference batches",
                unit="batch",
            )

        with self.output_path.open("w", encoding="utf-8") as f_out:
            for start in batch_iter:
                batch_records = records[start : start + self.batch_size]
                prompts = [build_prompt_from_record(r) for r in batch_records]
                completions = generate_batch(prompts)
                for record, completion in zip(batch_records, completions, strict=True):
                    updated = _record_with_completion(
                        record,
                        model_id=self.model,
                        completion=completion,
                    )
                    f_out.write(json.dumps(updated.to_dict(), ensure_ascii=False) + "\n")

    def run(self, *, limit: int | None = None) -> Path:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input datapoints file not found: {self.input_path}")

        records = self._load_records(limit=limit)
        if not records:
            raise ValueError(f"No records found in {self.input_path}")

        if self.use_vllm:
            backend = VllmBatchBackend(self.model, gen_config=self.gen_config)
            if not backend.load():
                raise RuntimeError(
                    "Failed to load vLLM backend. Install vllm or pass use_vllm=False."
                )
            try:
                self._with_backend(backend.generate_batch, records)
            finally:
                backend.unload()
        else:
            from utils.model_loader import load_model_and_processor

            hf_model, tokenizer = load_model_and_processor(self.model)
            if not hasattr(tokenizer, "encode"):
                raise RuntimeError(
                    "Expected a HuggingFace tokenizer from load_model_and_processor; "
                    "got a processor without encode(). Use use_vllm=True for this model."
                )
            backend = TransformersBatchBackend(
                hf_model,
                tokenizer,
                gen_config=self.gen_config,
            )
            self._with_backend(backend.generate_batch, records)

        return self.output_path
