from __future__ import annotations

"""
Generate SFT training data for the codebook QA task.

This script:
- Samples 1000 datapoints from CodebookQADataset (SimpleStories + final_selection).
- For each datapoint, queries a teacher model (GPT-5-mini) to produce:
    - A reasoning trace in <thinking>...</thinking> format
    - A final answer line, e.g. "Yes, the story is dense."
- Writes results to JSONL with 'text' (prompt + completion) plus structured fields
  for reward evaluation and debugging.
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from dataloader.trl_adapters import (
    build_base_prompt,
    build_task_prefix,
    gold_attr_values_from_graph,
)


load_dotenv()


def build_teacher_prompt(sample) -> str:
    """
    Prompt sent to the GPT teacher: full instructions + example content.
    """
    task_prefix = build_task_prefix("full")
    return task_prefix + build_base_prompt(sample, prompt_style="none")


def build_sft_text(sample) -> str:
    """
    Training example text stored in JSONL (no long prefix).
    """
    return build_base_prompt(sample, prompt_style="none")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Annotate Codebook QA datapoints with GPT-5-mini for SFT."
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of annotated examples to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/codebook_qa_sft_1000.jsonl",
        help="Output JSONL file for SFT data.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini-2025-08-07",
        help="Teacher model to use via OpenAI API.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for sampling the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help=(
            "Logical split to draw stories from "
            "(train, validation, test; validation maps to the dataset's test split)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel GPT calls.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize teacher model client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set in the environment.")
    client = OpenAI(api_key=api_key)

    # Initialize dataset (uses SimpleStories by default)
    dataset = CodebookQADataset(
        split=args.split,
        difficulties=[(GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=6), 1.0)],
        seed=args.seed,
    )

    def annotate_one(_idx: int) -> Dict[str, Any]:
        """Single annotation job for use in a thread pool."""
        sample = dataset.sample()
        # Prompt the teacher with full instructions
        teacher_prompt = build_teacher_prompt(sample)

        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert reasoning assistant. Always follow the "
                        "requested output format exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": teacher_prompt,
                },
            ],
        )

        completion_text = completion.choices[0].message.content or ""
        prompt = build_sft_text(sample)

        sink_node = sample.reasoning_graph.get_node_by_id(sample.sink_id)
        sink_label = sink_node.label if sink_node is not None else sample.sink_id

        record: Dict[str, Any] = {
            "text": prompt + completion_text,
            "prompt": prompt,
            "completion": completion_text,
            "story": sample.story,
            "codebook": sample.codebook_text,
            "question": sample.question,
            "sink_id": sample.sink_id,
            "sink_label": sink_label,
            "answer": bool(sample.answer),
            "node_ids": [node.id.upper() for node in sample.reasoning_graph.get_nodes()],
            "gold_attr_values": gold_attr_values_from_graph(sample.reasoning_graph),
        }
        return record

    with output_path.open("w", encoding="utf-8") as f_out, ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor, tqdm(
        total=args.num_examples, desc="Annotating (GPT teacher)"
    ) as pbar:
        futures = [executor.submit(annotate_one, i) for i in range(args.num_examples)]
        for fut in as_completed(futures):
            record = fut.result()
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)


if __name__ == "__main__":
    main()

