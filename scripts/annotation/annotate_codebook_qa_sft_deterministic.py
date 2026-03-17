from __future__ import annotations

"""
Deterministic SFT data generation for the codebook QA task.

This script:
- Samples datapoints from CodebookQADataset (SimpleStories + final_selection).
- Uses the gold reasoning graph (sink + ancestors) and leaf values to render
  a reasoning trace in the required structured format:

    <thinking>
    ... paragraph with [ATTR] citations ... (ATTR : True/False)

    ... next paragraph ...
    </thinking>
    Yes, the story is ...

- Writes results to JSONL with a single 'text' field per row suitable for SFT.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from dataloader.trl_adapters import build_base_prompt, render_reasoning_trace


def build_sft_text(sample) -> str:
    reasoning = render_reasoning_trace(sample.reasoning_graph, sample.sink_id)
    return build_base_prompt(sample, prompt_style="none") + reasoning


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Deterministically annotate Codebook QA datapoints using gold reasoning graphs."
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
        default="data/codebook_qa_sft_deterministic_1000.jsonl",
        help="Output JSONL file for SFT data.",
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
        choices=["train", "test", "eval"],
        help="Which local split to draw stories from.",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize dataset (uses SimpleStories by default)
    dataset = CodebookQADataset(
        split=args.split,
        difficulties=[(GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=6), 1.0)],
        seed=args.seed,
    )

    with output_path.open("w", encoding="utf-8") as f_out, tqdm(
        total=args.num_examples, desc="Annotating (deterministic)"
    ) as pbar:
        for _ in range(args.num_examples):
            sample = dataset.sample()
            text = build_sft_text(sample)

            record: Dict[str, Any] = {
                "text": text,
                "story": sample.story,
                "question": sample.question,
                "sink_id": sample.sink_id,
                "answer": bool(sample.answer),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)


if __name__ == "__main__":
    main()

