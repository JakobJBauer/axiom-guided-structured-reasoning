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
import os
import sys
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset  # noqa: E402
from graph.graph import Graph  # noqa: E402


def render_reasoning_trace(graph: Graph, sink_id: str) -> str:
    """
    Render a reasoning trace from a reasoning subgraph (sink + ancestors).

    Each node with a defined value becomes one paragraph:
    - Leaf nodes (no incoming edges): basic attribute statements
    - Non-leaf nodes: combine sources using citations [ATTR] and a verdict

    All ATTR tokens are rendered in uppercase to align with ReasoningParser.
    """
    # Topological order ensures parents come before children
    topo = graph.topological_sort()
    paragraphs = []

    for node in topo:
        if node.value is None:
            continue

        attr = node.id.upper()
        value_str = "True" if bool(node.value) else "False"
        parents = graph.get_incoming_nodes(node)

        if not parents:
            # Leaf node: basic attribute based on dataset features
            para = (
                f"The story has the basic attribute [{attr}] according to the "
                f"dataset features. ({attr} : {value_str})"
            )
        else:
            parent_attrs = [p.id.upper() for p in parents]
            parent_citations = ", ".join(f"[{a}]" for a in parent_attrs)

            if bool(node.value):
                para = (
                    f"Because {parent_citations} are true, I conclude that the story "
                    f"is [{attr}]. ({attr} : True)"
                )
            else:
                para = (
                    f"Even though {parent_citations} hold, I conclude that the story "
                    f"is not [{attr}]. ({attr} : False)"
                )

        paragraphs.append(para)

    thinking = "<thinking>\n" + "\n\n".join(paragraphs) + "\n</thinking>"

    # Final answer based on sink node's value
    sink = graph.get_node_by_id(sink_id)
    if sink is None or sink.value is None:
        final_answer = "I cannot determine whether the story satisfies the target attribute."
    else:
        sink_label = sink.label or sink.id
        if bool(sink.value):
            final_answer = f"Yes, the story is {sink_label.lower()}."
        else:
            final_answer = f"No, the story is not {sink_label.lower()}."

    return thinking + "\n" + final_answer + "\n"


def build_sft_text(sample) -> str:
    """
    Build a single training example string combining:
    - story
    - codebook
    - question
    - deterministic reasoning + answer.
    """
    reasoning = render_reasoning_trace(sample.reasoning_graph, sample.sink_id)

    text = (
        "Story:\n"
        f"{sample.story}\n\n"
        "Codebook:\n"
        f"{sample.codebook_text}\n\n"
        "Question:\n"
        f"{sample.question}\n\n"
        "Assistant:\n"
        f"{reasoning}"
    )
    return text


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
        help=(
            "Logical split to draw stories from "
            "(train, validation, test; validation maps to the dataset's test split)."
        ),
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize dataset (uses SimpleStories by default)
    dataset = CodebookQADataset(
        stories=None,
        stories_story_key="story",
        codebooks_root=Path("codebooks") / "final_selection",
        simplestories_split=args.split,
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
                "codebook_path": str(sample.codebook_path),
                "question": sample.question,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pbar.update(1)


if __name__ == "__main__":
    main()

