from __future__ import annotations

"""
Generate SFT training data for the codebook QA task.

This script:
- Samples 1000 datapoints from CodebookQADataset (SimpleStories + final_selection).
- For each datapoint, queries a teacher model (GPT-5-mini) to produce:
    - A reasoning trace in <thinking>...</thinking> format
    - A final answer line, e.g. "Yes, the story is dense."
- Writes results to JSONL with a single 'text' field per row suitable for SFT.
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

from dataloader.codebook_qa import CodebookQADataset


load_dotenv()


def build_sft_text(sample) -> str:
    """
    Build a single training example string combining:
    - story
    - codebook
    - question
    - teacher answer (to be filled in by GPT-5-mini)
    """
    # The teacher will fill in everything after "Assistant:"
    prefix = (
        "You are an expert reasoning assistant. Given a story, a codebook, and a "
        "yes/no question about whether the story satisfies a particular attribute, "
        "answer using the following STRICT format:\n\n"
        "<thinking>\n"
        "- The thinking section consists of multiple PARAGRAPHS.\n"
        "- Each paragraph is ONE argument.\n"
        "- Inside each paragraph, refer to attributes in ALL CAPS in square brackets, "
        "e.g. [SHORT], [NOUN], [NON-NOUN], [DENSE]. These are the nodes/attributes.\n"
        "- Each paragraph MUST END with a citation of the form:\n"
        "    (ATTR : True)\n"
        "  or\n"
        "    (ATTR : False)\n"
        "  where ATTR is the (uppercase) attribute name that this paragraph is "
        "concluding about.\n"
        "- The cited ATTR at the end of the paragraph MUST appear in square brackets "
        "somewhere in that paragraph as [ATTR].\n"
        "- Use one blank line between paragraphs.\n"
        "- Base your arguments on the story, codebook, and question.\n"
        "\n"
        "For example:\n"
        "<thinking>\n"
        "The estimated number of characters for this story is 3000. Since 3000 is "
        "lower than 5000, I conclude that the story is [SHORT]. (SHORT : True)\n"
        "\n"
        "Since the story starts with an adjective (\"Wet raindrops fall...\") it is "
        "not [NOUN]. (NOUN : False)\n"
        "\n"
        "The story is also [NON-NOUN] because it is not [NOUN]. (NON-NOUN : True)\n"
        "\n"
        "Therefore, the story is [DENSE] because both [SHORT] and [NON-NOUN] are "
        "true. (DENSE : True)\n"
        "</thinking>\n"
        "Yes, the story is dense.\n"
        "\n"
        "Now follow exactly this structure for the given story, codebook, and "
        "question.\n"
        "</thinking>\n"
        "Yes, the story is ...\n"
        "# OR\n"
        "No, the story is not ...\n\n"
        "Story:\n"
        f"{sample.story}\n\n"
        "Codebook:\n"
        f"{sample.codebook_text}\n\n"
        "Question:\n"
        f"{sample.question}\n\n"
        "Assistant:\n"
    )
    return prefix


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
        stories=None,
        stories_story_key="story",
        codebooks_root=Path("codebooks") / "final_selection",
        simplestories_split=args.split,
        seed=args.seed,
    )

    def annotate_one(_idx: int) -> Dict[str, Any]:
        """Single annotation job for use in a thread pool."""
        sample = dataset.sample()
        prompt = build_sft_text(sample)

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
                    "content": prompt,
                },
            ],
        )

        answer = completion.choices[0].message.content

        # Final SFT text: user context + teacher answer
        text = prompt + (answer or "")

        record: Dict[str, Any] = {
            "text": text,
            "story": sample.story,
            "codebook_path": str(sample.codebook_path),
            "question": sample.question,
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

