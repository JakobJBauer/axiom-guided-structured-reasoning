from __future__ import annotations

"""
GRPO training script for the codebook QA task.

Uses a reward function that encourages the model to:
- Produce a <thinking>...</thinking> block.
- Follow it with a clear yes/no answer of the form:
  "Yes, the story is ..." or "No, the story is not ...".

The prompts are the same 'text' field used for SFT, but GRPO focuses on
format adherence rather than teacher matching.
"""

import re
from pathlib import Path
import sys

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer


THINKING_OPEN = "<thinking>"
THINKING_CLOSE = "</thinking>"


def reward_structure(completions, **kwargs):
    """
    Reward adherence to the desired output structure.

    Components:
    - +1.0 if there is a <thinking>...</thinking> block in correct order.
    - +1.0 if the final non-empty line after </thinking> is a yes/no answer
      of the form:
         "Yes, the story is ..." or "No, the story is not ..."
    - +0.5 if there is at least one predicate verdict like
      "(ATTR : True)" or "(ATTR : False)" inside the thinking block.
    - +0.5 if:
        * There is at least one [ATTR] citation in the thinking block, and
        * For every (ATTR : ...) verdict inside the thinking block, that ATTR
          also appears as [ATTR] somewhere in the thinking block.
    """
    rewards = []

    for completion in completions:
        # TRL GRPO passes a list of message dicts per completion; we assume
        # the first element is the assistant content string.
        if isinstance(completion, list) and completion and isinstance(
            completion[0], dict
        ):
            text = completion[0].get("content", "") or ""
        else:
            text = str(completion)

        r = 0.0
        lower = text.lower()

        # 1) Check thinking block structure
        start = lower.find(THINKING_OPEN)
        end = lower.find(THINKING_CLOSE)
        if start != -1 and end != -1 and end > start:
            r += 1.0

            thinking_block = text[start + len(THINKING_OPEN) : end]
            # 3) Check for predicate verdict patterns inside thinking block
            verdict_pattern = r"\(([A-Z0-9\-_]+)\s*:\s*(True|False)\)"
            verdict_matches = re.findall(verdict_pattern, thinking_block)
            if verdict_matches:
                r += 0.5

            # Check for [ATTR] citations and alignment with (ATTR : ...) verdicts
            citation_pattern = r"\[([A-Z0-9\-_]+)\]"
            citations = re.findall(citation_pattern, thinking_block)
            citation_set = {c.upper() for c in citations}

            if citation_set and verdict_matches:
                # All verdict ATTRs must also appear as [ATTR] somewhere
                verdict_attrs = {name.upper() for name, _ in verdict_matches}
                if verdict_attrs.issubset(citation_set):
                    r += 0.5

            # 2) Check final answer line after </thinking>
            after = text[end + len(THINKING_CLOSE) :].strip().splitlines()
            # take last non-empty line
            last_non_empty = ""
            for line in reversed(after):
                line = line.strip()
                if line:
                    last_non_empty = line
                    break

            ans = last_non_empty.lower()
            if ans.startswith("yes, the story is") or ans.startswith(
                "no, the story is"
            ):
                r += 1.0
        else:
            # If no thinking block, small penalty
            r -= 0.5

        rewards.append(float(r))

    return rewards


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training for codebook QA structural adherence."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/codebook_qa_sft_1000.jsonl",
        help="Path to JSONL file with prompts (uses 'text' field).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Base model (e.g., SFT checkpoint) to further train with GRPO.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen2-CodebookQA-GRPO",
        help="Directory for GRPO fine-tuned model/checkpoints.",
    )

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}. Run annotate_codebook_qa_sft.py first."
        )

    dataset = load_dataset("json", data_files=str(data_path), split="train")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=args.model,
        reward_funcs=reward_structure,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="text",
    )

    trainer.train()


if __name__ == "__main__":
    main()

