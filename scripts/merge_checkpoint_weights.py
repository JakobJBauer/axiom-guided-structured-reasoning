from __future__ import annotations

"""
Fill missing weight tensors (e.g. vision tower) from a base model into a trained checkpoint.

Use after full fine-tune on text-only data when SFTTrainer saved language weights
but dropped multimodal tensors. For LoRA checkpoints, use merge_peft_checkpoint.py
with --fill-missing-from-base instead.

Example:
  uv run scripts/merge_checkpoint_weights.py \\
    --base Qwen/Qwen3.5-4B \\
    --trained models/qwen35-4B-full-sft \\
    --output models/qwen35-4B-full-sft/merged
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.checkpoint_merge import env_bool, merge_checkpoint_with_base_weights


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge trained safetensors with a base model (vision / vLLM hotfix)."
    )
    parser.add_argument(
        "--base",
        type=str,
        required=True,
        help="Base model directory or HF id with full weights (e.g. Qwen/Qwen3.5-4B).",
    )
    parser.add_argument(
        "--trained",
        type=str,
        required=True,
        help="Trained checkpoint directory (typically text-only SFT output).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: <trained>/merged, or --trained if --merge-same-dir).",
    )
    parser.add_argument(
        "--merge-same-dir",
        action="store_true",
        help="Write merged weights into --trained (also respects MERGE_SAME_DIR=1).",
    )
    args = parser.parse_args()

    merge_same_dir = args.merge_same_dir or env_bool("MERGE_SAME_DIR", False)
    trained = Path(args.trained).resolve()
    output = args.output
    if output is None:
        output = str(trained) if merge_same_dir else str(trained / "merged")

    print(f"Base:    {args.base}")
    print(f"Trained: {trained}")
    print(f"Output:  {output} (merge_same_dir={merge_same_dir})")

    out = merge_checkpoint_with_base_weights(
        base_model_dir=args.base,
        trained_checkpoint_dir=str(trained),
        output_dir=output,
        merge_same_dir=merge_same_dir,
    )
    print(f"Done. Merged weights saved to {out}")


if __name__ == "__main__":
    main()
