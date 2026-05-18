from __future__ import annotations

"""
Serialize CodebookQA datapoints (JSONL + per-example graph JSON).

Run from repo root or anywhere; project root is added to sys.path.

Example:
  python scripts/serialize_codebook_datapoints.py --output data/my_serialized --num-examples 100
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from utils.datapoint_serializer import CodebookDatapointSerializer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serialize CodebookQA datapoints to disk.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write datapoints.jsonl and graphs/ under.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of dataset indices to serialize (0 .. num-1).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=("train", "test", "eval"),
        help="Which local split to draw from.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Dataset RNG seed.")
    parser.add_argument("--goal-depth", type=int, default=2, help="Graph difficulty: goal_depth.")
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=6,
        help="Graph difficulty: max_leaf_nodes.",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="none",
        choices=("none", "abbr", "full"),
        help="Stored in each record as prompt_strategy (for later prompt building).",
    )
    parser.add_argument(
        "--datapoints-filename",
        type=str,
        default="datapoints.jsonl",
        help="JSONL filename under --output.",
    )
    args = parser.parse_args()

    difficulty = GraphDifficultyConfig(goal_depth=args.goal_depth, max_leaf_nodes=args.max_leaf_nodes)
    serializer = CodebookDatapointSerializer(
        output_path=args.output,
        difficulty=difficulty,
        split=args.split,
        seed=args.seed,
        prompt_strategy=args.prompt_strategy,
        datapoints_filename=args.datapoints_filename,
    )

    n = args.num_examples
    ds = CodebookQADataset(
        split=args.split,
        difficulties=[(difficulty, 1.0)],
        seed=args.seed,
    )
    if n > len(ds):
        raise SystemExit(f"--num-examples ({n}) exceeds split length ({len(ds)}).")

    out = serializer.serialize(n)
    print(f"Wrote {n} records to {out}")


if __name__ == "__main__":
    main()
