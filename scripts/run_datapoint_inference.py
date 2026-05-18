from __future__ import annotations

"""
Run model inference on serialized CodebookQA datapoints.

Example:
  python scripts/run_datapoint_inference.py \\
    --base-dir output/model_outputs/test1 \\
    --model /path/to/checkpoint
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.datapoint_inference import CodebookDatapointInferencer
from utils.datapoint_serializer import DEFAULT_DATAPOINTS_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run model inference on serialized CodebookQA datapoints."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="Directory containing datapoints.jsonl and graphs/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model id or local checkpoint path.",
    )
    parser.add_argument(
        "--input-filename",
        type=str,
        default=DEFAULT_DATAPOINTS_FILENAME,
        help="Input JSONL filename under --base-dir.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=None,
        help="Output JSONL filename under --base-dir (default: datapoints_inferred.jsonl).",
    )
    parser.add_argument(
        "--use-vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use vLLM for batch generation (default: true).",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on number of records to process.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Greedy decoding (temperature ignored).",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar.",
    )
    args = parser.parse_args()

    inferencer = CodebookDatapointInferencer(
        base_dir=args.base_dir,
        model=args.model,
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        use_vllm=args.use_vllm,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        batch_size=args.batch_size,
        do_sample=not args.no_sample,
        show_progress=not args.no_progress,
    )

    out = inferencer.run(limit=args.max_examples)
    print(f"Wrote inferred datapoints to {out}")


if __name__ == "__main__":
    main()
