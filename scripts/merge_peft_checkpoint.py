from __future__ import annotations

"""
Merge a PEFT LoRA checkpoint into its base model and save a full HF checkpoint.

Example:
  python scripts/merge_peft_checkpoint.py \\
    --checkpoint models2/Qwen3.5-4B-sft-process/checkpoint-700 \\
    --output models2/Qwen3.5-4B-sft-process/checkpoint-700-merged
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_base_model(base_model: str):
    """Load base weights on CPU (dtype-aware for large models)."""
    kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
    name = str(base_model)
    if "Qwen3.5" in name:
        return AutoModelForImageTextToText.from_pretrained(name, **kwargs)
    return AutoModelForCausalLM.from_pretrained(name, **kwargs)


def _load_tokenizer_or_processor(model_id: str, checkpoint: Path):
    """Prefer tokenizer/processor files from the adapter checkpoint, then base."""
    if (checkpoint / "tokenizer_config.json").exists():
        try:
            return AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        except Exception:
            return AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if "Qwen3.5" in str(model_id):
        return AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def merge_peft_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str | None = None,
) -> Path:
    checkpoint = Path(checkpoint_dir).resolve()
    output = Path(output_dir).resolve()

    if not (checkpoint / "adapter_config.json").exists():
        raise FileNotFoundError(
            f"No adapter_config.json in {checkpoint}. Pass a PEFT checkpoint directory."
        )

    peft_cfg = PeftConfig.from_pretrained(str(checkpoint))
    base_id = base_model or peft_cfg.base_model_name_or_path
    if not base_id:
        raise ValueError(
            "base_model_name_or_path missing from adapter_config.json; pass --base-model."
        )

    print(f"Base model: {base_id}")
    print(f"Adapter:   {checkpoint}")
    print(f"Output:    {output}")

    base = _load_base_model(base_id)
    model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=False)

    print("Merging LoRA into base weights on CPU...")
    if torch.cuda.is_available():
        model = model.to("cpu")
        torch.cuda.empty_cache()

    merged = model.merge_and_unload()

    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))

    proc = _load_tokenizer_or_processor(base_id, checkpoint)
    proc.save_pretrained(str(output))

    meta = {
        "merged_from_checkpoint": str(checkpoint),
        "base_model_name_or_path": base_id,
    }
    (output / "merge_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Done. Merged model saved to {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a PEFT LoRA checkpoint into a full model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="PEFT checkpoint directory (contains adapter_config.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to write the merged full model.",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Override base model path/id (default: read from adapter_config.json).",
    )
    args = parser.parse_args()

    merge_peft_checkpoint(
        args.checkpoint,
        args.output,
        base_model=args.base_model,
    )


if __name__ == "__main__":
    main()
