from __future__ import annotations

"""
Supervised fine-tuning (SFT) script for the codebook QA task.

Trains a local model to take (story, codebook, question) as input and
generate a reasoning trace plus final answer, using SFT data produced by
annotate_codebook_qa_sft.py (JSONL with a 'text' field).
"""

from pathlib import Path
import sys
from dotenv import load_dotenv
import torch

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from utils import load_model_and_processor
from peft import LoraConfig

load_dotenv()

DEFAULT_GPT_JSONL = "data/codebook_qa_sft_gpt_1000.jsonl"
DEFAULT_DETERMINISTIC_JSONL = "data/codebook_qa_sft_deterministic_1000.jsonl"

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an SFT model on codebook QA reasoning traces."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_GPT_JSONL,
        help="Path to JSONL file with SFT data (must have 'text' field).",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="jsonl",
        choices=["jsonl", "live"],
        help="Train from a JSONL snapshot or from the live on-the-fly dataloader.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="full",
        choices=["full", "abbr", "none"],
        help=(
            "Prompt prefix style. For existing JSONL snapshots: "
            "GPT JSONL is `full`, deterministic JSONL is `none`."
        ),
    )
    parser.add_argument(
        "--abbr-prefix",
        type=str,
        default="TASK: CODEBOOK_QA\n\n",
        help="Abbreviated prefix string when --prompt-style=abbr.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test", "eval"],
        help="Which split to draw from when using the live dataloader.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10000,
        help="Number of SFT examples to serve when using the live dataloader.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen3.5-4B-CodebookQA-SFT",
        help="Directory where the fine-tuned model will be saved.",
    )

    args = parser.parse_args()

    if args.data_source == "live":
        from dataloader import CodebookQADataset, GraphDifficultyConfig
        from dataloader.trl_adapters import CodebookQASFTDataset, SFTAdapterConfig

        base = CodebookQADataset(
            split=args.split,
            difficulties=[(GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=6), 1.0)],
            seed=42,
        )
        dataset = CodebookQASFTDataset(
            base_dataset=base,
            num_examples=args.num_examples,
            config=SFTAdapterConfig(
                prompt_style=args.prompt_style,
                abbr_prefix=args.abbr_prefix,
                completion_mode="deterministic_trace",
            ),
        )
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"SFT data file not found: {data_path}. "
                "Either pass --data-path to an existing JSONL, or use --data-source=live."
            )
        dataset = load_dataset("json", data_files=str(data_path), split="train")
        # For JSONL snapshots we cannot reliably *remove* an existing long prefix.
        # But we can safely *prepend* prefixes if desired.
        if args.prompt_style != "none":
            from dataloader.trl_adapters import build_task_prefix

            prefix = build_task_prefix(args.prompt_style, abbr_prefix=args.abbr_prefix)
            if prefix:
                # Keep this as a real HF Dataset so TRL can introspect `column_names`.
                def _prepend_prefix(ex):
                    return {"text": prefix + ex.get("text", "")}

                dataset = dataset.map(_prepend_prefix)

    model, processor = load_model_and_processor(args.model)

    training_args = SFTConfig(
        run_name=f"sft-{Path(args.model).name}-{Path(args.output_dir).name}",
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        max_length=4096,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_steps=250,
        report_to="wandb",
        gradient_checkpointing=True,
        warmup_steps=5,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="epoch",
        dataset_text_field="text",
        seed=42,
        bf16=True,
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # attention
            "gate_proj", "up_proj", "down_proj",       # MLP
        ],
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # merged_dir = Path(args.output_dir) / "merged"
    # merged_dir.mkdir(parents=True, exist_ok=True)

    # model_to_merge = trainer.model

    # # Merge on CPU to avoid VRAM spikes / OOM during adapter merge.
    # if torch.cuda.is_available():
    #     try:
    #         model_to_merge = model_to_merge.to("cpu")
    #     finally:
    #         torch.cuda.empty_cache()

    # merged_model = model_to_merge.merge_and_unload()
    # merged_model.save_pretrained(str(merged_dir))
    # processor.save_pretrained(str(merged_dir))


if __name__ == "__main__":
    main()

