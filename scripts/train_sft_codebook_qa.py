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

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

load_dotenv()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an SFT model on codebook QA reasoning traces."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/codebook_qa_sft_gpt_1000.jsonl",
        help="Path to JSONL file with SFT data (must have 'text' field).",
    )
    parser.add_argument(
        "--use-live-dataloader",
        action="store_true",
        help=(
            "Use the on-the-fly CodebookQADataset instead of JSONL. "
            "This produces deterministic traces from the inferred reasoning graph."
        ),
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
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Base model to fine-tune.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen2-CodebookQA-SFT",
        help="Directory where the fine-tuned model will be saved.",
    )

    args = parser.parse_args()

    if args.use_live_dataloader:
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
            config=SFTAdapterConfig(),
        )
    else:
        data_path = Path(args.data_path)
        if not data_path.exists():
            raise FileNotFoundError(
                f"SFT data file not found: {data_path}. "
                "Either pass --data-path to an existing JSONL, or use --use-live-dataloader."
            )
        dataset = load_dataset("json", data_files=str(data_path), split="train")

    training_args = SFTConfig(
        run_name=f"sft-{Path(args.model).name}-{Path(args.output_dir).name}",
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-5,
        save_steps=200,
        report_to="wandb",
        gradient_checkpointing=True,
        warmup_steps=5,
        max_grad_norm=1.0,
        logging_steps=5,
        save_strategy="epoch",
        dataset_text_field="text",
        seed=42,
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"],
    )

    trainer = SFTTrainer(
        model=args.model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()

