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
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from dataloader.trl_adapters import CodebookQAGRPODataset
from utils import load_model_and_processor
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from dotenv import load_dotenv


load_dotenv()

THINKING_OPEN, THINKING_CLOSE = "<thinking>", "</thinking>"
CITATION_PATTERN = re.compile(r'\(([A-Z][A-Z\-]*) : (True|False)\)', re.IGNORECASE)

def extract_responses(completions):
    responses = []
    for completion in completions:
        if isinstance(completion, list) and completion and isinstance(
            completion[0], dict
        ):
            text = completion[0].get("content", "") or ""
        else:
            text = str(completion)
        responses.append(text)
    return responses

def thinking_tags_reward(completions, **kwargs):
    # 0 - 1 reward depending on the presence of <thinking>...</thinking> tags
    responses = extract_responses(completions)
    rewards = []
    for response in responses:
        response = response.lower()
        reward = 0.0
        open_count, closed_count = response.count(THINKING_OPEN), response.count(THINKING_CLOSE)
        if open_count == 1 and closed_count == 1:
            open_idx, closed_idx = response.index(THINKING_OPEN), response.index(THINKING_CLOSE)
            if open_idx < closed_idx:
                reward += 0.8
                content = response[open_idx + len(THINKING_OPEN):closed_idx].strip()
                if len(content.splitlines()) >= 2: reward += 0.2 # Give extra credit for multiple paragraphs.
            else: reward += 0.6
        elif open_count == 1 or closed_count == 1: reward += 0.5;
        rewards.append(reward)
    return rewards

def citation_format_reward(completions, **kwargs):
    # 0 - 1 reward depending on the presence of (ATTR : True) or (ATTR : False) tags. Gives partial credit.
    responses = extract_responses(completions)
    rewards = []
    for response in responses:
        response = response.lower()
        start = response.find(THINKING_OPEN)
        end = response.rfind(THINKING_CLOSE)
        if start == -1 or end == -1:
            rewards.append(0.0)
            continue
        
        reasoning = response[start + len(THINKING_OPEN):end].strip()

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', reasoning) if p.strip()]
        if not paragraphs:
            rewards.append(0.0)
            continue

        matching = 0.0
        for paragraph in paragraphs:
            last_line = paragraph.splitlines()[-1].strip()
            citation_match = CITATION_PATTERN.search(last_line) # get the citation
            if not citation_match: continue
            matching += 0.3
            if last_line.endswith(")"): matching += 0.1 # Make sure it actually ends in the citation.

            attr = citation_match.group(1).upper()
            if f"[{attr}]" in paragraph.upper(): matching += 0.6 # We give extra credit when the citation is relevant to the paragraph
        
        rewards.append(matching / len(paragraphs))
    return rewards
        
def answer_format_reward(completions, sink_id, **kwargs):
    # 0 - 0.5 reward depending on the presence of yes/no answer
    responses = extract_responses(completions)
    rewards = []
    for response, sink in zip(responses, sink_id):
        response = response.lower()
        sink = str(sink).lower()
        end = response.rfind(THINKING_CLOSE)
        out = response[end + len(THINKING_CLOSE):].strip() if end != -1 else response

        EXPECTED_RESPONSE = f"yes, the story is {sink}", f"no, the story is not {sink}"

        if any(out.startswith(expected) for expected in EXPECTED_RESPONSE): rewards.append(0.5)
        else: rewards.append(0.0)
    return rewards


def run_grpo_training(
    train_dataset,
    model_name_or_path,
    output_dir: str,
    num_examples: int,
    max_steps: int = -1,
) -> None:
    """
    Run GRPO training given a pre-built training dataset.

    Callers are responsible for constructing `train_dataset` (e.g., via
    `CodebookQAGRPODataset`) and loading the model object, so the trainer
    does not depend on any particular data path.
    """
    model, processor = load_model_and_processor(model_name_or_path)

    from pathlib import Path

    base_name = Path(str(model_name_or_path)).name

    per_device_train_batch_size = 8
    gradient_accumulation_steps = 1

    # HF Trainer requires `max_steps > 0` when dataset has no `__len__`.
    # The GRPO adapter dataset is iterable, so derive a sensible default.
    if max_steps <= 0:
        effective_batch = per_device_train_batch_size * gradient_accumulation_steps
        max_steps = max(1, num_examples // effective_batch)
        print(
            f"Dataset has no static length; setting max_steps={max_steps} "
            f"(num_examples={num_examples}, effective_batch={effective_batch})."
        )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_generations=4,
        num_train_epochs=1,  # ignored when max_steps > 0
        max_steps=max_steps,
        learning_rate=5e-6,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        
        # GRPO specific hyperparameters
        beta=0.1,
        bf16=True,

        # Tracking info
        report_to="wandb",
        save_strategy="steps",
        logging_steps=10,
        save_steps=100,
        run_name=f"grpo-{base_name}",

        # Fast inference with VLLM
        # use_vllm=True,
        # vllm_mode="colocate",
    )

    if not hasattr(model, "peft_config") and bool(getattr(model, "peft_config")):
        raise ValueError("Model does not have a PEFT config.")


    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[thinking_tags_reward, citation_format_reward, answer_format_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training for codebook QA structural adherence."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3.5-4B",
        help=(
            "Base model ID/path, or SFT LoRA checkpoint directory. "
            "If you pass an SFT PEFT directory, GRPO reuses that adapter."
        ),
    )
    # parser.add_argument(
    #     "--adapter-model",
    #     type=str,
    #     default=None,
    #     help=(
    #         "(deprecated; ignored in full fine-tuning mode) "
    #         "LoRA/PEFT adapter directory produced by the SFT script."
    #     ),
    # )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen3.5-4B-CodebookQA-GRPO",
        help="Directory for GRPO fine-tuned model/checkpoints.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of GRPO training examples to sample from the dataloader.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="full",
        choices=["full", "abbr", "none"],
        help="Prompt prefix style: full instructions, abbreviated task tag, or none.",
    )
    parser.add_argument(
        "--abbr-prefix",
        type=str,
        default="TASK: CODEBOOK_QA\n\n",
        help="Abbreviated prefix string when --prompt-style=abbr.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the CodebookQADataset dataloader.",
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
        "--max-steps",
        type=int,
        default=-1,
        help=(
            "Override GRPO max update steps. If <= 0, auto-compute from "
            "--num-examples and effective batch size."
        ),
    )

    args = parser.parse_args()
    # if args.adapter_model:
    #     print(
    #         "Warning: --adapter-model is ignored in full fine-tuning mode. "
    #         "Pass the full SFT checkpoint directory via --model."
    #     )

    # Use the shared CodebookQADataset dataloader for stories + codebooks.
    base_dataset = CodebookQADataset(
        split=args.split,
        difficulties=[(GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=6), 1.0)],
        seed=args.seed,
    )
    train_dataset = CodebookQAGRPODataset(
        base_dataset=base_dataset,
        num_examples=args.num_examples,
        prompt_style=args.prompt_style,
        abbr_prefix=args.abbr_prefix,
    )
    run_grpo_training(
        train_dataset=train_dataset,
        model_name_or_path=args.model,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()

