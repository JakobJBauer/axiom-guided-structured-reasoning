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

from dataloader.codebook_qa import CodebookQADataset
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv


load_dotenv()

THINKING_OPEN, THINKING_CLOSE = "<thinking>", "</thinking>"

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
        if response.count(THINKING_OPEN) == 1 and response.count(THINKING_CLOSE) == 1:
            if response.index(THINKING_OPEN) < response.index(THINKING_CLOSE):
                reward += 1.0
        elif response.count(THINKING_OPEN) == 1:
            reward += 0.5;
        elif response.count(THINKING_CLOSE) == 1:
            reward += 0.5;
        rewards.append(reward)
    return rewards

def citation_format_reward(completions, **kwargs):
    # 0 - 0.5 reward depending on the presence of (ATTR : True) or (ATTR : False) tags
    responses = extract_responses(completions)
    rewards = []
    for response in responses:
        response = response.lower()
        reward = 0.0
        start = response.find(THINKING_OPEN)
        end = response.rfind(THINKING_CLOSE)
        if start == -1 or end == -1: reasoning = response
        else: reasoning = response[start + len(THINKING_OPEN):end].strip()
        
        # Here we check whether each argument ends with a citation of the form (ATTR : True) or (ATTR : False)
        # For now we just check presence
        if reasoning.count("(ATTR : True)") + reasoning.count("(ATTR : False)") > 0: reward = 0.5
        else: reward = 0.0


        rewards.append(reward)
    return rewards
        
def answer_format_reward(completions, **kwargs):
    # 0 - 0.5 reward depending on the presence of yes/no answer
    responses = extract_responses(completions)
    rewards = []
    for response in responses:
        response = response.lower()
        reward = 0.0

        # cut to after the last </thinking> tag. Fallback use the whole response.
        end = response.rfind(THINKING_CLOSE)
        if end == -1: answer = response
        else: answer = response[end + len(THINKING_CLOSE):].strip()
        if answer.startswith("yes, the story is") or answer.startswith("no, the story is not"):
            reward += 0.5
        rewards.append(reward)
    return rewards


def load_model_for_grpo(base_model_name_or_path: str, adapter_model_name_or_path: str):
    """
    Load a causal LM for GRPO training.

    Supports either:
    - A base HF model ID or path with a full `config.json`.
    - A LoRA/PEFT adapter directory (like the SFT output), in which case
      we load it via `AutoPeftModelForCausalLM` so the base model is
      automatically resolved and the adapter weights are applied.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading Base Model: {base_model_name_or_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, trust_remote_code=True)

    print(f"Loading SFT Adapter from {adapter_model_name_or_path} (parameter-efficient)...")
    model = PeftModel.from_pretrained(base_model, adapter_model_name_or_path)

    # At this point, `model` is a PEFT-wrapped base model where only adapter
    # parameters are trainable (base weights are frozen), which is exactly
    # the parameter-efficient GRPO setup we want.
    return model, tokenizer


def build_grpo_prompt(sample) -> str:
    """
    Build the user-facing prompt for GRPO from a CodebookQADataset sample.

    This mirrors the SFT prompt used in annotate_codebook_qa_sft.py but
    omits any teacher answer so the model's completion can be evaluated
    by the reward function.
    """
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


class CodebookQAGRPODataset:
    """
    Lightweight dataset wrapper for GRPO that uses CodebookQADataset as the
    underlying dataloader and exposes 'text' fields for TRL.
    """

    def __init__(
        self,
        base_dataset: CodebookQADataset,
        num_examples: int = 1000,
    ) -> None:
        self._base = base_dataset
        self._num_examples = num_examples

    def __len__(self) -> int:
        return self._num_examples

    def __getitem__(self, idx):
        # We ignore idx and sample randomly to keep things simple; CodebookQADataset
        # already handles sampling with its own RNG/seed.
        sample = self._base.sample()
        prompt = build_grpo_prompt(sample)
        # TRL's GRPOTrainer expects a 'prompt' column.
        return {"prompt": prompt}


def run_grpo_training(train_dataset, base_model_name_or_path, adapter_model_name_or_path, output_dir: str) -> None:
    """
    Run GRPO training given a pre-built training dataset.

    Callers are responsible for constructing `train_dataset` (e.g., via
    `CodebookQAGRPODataset`) and loading the model object, so the trainer
    does not depend on any particular data path.
    """
    model, tokenizer = load_model_for_grpo(
        base_model_name_or_path, adapter_model_name_or_path
    )

    from pathlib import Path

    base_name = Path(str(base_model_name_or_path)).name
    adapter_name = Path(str(adapter_model_name_or_path)).name

    training_args = GRPOConfig(
        run_name=f"grpo-{base_name}-{adapter_name}",
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        num_generations=4,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=10,
        report_to="wandb",
        save_strategy="steps",
        save_steps=100,
        beta=0.1,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[thinking_tags_reward, citation_format_reward, answer_format_reward],
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training for codebook QA structural adherence."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B-Instruct",
        help=(
            "Base model ID or local checkpoint directory. "
            "Can also be an SFT LoRA/PEFT adapter directory produced by the SFT script."
        ),
    )
    parser.add_argument(
        "--adapter-model",
        type=str,
        default="Qwen2-CodebookQA-SFT",
        help="Adapter model ID or local checkpoint directory. ",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="Qwen2-CodebookQA-GRPO",
        help="Directory for GRPO fine-tuned model/checkpoints.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=1000,
        help="Number of GRPO training examples to sample from the dataloader.",
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

    args = parser.parse_args()

    # Use the shared CodebookQADataset dataloader for stories + codebooks.
    base_dataset = CodebookQADataset(
        stories=None,
        stories_story_key="story",
        codebooks_root=Path("codebooks") / "final_selection",
        simplestories_split=args.split,
        seed=args.seed,
    )
    train_dataset = CodebookQAGRPODataset(
        base_dataset=base_dataset,
        num_examples=args.num_examples,
    )
    run_grpo_training(
        train_dataset=train_dataset,
        base_model_name_or_path=args.model,
        adapter_model_name_or_path=args.adapter_model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

