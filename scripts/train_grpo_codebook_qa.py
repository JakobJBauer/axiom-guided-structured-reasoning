from __future__ import annotations

"""
GRPO training script for the codebook QA task.

Reward modes:
  - structure: encourages <thinking> tags + citation formatting + a template answer line
  - answer_only: scores ONLY whether the final answer is correct (ignores all other text/format)
  - process: structure rewards + intermediate citation accuracy when parseable
"""

import re
from pathlib import Path
import sys
from typing import Any, Literal
import os

# Ensure project root is on sys.path so we can import local modules
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from dataloader.trl_adapters import CodebookQAGRPODataset
from utils import load_model_and_processor
from trl import GRPOConfig, GRPOTrainer
from dotenv import load_dotenv


load_dotenv()

THINKING_OPEN, THINKING_CLOSE = "<thinking>", "</thinking>"
CITATION_PATTERN = re.compile(r'\(\s?([A-Z][A-Z0-9_\-]*)\s?:\s?(True|False)\s?\)', re.IGNORECASE)

RewardMode = Literal["structure", "answer_only", "process"]


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


def _final_answer_tail(response_lower: str) -> str:
    end = response_lower.rfind(THINKING_CLOSE)
    if end != -1:
        return response_lower[end + len(THINKING_CLOSE) :].strip()
    return response_lower.strip()


def _strict_reasoning_citations(response: str) -> dict[str, bool] | None:
    """
    If <thinking> is well-formed and every non-empty reasoning paragraph ends with
    a valid trailing citation, return ATTR_UPPER -> bool (last assignment wins).
    Otherwise None.
    """
    r = response.lower()
    start = r.find(THINKING_OPEN)
    end = r.rfind(THINKING_CLOSE)
    if start == -1 or end == -1 or start >= end:
        return None
    reasoning = r[start + len(THINKING_OPEN) : end].strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", reasoning) if p.strip()]
    if not paragraphs:
        return None
    pred: dict[str, bool] = {}
    for paragraph in paragraphs:
        last_line = paragraph.splitlines()[-1].strip()
        matches = list(CITATION_PATTERN.finditer(last_line))
        if not matches:
            return None
        citation_match = matches[-1]
        if last_line[citation_match.end() :].strip():
            return None
        attr = citation_match.group(1).upper()
        pred[attr] = citation_match.group(2).lower() == "true"
    return pred


def thinking_tags_reward(completions, **kwargs):
    # 0 - 0.5 reward depending on the presence of <thinking>...</thinking> tags
    responses = extract_responses(completions)
    rewards = []
    for response in responses:
        response = response.lower()
        reward = 0.0
        open_count, closed_count = response.count(THINKING_OPEN), response.count(THINKING_CLOSE)
        if open_count == 1 and closed_count == 1:
            open_idx, closed_idx = response.index(THINKING_OPEN), response.index(THINKING_CLOSE)
            if open_idx < closed_idx:
                reward += 0.4
                content = response[open_idx + len(THINKING_OPEN):closed_idx].strip()
                if len(content.splitlines()) >= 2: reward += 0.1 # Give extra credit for multiple paragraphs.
            else: reward += 0.3
        rewards.append(reward)
        if os.environ.get("DEBUG_THINKING_TAGS", "false").lower() == "true": print(f"---------------------\nResponse: {response}\nReward: {rewards[-1]} for thinking tags.\nPASSAGE END ---------------\n")

    return rewards

def citation_format_reward(completions, node_ids, **kwargs):
    # 0 - 2 reward depending on the presence of (ATTR : True) or (ATTR : False) tags. Gives partial credit.
    # Only gives credit when the cited ATTR exists in this example's codebook graph.
    responses = extract_responses(completions)
    rewards = []
    for response, valid_nodes in zip(responses, node_ids):
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

        valid_set = {str(n).upper() for n in (valid_nodes or [])}
        matching = 0.0
        for paragraph in paragraphs:
            last_line = paragraph.splitlines()[-1].strip()
            matches = list(CITATION_PATTERN.finditer(last_line))
            if not matches: continue
            citation_match = matches[-1]
            if last_line[citation_match.end():].strip(): continue
            cited_attr = citation_match.group(1).upper()
            if cited_attr not in valid_set: continue
            matching += 1.0

            # attr = citation_match.group(1).upper()
            # if f"[{attr}]" in paragraph.upper(): matching += 0.5 # We give extra credit when the citation is relevant to the paragraph
        reward = 2.0 * matching / len(paragraphs)
        rewards.append(reward)
        if os.environ.get("DEBUG_CITATION_FORMAT", "false").lower() == "true": print(f"---------------------\nResponse: {response}\nReward: {rewards[-1]} for citation format.\nPASSAGE END ---------------\n")
    return rewards
        
def _matches_answer_format_line(out: str, sink: str, label: str | None = None) -> bool:
    alts = re.escape(sink.lower()) + "|" + re.escape(label.lower()) if label else re.escape(sink)
    pat = re.compile(
        rf"^(?:(yes)?,?\s*the\s*story\s*is|(no)?,?\s*the\s*story\s*is\s*not)\s*[\[\(]?(?:{alts})[\]\)]?.*\b",
        re.IGNORECASE,
    )
    return pat.match(out) is not None


def answer_format_reward(completions, sink_id, sink_label=None, **kwargs):
    # 0 - 0.5 reward depending on the presence of yes/no answer
    responses = extract_responses(completions)
    labels = sink_label
    if labels is None:
        labels = [None] * len(sink_id)
    rewards = []
    for response, sink, label in zip(responses, sink_id, labels, strict=True):
        out = _final_answer_tail(response.lower())
        matched = _matches_answer_format_line(out, sink, label)
        rewards.append(0.5 if matched else 0.0)
        if os.environ.get("DEBUG_ANSWER_FORMAT", "false").lower() == "true":
            print(
                f"---------------------\nResponse: {response}\nReward: {rewards[-1]} "
                f"for answer format. Sink tokens: {sink_tokens}.\nPASSAGE END ---------------\n"
            )
    return rewards


def answer_accuracy_reward(completions, sink_id, answer, **kwargs):
    """1.0 if the final line matches the gold boolean answer; 0.0 otherwise."""
    _BOOL_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)
    def _parse_final_boolean_answer(response: str) -> bool | None:
        """
        Parse the final boolean decision from the model output.
        We intentionally ignore everything except the *last* yes/no/true/false token
        in the tail after </thinking> (if present).
        """
        tail = _final_answer_tail(response.lower())
        matches = list(_BOOL_TOKEN_RE.finditer(tail))
        if not matches:
            return None
        token = matches[-1].group(1).lower()
        if token in {"yes", "true"}:
            return True
        if token in {"no", "false"}:
            return False
        return None

    responses = extract_responses(completions)
    rewards = []
    for response, sink, gold_bool in zip(responses, sink_id, answer, strict=True):
        response = response.lower()
        sink = str(sink).lower()
        final_response = _parse_final_boolean_answer(response)
        if final_response is None: rewards.append(0.0)
        elif final_response == gold_bool: rewards.append(1.0)
        else: rewards.append(0.0)

        if os.environ.get("DEBUG_ANSWER_ACCURACY", "false").lower() == "true": print(f"---------------------\nResponse: {response}\nReward: {rewards[-1]} for answer accuracy. Gold answer: {gold_bool}. Sink: {sink}.\nPASSAGE END ---------------\n")
    return rewards


def intermediate_steps_reward(completions, gold_attr_values, **kwargs):
    """
    Fraction of gold (attr -> bool) pairs that match parsed citations when the
    thinking block is fully citation-valid; otherwise None (skipped in GRPO sum).

    Maximum reward is 3.0
    """
    responses = extract_responses(completions)
    rewards = []
    for response, gold in zip(responses, gold_attr_values, strict=True):
        pred = _strict_reasoning_citations(response)
        if pred is None:
            rewards.append(None)
            continue
        gold_map = gold or {}
        if not gold_map:
            rewards.append(0.0)
            continue
        correct = sum(1 for attr, pred_val in pred.items() if pred_val == gold_map.get(attr))
        rewards.append(3.0 * correct / len(pred))

        if os.environ.get("DEBUG_INTERMEDIATE_STEPS", "false").lower() == "true": print(f"---------------------\nResponse: {response}\nReward: {rewards[-1]} for intermediate steps. Gold: {gold}. Pred: {pred}.\nPASSAGE END ---------------\n")
    return rewards


def reward_functions_for_mode(mode: RewardMode):
    if mode == "structure":
        return [thinking_tags_reward, citation_format_reward, answer_format_reward]
    if mode == "answer_only" or mode == "answer-only":
        return [answer_accuracy_reward]
    if mode == "process":
        return [
            thinking_tags_reward,
            citation_format_reward,
            intermediate_steps_reward,
            answer_format_reward,
            answer_accuracy_reward,
        ]
    raise ValueError(f"Unknown reward mode: {mode!r}")


def run_grpo_training(
    train_dataset,
    model_name_or_path,
    output_dir: str,
    num_examples: int,
    max_steps: int = -1,
    reward_mode: RewardMode = "structure",
    per_device_train_batch_size: int = 8,
    max_completion_length: int = 2048,
    use_vllm: bool = False,
    peft: bool = True,
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

    per_device_train_batch_size = int(per_device_train_batch_size)
    # As requested: gradient accumulation is 8, or the batch size if batch size > 8.
    gradient_accumulation_steps = max(16 // per_device_train_batch_size, 1)

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
        max_completion_length=max_completion_length,
        beta=0.1,
        bf16=True,

        # Tracking info
        report_to="wandb",
        save_strategy="steps",
        logging_steps=10,
        save_steps=100,
        run_name=f"grpo-{reward_mode}-{base_name}",

        # Fast inference with VLLM
        use_vllm=use_vllm,
        vllm_mode="colocate",
    )

    peft_config = None
    if peft:
        # If the loader already attached a PEFT adapter (e.g., you passed an SFT LoRA dir),
        # keep training that adapter. GRPOTrainer will error if we pass both a PeftModel
        # and a new `peft_config`.
        model_has_adapter = bool(getattr(model, "peft_config", None))
        if model_has_adapter:
            print("Model already has a PEFT adapter attached; continuing PEFT training from it.")
        else:
            from peft import LoraConfig

            # Match the SFT script's LoRA config exactly.
            peft_config = LoraConfig(
                task_type="CAUSAL_LM",
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                bias="none",
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",  # attention
                    "gate_proj",
                    "up_proj",
                    "down_proj",  # MLP
                ],
            )


    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_functions_for_mode(reward_mode),
        args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GRPO training for codebook QA (structure / answer-only / process rewards)."
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
    def _str2bool(v: str) -> bool:
        s = str(v).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise argparse.ArgumentTypeError(f"Expected a boolean (true/false), got: {v!r}")

    parser.add_argument(
        "--peft",
        type=_str2bool,
        default=True,
        help="Train with PEFT/LoRA (True/False). Default: True.",
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
        "--batch-size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--prompt-style",
        type=str,
        default=None,
        choices=["full", "abbr", "none"],
        help=(
            "Prompt prefix style: full instructions, abbreviated task tag, or none. "
            "If omitted, defaults to: answer_only -> none, otherwise -> full."
        ),
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
    parser.add_argument(
        "--reward-mode",
        type=str,
        default="structure",
        choices=["structure", "answer_only", "answer-only", "process"],
        help=(
            "structure: thinking + citations + template answer line; "
            "answer_only: final answer correctness only; "
            "process: structure rewards + intermediate citation accuracy when parseable."
        ),
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=2048,
        help=(
            "Max new tokens per GRPO completion (TRL default 256 is usually too small for "
            "full thinking traces). Lower if you run out of VRAM during generation/logprob."
        ),
    )
    parser.add_argument(
        "--use-vllm",
        action="store_true",
        default=False,
        help="Use VLLM for generation.",
    )

    args = parser.parse_args()

    if args.prompt_style is None:
        args.prompt_style = "none" if args.reward_mode == "answer_only" else "full"
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
        reward_mode=args.reward_mode,
        per_device_train_batch_size=args.batch_size,
        max_completion_length=args.max_completion_length,
        use_vllm=args.use_vllm,
        peft=args.peft,
    )


if __name__ == "__main__":
    main()

