import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

import json
from tqdm.auto import tqdm
from scripts.train_grpo_codebook_qa import thinking_tags_reward, citation_format_reward, intermediate_steps_reward, answer_format_reward, answer_accuracy_reward
ALL_REWARD_FUNCTIONS = [thinking_tags_reward, citation_format_reward, intermediate_steps_reward, answer_format_reward, answer_accuracy_reward]

def completions_from_rows(rows: list[dict]) -> list[list[dict[str, str]]]:
    return [[{"role": "assistant", "content": row["completion"]}] for row in rows]


def reward_kwargs_from_rows(rows: list[dict]) -> dict:
    return {
        "sink_id": [r["sink_id"] for r in rows],
        "sink_label": [r["sink_label"] if "sink_label" in r else None for r in rows],
        "answer": [r["answer"] for r in rows],
        "node_ids": [r["node_ids"] for r in rows],
        "gold_attr_values": [r["gold_attr_values"] for r in rows],
    }

def main(dataset_path: str, output_path: str):
    with open(dataset_path, "r") as f: rows = [json.loads(line) for line in f]
    completions = completions_from_rows(rows)
    kwargs = reward_kwargs_from_rows(rows)

    for reward_function in tqdm(ALL_REWARD_FUNCTIONS, desc=f"Calculating rewards", leave=True):
        rewards = reward_function(completions, **kwargs)
        for row, score in zip(rows, rewards): row.setdefault("rewards", {})[reward_function.__name__] = score
    
    with open(output_path, "w") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=False, default="data/codebook_qa_sft_gpt_1000_complete.jsonl")
    parser.add_argument("--output-path", type=str, required=False, default="data/codebook_qa_sft_gpt_1000_complete_rewards.jsonl")
    args = parser.parse_args()

    main(dataset_path=args.dataset_path, output_path=args.output_path)