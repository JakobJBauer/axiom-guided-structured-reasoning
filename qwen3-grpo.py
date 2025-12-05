from trl import GRPOConfig, GRPOTrainer
from datasets import load_dataset

dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

def reward_num_unique_letters(completions, **kwargs):
    completions_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completions_contents]

training_args = GRPOConfig(
    output_dir="Qwen2-Instruct-GRPO"
    report_to="wandb",
    run_name="qwen2-grpo-unique-letters",
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=reward_num_unique_letters,
    args=training_args,
    train_dataset=dataset
)

trainer.train()