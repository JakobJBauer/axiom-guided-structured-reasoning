# GRPO Structure, no SFT

# Qwen3.5 0.8B
python scripts/train_grpo_codebook_qa.py   --model "/hf/models/Qwen/Qwen3.5-0.8B"   --output-dir "models/Qwen3.5-0.8B-structure"   --reward-mode structure   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen3.5 4B
python scripts/train_grpo_codebook_qa.py   --model "/hf/models/Qwen/Qwen3.5-4B"   --output-dir "models/Qwen3.5-4B-structure"   --reward-mode structure   --num-examples 16000   --batch-size 2 --use-vllm

# Qwen2.5 0.5B
python scripts/train_grpo_codebook_qa.py   --model "Qwen/Qwen2.5-0.5B-Instruct"   --output-dir "models/Qwen2.5-0.5B-Instruct-structure"   --reward-mode structure   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen2.5 3B
python scripts/train_grpo_codebook_qa.py   --model "Qwen/Qwen2.5-3B-Instruct"   --output-dir "models/Qwen2.5-3B-Instruct-structure"   --reward-mode structure   --num-examples 16000   --batch-size 2 --use-vllm


# Just SFT

# Qwen3.5 0.8B
python scripts/train_sft_codebook_qa.py   \
--model "/hf/models/Qwen/Qwen3.5-0.8B"   \
--data-source jsonl   \
--data-path "data/codebook_qa_sft_gpt_1000.jsonl"   \
--prompt-style full   \
--output-dir "models/Qwen3.5-0.8B-sft" \
--batch-size 8 \
--grad-accum 1 \
--max-length 2048

# Qwen3.5 4B
python scripts/train_sft_codebook_qa.py   \
--model "/hf/models/Qwen/Qwen3.5-4B"   \
--data-source jsonl   \
--data-path "data/codebook_qa_sft_gpt_1000.jsonl"   \
--prompt-style full   \
--output-dir "models/Qwen3.5-4B-sft" \
--batch-size 4 \
--grad-accum 1 \
--max-length 2048

# Qwen2.5 0.5B
python scripts/train_sft_codebook_qa.py   \
--model "Qwen/Qwen2.5-0.5B-Instruct"   \
--data-source jsonl   \
--data-path "data/codebook_qa_sft_gpt_1000.jsonl"   \
--prompt-style full   \
--output-dir "models/Qwen2.5-0.5B-Instruct-sft" \
--batch-size 8 \
--grad-accum 1 \
--max-length 2048

# Qwen2.5 3B
python scripts/train_sft_codebook_qa.py   \
--model "Qwen/Qwen2.5-3B-Instruct"   \
--data-source jsonl   \
--data-path "data/codebook_qa_sft_gpt_1000.jsonl"   \
--prompt-style full   \
--output-dir "models/Qwen2.5-3B-Instruct-sft" \
--batch-size 4 \
--grad-accum 1 \
--max-length 2048


# GRPO Structure on SFT
# Qwen3.5 0.8B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen3.5-0.8B-sft"   --output-dir "models/Qwen3.5-0.8B-sft-structure"   --reward-mode structure   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen3.5 4B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen3.5-4B-sft"   --output-dir "models/Qwen3.5-4B-sft-structure"   --reward-mode structure   --num-examples 16000   --batch-size 2 --use-vllm

# Qwen2.5 0.5B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen2.5-0.5B-Instruct-sft"   --output-dir "models/Qwen2.5-0.5B-Instruct-sft-structure"   --reward-mode structure   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen2.5 3B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen2.5-3B-Instruct-sft"   --output-dir "models/Qwen2.5-3B-Instruct-sft-structure"   --reward-mode structure   --num-examples 16000   --batch-size 2 --use-vllm



# GRPO Answer Only, no SFT

# Qwen3.5 0.8B
python scripts/train_grpo_codebook_qa.py   --model "/hf/models/Qwen/Qwen3.5-0.8B"   --output-dir "models/Qwen3.5-0.8B-answer"   --reward-mode answer-only   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen3.5 4B
python scripts/train_grpo_codebook_qa.py   --model "/hf/models/Qwen/Qwen3.5-4B"   --output-dir "models/Qwen3.5-4B-answer"   --reward-mode answer-only   --num-examples 16000   --batch-size 2 --use-vllm

# Qwen2.5 0.5B
python scripts/train_grpo_codebook_qa.py   --model "Qwen/Qwen2.5-0.5B-Instruct"   --output-dir "models/Qwen2.5-0.5B-Instruct-answer"   --reward-mode answer-only --num-examples 16000   --batch-size 4 --use-vllm

# Qwen2.5 3B
python scripts/train_grpo_codebook_qa.py   --model "Qwen/Qwen2.5-3B-Instruct"   --output-dir "models/Qwen2.5-3B-Instruct-answer"   --reward-mode answer-only --num-examples 16000   --batch-size 2 --use-vllm


# GRPO Answer Only, with SFT

# Qwen3.5 0.8B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen3.5-0.8B-sft"   --output-dir "models/Qwen3.5-0.8B-sft-answer"   --reward-mode answer-only   --num-examples 16000   --batch-size 4 --use-vllm

# Qwen3.5 4B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen3.5-4B-sft"   --output-dir "models/Qwen3.5-4B-sft-answer"   --reward-mode answer-only   --num-examples 16000   --batch-size 2 --use-vllm

# Qwen2.5 0.5B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen2.5-0.5B-Instruct-sft"   --output-dir "models/Qwen2.5-0.5B-Instruct-sft-answer"   --reward-mode answer-only --num-examples 16000   --batch-size 4 --use-vllm

# Qwen2.5 3B
python scripts/train_grpo_codebook_qa.py   --model "models/Qwen2.5-3B-Instruct-sft"   --output-dir "models/Qwen2.5-3B-Instruct-sft-answer"   --reward-mode answer-only --num-examples 16000   --batch-size 2 --use-vllm


# We continue with process rewards after answers have been validated.