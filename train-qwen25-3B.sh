#!/usr/bin/env bash
# Train Qwen2.5-3B-Instruct with optional SFT + GRPO.
#
# Flags (env or prompt if unset):
#   PROMPT_STYLE         full | abbr | none
#   CUDA_VISIBLE_DEVICES e.g. 0 or 1
#   USE_SFT              true | false
#     true  — GRPO from SFT checkpoint; run SFT first only if that dir is missing
#     false — GRPO from base /hf model; no SFT step
#
# Examples:
#   PROMPT_STYLE=full CUDA_VISIBLE_DEVICES=0 USE_SFT=true ./train-qwen25-3B.sh
#   PROMPT_STYLE=full CUDA_VISIBLE_DEVICES=1 USE_SFT=false ./train-qwen25-3B.sh
#
# Pipe prompts (style, GPU, USE_SFT):
#   printf 'full\n0\ntrue\n' | ./train-qwen25-3B.sh

set -euo pipefail

_parse_bool() {
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    1 | true | t | yes | y | on) echo "true" ;;
    0 | false | f | no | n | off) echo "false" ;;
    *)
      echo "Invalid boolean: $1 (use true/false)" >&2
      exit 1
      ;;
  esac
}

if [ -z "${PROMPT_STYLE:-}" ]; then
  read -r -p "Prompt style (full/abbr/none): " PROMPT_STYLE
fi
export PROMPT_STYLE

export PEFT="${PEFT:-False}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}" # only for SFT
export MAX_LENGTH="${MAX_LENGTH:-4096}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  read -r -p "CUDA_VISIBLE_DEVICES (e.g. 0 or 1): " CUDA_VISIBLE_DEVICES
fi
export CUDA_VISIBLE_DEVICES

if [ -z "${USE_SFT:-}" ]; then
  read -r -p "Use SFT checkpoint for GRPO? (true=train SFT if missing, then GRPO from it; false=GRPO from base): " USE_SFT
fi
USE_SFT="$(_parse_bool "$USE_SFT")"

export MODEL_NAME="Qwen2.5-3B-Instruct"
export MODEL_PATH="Qwen/${MODEL_NAME}" # no local option
export MODEL_IDENTIFIER="qwen25-3B"

if [ "$PEFT" = "True" ] || [ "$PEFT" = "true" ]; then
  export PEFT_PATH="peft"
else
  export PEFT_PATH="nopeft"
fi

if [ "$PROMPT_STYLE" = "full" ]; then
  export PROMPT_STYLE_PATH="fullprompt"
elif [ "$PROMPT_STYLE" = "abbr" ]; then
  export PROMPT_STYLE_PATH="abbrprompt"
elif [ "$PROMPT_STYLE" = "none" ]; then
  export PROMPT_STYLE_PATH="noneprompt"
else
  echo "Invalid prompt style: $PROMPT_STYLE"
  exit 1
fi

export MODEL_OUTPUT_DIR="models/${PROMPT_STYLE_PATH}/${PEFT_PATH}"
export SFT_MODEL_PATH="${MODEL_OUTPUT_DIR}/${MODEL_IDENTIFIER}-sft"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/agsr-venv}"

run_uv() {
  uv run "$@"
}

dir_has_contents() {
  local dir="$1"
  [ -d "$dir" ] || return 1
  local entries=("$dir"/* "$dir"/.[!.]* "$dir"/..?*)
  [ -e "${entries[0]}" ]
}

if [ "$USE_SFT" = "true" ]; then
  if ! dir_has_contents "$SFT_MODEL_PATH"; then
    echo "SFT checkpoint missing; training SFT on GPU ${CUDA_VISIBLE_DEVICES} -> $SFT_MODEL_PATH"
    run_uv scripts/train_sft_codebook_qa.py \
      --model "$MODEL_PATH" \
      --data-source jsonl \
      --data-path data/codebook_qa_sft_gpt_1000.jsonl \
      --output-dir "$SFT_MODEL_PATH" \
      --prompt-style "$PROMPT_STYLE" \
      --batch-size "$BATCH_SIZE" \
      --grad-accum "$GRAD_ACCUM" \
      --peft "$PEFT" \
      --max-length "$MAX_LENGTH"
  else
    echo "SFT checkpoint exists at $SFT_MODEL_PATH"
  fi
  GRPO_MODEL="$SFT_MODEL_PATH"
  GRPO_OUTPUT_PREFIX="$SFT_MODEL_PATH"
else
  echo "GRPO from base model (no SFT)"
  GRPO_MODEL="$MODEL_PATH"
  GRPO_OUTPUT_PREFIX="${MODEL_OUTPUT_DIR}/${MODEL_IDENTIFIER}"
fi

echo "Config: GPU=${CUDA_VISIBLE_DEVICES} USE_SFT=${USE_SFT}"
echo "  GRPO model: ${GRPO_MODEL}"

for REWARD_MODE in process structure answer-only; do
  GRPO_OUTPUT_DIR="${GRPO_OUTPUT_PREFIX}-${REWARD_MODE}"
  if dir_has_contents "$GRPO_OUTPUT_DIR"; then
    echo "GRPO ($REWARD_MODE) already exists at $GRPO_OUTPUT_DIR"
  else
    echo "Training GRPO ($REWARD_MODE) -> $GRPO_OUTPUT_DIR"
    run_uv scripts/train_grpo_codebook_qa.py \
      --model "$GRPO_MODEL" \
      --prompt-style "$PROMPT_STYLE" \
      --output-dir "$GRPO_OUTPUT_DIR" \
      --reward-mode "$REWARD_MODE" \
      --num-examples 16000 \
      --batch-size "$BATCH_SIZE" \
      --use-vllm \
      --peft "$PEFT"
  fi
done
