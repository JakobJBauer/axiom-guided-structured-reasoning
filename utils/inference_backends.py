from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


class VllmBatchBackend:
    """Batch text generation via vLLM (see open-ended-csp/utils/lm_utils/inference_backends.py)."""

    def __init__(self, checkpoint_path: str, *, gen_config: GenerationConfig) -> None:
        self.checkpoint_path = checkpoint_path
        self.gen_config = gen_config
        self._llm = None

    def load(self) -> bool:
        if self._llm is not None:
            return True
        try:
            from vllm import LLM  # type: ignore
        except Exception as e:
            print(f"vLLM import failed: {e}")
            return False

        max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "16384"))
        gpu_memory_utilization = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self._llm = LLM(
            model=self.checkpoint_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        return True

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        if self._llm is None:
            raise RuntimeError("vLLM backend is not loaded")
        from vllm import SamplingParams  # type: ignore

        cfg = self.gen_config
        params = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature if cfg.do_sample else 0.0,
            top_p=cfg.top_p,
        )
        outputs = self._llm.generate(list(prompts), params, use_tqdm=False)
        return [out.outputs[0].text for out in outputs]


class TransformersBatchBackend:
    """Batch generation with HuggingFace causal LM + tokenizer."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        gen_config: GenerationConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.device = next(model.parameters()).device

    def generate_batch(self, prompts: Sequence[str]) -> list[str]:
        tokenizer = self.tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        encoded = tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_lengths = attention_mask.sum(dim=1)

        cfg = self.gen_config
        gen_kwargs: dict = {
            "max_new_tokens": cfg.max_new_tokens,
            "do_sample": cfg.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if cfg.do_sample:
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )

        texts: list[str] = []
        for i in range(len(prompts)):
            new_tokens = output_ids[i, prompt_lengths[i] :]
            texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return texts
