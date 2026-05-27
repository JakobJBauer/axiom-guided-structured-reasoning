import json
import os
import torch
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

def _strtobool_env(name: str, default: str = "false") -> bool:
    v = str(os.environ.get(name, default)).strip().lower()
    return v in {"1", "true", "t", "yes", "y", "on"}

def _multi_gpu():
    return os.environ.get("LOCAL_RANK") is not None


def _read_config_hints(model_name_or_path: str) -> tuple[str | None, list[str]]:
    """Return (model_type, architectures) from a local or Hub model path."""
    p = Path(model_name_or_path)
    if p.is_dir() and (p / "config.json").is_file():
        cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        return cfg.get("model_type"), list(cfg.get("architectures") or [])

    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        return getattr(config, "model_type", None), list(getattr(config, "architectures", None) or [])
    except Exception:
        return None, []


def _is_qwen35_model(model_name_or_path: str) -> bool:
    model_type, architectures = _read_config_hints(model_name_or_path)
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        return True
    return any("qwen3_5" in arch.lower() for arch in architectures)


def _register_qwen35_rope_delta_batch_guard(model: torch.nn.Module) -> None:
    """
    Qwen3_5Model caches `rope_deltas` from the previous forward. GRPO runs
    generation with batch size (prompts * num_generations), then TRL chunks
    reference / old-policy logprob forwards with `per_device_train_batch_size`.
    If the cache is larger than the current batch, `compute_3d_position_ids`
    uses `repeat_interleave(batch // cache_batch)` which becomes 0 and
    crashes before reward functions run.
    """
    def _pre_hook(module, args, kwargs):
        input_ids = kwargs.get("input_ids") if kwargs else None
        if input_ids is None and args:
            input_ids = args[0]
        if input_ids is None:
            return args, kwargs
        rd = getattr(module, "rope_deltas", None)
        if rd is not None and rd.shape[0] != input_ids.shape[0]:
            module.rope_deltas = None
        return args, kwargs

    for m in model.modules():
        if type(m).__name__ == "Qwen3_5Model" and hasattr(m, "rope_deltas"):
            m.register_forward_pre_hook(_pre_hook, with_kwargs=True)


def load_model_and_processor(model_name_or_path: str):
    device = None if _multi_gpu() else "cuda" if torch.cuda.is_available() else "cpu"
    if device is None: print("Multi-GPU detected. Using CPU for model loading.")

    # If `model_name_or_path` is a PEFT adapter directory, load base + attach adapter.
    p = Path(model_name_or_path)
    if p.exists() and (p / "adapter_config.json").exists():
        print(f"Loading PEFT Adapter: {model_name_or_path}...")
        from peft import PeftConfig, PeftModel

        peft_cfg = PeftConfig.from_pretrained(str(p))
        base_id = peft_cfg.base_model_name_or_path
        if not base_id: raise ValueError(f"PEFT adapter at {model_name_or_path!r} does not specify base_model_name_or_path.")

        base_model, base_tokenizer = load_model_and_processor(str(base_id))
        model = PeftModel.from_pretrained(base_model, str(p))

        return model, base_tokenizer


    print(f"Loading Training Model: {model_name_or_path}...")
    if _is_qwen35_model(model_name_or_path):
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None
        )
        if device is not None:  model = model.to(device)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        _register_qwen35_rope_delta_batch_guard(model)
        return model, processor

    elif "Qwen2.5" in model_name_or_path:
        tok_kwargs = {"trust_remote_code": True}
        if _strtobool_env("FIX_MISTRAL_REGEX", "true"):
            # Some tokenizers accept this kwarg; if unsupported, fall back silently.
            tok_kwargs["fix_mistral_regex"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
        except TypeError:
            tok_kwargs.pop("fix_mistral_regex", None)
            print("Fix mistral regex not supported. Trying without fix_mistral_regex...")
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if device is not None: model = model.to(device)

        # PEFT/LoRA adapter loading (kept as commented-out reference):
        # base_model = AutoModelForCausalLM.from_pretrained(
        #     model_name_or_path, trust_remote_code=True
        # )
        # model = PeftModel.from_pretrained(base_model, adapter_model_name_or_path)
        return model, tokenizer
    
    else:
        print(f"Generic Fallback. Loading Full Training Model: {model_name_or_path}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        if device is not None: model = model.to(device)
        tok_kwargs = {"trust_remote_code": True}
        if _strtobool_env("FIX_MISTRAL_REGEX", "true"):
            tok_kwargs["fix_mistral_regex"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
        except TypeError:
            print("Fix mistral regex not supported. Trying without fix_mistral_regex...")
            tok_kwargs.pop("fix_mistral_regex", None)
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
        return model, tokenizer