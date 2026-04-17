import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

def load_model_and_processor(model_name_or_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    if "Qwen3.5" in model_name_or_path:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None
        ).to(device)
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        return model, processor

    elif "Qwen2.5" in model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to(device)

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
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        return model, tokenizer