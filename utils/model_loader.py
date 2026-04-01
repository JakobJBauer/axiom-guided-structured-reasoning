import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

def load_model_and_processor(model_name_or_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if "Qwen3.5" in model_name_or_path:
        print(f"Loading Full Training Model: {model_name_or_path}...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, 
            trust_remote_code=True,
            dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            device_map="auto"
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

        print(f"Loading Full Training Model: {model_name_or_path}...")
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