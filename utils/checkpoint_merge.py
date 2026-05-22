"""
Merge trained checkpoint weights with a base model on disk (safetensors).

Useful when text-only SFT on a multimodal model omits vision-tower tensors but
the saved config still describes a multimodal architecture (vLLM load failures).
"""

from __future__ import annotations

import os
import shutil
from typing import Iterable

from safetensors import safe_open
from safetensors.torch import load_file, save_file


def env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def iter_safetensor_files(model_dir: str) -> list[str]:
    """
    Return all safetensors shards in a directory.

    Supports either:
    - model.safetensors
    - model.safetensors-00001-of-0000N.safetensors (HF-style sharding)
    """
    single = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]

    shards = [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.startswith("model.safetensors-") and f.endswith(".safetensors")
    ]
    shards.sort()
    if shards:
        return shards

    raise FileNotFoundError(
        f"No safetensors weights found in {model_dir!r}. "
        "Expected model.safetensors or model.safetensors-*.safetensors."
    )


def keys_from_safetensors(paths: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for p in paths:
        with safe_open(p, framework="pt", device="cpu") as f:
            out.update(f.keys())
    return out


def merge_checkpoint_with_base_weights(
    *,
    base_model_dir: str,
    trained_checkpoint_dir: str,
    output_dir: str,
    copy_non_weight_files_from: str | None = None,
    merge_same_dir: bool | None = None,
) -> str:
    """
    Build a vLLM-loadable checkpoint by merging:
    - all tensors from `trained_checkpoint_dir` (text-only training output)
    - any missing tensors from `base_model_dir` (e.g. vision tower)

    Trained tensors are never overwritten; base tensors only fill missing keys.
    Output is a single `model.safetensors` plus sidecar files copied from the
    trained checkpoint (config, tokenizer, processor, etc.).
    """
    base_model_dir = os.path.abspath(base_model_dir)
    trained_checkpoint_dir = os.path.abspath(trained_checkpoint_dir)
    output_dir = os.path.abspath(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    if merge_same_dir is None:
        merge_same_dir = env_bool("MERGE_SAME_DIR", False)

    base_weight_files = iter_safetensor_files(base_model_dir)
    trained_weight_files = iter_safetensor_files(trained_checkpoint_dir)

    src_for_sidecars = copy_non_weight_files_from or trained_checkpoint_dir
    if not merge_same_dir and output_dir != trained_checkpoint_dir:
        for name in os.listdir(src_for_sidecars):
            if name.endswith((".safetensors", ".bin", ".pt")):
                continue
            src = os.path.join(src_for_sidecars, name)
            dst = os.path.join(output_dir, name)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    merged_tensors: dict[str, object] = {}
    for p in trained_weight_files:
        merged_tensors.update(load_file(p, device="cpu"))

    existing_keys = set(merged_tensors.keys())
    for p in base_weight_files:
        with safe_open(p, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in existing_keys:
                    continue
                merged_tensors[k] = f.get_tensor(k)
                existing_keys.add(k)

    has_visual = any(k.startswith("model.visual.") for k in merged_tensors.keys())
    if not has_visual:
        base_keys = keys_from_safetensors(base_weight_files)
        if any(k.startswith("model.visual.") for k in base_keys):
            raise RuntimeError(
                "Merge produced no 'model.visual.*' tensors even though the base model has them. "
                "This likely indicates an unexpected key prefix mismatch."
            )

    out_weights = os.path.join(output_dir, "model.safetensors")
    tmp_weights = out_weights + ".tmp"
    save_file(merged_tensors, tmp_weights)
    os.replace(tmp_weights, out_weights)
    return output_dir
