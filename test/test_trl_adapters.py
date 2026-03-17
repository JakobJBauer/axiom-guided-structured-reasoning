from __future__ import annotations

from dataloader.codebook_qa import CodebookQADataset, GraphDifficultyConfig
from dataloader.trl_adapters import (
    DEFAULT_ABBR_PREFIX,
    CodebookQAGRPODataset,
    CodebookQASFTDataset,
    SFTAdapterConfig,
    build_task_prefix,
)


def _make_base_dataset() -> CodebookQADataset:
    # Use a tiny, deterministic config so tests are cheap.
    return CodebookQADataset(
        split="train",
        difficulties=[(GraphDifficultyConfig(goal_depth=2, max_leaf_nodes=4), 1.0)],
        seed=0,
    )


def test_build_task_prefix_variants():
    assert build_task_prefix("none") == ""
    assert build_task_prefix("abbr") == DEFAULT_ABBR_PREFIX
    full = build_task_prefix("full")
    assert "You are an expert reasoning assistant." in full
    assert "<thinking>" in full


def test_sft_adapter_prompt_styles():
    base = _make_base_dataset()

    # none
    ds_none = CodebookQASFTDataset(
        base_dataset=base,
        num_examples=1,
        config=SFTAdapterConfig(prompt_style="none"),
    )
    item_none = ds_none[0]
    text_none = item_none["text"]
    assert text_none.startswith("Story:\n")

    # abbr
    ds_abbr = CodebookQASFTDataset(
        base_dataset=base,
        num_examples=1,
        config=SFTAdapterConfig(prompt_style="abbr"),
    )
    text_abbr = ds_abbr[0]["text"]
    assert text_abbr.startswith(DEFAULT_ABBR_PREFIX + "Story:\n")

    # full
    ds_full = CodebookQASFTDataset(
        base_dataset=base,
        num_examples=1,
        config=SFTAdapterConfig(prompt_style="full"),
    )
    text_full = ds_full[0]["text"]
    assert "You are an expert reasoning assistant." in text_full.split("Story:\n", 1)[0]
    assert "Story:\n" in text_full


def test_grpo_adapter_prompt_styles():
    base = _make_base_dataset()

    ds = CodebookQAGRPODataset(
        base_dataset=base,
        num_examples=3,
        include_answer=True,
        prompt_style="abbr",
    )
    item = ds[0]
    assert "prompt" in item
    assert isinstance(item["answer"], bool)
    assert item["prompt"].startswith(DEFAULT_ABBR_PREFIX + "Story:\n")

