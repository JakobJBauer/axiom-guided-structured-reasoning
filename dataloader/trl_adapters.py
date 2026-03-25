from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from torch.utils.data import Dataset

from .codebook_qa import CodebookQADataset, CodebookSample
from graph.graph import Graph


THINKING_OPEN, THINKING_CLOSE = "<thinking>", "</thinking>"


PromptStyle = Literal["full", "abbr", "none"]


DEFAULT_ABBR_PREFIX = "TASK: CODEBOOK_QA\n\n"


def _full_task_prefix() -> str:
    # Keep this stable across SFT/GRPO when style="full".
    return (
        "You are an expert reasoning assistant. Given a story, a codebook, and a "
        "yes/no question about whether the story satisfies a particular attribute, "
        "answer using the following STRICT format:\n\n"
        "<thinking>\n"
        "- The thinking section consists of multiple PARAGRAPHS.\n"
        "- Each paragraph is ONE argument.\n"
        "- Inside each paragraph, refer to attributes in ALL CAPS in square brackets, "
        "e.g. [SHORT], [NOUN], [NON-NOUN], [DENSE]. These are the nodes/attributes.\n"
        "- Each paragraph MUST END with a citation of the form:\n"
        "    (ATTR : True)\n"
        "  or\n"
        "    (ATTR : False)\n"
        "  where ATTR is the (uppercase) attribute name that this paragraph is "
        "concluding about.\n"
        "- The cited ATTR at the end of the paragraph MUST appear in square brackets "
        "somewhere in that paragraph as [ATTR].\n"
        "- Use one blank line between paragraphs.\n"
        "- Base your arguments on the story, codebook, and question.\n"
        "\n"
        "After the thinking block, output ONLY one final line starting with either:\n"
        "- \"Yes, the story is ...\"\n"
        "- \"No, the story is not ...\"\n\n"
    )


def build_task_prefix(style: PromptStyle, abbr_prefix: str = DEFAULT_ABBR_PREFIX) -> str:
    if style == "none":
        return ""
    if style == "abbr":
        return abbr_prefix
    if style == "full":
        return _full_task_prefix()
    raise ValueError(f"Unknown prompt style: {style!r}")


def build_base_prompt(
    sample: CodebookSample,
    *,
    prompt_style: PromptStyle = "none",
    abbr_prefix: str = DEFAULT_ABBR_PREFIX,
) -> str:
    """
    Shared prompt shape for SFT/GRPO.

    Keeps the fields minimal and stable:
    - Story
    - Codebook
    - Question
    """
    return build_task_prefix(prompt_style, abbr_prefix=abbr_prefix) + (
        "Story:\n"
        f"{sample.story}\n\n"
        "Codebook:\n"
        f"{sample.codebook_text}\n\n"
        "Question:\n"
        f"{sample.question}\n\n"
        "Assistant:\n"
    )


def render_reasoning_trace(graph: Graph, sink_id: str) -> str:
    """
    Deterministic structured trace from the inferred reasoning graph.

    One paragraph per node with a defined value; ends with a yes/no line.
    """
    topo = graph.topological_sort()
    paragraphs = []
    for node in topo:
        if node.value is None:
            continue

        attr = node.id.upper()
        value_str = "True" if bool(node.value) else "False"
        parents = graph.get_incoming_nodes(node)

        if not parents:
            para = (
                f"The story has the basic attribute [{attr}] according to the dataset features. "
                f"({attr} : {value_str})"
            )
        else:
            parent_attrs = [p.id.upper() for p in parents]
            parent_citations = ", ".join(f"[{a}]" for a in parent_attrs)
            if bool(node.value):
                para = (
                    f"Because {parent_citations} are true, I conclude that the story is [{attr}]. "
                    f"({attr} : True)"
                )
            else:
                para = (
                    f"Even though {parent_citations} hold, I conclude that the story is not [{attr}]. "
                    f"({attr} : False)"
                )
        paragraphs.append(para)

    thinking = THINKING_OPEN + "\n" + "\n\n".join(paragraphs) + "\n" + THINKING_CLOSE

    sink = graph.get_node_by_id(sink_id)
    if sink is None or sink.value is None:
        final_answer = "I cannot determine whether the story satisfies the target attribute."
    else:
        sink_label = (sink.label or sink.id).lower()
        final_answer = f"Yes, the story is {sink_label}." if bool(sink.value) else f"No, the story is not {sink_label}."

    return thinking + "\n" + final_answer + "\n"


def build_grpo_prompt(
    sample: CodebookSample,
    *,
    prompt_style: PromptStyle = "full",
    abbr_prefix: str = DEFAULT_ABBR_PREFIX,
) -> str:
    """
    GRPO prompt: enforce strict format, but do not include any teacher answer.
    """
    return build_base_prompt(
        sample,
        prompt_style=prompt_style,
        abbr_prefix=abbr_prefix,
    )


@dataclass(frozen=True)
class SFTAdapterConfig:
    prompt_style: PromptStyle = "none"
    abbr_prefix: str = DEFAULT_ABBR_PREFIX
    completion_mode: Literal["deterministic_trace", "boolean_only"] = "deterministic_trace"


class CodebookQASFTDataset(Dataset):
    """
    Torch dataset for TRL SFTTrainer.

    Returns a dict with a single `text` field by default (SFTConfig.dataset_text_field="text").
    """

    def __init__(
        self,
        base_dataset: CodebookQADataset,
        num_examples: Optional[int] = None,
        config: SFTAdapterConfig = SFTAdapterConfig(),
    ) -> None:
        self._base = base_dataset
        self._n = len(base_dataset) if num_examples is None else int(num_examples)
        self._config = config

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._base[idx % len(self._base)]
        prompt = build_base_prompt(
            sample,
            prompt_style=self._config.prompt_style,
            abbr_prefix=self._config.abbr_prefix,
        )

        if self._config.completion_mode == "boolean_only":
            completion = ("True" if sample.answer else "False") + "\n"
        else:
            completion = render_reasoning_trace(sample.reasoning_graph, sample.sink_id)

        return {"text": prompt + completion}


class CodebookQAGRPODataset(Dataset):
    """
    Torch dataset for TRL GRPOTrainer.

    Returns `prompt` for generation. Optionally includes `answer` for reward/debug.
    """

    def __init__(
        self,
        base_dataset: CodebookQADataset,
        num_examples: int = 1000,
        include_answer: bool = True,
        prompt_style: PromptStyle = "full",
        abbr_prefix: str = DEFAULT_ABBR_PREFIX,
    ) -> None:
        self._base = base_dataset
        self._n = int(num_examples)
        self._include_answer = include_answer
        self._prompt_style = prompt_style
        self._abbr_prefix = abbr_prefix

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self._base[idx % len(self._base)]
        out: Dict[str, Any] = {
            "prompt": build_grpo_prompt(
                sample,
                prompt_style=self._prompt_style,
                abbr_prefix=self._abbr_prefix,
            ),
            "sink_id": sample.sink_id,
        }
        if self._include_answer:
            out["answer"] = bool(sample.answer)
        return out

