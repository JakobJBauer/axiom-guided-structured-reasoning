from __future__ import annotations

import re

THINKING_CLOSE = "</thinking>"
_BOOL_TOKEN_RE = re.compile(r"\b(yes|no|true|false)\b", re.IGNORECASE)


def final_answer_tail(response: str) -> str:
    """Text after the closing thinking tag, or the full response if absent."""
    lower = response.lower()
    end = lower.rfind(THINKING_CLOSE)
    if end != -1:
        return response[end + len(THINKING_CLOSE) :].strip()
    return response.strip()


def parse_final_boolean_answer(response: str) -> bool | None:
    """
    Parse yes/no from the final answer tail (after </thinking> if present).
    Uses the last yes/no/true/false token in that tail.
    """
    tail = final_answer_tail(response).lower()
    matches = list(_BOOL_TOKEN_RE.finditer(tail))
    if not matches:
        return None
    token = matches[-1].group(1).lower()
    if token in {"yes", "true"}:
        return True
    if token in {"no", "false"}:
        return False
    return None
