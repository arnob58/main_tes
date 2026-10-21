"""Continuation prompts and merging, built from externalised state only.

SRP: turn a SourceState (from disk, not from model memory) into a compact
fresh prompt, and splice the model's continuation back onto the artifact
without duplicating lines the model chose to repeat.
"""

from __future__ import annotations

import re
from typing import Callable

from ..core.types import Message
from .source_analyzer import SourceState

WRITE_PROMPT = """Stage: codewriter.
Write the complete Python module described by the task. Output raw Python only — no markdown fences, no commentary. Write top-level definitions one after another, each finished before the next."""

CONTINUE_PROMPT = """Stage: codecontinue.
You are resuming a Python module that was saved to disk before your context filled up. The earlier parts already exist; do NOT rewrite them. Output raw Python only (no markdown), continuing EXACTLY from the last character of CURRENT INCOMPLETE SECTION. If that section is empty, start the next definition. Finish the whole module."""

_FENCE_OPEN = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n")
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def strip_fences(text: str) -> str:
    text = _FENCE_OPEN.sub("", text, count=1)
    return _FENCE_CLOSE.sub("\n", text)


def merge_continuation(existing: str, continuation: str, max_overlap_lines: int = 30) -> str:
    """Append `continuation`, dropping a repeated overlap at the seam."""
    if not existing:
        return continuation
    if not continuation:
        return existing
    # Partial last line repeated in full by the model.
    if not existing.endswith("\n"):
        head, _, partial = existing.rpartition("\n")
        first, nl, rest = continuation.partition("\n")
        if partial.strip() and first.startswith(partial):
            return (head + "\n" if head or existing.startswith("\n") else "") + first + nl + rest
        return existing + continuation
    old = existing.splitlines()
    new = continuation.splitlines(keepends=True)
    for k in range(min(max_overlap_lines, len(old), len(new)), 0, -1):
        tail = [l.rstrip() for l in old[-k:]]
        if any(tail) and tail == [l.rstrip() for l in new[:k]]:
            return existing + "".join(new[k:])
    return existing + continuation


def build_continuation_messages(
    objective: str,
    state: SourceState,
    *,
    completed_steps: list[str],
    next_operation: str,
    count: Callable[[list[Message]], int],
    max_prompt_tokens: int,
) -> list[Message]:
    """Fit the prompt into `max_prompt_tokens`, degrading in stages.

    The incomplete section and the task are never cut. Shrink order:
    source excerpt (but keep at least its last 2 lines as the resume anchor)
    → completed-work list becomes a count → interfaces become names only →
    only the most recent interface names. The caller checks the final size.
    """
    prefix_lines = state.complete_prefix.splitlines()
    full_interfaces = state.interfaces_text() or "(none)"
    names = [f.name for f in state.functions] + [c.name for c in state.classes]
    suffix = state.incomplete_suffix
    keep = len(prefix_lines)
    min_keep = min(2, len(prefix_lines))
    level = 0

    while True:
        if level == 0:
            done = "\n".join(f"- {s}" for s in completed_steps) or "- none"
        else:
            done = f"- {len(completed_steps)} definitions written (see interfaces)"
        if level <= 1:
            interfaces = full_interfaces
        elif level == 2:
            interfaces = "Defined so far: " + ", ".join(names) if names else "(none)"
        else:
            recent = names[-10:]
            interfaces = f"Defined so far ({len(names)} total, most recent): " + ", ".join(recent)
        omitted = len(prefix_lines) - keep
        excerpt = "\n".join(prefix_lines[-keep:]) if keep else "(omitted)"
        body = (
            f"ORIGINAL TASK:\n{objective}\n\n"
            f"COMPLETED WORK:\n{done}\n\n"
            f"EXISTING INTERFACES:\n{interfaces}\n\n"
            f"RELEVANT SOURCE CODE (end of the completed part{f'; {omitted} earlier lines omitted' if omitted else ''}):\n"
            f"{excerpt}\n\n"
            f"CURRENT INCOMPLETE SECTION:\n{suffix if suffix else '(empty)'}\n\n"
            f"NEXT REQUIRED OPERATION:\n{next_operation}"
        )
        msgs = [Message.system(CONTINUE_PROMPT), Message.user(body)]
        if count(msgs) <= max_prompt_tokens:
            return msgs
        if keep > min_keep:
            keep = max(min_keep, keep // 2)
        elif level < 3:
            level += 1
        else:
            return msgs  # smallest form; the generator decides whether it fits
