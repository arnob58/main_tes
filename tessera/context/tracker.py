"""Context usage measurement with the model's real tokenizer.

SRP: answer "how full would the window be?" and "how many output tokens can
I allow before crossing the recovery threshold?". It never edits prompts.

usage = (prompt tokens + reserved output tokens) / context window
Recovery triggers at usage >= threshold (default 0.90, i.e. 10% remaining).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.interfaces import TokenCounter
from ..core.types import Message


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    reserved_output: int
    window: int
    threshold: float

    @property
    def ratio(self) -> float:
        return (self.prompt_tokens + self.reserved_output) / self.window

    @property
    def needs_recovery(self) -> bool:
        return self.ratio >= self.threshold

    @property
    def safe_output_budget(self) -> int:
        """Output tokens allowed before generation would cross the threshold."""
        return max(0, int(self.window * self.threshold) - self.prompt_tokens)


class ContextTracker:
    def __init__(self, counter: TokenCounter, threshold: float = 0.90) -> None:
        if not 0.1 <= threshold < 1.0:
            raise ValueError("threshold must be in [0.1, 1.0)")
        self._counter = counter
        self.threshold = threshold

    @property
    def window(self) -> int:
        return self._counter.context_window

    def measure(self, messages: Sequence[Message], reserved_output: int = 0) -> Usage:
        return Usage(self._counter.count_prompt_tokens(messages), reserved_output, self.window, self.threshold)

    def count(self, messages: Sequence[Message]) -> int:
        return self._counter.count_prompt_tokens(messages)
