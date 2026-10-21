"""A deterministic stand-in for the real model.

Used by the test suite and by `--scripted` dry runs so control flow can be
exercised without a GPU. It never pretends to be Gemma: every reply comes
from an explicit rule or queue supplied by the caller, and every call is
recorded for assertions.

Token counts here are whitespace word counts — fine for control-flow tests,
and clearly labelled as such in metadata. Real counting lives in the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..core.types import GenerationResult, Message

Rule = Callable[[Sequence[Message]], "str | None"]


@dataclass
class ScriptedCall:
    messages: list[Message]
    max_output_tokens: int
    temperature: float
    grammar: str | None


@dataclass
class ScriptedModel:
    rules: list[Rule] = field(default_factory=list)
    queue: list[str] = field(default_factory=list)
    default: str = "OK."
    context_window: int = 4096
    calls: list[ScriptedCall] = field(default_factory=list)

    def when(self, predicate: Callable[[str], bool], reply: str | Callable[[Sequence[Message]], str]) -> "ScriptedModel":
        """Reply when `predicate(system_prompt)` is true."""

        def rule(messages: Sequence[Message]) -> str | None:
            system = messages[0].content if messages else ""
            if predicate(system):
                return reply(messages) if callable(reply) else reply
            return None

        self.rules.append(rule)
        return self

    def when_agent(self, agent: str, reply: str | Callable[[Sequence[Message]], str]) -> "ScriptedModel":
        """Reply when the system prompt belongs to stage `agent` (or "router")."""
        return self.when(lambda s, a=agent: s.startswith(f"Stage: {a}."), reply)

    # ModelProvider
    def generate(
        self,
        messages: Sequence[Message],
        *,
        max_output_tokens: int = 256,
        temperature: float = 0.3,
        grammar: str | None = None,
        stop: Sequence[str] | None = None,
        repeat_penalty: float | None = None,
    ) -> GenerationResult:
        self.calls.append(ScriptedCall(list(messages), max_output_tokens, temperature, grammar))
        text = None
        if self.queue:
            text = self.queue.pop(0)
        else:
            for rule in self.rules:
                text = rule(messages)
                if text is not None:
                    break
        if text is None:
            text = self.default
        spans = [m.end() for m in re.finditer(r"\S+", text)]
        finish = "stop"
        if len(spans) > max_output_tokens:
            # cut after the Nth word, preserving the original whitespace
            text, finish = text[: spans[max_output_tokens - 1]] if max_output_tokens else "", "length"
            spans = spans[:max_output_tokens]
        return GenerationResult(
            text=text,
            input_tokens=self.count_prompt_tokens(messages),
            output_tokens=len(spans),
            finish_reason=finish,
            metadata={"backend": "scripted", "token_counts": "approximate-word-count"},
        )

    # TokenCounter
    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        return sum(len(m.content.split()) + 4 for m in messages)
