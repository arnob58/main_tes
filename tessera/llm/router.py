"""Adaptive router: one tiny model call → (complexity, needs_skills).

SRP: classification only. It does not choose skills or run agents.

Robustness layers, cheapest first, with no extra model calls:
1. GBNF grammar constrains a llama.cpp backend to exactly `Medium|yes`-style
   output.
2. A strict regex accepts the canonical form with any casing/whitespace.
3. A lenient scan picks up the words if the model wrapped them in prose.
4. A fixed fallback (Medium|no) with `parsed_cleanly=False`, so the
   orchestrator and metrics can see that recovery happened.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..core.interfaces import ModelProvider
from ..core.types import Complexity, Message, RouteDecision
from .prompts import ROUTER_GRAMMAR, ROUTER_PROMPT

_STRICT = re.compile(r"^\s*(instant|medium|complex)\s*\|\s*(yes|no)\s*\.?\s*$", re.I)
_COMPLEXITY = re.compile(r"\b(instant|medium|complex)\b", re.I)
_FLAG = re.compile(r"\|\s*(yes|no)\b|\b(yes|no)\b", re.I)

FALLBACK = (Complexity.MEDIUM, False)


def parse_route(raw: str) -> RouteDecision:
    first_line = raw.strip().splitlines()[0] if raw.strip() else ""
    m = _STRICT.match(first_line)
    if m:
        return RouteDecision(Complexity(m.group(1).lower()), m.group(2).lower() == "yes", raw, True)

    cm = _COMPLEXITY.search(raw)
    fm = _FLAG.search(raw)
    complexity = Complexity(cm.group(1).lower()) if cm else FALLBACK[0]
    if fm:
        needs = (fm.group(1) or fm.group(2)).lower() == "yes"
    else:
        needs = FALLBACK[1]
    return RouteDecision(complexity, needs, raw, False)


class Router:
    def __init__(
        self,
        model: ModelProvider,
        *,
        max_tokens: int = 8,
        temperature: float = 0.0,
        context_chars: int = 300,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._context_chars = context_chars

    def build_messages(self, request: str, recent: Sequence[Message] = ()) -> list[Message]:
        # Only a sliver of recent context: enough to resolve "do that again",
        # cheap enough to keep routing fast.
        prior = ""
        if recent:
            tail = recent[-2:]
            lines = [f"{m.role.value}: {m.content[: self._context_chars]}" for m in tail]
            prior = "Recent conversation:\n" + "\n".join(lines) + "\n\n"
        return [Message.system(ROUTER_PROMPT), Message.user(f"{prior}Request: {request}")]

    def route(self, request: str, recent: Sequence[Message] = ()) -> RouteDecision:
        result = self._model.generate(
            self.build_messages(request, recent),
            max_output_tokens=self._max_tokens,
            temperature=self._temperature,
            grammar=ROUTER_GRAMMAR,
        )
        return parse_route(result.text)
