"""Protocols that high-level code depends on (Dependency Inversion).

The orchestrator, router and agents only ever see these interfaces. Concrete
backends (llama.cpp, a llama-server HTTP client, the scripted test model,
the future skill engine) satisfy them structurally — no inheritance needed.

Interfaces are deliberately small (Interface Segregation): a component that
only needs to count tokens depends on TokenCounter, not on the whole model.
"""

from __future__ import annotations

from typing import Callable, Protocol, Sequence, runtime_checkable

from .types import GenerationResult, Message, Observation, SkillRequest


@runtime_checkable
class ModelProvider(Protocol):
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
        """Run one fresh, stateless chat completion.

        `grammar` is GBNF text; backends that cannot honour it must ignore it
        (callers always validate output themselves).
        """
        ...


@runtime_checkable
class TokenCounter(Protocol):
    @property
    def context_window(self) -> int: ...

    def count_prompt_tokens(self, messages: Sequence[Message]) -> int:
        """Exact token count of `messages` once rendered with the chat template."""
        ...


@runtime_checkable
class ToolBridge(Protocol):
    """Where agents send skill requests. Phase 1 ships a null implementation;
    Phase 3's skill engine provides the real one without touching agents."""

    def describe_capabilities(self) -> str: ...

    def execute(self, request: SkillRequest) -> Observation: ...


@runtime_checkable
class Approver(Protocol):
    """Gate for consequential actions (installs, risky code, GUI input, submissions, patches)."""

    def approve(self, action: str, detail: str) -> bool: ...


@runtime_checkable
class MemoryRetriever(Protocol):
    def recall_text(self, query: str, k: int = 5) -> str:
        """Prompt-ready block of possibly relevant evidence, or ''."""
        ...


class EventSink(Protocol):
    def emit(self, channel: str, message: str) -> None: ...


EvictionHook = Callable[[Message], None]
